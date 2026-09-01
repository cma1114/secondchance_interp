from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSDPAQuerySourceAttentionAblator
from .io import atomic_save_npz
from .modeling import get_tokenizer, load_model_and_processor, resolve_answer_tokens
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_all_candidate_matched_relay import _specs
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_fixed_a_final_query_edge_ablation import _option_line_positions
from .run_option_newline_all_four_projection import CompleteSequenceGLA


SCENARIOS = (
    "trusted_natural",
    "complete_sequence_natural",
    "identity_hook",
    "fresh_scrub",
    "matching_history_blockade",
    "matching_plus_fresh",
    "dose_matched_random",
    "matching_plus_random",
)
TRUSTED_NATURAL, COMPLETE_NATURAL, IDENTITY, FRESH, MATCHING, JOINT, RANDOM, MATCHING_RANDOM = range(8)
GROUPS = ("semantic_wordpieces", "option_newline")
CONTENT_SUMMARY = 1
NEWLINE_SUMMARY = 3
OLD_TARGET = 0
FRESH_TARGET = 1
LAYERS = tuple(range(64))
ORDINARY_LAYERS = tuple(range(3, 64, 4))
SCRUB_ITERATIONS = 8


def _hidden(output: Any) -> Any:
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_hidden(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


def _unit_unique_fresh(old: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    old = np.asarray(old, dtype=np.float64)
    fresh = np.asarray(fresh, dtype=np.float64)
    old = old / max(float(np.linalg.norm(old)), 1e-12)
    unique = fresh - float(fresh @ old) * old
    norm = float(np.linalg.norm(unique))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Fresh direction is degenerate after removing old direction")
    return (unique / norm).astype(np.float32)


def _random_control(old: np.ndarray, fresh: np.ndarray, seed: int) -> np.ndarray:
    old = np.asarray(old, dtype=np.float64)
    fresh = np.asarray(fresh, dtype=np.float64)
    old /= max(float(np.linalg.norm(old)), 1e-12)
    fresh /= max(float(np.linalg.norm(fresh)), 1e-12)
    rng = np.random.default_rng(int(seed))
    value = rng.standard_normal(old.shape[0])
    # Two passes make the finite-precision orthogonalization stable.
    for _ in range(2):
        value -= float(value @ old) * old
        value -= float(value @ fresh) * fresh
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Random control direction is degenerate")
    return (value / norm).astype(np.float32)


def _direction_geometry(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if tuple(directions.shape[:3]) != (64, 4, 2):
        raise ValueError(f"Unexpected score-direction shape: {directions.shape}")
    width = int(directions.shape[-1])
    fresh = np.empty((64, 2, width), dtype=np.float32)
    random = np.empty_like(fresh)
    diagnostics: dict[str, Any] = {"model_width": width, "groups": {}}
    for group_index, (group, summary) in enumerate(
        zip(GROUPS, (CONTENT_SUMMARY, NEWLINE_SUMMARY))
    ):
        cosines = []
        residual_norms = []
        random_old = []
        random_fresh = []
        for layer in LAYERS:
            old = np.asarray(directions[layer, summary, OLD_TARGET], dtype=np.float32)
            raw_fresh = np.asarray(directions[layer, summary, FRESH_TARGET], dtype=np.float32)
            old_unit = old / max(float(np.linalg.norm(old)), 1e-12)
            raw_unit = raw_fresh / max(float(np.linalg.norm(raw_fresh)), 1e-12)
            fresh[layer, group_index] = _unit_unique_fresh(old_unit, raw_unit)
            random[layer, group_index] = _random_control(
                old_unit,
                fresh[layer, group_index],
                seed=20260828 + layer * 101 + group_index * 10007,
            )
            cosines.append(float(old_unit @ raw_unit))
            residual_norms.append(float(np.linalg.norm(raw_unit - (raw_unit @ old_unit) * old_unit)))
            random_old.append(float(random[layer, group_index] @ old_unit))
            random_fresh.append(float(random[layer, group_index] @ fresh[layer, group_index]))
        diagnostics["groups"][group] = {
            "raw_old_fresh_cosine_by_layer": cosines,
            "unique_fresh_pre_normalization_norm_by_layer": residual_norms,
            "max_abs_random_old_dot": float(np.max(np.abs(random_old))),
            "max_abs_random_unique_fresh_dot": float(np.max(np.abs(random_fresh))),
        }
    return fresh, random, diagnostics


def _mean_states(hidden: Any, groups: list[list[list[int]]]) -> Any:
    import torch

    rows = []
    for row, candidates in enumerate(groups):
        rows.append(
            torch.stack(
                [hidden[row, torch.as_tensor(columns, device=hidden.device)].float().mean(0) for columns in candidates]
            )
        )
    return torch.stack(rows)


def _coordinates(means: Any, old: Any, fresh: Any) -> tuple[Any, Any, Any, Any]:
    rms = means.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
    normalized = means / rms
    centered = normalized - normalized.mean(dim=1, keepdim=True)
    old_coordinate = (centered * old[None, None]).sum(dim=-1)
    fresh_coordinate = (centered * fresh[None, None]).sum(dim=-1)
    return old_coordinate, fresh_coordinate, rms, normalized


class FreshOptionLineScrubber:
    """Repeatedly remove unique fresh-score state from the four 2P option lines."""

    def __init__(
        self,
        parts: Any,
        content_positions: list[list[list[int]]],
        newline_positions: list[list[list[int]]],
        old_directions: np.ndarray,
        fresh_directions: np.ndarray,
        random_directions: np.ndarray,
        mode: str,
        prescribed_dose: np.ndarray | None = None,
    ) -> None:
        import torch

        if mode not in {"identity", "fresh", "random"}:
            raise ValueError(f"Unknown scrub mode: {mode}")
        self.mode = mode
        self.groups = (content_positions, newline_positions)
        self.old = torch.from_numpy(np.asarray(old_directions, dtype=np.float32)).cpu()
        self.fresh = torch.from_numpy(np.asarray(fresh_directions, dtype=np.float32)).cpu()
        self.random = torch.from_numpy(np.asarray(random_directions, dtype=np.float32)).cpu()
        self.prescribed_dose = (
            None
            if prescribed_dose is None
            else torch.from_numpy(np.asarray(prescribed_dose, dtype=np.float32)).cpu()
        )
        if mode == "random" and self.prescribed_dose is None:
            raise ValueError("Random control requires a frozen dose schedule")
        batch = len(content_positions)
        shape = (64, batch, 2, 4)
        self.pre_fresh = torch.full(shape, float("nan"))
        self.post_fresh = torch.full(shape, float("nan"))
        self.pre_old = torch.full(shape, float("nan"))
        self.post_old = torch.full(shape, float("nan"))
        self.dose = torch.full(shape, float("nan"))
        self.handles = [
            layer.register_forward_hook(self._hook(layer_index))
            for layer_index, layer in enumerate(parts.layers)
        ]

    def _hook(self, layer_index: int):
        def intervene(_module: Any, _inputs: Any, output: Any) -> Any:
            import torch

            hidden = _hidden(output)
            if hidden.ndim != 3:
                raise RuntimeError(f"Layer {layer_index + 1}: unexpected hidden shape {hidden.shape}")
            changed = hidden.clone() if self.mode != "identity" else hidden
            for group_index, positions in enumerate(self.groups):
                old = self.old[layer_index, group_index].to(hidden.device)
                fresh = self.fresh[layer_index, group_index].to(hidden.device)
                random = self.random[layer_index, group_index].to(hidden.device)
                means = _mean_states(hidden, positions)
                old_pre, fresh_pre, _rms, _normalized = _coordinates(means, old, fresh)
                self.pre_old[layer_index, :, group_index] = old_pre.detach().cpu()
                self.pre_fresh[layer_index, :, group_index] = fresh_pre.detach().cpu()

                if self.mode == "identity":
                    old_post, fresh_post = old_pre, fresh_pre
                    dose = torch.zeros_like(fresh_pre)
                else:
                    # Construct the exact fresh-scrub displacement in mean-state
                    # space. The same candidate-wise displacement magnitudes are
                    # used for the random-direction control.
                    target_means = means.clone()
                    for _ in range(SCRUB_ITERATIONS):
                        _old_now, fresh_now, rms_now, _normalized_now = _coordinates(
                            target_means, old, fresh
                        )
                        target_means = target_means - fresh_now[..., None] * fresh[None, None] * rms_now
                        old_now, _fresh_now, rms_now, _normalized_now = _coordinates(
                            target_means, old, fresh
                        )
                        target_means = target_means + (
                            (old_pre - old_now)[..., None] * old[None, None] * rms_now
                        )
                    fresh_delta = means - target_means
                    if self.mode == "fresh":
                        delta = fresh_delta
                    else:
                        magnitude = self.prescribed_dose[layer_index, :, group_index].to(
                            hidden.device
                        )
                        sign = torch.where(fresh_pre >= 0, 1.0, -1.0)
                        delta = sign[..., None] * magnitude[..., None] * random[None, None]

                    for row, candidates in enumerate(positions):
                        for candidate, columns in enumerate(candidates):
                            column_tensor = torch.as_tensor(columns, device=hidden.device)
                            changed[row, column_tensor] = (
                                changed[row, column_tensor].float() - delta[row, candidate]
                            ).to(hidden.dtype)
                    post_means = _mean_states(changed, positions)
                    old_post, fresh_post, _post_rms, _post_normalized = _coordinates(
                        post_means, old, fresh
                    )
                    dose = delta.norm(dim=-1)

                self.post_old[layer_index, :, group_index] = old_post.detach().cpu()
                self.post_fresh[layer_index, :, group_index] = fresh_post.detach().cpu()
                self.dose[layer_index, :, group_index] = dose.detach().cpu()
            return output if self.mode == "identity" else _replace_hidden(output, changed)

        return intervene

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "pre_fresh": self.pre_fresh.numpy(),
            "post_fresh": self.post_fresh.numpy(),
            "pre_old": self.pre_old.numpy(),
            "post_old": self.post_old.numpy(),
            "dose_l2": self.dose.numpy(),
        }
        for name, value in arrays.items():
            if not np.isfinite(value).all():
                raise RuntimeError(f"Missing or non-finite scrub audit values: {name}")
        return arrays

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []

    def __enter__(self) -> "FreshOptionLineScrubber":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _initialize(path: Path, qids: list[str], split: np.ndarray) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses another question order")
        return arrays
    n = len(qids)
    audit_shape = (2, len(SCENARIOS), n, 64, 2, 4)
    return {
        "question_ids": np.asarray(qids),
        "split": split,
        "completed": np.zeros(n, dtype=bool),
        "baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "rank_letters": np.full((n, 4), "", dtype="<U1"),
        "logits": np.full((2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32),
        "pre_fresh": np.full(audit_shape, np.nan, dtype=np.float32),
        "post_fresh": np.full(audit_shape, np.nan, dtype=np.float32),
        "pre_old": np.full(audit_shape, np.nan, dtype=np.float32),
        "post_old": np.full(audit_shape, np.nan, dtype=np.float32),
        "dose_l2": np.full(audit_shape, np.nan, dtype=np.float32),
        "trusted_max_abs_error": np.full((2, n), np.nan, dtype=np.float32),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
    }


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(args.config)
    if (
        config.prompt_mode != "baseline_matched_empty_history"
        or config.feedback_variant != "token_matched_test"
        or config.chat_serialization != "raw_qwen_chatml"
        or config.attn_implementation != "sdpa"
        or int(config.batch_size) != 4
    ):
        raise ValueError("Requires the exact canonical empty-history batch-four SDPA regime")

    manifest = json.loads(Path(config.manifest_path).read_text())["questions"]
    qids = [row["id"] for row in manifest]
    if len(qids) != 500:
        raise ValueError(f"Expected 500 questions, got {len(qids)}")
    questions = {row["id"]: row for row in manifest}
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    if len(discovery_ids) != 251 or len(set(qids) - discovery_ids) != 249:
        raise ValueError("Expected frozen 251/249 split")
    split = np.asarray(["discovery" if qid in discovery_ids else "confirmation" for qid in qids])
    mappings = {
        row["question_id"]: row for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    if set(mappings) != set(qids):
        raise ValueError("Remapping plan does not cover the frozen manifest exactly")
    baseline = json.loads(args.baseline.read_text())["results"]
    trusted = [
        json.loads(args.trusted_game.read_text())["results"],
        json.loads(args.trusted_neutral.read_text())["results"],
    ]

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    loaded_directions = torch.load(args.score_directions, map_location="cpu", weights_only=True)
    if hasattr(loaded_directions, "numpy"):
        direction_array = loaded_directions.float().numpy()
    elif isinstance(loaded_directions, dict) and "directions" in loaded_directions:
        direction_array = loaded_directions["directions"].float().numpy()
    else:
        raise ValueError("Unrecognized score-direction artifact")
    unique_fresh, random_directions, geometry = _direction_geometry(direction_array)
    old_directions = np.stack(
        [direction_array[:, CONTENT_SUMMARY, OLD_TARGET], direction_array[:, NEWLINE_SUMMARY, OLD_TARGET]],
        axis=1,
    ).astype(np.float32)
    old_directions /= np.maximum(np.linalg.norm(old_directions, axis=-1, keepdims=True), 1e-12)

    if tuple(index for index, layer in enumerate(parts.layers) if getattr(layer, "self_attn", None) is not None) != ORDINARY_LAYERS:
        raise RuntimeError("Unexpected complete ordinary-attention inventory")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.npz"
    arrays = _initialize(result_path, qids, split)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = args.output_dir / "prompt_audit.json"
    started = time.monotonic()
    durations: list[float] = []
    completed_cohorts = 0
    pending = {qid for qid, done in zip(qids, arrays["completed"]) if not bool(done)}

    for start in range(0, len(qids), 4):
        cohort = qids[start : start + 4]
        if not set(cohort) & pending:
            continue
        cohort_started = time.monotonic()
        indices = [qid_index[qid] for qid in cohort]
        for qid in cohort:
            qi = qid_index[qid]
            old_logits = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=np.float32)
            arrays["baseline_logits"][qi] = old_logits
            arrays["rank_letters"][qi] = np.asarray(
                [LETTERS[int(index)] for index in np.argsort(-old_logits, kind="stable")]
            )

        for condition_index, condition in enumerate(CONDITIONS):
            batch = _build_batch(config, processor, tokenizer, questions, mappings, cohort, condition)
            width = int(batch["input_ids"].shape[1])
            source_positions: list[list[list[int]]] = []
            query_positions: list[list[list[int]]] = []
            content_positions: list[list[list[int]]] = []
            newline_positions: list[list[list[int]]] = []
            row_audits: list[dict[str, Any]] = []
            for row, qid in enumerate(cohort):
                qi = qid_index[qid]
                left_pad = width - len(batch["token_rows"][row])
                remapped_question = {
                    **questions[qid],
                    "options": {
                        new: questions[qid]["options"][old]
                        for new, old in mappings[qid]["new_to_original"].items()
                    },
                }
                first_lines, first_audit = _option_line_positions(
                    tokenizer, batch["prompts"][row], questions[qid]
                )
                second_lines, second_audit = _option_line_positions(
                    tokenizer, batch["prompts"][row], remapped_question
                )
                ranks = arrays["rank_letters"][qi].astype(str).tolist()
                sources: list[list[int]] = []
                queries: list[list[int]] = []
                contents: list[list[int]] = []
                newlines: list[list[int]] = []
                for original in ranks:
                    second_letter = mappings[qid]["original_to_new"][original]
                    source = [left_pad + int(value) for value in first_lines[original]]
                    query = [left_pad + int(value) for value in second_lines[second_letter]]
                    if len(query) < 5 or not source or max(source) >= min(query):
                        raise RuntimeError(f"{qid}: invalid matching 1P/2P option positions")
                    if int(batch["input_ids"][row, query[-1]]) != 198:
                        raise RuntimeError(f"{qid}: expected newline token 198 at end of 2P line")
                    sources.append(source)
                    queries.append(query)
                    contents.append(query[3:-1])
                    newlines.append([query[-1]])
                source_positions.append(sources)
                query_positions.append(queries)
                content_positions.append(contents)
                newline_positions.append(newlines)
                arrays["prompt_hashes"][condition_index, qi] = _hash_prompt(batch["prompts"][row])
                if arrays["prompt_hashes"][condition_index, qi] != trusted[condition_index][qid]["prompt_hash"]:
                    raise RuntimeError(f"{qid}: {condition} trusted prompt hash mismatch")
                row_audits.append(
                    {
                        "question_id": qid,
                        "rank_letters": ranks,
                        "first": first_audit,
                        "second": second_audit,
                        "content_positions_padded": contents,
                        "newline_positions_padded": newlines,
                    }
                )

            matching_specs = _specs(
                ORDINARY_LAYERS, source_positions, query_positions, tuple(range(4)), False
            )
            dose_schedules: dict[str, np.ndarray] = {}
            for scenario_index, scenario in enumerate(SCENARIOS):
                hook = None
                with contextlib.ExitStack() as stack:
                    if scenario != "trusted_natural":
                        stack.enter_context(CompleteSequenceGLA(parts))
                    if scenario in {"identity_hook", "fresh_scrub", "matching_plus_fresh", "dose_matched_random", "matching_plus_random"}:
                        mode = (
                            "identity" if scenario == "identity_hook"
                            else "random" if scenario in {"dose_matched_random", "matching_plus_random"}
                            else "fresh"
                        )
                        hook = stack.enter_context(
                            FreshOptionLineScrubber(
                                parts,
                                content_positions,
                                newline_positions,
                                old_directions,
                                unique_fresh,
                                random_directions,
                                mode,
                                prescribed_dose=(
                                    dose_schedules[
                                        "joint" if scenario == "matching_plus_random" else "fresh"
                                    ]
                                    if mode == "random"
                                    else None
                                ),
                            )
                        )
                    if scenario in {"matching_history_blockade", "matching_plus_fresh", "matching_plus_random"}:
                        stack.enter_context(
                            BatchedSDPAQuerySourceAttentionAblator(parts, matching_specs)
                        )
                    output = _aggregate_logits(
                        _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                        variant_ids,
                    )
                if not np.all(np.isfinite(output)):
                    raise RuntimeError(f"Non-finite logits in {condition}/{scenario}")
                arrays["logits"][condition_index, scenario_index, indices] = output
                if hook is not None:
                    local = hook.arrays()
                    if scenario == "fresh_scrub":
                        dose_schedules["fresh"] = local["dose_l2"]
                    elif scenario == "matching_plus_fresh":
                        dose_schedules["joint"] = local["dose_l2"]
                    for row, qi in enumerate(indices):
                        for name in ("pre_fresh", "post_fresh", "pre_old", "post_old", "dose_l2"):
                            arrays[name][condition_index, scenario_index, qi] = local[name][:, row]

            for row, qid in enumerate(cohort):
                qi = qid_index[qid]
                reference = np.asarray(trusted[condition_index][qid]["aggregated_ad_logits"], dtype=np.float32)
                arrays["trusted_max_abs_error"][condition_index, qi] = float(
                    np.max(np.abs(arrays["logits"][condition_index, TRUSTED_NATURAL, qi] - reference))
                )
            if not audit_path.exists():
                audit_path.write_text(
                    json.dumps(
                        {
                            "condition": condition,
                            "rendered_prompt": batch["prompts"][0],
                            "prompt_hash": arrays["prompt_hashes"][condition_index, indices[0]].item(),
                            "rows": row_audits,
                            "ordinary_attention_layers_one_based": [value + 1 for value in ORDINARY_LAYERS],
                            "scrub_layers_one_based": [value + 1 for value in LAYERS],
                            "groups": list(GROUPS),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        pending.difference_update(cohort)
        completed_cohorts += 1
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        elapsed = time.monotonic() - started
        remaining = sum(bool(set(qids[offset : offset + 4]) & pending) for offset in range(0, len(qids), 4))
        eta = elapsed / completed_cohorts * remaining
        print(
            f"fresh-history double dissociation: {int(arrays['completed'].sum())}/500; "
            f"cohort={duration:.2f}s elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
            flush=True,
        )
        if args.max_cohorts is not None and completed_cohorts >= int(args.max_cohorts):
            print(f"Stopped after {args.max_cohorts} benchmark cohorts", flush=True)
            break

    completed = arrays["completed"]
    identity_error = float(
        np.nanmax(
            np.abs(
                arrays["logits"][:, IDENTITY, completed]
                - arrays["logits"][:, COMPLETE_NATURAL, completed]
            )
        )
    )
    metadata = {
        "experiment": "fresh-2P by matching-history double dissociation",
        "config": config.as_dict(),
        "n_questions": len(qids),
        "conditions": list(CONDITIONS),
        "scenarios": list(SCENARIOS),
        "groups": list(GROUPS),
        "ordinary_attention_layers_one_based": [value + 1 for value in ORDINARY_LAYERS],
        "scrub_layers_one_based": [value + 1 for value in LAYERS],
        "complete_model_forwards_per_cohort": 16,
        "complete_model_work": "Per task: one trusted native natural plus seven complete-sequence forwards (natural, identity, fresh, matching, joint, random, matching+random).",
        "complete": bool(completed.all()),
        "natural_validation": {
            "max_abs_trusted_logit_error": float(np.nanmax(arrays["trusted_max_abs_error"][:, completed])),
        },
        "identity_validation": {
            "max_abs_complete_sequence_natural_logit_error": identity_error,
        },
        "direction_geometry": geometry,
        "score_directions_path": str(args.score_directions),
        "score_directions_sha256": hashlib.sha256(args.score_directions.read_bytes()).hexdigest(),
        "elapsed_seconds_after_model_load": time.monotonic() - started,
        "cohort_seconds": durations,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--score-directions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
