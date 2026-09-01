from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import prompt_hash
from .run_first_decision_cross_order_patching import _aggregate_logits, _decision_position
from .run_option_newline_all_four_projection import (
    _load_decision_answer_bases,
    _trusted,
)
from .run_semantic_binding_module_factorial import _messages, _remap_question


LETTERS = "ABCD"
CONDITIONS = ("incorrect_again", "lost_again")
CONDITION_NAMES = ("game", "neutral")
MODES = ("natural", "identity_hook", "continuous_letter_scrub")
NATURAL, IDENTITY, SCRUB = range(3)
EXECUTION_BATCH_SIZE = 4
TARGET_READOUTS = tuple(range(48, 64))


def _initialize(path: Path, qids: list[str], split: np.ndarray) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses another question order")
        return arrays
    n = len(qids)
    shape = (len(CONDITIONS), len(MODES), n)
    layer_shape = shape + (64,)
    return {
        "question_ids": np.asarray(qids),
        "split": split,
        "completed": np.zeros(n, dtype=bool),
        "logits": np.full(shape + (4,), np.nan, dtype=np.float32),
        "pre_ad_norm": np.full(layer_shape, np.nan, dtype=np.float32),
        "post_ad_norm": np.full(layer_shape, np.nan, dtype=np.float32),
        "residual_norm": np.full(layer_shape, np.nan, dtype=np.float32),
        "dose_l2": np.full(layer_shape, np.nan, dtype=np.float32),
        "trusted_max_abs_error": np.full((2, n), np.nan, dtype=np.float32),
        "trusted_choice_match": np.zeros((2, n), dtype=bool),
        "first_decision_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "first_decision_matches_baseline": np.zeros(n, dtype=bool),
    }


class ContinuousDecisionLetterScrub:
    """Remove centered A--D identity from one live token at every readout."""

    def __init__(
        self,
        parts: Any,
        positions: list[int],
        bases: dict[int, Any],
        project: bool,
    ) -> None:
        import torch

        self.positions = tuple(int(value) for value in positions)
        self.bases = {key: value.float().cpu() for key, value in bases.items()}
        self.project = bool(project)
        self.pre = torch.full((64, len(positions)), float("nan"))
        self.post = torch.full((64, len(positions)), float("nan"))
        self.norm = torch.full((64, len(positions)), float("nan"))
        self.dose = torch.full((64, len(positions)), float("nan"))
        # parts.layers[r] consumes post-block readout r as the input to block r+1.
        self.handles = [
            parts.layers[readout].register_forward_pre_hook(self._hook(readout))
            for readout in TARGET_READOUTS
        ]

    def _hook(self, readout: int):
        def intervene(_module: Any, inputs: Any) -> Any:
            import torch

            hidden = inputs[0]
            if hidden.ndim != 3:
                raise RuntimeError(f"Readout {readout}: unexpected hidden shape {hidden.shape}")
            rows = torch.arange(len(self.positions), device=hidden.device)
            columns = torch.as_tensor(self.positions, device=hidden.device)
            current = hidden[rows, columns].float()
            basis = self.bases[readout - 1].to(current.device)
            coefficients = current @ basis.T
            carrier = current - coefficients @ basis
            post_coefficients = carrier @ basis.T
            if not torch.isfinite(carrier).all():
                raise RuntimeError(f"Readout {readout}: non-finite projected residual")
            self.pre[readout] = coefficients.norm(dim=-1).detach().cpu()
            self.norm[readout] = current.norm(dim=-1).detach().cpu()
            if self.project:
                self.post[readout] = post_coefficients.norm(dim=-1).detach().cpu()
                self.dose[readout] = (carrier - current).norm(dim=-1).detach().cpu()
                changed = hidden.clone()
                changed[rows, columns] = carrier.to(hidden.dtype)
                return (changed, *inputs[1:])
            self.post[readout] = coefficients.norm(dim=-1).detach().cpu()
            self.dose[readout] = 0.0
            return None

        return intervene

    def arrays(self) -> dict[str, np.ndarray]:
        targets = np.asarray(TARGET_READOUTS, dtype=np.int64)
        values = {
            "pre_ad_norm": self.pre.numpy(),
            "post_ad_norm": self.post.numpy(),
            "residual_norm": self.norm.numpy(),
            "dose_l2": self.dose.numpy(),
        }
        for name, value in values.items():
            if not np.all(np.isfinite(value[targets])):
                raise RuntimeError(f"Missing/non-finite continuous scrub audit: {name}")
        if self.project and float(np.max(values["post_ad_norm"][targets])) > 2e-3:
            raise RuntimeError("A-D projection did not numerically zero the target subspace")
        return values

    def close(self) -> None:
        for handle in reversed(getattr(self, "handles", [])):
            handle.remove()
        self.handles = []

    def __enter__(self) -> "ContinuousDecisionLetterScrub":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _padded_rows(rows: list[list[int]], pad_id: int) -> tuple[Any, Any]:
    import torch

    width = max(len(row) for row in rows)
    ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        offset = width - len(row)
        ids[index, offset:] = torch.as_tensor(row)
        mask[index, offset:] = 1
    return ids, mask


def run(
    config_path: Path,
    discovery_plan_path: Path,
    second_mapping_path: Path,
    baseline_path: Path,
    trusted_game_path: Path,
    trusted_neutral_path: Path,
    output_dir: Path,
    max_cohorts: int | None,
    lens_repo: str,
    lens_filename: str,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
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
    discovery_ids = set(json.loads(discovery_plan_path.read_text())["question_ids"])
    if len(discovery_ids) != 251 or len(set(qids) - discovery_ids) != 249:
        raise ValueError("Expected frozen 251/249 question split")
    split = np.asarray(["discovery" if qid in discovery_ids else "confirmation" for qid in qids])
    baseline = json.loads(baseline_path.read_text())["results"]
    second_rows = {
        row["question_id"]: row
        for row in json.loads(second_mapping_path.read_text())["rows"]
    }
    trusted = {
        "game": _trusted(trusted_game_path),
        "neutral": _trusted(trusted_neutral_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids, split)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    canonical_ids = [resolved[letter][0][1] for letter in LETTERS]
    bases, basis_diagnostics = _load_decision_answer_bases(
        parts,
        canonical_ids,
        lens_repo,
        lens_filename,
        TARGET_READOUTS,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    audit_path = output_dir / "prompt_audit.json"
    started = time.monotonic()
    total_model_calls = 0
    durations: list[float] = []
    completed_cohorts = 0
    pending = {qid for qid, done in zip(qids, arrays["completed"]) if not bool(done)}
    total_cohorts = sum(
        bool(set(qids[start : start + 4]) & pending) for start in range(0, 500, 4)
    )

    for start in range(0, 500, EXECUTION_BATCH_SIZE):
        group_qids = qids[start : start + EXECUTION_BATCH_SIZE]
        if not set(group_qids) & pending:
            continue
        cohort_started = time.monotonic()
        canonical_prefix_rows: list[list[int]] | None = None

        for condition_index, (condition, condition_name) in enumerate(zip(CONDITIONS, CONDITION_NAMES)):
            prompts: list[str] = []
            token_rows: list[list[int]] = []
            boundaries: list[int] = []
            lengths: list[int] = []
            for qid in group_qids:
                second = _remap_question(questions[qid], second_rows[qid]["new_to_original"])
                prompt = render_chat(
                    processor,
                    _messages(config, questions[qid], second, condition),
                    config.disable_thinking,
                    config.chat_serialization,
                )
                ids = [int(value) for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
                boundary, boundary_ids = _decision_position(tokenizer, prompt)
                if ids != [int(value) for value in boundary_ids]:
                    raise RuntimeError(f"{qid}: decision-boundary tokenization mismatch")
                if prompt_hash(prompt) != trusted[condition_name][qid]["prompt_hash"]:
                    raise RuntimeError(f"{qid}: {condition_name} prompt hash mismatch")
                prompts.append(prompt)
                token_rows.append(ids)
                boundaries.append(int(boundary))
                lengths.append(len(ids))

            input_ids, attention_mask, _last_indices = tokenize_batch(tokenizer, prompts)
            width = int(input_ids.shape[1])
            padded_boundaries = [boundary + width - length for boundary, length in zip(boundaries, lengths)]
            boundary_tokens = [tokenizer.decode([int(input_ids[row, position])]) for row, position in enumerate(padded_boundaries)]
            if len(set(boundary_tokens)) != 1:
                raise RuntimeError(f"First-decision tokens differ within cohort: {boundary_tokens}")

            prefix_rows = [ids[: boundary + 1] for ids, boundary in zip(token_rows, boundaries)]
            if condition_index == 0:
                canonical_prefix_rows = [list(row) for row in prefix_rows]
                prefix_ids, prefix_mask = _padded_rows(prefix_rows, int(pad_id))
                with torch.inference_mode():
                    kwargs = dict(
                        input_ids=prefix_ids.to(model_input_device(parts)),
                        attention_mask=prefix_mask.to(model_input_device(parts)),
                        use_cache=False,
                        return_dict=True,
                    )
                    try:
                        prefix_output = model(**kwargs, logits_to_keep=1)
                    except TypeError:
                        prefix_output = model(**kwargs)
                total_model_calls += 1
                prefix_final = prefix_output.logits[:, 0] if int(prefix_output.logits.shape[1]) == 1 else prefix_output.logits[:, -1]
                first_logits = _aggregate_logits(prefix_final.detach().float(), variant_ids).cpu().numpy()
                for row, qid in enumerate(group_qids):
                    qi = qids.index(qid)
                    arrays["first_decision_logits"][qi] = first_logits[row]
                    arrays["first_decision_matches_baseline"][qi] = bool(
                        LETTERS[int(np.argmax(first_logits[row]))] == baseline[qid]["answer"]
                    )
            elif canonical_prefix_rows != [list(row) for row in prefix_rows]:
                raise RuntimeError("Game and Neutral prefixes differ through first decision")

            for mode_index, mode in enumerate(MODES):
                hook = None
                with contextlib.ExitStack() as stack:
                    if mode != "natural":
                        hook = stack.enter_context(
                            ContinuousDecisionLetterScrub(
                                parts,
                                padded_boundaries,
                                bases,
                                project=(mode == "continuous_letter_scrub"),
                            )
                        )
                    with torch.inference_mode():
                        kwargs = dict(
                            input_ids=input_ids.to(model_input_device(parts)),
                            attention_mask=attention_mask.to(model_input_device(parts)),
                            use_cache=False,
                            return_dict=True,
                        )
                        try:
                            output = model(**kwargs, logits_to_keep=1)
                        except TypeError:
                            output = model(**kwargs)
                total_model_calls += 1
                final = output.logits[:, 0] if int(output.logits.shape[1]) == 1 else output.logits[:, -1]
                logits = _aggregate_logits(final.detach().float(), variant_ids).cpu().numpy()
                local = hook.arrays() if hook is not None else None
                for row, qid in enumerate(group_qids):
                    qi = qids.index(qid)
                    arrays["logits"][condition_index, mode_index, qi] = logits[row]
                    if local is not None:
                        for name in ("pre_ad_norm", "post_ad_norm", "residual_norm", "dose_l2"):
                            arrays[name][condition_index, mode_index, qi] = local[name][:, row]
                    if mode == "natural":
                        reference = np.asarray(trusted[condition_name][qid]["aggregated_ad_logits"], dtype=np.float32)
                        arrays["trusted_max_abs_error"][condition_index, qi] = float(np.max(np.abs(logits[row] - reference)))
                        arrays["trusted_choice_match"][condition_index, qi] = bool(np.argmax(logits[row]) == np.argmax(reference))

                if mode == "continuous_letter_scrub" and not audit_path.exists() and local is not None:
                    audit_path.write_text(json.dumps({
                        "question_id": group_qids[0],
                        "condition": condition_name,
                        "historical_group_qids": group_qids,
                        "prompt_hash": prompt_hash(prompts[0]),
                        "decision_position_unpadded": boundaries[0],
                        "decision_position_padded": padded_boundaries[0],
                        "decision_token": boundary_tokens[0],
                        "target_readouts": list(TARGET_READOUTS),
                        "pre_ad_norm": local["pre_ad_norm"][list(TARGET_READOUTS), 0].tolist(),
                        "post_ad_norm": local["post_ad_norm"][list(TARGET_READOUTS), 0].tolist(),
                        "dose_l2": local["dose_l2"][list(TARGET_READOUTS), 0].tolist(),
                    }, indent=2, sort_keys=True) + "\n")

        for qid in group_qids:
            arrays["completed"][qids.index(qid)] = True
        atomic_save_npz(result_path, **arrays)
        pending.difference_update(group_qids)
        completed_cohorts += 1
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        elapsed = time.monotonic() - started
        eta = elapsed / completed_cohorts * (total_cohorts - completed_cohorts)
        print(
            f"continuous decision-letter scrub: {int(arrays['completed'].sum())}/500; "
            f"cohort={duration:.1f}s elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
            flush=True,
        )
        if max_cohorts is not None and completed_cohorts >= max_cohorts:
            print(f"Stopped after {max_cohorts} benchmark cohorts", flush=True)
            break

    completed = arrays["completed"]
    target = np.asarray(TARGET_READOUTS, dtype=np.int64)
    metadata = {
        "experiment": "Continuous first-decision centered A-D residual scrub",
        "config": config.as_dict(),
        "n_questions": len(qids),
        "conditions": list(CONDITION_NAMES),
        "prompt_conditions": list(CONDITIONS),
        "modes": list(MODES),
        "target_readouts": list(TARGET_READOUTS),
        "intervention_position": "exact empty first-presentation decision token",
        "intervention_geometry": (
            "At each readout 48-63, orthogonally subtract the complete centered "
            "three-dimensional A-D JLens decoder subspace from the live residual "
            "before the next block. Repeating the edit removes any A-D component "
            "reconstructed by intervening blocks."
        ),
        "execution_batch_size": EXECUTION_BATCH_SIZE,
        "complete_prompt_executions_per_cohort": 6,
        "prefix_executions_per_cohort": 1,
        "complete_model_forwards_per_cohort": 7,
        "model_forward_calls_total": total_model_calls,
        "mean_model_forward_calls_per_completed_cohort": total_model_calls / completed_cohorts if completed_cohorts else None,
        "complete": bool(completed.all()),
        "natural_validation": {
            "max_abs_trusted_logit_error": float(np.nanmax(arrays["trusted_max_abs_error"][:, completed])),
            "trusted_choice_agreement": float(arrays["trusted_choice_match"][:, completed].mean()),
        },
        "identity_validation": {
            "max_abs_natural_logit_error": float(np.nanmax(np.abs(arrays["logits"][:, IDENTITY, completed] - arrays["logits"][:, NATURAL, completed]))),
            "choice_changes": int(np.sum(arrays["logits"][:, IDENTITY, completed].argmax(-1) != arrays["logits"][:, NATURAL, completed].argmax(-1))),
        },
        "scrub_validation": {
            "max_post_ad_norm": float(np.nanmax(arrays["post_ad_norm"][:, SCRUB, completed][:, :, target])),
            "mean_dose_l2": float(np.nanmean(arrays["dose_l2"][:, SCRUB, completed][:, :, target])),
        },
        "first_decision_baseline_choice_agreement": float(arrays["first_decision_matches_baseline"][completed].mean()),
        "elapsed_seconds_after_model_load": time.monotonic() - started,
        "completed_cohort_durations_seconds": durations,
        "resolved_answer_tokens": resolved,
        "jlens": {"repo": lens_repo, "filename": lens_filename, "diagnostics": basis_diagnostics},
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--second-mapping", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument("--lens-filename", default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")
    args = parser.parse_args()
    run(
        args.config,
        args.discovery_plan,
        args.second_mapping,
        args.baseline,
        args.trusted_game,
        args.trusted_neutral,
        args.output_dir,
        args.max_cohorts,
        args.lens_repo,
        args.lens_filename,
    )


if __name__ == "__main__":
    main()
