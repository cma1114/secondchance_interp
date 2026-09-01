from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .analyze_second_presentation_score_residuals import (
    _align_remapped,
    _correlation,
    _direction_and_projection,
    _position_controls,
    _residualize_target,
    _rms_normalize,
)
from .collect_cross_model_behavioral_gate import (
    _assert_prompt_pair,
    _remap_question,
    _scenario_messages,
)
from .config import ExperimentConfig
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .run_fixed_a_final_query_edge_ablation import _option_line_positions
from .run_seed_oss_matching_history_blockade import (
    ATTENTION_LAYERS_ONE_BASED,
    MODEL_ID,
    MODEL_REVISION,
    _aggregate_final_logits,
    _forward_final_logits,
)


GROUPS = ("semantic_wordpieces", "option_newline")
CONDITIONS = ("Game", "Neutral")
TRUSTED_SCENARIOS = ("incorrect_again_remapped", "lost_again_remapped")
TARGETS = ("old_unique", "fresh_unique")


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class SummaryCapture:
    """Capture two candidate summaries after every Seed block.

    Positions are batch-padded absolute columns.  Summaries are reduced on the
    layer device and only the small BF16 means are transferred to CPU.
    """

    def __init__(self, parts: Any, positions: list[tuple[list[list[int]], list[list[int]]]]):
        import torch

        self.positions = positions
        self.values: list[Any] = [None] * len(parts.layers)
        self.handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(parts.layers)
        ]
        self.torch = torch

    def _hook(self, index: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            rows = []
            for row, grouped in enumerate(self.positions):
                candidates = []
                for candidate in range(4):
                    summaries = []
                    for group in grouped:
                        columns = self.torch.as_tensor(
                            group[candidate], device=hidden.device, dtype=self.torch.long
                        )
                        summaries.append(hidden[row].index_select(0, columns).float().mean(0))
                    candidates.append(self.torch.stack(summaries, dim=0))
                rows.append(self.torch.stack(candidates, dim=0))
            self.values[index] = self.torch.stack(rows, dim=0).detach().to(
                "cpu", dtype=self.torch.bfloat16
            )

        return capture

    def stacked(self):
        if any(value is None for value in self.values):
            missing = [index + 1 for index, value in enumerate(self.values) if value is None]
            raise RuntimeError(f"Hooks did not capture Seed layers {missing}")
        # batch x layer x candidate x group x width
        return self.torch.stack(self.values, dim=1)

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _positions(
    tokenizer: Any,
    prompt: str,
    original: dict[str, Any],
    remapped: dict[str, Any],
    mapping: dict[str, Any],
    left_pad: int,
) -> tuple[tuple[list[list[int]], list[list[int]]], dict[str, Any]]:
    del original  # The second-presentation spans are located in `remapped`.
    second, audit = _option_line_positions(tokenizer, prompt, remapped)
    semantic: list[list[int]] = []
    newline: list[list[int]] = []
    for original_letter in LETTERS:
        displayed = mapping["original_to_new"][original_letter]
        raw_line = [int(value) for value in second[displayed]]
        line = [left_pad + value for value in raw_line]
        # Seed's audited native tokenization is: two indentation/letter tokens,
        # colon, one-or-more option-content wordpieces, standalone newline.
        if len(line) < 5:
            raise RuntimeError("Unexpectedly short Seed 2P option line")
        token_strings = audit[displayed]["tokens"]
        if len(token_strings) != len(line):
            raise RuntimeError("2P token audit and positions disagree")
        if "\n" not in tokenizer.decode(
            tokenizer(prompt, add_special_tokens=False)["input_ids"][raw_line[-1] : raw_line[-1] + 1]
        ):
            raise RuntimeError("Seed 2P boundary is not a standalone newline token")
        semantic.append(line[3:-1])
        newline.append([line[-1]])
    return (semantic, newline), audit


def _load_discovery(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    values = payload.get("question_ids", payload.get("discovery_question_ids"))
    if values is None:
        raise ValueError("Split artifact has neither question_ids nor discovery_question_ids")
    return {str(value) for value in values}


def _load_targets(
    qids: list[str],
    trusted: dict[str, Any],
    fresh_payload: dict[str, Any],
    mappings: dict[str, dict[str, Any]],
    discovery: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, list[float]], np.ndarray, np.ndarray]:
    baseline = trusted["scenarios"]["baseline"]
    fresh_rows = fresh_payload["results"]
    if any(qid not in baseline or qid not in fresh_rows for qid in qids):
        raise ValueError("Old or fresh score artifact is missing requested questions")
    old = _center(
        np.asarray([baseline[qid]["aggregated_ad_logits"] for qid in qids], dtype=np.float64)
    )
    fresh_raw = np.asarray(
        [fresh_rows[qid]["aggregated_ad_logits"] for qid in qids], dtype=np.float64
    )
    fresh = _center(_align_remapped(fresh_raw, qids, mappings))
    controls = _position_controls(qids, mappings)
    old_unique, old_coefficients = _residualize_target(old, fresh, controls, discovery)
    fresh_unique, fresh_coefficients = _residualize_target(fresh, old, controls, discovery)
    return (
        (old_unique, fresh_unique),
        {
            "old_unique": old_coefficients.tolist(),
            "fresh_unique": fresh_coefficients.tolist(),
        },
        old,
        fresh,
    )


def _state_slice(states: np.memmap, condition: int, layer: int, group: int):
    import torch

    raw = np.asarray(states[condition, :, layer, :, group]).copy()
    return torch.from_numpy(raw).view(torch.bfloat16)


def _fit_and_validate(
    output_dir: Path,
    states: np.memmap,
    qids: list[str],
    discovery: np.ndarray,
    targets: tuple[np.ndarray, np.ndarray],
    coefficients: dict[str, list[float]],
    old: np.ndarray,
    fresh: np.ndarray,
    width: int,
) -> dict[str, Any]:
    import torch

    confirmation = ~discovery
    directions = torch.empty((64, 2, 2, width), dtype=torch.float32)
    projections = np.empty((2, len(qids), 64, 4, 2, 2), dtype=np.float32)
    trajectory: dict[str, Any] = {}
    for layer in range(64):
        layer_rows: dict[str, Any] = {}
        for group_index, group_name in enumerate(GROUPS):
            game = _state_slice(states, 0, layer, group_index)
            neutral = _state_slice(states, 1, layer, group_index)
            source = (game.float() + neutral.float()) / 2.0
            group_rows: dict[str, Any] = {}
            for target_index, (target_name, target) in enumerate(zip(TARGETS, targets)):
                direction, shared = _direction_and_projection(source, target, discovery)
                directions[layer, group_index, target_index] = direction
                rows: dict[str, Any] = {
                    "shared_discovery_correlation": _correlation(
                        shared[discovery], target[discovery]
                    ),
                    "shared_confirmation_correlation": _correlation(
                        shared[confirmation], target[confirmation]
                    ),
                }
                for condition_index, (condition_name, state) in enumerate(
                    zip(CONDITIONS, (game, neutral))
                ):
                    normalized = _rms_normalize(state)
                    normalized -= normalized.mean(1, keepdim=True)
                    projection = torch.einsum("qcd,d->qc", normalized, direction).numpy()
                    projections[
                        condition_index, :, layer, :, group_index, target_index
                    ] = projection
                    rows[condition_name] = {
                        "discovery_correlation": _correlation(
                            projection[discovery], target[discovery]
                        ),
                        "confirmation_correlation": _correlation(
                            projection[confirmation], target[confirmation]
                        ),
                    }
                group_rows[target_name] = rows
            layer_rows[group_name] = group_rows
        trajectory[str(layer + 1)] = layer_rows
        if layer == 0 or (layer + 1) % 8 == 0:
            print(f"DIRECTION_FIT layer={layer + 1}/64", flush=True)

    result = {
        "complete": True,
        "definition": {
            "old_score": "candidate-centered Seed standalone 1P A-D logit",
            "fresh_score": "candidate-centered Seed standalone remapped A-D logit aligned to semantic identity",
            "state": "post-block residual mean at each 2P option's semantic wordpieces or boundary newline",
            "fit": "condition-mean RMS-normalized candidate-centered state; old/fresh scores mutually residualized with 1P/2P displayed-position controls; fit on discovery only",
        },
        "validation": {
            "questions": len(qids),
            "discovery": int(discovery.sum()),
            "confirmation": int(confirmation.sum()),
            "layers": 64,
            "groups": list(GROUPS),
            "model_width": width,
        },
        "target_control_coefficients": coefficients,
        "trajectory": trajectory,
    }
    _atomic_json(output_dir / "score_direction_validation.json", result)
    torch.save(directions.to(torch.float16), output_dir / "score_directions.pt")
    temporary = output_dir / "score_projections.tmp.npz"
    np.savez_compressed(
        temporary,
        question_ids=np.asarray(qids),
        discovery=discovery,
        old_score=old.astype(np.float32),
        fresh_score=fresh.astype(np.float32),
        old_unique=targets[0].astype(np.float32),
        fresh_unique=targets[1].astype(np.float32),
        projections=projections.astype(np.float16),
    )
    os.replace(temporary, output_dir / "score_projections.npz")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--trusted-behavior", type=Path, required=True)
    parser.add_argument("--fresh-score-results", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    return parser


def run(args: argparse.Namespace) -> None:
    import torch

    config = ExperimentConfig.load(args.config)
    if config.model_id != MODEL_ID or config.model_revision != MODEL_REVISION:
        raise ValueError("Requires the pinned Seed-OSS 36B revision")
    if (
        config.chat_serialization != "hf_template"
        or config.model_loader != "causal_lm"
        or config.attn_implementation != "sdpa"
        or config.batch_size != 4
    ):
        raise ValueError("Requires Seed causal_lm + native HF template + SDPA + batch 4")
    manifest = json.loads(Path(config.manifest_path).read_text())["questions"]
    questions = {str(row["id"]): row for row in manifest}
    qids = [str(row["id"]) for row in manifest]
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    trusted = json.loads(args.trusted_behavior.read_text())
    fresh_payload = json.loads(args.fresh_score_results.read_text())
    discovery_ids = _load_discovery(args.discovery_plan)
    if len(qids) != 500 or set(qids) != set(mappings):
        raise RuntimeError("Expected the complete frozen 500-question inventory")
    discovery = np.asarray([qid in discovery_ids for qid in qids], dtype=bool)
    if not discovery.any() or discovery.all():
        raise RuntimeError("Frozen discovery/confirmation split is degenerate")
    trusted_tasks = [trusted["scenarios"][name] for name in TRUSTED_SCENARIOS]
    if any(qid not in rows for rows in trusted_tasks for qid in qids):
        raise ValueError("Trusted Game/Neutral artifact is incomplete")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = args.output_dir / "completed.npy"
    completed = (
        np.load(completed_path).astype(bool)
        if completed_path.exists()
        else np.zeros(500, dtype=bool)
    )
    if completed.shape != (500,):
        raise RuntimeError("Existing completion checkpoint has the wrong shape")

    load_started = time.monotonic()
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _text, token_id in resolved[letter]})
        for letter in LETTERS
    }
    if [
        i + 1 for i, layer in enumerate(parts.layers) if getattr(layer, "self_attn", None)
    ] != list(ATTENTION_LAYERS_ONE_BASED):
        raise RuntimeError("Unexpected Seed layer inventory")
    width = int(parts.layers[0].input_layernorm.weight.numel())
    state_path = args.output_dir / "summary_states_bf16.npy"
    shape = (2, 500, 64, 4, 2, width)
    states = np.lib.format.open_memmap(
        state_path,
        mode="r+" if state_path.exists() else "w+",
        dtype=np.uint16,
        shape=shape,
    )
    print(f"MODEL_LOADED seconds={time.monotonic() - load_started:.3f}", flush=True)

    prompt_audit: dict[str, Any] = {}
    durations: list[float] = []
    cohort_limit = 125 if args.max_cohorts is None else min(125, int(args.max_cohorts))
    for cohort_index in range(cohort_limit):
        start = cohort_index * 4
        indices = list(range(start, start + 4))
        if completed[indices].all():
            continue
        cohort_qids = [qids[index] for index in indices]
        cohort_started = time.monotonic()
        prompts_by_condition: list[list[str]] = []
        remapped_by_condition: list[list[dict[str, Any]]] = []
        for scenario in TRUSTED_SCENARIOS:
            prompts: list[str] = []
            remapped_questions: list[dict[str, Any]] = []
            for qid in cohort_qids:
                mapping = mappings[qid]
                messages, remapped = _scenario_messages(
                    scenario, questions[qid], mapping["new_to_original"]
                )
                if remapped is None:
                    raise RuntimeError("Direction collector requires remapped 2P prompts")
                prompt = render_chat(
                    processor,
                    messages,
                    config.disable_thinking,
                    config.chat_serialization,
                    config.chat_template_kwargs,
                )
                expected_hash = trusted["scenarios"][scenario][qid]["prompt_hash"]
                if _prompt_hash(prompt) != expected_hash:
                    raise RuntimeError(f"Trusted prompt mismatch for {scenario} {qid}")
                prompts.append(prompt)
                remapped_questions.append(remapped)
            prompts_by_condition.append(prompts)
            remapped_by_condition.append(remapped_questions)
        for row in range(4):
            _assert_prompt_pair(prompts_by_condition[0][row], prompts_by_condition[1][row])

        for condition_index, prompts in enumerate(prompts_by_condition):
            input_ids, attention_mask, _last_indices = tokenize_batch(tokenizer, prompts)
            positions = []
            audits = []
            for row, (qid, prompt, remapped) in enumerate(
                zip(cohort_qids, prompts, remapped_by_condition[condition_index])
            ):
                unpadded = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                left_pad = int(input_ids.shape[1] - len(unpadded))
                grouped, audit = _positions(
                    tokenizer,
                    prompt,
                    questions[qid],
                    remapped,
                    mappings[qid],
                    left_pad,
                )
                positions.append(grouped)
                audits.append(audit)
            capture = SummaryCapture(parts, positions)
            try:
                final = _forward_final_logits(model, parts, input_ids, attention_mask)
                summary = capture.stacked()
            finally:
                capture.close()
            logits = _aggregate_final_logits(final, variant_ids)
            trusted_logits = np.asarray(
                [trusted_tasks[condition_index][qid]["aggregated_ad_logits"] for qid in cohort_qids],
                dtype=np.float32,
            )
            if not np.array_equal(logits, trusted_logits):
                error = float(np.max(np.abs(logits - trusted_logits)))
                raise RuntimeError(f"Natural reproduction failed: max error {error}")
            states[condition_index, indices] = summary.view(torch.uint16).numpy()
            if cohort_index == 0:
                prompt_audit[CONDITIONS[condition_index]] = {
                    "question_ids": cohort_qids,
                    "prompt_hashes": [_prompt_hash(value) for value in prompts],
                    "second_option_lines": audits,
                    "semantic_position_counts": [
                        [len(group) for group in positions[row][0]] for row in range(4)
                    ],
                    "newline_position_counts": [
                        [len(group) for group in positions[row][1]] for row in range(4)
                    ],
                }
        states.flush()
        completed[indices] = True
        _atomic_npy(completed_path, completed)
        durations.append(time.monotonic() - cohort_started)
        print(
            f"COHORT_COMPLETE cohort={cohort_index + 1}/{cohort_limit} "
            f"questions={int(completed.sum())}/500 seconds={durations[-1]:.3f}",
            flush=True,
        )

    if prompt_audit:
        _atomic_json(args.output_dir / "prompt_audit.json", prompt_audit)
    if args.max_cohorts is not None:
        _atomic_json(
            args.output_dir / "benchmark.json",
            {
                "complete": False,
                "cohorts": cohort_limit,
                "complete_forwards_per_cohort": 2,
                "cohort_seconds": durations,
                "mean_cohort_seconds": float(np.mean(durations)) if durations else None,
                "projected_seconds_125_cohorts": (
                    float(np.mean(durations) * 125) if durations else None
                ),
                "natural_reproduction_error": 0.0,
            },
        )
        return
    if not completed.all():
        raise RuntimeError("Full collector ended before all 500 questions completed")

    targets, coefficients, old, fresh = _load_targets(
        qids, trusted, fresh_payload, mappings, discovery
    )
    validation = _fit_and_validate(
        args.output_dir,
        states,
        qids,
        discovery,
        targets,
        coefficients,
        old,
        fresh,
        width,
    )
    _atomic_json(
        args.output_dir / "run_metadata.json",
        {
            "experiment": "Seed-OSS all-layer 2P old/fresh score directions",
            "complete": True,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "questions": 500,
            "conditions": list(CONDITIONS),
            "layers_one_based": list(ATTENTION_LAYERS_ONE_BASED),
            "groups": list(GROUPS),
            "complete_forwards_per_cohort": 2,
            "natural_reproduction_error": 0.0,
            "cohort_seconds": durations,
            "validation": validation["validation"],
            "paths": {
                "config": str(args.config),
                "remapping_plan": str(args.remapping_plan),
                "trusted_behavior": str(args.trusted_behavior),
                "fresh_scores": str(args.fresh_score_results),
                "discovery_plan": str(args.discovery_plan),
            },
        },
    )
    print(json.dumps({"complete": True, "questions": 500, "layers": 64}), flush=True)


if __name__ == "__main__":
    run(build_parser().parse_args())
