from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .collect_contextual_option_representations import (
    ANCHORS,
    BatchedPositionCollector,
    _positions,
)
from .collect_remapped_behavior import _messages, _remap_question
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
from .prompts import build_messages, prompt_hash
from .run_first_decision_cross_order_patching import (
    CONDITIONS,
    LETTERS,
    _aggregate_logits,
    _load_mapping_plans,
    _question_ids,
)
from .run_historical_answer_intervention import _forward


def _hidden(output: Any):
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_hidden(output: Any, hidden: Any):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


class FinalDecisionSemanticHook:
    """Measure and optionally erase target-answer semantics at the final position.

    Directions are question- and readout-specific unit vectors with shape
    ``[n_layers, batch, hidden]``. Only ``selected_rows`` are modified, so each
    target retains its exact historical four-question physical cohort.
    """

    def __init__(
        self,
        parts: Any,
        directions: np.ndarray,
        selected_rows: list[int],
        intervention_mode: str,
    ) -> None:
        import torch

        if directions.ndim != 3 or directions.shape[0] != len(parts.layers):
            raise ValueError("Semantic directions must be [layer, batch, hidden]")
        self.selected_rows = tuple(sorted(set(int(row) for row in selected_rows)))
        if not self.selected_rows:
            raise ValueError("At least one batch row must be selected")
        if intervention_mode not in {"none", "signed", "positive_only", "negative_only"}:
            raise ValueError(f"Unknown intervention mode: {intervention_mode}")
        self.intervention_mode = intervention_mode
        self.directions = torch.from_numpy(directions).float().to(
            model_input_device(parts)
        )
        self.coefficients: list[Any] = [None] * len(parts.layers)
        self.residual_norms: list[Any] = [None] * len(parts.layers)
        self.after_coefficients: list[Any] = [None] * len(parts.layers)
        self.handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(parts.layers)
        ]

    def _hook(self, index: int):
        def intervene(_module: Any, _inputs: Any, output: Any) -> Any:
            import torch

            hidden = _hidden(output)
            current = hidden[:, -1].float()
            direction = self.directions[index].to(current.device)
            coefficients = (current * direction).sum(dim=-1)
            norms = current.norm(dim=-1)
            self.coefficients[index] = coefficients.detach().cpu()
            self.residual_norms[index] = norms.detach().cpu()
            if self.intervention_mode == "none":
                self.after_coefficients[index] = coefficients.detach().cpu()
                return output

            changed = hidden.clone()
            rows = torch.as_tensor(self.selected_rows, device=current.device)
            removal = coefficients[rows]
            if self.intervention_mode == "positive_only":
                removal = torch.clamp(removal, min=0)
            elif self.intervention_mode == "negative_only":
                removal = torch.clamp(removal, max=0)
            updated = current[rows] - removal[:, None] * direction[rows]
            changed[rows, -1] = updated.to(hidden.dtype)
            actual_after = (
                changed[:, -1].float() * direction
            ).sum(dim=-1)
            self.after_coefficients[index] = actual_after.detach().cpu()
            return _replace_hidden(output, changed)

        return intervene

    def arrays(self) -> dict[str, np.ndarray]:
        import torch

        if any(value is None for value in self.coefficients):
            raise RuntimeError("Not every semantic hook executed")
        return {
            "projection": torch.stack(self.coefficients).numpy(),
            "residual_norm": torch.stack(self.residual_norms).numpy(),
            "projection_after": torch.stack(self.after_coefficients).numpy(),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __enter__(self) -> "FinalDecisionSemanticHook":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _initialize(path: Path, qids: list[str], n_layers: int) -> dict[str, np.ndarray]:
    if path.exists():
        arrays = dict(np.load(path, allow_pickle=False))
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Question IDs changed")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "ablated_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "natural_projection": np.full((2, n, n_layers), np.nan, dtype=np.float32),
        "natural_residual_norm": np.full((2, n, n_layers), np.nan, dtype=np.float32),
        "ablated_pre_projection": np.full((2, n, n_layers), np.nan, dtype=np.float32),
        "ablated_residual_norm": np.full((2, n, n_layers), np.nan, dtype=np.float32),
        "ablated_projection_after": np.full((2, n, n_layers), np.nan, dtype=np.float32),
    }


def _mapping_questions(
    questions: dict[str, dict[str, Any]],
    group_qids: list[str],
    mapping_plans: list[dict[str, dict[str, str]]],
) -> list[tuple[list[dict[str, Any]], list[dict[str, str]]]]:
    identity = {letter: letter for letter in LETTERS}
    outputs = [
        ([questions[qid] for qid in group_qids], [identity for _ in group_qids])
    ]
    for plan in mapping_plans:
        outputs.append(
            (
                [
                    _remap_question(questions[qid], plan[qid]["new_to_original"])
                    for qid in group_qids
                ],
                [plan[qid]["original_to_new"] for qid in group_qids],
            )
        )
    return outputs


def _semantic_directions(
    model: Any,
    processor: Any,
    parts: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    questions: dict[str, dict[str, Any]],
    group_qids: list[str],
    mapping_plans: list[dict[str, dict[str, str]]],
    target_answers: dict[str, str],
) -> tuple[np.ndarray, dict[str, Any]]:
    mapping_values: list[np.ndarray] = []
    mappings: list[list[dict[str, str]]] = []
    line_indices = [ANCHORS.index(f"line_end_{letter}") for letter in LETTERS]
    audit: dict[str, Any] = {"mapping_prompts": []}

    for questions_for_mapping, original_to_new in _mapping_questions(
        questions, group_qids, mapping_plans
    ):
        prompts = []
        positions = []
        lengths = []
        position_audits = []
        for question in questions_for_mapping:
            messages = build_messages(
                question, "baseline", config.prompt_mode, config.feedback_variant
            )
            prompt = render_chat(
                processor, messages, config.disable_thinking, config.chat_serialization
            )
            row_positions, row_audit = _positions(tokenizer, prompt, question)
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompts.append(prompt)
            positions.append(row_positions)
            lengths.append(len(ids))
            position_audits.append(row_audit)
        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        width = int(input_ids.shape[1])
        padded = [
            [int(position) + width - length for position in row]
            for row, length in zip(positions, lengths)
        ]
        collector = BatchedPositionCollector(parts.layers, padded)
        try:
            # Preserve the historical exact-reproduction regime. The first pass
            # is a warm-up; the second pass supplies the retained residuals.
            # Removing the warm-up changes SDPA/bfloat16 numerics materially.
            _forward(model, parts, input_ids, attention_mask)
            _forward(model, parts, input_ids, attention_mask)
            values = collector.stacked().numpy().transpose(0, 1, 2, 3)
        finally:
            collector.close()
        mapping_values.append(values[:, :, line_indices].astype(np.float32))
        mappings.append(original_to_new)
        if not audit["mapping_prompts"]:
            audit["mapping_prompts"] = [
                {
                    "question_id": qid,
                    "prompt_hash": prompt_hash(prompt),
                    "line_end_positions_unpadded": [positions[row][i] for i in line_indices],
                    "line_end_tokens": [
                        tokenizer.decode([
                            tokenizer(prompt, add_special_tokens=False)["input_ids"][positions[row][i]]
                        ])
                        for i in line_indices
                    ],
                }
                for row, (qid, prompt) in enumerate(zip(group_qids, prompts))
            ]

    layers = len(parts.layers)
    batch = len(group_qids)
    hidden = int(mapping_values[0].shape[-1])
    aligned = np.empty((4, layers, batch, 4, hidden), dtype=np.float32)
    for mapping_index, values in enumerate(mapping_values):
        for row in range(batch):
            indices = [
                LETTERS.index(mappings[mapping_index][row][content])
                for content in LETTERS
            ]
            aligned[mapping_index, :, row] = values[:, row, indices]
    average = aligned.mean(axis=0, dtype=np.float32)
    centered = average - average.mean(axis=2, keepdims=True)
    directions = np.empty((layers, batch, hidden), dtype=np.float32)
    for row, qid in enumerate(group_qids):
        target = LETTERS.index(target_answers[qid])
        vector = centered[:, row, target]
        norms = np.linalg.norm(vector, axis=-1, keepdims=True)
        if np.any(norms <= 1e-8):
            raise RuntimeError(f"{qid}: degenerate target-answer semantic vector")
        directions[:, row] = vector / norms
    audit["direction_norm_range"] = [
        float(np.linalg.norm(directions, axis=-1).min()),
        float(np.linalg.norm(directions, axis=-1).max()),
    ]
    return directions, audit


def run(
    config_path: Path,
    plan_path: Path,
    baseline_path: Path,
    mapping_plan_paths: list[Path],
    output: Path,
    intervention_mode: str,
    all_questions: bool,
    natural_source: Path | None,
    natural_validation_source: Path | None,
    direction_cache: Path | None,
    target_answer: str,
    remapped_baseline_path: Path | None,
    max_cohorts: int | None,
    cohort_shard_index: int | None,
    cohort_shard_count: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires the historical batch-size-4 SDPA regime")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    all_qids = [row["id"] for row in manifest["questions"]]
    if (cohort_shard_index is None) != (cohort_shard_count is None):
        raise ValueError("Both cohort shard arguments must be supplied together")
    if cohort_shard_count is not None:
        if all_questions:
            raise ValueError("Cohort sharding and --all-questions are mutually exclusive")
        if cohort_shard_count <= 0 or not 0 <= cohort_shard_index < cohort_shard_count:
            raise ValueError("Invalid cohort shard")
        cohorts = [
            all_qids[start : start + config.batch_size]
            for start in range(0, len(all_qids), config.batch_size)
        ]
        shard_start = len(cohorts) * cohort_shard_index // cohort_shard_count
        shard_end = len(cohorts) * (cohort_shard_index + 1) // cohort_shard_count
        qids = [qid for cohort in cohorts[shard_start:shard_end] for qid in cohort]
    else:
        qids = all_qids if all_questions else _question_ids(plan_path)
    qid_set = set(qids)
    baseline = json.loads(baseline_path.read_text())["results"]
    if not set(all_qids) <= set(baseline):
        raise ValueError("Baseline results are incomplete")
    mapping_plans = _load_mapping_plans(mapping_plan_paths, all_qids)
    second_mapping = mapping_plans[0]
    if target_answer == "w1":
        target_answers = {qid: baseline[qid]["answer"] for qid in all_qids}
    elif target_answer == "w2":
        if remapped_baseline_path is None:
            raise ValueError("--remapped-baseline is required for --target-answer w2")
        remapped_baseline = json.loads(remapped_baseline_path.read_text())["results"]
        target_answers = {
            qid: remapped_baseline[qid]["answer_original_content"]
            for qid in all_qids
        }
        invalid = [qid for qid, answer in target_answers.items() if answer not in LETTERS]
        if invalid:
            raise ValueError(f"W2 is unavailable for {len(invalid)} questions")
    else:
        raise ValueError(f"Unknown target answer: {target_answer}")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    n_layers = len(parts.layers)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.npz"
    arrays = _initialize(result_path, qids, n_layers)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = output / "prompt_audit.json"

    if natural_source is not None:
        source = dict(np.load(natural_source, allow_pickle=False))
        source_qids = source["question_ids"].astype(str).tolist()
        source_index = {qid: index for index, qid in enumerate(source_qids)}
        missing = [qid for qid in qids if qid not in source_index]
        if missing:
            raise ValueError(f"Natural source is missing {len(missing)} questions")
        for target_key in (
            "natural_logits", "natural_projection", "natural_residual_norm"
        ):
            if target_key not in source:
                raise ValueError(f"Natural source lacks {target_key}")
            for qi, qid in enumerate(qids):
                arrays[target_key][:, qi] = source[target_key][:, source_index[qid]]
        if not all(
            np.all(np.isfinite(arrays[key]))
            for key in ("natural_logits", "natural_projection", "natural_residual_norm")
        ):
            raise ValueError("Natural source contains non-finite values")
    validation_source = None
    validation_index = None
    if natural_validation_source is not None:
        validation_source = dict(np.load(natural_validation_source, allow_pickle=False))
        validation_qids = validation_source["question_ids"].astype(str).tolist()
        validation_index = {qid: index for index, qid in enumerate(validation_qids)}
        missing = [qid for qid in qids if qid not in validation_index]
        if missing:
            raise ValueError(f"Natural validation source is missing {len(missing)} questions")

    if direction_cache is not None:
        direction_cache.mkdir(parents=True, exist_ok=True)
    run_started = time.monotonic()
    completed_groups = 0
    pending_qids = {
        qid for qid in qids if not bool(arrays["completed"][qid_index[qid]])
    }
    total_target_groups = sum(
        bool(set(all_qids[start:start + config.batch_size]) & pending_qids)
        for start in range(0, len(all_qids), config.batch_size)
    )

    for group_start in range(0, len(all_qids), config.batch_size):
        group_qids = all_qids[group_start : group_start + config.batch_size]
        targets = [
            qid for qid in group_qids
            if qid in qid_set and not bool(arrays["completed"][qid_index[qid]])
        ]
        if not targets:
            continue
        selected_rows = [group_qids.index(qid) for qid in targets]
        group_started = time.monotonic()
        cache_path = (
            direction_cache / f"cohort_{group_start:04d}.npz"
            if direction_cache is not None else None
        )
        if cache_path is not None and cache_path.exists():
            cached = dict(np.load(cache_path, allow_pickle=False))
            if cached["question_ids"].astype(str).tolist() != group_qids:
                raise ValueError(f"Direction-cache cohort mismatch: {cache_path}")
            if "target_answer" in cached:
                cached_target = str(cached["target_answer"].item())
                if cached_target != target_answer:
                    raise ValueError(
                        f"Direction-cache target mismatch: {cached_target} != {target_answer}"
                    )
            elif target_answer != "w1":
                raise ValueError(f"Unlabeled cache cannot be used for {target_answer}: {cache_path}")
            directions = cached["directions"].astype(np.float32, copy=False)
            vector_audit = {"source": "cache", "path": str(cache_path)}
        else:
            directions, vector_audit = _semantic_directions(
                model, processor, parts, tokenizer, config, questions,
                group_qids, mapping_plans, target_answers,
            )
            if cache_path is not None:
                atomic_save_npz(
                    cache_path,
                    question_ids=np.asarray(group_qids),
                    directions=directions.astype(np.float32),
                    target_answer=np.asarray(target_answer),
                )

        for condition_index, condition in enumerate(CONDITIONS):
            prompts = []
            for qid in group_qids:
                remapped = _remap_question(
                    questions[qid], second_mapping[qid]["new_to_original"]
                )
                messages = _messages(config, questions[qid], remapped, condition)
                prompts.append(
                    render_chat(
                        processor, messages, config.disable_thinking,
                        config.chat_serialization,
                    )
                )
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            if any(int(index) != int(input_ids.shape[1] - 1) for index in last_indices):
                raise RuntimeError("Final decision positions are not batch-final tokens")

            if natural_source is None:
                with FinalDecisionSemanticHook(
                    parts, directions, selected_rows, intervention_mode="none"
                ) as hook:
                    natural_output = _forward(model, parts, input_ids, attention_mask)
                    natural_audit = hook.arrays()
                natural_logits = _aggregate_logits(
                    natural_output.logits[:, -1].float(), variant_ids
                ).detach().cpu().numpy()
                if validation_source is not None and validation_index is not None:
                    for qid in targets:
                        row = group_qids.index(qid)
                        reference = validation_source["natural_logits"][
                            condition_index, validation_index[qid]
                        ]
                        if not np.array_equal(natural_logits[row], reference):
                            difference = float(np.max(np.abs(natural_logits[row] - reference)))
                            raise RuntimeError(
                                f"{qid}: natural logits failed exact validation; max diff={difference}"
                            )

            with FinalDecisionSemanticHook(
                parts, directions, selected_rows, intervention_mode=intervention_mode
            ) as hook:
                ablated_output = _forward(model, parts, input_ids, attention_mask)
                ablated_audit = hook.arrays()
            ablated_logits = _aggregate_logits(
                ablated_output.logits[:, -1].float(), variant_ids
            ).detach().cpu().numpy()

            for qid in targets:
                row = group_qids.index(qid)
                qi = qid_index[qid]
                if natural_source is None:
                    arrays["natural_logits"][condition_index, qi] = natural_logits[row]
                arrays["ablated_logits"][condition_index, qi] = ablated_logits[row]
                if natural_source is None:
                    arrays["natural_projection"][condition_index, qi] = natural_audit[
                        "projection"
                    ][:, row]
                    arrays["natural_residual_norm"][condition_index, qi] = natural_audit[
                        "residual_norm"
                    ][:, row]
                arrays["ablated_pre_projection"][condition_index, qi] = ablated_audit[
                    "projection"
                ][:, row]
                arrays["ablated_residual_norm"][condition_index, qi] = ablated_audit[
                    "residual_norm"
                ][:, row]
                arrays["ablated_projection_after"][condition_index, qi] = ablated_audit[
                    "projection_after"
                ][:, row]

            if not audit_path.exists():
                row = selected_rows[0]
                audit_qid = group_qids[row]
                audit_qi = qid_index[audit_qid]
                audit_path.write_text(
                    json.dumps(
                        {
                            "question_id": audit_qid,
                            "condition": condition,
                            "historical_group_qids": group_qids,
                            "target_rows": selected_rows,
                            "final_decision_position_physical": int(input_ids.shape[1] - 1),
                            "final_decision_token": tokenizer.decode(
                                [int(input_ids[row, -1])]
                            ),
                            "prompt_hash": prompt_hash(prompts[row]),
                            "rendered_prompt": prompts[row],
                            "vector_audit": vector_audit,
                            "natural_projection_first_last": [
                                float(arrays["natural_projection"][condition_index, audit_qi, 0]),
                                float(arrays["natural_projection"][condition_index, audit_qi, -1]),
                            ],
                            "ablated_projection_after_max_abs": float(
                                np.abs(ablated_audit["projection_after"][:, row]).max()
                            ),
                        },
                        indent=2,
                        sort_keys=True,
                    ) + "\n"
                )

        for qid in targets:
            arrays["completed"][qid_index[qid]] = True
        atomic_save_npz(result_path, **arrays)
        done = int(arrays["completed"].sum())
        completed_groups += 1
        elapsed = time.monotonic() - run_started
        group_seconds = time.monotonic() - group_started
        eta = elapsed / completed_groups * (total_target_groups - completed_groups)
        print(
            f"final-decision semantic ablation: {done}/{len(qids)}; "
            f"cohort={group_seconds:.1f}s elapsed={elapsed/60:.1f}m "
            f"eta={eta/60:.1f}m cache={'hit' if vector_audit.get('source') == 'cache' else 'miss'}",
            flush=True,
        )
        if max_cohorts is not None and completed_groups >= max_cohorts:
            print(f"Stopped after benchmark limit of {max_cohorts} cohorts", flush=True)
            break

    metadata = {
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "baseline_path": str(baseline_path),
        "mapping_plan_paths": [str(path) for path in mapping_plan_paths],
        "n_questions": len(qids),
        "all_questions_single_pass": bool(all_questions),
        "n_layers": n_layers,
        "position": "final token of the second user prompt (the second-answer decision position)",
        "vector_definition": (
            "For each question and post-block readout, align option-closing-newline "
            "residuals by semantic content across four Baseline mappings where each "
            "content occupies A-D once; average mappings; subtract the within-question "
            "mean option; select and normalize "
            + (
                "the original Baseline winner W1."
                if target_answer == "w1"
                else "the fresh remapped-Baseline winner W2."
            )
        ),
        "target_answer": target_answer,
        "remapped_baseline_path": (
            str(remapped_baseline_path) if remapped_baseline_path is not None else None
        ),
        "intervention": (
            "After every model block 1-64, record the live projection of the final "
            f"decision residual onto the layer-specific {target_answer.upper()} semantic direction and apply "
            + (
                "signed projection removal before the next block or final norm/unembedding."
                if intervention_mode == "signed"
                else (
                    "positive-only projection removal, subtracting max(projection, 0), before the next block or final norm/unembedding."
                    if intervention_mode == "positive_only"
                    else "negative-only projection clamping, subtracting min(projection, 0), before the next block or final norm/unembedding."
                )
            )
        ),
        "intervention_mode": intervention_mode,
        "natural_companion": (
            f"Natural logits, {target_answer.upper()} projections, and residual norms were copied from "
            f"the bit-exact saved companion at {natural_source}."
            if natural_source is not None
            else f"An unmodified exact-cohort forward in the same job records the natural {target_answer.upper()} projection and residual norm at every readout."
        ),
        "natural_validation_source": (
            str(natural_validation_source)
            if natural_validation_source is not None else None
        ),
        "direction_cache": str(direction_cache) if direction_cache is not None else None,
        "cohort_shard_index": cohort_shard_index,
        "cohort_shard_count": cohort_shard_count,
        "resolved_answer_tokens": resolved,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--mapping-plans", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--intervention-mode",
        choices=("signed", "positive_only", "negative_only"),
        default="signed",
    )
    parser.add_argument(
        "--all-questions",
        action="store_true",
        help="Process all manifest questions once while preserving physical cohorts.",
    )
    parser.add_argument(
        "--natural-source", type=Path,
        help="Reuse natural logits/projections/norms from an exact prior result.",
    )
    parser.add_argument(
        "--natural-validation-source", type=Path,
        help="Run a natural companion but require its logits to match this saved result exactly.",
    )
    parser.add_argument(
        "--direction-cache", type=Path,
        help="Read/write exact float32 semantic directions by physical cohort.",
    )
    parser.add_argument("--target-answer", choices=("w1", "w2"), default="w1")
    parser.add_argument("--remapped-baseline", type=Path)
    parser.add_argument(
        "--max-cohorts", type=int,
        help="Process at most this many pending physical cohorts (benchmarking).",
    )
    parser.add_argument("--cohort-shard-index", type=int)
    parser.add_argument("--cohort-shard-count", type=int)
    args = parser.parse_args()
    run(
        args.config, args.plan, args.baseline, args.mapping_plans, args.output,
        args.intervention_mode, args.all_questions, args.natural_source,
        args.natural_validation_source,
        args.direction_cache, args.target_answer, args.remapped_baseline,
        args.max_cohorts, args.cohort_shard_index,
        args.cohort_shard_count,
    )


if __name__ == "__main__":
    main()
