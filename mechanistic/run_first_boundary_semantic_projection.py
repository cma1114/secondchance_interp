from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .collect_remapped_behavior import _messages, _remap_question
from .config import ExperimentConfig
from .downstream_source_intervention import BatchedGDNSourceWriteAblator
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import prompt_hash
from .run_final_decision_semantic_ablation import (
    FinalDecisionSemanticHook,
    _semantic_directions,
)
from .run_first_decision_cross_order_patching import (
    CONDITIONS,
    LETTERS,
    _aggregate_logits,
    _load_mapping_plans,
    _question_ids,
)
from .run_first_span_gla_ablation import _source_positions
from .run_historical_answer_intervention import _forward


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
        "lesioned_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "natural_reference_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "lesion_reference_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "natural_projection": np.full((2, n, n_layers), np.nan, dtype=np.float32),
        "natural_residual_norm": np.full((2, n, n_layers), np.nan, dtype=np.float32),
        "lesioned_projection": np.full((2, n, n_layers), np.nan, dtype=np.float32),
        "lesioned_residual_norm": np.full((2, n, n_layers), np.nan, dtype=np.float32),
    }


def _load_natural_source(
    qids: list[str], natural_source: Path
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    source = dict(np.load(natural_source, allow_pickle=False))
    source_qids = source["question_ids"].astype(str).tolist()
    source_index = {qid: index for index, qid in enumerate(source_qids)}
    missing = [qid for qid in qids if qid not in source_index]
    if missing:
        raise ValueError(f"Natural source is missing {len(missing)} questions")
    for target_key in ("natural_logits", "natural_projection", "natural_residual_norm"):
        if target_key not in source:
            raise ValueError(f"Natural source lacks {target_key}")
    return source, source_index


def run(
    config_path: Path,
    plan_path: Path,
    baseline_path: Path,
    mapping_plan_paths: list[Path],
    natural_source: Path,
    lesion_validation_source: Path,
    direction_cache: Path,
    output: Path,
    max_cohorts: int | None,
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
    qids = _question_ids(plan_path)
    qid_set = set(qids)
    baseline = json.loads(baseline_path.read_text())["results"]
    target_answers = {qid: baseline[qid]["answer"] for qid in all_qids}
    mapping_plans = _load_mapping_plans(mapping_plan_paths, all_qids)
    second_mapping = mapping_plans[0]

    validation = dict(np.load(lesion_validation_source, allow_pickle=False))
    validation_qids = validation["question_ids"].astype(str).tolist()
    validation_index = {qid: index for index, qid in enumerate(validation_qids)}
    scenario_ids = validation["scenario_ids"].astype(str).tolist()
    boundary_scenario = scenario_ids.index("first_answer_boundary")
    missing = [qid for qid in qids if qid not in validation_index]
    if missing:
        raise ValueError(f"Lesion validation source is missing {len(missing)} questions")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    n_layers = len(parts.layers)
    direction_cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.npz"
    arrays = _initialize(result_path, qids, n_layers)
    natural_reference, natural_reference_index = _load_natural_source(qids, natural_source)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = output / "prompt_audit.json"

    pending = {qid for qid in qids if not bool(arrays["completed"][qid_index[qid]])}
    total_groups = sum(
        bool(set(all_qids[start : start + config.batch_size]) & pending)
        for start in range(0, len(all_qids), config.batch_size)
    )
    completed_groups = 0
    started = time.monotonic()

    for group_start in range(0, len(all_qids), config.batch_size):
        group_qids = all_qids[group_start : group_start + config.batch_size]
        targets = [
            qid for qid in group_qids
            if qid in qid_set and not bool(arrays["completed"][qid_index[qid]])
        ]
        if not targets:
            continue
        group_started = time.monotonic()
        selected_rows = [group_qids.index(qid) for qid in targets]
        cache_path = direction_cache / f"cohort_{group_start:04d}.npz"
        if cache_path.exists():
            cached = dict(np.load(cache_path, allow_pickle=False))
            if cached["question_ids"].astype(str).tolist() != group_qids:
                raise ValueError(f"Direction-cache cohort mismatch: {cache_path}")
            if str(cached.get("target_answer", np.asarray("w1")).item()) != "w1":
                raise ValueError(f"Direction cache is not W1: {cache_path}")
            directions = cached["directions"].astype(np.float32, copy=False)
            vector_audit: dict[str, Any] = {"source": "cache", "path": str(cache_path)}
        else:
            directions, vector_audit = _semantic_directions(
                model,
                processor,
                parts,
                tokenizer,
                config,
                questions,
                group_qids,
                mapping_plans,
                target_answers,
            )
            atomic_save_npz(
                cache_path,
                question_ids=np.asarray(group_qids),
                directions=directions.astype(np.float32),
                target_answer=np.asarray("w1"),
            )

        condition_batches: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            prompts: list[str] = []
            unpadded_spans: list[list[int]] = []
            token_rows: list[list[int]] = []
            for qid in group_qids:
                remapped = _remap_question(
                    questions[qid], second_mapping[qid]["new_to_original"]
                )
                messages = _messages(config, questions[qid], remapped, condition)
                prompt = render_chat(
                    processor,
                    messages,
                    config.disable_thinking,
                    config.chat_serialization,
                )
                ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                prompts.append(prompt)
                token_rows.append([int(value) for value in ids])
                unpadded_spans.append(
                    _source_positions(tokenizer, prompt, messages, questions[qid])[
                        "first_answer_boundary"
                    ]
                )
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            if any(int(index) != int(input_ids.shape[1] - 1) for index in last_indices):
                raise RuntimeError("Final decision positions are not batch-final tokens")
            width = int(input_ids.shape[1])
            physical_spans: list[list[int]] = []
            for row, (ids, span) in enumerate(zip(token_rows, unpadded_spans)):
                left_pad = width - len(ids)
                if input_ids[row, left_pad:].tolist() != ids:
                    raise RuntimeError("Historical cohort tokenization changed a prompt")
                physical_spans.append([left_pad + position for position in span])

            # Reproduce the original lesion runner's per-condition natural pass,
            # while read-only hooks record a same-host natural projection companion.
            natural_hook = FinalDecisionSemanticHook(
                parts, directions, selected_rows, intervention_mode="none"
            )
            try:
                natural_output = _forward(model, parts, input_ids, attention_mask)
                natural_audit = natural_hook.arrays()
            finally:
                natural_hook.close()
            natural_logits = _aggregate_logits(
                natural_output.logits[:, -1].float(), variant_ids
            ).detach().cpu().numpy()
            condition_index = CONDITIONS.index(condition)
            for qid in targets:
                row = group_qids.index(qid)
                qi = qid_index[qid]
                reference_qi = natural_reference_index[qid]
                arrays["natural_logits"][condition_index, qi] = natural_logits[row]
                arrays["natural_reference_logits"][condition_index, qi] = natural_reference[
                    "natural_logits"
                ][condition_index, reference_qi]
                arrays["natural_projection"][condition_index, qi] = natural_audit[
                    "projection"
                ][:, row]
                arrays["natural_residual_norm"][condition_index, qi] = natural_audit[
                    "residual_norm"
                ][:, row]
            condition_batches[condition] = {
                "prompts": prompts,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "physical_spans": physical_spans,
            }

        for qid in targets:
            row = group_qids.index(qid)
            for condition_index, condition in enumerate(CONDITIONS):
                batch = condition_batches[condition]
                ablator = BatchedGDNSourceWriteAblator(
                    parts, {row: batch["physical_spans"][row]}
                )
                hook = FinalDecisionSemanticHook(
                    parts, directions, [row], intervention_mode="none"
                )
                try:
                    lesioned_output = _forward(
                        model, parts, batch["input_ids"], batch["attention_mask"]
                    )
                    ablator.assert_fired()
                    lesion_audit = hook.arrays()
                finally:
                    hook.close()
                    ablator.close()
                lesioned_logits = _aggregate_logits(
                    lesioned_output.logits[:, -1].float(), variant_ids
                ).detach().cpu().numpy()

                row = group_qids.index(qid)
                qi = qid_index[qid]
                reference = validation["ablated_logits"][
                    condition_index, boundary_scenario, validation_index[qid]
                ]
                arrays["lesioned_logits"][condition_index, qi] = lesioned_logits[row]
                arrays["lesion_reference_logits"][condition_index, qi] = reference
                arrays["lesioned_projection"][condition_index, qi] = lesion_audit[
                    "projection"
                ][:, row]
                arrays["lesioned_residual_norm"][condition_index, qi] = lesion_audit[
                    "residual_norm"
                ][:, row]

                if not audit_path.exists():
                    audit_path.write_text(
                        json.dumps(
                            {
                                "question_id": qid,
                                "condition": condition,
                                "historical_group_qids": group_qids,
                                "selected_row": row,
                                "first_answer_boundary_positions_physical": batch[
                                    "physical_spans"
                                ][row],
                                "first_answer_boundary_tokens": tokenizer.convert_ids_to_tokens(
                                    batch["input_ids"][
                                        row, batch["physical_spans"][row]
                                    ].tolist()
                                ),
                                "final_decision_position_physical": int(
                                    batch["input_ids"].shape[1] - 1
                                ),
                                "prompt_hash": prompt_hash(batch["prompts"][row]),
                                "rendered_prompt": batch["prompts"][row],
                                "vector_audit": vector_audit,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    )

        for qid in targets:
            arrays["completed"][qid_index[qid]] = True
        atomic_save_npz(result_path, **arrays)
        completed_groups += 1
        elapsed = time.monotonic() - started
        group_seconds = time.monotonic() - group_started
        eta = elapsed / completed_groups * (total_groups - completed_groups)
        print(
            f"first-boundary semantic projection: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort={group_seconds:.1f}s elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m "
            f"cache={'hit' if vector_audit.get('source') == 'cache' else 'miss'}",
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
        "natural_source": str(natural_source),
        "lesion_validation_source": str(lesion_validation_source),
        "direction_cache": str(direction_cache),
        "n_questions": len(qids),
        "n_layers": n_layers,
        "intervention": (
            "Set beta=0 at every token in the first-answer-boundary span in all 48 "
            "Gated DeltaNet layers for all selected rows, while recording the "
            "final-decision residual projection onto the exact question- and layer-specific "
            "four-mapping W1 semantic direction after every model block."
        ),
        "validation": (
            "The same-host natural and lesion logits are retained alongside the exact "
            "historical natural and first_answer_boundary reference rows. Cross-host "
            "maximum deviations and A-D argmax agreement are reported; the causal "
            "projection effect is always same-host lesion minus same-host natural."
        ),
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
    parser.add_argument("--natural-source", type=Path, required=True)
    parser.add_argument("--lesion-validation-source", type=Path, required=True)
    parser.add_argument("--direction-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.plan,
        args.baseline,
        args.mapping_plans,
        args.natural_source,
        args.lesion_validation_source,
        args.direction_cache,
        args.output,
        args.max_cohorts,
    )


if __name__ == "__main__":
    main()
