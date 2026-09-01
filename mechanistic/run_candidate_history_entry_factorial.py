from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSDPAQuerySourceAttentionAblator
from .io import atomic_save_npz
from .modeling import get_tokenizer, load_model_and_processor, resolve_answer_tokens
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_fixed_a_final_query_edge_ablation import _option_line_positions

ORDINARY_LAYERS = tuple(range(4, 65, 4))
RANKS = ("R1", "R2", "R3", "R4")
TOKEN_CLASSES = (
    "leading_space",
    "option_letter",
    "colon",
    "semantic",
    "newline",
)
ALL_OPEN_MASK = (1 << len(TOKEN_CLASSES)) - 1
SOURCE_MODES = ("matching", "balanced_wrong")
SourceMode = Literal["matching", "balanced_wrong"]


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _wrong_source_rank(question_index: int, target_rank: int) -> int:
    """Balanced deterministic assignment to one of the three wrong ranks."""

    return (target_rank + 1 + ((question_index + target_rank) % 3)) % 4


def _partition_option_line(
    positions: list[int], tokens: list[str]
) -> tuple[list[list[int]], dict[str, Any]]:
    """Partition one canonical option line into five exact physical classes."""

    if len(positions) != len(tokens):
        raise ValueError("Option-line position and token counts differ")
    if len(positions) < 5:
        raise ValueError(f"Option line is too short for canonical partition: {tokens}")
    if tokens[0] not in {"Ġ", "▁"}:
        raise ValueError(f"Unexpected leading-space token: {tokens[0]!r}")
    if tokens[2] != ":":
        raise ValueError(f"Unexpected colon token: {tokens[2]!r}")
    if tokens[-1] not in {"Ċ", "\n"}:
        raise ValueError(f"Unexpected newline token: {tokens[-1]!r}")
    if not any(letter in tokens[1] for letter in LETTERS):
        raise ValueError(f"Unexpected option-letter token: {tokens[1]!r}")

    classes = [
        [int(positions[0])],
        [int(positions[1])],
        [int(positions[2])],
        [int(value) for value in positions[3:-1]],
        [int(positions[-1])],
    ]
    flat = [value for values in classes for value in values]
    if flat != [int(value) for value in positions] or len(set(flat)) != len(flat):
        raise RuntimeError("Token-class partition is not an exact disjoint cover")
    if not classes[3]:
        raise RuntimeError("Semantic token class is empty")
    audit = {
        name: {
            "positions": values,
            "tokens": [tokens[positions.index(value)] for value in values],
        }
        for name, values in zip(TOKEN_CLASSES, classes)
    }
    return classes, audit


def _blocked_class_indices(availability_mask: int) -> tuple[int, ...]:
    if availability_mask < 0 or availability_mask > ALL_OPEN_MASK:
        raise ValueError(f"Invalid availability mask: {availability_mask}")
    return tuple(
        index
        for index in range(len(TOKEN_CLASSES))
        if not (availability_mask & (1 << index))
    )


def _availability_label(mask: int) -> str:
    available = [
        name for index, name in enumerate(TOKEN_CLASSES) if mask & (1 << index)
    ]
    return "+".join(available) if available else "none"


def _factorial_specs(
    layer_indices: tuple[int, ...],
    source_positions: list[list[list[int]]],
    query_classes: list[list[list[list[int]]]],
    wrong_source_ranks: np.ndarray,
    availability_mask: int,
    source_mode: SourceMode,
) -> dict[int, dict[int, dict[int, list[int]]]]:
    blocked_classes = _blocked_class_indices(availability_mask)
    if not blocked_classes:
        raise ValueError("The all-open mask has no intervention specs")
    specs: dict[int, dict[int, dict[int, list[int]]]] = {}
    for layer_index in layer_indices:
        rows: dict[int, dict[int, list[int]]] = {}
        for row in range(len(source_positions)):
            row_specs: dict[int, list[int]] = {}
            for target_rank in range(4):
                source_rank = (
                    target_rank
                    if source_mode == "matching"
                    else int(wrong_source_ranks[row, target_rank])
                )
                if source_rank == target_rank and source_mode != "matching":
                    raise RuntimeError(
                        "Wrong-line control selected the matching source"
                    )
                sources = [int(value) for value in source_positions[row][source_rank]]
                if not sources:
                    raise RuntimeError("Empty source line")
                for class_index in blocked_classes:
                    for query in query_classes[row][target_rank][class_index]:
                        if int(query) in row_specs:
                            raise RuntimeError(
                                "Destination query assigned more than once"
                            )
                        row_specs[int(query)] = sources
            if not row_specs:
                raise RuntimeError("Intervention row has no destination queries")
            rows[row] = row_specs
        specs[layer_index] = rows
    return specs


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses a different question order")
        if arrays["token_classes"].astype(str).tolist() != list(TOKEN_CLASSES):
            raise ValueError("Existing checkpoint uses a different token partition")
        if arrays["availability_masks"].tolist() != list(range(32)):
            raise ValueError("Existing checkpoint does not contain the full factorial")
        return arrays

    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "conditions": np.asarray(CONDITIONS),
        "source_modes": np.asarray(SOURCE_MODES),
        "ranks": np.asarray(RANKS),
        "token_classes": np.asarray(TOKEN_CLASSES),
        "availability_masks": np.arange(32, dtype=np.int8),
        "ordinary_layers_one_based": np.asarray(ORDINARY_LAYERS, dtype=np.int16),
        "completed": np.zeros(n, dtype=bool),
        "rank_contents": np.full((n, 4), "", dtype="<U1"),
        "baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        # condition, source mode, availability mask, question, output letter
        "factorial_logits": np.full((2, 2, 32, n, 4), np.nan, dtype=np.float32),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "wrong_source_ranks": np.full((n, 4), -1, dtype=np.int8),
        "source_position_counts": np.zeros((n, 4), dtype=np.int16),
        "wrong_source_position_counts": np.zeros((n, 4), dtype=np.int16),
        "query_class_counts": np.zeros((n, 4, 5), dtype=np.int16),
    }


def _validate_completed(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    completed = arrays["completed"].astype(bool)
    if not np.any(completed):
        return {"n_completed": 0.0, "natural_max_abs_error": float("nan")}
    completed_indices = np.flatnonzero(completed)
    completed_factorial = np.take(arrays["factorial_logits"], completed_indices, axis=3)
    completed_natural = np.take(arrays["natural_logits"], completed_indices, axis=1)
    completed_trusted = np.take(
        arrays["trusted_natural_logits"], completed_indices, axis=1
    )
    finite_arrays = (
        arrays["baseline_logits"][completed],
        completed_trusted,
        completed_natural,
        completed_factorial,
    )
    if not all(np.all(np.isfinite(values)) for values in finite_arrays):
        raise RuntimeError("Checkpoint contains non-finite completed outputs")
    natural_error = float(np.max(np.abs(completed_natural - completed_trusted)))
    factorial_identity_error = float(
        np.max(
            np.abs(
                completed_factorial[:, :, ALL_OPEN_MASK] - completed_natural[:, None]
            )
        )
    )
    if natural_error != 0.0:
        raise RuntimeError(f"Trusted natural reproduction error is {natural_error}")
    if factorial_identity_error != 0.0:
        raise RuntimeError(
            f"All-open factorial identity error is {factorial_identity_error}"
        )
    return {
        "n_completed": float(completed.sum()),
        "natural_max_abs_error": natural_error,
        "all_open_identity_max_abs_error": factorial_identity_error,
    }


def run(
    config_path: Path,
    remapping_plan_path: Path,
    baseline_path: Path,
    trusted_game_path: Path,
    trusted_neutral_path: Path,
    output_dir: Path,
    max_cohorts: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires exact historical batch-size-4 SDPA execution")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires action-matched incorrect/lost feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw_qwen_chatml serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    all_qids = [row["id"] for row in manifest["questions"]]
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = all_qids
    if max_cohorts is not None:
        qids = qids[: int(max_cohorts) * config.batch_size]
    global_qid_index = {qid: index for index, qid in enumerate(all_qids)}
    local_qid_index = {qid: index for index, qid in enumerate(qids)}
    mappings = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    baseline = json.loads(baseline_path.read_text())["results"]
    trusted = [
        json.loads(trusted_game_path.read_text())["results"],
        json.loads(trusted_neutral_path.read_text())["results"],
    ]
    required_record_key = "aggregated_ad_logits"
    for qid in qids:
        if qid not in baseline or required_record_key not in baseline[qid]:
            raise ValueError(
                f"Baseline artifact lacks {required_record_key!r} for {qid}; "
                "use the canonical local Qwen baseline_results.json artifact"
            )
        for condition_index, condition in enumerate(CONDITIONS):
            if (
                qid not in trusted[condition_index]
                or required_record_key not in trusted[condition_index][qid]
            ):
                raise ValueError(
                    f"Trusted {condition} artifact lacks {required_record_key!r} for {qid}"
                )

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    layer_indices = tuple(
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if tuple(index + 1 for index in layer_indices) != ORDINARY_LAYERS:
        raise RuntimeError(f"Unexpected ordinary-attention layers: {layer_indices}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    audit_path = output_dir / "prompt_audit.json"
    arrays = _initialize(result_path, qids)
    started = time.monotonic()
    durations: list[float] = []

    for start in range(0, len(qids), config.batch_size):
        cohort = qids[start : start + config.batch_size]
        indices = [local_qid_index[qid] for qid in cohort]
        if np.all(arrays["completed"][indices]):
            continue
        cohort_started = time.monotonic()
        batches = [
            _build_batch(
                config, processor, tokenizer, questions, mappings, cohort, condition
            )
            for condition in CONDITIONS
        ]
        cohort_audit: dict[str, Any] = {"question_ids": cohort, "conditions": {}}

        for qid in cohort:
            qi = local_qid_index[qid]
            baseline_logits = np.asarray(
                baseline[qid]["aggregated_ad_logits"], dtype=np.float32
            )
            order = np.argsort(-baseline_logits, kind="stable")
            arrays["baseline_logits"][qi] = baseline_logits
            arrays["rank_contents"][qi] = np.asarray(
                [LETTERS[int(index)] for index in order]
            )
            for target_rank in range(4):
                arrays["wrong_source_ranks"][qi, target_rank] = _wrong_source_rank(
                    global_qid_index[qid], target_rank
                )

        for condition_index, (condition, batch) in enumerate(zip(CONDITIONS, batches)):
            width = int(batch["input_ids"].shape[1])
            source_physical: list[list[list[int]]] = []
            query_classes_physical: list[list[list[list[int]]]] = []
            wrong_ranks = np.empty((len(cohort), 4), dtype=np.int8)
            row_audits: list[dict[str, Any]] = []

            for row, qid in enumerate(cohort):
                qi = local_qid_index[qid]
                left_pad = width - len(batch["token_rows"][row])
                remapped_question = {
                    **questions[qid],
                    "options": {
                        new: questions[qid]["options"][old]
                        for new, old in mappings[qid]["new_to_original"].items()
                    },
                }
                first_positions, first_audit = _option_line_positions(
                    tokenizer, batch["prompts"][row], questions[qid]
                )
                second_positions, second_audit = _option_line_positions(
                    tokenizer, batch["prompts"][row], remapped_question
                )
                ranks = arrays["rank_contents"][qi].astype(str).tolist()
                row_sources: list[list[int]] = []
                row_query_classes: list[list[list[int]]] = []
                partition_audit: dict[str, Any] = {}

                for target_rank, content in enumerate(ranks):
                    second_letter = mappings[qid]["original_to_new"][content]
                    sources = [left_pad + value for value in first_positions[content]]
                    raw_queries = second_positions[second_letter]
                    raw_tokens = second_audit[second_letter]["tokens"]
                    query_partition, query_audit = _partition_option_line(
                        raw_queries, raw_tokens
                    )
                    physical_partition = [
                        [left_pad + value for value in values]
                        for values in query_partition
                    ]
                    if max(sources) >= min(
                        value for values in physical_partition for value in values
                    ):
                        raise RuntimeError(
                            "Source option line is not causally before receiver"
                        )
                    if [value for values in physical_partition for value in values] != [
                        left_pad + value for value in raw_queries
                    ]:
                        raise RuntimeError(
                            "Physical query partition failed exact cover"
                        )
                    row_sources.append(sources)
                    row_query_classes.append(physical_partition)
                    arrays["source_position_counts"][qi, target_rank] = len(sources)
                    arrays["query_class_counts"][qi, target_rank] = np.asarray(
                        [len(values) for values in physical_partition], dtype=np.int16
                    )
                    wrong_rank = int(arrays["wrong_source_ranks"][qi, target_rank])
                    wrong_ranks[row, target_rank] = wrong_rank
                    partition_audit[RANKS[target_rank]] = {
                        "semantic_content": content,
                        "display_letter_2p": second_letter,
                        "wrong_source_rank": RANKS[wrong_rank],
                        "classes": query_audit,
                    }

                for target_rank in range(4):
                    wrong_rank = int(wrong_ranks[row, target_rank])
                    arrays["wrong_source_position_counts"][qi, target_rank] = len(
                        row_sources[wrong_rank]
                    )
                source_physical.append(row_sources)
                query_classes_physical.append(row_query_classes)
                arrays["prompt_hashes"][condition_index, qi] = _hash_prompt(
                    batch["prompts"][row]
                )
                row_audits.append(
                    {
                        "question_id": qid,
                        "rank_contents": ranks,
                        "first_option_lines": first_audit,
                        "second_option_lines": second_audit,
                        "receiver_partition": partition_audit,
                    }
                )

            natural = _aggregate_logits(
                _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                variant_ids,
            )
            if not np.all(np.isfinite(natural)):
                raise RuntimeError("Non-finite natural logits")
            arrays["natural_logits"][condition_index, indices] = natural
            for source_mode_index in range(len(SOURCE_MODES)):
                arrays["factorial_logits"][
                    condition_index, source_mode_index, ALL_OPEN_MASK, indices
                ] = natural
            for row, qid in enumerate(cohort):
                qi = local_qid_index[qid]
                arrays["trusted_natural_logits"][condition_index, qi] = np.asarray(
                    trusted[condition_index][qid]["aggregated_ad_logits"],
                    dtype=np.float32,
                )

            for availability_mask in range(ALL_OPEN_MASK):
                for source_mode_index, source_mode in enumerate(SOURCE_MODES):
                    specs = _factorial_specs(
                        layer_indices,
                        source_physical,
                        query_classes_physical,
                        wrong_ranks,
                        availability_mask,
                        source_mode,
                    )
                    with BatchedSDPAQuerySourceAttentionAblator(parts, specs):
                        output = _aggregate_logits(
                            _forward(
                                model,
                                parts,
                                batch["input_ids"],
                                batch["attention_mask"],
                            ),
                            variant_ids,
                        )
                    if not np.all(np.isfinite(output)):
                        raise RuntimeError(
                            f"Non-finite {source_mode} mask {availability_mask} logits"
                        )
                    arrays["factorial_logits"][
                        condition_index,
                        source_mode_index,
                        availability_mask,
                        indices,
                    ] = output

            cohort_audit["conditions"][condition] = {
                "rendered_prompt": batch["prompts"][0],
                "prompt_hash": arrays["prompt_hashes"][
                    condition_index, indices[0]
                ].item(),
                "rows": row_audits,
            }

        arrays["completed"][indices] = True
        validation = _validate_completed(arrays)
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"candidate-history entry factorial: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.2f}; natural_error={validation['natural_max_abs_error']}",
            flush=True,
        )
        if not audit_path.exists():
            cohort_audit["token_classes"] = list(TOKEN_CLASSES)
            cohort_audit["availability_masks"] = {
                str(mask): _availability_label(mask) for mask in range(32)
            }
            cohort_audit["wrong_source_assignment"] = (
                "For question index q and target rank r, use wrong source rank "
                "(r + 1 + ((q + r) mod 3)) mod 4."
            )
            audit_path.write_text(
                json.dumps(cohort_audit, indent=2, ensure_ascii=False) + "\n"
            )

    validation = _validate_completed(arrays)
    wrong_ranks_all = arrays["wrong_source_ranks"][arrays["completed"]]
    wrong_offsets = np.mod(wrong_ranks_all - np.arange(4, dtype=np.int8)[None, :], 4)
    balance = {
        str(offset): int(np.sum(wrong_offsets == offset)) for offset in (1, 2, 3)
    }
    if int(np.sum(wrong_offsets == 0)):
        raise RuntimeError("Balanced wrong-line assignment contains matching sources")
    metadata = {
        "experiment": "candidate-history exhaustive 2P receiver-token entry factorial",
        "config": config.as_dict(),
        "n_questions": len(qids),
        "conditions": list(CONDITIONS),
        "ranks": list(RANKS),
        "token_classes": list(TOKEN_CLASSES),
        "availability_masks": {
            str(mask): _availability_label(mask) for mask in range(32)
        },
        "ordinary_layers_one_based": list(ORDINARY_LAYERS),
        "complete_model_forwards_per_cohort": 126,
        "complete_model_work": (
            "Per condition: one natural forward, then matching-source and balanced-"
            "wrong-source interventions for each of 31 non-natural availability masks."
        ),
        "wrong_source_offset_counts": balance,
        "validation": validation,
        "elapsed_seconds_after_load": time.monotonic() - started,
        "cohort_seconds": durations,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.remapping_plan,
        args.baseline,
        args.trusted_game,
        args.trusted_neutral,
        args.output_dir,
        args.max_cohorts,
    )


if __name__ == "__main__":
    main()
