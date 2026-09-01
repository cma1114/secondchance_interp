from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSDPAFinalQueryAttentionAblator
from .io import atomic_save_npz
from .modeling import get_tokenizer, load_model_and_processor, resolve_answer_tokens
from .prompts import present_question
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_fixed_a_final_query_edge_ablation import _token_positions_for_interval


SOURCE_ROLES = ("w1", "other_1", "other_2", "other_3")
ORDINARY_BLOCKS = tuple(range(4, 65, 4))


def _second_option_line_positions(
    tokenizer: Any,
    prompt: str,
    second_question: dict[str, Any],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    encoded = tokenizer(
        prompt, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    question_text = present_question(second_question)
    question_start = prompt.rfind(question_text)
    if question_start < 0:
        raise RuntimeError("Could not locate the second presentation")
    question_end = question_start + len(question_text)
    positions: dict[str, list[int]] = {}
    audit: dict[str, Any] = {}
    for displayed in LETTERS:
        line = f"  {displayed}: {second_question['options'][displayed]}\n"
        start = prompt.find(line, question_start, question_end)
        if start < 0:
            raise RuntimeError(f"Could not locate second option line {line!r}")
        row = _token_positions_for_interval(offsets, start, start + len(line))
        if not row:
            raise RuntimeError(f"Second option line {displayed} has no tokens")
        positions[displayed] = row
        audit[displayed] = {
            "text": line.rstrip("\n"),
            "positions": row,
            "tokens": tokenizer.convert_ids_to_tokens([ids[index] for index in row]),
        }
    return positions, audit


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Question IDs changed")
        if arrays["source_roles"].astype(str).tolist() != list(SOURCE_ROLES):
            raise ValueError("Source roles changed")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "source_roles": np.asarray(SOURCE_ROLES),
        "completed": np.zeros(n, dtype=bool),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "intervention_logits": np.full((2, 4, n, 4), np.nan, dtype=np.float32),
        "source_position_counts": np.zeros((2, 4, n), dtype=np.int16),
        "source_original_letters": np.full((4, n), "", dtype="<U1"),
        "source_display_letters": np.full((4, n), "", dtype="<U1"),
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
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw_qwen_chatml serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    if max_cohorts is not None:
        qids = qids[: int(max_cohorts) * config.batch_size]
    mappings = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    baseline = json.loads(baseline_path.read_text())["results"]
    trusted = [
        json.loads(trusted_game_path.read_text())["results"],
        json.loads(trusted_neutral_path.read_text())["results"],
    ]

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    ordinary_blocks = tuple(
        index + 1
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if ordinary_blocks != ORDINARY_BLOCKS:
        raise RuntimeError(f"Unexpected ordinary-attention blocks: {ordinary_blocks}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = output_dir / "prompt_audit.json"
    started = time.monotonic()

    for start in range(0, len(qids), config.batch_size):
        cohort = qids[start : start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if np.all(arrays["completed"][indices]):
            continue
        cohort_started = time.monotonic()
        batches = [
            _build_batch(config, processor, tokenizer, questions, mappings, cohort, condition)
            for condition in CONDITIONS
        ]
        cohort_audit: dict[str, Any] = {"question_ids": cohort, "conditions": {}}

        for ci, (condition, batch) in enumerate(zip(CONDITIONS, batches)):
            natural = _aggregate_logits(
                _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                variant_ids,
            )
            width = int(batch["input_ids"].shape[1])
            source_positions: list[list[list[int]]] = []
            audits: list[dict[str, Any]] = []
            for row, qid in enumerate(cohort):
                mapping = mappings[qid]
                remapped_question = {
                    **questions[qid],
                    "options": {
                        new: questions[qid]["options"][old]
                        for new, old in mapping["new_to_original"].items()
                    },
                }
                positions, option_audit = _second_option_line_positions(
                    tokenizer, batch["prompts"][row], remapped_question
                )
                w1 = baseline[qid].get("answer", baseline[qid].get("subject_answer"))
                if w1 not in LETTERS:
                    raise RuntimeError(f"Invalid W1 for {qid}: {w1!r}")
                semantic_order = [w1] + [letter for letter in LETTERS if letter != w1]
                displayed_order = [mapping["original_to_new"][letter] for letter in semantic_order]
                left_pad = width - len(batch["token_rows"][row])
                row_positions = [
                    [left_pad + position for position in positions[displayed]]
                    for displayed in displayed_order
                ]
                source_positions.append(row_positions)
                qi = qid_index[qid]
                arrays["source_original_letters"][:, qi] = semantic_order
                arrays["source_display_letters"][:, qi] = displayed_order
                for source_index, values in enumerate(row_positions):
                    arrays["source_position_counts"][ci, source_index, qi] = len(values)
                audits.append({
                    "W1_original_letter": w1,
                    "source_lines": [
                        {
                            "role": SOURCE_ROLES[source_index],
                            "original_semantic_letter": semantic_order[source_index],
                            "displayed_letter": displayed_order[source_index],
                            **option_audit[displayed_order[source_index]],
                        }
                        for source_index in range(4)
                    ],
                })

            for local, qid in enumerate(cohort):
                qi = qid_index[qid]
                arrays["same_batch_natural_logits"][ci, qi] = natural[local]
                arrays["trusted_natural_logits"][ci, qi] = np.asarray(
                    trusted[ci][qid]["aggregated_ad_logits"], dtype=np.float32
                )

            for source_index in range(4):
                layer_specs = {
                    block - 1: {
                        row: source_positions[row][source_index]
                        for row in range(len(cohort))
                    }
                    for block in ORDINARY_BLOCKS
                }
                with BatchedSDPAFinalQueryAttentionAblator(parts, layer_specs):
                    intervened = _aggregate_logits(
                        _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                        variant_ids,
                    )
                for local, qid in enumerate(cohort):
                    arrays["intervention_logits"][ci, source_index, qid_index[qid]] = intervened[local]

            cohort_audit["conditions"][condition] = {
                "rendered_prompt": batch["prompts"][0],
                "final_query_position_physical": width - 1,
                "final_query_token": tokenizer.decode([int(batch["input_ids"][0, -1])]),
                "ordinary_blocks": list(ORDINARY_BLOCKS),
                "rows": audits,
            }

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        print(
            f"final-query repeated-option edge: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={time.monotonic() - cohort_started:.1f}",
            flush=True,
        )
        if not audit_path.exists():
            audit_path.write_text(json.dumps(cohort_audit, indent=2, ensure_ascii=False) + "\n")

    metadata = {
        "experiment": "canonical remapped final-query to repeated-option attention-edge ablation",
        "config": config.as_dict(),
        "remapping_plan": str(remapping_plan_path),
        "baseline": str(baseline_path),
        "trusted_game": str(trusted_game_path),
        "trusted_neutral": str(trusted_neutral_path),
        "n_questions": len(qids),
        "conditions": list(CONDITIONS),
        "source_roles": list(SOURCE_ROLES),
        "ordinary_blocks_one_based": list(ORDINARY_BLOCKS),
        "historical_batch_size": config.batch_size,
        "complete_model_forwards_per_cohort": 2 * (1 + len(SOURCE_ROLES)),
        "intervention": (
            "At only the final pre-answer query, block every head in all ordinary-attention "
            "blocks 4-64 from reading one complete second-presentation option line. The primary "
            "source is the line containing semantic W1; all three other repeated option lines "
            "are separately tested controls."
        ),
        "elapsed_seconds_after_load": time.monotonic() - started,
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
