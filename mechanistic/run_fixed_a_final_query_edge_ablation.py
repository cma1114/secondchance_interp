from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSDPAFinalQueryAttentionAblator
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import present_question
from .run_first_decision_cross_order_patching import _decision_position
from .run_semantic_binding_module_factorial import (
    CELLS,
    CONDITIONS,
    _aggregate_logits,
    _forward,
    _messages,
    _remap_question,
)


BLOCK_SETS = {
    "block_44": (44,),
    "band_36_48": (36, 40, 44, 48),
    "all_04_48": tuple(range(4, 49, 4)),
}
INTERVENTION_CELLS = tuple(
    f"{name}_{source}"
    for name in BLOCK_SETS
    for source in ("selected", "matched_control")
)


def _token_positions_for_interval(
    offsets: list[tuple[int, int]], start: int, end: int
) -> list[int]:
    return [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    ]


def _option_line_positions(
    tokenizer: Any,
    prompt: str,
    first_question: dict[str, Any],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    encoded = tokenizer(
        prompt, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    question_text = present_question(first_question)
    question_start = prompt.find(question_text)
    if question_start < 0:
        raise RuntimeError("Could not locate the first presentation")
    question_end = question_start + len(question_text)
    positions: dict[str, list[int]] = {}
    audit: dict[str, Any] = {}
    for letter in "ABCD":
        line = f"  {letter}: {first_question['options'][letter]}\n"
        start = prompt.find(line, question_start, question_end)
        if start < 0:
            raise RuntimeError(f"Could not locate first option line {line!r}")
        row = _token_positions_for_interval(offsets, start, start + len(line))
        if not row:
            raise RuntimeError(f"First option line {letter} has no tokens")
        positions[letter] = row
        audit[letter] = {
            "text": line.rstrip("\n"),
            "positions": row,
            "tokens": tokenizer.convert_ids_to_tokens([ids[index] for index in row]),
        }
    return positions, audit


def _matched_control(option_positions: dict[str, list[int]]) -> str:
    selected_count = len(option_positions["A"])
    # Deterministic nearest-token-count negative control.  Ties are resolved
    # alphabetically and recorded in the prompt audit.
    return min("BCD", key=lambda letter: (abs(len(option_positions[letter]) - selected_count), letter))


def _initialize(path: Path, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    qids = [row["question_id"] for row in rows]
    if path.exists():
        arrays = dict(np.load(path, allow_pickle=False))
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing output uses a different question order")
        if arrays["intervention_cells"].astype(str).tolist() != list(INTERVENTION_CELLS):
            raise ValueError("Existing output uses different intervention cells")
        return arrays
    n = len(rows)
    return {
        "question_ids": np.asarray(qids),
        "x_second_letter": np.asarray([row["x_second_letter"] for row in rows]),
        "y_second_letter": np.asarray([row["y_second_letter"] for row in rows]),
        "intervention_cells": np.asarray(INTERVENTION_CELLS),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "first_decision_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "first_decision_valid": np.zeros(n, dtype=bool),
        "intervention_logits": np.full(
            (len(INTERVENTION_CELLS), 4, n, 4), np.nan, dtype=np.float32
        ),
        "selected_position_counts": np.zeros((4, n), dtype=np.int16),
        "control_position_counts": np.zeros((4, n), dtype=np.int16),
        "control_letters": np.full((4, n), "", dtype="<U1"),
    }


def _prefix_forward(model: Any, parts: Any, input_ids: Any, attention_mask: Any):
    import torch

    with torch.inference_mode():
        kwargs = {
            "input_ids": input_ids.to(model_input_device(parts)),
            "attention_mask": attention_mask.to(model_input_device(parts)),
            "use_cache": False,
            "return_dict": True,
        }
        try:
            return model(**kwargs, logits_to_keep=1)
        except TypeError:
            return model(**kwargs)


def run(
    config_path: Path,
    cohort_path: Path,
    output_dir: Path,
    split: str,
    max_questions: int | None,
) -> None:
    config = ExperimentConfig.load(config_path)
    if config.attn_implementation != "sdpa" or config.batch_size != 4:
        raise ValueError("Requires the established batch-of-four SDPA regime")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw_qwen_chatml serialization")

    cohort = json.loads(cohort_path.read_text())
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    rows = [row for row in cohort["rows"] if row["split"] == split]
    if max_questions is not None:
        rows = rows[: int(max_questions)]
    if not rows:
        raise ValueError("No cohort rows selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, rows)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    variants = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: [token_id for _, token_id in variants[letter]]
        for letter in "ABCD"
    }
    ordinary_blocks = tuple(
        index + 1
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    expected_blocks = tuple(range(4, 65, 4))
    if ordinary_blocks != expected_blocks:
        raise RuntimeError(
            f"Expected ordinary attention at {expected_blocks}; found {ordinary_blocks}"
        )

    metadata = {
        "experiment": "fixed-A final-query selected-option attention-edge ablation",
        "config": config.as_dict(),
        "config_path": str(config_path),
        "cohort_path": str(cohort_path),
        "split": split,
        "n_questions": len(rows),
        "cells": list(CELLS),
        "intervention_cells": list(INTERVENTION_CELLS),
        "block_sets_one_based": {key: list(value) for key, value in BLOCK_SETS.items()},
        "intervention": (
            "Set only the final-decision-query attention-mask entries to the "
            "first-presentation selected A option line to -inf in all heads of "
            "the specified ordinary-attention blocks; matched controls use the "
            "nearest-token-count unselected option line."
        ),
        "complete_model_work_per_question": (
            "one natural full-prompt batch, one first-decision prefix batch, and "
            f"{len(INTERVENTION_CELLS)} intervened full-prompt batches"
        ),
        "python": sys.version,
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    for qi, row in enumerate(rows):
        if arrays["completed"][qi]:
            continue
        question = questions[row["question_id"]]
        first_x = _remap_question(question, row["first_x_new_to_original"])
        first_y = _remap_question(question, row["first_y_new_to_original"])
        second = _remap_question(question, row["second_new_to_original"])
        prompts: list[str] = []
        token_rows: list[list[int]] = []
        boundaries: list[int] = []
        selected_positions: list[list[int]] = []
        control_positions: list[list[int]] = []
        control_letters: list[str] = []
        audits: list[dict[str, Any]] = []
        for first, condition in zip((first_x, first_x, first_y, first_y), CONDITIONS):
            prompt = render_chat(
                processor,
                _messages(config, first, second, condition),
                config.disable_thinking,
                config.chat_serialization,
            )
            boundary, ids = _decision_position(tokenizer, prompt)
            option_positions, option_audit = _option_line_positions(
                tokenizer, prompt, first
            )
            control = _matched_control(option_positions)
            prompts.append(prompt)
            token_rows.append(ids)
            boundaries.append(boundary)
            selected_positions.append(option_positions["A"])
            control_positions.append(option_positions[control])
            control_letters.append(control)
            audits.append({"options": option_audit, "control_letter": control})

        lengths = [len(ids) for ids in token_rows]
        if len(set(lengths)) != 1 or len(set(boundaries)) != 1:
            raise RuntimeError("Fixed-A prompts are not token-aligned")
        cut = boundaries[0] + 1
        if token_rows[0][cut:] != token_rows[2][cut:]:
            raise RuntimeError("Evaluation X/Y suffixes differ after the first decision")
        if token_rows[1][cut:] != token_rows[3][cut:]:
            raise RuntimeError("Neutral X/Y suffixes differ after the first decision")

        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        natural = _forward(model, parts, input_ids, attention_mask)
        arrays["natural_logits"][:, qi] = _aggregate_logits(natural, variant_ids)
        prefix = _prefix_forward(
            model, parts, input_ids[:, :cut], attention_mask[:, :cut]
        )
        arrays["first_decision_logits"][:, qi] = _aggregate_logits(prefix, variant_ids)
        arrays["first_decision_valid"][qi] = bool(
            np.all(arrays["first_decision_logits"][:, qi].argmax(axis=-1) == 0)
        )
        arrays["selected_position_counts"][:, qi] = [
            len(values) for values in selected_positions
        ]
        arrays["control_position_counts"][:, qi] = [
            len(values) for values in control_positions
        ]
        arrays["control_letters"][:, qi] = np.asarray(control_letters)

        if arrays["first_decision_valid"][qi]:
            for cell_index, cell in enumerate(INTERVENTION_CELLS):
                block_name, source_name = cell.rsplit("_", 1)
                # ``matched_control`` contains an underscore, so repair the
                # deterministic cell split.
                if cell.endswith("_matched_control"):
                    block_name = cell.removesuffix("_matched_control")
                    source_name = "matched_control"
                elif cell.endswith("_selected"):
                    block_name = cell.removesuffix("_selected")
                    source_name = "selected"
                positions = (
                    selected_positions if source_name == "selected" else control_positions
                )
                layer_specs = {
                    block - 1: {batch_row: values for batch_row, values in enumerate(positions)}
                    for block in BLOCK_SETS[block_name]
                }
                with BatchedSDPAFinalQueryAttentionAblator(parts, layer_specs):
                    output = _forward(model, parts, input_ids, attention_mask)
                arrays["intervention_logits"][cell_index, :, qi] = _aggregate_logits(
                    output, variant_ids
                )

        arrays["completed"][qi] = True
        atomic_save_npz(result_path, **arrays)
        done = int(arrays["completed"].sum())
        if done == 1 or done % 5 == 0 or done == len(rows):
            status = "eligible" if arrays["first_decision_valid"][qi] else "screened"
            print(
                f"fixed-A final-query edge {split}: {done}/{len(rows)} ({status})",
                flush=True,
            )
        if qi == 0 and not (output_dir / "prompt_audit.json").exists():
            (output_dir / "prompt_audit.json").write_text(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "cut": cut,
                        "first_decision_token": tokenizer.decode([token_rows[0][boundaries[0]]]),
                        "final_query_position": len(token_rows[0]) - 1,
                        "final_query_token": tokenizer.decode([token_rows[0][-1]]),
                        "rows": dict(zip(CELLS, audits)),
                        "rendered_prompts": dict(zip(CELLS, prompts)),
                    },
                    indent=2,
                    ensure_ascii=False,
                ) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    run(args.config, args.cohort, args.output, args.split, args.max_questions)


if __name__ == "__main__":
    main()
