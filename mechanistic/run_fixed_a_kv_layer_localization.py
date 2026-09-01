from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
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
from .run_first_decision_cross_order_patching import _decision_position
from .run_fixed_a_full_cache_factorial import DONOR_ROWS, _aggregate_logits, _cache_inventory, _cached_forward
from .run_fixed_a_kv_source_transplant import _source_positions
from .run_semantic_binding_module_factorial import CELLS, CONDITIONS, _forward, _messages, _remap_question


ATTENTION_BLOCKS = tuple(range(4, 65, 4))
BANDS = {
    "band_04_16": (4, 8, 12, 16),
    "band_20_32": (20, 24, 28, 32),
    "band_36_48": (36, 40, 44, 48),
    "band_52_64": (52, 56, 60, 64),
}
LAYER_CELLS = (
    "identity",
    *(f"block_{block:02d}" for block in ATTENTION_BLOCKS),
    *BANDS,
    *(f"without_{name}" for name in BANDS),
    "all_selected_option",
)


def _target_blocks(cell: str) -> tuple[int, ...]:
    if cell == "identity":
        return ()
    if cell.startswith("block_"):
        return (int(cell.rsplit("_", 1)[1]),)
    if cell in BANDS:
        return BANDS[cell]
    if cell.startswith("without_"):
        omitted = set(BANDS[cell.removeprefix("without_")])
        return tuple(block for block in ATTENTION_BLOCKS if block not in omitted)
    if cell == "all_selected_option":
        return ATTENTION_BLOCKS
    raise KeyError(cell)


def _initialize(path: Path, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    qids = [row["question_id"] for row in rows]
    if path.exists():
        arrays = dict(np.load(path, allow_pickle=False))
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing output uses a different question order")
        if arrays["layer_cells"].astype(str).tolist() != list(LAYER_CELLS):
            raise ValueError("Existing output uses different layer cells")
        return arrays
    n = len(rows)
    return {
        "question_ids": np.asarray(qids),
        "x_second_letter": np.asarray([row["x_second_letter"] for row in rows]),
        "y_second_letter": np.asarray([row["y_second_letter"] for row in rows]),
        "layer_cells": np.asarray(LAYER_CELLS),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "first_decision_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "first_decision_valid": np.zeros(n, dtype=bool),
        "layer_logits": np.full((len(LAYER_CELLS), 4, n, 4), np.nan, dtype=np.float32),
        "selected_position_counts": np.zeros((4, n), dtype=np.int16),
        "patched_layer_counts": np.zeros(len(LAYER_CELLS), dtype=np.int16),
    }


def _patch_selected_option_layers(
    source_cache: Any,
    positions_by_row: list[list[int]],
    target_blocks: tuple[int, ...],
) -> tuple[Any, tuple[int, ...]]:
    import torch

    target = copy.deepcopy(source_cache)
    target_set = set(target_blocks)
    ordinary_blocks: list[int] = []
    patched_blocks: list[int] = []
    for block, (source_layer, target_layer) in enumerate(
        zip(source_cache.layers, target.layers), start=1
    ):
        keys = getattr(source_layer, "keys", None)
        values = getattr(source_layer, "values", None)
        if keys is None and values is None:
            continue
        if keys is None or values is None or keys.numel() == 0 or values.numel() == 0:
            raise RuntimeError(f"Conventional-attention K/V mismatch at block {block}")
        ordinary_blocks.append(block)
        if block not in target_set:
            continue
        for recipient, donor in enumerate(DONOR_ROWS.tolist()):
            positions = positions_by_row[int(donor)]
            index = torch.as_tensor(positions, dtype=torch.long, device=keys.device)
            target_layer.keys[recipient, :, index, :] = keys[int(donor), :, index, :]
            value_index = index.to(values.device)
            target_layer.values[recipient, :, value_index, :] = values[
                int(donor), :, value_index, :
            ]
        patched_blocks.append(block)
    if tuple(ordinary_blocks) != ATTENTION_BLOCKS:
        raise RuntimeError(
            f"Expected ordinary attention at {ATTENTION_BLOCKS}; found {tuple(ordinary_blocks)}"
        )
    if tuple(patched_blocks) != tuple(target_blocks):
        raise RuntimeError(
            f"Requested blocks {target_blocks}; patched {tuple(patched_blocks)}"
        )
    return target, tuple(patched_blocks)


def run(
    config_path: Path,
    cohort_path: Path,
    output_dir: Path,
    split: str,
    max_questions: int | None,
) -> None:
    import torch

    config = ExperimentConfig.load(config_path)
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
    variant_tokens = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: [token_id for _, token_id in variant_tokens[letter]]
        for letter in "ABCD"
    }
    metadata = {
        "experiment": "fixed-A selected-option conventional-attention K/V layer localization",
        "config": config.as_dict(),
        "config_path": str(config_path),
        "cohort_path": str(cohort_path),
        "split": split,
        "n_questions": len(rows),
        "cells": list(CELLS),
        "layer_cells": list(LAYER_CELLS),
        "attention_blocks_one_based": list(ATTENTION_BLOCKS),
        "bands": {key: list(value) for key, value in BANDS.items()},
        "donor_rows": DONOR_ROWS.tolist(),
        "complete_model_work_per_question": (
            f"one unsplit full forward, one cached prefix forward, and {len(LAYER_CELLS)} "
            "cached suffix forwards; screened questions stop after the prefix"
        ),
        "python": sys.version,
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    expected_inventory = {"attention_kv": 16, "gla_conv": 48, "gla_recurrent": 48}

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
        audits: list[dict[str, Any]] = []
        for first, condition in zip((first_x, first_x, first_y, first_y), CONDITIONS):
            prompt = render_chat(
                processor,
                _messages(config, first, second, condition),
                config.disable_thinking,
                config.chat_serialization,
            )
            boundary, ids = _decision_position(tokenizer, prompt)
            positions, audit = _source_positions(tokenizer, prompt, first, boundary)
            prompts.append(prompt)
            token_rows.append(ids)
            boundaries.append(boundary)
            selected_positions.append(positions["selected_option"])
            audits.append(audit)
        lengths = [len(ids) for ids in token_rows]
        if len(set(lengths)) != 1 or len(set(boundaries)) != 1:
            raise RuntimeError("Fixed-A prompts are not token-aligned")
        cut = boundaries[0] + 1
        if token_rows[0][cut:] != token_rows[2][cut:]:
            raise RuntimeError("Evaluation X/Y suffixes differ after the boundary")
        if token_rows[1][cut:] != token_rows[3][cut:]:
            raise RuntimeError("Neutral X/Y suffixes differ after the boundary")

        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        device = model_input_device(parts)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        natural_output = _forward(model, parts, input_ids, attention_mask)
        arrays["natural_logits"][:, qi] = _aggregate_logits(natural_output, variant_ids)
        prefix_output = _cached_forward(model, parts, input_ids[:, :cut], attention_mask[:, :cut])
        source_cache = prefix_output.past_key_values
        arrays["first_decision_logits"][:, qi] = _aggregate_logits(prefix_output, variant_ids)
        valid = bool(np.all(arrays["first_decision_logits"][:, qi].argmax(axis=-1) == 0))
        arrays["first_decision_valid"][qi] = valid
        arrays["selected_position_counts"][:, qi] = [len(x) for x in selected_positions]
        inventory = _cache_inventory(source_cache)
        if inventory != expected_inventory:
            raise RuntimeError(f"Unexpected cache inventory {inventory}; expected {expected_inventory}")
        if not valid:
            arrays["completed"][qi] = True
            atomic_save_npz(result_path, **arrays)
            print(f"fixed-A K/V layers {split}: screened {row['question_id']}", flush=True)
            continue

        suffix_ids = input_ids[:, cut:]
        for cell_index, cell in enumerate(LAYER_CELLS):
            targets = _target_blocks(cell)
            cache, patched = _patch_selected_option_layers(source_cache, selected_positions, targets)
            arrays["patched_layer_counts"][cell_index] = len(patched)
            output = _cached_forward(
                model, parts, suffix_ids, attention_mask, past_key_values=cache
            )
            arrays["layer_logits"][cell_index, :, qi] = _aggregate_logits(output, variant_ids)
            del cache, output

        arrays["completed"][qi] = True
        atomic_save_npz(result_path, **arrays)
        done = int(arrays["completed"].sum())
        if done == 1 or done % 5 == 0 or done == len(rows):
            print(f"fixed-A K/V layers {split}: {done}/{len(rows)}", flush=True)
        if qi == 0 and not (output_dir / "prompt_audit.json").exists():
            (output_dir / "prompt_audit.json").write_text(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "cut": cut,
                        "cache_inventory": inventory,
                        "selected_option_audits": dict(zip(CELLS, audits)),
                        "rendered_prompts": dict(zip(CELLS, prompts)),
                    },
                    indent=2,
                    ensure_ascii=False,
                ) + "\n"
            )
        del source_cache, prefix_output, natural_output
        torch.cuda.empty_cache()


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
