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
from .run_fixed_a_full_cache_factorial import (
    DONOR_ROWS,
    _aggregate_logits,
    _cache_inventory,
    _cached_forward,
    _swap_cache_families,
)
from .run_semantic_binding_module_factorial import (
    CELLS,
    CONDITIONS,
    _forward,
    _messages,
    _remap_question,
)
from .prompts import present_question


SOURCE_CELLS = (
    "identity",
    "selected_option",
    "question_without_selected",
    "first_question",
    "decision_boundary",
    "post_question_without_boundary",
    "selected_plus_boundary",
    "informative_prefix",
    "all_attention_kv",
    "complete_causal_cache",
)


def _initialize(path: Path, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    qids = [row["question_id"] for row in rows]
    if path.exists():
        arrays = dict(np.load(path, allow_pickle=False))
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing output uses a different question order")
        if arrays["source_cells"].astype(str).tolist() != list(SOURCE_CELLS):
            raise ValueError("Existing output uses different source cells")
        return arrays
    n = len(rows)
    return {
        "question_ids": np.asarray(qids),
        "x_second_letter": np.asarray([row["x_second_letter"] for row in rows]),
        "y_second_letter": np.asarray([row["y_second_letter"] for row in rows]),
        "source_cells": np.asarray(SOURCE_CELLS),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "first_decision_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "first_decision_valid": np.zeros(n, dtype=bool),
        "source_logits": np.full(
            (len(SOURCE_CELLS), 4, n, 4), np.nan, dtype=np.float32
        ),
        "source_position_counts": np.zeros(
            (len(SOURCE_CELLS), 4, n), dtype=np.int16
        ),
        "complete_cache_donor_max_abs_error": np.full(n, np.nan, dtype=np.float32),
        "informative_prefix_vs_all_kv_max_abs_error": np.full(
            n, np.nan, dtype=np.float32
        ),
    }


def _token_positions_for_interval(
    offsets: list[tuple[int, int]], start: int, end: int
) -> list[int]:
    return [
        i
        for i, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    ]


def _source_positions(
    tokenizer: Any,
    prompt: str,
    first_question: dict[str, Any],
    boundary: int,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    encoded = tokenizer(
        prompt, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    if boundary >= len(ids):
        raise RuntimeError(f"Boundary {boundary} exceeds prompt length {len(ids)}")

    question_text = present_question(first_question)
    question_start = prompt.find(question_text)
    if question_start < 0:
        raise RuntimeError("Could not locate the first presentation")
    question_end = question_start + len(question_text)
    question_positions = _token_positions_for_interval(
        offsets, question_start, question_end
    )
    if not question_positions:
        raise RuntimeError("First presentation has no token positions")

    option_line = f"  A: {first_question['options']['A']}\n"
    option_start = prompt.find(option_line, question_start, question_end)
    if option_start < 0:
        raise RuntimeError(f"Could not locate selected option line {option_line!r}")
    option_positions = _token_positions_for_interval(
        offsets, option_start, option_start + len(option_line)
    )
    if not option_positions:
        raise RuntimeError("Selected option line has no token positions")

    question_set = set(question_positions)
    option_set = set(option_positions)
    question_without_selected = sorted(question_set - option_set)
    post_question = list(range(max(question_positions) + 1, boundary))
    informative_prefix = list(range(min(question_positions), boundary + 1))
    positions = {
        "identity": [],
        "selected_option": sorted(option_set),
        "question_without_selected": question_without_selected,
        "first_question": sorted(question_set),
        "decision_boundary": [boundary],
        "post_question_without_boundary": post_question,
        "selected_plus_boundary": sorted(option_set | {boundary}),
        "informative_prefix": informative_prefix,
        "all_attention_kv": list(range(boundary + 1)),
        "complete_causal_cache": list(range(boundary + 1)),
    }
    if set(positions) != set(SOURCE_CELLS):
        raise RuntimeError("Source-position inventory does not match source cells")
    if any(position > boundary for values in positions.values() for position in values):
        raise RuntimeError("A source position lies after the transplant boundary")
    audit = {
        "question_character_interval": [question_start, question_end],
        "selected_option_character_interval": [
            option_start,
            option_start + len(option_line),
        ],
        "positions": positions,
        "position_counts": {name: len(values) for name, values in positions.items()},
        "selected_option_text": option_line.rstrip("\n"),
        "boundary_token": tokenizer.decode([ids[boundary]]),
        "selected_option_tokens": tokenizer.convert_ids_to_tokens(
            [ids[i] for i in option_positions]
        ),
        "post_question_tokens": tokenizer.convert_ids_to_tokens(
            [ids[i] for i in post_question]
        ),
    }
    return positions, audit


def _patch_attention_kv_positions(
    source_cache: Any,
    positions_by_row: list[list[int]],
    donor_rows: np.ndarray = DONOR_ROWS,
) -> tuple[Any, int]:
    """Replace donor-position K/V entries while retaining all other cache state."""
    import torch

    if len(positions_by_row) != len(donor_rows):
        raise ValueError("Expected one position list per batch row")
    target = copy.deepcopy(source_cache)
    conventional_layers = 0
    for source_layer, target_layer in zip(source_cache.layers, target.layers):
        keys = getattr(source_layer, "keys", None)
        values = getattr(source_layer, "values", None)
        if keys is None and values is None:
            continue
        if keys is None or values is None or keys.numel() == 0 or values.numel() == 0:
            raise RuntimeError("Conventional-attention K/V initialization mismatch")
        if keys.shape[0] != len(donor_rows) or values.shape[0] != len(donor_rows):
            raise RuntimeError("Unexpected K/V batch dimension")
        if keys.shape[-2] != values.shape[-2]:
            raise RuntimeError("K/V sequence dimensions differ")
        for recipient, donor in enumerate(donor_rows.tolist()):
            positions = positions_by_row[int(donor)]
            if not positions:
                continue
            index = torch.as_tensor(positions, dtype=torch.long, device=keys.device)
            target_layer.keys[recipient, :, index, :] = keys[int(donor), :, index, :]
            value_index = index.to(values.device)
            target_layer.values[recipient, :, value_index, :] = values[
                int(donor), :, value_index, :
            ]
        conventional_layers += 1
    return target, conventional_layers


def run(
    config_path: Path,
    cohort_path: Path,
    output_dir: Path,
    split: str,
    max_questions: int | None,
    shard_index: int,
    num_shards: int,
    full_cache_tolerance: float,
) -> None:
    import torch

    config = ExperimentConfig.load(config_path)
    cohort = json.loads(cohort_path.read_text())
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    rows = [row for row in cohort["rows"] if row["split"] == split]
    rows = [row for i, row in enumerate(rows) if i % num_shards == shard_index]
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

    metadata: dict[str, Any] = {
        "experiment": "fixed-A conventional-attention K/V source-region transplant",
        "config": config.as_dict(),
        "config_path": str(config_path),
        "cohort_path": str(cohort_path),
        "split": split,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "n_questions": len(rows),
        "cells": list(CELLS),
        "source_cells": list(SOURCE_CELLS),
        "donor_rows": DONOR_ROWS.tolist(),
        "complete_model_work_per_question": (
            "one unsplit natural pass; one cached prefix pass; ten cached suffix "
            "passes (identity, seven restricted source-region interventions, "
            "all-attention-K/V control, and complete-causal-cache control)"
        ),
        "full_cache_tolerance": full_cache_tolerance,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    expected_inventory = {
        "attention_kv": 16,
        "gla_conv": 48,
        "gla_recurrent": 48,
    }

    for qi, row in enumerate(rows):
        if arrays["completed"][qi]:
            continue
        question = questions[row["question_id"]]
        first_x = _remap_question(question, row["first_x_new_to_original"])
        first_y = _remap_question(question, row["first_y_new_to_original"])
        second = _remap_question(question, row["second_new_to_original"])

        prompts: list[str] = []
        token_rows: list[list[int]] = []
        boundary_positions: list[int] = []
        source_positions_by_row: list[dict[str, list[int]]] = []
        source_audits: list[dict[str, Any]] = []
        for first, condition in zip((first_x, first_x, first_y, first_y), CONDITIONS):
            prompt = render_chat(
                processor,
                _messages(config, first, second, condition),
                config.disable_thinking,
                config.chat_serialization,
            )
            boundary, ids = _decision_position(tokenizer, prompt)
            positions, audit = _source_positions(
                tokenizer, prompt, first, boundary
            )
            prompts.append(prompt)
            token_rows.append(ids)
            boundary_positions.append(boundary)
            source_positions_by_row.append(positions)
            source_audits.append(audit)

        lengths = [len(ids) for ids in token_rows]
        if len(set(lengths)) != 1 or len(set(boundary_positions)) != 1:
            raise RuntimeError("Fixed-A prompts are not token-aligned")
        cut = boundary_positions[0] + 1
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
        prefix_output = _cached_forward(
            model, parts, input_ids[:, :cut], attention_mask[:, :cut]
        )
        source_cache = prefix_output.past_key_values
        arrays["first_decision_logits"][:, qi] = _aggregate_logits(
            prefix_output, variant_ids
        )
        valid = bool(
            np.all(arrays["first_decision_logits"][:, qi].argmax(axis=-1) == 0)
        )
        arrays["first_decision_valid"][qi] = valid
        inventory = _cache_inventory(source_cache)
        if inventory != expected_inventory:
            raise RuntimeError(
                f"Unexpected cache inventory {inventory}; expected {expected_inventory}"
            )
        if not valid:
            arrays["completed"][qi] = True
            atomic_save_npz(result_path, **arrays)
            answers = "".join(
                "ABCD"[i]
                for i in arrays["first_decision_logits"][:, qi].argmax(axis=-1)
            )
            print(
                f"fixed-A K/V source {split} shard {shard_index}: screened out "
                f"{row['question_id']} (first decisions {answers})",
                flush=True,
            )
            continue

        suffix_ids = input_ids[:, cut:]
        for cell_index, cell in enumerate(SOURCE_CELLS):
            positions_by_row = [row_positions[cell] for row_positions in source_positions_by_row]
            arrays["source_position_counts"][cell_index, :, qi] = [
                len(values) for values in positions_by_row
            ]
            if cell == "complete_causal_cache":
                cache, counts = _swap_cache_families(source_cache, 7)
                if counts != expected_inventory:
                    raise RuntimeError(
                        f"Complete causal cache swapped {counts}; expected {expected_inventory}"
                    )
            else:
                cache, layer_count = _patch_attention_kv_positions(
                    source_cache, positions_by_row
                )
                if layer_count != 16:
                    raise RuntimeError(
                        f"Cell {cell} patched {layer_count} attention layers, expected 16"
                    )
            output = _cached_forward(
                model,
                parts,
                suffix_ids,
                attention_mask,
                past_key_values=cache,
            )
            arrays["source_logits"][cell_index, :, qi] = _aggregate_logits(
                output, variant_ids
            )
            del cache, output

        identity = arrays["source_logits"][0, :, qi]
        donor_identity = identity[DONOR_ROWS]
        complete = arrays["source_logits"][
            SOURCE_CELLS.index("complete_causal_cache"), :, qi
        ]
        all_kv = arrays["source_logits"][
            SOURCE_CELLS.index("all_attention_kv"), :, qi
        ]
        informative = arrays["source_logits"][
            SOURCE_CELLS.index("informative_prefix"), :, qi
        ]
        complete_error = float(np.max(np.abs(complete - donor_identity)))
        informative_error = float(np.max(np.abs(informative - all_kv)))
        arrays["complete_cache_donor_max_abs_error"][qi] = complete_error
        arrays["informative_prefix_vs_all_kv_max_abs_error"][qi] = informative_error
        if complete_error > full_cache_tolerance:
            raise RuntimeError(
                f"Complete-cache positive control failed on {row['question_id']}: "
                f"max A-D error {complete_error:.6g} > {full_cache_tolerance:.6g}"
            )

        arrays["completed"][qi] = True
        atomic_save_npz(result_path, **arrays)
        done = int(arrays["completed"].sum())
        if done == 1 or done % 5 == 0 or done == len(rows):
            print(
                f"fixed-A K/V source {split} shard {shard_index}: "
                f"{done}/{len(rows)}; complete={complete_error:.3g}; "
                f"informative-vs-all-kv={informative_error:.3g}",
                flush=True,
            )

        if qi == 0 and not (output_dir / "prompt_audit.json").exists():
            (output_dir / "prompt_audit.json").write_text(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "literal_first_answer": row["literal_first_answer"],
                        "boundary_positions": boundary_positions,
                        "cut": cut,
                        "cache_inventory": inventory,
                        "source_audits": dict(zip(CELLS, source_audits)),
                        "suffix_identity": {
                            "evaluation_x_equals_y": token_rows[0][cut:] == token_rows[2][cut:],
                            "neutral_x_equals_y": token_rows[1][cut:] == token_rows[3][cut:],
                        },
                        "rendered_prompts": dict(zip(CELLS, prompts)),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
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
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--full-cache-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    run(
        args.config,
        args.cohort,
        args.output,
        args.split,
        args.max_questions,
        args.shard_index,
        args.num_shards,
        args.full_cache_tolerance,
    )


if __name__ == "__main__":
    main()
