from __future__ import annotations

import argparse
import copy
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
from .prompts import present_question, prompt_hash
from .run_first_decision_cross_order_patching import _decision_position
from .run_fixed_a_full_cache_factorial import (
    DONOR_ROWS,
    _aggregate_logits,
    _cache_inventory,
    _cached_forward,
    _swap_cache_families,
)
from .run_fixed_a_kv_source_transplant import _patch_attention_kv_positions
from .run_semantic_binding_module_factorial import CONDITIONS, _forward, _messages, _remap_question


SCENARIOS = ("recipient_open", "donor_open")


def option_line_positions(
    tokenizer: Any,
    prompt: str,
    question: dict[str, Any],
    occurrence: str,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    question_text = present_question(question)
    question_start = prompt.find(question_text) if occurrence == "first" else prompt.rfind(question_text)
    if question_start < 0:
        raise RuntimeError(f"Could not locate {occurrence} question")
    question_end = question_start + len(question_text)
    positions: dict[str, list[int]] = {}
    audit: dict[str, Any] = {}
    for letter in "ABCD":
        line = f"  {letter}: {question['options'][letter]}\n"
        start = prompt.find(line, question_start, question_end)
        if start < 0:
            raise RuntimeError(f"Could not locate {occurrence} option line {line!r}")
        end = start + len(line)
        row = [
            index for index, (left, right) in enumerate(offsets)
            if right > left and left < end and right > start
        ]
        if not row or row != list(range(row[0], row[-1] + 1)):
            raise RuntimeError("Option-line token positions are absent or noncontiguous")
        positions[letter] = row
        audit[letter] = {
            "text": line.rstrip("\n"),
            "positions": row,
            "tokens": tokenizer.convert_ids_to_tokens([ids[index] for index in row]),
        }
    return positions, audit


def _initialize(path: Path, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    pair_ids = [f"{row['question_id']}:{row['literal_first_answer']}" for row in rows]
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["pair_ids"].astype(str).tolist() != pair_ids:
            raise ValueError("Existing checkpoint uses different pairs")
        return arrays
    n = len(rows)
    return {
        "pair_ids": np.asarray(pair_ids),
        "question_ids": np.asarray([row["question_id"] for row in rows]),
        "literal_letters": np.asarray([row["literal_first_answer"] for row in rows]),
        "x_content": np.asarray([row["x_content_original_letter"] for row in rows]),
        "y_content": np.asarray([row["y_content_original_letter"] for row in rows]),
        "x_second_letter": np.asarray([row["x_second_letter"] for row in rows]),
        "y_second_letter": np.asarray([row["y_second_letter"] for row in rows]),
        "scenarios": np.asarray(SCENARIOS),
        "completed": np.zeros(n, dtype=bool),
        "first_decision_valid": np.zeros(n, dtype=bool),
        "first_decision_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "natural_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "scenario_logits": np.full((2, 4, n, 4), np.nan, dtype=np.float32),
        "source_position_counts": np.zeros((4, n), dtype=np.int16),
        "cached_identity_max_abs_error": np.full(n, np.nan, dtype=np.float32),
        "complete_cache_donor_max_abs_error": np.full(n, np.nan, dtype=np.float32),
    }


def run(
    config_path: Path,
    cohort_path: Path,
    output_dir: Path,
    split: str,
    max_questions: int | None,
    validate_complete_cache: bool,
    complete_cache_tolerance: float,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.attn_implementation != "sdpa" or int(config.batch_size) != 4:
        raise ValueError("Requires the canonical batch-of-four SDPA regime")
    cohort = json.loads(cohort_path.read_text())
    rows = [row for row in cohort["rows"] if row["split"] == split]
    if max_questions is not None:
        rows = rows[: int(max_questions)]
    if not rows:
        raise ValueError("No cohort rows selected")
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, rows)

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    variants = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {letter: [token_id for _, token_id in variants[letter]] for letter in "ABCD"}
    expected_inventory = {"attention_kv": 16, "gla_conv": 48, "gla_recurrent": 48}
    ordinary_blocks = tuple(
        index + 1 for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if ordinary_blocks != tuple(range(4, 65, 4)):
        raise RuntimeError(f"Unexpected ordinary-attention layers {ordinary_blocks}")
    started = time.monotonic()
    durations: list[float] = []

    for qi, row in enumerate(rows):
        if arrays["completed"][qi]:
            continue
        question_started = time.monotonic()
        question = questions[row["question_id"]]
        first_x = _remap_question(question, row["first_x_new_to_original"])
        first_y = _remap_question(question, row["first_y_new_to_original"])
        second = _remap_question(question, row["second_new_to_original"])
        literal = row["literal_first_answer"]

        prompts: list[str] = []
        token_rows: list[list[int]] = []
        boundaries: list[int] = []
        source_rows: list[list[int]] = []
        audits: list[dict[str, Any]] = []
        for cell, (first, condition) in enumerate(zip((first_x, first_x, first_y, first_y), CONDITIONS)):
            prompt = render_chat(
                processor,
                _messages(config, first, second, condition),
                config.disable_thinking,
                config.chat_serialization,
            )
            boundary, ids = _decision_position(tokenizer, prompt)
            first_positions, first_audit = option_line_positions(tokenizer, prompt, first, "first")
            prompts.append(prompt)
            token_rows.append(ids)
            boundaries.append(boundary)
            source_rows.append(first_positions[literal])
            audits.append({
                "cell": cell,
                "condition": condition,
                "literal": literal,
                "source": first_audit[literal],
                "prompt_hash": prompt_hash(prompt),
            })

        if len(set(map(len, token_rows))) != 1 or len(set(boundaries)) != 1:
            raise RuntimeError("Paired prompts are not token aligned")
        if any(values != source_rows[0] for values in source_rows[1:]):
            raise RuntimeError("Selected source lines do not occupy identical token positions")
        expected_positions = list(range(row["selected_line_start"], row["selected_line_end"]))
        if source_rows[0] != expected_positions:
            raise RuntimeError("Causal prompt source span differs from screened source span")
        if token_rows[0] == token_rows[2]:
            raise RuntimeError("Donor and recipient first histories are identical")
        cut = boundaries[0] + 1
        if token_rows[0][cut:] != token_rows[2][cut:] or token_rows[1][cut:] != token_rows[3][cut:]:
            raise RuntimeError("X/Y suffixes differ after the first-decision boundary")

        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        device = model_input_device(parts)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        natural = _aggregate_logits(_forward(model, parts, input_ids, attention_mask), variant_ids)
        arrays["natural_logits"][:, qi] = natural
        prefix = _cached_forward(model, parts, input_ids[:, :cut], attention_mask[:, :cut])
        source_cache = prefix.past_key_values
        arrays["first_decision_logits"][:, qi] = _aggregate_logits(prefix, variant_ids)
        target_index = "ABCD".index(literal)
        valid = bool(np.all(arrays["first_decision_logits"][:, qi].argmax(axis=-1) == target_index))
        arrays["first_decision_valid"][qi] = valid
        if _cache_inventory(source_cache) != expected_inventory:
            raise RuntimeError("Unexpected hybrid-cache inventory")
        arrays["source_position_counts"][:, qi] = [len(values) for values in source_rows]
        if not valid:
            arrays["completed"][qi] = True
            atomic_save_npz(result_path, **arrays)
            print(f"fixed-{literal} transplant {split}: screened {row['question_id']}", flush=True)
            continue

        suffix_ids = input_ids[:, cut:]
        identity_output = _cached_forward(
            model, parts, suffix_ids, attention_mask, past_key_values=copy.deepcopy(source_cache)
        )
        identity_logits = _aggregate_logits(identity_output, variant_ids)
        arrays["scenario_logits"][0, :, qi] = identity_logits
        identity_error = float(np.max(np.abs(identity_logits - natural)))
        arrays["cached_identity_max_abs_error"][qi] = identity_error
        # A split cached-prefix execution is not numerically identical to an
        # unsplit full forward in this hybrid architecture. Prior validated
        # fixed-A runs observed up to ~0.87 A-D-logit error here. The causal
        # contrast is donor_open versus recipient_open within the same cached
        # execution path, so record this diagnostic but do not use it as an
        # exactness gate. Exact donor reproduction under a complete-cache swap
        # below remains the causal-path validation.
        if not np.isfinite(identity_error):
            raise RuntimeError("Cached identity comparison is non-finite")

        donor_cache, count = _patch_attention_kv_positions(source_cache, source_rows)
        if count != 16:
            raise RuntimeError("Selected-line transplant missed ordinary-attention layers")
        donor_output = _cached_forward(
            model, parts, suffix_ids, attention_mask, past_key_values=donor_cache
        )
        arrays["scenario_logits"][1, :, qi] = _aggregate_logits(donor_output, variant_ids)

        if validate_complete_cache:
            complete_cache, counts = _swap_cache_families(source_cache, 7)
            if counts != expected_inventory:
                raise RuntimeError("Complete-cache donor swap missed state families")
            complete_output = _cached_forward(
                model, parts, suffix_ids, attention_mask, past_key_values=complete_cache
            )
            complete_logits = _aggregate_logits(complete_output, variant_ids)
            # Exact donor reproduction must be evaluated within the same split
            # cached execution path. Comparing against the unsplit natural
            # forward confounds donor-cache correctness with the known
            # split-versus-unsplit numerical difference.
            complete_error = float(
                np.max(np.abs(complete_logits - identity_logits[DONOR_ROWS]))
            )
            arrays["complete_cache_donor_max_abs_error"][qi] = complete_error
            if complete_error > complete_cache_tolerance:
                raise RuntimeError(
                    f"Complete-cache donor error {complete_error:.6g} exceeds "
                    f"{complete_cache_tolerance}"
                )

        arrays["completed"][qi] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - question_started
        durations.append(duration)
        print(
            f"fixed-{literal} transplant {split}: {int(arrays['completed'].sum())}/{len(rows)}; "
            f"valid={int(arrays['first_decision_valid'].sum())}; seconds={duration:.2f}",
            flush=True,
        )
        if not (output_dir / "prompt_audit.json").exists():
            (output_dir / "prompt_audit.json").write_text(json.dumps({
                "pair_id": arrays["pair_ids"][qi].item(),
                "cut": cut,
                "source_positions_identical": all(values == source_rows[0] for values in source_rows),
                "suffix_identity": [token_rows[0][cut:] == token_rows[2][cut:], token_rows[1][cut:] == token_rows[3][cut:]],
                "cells": audits,
            }, indent=2) + "\n")

    metadata = {
        "experiment": "fixed-letter whole-selected-line ordinary-attention K/V transplant",
        "config": config.as_dict(),
        "cohort": str(cohort_path),
        "split": split,
        "n_pairs": len(rows),
        "scenarios": list(SCENARIOS),
        "ordinary_layers_one_based": list(ordinary_blocks),
        "complete_model_forwards_per_valid_pair": 5 if validate_complete_cache else 4,
        "complete_model_work": "one full natural, one cached prefix, recipient suffix, donor-line suffix" + (", complete-cache donor suffix" if validate_complete_cache else ""),
        "elapsed_seconds_after_load": time.monotonic() - started,
        "pair_seconds": durations,
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--validate-complete-cache", action="store_true")
    parser.add_argument("--complete-cache-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    run(
        args.config, args.cohort, args.output_dir, args.split,
        args.max_questions, args.validate_complete_cache, args.complete_cache_tolerance,
    )


if __name__ == "__main__":
    main()
