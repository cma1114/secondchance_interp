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
from .modeling import get_tokenizer, load_model_and_processor, model_input_device, render_chat, resolve_answer_tokens, tokenize_batch
from .prompts import prompt_hash
from .run_first_decision_cross_order_patching import _decision_position
from .run_fixed_a_donor_receiver_mediation import CachedQuerySourceAblator, SCENARIOS, _specs
from .run_fixed_a_full_cache_factorial import _aggregate_logits, _cache_inventory, _cached_forward
from .run_fixed_a_kv_source_transplant import _patch_attention_kv_positions
from .run_fixed_bcd_line_transplant import option_line_positions
from .run_semantic_binding_module_factorial import CONDITIONS, _forward, _messages, _remap_question


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
        "x_second_letter": np.asarray([row["x_second_letter"] for row in rows]),
        "y_second_letter": np.asarray([row["y_second_letter"] for row in rows]),
        "scenarios": np.asarray(SCENARIOS),
        "completed": np.zeros(n, dtype=bool),
        "first_decision_valid": np.zeros(n, dtype=bool),
        "first_decision_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "natural_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "scenario_logits": np.full((len(SCENARIOS), 4, n, 4), np.nan, dtype=np.float32),
        "source_position_counts": np.zeros((4, n), dtype=np.int16),
        "matching_query_counts": np.zeros((4, n), dtype=np.int16),
        "control_query_counts": np.zeros((4, n), dtype=np.int16),
        "matching_query_letters": np.full((4, n), "", dtype="<U1"),
        "control_query_letters": np.full((4, n), "", dtype="<U1"),
    }


def run(
    config_path: Path,
    cohort_path: Path,
    gate_path: Path,
    output_dir: Path,
    split: str,
    max_questions: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.attn_implementation != "sdpa" or int(config.batch_size) != 4:
        raise ValueError("Requires the canonical batch-of-four SDPA regime")
    gated = set(json.loads(gate_path.read_text())["letters"])
    if not gated:
        raise ValueError("No letters passed the discovery mediation gate")
    cohort = json.loads(cohort_path.read_text())
    rows = [
        row for row in cohort["rows"]
        if row["split"] == split and row["literal_first_answer"] in gated
    ]
    if max_questions is not None:
        rows = rows[: int(max_questions)]
    if not rows:
        raise ValueError("No gated cohort rows selected")
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _initialize(output_dir / "results.npz", rows)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    variants = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {letter: [token_id for _, token_id in variants[letter]] for letter in "ABCD"}
    layer_indices = tuple(
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if tuple(index + 1 for index in layer_indices) != tuple(range(4, 65, 4)):
        raise RuntimeError("Unexpected ordinary-attention layer inventory")
    expected_inventory = {"attention_kv": 16, "gla_conv": 48, "gla_recurrent": 48}
    started = time.monotonic()
    durations: list[float] = []

    for qi, row in enumerate(rows):
        if arrays["completed"][qi]:
            continue
        pair_started = time.monotonic()
        question = questions[row["question_id"]]
        first_x = _remap_question(question, row["first_x_new_to_original"])
        first_y = _remap_question(question, row["first_y_new_to_original"])
        second = _remap_question(question, row["second_new_to_original"])
        literal = row["literal_first_answer"]
        prompts: list[str] = []
        token_rows: list[list[int]] = []
        boundaries: list[int] = []
        source_rows: list[list[int]] = []
        matching_absolute: list[list[int]] = []
        control_absolute: list[list[int]] = []
        audits: list[dict[str, Any]] = []

        for cell, (first, condition) in enumerate(zip((first_x, first_x, first_y, first_y), CONDITIONS)):
            prompt = render_chat(
                processor, _messages(config, first, second, condition),
                config.disable_thinking, config.chat_serialization,
            )
            boundary, ids = _decision_position(tokenizer, prompt)
            first_positions, first_audit = option_line_positions(tokenizer, prompt, first, "first")
            second_positions, second_audit = option_line_positions(tokenizer, prompt, second, "last")
            donor_target = row["y_second_letter"] if cell < 2 else row["x_second_letter"]
            alternatives = [letter for letter in "ABCD" if letter != donor_target]
            control = min(
                alternatives,
                key=lambda letter: (abs(len(second_positions[letter]) - len(second_positions[donor_target])), letter),
            )
            prompts.append(prompt)
            token_rows.append(ids)
            boundaries.append(boundary)
            source_rows.append(first_positions[literal])
            matching_absolute.append(second_positions[donor_target])
            control_absolute.append(second_positions[control])
            arrays["source_position_counts"][cell, qi] = len(first_positions[literal])
            arrays["matching_query_counts"][cell, qi] = len(second_positions[donor_target])
            arrays["control_query_counts"][cell, qi] = len(second_positions[control])
            arrays["matching_query_letters"][cell, qi] = donor_target
            arrays["control_query_letters"][cell, qi] = control
            audits.append({
                "cell": cell,
                "condition": condition,
                "literal": literal,
                "source": first_audit[literal],
                "matching": second_audit[donor_target],
                "control": second_audit[control],
                "prompt_hash": prompt_hash(prompt),
            })

        if len(set(map(len, token_rows))) != 1 or len(set(boundaries)) != 1:
            raise RuntimeError("Paired mediation prompts are not token aligned")
        if any(values != source_rows[0] for values in source_rows[1:]):
            raise RuntimeError("Selected source spans are not exactly aligned")
        if source_rows[0] != list(range(row["selected_line_start"], row["selected_line_end"])):
            raise RuntimeError("Causal source span differs from screen")
        cut = boundaries[0] + 1
        if token_rows[0][cut:] != token_rows[2][cut:] or token_rows[1][cut:] != token_rows[3][cut:]:
            raise RuntimeError("X/Y suffixes differ")
        matching_local = [[position - cut for position in values] for values in matching_absolute]
        control_local = [[position - cut for position in values] for values in control_absolute]
        if any(not values or min(values) < 0 for values in matching_local + control_local):
            raise RuntimeError("Receiver query is outside the cached suffix")

        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        device = model_input_device(parts)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        arrays["natural_logits"][:, qi] = _aggregate_logits(
            _forward(model, parts, input_ids, attention_mask), variant_ids
        )
        prefix = _cached_forward(model, parts, input_ids[:, :cut], attention_mask[:, :cut])
        source_cache = prefix.past_key_values
        arrays["first_decision_logits"][:, qi] = _aggregate_logits(prefix, variant_ids)
        valid = bool(np.all(
            arrays["first_decision_logits"][:, qi].argmax(axis=-1) == "ABCD".index(literal)
        ))
        arrays["first_decision_valid"][qi] = valid
        if _cache_inventory(source_cache) != expected_inventory:
            raise RuntimeError("Unexpected hybrid-cache inventory")
        if not valid:
            arrays["completed"][qi] = True
            atomic_save_npz(output_dir / "results.npz", **arrays)
            continue

        suffix_ids = input_ids[:, cut:]
        donor_cache, count = _patch_attention_kv_positions(source_cache, source_rows)
        if count != 16:
            raise RuntimeError("Donor transplant missed ordinary-attention layers")
        matching_specs = _specs(layer_indices, source_rows, matching_local)
        control_specs = _specs(layer_indices, source_rows, control_local)
        definitions = (
            (source_cache, None),
            (donor_cache, None),
            (source_cache, matching_specs),
            (donor_cache, matching_specs),
            (source_cache, control_specs),
            (donor_cache, control_specs),
        )
        for scenario, (base_cache, specs) in enumerate(definitions):
            cache = copy.deepcopy(base_cache)
            if specs is None:
                output = _cached_forward(model, parts, suffix_ids, attention_mask, past_key_values=cache)
            else:
                with CachedQuerySourceAblator(parts, specs, cut):
                    output = _cached_forward(model, parts, suffix_ids, attention_mask, past_key_values=cache)
            arrays["scenario_logits"][scenario, :, qi] = _aggregate_logits(output, variant_ids)
            del cache, output
        arrays["completed"][qi] = True
        atomic_save_npz(output_dir / "results.npz", **arrays)
        duration = time.monotonic() - pair_started
        durations.append(duration)
        print(
            f"fixed-{literal} mediation {split}: {int(arrays['completed'].sum())}/{len(rows)}; "
            f"valid={int(arrays['first_decision_valid'].sum())}; seconds={duration:.2f}",
            flush=True,
        )
        if not (output_dir / "prompt_audit.json").exists():
            (output_dir / "prompt_audit.json").write_text(json.dumps({
                "pair_id": arrays["pair_ids"][qi].item(), "cut": cut, "cells": audits,
            }, indent=2) + "\n")

    metadata = {
        "experiment": "fixed-letter selected-line donor-to-matching-repeat mediation",
        "config": config.as_dict(),
        "cohort": str(cohort_path),
        "gate": str(gate_path),
        "gated_letters": sorted(gated),
        "split": split,
        "n_pairs": len(rows),
        "scenarios": list(SCENARIOS),
        "ordinary_layers_one_based": [index + 1 for index in layer_indices],
        "complete_model_forwards_per_valid_pair": 8,
        "complete_model_work": "one full natural, one cached prefix, six cached suffix scenarios",
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
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    run(args.config, args.cohort, args.gate, args.output_dir, args.split, args.max_questions)


if __name__ == "__main__":
    main()
