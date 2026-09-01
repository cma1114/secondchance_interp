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
from .run_semantic_binding_module_factorial import (
    CELLS,
    CONDITIONS,
    _aggregate_logits,
    _forward,
    _messages,
    _remap_question,
)


FAMILIES = ("attention_kv", "gla_conv", "gla_recurrent")
FACTORIAL_MASKS = tuple(range(8))
DONOR_ROWS = np.asarray([2, 3, 0, 1], dtype=np.int64)


def _initialize(path: Path, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    qids = [row["question_id"] for row in rows]
    if path.exists():
        arrays = dict(np.load(path, allow_pickle=False))
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing output uses a different question order")
        if arrays["factorial_masks"].astype(int).tolist() != list(FACTORIAL_MASKS):
            raise ValueError("Existing output uses different factorial masks")
        # Older checkpoints predate exact-regime eligibility recording. Recover
        # it for completed rows; incomplete rows are screened when resumed.
        if "first_decision_valid" not in arrays:
            valid = np.zeros(len(qids), dtype=bool)
            completed = arrays["completed"].astype(bool)
            if np.any(completed):
                valid[completed] = np.all(
                    arrays["first_decision_logits"][:, completed].argmax(axis=-1) == 0,
                    axis=0,
                )
            arrays["first_decision_valid"] = valid
        return arrays
    n = len(rows)
    return {
        "question_ids": np.asarray(qids),
        "x_second_letter": np.asarray([row["x_second_letter"] for row in rows]),
        "y_second_letter": np.asarray([row["y_second_letter"] for row in rows]),
        "factorial_masks": np.asarray(FACTORIAL_MASKS, dtype=np.int8),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "first_decision_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "first_decision_valid": np.zeros(n, dtype=bool),
        "cached_identity_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "factorial_logits": np.full((8, 4, n, 4), np.nan, dtype=np.float32),
        "full_cache_donor_max_abs_error": np.full(n, np.nan, dtype=np.float32),
    }


def _cached_forward(
    model: Any,
    parts: Any,
    input_ids: Any,
    attention_mask: Any,
    past_key_values: Any | None = None,
) -> Any:
    import torch

    kwargs = {
        "input_ids": input_ids.to(model_input_device(parts)),
        "attention_mask": attention_mask.to(model_input_device(parts)),
        "past_key_values": past_key_values,
        "use_cache": True,
        "return_dict": True,
    }
    with torch.inference_mode():
        try:
            return model(**kwargs, logits_to_keep=1)
        except TypeError:
            return model(**kwargs)


def _initialized_tensor(value: Any) -> bool:
    return value is not None and getattr(value, "numel", lambda: 0)() > 0


def _swap_cache_families(
    source_cache: Any,
    family_mask: int,
    donor_rows: np.ndarray = DONOR_ROWS,
) -> tuple[Any, dict[str, int]]:
    """Clone a recipient cache and replace selected state families by donor rows."""
    import torch

    target = copy.deepcopy(source_cache)
    donor = torch.as_tensor(donor_rows, dtype=torch.long)
    counts = {family: 0 for family in FAMILIES}

    for source_layer, target_layer in zip(source_cache.layers, target.layers):
        if family_mask & 1:
            keys = getattr(source_layer, "keys", None)
            values = getattr(source_layer, "values", None)
            if _initialized_tensor(keys) or _initialized_tensor(values):
                if not (_initialized_tensor(keys) and _initialized_tensor(values)):
                    raise RuntimeError("Only one conventional-attention cache tensor is initialized")
                target_layer.keys = keys.index_select(0, donor.to(keys.device)).clone()
                target_layer.values = values.index_select(0, donor.to(values.device)).clone()
                counts["attention_kv"] += 1

        if family_mask & 2 and hasattr(source_layer, "conv_states"):
            for state_index, state in source_layer.conv_states.items():
                initialized = bool(source_layer.is_conv_states_initialized[state_index])
                if initialized:
                    if state is None:
                        raise RuntimeError("Initialized GLA convolution state is absent")
                    target_layer.conv_states[state_index] = state.index_select(
                        0, donor.to(state.device)
                    ).clone()
                    counts["gla_conv"] += 1

        if family_mask & 4 and hasattr(source_layer, "recurrent_states"):
            for state_index, state in source_layer.recurrent_states.items():
                initialized = bool(source_layer.is_recurrent_states_initialized[state_index])
                if initialized:
                    if state is None:
                        raise RuntimeError("Initialized GLA recurrent state is absent")
                    target_layer.recurrent_states[state_index] = state.index_select(
                        0, donor.to(state.device)
                    ).clone()
                    counts["gla_recurrent"] += 1
    return target, counts


def _cache_inventory(cache: Any) -> dict[str, int]:
    counts = {family: 0 for family in FAMILIES}
    for layer in cache.layers:
        if _initialized_tensor(getattr(layer, "keys", None)):
            counts["attention_kv"] += 1
        if hasattr(layer, "conv_states"):
            counts["gla_conv"] += sum(
                bool(layer.is_conv_states_initialized[i]) for i in layer.conv_states
            )
        if hasattr(layer, "recurrent_states"):
            counts["gla_recurrent"] += sum(
                bool(layer.is_recurrent_states_initialized[i]) for i in layer.recurrent_states
            )
    return counts


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
    rows = [row for index, row in enumerate(rows) if index % num_shards == shard_index]
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
        "experiment": "fixed-A complete causal-cache factorial transplant",
        "config": config.as_dict(),
        "config_path": str(config_path),
        "cohort_path": str(cohort_path),
        "split": split,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "n_questions": len(rows),
        "cells": list(CELLS),
        "families": list(FAMILIES),
        "factorial_masks": {
            str(mask): [FAMILIES[i] for i in range(3) if mask & (1 << i)]
            for mask in FACTORIAL_MASKS
        },
        "complete_model_work_per_question": (
            "one unsplit natural pass; one cached prefix pass; eight cached suffix "
            "passes (identity plus seven family combinations)"
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
        for first, condition in zip((first_x, first_x, first_y, first_y), CONDITIONS):
            prompt = render_chat(
                processor,
                _messages(config, first, second, condition),
                config.disable_thinking,
                config.chat_serialization,
            )
            boundary, ids = _decision_position(tokenizer, prompt)
            prompts.append(prompt)
            token_rows.append(ids)
            boundary_positions.append(boundary)

        lengths = [len(ids) for ids in token_rows]
        if len(set(lengths)) != 1 or len(set(boundary_positions)) != 1:
            raise RuntimeError("Fixed-A prompts are not token-aligned")
        cut = boundary_positions[0] + 1
        if token_rows[0][cut:] != token_rows[2][cut:]:
            raise RuntimeError("Evaluation X/Y suffixes differ after the transplant boundary")
        if token_rows[1][cut:] != token_rows[3][cut:]:
            raise RuntimeError("Neutral X/Y suffixes differ after the transplant boundary")

        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        device = model_input_device(parts)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        natural_output = _forward(model, parts, input_ids, attention_mask)
        arrays["natural_logits"][:, qi] = _aggregate_logits(natural_output, variant_ids)

        prefix_output = _cached_forward(
            model,
            parts,
            input_ids[:, :cut],
            attention_mask[:, :cut],
        )
        source_cache = prefix_output.past_key_values
        arrays["first_decision_logits"][:, qi] = _aggregate_logits(
            prefix_output, variant_ids
        )
        first_decision_valid = bool(
            np.all(arrays["first_decision_logits"][:, qi].argmax(axis=-1) == 0)
        )
        arrays["first_decision_valid"][qi] = first_decision_valid
        inventory = _cache_inventory(source_cache)
        if inventory != expected_inventory:
            raise RuntimeError(
                f"Unexpected cache inventory {inventory}; expected {expected_inventory}"
            )

        # The historical cohort was selected from separately batched Baseline
        # runs. Qwen is numerically batch-sensitive, so the fixed-A eligibility
        # condition must be rechecked under this experiment's exact four-cell
        # execution regime. This screen uses only the pre-feedback decision and
        # therefore cannot select on the intervention outcome.
        if not first_decision_valid:
            arrays["completed"][qi] = True
            atomic_save_npz(result_path, **arrays)
            answers = "".join(
                "ABCD"[i]
                for i in arrays["first_decision_logits"][:, qi].argmax(axis=-1)
            )
            print(
                f"fixed-A full-cache factorial {split} shard {shard_index}: "
                f"screened out {row['question_id']} (first decisions {answers})",
                flush=True,
            )
            continue

        suffix_ids = input_ids[:, cut:]
        for mask in FACTORIAL_MASKS:
            cache, counts = _swap_cache_families(source_cache, mask)
            expected_counts = {
                family: expected_inventory[family] if mask & (1 << i) else 0
                for i, family in enumerate(FAMILIES)
            }
            if counts != expected_counts:
                raise RuntimeError(
                    f"Mask {mask} swapped {counts}; expected {expected_counts}"
                )
            output = _cached_forward(
                model,
                parts,
                suffix_ids,
                attention_mask,
                past_key_values=cache,
            )
            arrays["factorial_logits"][mask, :, qi] = _aggregate_logits(
                output, variant_ids
            )
            del cache, output

        arrays["cached_identity_logits"][:, qi] = arrays["factorial_logits"][0, :, qi]
        full = arrays["factorial_logits"][7, :, qi]
        donor_identity = arrays["cached_identity_logits"][DONOR_ROWS, qi]
        full_error = float(np.max(np.abs(full - donor_identity)))
        arrays["full_cache_donor_max_abs_error"][qi] = full_error
        if full_error > full_cache_tolerance:
            raise RuntimeError(
                f"Full-cache positive control failed on {row['question_id']}: "
                f"max A-D error {full_error:.6g} > {full_cache_tolerance:.6g}"
            )

        arrays["completed"][qi] = True
        atomic_save_npz(result_path, **arrays)
        done = int(arrays["completed"].sum())
        if done == 1 or done % 5 == 0 or done == len(rows):
            print(
                f"fixed-A full-cache factorial {split} shard {shard_index}: "
                f"{done}/{len(rows)}; full-cache error={full_error:.3g}",
                flush=True,
            )

        if qi == 0 and not (output_dir / "prompt_audit.json").exists():
            (output_dir / "prompt_audit.json").write_text(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "literal_first_answer": row["literal_first_answer"],
                        "boundary_positions": boundary_positions,
                        "boundary_tokens": tokenizer.convert_ids_to_tokens(
                            [ids[pos] for ids, pos in zip(token_rows, boundary_positions)]
                        ),
                        "cut": cut,
                        "cache_inventory": inventory,
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
