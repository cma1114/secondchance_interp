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
from .prompts import prompt_hash
from .run_decision_boundary_crossover import (
    BoundaryTrajectoryCollector,
    _run_boundary,
)
from .run_first_decision_cross_order_patching import _decision_position
from .run_fixed_a_full_cache_factorial import (
    DONOR_ROWS,
    _aggregate_logits,
    _cache_inventory,
    _cached_forward,
    _swap_cache_families,
)
from .run_fixed_a_kv_source_transplant import _patch_attention_kv_positions
from .run_fixed_bcd_line_transplant import option_line_positions
from .run_semantic_binding_module_factorial import CONDITIONS, _messages, _remap_question


LETTERS = "ABCD"
CURRENT_STRATA = ("low", "high")
SCENARIOS = (
    "identity_cached",
    "d_line_kv",
    "identity_trajectory",
    "d_closing_trajectory",
    "full_history",
)
EXPECTED_INVENTORY = {"attention_kv": 16, "gla_conv": 48, "gla_recurrent": 48}
N_LAYERS = 64


def _initialize(path: Path, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    qids = [row["question_id"] for row in rows]
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses another question order")
        return arrays
    n = len(rows)
    return {
        "question_ids": np.asarray(qids),
        "semantic_targets": np.asarray([row["semantic_target"] for row in rows]),
        "current_strata": np.asarray(CURRENT_STRATA),
        "scenarios": np.asarray(SCENARIOS),
        "completed": np.zeros(n, dtype=bool),
        "exact_eligible": np.zeros(n, dtype=bool),
        "screen_old_score_gap": np.asarray([row["old_score_gap"] for row in rows], dtype=np.float32),
        "screen_old_low_rank": np.asarray([row["old_low_rank"] for row in rows], dtype=np.int8),
        "screen_old_high_rank": np.asarray([row["old_high_rank"] for row in rows], dtype=np.int8),
        "winner_crossing": np.asarray([row["winner_crossing"] for row in rows], dtype=bool),
        "first_decision_logits": np.full((len(SCENARIOS), 4, n, 4), np.nan, dtype=np.float32),
        "final_logits": np.full((len(CURRENT_STRATA), len(SCENARIOS), 4, n, 4), np.nan, dtype=np.float32),
        "actual_old_score_gap": np.full(n, np.nan, dtype=np.float32),
        "d_line_position_count": np.zeros(n, dtype=np.int16),
        "ordinary_layer_count": np.zeros(n, dtype=np.int16),
        "d_closing_trajectory_dose": np.full((n, N_LAYERS), np.nan, dtype=np.float32),
        "trajectory_identity_decision_max_error": np.full(n, np.nan, dtype=np.float32),
        "trajectory_identity_final_max_error": np.full((len(CURRENT_STRATA), n), np.nan, dtype=np.float32),
        "full_history_decision_max_error": np.full(n, np.nan, dtype=np.float32),
        "full_history_final_max_error": np.full((len(CURRENT_STRATA), n), np.nan, dtype=np.float32),
        "model_calls": np.zeros(n, dtype=np.int16),
        "duration_seconds": np.full(n, np.nan, dtype=np.float32),
    }


def _scenario_to_decision_and_final(
    model: Any,
    parts: Any,
    d_cache: Any,
    batches: list[dict[str, Any]],
    d_cut: int,
    boundary_cut: int,
    variant_ids: dict[str, list[int]],
) -> tuple[np.ndarray, np.ndarray, int]:
    # Prefixes are identical across current-score strata through the first
    # decision, so compute that continuation once and fork only afterwards.
    reference = batches[0]
    decision = _cached_forward(
        model,
        parts,
        reference["input_ids"][:, d_cut:boundary_cut],
        reference["attention_mask"][:, :boundary_cut],
        past_key_values=copy.deepcopy(d_cache),
    )
    decision_logits = _aggregate_logits(decision, variant_ids)
    final_logits = np.full((len(batches), 4, 4), np.nan, dtype=np.float32)
    calls = 1
    for current_index, batch in enumerate(batches):
        final = _cached_forward(
            model,
            parts,
            batch["input_ids"][:, boundary_cut:],
            batch["attention_mask"],
            past_key_values=copy.deepcopy(decision.past_key_values),
        )
        final_logits[current_index] = _aggregate_logits(final, variant_ids)
        calls += 1
        del final
    del decision
    return decision_logits, final_logits, calls


def run(
    config_path: Path,
    cohort_path: Path,
    output_dir: Path,
    split: str,
    max_questions: int | None,
    full_cache_tolerance: float,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if (
        config.prompt_mode != "baseline_matched_empty_history"
        or config.feedback_variant != "token_matched_test"
        or config.chat_serialization != "raw_qwen_chatml"
        or config.attn_implementation != "sdpa"
        or int(config.batch_size) != 4
    ):
        raise ValueError("Requires the exact canonical empty-history batch-four SDPA regime")
    cohort = json.loads(cohort_path.read_text())
    rows = [row for row in cohort["rows"] if row["split"] == split]
    if max_questions is not None:
        rows = rows[: int(max_questions)]
    if not rows:
        raise ValueError(f"No rows for split {split}")
    manifest = json.loads(Path(config.manifest_path).read_text())["questions"]
    questions = {row["id"]: row for row in manifest}
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, rows)

    model, processor, parts = load_model_and_processor(config)
    if len(parts.layers) != N_LAYERS:
        raise RuntimeError(f"Expected {N_LAYERS} layers")
    ordinary_layers = tuple(
        index + 1 for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if ordinary_layers != tuple(range(4, 65, 4)):
        raise RuntimeError(f"Unexpected ordinary-attention layers {ordinary_layers}")
    tokenizer = get_tokenizer(processor)
    variants = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {letter: [token_id for _, token_id in variants[letter]] for letter in LETTERS}
    device = model_input_device(parts)
    started = time.monotonic()

    for qi, row in enumerate(rows):
        if arrays["completed"][qi]:
            continue
        question_started = time.monotonic()
        question = questions[row["question_id"]]
        first_high = _remap_question(question, row["old_high_new_to_original"])
        first_low = _remap_question(question, row["old_low_new_to_original"])
        target = row["semantic_target"]
        firsts = (first_high, first_high, first_low, first_low)
        batches: list[dict[str, Any]] = []
        audit_batches: list[dict[str, Any]] = []

        for current_name in CURRENT_STRATA:
            second = _remap_question(question, row[f"current_{current_name}_new_to_original"])
            prompts = [
                render_chat(
                    processor,
                    _messages(config, first, second, condition),
                    config.disable_thinking,
                    config.chat_serialization,
                )
                for first, condition in zip(firsts, CONDITIONS)
            ]
            token_rows = [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
            boundaries = [_decision_position(tokenizer, prompt)[0] for prompt in prompts]
            position_rows = []
            position_audits = []
            for prompt, first in zip(prompts, firsts):
                positions, position_audit = option_line_positions(tokenizer, prompt, first, "first")
                position_rows.append(positions["D"])
                position_audits.append(position_audit["D"])
            if len(set(map(len, token_rows))) != 1 or len(set(boundaries)) != 1:
                raise RuntimeError("Paired prompts are not token aligned")
            if any(values != position_rows[0] for values in position_rows[1:]):
                raise RuntimeError("D lines occupy different token positions")
            if position_rows[0] != list(range(row["d_line_start"], row["d_line_end"])):
                raise RuntimeError("D-line span differs from the frozen screen")
            source_ids = [tuple(ids[index] for index in position_rows[cell]) for cell, ids in enumerate(token_rows)]
            if len(set(source_ids)) != 1:
                raise RuntimeError("The paired D lines are not token-identical")
            boundary_cut = boundaries[0] + 1
            if token_rows[0][boundary_cut:] != token_rows[2][boundary_cut:]:
                raise RuntimeError("Game suffix differs after the first-decision boundary")
            if token_rows[1][boundary_cut:] != token_rows[3][boundary_cut:]:
                raise RuntimeError("Neutral suffix differs after the first-decision boundary")
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
            batches.append(
                {
                    "name": current_name,
                    "prompts": prompts,
                    "token_rows": token_rows,
                    "input_ids": input_ids.to(device),
                    "attention_mask": attention_mask.to(device),
                    "boundary_cut": boundary_cut,
                    "d_positions": position_rows,
                }
            )
            audit_batches.append(
                {
                    "current": current_name,
                    "prompt_hashes": [prompt_hash(prompt) for prompt in prompts],
                    "d_line": position_audits,
                    "boundary_cut": boundary_cut,
                }
            )

        d_positions = batches[0]["d_positions"]
        d_cut = d_positions[0][-1] + 1
        boundary_cut = batches[0]["boundary_cut"]
        if batches[1]["boundary_cut"] != boundary_cut:
            raise RuntimeError("Current-score strata have different first-decision positions")
        if batches[0]["token_rows"][0][:boundary_cut] != batches[1]["token_rows"][0][:boundary_cut]:
            raise RuntimeError("Current-score strata differ before the first decision")
        arrays["d_line_position_count"][qi] = len(d_positions[0])
        calls = 0

        with BoundaryTrajectoryCollector(parts) as collector:
            inclusive = _cached_forward(
                model,
                parts,
                batches[0]["input_ids"][:, :d_cut],
                batches[0]["attention_mask"][:, :d_cut],
            )
        calls += 1
        trajectory = collector.values
        final_d_output = collector.final_output
        if final_d_output is None or set(trajectory) != set(range(N_LAYERS)):
            raise RuntimeError("Incomplete D-closing-token trajectory")
        d_cache = inclusive.past_key_values
        if _cache_inventory(d_cache) != EXPECTED_INVENTORY:
            raise RuntimeError("Unexpected inclusive D cache inventory")

        pre_d = _cached_forward(
            model,
            parts,
            batches[0]["input_ids"][:, : d_cut - 1],
            batches[0]["attention_mask"][:, : d_cut - 1],
        )
        calls += 1
        pre_d_cache = pre_d.past_key_values
        del pre_d

        decision_logits, final_logits, used = _scenario_to_decision_and_final(
            model, parts, d_cache, batches, d_cut, boundary_cut, variant_ids
        )
        calls += used
        arrays["first_decision_logits"][0, :, qi] = decision_logits
        arrays["final_logits"][:, 0, :, qi] = final_logits
        target_index = LETTERS.index(target)
        centered = decision_logits - decision_logits.mean(axis=-1, keepdims=True)
        high_score = float(centered[:2, target_index].mean())
        low_score = float(centered[2:, target_index].mean())
        arrays["actual_old_score_gap"][qi] = high_score - low_score
        exact = bool(np.isfinite(centered).all() and high_score > low_score)
        arrays["exact_eligible"][qi] = exact
        if not exact:
            arrays["model_calls"][qi] = calls
            arrays["duration_seconds"][qi] = time.monotonic() - question_started
            arrays["completed"][qi] = True
            atomic_save_npz(result_path, **arrays)
            print(f"D-line score {split}: screened {row['question_id']} (score order failed)", flush=True)
            del inclusive, d_cache, pre_d_cache
            torch.cuda.empty_cache()
            continue

        kv_cache, count = _patch_attention_kv_positions(d_cache, d_positions)
        if count != len(ordinary_layers):
            raise RuntimeError("D-line K/V transplant missed ordinary-attention layers")
        arrays["ordinary_layer_count"][qi] = count
        decision_logits, final_logits, used = _scenario_to_decision_and_final(
            model, parts, kv_cache, batches, d_cut, boundary_cut, variant_ids
        )
        calls += used
        arrays["first_decision_logits"][1, :, qi] = decision_logits
        arrays["final_logits"][:, 1, :, qi] = final_logits
        del kv_cache

        identity_d, identity_dose = _run_boundary(
            model,
            parts,
            batches[0]["input_ids"][:, d_cut - 1 : d_cut],
            batches[0]["attention_mask"][:, :d_cut],
            copy.deepcopy(pre_d_cache),
            trajectory,
            final_d_output,
            np.arange(4, dtype=np.int64),
        )
        calls += 1
        decision_logits, final_logits, used = _scenario_to_decision_and_final(
            model, parts, identity_d.past_key_values, batches, d_cut, boundary_cut, variant_ids
        )
        calls += used
        arrays["first_decision_logits"][2, :, qi] = decision_logits
        arrays["final_logits"][:, 2, :, qi] = final_logits
        arrays["trajectory_identity_decision_max_error"][qi] = float(
            np.max(np.abs(arrays["first_decision_logits"][2, :, qi] - arrays["first_decision_logits"][0, :, qi]))
        )
        for current_index in range(len(CURRENT_STRATA)):
            arrays["trajectory_identity_final_max_error"][current_index, qi] = float(
                np.max(np.abs(arrays["final_logits"][current_index, 2, :, qi] - arrays["final_logits"][current_index, 0, :, qi]))
            )
        del identity_d

        crossed_d, dose = _run_boundary(
            model,
            parts,
            batches[0]["input_ids"][:, d_cut - 1 : d_cut],
            batches[0]["attention_mask"][:, :d_cut],
            copy.deepcopy(pre_d_cache),
            trajectory,
            final_d_output,
            DONOR_ROWS,
        )
        calls += 1
        arrays["d_closing_trajectory_dose"][qi] = dose.mean(axis=1)
        decision_logits, final_logits, used = _scenario_to_decision_and_final(
            model, parts, crossed_d.past_key_values, batches, d_cut, boundary_cut, variant_ids
        )
        calls += used
        arrays["first_decision_logits"][3, :, qi] = decision_logits
        arrays["final_logits"][:, 3, :, qi] = final_logits
        del crossed_d

        donor_pre_d, counts = _swap_cache_families(pre_d_cache, 7, donor_rows=DONOR_ROWS)
        if counts != EXPECTED_INVENTORY:
            raise RuntimeError("Complete pre-D donor swap missed a cache family")
        full_d, _ = _run_boundary(
            model,
            parts,
            batches[0]["input_ids"][:, d_cut - 1 : d_cut],
            batches[0]["attention_mask"][:, :d_cut],
            donor_pre_d,
            trajectory,
            final_d_output,
            DONOR_ROWS,
        )
        calls += 1
        decision_logits, final_logits, used = _scenario_to_decision_and_final(
            model, parts, full_d.past_key_values, batches, d_cut, boundary_cut, variant_ids
        )
        calls += used
        arrays["first_decision_logits"][4, :, qi] = decision_logits
        arrays["final_logits"][:, 4, :, qi] = final_logits
        del full_d

        decision_error = float(
            np.max(np.abs(arrays["first_decision_logits"][4, :, qi] - arrays["first_decision_logits"][2, DONOR_ROWS, qi]))
        )
        arrays["full_history_decision_max_error"][qi] = decision_error
        for current_index in range(len(CURRENT_STRATA)):
            arrays["full_history_final_max_error"][current_index, qi] = float(
                np.max(
                    np.abs(
                        arrays["final_logits"][current_index, 4, :, qi]
                        - arrays["final_logits"][current_index, 2, DONOR_ROWS, qi]
                    )
                )
            )
        if decision_error > full_cache_tolerance or np.nanmax(arrays["full_history_final_max_error"][:, qi]) > full_cache_tolerance:
            raise RuntimeError("Complete-history positive control failed exact donor reproduction")

        arrays["model_calls"][qi] = calls
        arrays["duration_seconds"][qi] = time.monotonic() - question_started
        arrays["completed"][qi] = True
        atomic_save_npz(result_path, **arrays)
        if not (output_dir / "prompt_audit.json").exists():
            closing = d_positions[0][-1]
            (output_dir / "prompt_audit.json").write_text(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "semantic_target": target,
                        "d_cut": d_cut,
                        "d_closing_position": closing,
                        "d_closing_token": tokenizer.convert_ids_to_tokens([batches[0]["token_rows"][0][closing]])[0],
                        "d_line_tokens_identical": True,
                        "ordinary_layers_one_based": list(ordinary_layers),
                        "batches": audit_batches,
                    },
                    indent=2,
                )
                + "\n"
            )
        duration = arrays["duration_seconds"][qi]
        print(
            f"D-line score {split}: {int(arrays['completed'].sum())}/{len(rows)}; "
            f"eligible={int(arrays['exact_eligible'].sum())}; calls={calls}; seconds={duration:.2f}",
            flush=True,
        )
        del inclusive, d_cache, pre_d_cache
        torch.cuda.empty_cache()

    metadata = {
        "experiment": "same-semantic D-line old-score K/V transfer and D-closing-state crossover",
        "config": config.as_dict(),
        "config_path": str(config_path),
        "cohort_path": str(cohort_path),
        "split": split,
        "n_rows": len(rows),
        "current_strata": list(CURRENT_STRATA),
        "scenarios": list(SCENARIOS),
        "cells": ["game_old_high", "neutral_old_high", "game_old_low", "neutral_old_low"],
        "ordinary_layers_one_based": list(ordinary_layers),
        "complete_model_calls_per_eligible_question": 20,
        "complete_model_work": (
            "inclusive D-closing prefix capture; pre-closing prefix; cached identity and all-layer "
            "D-line K/V paths; identity and crossed D-closing-trajectory paths; complete-history "
            "path; each of five paths continued under low- and high-current second presentations"
        ),
        "full_cache_tolerance": full_cache_tolerance,
        "elapsed_seconds_after_load": time.monotonic() - started,
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
    parser.add_argument("--full-cache-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    run(args.config, args.cohort, args.output_dir, args.split, args.max_questions, args.full_cache_tolerance)


if __name__ == "__main__":
    main()
