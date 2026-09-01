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
from .run_first_decision_cross_order_patching import _decision_position
from .run_fixed_a_full_cache_factorial import (
    _aggregate_logits,
    _cache_inventory,
    _cached_forward,
    _swap_cache_families,
)
from .run_semantic_binding_module_factorial import _forward, _messages, _remap_question


LETTERS = "ABCD"
CELLS = (
    "evaluation_donor",
    "neutral_donor",
    "evaluation_recipient",
    "neutral_recipient",
)
CONDITIONS = (
    "incorrect_again",
    "lost_again",
    "incorrect_again",
    "lost_again",
)
DONOR_ROWS = np.asarray([0, 1, 0, 1], dtype=np.int64)
EXPECTED_INVENTORY = {
    "attention_kv": 16,
    "gla_conv": 48,
    "gla_recurrent": 48,
}


def _initialize(path: Path, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    qids = [row["question_id"] for row in rows]
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing output uses a different question order")
        return arrays
    n = len(rows)
    return {
        "question_ids": np.asarray(qids),
        "target_second_letter": np.asarray(
            [row["target_second_letter"] for row in rows]
        ),
        "frozen_recipient_winner_second_letter": np.asarray(
            [row["screen_recipient_winner_second_letter"] for row in rows]
        ),
        "exact_recipient_winner_first_letter": np.full(n, "", dtype="<U1"),
        "exact_recipient_winner_original_content": np.full(n, "", dtype="<U1"),
        "exact_recipient_winner_second_letter": np.full(n, "", dtype="<U1"),
        "exact_eligible": np.zeros(n, dtype=bool),
        "completed": np.zeros(n, dtype=bool),
        "first_decision_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "natural_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "cached_identity_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "boundary_kv_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "complete_cache_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "boundary_kv_delta_l2": np.full(n, np.nan, dtype=np.float32),
        "boundary_kv_delta_relative": np.full(n, np.nan, dtype=np.float32),
        "boundary_donor_control_max_abs_error": np.full(
            n, np.nan, dtype=np.float32
        ),
        "complete_cache_donor_max_abs_error": np.full(
            n, np.nan, dtype=np.float32
        ),
    }


def _patch_boundary_attention_kv(
    source_cache: Any,
    boundary: int,
) -> tuple[Any, int, float, float]:
    """Replace donor K/V only at the first-decision boundary in recipient rows."""
    target = copy.deepcopy(source_cache)
    layer_count = 0
    delta_sq = 0.0
    recipient_sq = 0.0
    for source_layer, target_layer in zip(source_cache.layers, target.layers):
        keys = getattr(source_layer, "keys", None)
        values = getattr(source_layer, "values", None)
        if keys is None and values is None:
            continue
        if keys is None or values is None or keys.numel() == 0 or values.numel() == 0:
            raise RuntimeError("Conventional-attention K/V initialization mismatch")
        if keys.shape[0] != 4 or values.shape[0] != 4:
            raise RuntimeError("Expected four cache rows")
        if boundary >= keys.shape[-2] or boundary >= values.shape[-2]:
            raise RuntimeError("Boundary lies outside the K/V sequence")
        for recipient, donor in ((2, 0), (3, 1)):
            donor_key = keys[donor, :, boundary, :]
            recipient_key = keys[recipient, :, boundary, :]
            donor_value = values[donor, :, boundary, :]
            recipient_value = values[recipient, :, boundary, :]
            delta_sq += float(
                (donor_key.float() - recipient_key.float()).square().sum()
                + (donor_value.float() - recipient_value.float()).square().sum()
            )
            recipient_sq += float(
                recipient_key.float().square().sum()
                + recipient_value.float().square().sum()
            )
            target_layer.keys[recipient, :, boundary, :] = donor_key
            target_layer.values[recipient, :, boundary, :] = donor_value
        layer_count += 1
    delta_l2 = float(np.sqrt(delta_sq))
    relative = float(delta_l2 / max(np.sqrt(recipient_sq), 1e-12))
    return target, layer_count, delta_l2, relative


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
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires the action-matched incorrect/lost feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw Qwen ChatML")
    if config.attn_implementation != "sdpa" or int(config.batch_size) != 4:
        raise ValueError("Requires the established batch-four SDPA regime")
    if split not in {"discovery", "confirmation"}:
        raise ValueError("Unknown split")

    cohort = json.loads(cohort_path.read_text())
    rows = [row for row in cohort["rows"] if row["split"] == split]
    if not rows:
        raise ValueError("No selectedness pairs for this split")
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, rows)

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    started_after_load = time.monotonic()
    question_durations: list[float] = []
    audit_path = output_dir / "prompt_audit.json"

    metadata: dict[str, Any] = {
        "experiment": "W1=A first-decision selectedness ordinary-attention K/V transplant",
        "config": config.as_dict(),
        "config_path": str(config_path),
        "cohort_path": str(cohort_path),
        "split": split,
        "n_frozen_questions": len(rows),
        "cells": list(CELLS),
        "donor_rows": DONOR_ROWS.tolist(),
        "intervention": (
            "Replace only conventional-attention K/V at the final empty first-answer "
            "decision-boundary token in recipient rows, across all 16 ordinary-"
            "attention blocks."
        ),
        "complete_model_forwards_per_exact_eligible_question": 5,
        "complete_model_work": (
            "one cached prefix eligibility pass; one unsplit natural pass; cached "
            "identity, boundary-K/V, and complete-cache suffix passes"
        ),
        "full_cache_tolerance": full_cache_tolerance,
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

    for qi, row in enumerate(rows):
        # Keep the checkpoint indexed by the complete frozen split.  The limit
        # controls work performed in this invocation only, so increasing it can
        # safely resume the same checkpoint after a benchmark.
        if max_questions is not None and qi >= int(max_questions):
            break
        if arrays["completed"][qi]:
            continue
        question_started = time.monotonic()
        question = questions[row["question_id"]]
        donor_first = _remap_question(
            question, row["donor_first_new_to_original"]
        )
        recipient_first = _remap_question(
            question, row["recipient_first_new_to_original"]
        )
        second = _remap_question(question, row["second_new_to_original"])

        prompts: list[str] = []
        token_rows: list[list[int]] = []
        boundaries: list[int] = []
        for first, condition in zip(
            (donor_first, donor_first, recipient_first, recipient_first), CONDITIONS
        ):
            prompt = render_chat(
                processor,
                _messages(config, first, second, condition),
                config.disable_thinking,
                config.chat_serialization,
            )
            boundary, ids = _decision_position(tokenizer, prompt)
            prompts.append(prompt)
            token_rows.append(ids)
            boundaries.append(boundary)

        if len({len(ids) for ids in token_rows}) != 1:
            raise RuntimeError("Donor and recipient prompts are not token-aligned")
        if len(set(boundaries)) != 1:
            raise RuntimeError("First-decision boundary positions differ")
        cut = boundaries[0] + 1
        if token_rows[0][cut:] != token_rows[2][cut:]:
            raise RuntimeError("Evaluation donor/recipient suffixes differ")
        if token_rows[1][cut:] != token_rows[3][cut:]:
            raise RuntimeError("Neutral donor/recipient suffixes differ")

        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        if input_ids.shape[1] != len(token_rows[0]):
            raise RuntimeError("Unexpected left padding in token-aligned four-row batch")
        device = model_input_device(parts)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        prefix_output = _cached_forward(
            model, parts, input_ids[:, :cut], attention_mask[:, :cut]
        )
        source_cache = prefix_output.past_key_values
        prefix_logits = _aggregate_logits(prefix_output, variant_ids)
        arrays["first_decision_logits"][:, qi] = prefix_logits
        inventory = _cache_inventory(source_cache)
        if inventory != EXPECTED_INVENTORY:
            raise RuntimeError(
                f"Unexpected cache inventory {inventory}; expected {EXPECTED_INVENTORY}"
            )

        first_answers = prefix_logits.argmax(axis=-1)
        exact_eligible = bool(
            first_answers[0] == 0
            and first_answers[1] == 0
            and first_answers[2] != 0
            and first_answers[3] == first_answers[2]
        )
        arrays["exact_eligible"][qi] = exact_eligible
        if not exact_eligible:
            arrays["completed"][qi] = True
            atomic_save_npz(result_path, **arrays)
            answers = "".join(LETTERS[int(index)] for index in first_answers)
            print(
                f"selectedness boundary K/V {split}: screened out "
                f"{row['question_id']} (first decisions {answers})",
                flush=True,
            )
            del source_cache, prefix_output
            torch.cuda.empty_cache()
            continue

        recipient_first_letter = LETTERS[int(first_answers[2])]
        recipient_content = row["recipient_first_new_to_original"][
            recipient_first_letter
        ]
        recipient_second_letter = row["second_original_to_new"][recipient_content]
        arrays["exact_recipient_winner_first_letter"][qi] = recipient_first_letter
        arrays["exact_recipient_winner_original_content"][qi] = recipient_content
        arrays["exact_recipient_winner_second_letter"][qi] = recipient_second_letter

        natural_output = _forward(model, parts, input_ids, attention_mask)
        arrays["natural_logits"][:, qi] = _aggregate_logits(
            natural_output, variant_ids
        )
        suffix_ids = input_ids[:, cut:]

        identity_cache = copy.deepcopy(source_cache)
        identity_output = _cached_forward(
            model,
            parts,
            suffix_ids,
            attention_mask,
            past_key_values=identity_cache,
        )
        identity_logits = _aggregate_logits(identity_output, variant_ids)
        arrays["cached_identity_logits"][:, qi] = identity_logits

        boundary_cache, layer_count, delta_l2, delta_relative = (
            _patch_boundary_attention_kv(source_cache, boundaries[0])
        )
        if layer_count != 16:
            raise RuntimeError(f"Patched {layer_count} attention layers, expected 16")
        boundary_output = _cached_forward(
            model,
            parts,
            suffix_ids,
            attention_mask,
            past_key_values=boundary_cache,
        )
        boundary_logits = _aggregate_logits(boundary_output, variant_ids)
        arrays["boundary_kv_logits"][:, qi] = boundary_logits
        arrays["boundary_kv_delta_l2"][qi] = delta_l2
        arrays["boundary_kv_delta_relative"][qi] = delta_relative
        donor_control_error = float(
            np.max(np.abs(boundary_logits[:2] - identity_logits[:2]))
        )
        arrays["boundary_donor_control_max_abs_error"][qi] = donor_control_error
        if donor_control_error > full_cache_tolerance:
            raise RuntimeError(
                f"Untouched donor rows changed by {donor_control_error:.6g}"
            )

        complete_cache, counts = _swap_cache_families(
            source_cache, 7, donor_rows=DONOR_ROWS
        )
        if counts != EXPECTED_INVENTORY:
            raise RuntimeError(
                f"Complete cache swapped {counts}; expected {EXPECTED_INVENTORY}"
            )
        complete_output = _cached_forward(
            model,
            parts,
            suffix_ids,
            attention_mask,
            past_key_values=complete_cache,
        )
        complete_logits = _aggregate_logits(complete_output, variant_ids)
        arrays["complete_cache_logits"][:, qi] = complete_logits
        complete_error = float(
            np.max(np.abs(complete_logits - identity_logits[DONOR_ROWS]))
        )
        arrays["complete_cache_donor_max_abs_error"][qi] = complete_error
        if complete_error > full_cache_tolerance:
            raise RuntimeError(
                f"Complete-cache positive control failed on {row['question_id']}: "
                f"{complete_error:.6g} > {full_cache_tolerance:.6g}"
            )

        arrays["completed"][qi] = True
        atomic_save_npz(result_path, **arrays)
        question_durations.append(time.monotonic() - question_started)
        done = int(arrays["completed"].sum())
        print(
            f"selectedness boundary K/V {split}: {done}/{len(rows)}; "
            f"eligible={int(arrays['exact_eligible'].sum())}; "
            f"full-cache error={complete_error:.3g}",
            flush=True,
        )

        if not audit_path.exists():
            audit_path.write_text(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "cells": list(CELLS),
                        "first_decision_answers": [
                            LETTERS[int(index)] for index in first_answers
                        ],
                        "first_decision_boundaries": boundaries,
                        "boundary_tokens": tokenizer.convert_ids_to_tokens(
                            [ids[position] for ids, position in zip(token_rows, boundaries)]
                        ),
                        "cut": cut,
                        "cache_inventory": inventory,
                        "target_original_content": "A",
                        "target_second_letter": row["target_second_letter"],
                        "exact_recipient_winner_original_content": recipient_content,
                        "exact_recipient_winner_second_letter": recipient_second_letter,
                        "suffix_identity": {
                            "evaluation": token_rows[0][cut:] == token_rows[2][cut:],
                            "neutral": token_rows[1][cut:] == token_rows[3][cut:],
                        },
                        "rendered_prompts": dict(zip(CELLS, prompts)),
                    },
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

        del (
            source_cache,
            prefix_output,
            natural_output,
            identity_cache,
            identity_output,
            boundary_cache,
            boundary_output,
            complete_cache,
            complete_output,
        )
        torch.cuda.empty_cache()

    metadata.update(
        {
            "complete": bool(arrays["completed"].all()),
            "n_exact_eligible": int(arrays["exact_eligible"].sum()),
            "elapsed_seconds_after_model_load": time.monotonic() - started_after_load,
            "completed_question_durations_seconds": question_durations,
        }
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--full-cache-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    run(
        args.config,
        args.cohort,
        args.output,
        args.split,
        args.max_questions,
        args.full_cache_tolerance,
    )


if __name__ == "__main__":
    main()
