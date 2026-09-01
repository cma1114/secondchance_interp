from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSDPAQuerySourceAttentionAblator
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .run_first_decision_cross_order_patching import _decision_position
from .run_fixed_a_final_query_edge_ablation import _option_line_positions
from .run_fixed_a_full_cache_factorial import _cached_forward
from .run_semantic_binding_module_factorial import (
    _aggregate_logits,
    _forward,
    _messages,
    _remap_question,
)


LETTERS = "ABCD"
ROW_NAMES = ("chosen_game", "chosen_neutral", "unchosen_game", "unchosen_neutral")
CONDITIONS = ("incorrect_again", "lost_again", "incorrect_again", "lost_again")
SOURCE_CONTENTS = tuple(LETTERS)
ORDINARY_BLOCKS = tuple(range(4, 49, 4))


def _initialize(path: Path, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    qids = [row["question_id"] for row in rows]
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses a different question order")
        if arrays["row_names"].astype(str).tolist() != list(ROW_NAMES):
            raise ValueError("Existing checkpoint uses different physical rows")
        if arrays["source_contents"].astype(str).tolist() != list(SOURCE_CONTENTS):
            raise ValueError("Existing checkpoint uses different source contents")
        return arrays
    n = len(rows)
    return {
        "question_ids": np.asarray(qids),
        "row_names": np.asarray(ROW_NAMES),
        "source_contents": np.asarray(SOURCE_CONTENTS),
        "target_second_letters": np.asarray([row["target_second_letter"] for row in rows]),
        "completed": np.zeros(n, dtype=bool),
        "exact_eligible": np.zeros(n, dtype=bool),
        "prefix_identity": np.zeros(n, dtype=bool),
        "source_kv_max_abs_error": np.full(n, np.nan, dtype=np.float32),
        "first_decision_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "exact_unchosen_first_letter": np.full(n, "", dtype="<U1"),
        "exact_unchosen_original_content": np.full(n, "", dtype="<U1"),
        "exact_unchosen_second_letter": np.full(n, "", dtype="<U1"),
        "natural_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "intervention_logits": np.full((4, 4, n, 4), np.nan, dtype=np.float32),
        "source_position_counts": np.zeros((4, 4, n), dtype=np.int16),
        "query_position_counts": np.zeros((4, n), dtype=np.int16),
        "prompt_hashes": np.full((4, n), "", dtype="<U64"),
    }


def _cache_source_error(cache: Any, positions: list[int]) -> tuple[float, int]:
    """Compare ordinary-attention K/V for the identical A line across histories."""
    import torch

    maximum = 0.0
    layers = 0
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None and values is None:
            continue
        if keys is None or values is None or keys.numel() == 0 or values.numel() == 0:
            raise RuntimeError("Conventional-attention K/V cache initialization mismatch")
        if keys.shape[0] != 4 or values.shape[0] != 4:
            raise RuntimeError("Expected four physical cache rows")
        index = torch.as_tensor(positions, dtype=torch.long, device=keys.device)
        for left, right in ((0, 1), (2, 3), (0, 2), (1, 3)):
            maximum = max(
                maximum,
                float((keys[left, :, index] - keys[right, :, index]).abs().max()),
                float((values[left, :, index] - values[right, :, index]).abs().max()),
            )
        layers += 1
    return maximum, layers


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def run(
    config_path: Path,
    cohort_path: Path,
    output_dir: Path,
    split: str,
    max_questions: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires action-matched incorrect/lost feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw Qwen ChatML")
    if config.attn_implementation != "sdpa" or int(config.batch_size) != 4:
        raise ValueError("Requires the established batch-four SDPA regime")
    if split not in {"discovery", "confirmation"}:
        raise ValueError("Unknown split")

    cohort = json.loads(cohort_path.read_text())
    rows = [row for row in cohort["rows"] if row["split"] == split]
    if not rows:
        raise ValueError(f"No {split} rows")
    for row in rows:
        if row["target_original_content"] != "A":
            raise ValueError("The identical-source design requires semantic A")
        if row["donor_first_new_to_original"]["A"] != "A":
            raise ValueError("Chosen history moved semantic A")
        if row["recipient_first_new_to_original"]["A"] != "A":
            raise ValueError("Unchosen history moved semantic A")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, rows)

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    ordinary = tuple(
        index + 1
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if ordinary != tuple(range(4, 65, 4)):
        raise RuntimeError(f"Unexpected ordinary-attention blocks: {ordinary}")

    audit_path = output_dir / "prompt_audit.json"
    started = time.monotonic()
    durations: list[float] = []

    for qi, row in enumerate(rows):
        if max_questions is not None and qi >= int(max_questions):
            break
        if arrays["completed"][qi]:
            continue
        question_started = time.monotonic()
        question = questions[row["question_id"]]
        chosen_first = _remap_question(question, row["donor_first_new_to_original"])
        unchosen_first = _remap_question(question, row["recipient_first_new_to_original"])
        second = _remap_question(question, row["second_new_to_original"])
        first_questions = (chosen_first, chosen_first, unchosen_first, unchosen_first)

        prompts: list[str] = []
        token_rows: list[list[int]] = []
        boundaries: list[int] = []
        first_positions: list[dict[str, list[int]]] = []
        second_positions: list[dict[str, list[int]]] = []
        first_audits: list[dict[str, Any]] = []
        second_audits: list[dict[str, Any]] = []
        for first, condition in zip(first_questions, CONDITIONS):
            prompt = render_chat(
                processor,
                _messages(config, first, second, condition),
                config.disable_thinking,
                config.chat_serialization,
            )
            boundary, ids = _decision_position(tokenizer, prompt)
            source_positions, source_audit = _option_line_positions(tokenizer, prompt, first)
            repeated_positions, repeated_audit = _option_line_positions(tokenizer, prompt, second)
            prompts.append(prompt)
            token_rows.append(ids)
            boundaries.append(boundary)
            first_positions.append(source_positions)
            second_positions.append(repeated_positions)
            first_audits.append(source_audit)
            second_audits.append(repeated_audit)
            arrays["prompt_hashes"][len(prompts) - 1, qi] = _hash_prompt(prompt)

        if len({len(ids) for ids in token_rows}) != 1:
            raise RuntimeError("The four matched prompts are not token-aligned")
        if len(set(boundaries)) != 1:
            raise RuntimeError("First-decision boundary positions differ")
        cut = boundaries[0] + 1
        a_end = max(first_positions[0]["A"])
        prefix_identity = all(
            token_rows[0][: a_end + 1] == token_rows[index][: a_end + 1]
            for index in range(1, 4)
        )
        arrays["prefix_identity"][qi] = prefix_identity
        if not prefix_identity:
            raise RuntimeError("The supposedly identical A-line prefix differs")
        if any(first_positions[index]["A"] != first_positions[0]["A"] for index in range(1, 4)):
            raise RuntimeError("The A source positions differ across matched histories")

        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        if input_ids.shape[1] != len(token_rows[0]):
            raise RuntimeError("Unexpected padding in token-aligned four-row batch")
        prefix_output = _cached_forward(
            model, parts, input_ids[:, :cut], attention_mask[:, :cut]
        )
        prefix_logits = _aggregate_logits(prefix_output, variant_ids)
        if not np.all(np.isfinite(prefix_logits)):
            raise RuntimeError(
                "Non-finite first-decision logits; this host/model execution is invalid"
            )
        arrays["first_decision_logits"][:, qi] = prefix_logits
        source_error, cache_layers = _cache_source_error(
            prefix_output.past_key_values, first_positions[0]["A"]
        )
        arrays["source_kv_max_abs_error"][qi] = source_error
        if cache_layers != 16:
            raise RuntimeError(f"Expected 16 ordinary-attention caches, got {cache_layers}")
        if source_error != 0.0:
            raise RuntimeError(f"Identical A-source K/V differs by {source_error}")

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
            print(
                f"selectedness edge {split}: screened out {row['question_id']} "
                f"(first decisions {''.join(LETTERS[int(value)] for value in first_answers)})",
                flush=True,
            )
            del prefix_output
            torch.cuda.empty_cache()
            continue

        unchosen_letter = LETTERS[int(first_answers[2])]
        unchosen_content = row["recipient_first_new_to_original"][unchosen_letter]
        unchosen_second = row["second_original_to_new"][unchosen_content]
        arrays["exact_unchosen_first_letter"][qi] = unchosen_letter
        arrays["exact_unchosen_original_content"][qi] = unchosen_content
        arrays["exact_unchosen_second_letter"][qi] = unchosen_second
        del prefix_output
        torch.cuda.empty_cache()

        natural = _aggregate_logits(
            _forward(model, parts, input_ids, attention_mask), variant_ids
        )
        if not np.all(np.isfinite(natural)):
            raise RuntimeError("Non-finite natural logits")
        arrays["natural_logits"][:, qi] = natural
        width = int(input_ids.shape[1])
        target_display = row["target_second_letter"]

        for source_index, source_content in enumerate(SOURCE_CONTENTS):
            specs: dict[int, dict[int, dict[int, list[int]]]] = {}
            for block in ORDINARY_BLOCKS:
                row_specs: dict[int, dict[int, list[int]]] = {}
                for physical_row in range(4):
                    first_mapping = (
                        row["donor_first_new_to_original"]
                        if physical_row < 2
                        else row["recipient_first_new_to_original"]
                    )
                    original_to_new = {value: key for key, value in first_mapping.items()}
                    source_display = original_to_new[source_content]
                    left_pad = width - len(token_rows[physical_row])
                    sources = [
                        left_pad + value
                        for value in first_positions[physical_row][source_display]
                    ]
                    queries = [
                        left_pad + value
                        for value in second_positions[physical_row][target_display]
                    ]
                    if not queries or not sources:
                        raise RuntimeError("Missing source or repeated-A query positions")
                    if max(sources) >= min(queries):
                        raise RuntimeError("Original source does not precede repeated-A query")
                    row_specs[physical_row] = {query: sources for query in queries}
                    arrays["source_position_counts"][source_index, physical_row, qi] = len(sources)
                    arrays["query_position_counts"][physical_row, qi] = len(queries)
                specs[block - 1] = row_specs
            with BatchedSDPAQuerySourceAttentionAblator(parts, specs):
                intervened = _aggregate_logits(
                    _forward(model, parts, input_ids, attention_mask), variant_ids
                )
            if not np.all(np.isfinite(intervened)):
                raise RuntimeError(
                    f"Non-finite intervention logits for source {source_content}"
                )
            arrays["intervention_logits"][source_index, :, qi] = intervened

        arrays["completed"][qi] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - question_started
        durations.append(duration)
        print(
            f"selectedness edge {split}: {int(arrays['completed'].sum())}/{len(rows)}; "
            f"exact={int(arrays['exact_eligible'].sum())}; seconds={duration:.2f}",
            flush=True,
        )

        if not audit_path.exists():
            audit = {
                "question_id": row["question_id"],
                "row_names": list(ROW_NAMES),
                "conditions": list(CONDITIONS),
                "ordinary_blocks_one_based": list(ORDINARY_BLOCKS),
                "first_decisions": [LETTERS[int(value)] for value in first_answers],
                "target_second_letter": target_display,
                "prefix_identity_through_A": prefix_identity,
                "source_kv_max_abs_error": source_error,
                "rows": [
                    {
                        "name": ROW_NAMES[index],
                        "prompt_hash": arrays["prompt_hashes"][index, qi].item(),
                        "first_A": first_audits[index]["A"],
                        "second_A": second_audits[index][target_display],
                    }
                    for index in range(4)
                ],
            }
            audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    metadata = {
        "experiment": "W1=A identical-source selectedness attention-edge test",
        "config": config.as_dict(),
        "config_path": str(config_path),
        "cohort_path": str(cohort_path),
        "split": split,
        "n_frozen_questions": len(rows),
        "n_completed": int(arrays["completed"].sum()),
        "n_exact_eligible": int(arrays["exact_eligible"].sum()),
        "row_names": list(ROW_NAMES),
        "source_contents": list(SOURCE_CONTENTS),
        "ordinary_blocks_one_based": list(ORDINARY_BLOCKS),
        "complete_model_forwards_per_exact_question": 6,
        "complete_model_work": (
            "one cached first-decision eligibility/KV-identity forward, one full natural "
            "forward, and four original-option-source to repeated-A edge lesions"
        ),
        "primary_intervention": (
            "In ordinary-attention blocks 4-48, block every token of the repeated semantic-A "
            "option line from reading every token of the original semantic-A option line. "
            "Repeat separately for semantic B/C/D source lines as controls."
        ),
        "elapsed_seconds_after_load": time.monotonic() - started,
        "eligible_question_seconds": durations,
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
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    run(args.config, args.cohort, args.output_dir, args.split, args.max_questions)


if __name__ == "__main__":
    main()
