from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
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
from .sublayer import _hidden, _replace_hidden


DEFAULT_READOUTS = (1, 8, 16, 24, 32, 40, 48, 52, 56, 60, 63)
PATCH_CONDITIONS = ("evaluation", "neutral")


class FirstDecisionCollector:
    def __init__(self, parts: Any, positions: list[int], readouts: list[int]) -> None:
        import torch

        self.positions = torch.as_tensor(positions, dtype=torch.long)
        self.values: dict[int, Any] = {}
        self.handles = [
            parts.layers[readout - 1].register_forward_hook(self._hook(readout))
            for readout in readouts
        ]

    def _hook(self, readout: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = _hidden(output)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            positions = self.positions.to(hidden.device)
            self.values[readout] = hidden[rows, positions].detach().to(
                "cpu", dtype=torch.float16
            )

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class FixedASemanticSwap:
    """Cross-swap X/Y decision residuals in one condition; identity-patch the other."""

    def __init__(
        self,
        parts: Any,
        readout: int,
        positions: list[int],
        source: Any,
        patch_condition: int,
    ) -> None:
        self.positions = list(map(int, positions))
        self.source = source
        self.patch_rows = (0, 2) if patch_condition == 0 else (1, 3)
        self.donor = {0: 2, 2: 0, 1: 3, 3: 1}
        self.identity_error = np.nan
        self.donor_delta = np.nan
        self.handle = parts.layers[readout - 1].register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = _hidden(output)
        updated = hidden.clone()
        source = self.source.to(device=hidden.device, dtype=hidden.dtype)
        identity_errors = []
        donor_deltas = []
        for row in range(hidden.shape[0]):
            current = hidden[row, self.positions[row]]
            identity_errors.append(float((current.float() - source[row].float()).norm()))
            if row in self.patch_rows:
                replacement = source[self.donor[row]]
                donor_deltas.append(float((replacement.float() - source[row].float()).norm()))
            else:
                replacement = source[row]
            updated[row, self.positions[row]] = replacement
        self.identity_error = float(max(identity_errors))
        self.donor_delta = float(np.mean(donor_deltas))
        return _replace_hidden(output, updated)

    def close(self) -> None:
        self.handle.remove()


def _initialize(path: Path, rows: list[dict], readouts: list[int]) -> dict[str, np.ndarray]:
    qids = [row["question_id"] for row in rows]
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing output uses a different question order")
        if arrays["readouts"].astype(int).tolist() != list(readouts):
            raise ValueError("Existing output uses different readouts")
        return arrays
    n = len(rows)
    l = len(readouts)
    return {
        "question_ids": np.asarray(qids),
        "readouts": np.asarray(readouts, dtype=np.int16),
        "x_second_letter": np.asarray([row["x_second_letter"] for row in rows]),
        "y_second_letter": np.asarray([row["y_second_letter"] for row in rows]),
        "natural_completed": np.zeros(n, dtype=bool),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((4, n, 4), np.nan, np.float32),
        "patched_batch_logits": np.full((2, l, 4, n, 4), np.nan, np.float32),
        "identity_source_error_norm": np.full((2, l, n), np.nan, np.float32),
        "donor_identity_delta_norm": np.full((2, l, n), np.nan, np.float32),
    }


def run(
    config_path: Path,
    cohort_path: Path,
    output_dir: Path,
    split: str,
    readouts: list[int],
    max_questions: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")
    if config.attn_implementation != "sdpa" or config.batch_size != 4:
        raise ValueError("Requires the established batch-4 SDPA regime")
    if split not in {"discovery", "confirmation"}:
        raise ValueError("Unknown split")
    if not readouts or min(readouts) < 1 or max(readouts) > 63:
        raise ValueError("Readouts must lie in 1--63")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    cohort = json.loads(cohort_path.read_text())
    rows = [row for row in cohort["rows"] if row["split"] == split]
    if max_questions is not None:
        rows = rows[: int(max_questions)]
    if not rows:
        raise ValueError("No fixed-A questions")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, rows, readouts)
    audit_path = output_dir / "prompt_audit.json"

    metadata = {
        "config": config.as_dict(),
        "cohort_path": str(cohort_path),
        "split": split,
        "readouts": readouts,
        "cells": CELLS,
        "patch_conditions": PATCH_CONDITIONS,
        "complete_model_forward_passes_per_question": 1 + 2 * len(readouts),
        "intervention": (
            "At one post-block readout, exchange the complete first-decision residual "
            "between same-question X/Y histories whose literal first decision is A. "
            "The other feedback condition is identity-patched in the same pass."
        ),
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
        if arrays["completed"][qi]:
            continue
        question = questions[row["question_id"]]
        first_x = _remap_question(question, row["first_x_new_to_original"])
        first_y = _remap_question(question, row["first_y_new_to_original"])
        second = _remap_question(question, row["second_new_to_original"])
        prompts = []
        unpadded_positions = []
        token_ids = []
        for first, condition in zip((first_x, first_x, first_y, first_y), CONDITIONS):
            prompt = render_chat(
                processor,
                _messages(config, first, second, condition),
                config.disable_thinking,
                config.chat_serialization,
            )
            position, ids = _decision_position(tokenizer, prompt)
            prompts.append(prompt)
            unpadded_positions.append(position)
            token_ids.append(ids)
        if len({len(ids) for ids in token_ids}) != 1:
            raise RuntimeError("Fixed-A prompt token lengths differ")
        if len(set(unpadded_positions)) != 1:
            raise RuntimeError("First-decision positions differ across cells")

        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        width = int(input_ids.shape[1])
        positions = [
            position + width - len(ids)
            for position, ids in zip(unpadded_positions, token_ids)
        ]
        collector = FirstDecisionCollector(parts, positions, readouts)
        try:
            natural_output = _forward(model, parts, input_ids, attention_mask)
        finally:
            collector.close()
        natural_logits = _aggregate_logits(natural_output, variant_ids)
        arrays["natural_logits"][:, qi] = natural_logits
        arrays["natural_completed"][qi] = True

        for condition_index in range(2):
            for li, readout in enumerate(readouts):
                patcher = FixedASemanticSwap(
                    parts,
                    readout,
                    positions,
                    collector.values[readout],
                    condition_index,
                )
                try:
                    patched_output = _forward(model, parts, input_ids, attention_mask)
                finally:
                    patcher.close()
                arrays["patched_batch_logits"][condition_index, li, :, qi] = (
                    _aggregate_logits(patched_output, variant_ids)
                )
                arrays["identity_source_error_norm"][condition_index, li, qi] = (
                    patcher.identity_error
                )
                arrays["donor_identity_delta_norm"][condition_index, li, qi] = (
                    patcher.donor_delta
                )

        arrays["completed"][qi] = True
        atomic_save_npz(result_path, **arrays)
        done = int(arrays["completed"].sum())
        if done == 1 or done % 5 == 0 or done == len(rows):
            print(f"fixed-A first-decision transfer {split}: {done}/{len(rows)}", flush=True)

        if not audit_path.exists():
            audit_path.write_text(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "literal_first_answer": row["literal_first_answer"],
                        "x_content_original_letter": row["x_content_original_letter"],
                        "y_content_original_letter": row["y_content_original_letter"],
                        "first_decision_positions_unpadded": unpadded_positions,
                        "first_decision_tokens": [
                            tokenizer.decode([ids[pos]])
                            for ids, pos in zip(token_ids, unpadded_positions)
                        ],
                        "rendered_prompts": dict(zip(CELLS, prompts)),
                    },
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--readouts", nargs="+", type=int, default=list(DEFAULT_READOUTS))
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    run(args.config, args.cohort, args.output_dir, args.split, args.readouts, args.max_questions)


if __name__ == "__main__":
    main()
