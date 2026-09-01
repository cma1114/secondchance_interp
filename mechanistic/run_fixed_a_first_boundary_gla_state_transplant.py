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


PATCH_CONDITIONS = ("evaluation", "neutral")


def _initialize(path: Path, rows: list[dict[str, Any]], gla_layers: list[int]) -> dict[str, np.ndarray]:
    qids = [row["question_id"] for row in rows]
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing output uses a different question order")
        if arrays["gla_layer_indices"].astype(int).tolist() != gla_layers:
            raise ValueError("Existing output uses different GLA layers")
        return arrays
    n = len(rows)
    return {
        "question_ids": np.asarray(qids),
        "gla_layer_indices": np.asarray(gla_layers, dtype=np.int16),
        "x_second_letter": np.asarray([row["x_second_letter"] for row in rows]),
        "y_second_letter": np.asarray([row["y_second_letter"] for row in rows]),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "identity_state_logits": np.full((4, n, 4), np.nan, dtype=np.float32),
        "cross_state_logits": np.full((2, 4, n, 4), np.nan, dtype=np.float32),
        "cross_state_delta_norm": np.full((2, n, len(gla_layers)), np.nan, dtype=np.float32),
        "recipient_state_norm": np.full((4, n, len(gla_layers)), np.nan, dtype=np.float32),
    }


class BatchedGLAStateCollector:
    """Capture each accumulated GLA state immediately after a common batch boundary."""

    def __init__(self, parts: Any, gla_layers: list[int], cut: int) -> None:
        self.values: dict[int, np.ndarray] = {}
        self.originals: list[tuple[Any, Any]] = []
        self.cut = int(cut)
        for layer_index in gla_layers:
            module = getattr(parts.layers[layer_index], "linear_attn", None)
            if module is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not Gated DeltaNet")
            original = module.chunk_gated_delta_rule

            def wrapped(
                query: Any,
                key: Any,
                value: Any,
                *args: Any,
                _original=original,
                _layer_index=layer_index,
                **kwargs: Any,
            ):
                if kwargs.get("initial_state") is not None:
                    raise RuntimeError("Collector expects an uncached full-prompt pass")
                prefix_kwargs = dict(kwargs)
                prefix_kwargs["g"] = kwargs["g"][:, : self.cut]
                prefix_kwargs["beta"] = kwargs["beta"][:, : self.cut]
                prefix_kwargs["initial_state"] = None
                prefix_kwargs["output_final_state"] = True
                prefix_kwargs["cu_seqlens"] = None
                _, state = _original(
                    query[:, : self.cut],
                    key[:, : self.cut],
                    value[:, : self.cut],
                    *args,
                    **prefix_kwargs,
                )
                if state is None:
                    raise RuntimeError("GLA kernel did not return a recurrent state")
                self.values[int(_layer_index)] = state.detach().float().cpu().numpy()
                return _original(query, key, value, *args, **kwargs)

            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped

    def close(self) -> None:
        for module, original in reversed(getattr(self, "originals", [])):
            module.chunk_gated_delta_rule = original
        self.originals = []


class BatchedGLAStatePatcher:
    """Resume every GLA suffix from identity or cross-semantic cached states."""

    def __init__(self, parts: Any, cache: dict[int, np.ndarray], cut: int) -> None:
        self.originals: list[tuple[Any, Any]] = []
        self.cut = int(cut)
        for layer_index, layer_cache in cache.items():
            module = getattr(parts.layers[layer_index], "linear_attn", None)
            if module is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not Gated DeltaNet")
            original = module.chunk_gated_delta_rule

            def wrapped(
                query: Any,
                key: Any,
                value: Any,
                *args: Any,
                _original=original,
                _layer_cache=layer_cache,
                **kwargs: Any,
            ):
                import torch

                if kwargs.get("initial_state") is not None:
                    raise RuntimeError("Patcher expects an uncached full-prompt pass")
                prefix_kwargs = dict(kwargs)
                prefix_kwargs["g"] = kwargs["g"][:, : self.cut]
                prefix_kwargs["beta"] = kwargs["beta"][:, : self.cut]
                prefix_kwargs["initial_state"] = None
                prefix_kwargs["output_final_state"] = False
                prefix_kwargs["cu_seqlens"] = None
                prefix_out, _ = _original(
                    query[:, : self.cut],
                    key[:, : self.cut],
                    value[:, : self.cut],
                    *args,
                    **prefix_kwargs,
                )
                suffix_kwargs = dict(kwargs)
                suffix_kwargs["g"] = kwargs["g"][:, self.cut :]
                suffix_kwargs["beta"] = kwargs["beta"][:, self.cut :]
                suffix_kwargs["initial_state"] = torch.from_numpy(_layer_cache).to(
                    device=query.device, dtype=torch.float32
                )
                suffix_kwargs["cu_seqlens"] = None
                suffix_out, state = _original(
                    query[:, self.cut :],
                    key[:, self.cut :],
                    value[:, self.cut :],
                    *args,
                    **suffix_kwargs,
                )
                return torch.cat([prefix_out, suffix_out], dim=1), state

            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped

    def close(self) -> None:
        for module, original in reversed(getattr(self, "originals", [])):
            module.chunk_gated_delta_rule = original
        self.originals = []


def _cross_cache(
    natural: dict[int, np.ndarray], condition_index: int
) -> dict[int, np.ndarray]:
    donor_rows = np.arange(4)
    if condition_index == 0:
        donor_rows[[0, 2]] = donor_rows[[2, 0]]
    else:
        donor_rows[[1, 3]] = donor_rows[[3, 1]]
    return {layer: states[donor_rows].copy() for layer, states in natural.items()}


def run(
    config_path: Path,
    cohort_path: Path,
    output_dir: Path,
    split: str,
    max_questions: int | None,
    shard_index: int,
    num_shards: int,
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

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    cohort = json.loads(cohort_path.read_text())
    rows = [row for row in cohort["rows"] if row["split"] == split]
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard index/count")
    rows = [row for index, row in enumerate(rows) if index % num_shards == shard_index]
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
    gla_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if len(gla_layers) != 48:
        raise RuntimeError(f"Expected 48 GLA layers, found {len(gla_layers)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, rows, gla_layers)
    audit_path = output_dir / "prompt_audit.json"
    metadata = {
        "config": config.as_dict(),
        "cohort_path": str(cohort_path),
        "split": split,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "n_questions": len(rows),
        "cells": CELLS,
        "patch_conditions": PATCH_CONDITIONS,
        "gla_layer_indices_zero_based": gla_layers,
        "complete_model_forward_passes_per_question": 4,
        "intervention": (
            "At all 48 GLA layers jointly, exchange the complete accumulated recurrent "
            "state immediately after the first-answer boundary between same-question "
            "X/Y histories whose literal first decision is A. The recipient visible "
            "prompt, feedback, and second presentation remain fixed."
        ),
        "causal_comparator": (
            "Cross-semantic state transplantation is compared with recipient-state "
            "reinsertion, not the unsplit natural pass, to remove segmented-kernel "
            "numerical effects."
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
        prompts: list[str] = []
        token_rows: list[list[int]] = []
        unpadded_positions: list[int] = []
        for first, condition in zip((first_x, first_x, first_y, first_y), CONDITIONS):
            prompt = render_chat(
                processor,
                _messages(config, first, second, condition),
                config.disable_thinking,
                config.chat_serialization,
            )
            position, ids = _decision_position(tokenizer, prompt)
            prompts.append(prompt)
            token_rows.append(ids)
            unpadded_positions.append(position)
        if len({len(ids) for ids in token_rows}) != 1:
            raise RuntimeError("Fixed-A prompt token lengths differ")
        if len(set(unpadded_positions)) != 1:
            raise RuntimeError("First-decision boundary positions differ across cells")

        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        width = int(input_ids.shape[1])
        physical_positions = [
            position + width - len(ids)
            for position, ids in zip(unpadded_positions, token_rows)
        ]
        if len(set(physical_positions)) != 1:
            raise RuntimeError("Padded boundary positions differ across cells")
        cut = physical_positions[0] + 1

        collector = BatchedGLAStateCollector(parts, gla_layers, cut)
        try:
            natural_output = _forward(model, parts, input_ids, attention_mask)
        finally:
            collector.close()
        arrays["natural_logits"][:, qi] = _aggregate_logits(
            natural_output, variant_ids
        )

        identity_patcher = BatchedGLAStatePatcher(parts, collector.values, cut)
        try:
            identity_output = _forward(model, parts, input_ids, attention_mask)
        finally:
            identity_patcher.close()
        arrays["identity_state_logits"][:, qi] = _aggregate_logits(
            identity_output, variant_ids
        )

        for condition_index in range(2):
            cross = _cross_cache(collector.values, condition_index)
            patcher = BatchedGLAStatePatcher(parts, cross, cut)
            try:
                patched_output = _forward(model, parts, input_ids, attention_mask)
            finally:
                patcher.close()
            arrays["cross_state_logits"][condition_index, :, qi] = _aggregate_logits(
                patched_output, variant_ids
            )
            donor_rows = (2, 0) if condition_index == 0 else (3, 1)
            target_rows = (0, 2) if condition_index == 0 else (1, 3)
            for li, layer in enumerate(gla_layers):
                states = collector.values[layer].astype(np.float64)
                deltas = [
                    np.linalg.norm(states[donor] - states[target])
                    for target, donor in zip(target_rows, donor_rows)
                ]
                arrays["cross_state_delta_norm"][condition_index, qi, li] = np.mean(deltas)
                for cell in range(4):
                    arrays["recipient_state_norm"][cell, qi, li] = np.linalg.norm(states[cell])

        arrays["completed"][qi] = True
        atomic_save_npz(result_path, **arrays)
        done = int(arrays["completed"].sum())
        if done == 1 or done % 5 == 0 or done == len(rows):
            print(f"fixed-A GLA-state transplant {split}: {done}/{len(rows)}", flush=True)

        if not audit_path.exists():
            audit_path.write_text(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "literal_first_answer": row["literal_first_answer"],
                        "x_content_original_letter": row["x_content_original_letter"],
                        "y_content_original_letter": row["y_content_original_letter"],
                        "boundary_positions_unpadded": unpadded_positions,
                        "boundary_tokens": [
                            tokenizer.decode([ids[position]])
                            for ids, position in zip(token_rows, unpadded_positions)
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
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    run(
        args.config,
        args.cohort,
        args.output_dir,
        args.split,
        args.max_questions,
        args.shard_index,
        args.num_shards,
    )


if __name__ == "__main__":
    main()
