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
from .run_fixed_a_final_query_edge_ablation import _option_line_positions
from .run_fixed_a_full_cache_factorial import DONOR_ROWS, _aggregate_logits, _cache_inventory, _cached_forward
from .run_fixed_a_kv_source_transplant import _patch_attention_kv_positions, _source_positions
from .run_semantic_binding_module_factorial import CONDITIONS, _forward, _messages, _remap_question


SCENARIOS = (
    "recipient_open",
    "donor_open",
    "recipient_matching_block",
    "donor_matching_block",
    "recipient_control_block",
    "donor_control_block",
)
ORDINARY_BLOCKS = tuple(range(4, 65, 4))


class CachedQuerySourceAblator:
    """Block selected suffix-query to cached-prefix-source edges."""

    def __init__(
        self,
        parts: Any,
        specs: dict[int, dict[int, dict[int, list[int]]]],
        query_offset: int,
    ) -> None:
        import torch

        self.specs = specs
        self.query_offset = int(query_offset)
        self.active: int | None = None
        self.handles: list[Any] = []
        for layer_index in sorted(specs):
            attention = getattr(parts.layers[layer_index], "self_attn", None)
            if attention is None:
                self.close()
                raise ValueError(f"Layer {layer_index + 1} is not ordinary attention")
            self.handles.append(attention.register_forward_pre_hook(self._enter(layer_index)))
            self.handles.append(attention.register_forward_hook(self._leave(layer_index)))
        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = self._wrapped_sdpa

    def _enter(self, layer_index: int):
        def enter(_module: Any, _inputs: Any) -> None:
            if self.active is not None:
                raise RuntimeError("Nested ordinary-attention calls are unsupported")
            self.active = layer_index
        return enter

    def _leave(self, layer_index: int):
        def leave(_module: Any, _inputs: Any, _output: Any) -> None:
            if self.active != layer_index:
                raise RuntimeError("Ordinary-attention layer stack became inconsistent")
            self.active = None
        return leave

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        import torch

        layer = self.active
        if layer is None:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        if args:
            raise RuntimeError("Validated Qwen SDPA path uses keyword options")
        rows = self.specs[layer]
        batch, _heads, query_length, _dim = query.shape
        key_length = int(key.shape[-2])
        for row, queries in rows.items():
            if row >= batch:
                raise RuntimeError("Ablation row exceeds batch")
            for query_position, sources in queries.items():
                if query_position < 0 or query_position >= query_length:
                    raise RuntimeError("Suffix query position outside query length")
                absolute_query = self.query_offset + query_position
                for source in sources:
                    if source < 0 or source >= key_length or source >= absolute_query:
                        raise RuntimeError("Cached source is not before selected query")

        mask = kwargs.get("attn_mask")
        if mask is not None:
            if mask.ndim != 4:
                raise RuntimeError("Unexpected cached attention-mask rank")
            if mask.shape[0] == 1 and batch > 1:
                patched = mask.expand(batch, *mask.shape[1:]).clone()
            elif mask.shape[0] == batch:
                patched = mask.clone()
            else:
                raise RuntimeError("Cached attention-mask batch mismatch")
            for row, queries in rows.items():
                for query_position, sources in queries.items():
                    for source in sources:
                        if patched.dtype == torch.bool:
                            patched[row, :, query_position, source] = False
                        else:
                            patched[row, :, query_position, source] = -torch.inf
            options = dict(kwargs)
            options["attn_mask"] = patched
            return self.original_sdpa(query, key, value, **options)

        raise RuntimeError(
            "CachedQuerySourceAblator requires the model-supplied attention mask; "
            "the old mask-free reconstruction assumed bottom-right causal alignment "
            "and is intentionally disabled."
        )

    def close(self) -> None:
        import torch

        if hasattr(self, "original_sdpa"):
            torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in reversed(getattr(self, "handles", [])):
            handle.remove()
        self.handles = []
        self.active = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _initialize(path: Path, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    qids = [row["question_id"] for row in rows]
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses a different question order")
        return arrays
    n = len(rows)
    return {
        "question_ids": np.asarray(qids),
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


def _specs(
    layer_indices: tuple[int, ...],
    source_positions: list[list[int]],
    query_positions: list[list[int]],
) -> dict[int, dict[int, dict[int, list[int]]]]:
    return {
        layer: {
            row: {query: list(source_positions[row]) for query in query_positions[row]}
            for row in range(len(source_positions))
        }
        for layer in layer_indices
    }


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
    cohort = json.loads(cohort_path.read_text())
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    rows = [row for row in cohort["rows"] if row["split"] == split]
    if max_questions is not None:
        rows = rows[: int(max_questions)]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, rows)

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    variants = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {letter: [token_id for _, token_id in variants[letter]] for letter in "ABCD"}
    layer_indices = tuple(
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if tuple(index + 1 for index in layer_indices) != ORDINARY_BLOCKS:
        raise RuntimeError("Unexpected ordinary-attention layer inventory")
    expected_inventory = {"attention_kv": 16, "gla_conv": 48, "gla_recurrent": 48}
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
            source_map, source_audit = _source_positions(tokenizer, prompt, first, boundary)
            second_positions, second_audit = _option_line_positions(tokenizer, prompt, second)
            donor_target = row["y_second_letter"] if cell < 2 else row["x_second_letter"]
            alternatives = [letter for letter in "ABCD" if letter != donor_target]
            control = min(
                alternatives,
                key=lambda letter: (
                    abs(len(second_positions[letter]) - len(second_positions[donor_target])),
                    letter,
                ),
            )
            prompts.append(prompt)
            token_rows.append(ids)
            boundaries.append(boundary)
            source_rows.append(source_map["selected_option"])
            matching_absolute.append(second_positions[donor_target])
            control_absolute.append(second_positions[control])
            arrays["source_position_counts"][cell, qi] = len(source_map["selected_option"])
            arrays["matching_query_counts"][cell, qi] = len(second_positions[donor_target])
            arrays["control_query_counts"][cell, qi] = len(second_positions[control])
            arrays["matching_query_letters"][cell, qi] = donor_target
            arrays["control_query_letters"][cell, qi] = control
            audits.append({
                "cell": cell,
                "condition": condition,
                "source": source_audit,
                "second": second_audit,
                "donor_matching_letter": donor_target,
                "control_letter": control,
            })

        if len(set(map(len, token_rows))) != 1 or len(set(boundaries)) != 1:
            raise RuntimeError("Fixed-A mediation prompts are not token aligned")
        cut = boundaries[0] + 1
        if token_rows[0][cut:] != token_rows[2][cut:] or token_rows[1][cut:] != token_rows[3][cut:]:
            raise RuntimeError("X/Y suffixes differ after the first-decision boundary")
        matching_local = [[position - cut for position in row_positions] for row_positions in matching_absolute]
        control_local = [[position - cut for position in row_positions] for row_positions in control_absolute]
        if any(not positions or min(positions) < 0 for positions in matching_local + control_local):
            raise RuntimeError("Receiver query is not in cached suffix")

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
        valid = bool(np.all(arrays["first_decision_logits"][:, qi].argmax(axis=-1) == 0))
        arrays["first_decision_valid"][qi] = valid
        if _cache_inventory(source_cache) != expected_inventory:
            raise RuntimeError("Unexpected hybrid cache inventory")
        if not valid:
            arrays["completed"][qi] = True
            atomic_save_npz(result_path, **arrays)
            continue

        suffix_ids = input_ids[:, cut:]
        donor_cache, count = _patch_attention_kv_positions(source_cache, source_rows)
        if count != 16:
            raise RuntimeError("Donor transplant did not patch all ordinary-attention layers")
        matching_specs = _specs(layer_indices, source_rows, matching_local)
        control_specs = _specs(layer_indices, source_rows, control_local)

        scenario_defs = (
            (source_cache, None),
            (donor_cache, None),
            (source_cache, matching_specs),
            (donor_cache, matching_specs),
            (source_cache, control_specs),
            (donor_cache, control_specs),
        )
        for scenario, (base_cache, specs) in enumerate(scenario_defs):
            cache = copy.deepcopy(base_cache)
            if specs is None:
                output = _cached_forward(model, parts, suffix_ids, attention_mask, past_key_values=cache)
            else:
                with CachedQuerySourceAblator(parts, specs, cut):
                    output = _cached_forward(model, parts, suffix_ids, attention_mask, past_key_values=cache)
            arrays["scenario_logits"][scenario, :, qi] = _aggregate_logits(output, variant_ids)
            del cache, output

        arrays["completed"][qi] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - question_started
        durations.append(duration)
        print(
            f"fixed-A donor mediation {split}: {int(arrays['completed'].sum())}/{len(rows)}; "
            f"question_seconds={duration:.2f}", flush=True,
        )
        if not (output_dir / "prompt_audit.json").exists():
            (output_dir / "prompt_audit.json").write_text(
                json.dumps({"question_id": row["question_id"], "cut": cut, "cells": audits}, indent=2) + "\n"
            )

    metadata = {
        "experiment": "fixed-A donor-to-repeated-option serial mediation",
        "config": config.as_dict(),
        "split": split,
        "n_questions": len(rows),
        "scenarios": list(SCENARIOS),
        "ordinary_blocks_one_based": list(ORDINARY_BLOCKS),
        "complete_model_forwards_per_question": 8,
        "complete_model_work": "one full natural, one cached prefix, and six cached suffix scenarios",
        "elapsed_seconds_after_load": time.monotonic() - started,
        "question_seconds": durations,
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
    args = parser.parse_args()
    run(args.config, args.cohort, args.output_dir, args.split, args.max_questions)


if __name__ == "__main__":
    main()
