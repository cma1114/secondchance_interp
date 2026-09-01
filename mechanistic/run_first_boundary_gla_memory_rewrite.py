from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .collect_remapped_behavior import _messages, _remap_question
from .config import ExperimentConfig
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import prompt_hash
from .run_first_decision_cross_order_patching import (
    CONDITIONS,
    LETTERS,
    _aggregate_logits,
    _load_mapping_plans,
    _question_ids,
)
from .run_first_span_gla_ablation import _source_positions
from .run_historical_answer_intervention import _forward


TENSOR_NAMES = ("key", "value", "g", "beta")


def _initialize(path: Path, qids: list[str], gla_layers: list[int]) -> dict[str, np.ndarray]:
    if path.exists():
        arrays = dict(np.load(path, allow_pickle=False))
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Question IDs changed")
        if arrays["gla_layer_indices"].astype(int).tolist() != gla_layers:
            raise ValueError("GLA layer indices changed")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "gla_layer_indices": np.asarray(gla_layers, dtype=np.int16),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "donor_patched_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "identity_patched_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "donor_target_delta_norm": np.full(
            (2, n, len(gla_layers), len(TENSOR_NAMES)), np.nan, dtype=np.float32
        ),
        "target_write_norm": np.full(
            (2, n, len(gla_layers), len(TENSOR_NAMES)), np.nan, dtype=np.float32
        ),
        "donor_write_norm": np.full(
            (2, n, len(gla_layers), len(TENSOR_NAMES)), np.nan, dtype=np.float32
        ),
    }


def _batch(
    config: ExperimentConfig,
    processor: Any,
    tokenizer: Any,
    questions: dict[str, dict[str, Any]],
    group_qids: list[str],
    second_plan: dict[str, dict[str, Any]],
    condition: str,
    first_mapping: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    prompts: list[str] = []
    messages: list[list[dict[str, str]]] = []
    unpadded_spans: list[list[int]] = []
    token_rows: list[list[int]] = []
    first_questions: list[dict[str, Any]] = []
    for qid in group_qids:
        first_question = questions[qid]
        if first_mapping is not None:
            first_question = _remap_question(
                questions[qid], first_mapping[qid]["new_to_original"]
            )
        second_question = _remap_question(
            questions[qid], second_plan[qid]["new_to_original"]
        )
        row_messages = _messages(config, first_question, second_question, condition)
        prompt = render_chat(
            processor,
            row_messages,
            config.disable_thinking,
            config.chat_serialization,
        )
        ids = [
            int(value)
            for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ]
        spans = _source_positions(tokenizer, prompt, row_messages, first_question)
        prompts.append(prompt)
        messages.append(row_messages)
        token_rows.append(ids)
        unpadded_spans.append(spans["first_answer_boundary"])
        first_questions.append(first_question)

    input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
    width = int(input_ids.shape[1])
    physical_spans: list[list[int]] = []
    boundary_token_ids: list[list[int]] = []
    for row, (ids, span) in enumerate(zip(token_rows, unpadded_spans)):
        left_pad = width - len(ids)
        if input_ids[row, left_pad:].tolist() != ids:
            raise RuntimeError("Historical cohort tokenization changed a prompt")
        physical = [left_pad + position for position in span]
        physical_spans.append(physical)
        boundary_token_ids.append([ids[position] for position in span])
    return {
        "prompts": prompts,
        "messages": messages,
        "first_questions": first_questions,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "spans": physical_spans,
        "boundary_token_ids": boundary_token_ids,
    }


class GLAWriteCollector:
    """Capture the tensors that determine selected recurrent GLA writes."""

    def __init__(
        self,
        parts: Any,
        gla_layers: list[int],
        positions_by_row: dict[int, list[int]],
    ) -> None:
        self.values: dict[int, dict[int, dict[str, np.ndarray]]] = {}
        self.originals: list[tuple[Any, Any]] = []
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
                if "g" not in kwargs or "beta" not in kwargs:
                    raise RuntimeError("Qwen Gated DeltaNet did not pass g and beta")
                layer_values: dict[int, dict[str, np.ndarray]] = {}
                for row, positions in positions_by_row.items():
                    selected = list(positions)
                    layer_values[int(row)] = {
                        "key": key[row, selected].detach().float().cpu().numpy(),
                        "value": value[row, selected].detach().float().cpu().numpy(),
                        "g": kwargs["g"][row, selected].detach().float().cpu().numpy(),
                        "beta": kwargs["beta"][row, selected].detach().float().cpu().numpy(),
                    }
                self.values[int(_layer_index)] = layer_values
                return _original(query, key, value, *args, **kwargs)

            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped

    def close(self) -> None:
        for module, original in reversed(getattr(self, "originals", [])):
            module.chunk_gated_delta_rule = original
        self.originals = []


class CachedGLAWritePatcher:
    """Replace selected GLA memory writes with tensors captured in another pass."""

    def __init__(
        self,
        parts: Any,
        cache: dict[int, dict[int, dict[str, np.ndarray]]],
        positions_by_row: dict[int, list[int]],
    ) -> None:
        self.originals: list[tuple[Any, Any]] = []
        for layer_index, row_cache in cache.items():
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
                _row_cache=row_cache,
                **kwargs: Any,
            ):
                import torch

                if "g" not in kwargs or "beta" not in kwargs:
                    raise RuntimeError("Qwen Gated DeltaNet did not pass g and beta")
                patched_key = key.clone()
                patched_value = value.clone()
                patched_g = kwargs["g"].clone()
                patched_beta = kwargs["beta"].clone()
                for row, values in _row_cache.items():
                    selected = list(positions_by_row[int(row)])
                    if len(selected) != values["key"].shape[0]:
                        raise RuntimeError("Donor and target boundary spans differ")
                    for name, tensor in (
                        ("key", patched_key),
                        ("value", patched_value),
                        ("g", patched_g),
                        ("beta", patched_beta),
                    ):
                        source = torch.from_numpy(values[name]).to(
                            device=tensor.device, dtype=tensor.dtype
                        )
                        tensor[int(row), selected] = source
                kwargs["g"] = patched_g
                kwargs["beta"] = patched_beta
                return _original(query, patched_key, patched_value, *args, **kwargs)

            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped

    def close(self) -> None:
        for module, original in reversed(getattr(self, "originals", [])):
            module.chunk_gated_delta_rule = original
        self.originals = []


def _assemble_cache(
    candidate_caches: dict[int, dict[int, dict[int, dict[str, np.ndarray]]]],
    donor_rows: dict[str, dict[str, Any]],
    group_qids: list[str],
    targets: list[str],
    gla_layers: list[int],
) -> dict[int, dict[int, dict[str, np.ndarray]]]:
    output: dict[int, dict[int, dict[str, np.ndarray]]] = {
        layer: {} for layer in gla_layers
    }
    for qid in targets:
        row = group_qids.index(qid)
        mapping_index = int(donor_rows[qid]["donor"]["mapping_index"])
        for layer in gla_layers:
            output[layer][row] = candidate_caches[mapping_index][layer][row]
    return output


def _norms(
    target: dict[int, dict[int, dict[str, np.ndarray]]],
    donor: dict[int, dict[int, dict[str, np.ndarray]]],
    row: int,
    gla_layers: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = np.empty((len(gla_layers), len(TENSOR_NAMES)), dtype=np.float32)
    target_norm = np.empty_like(delta)
    donor_norm = np.empty_like(delta)
    for li, layer in enumerate(gla_layers):
        for ti, name in enumerate(TENSOR_NAMES):
            target_value = target[layer][row][name].astype(np.float64)
            donor_value = donor[layer][row][name].astype(np.float64)
            delta[li, ti] = np.linalg.norm(donor_value - target_value)
            target_norm[li, ti] = np.linalg.norm(target_value)
            donor_norm[li, ti] = np.linalg.norm(donor_value)
    return delta, target_norm, donor_norm


def run(
    config_path: Path,
    plan_path: Path,
    donor_plan_path: Path,
    second_mapping_plan_path: Path,
    mapping_plan_paths: list[Path],
    output: Path,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires the historical batch-size-4 SDPA regime")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")

    qids = _question_ids(plan_path)
    qid_set = set(qids)
    donor_payload = json.loads(donor_plan_path.read_text())
    donor_rows = {row["question_id"]: row for row in donor_payload["rows"]}
    if not qid_set <= set(donor_rows):
        raise ValueError("Donor plan does not cover the requested split")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    all_qids = [row["id"] for row in manifest["questions"]]
    second_plan = {
        row["question_id"]: row
        for row in json.loads(second_mapping_plan_path.read_text())["rows"]
    }
    mapping_plans = _load_mapping_plans(mapping_plan_paths, all_qids)

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    gla_layers = [
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if len(gla_layers) != 48:
        raise RuntimeError(f"Expected 48 GLA layers, found {len(gla_layers)}")

    output.mkdir(parents=True, exist_ok=True)
    frozen_rows = [donor_rows[qid] for qid in qids]
    (output / "donor_plan.json").write_text(
        json.dumps(
            {
                "question_ids": qids,
                "source_donor_plan": str(donor_plan_path),
                "n_primary": int(
                    sum(
                        row["primary_letter_decoupled_changed_winner"]
                        for row in frozen_rows
                    )
                ),
                "rows": frozen_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    result_path = output / "results.npz"
    arrays = _initialize(result_path, qids, gla_layers)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = output / "prompt_audit.json"

    for group_start in range(0, len(all_qids), config.batch_size):
        group_qids = all_qids[group_start : group_start + config.batch_size]
        targets = [
            qid
            for qid in group_qids
            if qid in qid_set and not bool(arrays["completed"][qid_index[qid]])
        ]
        if not targets:
            continue
        target_rows = {group_qids.index(qid): None for qid in targets}

        needed_mappings = sorted(
            {int(donor_rows[qid]["donor"]["mapping_index"]) for qid in targets}
        )
        candidate_caches: dict[
            int, dict[int, dict[int, dict[str, np.ndarray]]]
        ] = {}
        candidate_batches: dict[int, dict[str, Any]] = {}
        for mapping_index in needed_mappings:
            donor_batch = _batch(
                config,
                processor,
                tokenizer,
                questions,
                group_qids,
                second_plan,
                "incorrect",
                mapping_plans[mapping_index - 1],
            )
            collector = GLAWriteCollector(
                parts, gla_layers, {row: donor_batch["spans"][row] for row in target_rows}
            )
            try:
                _forward(
                    model,
                    parts,
                    donor_batch["input_ids"],
                    donor_batch["attention_mask"],
                )
            finally:
                collector.close()
            candidate_caches[mapping_index] = collector.values
            candidate_batches[mapping_index] = donor_batch

        donor_cache = _assemble_cache(
            candidate_caches, donor_rows, group_qids, targets, gla_layers
        )

        for condition_index, condition in enumerate(CONDITIONS):
            target_batch = _batch(
                config,
                processor,
                tokenizer,
                questions,
                group_qids,
                second_plan,
                condition,
            )
            positions_by_row = {
                row: target_batch["spans"][row] for row in target_rows
            }
            for qid in targets:
                row = group_qids.index(qid)
                mapping_index = int(donor_rows[qid]["donor"]["mapping_index"])
                donor_tokens = candidate_batches[mapping_index]["boundary_token_ids"][row]
                target_tokens = target_batch["boundary_token_ids"][row]
                if donor_tokens != target_tokens:
                    raise RuntimeError(f"{qid}: donor/target boundary tokens differ")

            natural_collector = GLAWriteCollector(parts, gla_layers, positions_by_row)
            try:
                natural_output = _forward(
                    model,
                    parts,
                    target_batch["input_ids"],
                    target_batch["attention_mask"],
                )
            finally:
                natural_collector.close()
            natural_logits = _aggregate_logits(
                natural_output.logits[:, -1].float(), variant_ids
            ).detach().cpu().numpy()

            donor_patcher = CachedGLAWritePatcher(
                parts, donor_cache, positions_by_row
            )
            try:
                donor_output = _forward(
                    model,
                    parts,
                    target_batch["input_ids"],
                    target_batch["attention_mask"],
                )
            finally:
                donor_patcher.close()
            donor_logits = _aggregate_logits(
                donor_output.logits[:, -1].float(), variant_ids
            ).detach().cpu().numpy()

            identity_cache = {
                layer: {
                    row: natural_collector.values[layer][row]
                    for row in target_rows
                }
                for layer in gla_layers
            }
            identity_patcher = CachedGLAWritePatcher(
                parts, identity_cache, positions_by_row
            )
            try:
                identity_output = _forward(
                    model,
                    parts,
                    target_batch["input_ids"],
                    target_batch["attention_mask"],
                )
            finally:
                identity_patcher.close()
            identity_logits = _aggregate_logits(
                identity_output.logits[:, -1].float(), variant_ids
            ).detach().cpu().numpy()

            for qid in targets:
                qi = qid_index[qid]
                row = group_qids.index(qid)
                arrays["natural_logits"][condition_index, qi] = natural_logits[row]
                arrays["donor_patched_logits"][condition_index, qi] = donor_logits[row]
                arrays["identity_patched_logits"][condition_index, qi] = identity_logits[row]
                delta, target_norm, donor_norm = _norms(
                    natural_collector.values, donor_cache, row, gla_layers
                )
                arrays["donor_target_delta_norm"][condition_index, qi] = delta
                arrays["target_write_norm"][condition_index, qi] = target_norm
                arrays["donor_write_norm"][condition_index, qi] = donor_norm

                if not audit_path.exists():
                    mapping_index = int(donor_rows[qid]["donor"]["mapping_index"])
                    donor_batch = candidate_batches[mapping_index]
                    audit_path.write_text(
                        json.dumps(
                            {
                                "question_id": qid,
                                "condition": condition,
                                "historical_group_qids": group_qids,
                                "target_row": row,
                                "donor": donor_rows[qid],
                                "target_prompt_hash": prompt_hash(target_batch["prompts"][row]),
                                "donor_prompt_hash": prompt_hash(donor_batch["prompts"][row]),
                                "boundary_positions_target": target_batch["spans"][row],
                                "boundary_positions_donor": donor_batch["spans"][row],
                                "boundary_tokens": tokenizer.convert_ids_to_tokens(
                                    target_batch["boundary_token_ids"][row]
                                ),
                                "target_prompt": target_batch["prompts"][row],
                                "donor_prompt": donor_batch["prompts"][row],
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    )

        for qid in targets:
            arrays["completed"][qid_index[qid]] = True
        atomic_save_npz(result_path, **arrays)
        done = int(arrays["completed"].sum())
        if done == len(targets) or done % 5 == 0 or done == len(qids):
            print(f"first-boundary GLA memory rewrite: {done}/{len(qids)}", flush=True)

    metadata = {
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "donor_plan_path": str(donor_plan_path),
        "second_mapping_plan_path": str(second_mapping_plan_path),
        "mapping_plan_paths": [str(path) for path in mapping_plan_paths],
        "n_questions": len(qids),
        "n_primary_letter_decoupled_changed_winner": int(
            sum(
                donor_rows[qid]["primary_letter_decoupled_changed_winner"]
                for qid in qids
            )
        ),
        "gla_layer_indices_zero_based": gla_layers,
        "source_span": (
            "The same first-answer-boundary token span used by the validated GLA "
            "source-write lesion."
        ),
        "intervention": (
            "At all 48 GLA layers, replace the target's key, value, decay gate g, "
            "and write strength beta over the first-answer-boundary span with tensors "
            "captured from the same question under a frozen alternative first-option "
            "mapping whose live Baseline winner is a different semantic answer."
        ),
        "prediction": (
            "Game should disfavor the transplanted donor answer, while Neutral should "
            "favor or repeat it; effects should follow semantic content rather than "
            "the donor's old literal answer letter."
        ),
        "numerical_control": (
            "Natural and patched targets preserve the exact historical four-question "
            "cohort, row, padding, batch size 4, SDPA implementation, and software."
        ),
        "resolved_answer_tokens": resolved,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--donor-plan", type=Path, required=True)
    parser.add_argument("--second-mapping-plan", type=Path, required=True)
    parser.add_argument("--mapping-plans", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.config,
        args.plan,
        args.donor_plan,
        args.second_mapping_plan,
        args.mapping_plans,
        args.output,
    )


if __name__ == "__main__":
    main()
