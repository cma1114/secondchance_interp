from __future__ import annotations

import argparse
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
    resolve_answer_tokens,
)
from .prompts import prompt_hash
from .run_first_boundary_gla_memory_rewrite import (
    CONDITIONS,
    LETTERS,
    _aggregate_logits,
    _batch,
    _load_mapping_plans,
    _question_ids,
)
from .run_historical_answer_intervention import _forward


def _choose_same_winner(row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        candidate
        for candidate in row["all_candidate_winners"]
        if not candidate["changed_winner"] and candidate["letter_decoupled"]
    ]
    if not candidates:
        candidates = [
            candidate
            for candidate in row["all_candidate_winners"]
            if not candidate["changed_winner"]
        ]
    return max(candidates, key=lambda candidate: candidate["margin"]) if candidates else None


def _initialize(path: Path, qids: list[str], gla_layers: list[int]) -> dict[str, np.ndarray]:
    if path.exists():
        arrays = dict(np.load(path, allow_pickle=False))
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Question IDs changed")
        if arrays["gla_layer_indices"].astype(int).tolist() != gla_layers:
            raise ValueError("GLA layer indices changed")
        return arrays
    n = len(qids)
    l = len(gla_layers)
    return {
        "question_ids": np.asarray(qids),
        "gla_layer_indices": np.asarray(gla_layers, dtype=np.int16),
        "completed": np.zeros(n, dtype=bool),
        "has_same_winner_control": np.zeros(n, dtype=bool),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "identity_state_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "different_winner_state_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_winner_state_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "different_state_delta_norm": np.full((n, l), np.nan, dtype=np.float32),
        "same_state_delta_norm": np.full((n, l), np.nan, dtype=np.float32),
        "recipient_state_norm": np.full((n, l), np.nan, dtype=np.float32),
    }


class GLAStateCollector:
    """Capture the accumulated recurrent GLA state just after a row-specific boundary."""

    def __init__(
        self,
        parts: Any,
        gla_layers: list[int],
        positions_by_row: dict[int, list[int]],
    ) -> None:
        self.values: dict[int, dict[int, np.ndarray]] = {}
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
                if kwargs.get("initial_state") is not None:
                    raise RuntimeError("Collector expects an uncached full-prompt pass")
                if "g" not in kwargs or "beta" not in kwargs:
                    raise RuntimeError("Qwen Gated DeltaNet did not pass g and beta")
                layer_values: dict[int, np.ndarray] = {}
                for row, positions in positions_by_row.items():
                    cut = max(positions) + 1
                    prefix_kwargs = dict(kwargs)
                    prefix_kwargs["g"] = kwargs["g"][row : row + 1, :cut]
                    prefix_kwargs["beta"] = kwargs["beta"][row : row + 1, :cut]
                    prefix_kwargs["initial_state"] = None
                    prefix_kwargs["output_final_state"] = True
                    prefix_kwargs["cu_seqlens"] = None
                    _, state = _original(
                        query[row : row + 1, :cut],
                        key[row : row + 1, :cut],
                        value[row : row + 1, :cut],
                        *args,
                        **prefix_kwargs,
                    )
                    if state is None:
                        raise RuntimeError("GLA kernel did not return a recurrent state")
                    layer_values[int(row)] = state.detach().float().cpu().numpy()
                self.values[int(_layer_index)] = layer_values
                return _original(query, key, value, *args, **kwargs)

            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped

    def close(self) -> None:
        for module, original in reversed(getattr(self, "originals", [])):
            module.chunk_gated_delta_rule = original
        self.originals = []


class CachedGLAStatePatcher:
    """Resume each GLA after the first-answer boundary from a cached donor state."""

    def __init__(
        self,
        parts: Any,
        cache: dict[int, dict[int, np.ndarray]],
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

                if kwargs.get("initial_state") is not None:
                    raise RuntimeError("Patcher expects an uncached full-prompt pass")
                outputs = []
                final_states = []
                for row in range(query.shape[0]):
                    if row not in _row_cache:
                        row_kwargs = dict(kwargs)
                        row_kwargs["g"] = kwargs["g"][row : row + 1]
                        row_kwargs["beta"] = kwargs["beta"][row : row + 1]
                        row_kwargs["cu_seqlens"] = None
                        out, state = _original(
                            query[row : row + 1], key[row : row + 1], value[row : row + 1],
                            *args, **row_kwargs,
                        )
                        outputs.append(out)
                        if state is not None:
                            final_states.append(state)
                        continue
                    cut = max(positions_by_row[row]) + 1
                    prefix_kwargs = dict(kwargs)
                    prefix_kwargs["g"] = kwargs["g"][row : row + 1, :cut]
                    prefix_kwargs["beta"] = kwargs["beta"][row : row + 1, :cut]
                    prefix_kwargs["initial_state"] = None
                    prefix_kwargs["output_final_state"] = False
                    prefix_kwargs["cu_seqlens"] = None
                    prefix_out, _ = _original(
                        query[row : row + 1, :cut], key[row : row + 1, :cut],
                        value[row : row + 1, :cut], *args, **prefix_kwargs,
                    )
                    suffix_kwargs = dict(kwargs)
                    suffix_kwargs["g"] = kwargs["g"][row : row + 1, cut:]
                    suffix_kwargs["beta"] = kwargs["beta"][row : row + 1, cut:]
                    suffix_kwargs["initial_state"] = torch.from_numpy(_row_cache[row]).to(
                        device=query.device, dtype=torch.float32
                    )
                    suffix_kwargs["cu_seqlens"] = None
                    suffix_out, state = _original(
                        query[row : row + 1, cut:], key[row : row + 1, cut:],
                        value[row : row + 1, cut:], *args, **suffix_kwargs,
                    )
                    outputs.append(torch.cat([prefix_out, suffix_out], dim=1))
                    if state is not None:
                        final_states.append(state)
                output = torch.cat(outputs, dim=0)
                final_state = torch.cat(final_states, dim=0) if final_states else None
                return output, final_state

            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped

    def close(self) -> None:
        for module, original in reversed(getattr(self, "originals", [])):
            module.chunk_gated_delta_rule = original
        self.originals = []


def _assemble_cache(
    candidate_caches: dict[int, dict[int, dict[int, np.ndarray]]],
    rows: dict[str, dict[str, Any]],
    group_qids: list[str],
    targets: list[str],
    gla_layers: list[int],
    donor_key: str,
) -> dict[int, dict[int, np.ndarray]]:
    output = {layer: {} for layer in gla_layers}
    for qid in targets:
        donor = rows[qid][donor_key]
        if donor is None:
            continue
        row = group_qids.index(qid)
        mapping_index = int(donor["mapping_index"])
        for layer in gla_layers:
            output[layer][row] = candidate_caches[mapping_index][layer][row]
    return output


def _state_norms(
    recipient: dict[int, dict[int, np.ndarray]],
    donor: dict[int, dict[int, np.ndarray]],
    row: int,
    gla_layers: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.full(len(gla_layers), np.nan, dtype=np.float32)
    norm = np.full(len(gla_layers), np.nan, dtype=np.float32)
    for li, layer in enumerate(gla_layers):
        target = recipient[layer][row].astype(np.float64)
        norm[li] = np.linalg.norm(target)
        if row in donor[layer]:
            delta[li] = np.linalg.norm(donor[layer][row].astype(np.float64) - target)
    return delta, norm


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
    source_rows = {
        row["question_id"]: row
        for row in json.loads(donor_plan_path.read_text())["rows"]
    }
    rows = {}
    for qid in qids:
        row = dict(source_rows[qid])
        row["different_winner_donor"] = row["donor"]
        row["same_winner_donor"] = _choose_same_winner(row)
        rows[qid] = row

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
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if len(gla_layers) != 48:
        raise RuntimeError(f"Expected 48 GLA layers, found {len(gla_layers)}")

    output.mkdir(parents=True, exist_ok=True)
    frozen_rows = [rows[qid] for qid in qids]
    (output / "donor_plan.json").write_text(json.dumps({
        "question_ids": qids,
        "source_donor_plan": str(donor_plan_path),
        "n_different_winner_primary": int(sum(
            row["primary_letter_decoupled_changed_winner"] for row in frozen_rows
        )),
        "n_same_winner_control": int(sum(
            row["same_winner_donor"] is not None for row in frozen_rows
        )),
        "rows": frozen_rows,
    }, indent=2, sort_keys=True) + "\n")
    result_path = output / "results.npz"
    arrays = _initialize(result_path, qids, gla_layers)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = output / "prompt_audit.json"

    for group_start in range(0, len(all_qids), config.batch_size):
        group_qids = all_qids[group_start : group_start + config.batch_size]
        targets = [
            qid for qid in group_qids
            if qid in qid_set and not bool(arrays["completed"][qid_index[qid]])
        ]
        if not targets:
            continue
        target_rows = {group_qids.index(qid): None for qid in targets}
        needed_mappings = sorted({
            int(donor["mapping_index"])
            for qid in targets
            for donor in (rows[qid]["different_winner_donor"], rows[qid]["same_winner_donor"])
            if donor is not None
        })
        candidate_caches: dict[int, dict[int, dict[int, np.ndarray]]] = {}
        candidate_batches: dict[int, dict[str, Any]] = {}
        for mapping_index in needed_mappings:
            donor_batch = _batch(
                config, processor, tokenizer, questions, group_qids, second_plan,
                "incorrect", mapping_plans[mapping_index - 1],
            )
            collector = GLAStateCollector(
                parts, gla_layers,
                {row: donor_batch["spans"][row] for row in target_rows},
            )
            try:
                _forward(model, parts, donor_batch["input_ids"], donor_batch["attention_mask"])
            finally:
                collector.close()
            candidate_caches[mapping_index] = collector.values
            candidate_batches[mapping_index] = donor_batch

        different_cache = _assemble_cache(
            candidate_caches, rows, group_qids, targets, gla_layers,
            "different_winner_donor",
        )
        same_cache = _assemble_cache(
            candidate_caches, rows, group_qids, targets, gla_layers,
            "same_winner_donor",
        )

        for condition_index, condition in enumerate(CONDITIONS):
            target_batch = _batch(
                config, processor, tokenizer, questions, group_qids, second_plan, condition,
            )
            positions_by_row = {row: target_batch["spans"][row] for row in target_rows}
            recipient_collector = GLAStateCollector(parts, gla_layers, positions_by_row)
            try:
                natural_output = _forward(
                    model, parts, target_batch["input_ids"], target_batch["attention_mask"]
                )
            finally:
                recipient_collector.close()
            natural_logits = _aggregate_logits(
                natural_output.logits[:, -1].float(), variant_ids
            ).detach().cpu().numpy()

            scenarios = {
                "identity_state_logits": recipient_collector.values,
                "different_winner_state_logits": different_cache,
                "same_winner_state_logits": same_cache,
            }
            scenario_logits = {}
            for name, cache in scenarios.items():
                patcher = CachedGLAStatePatcher(parts, cache, positions_by_row)
                try:
                    patched_output = _forward(
                        model, parts, target_batch["input_ids"], target_batch["attention_mask"]
                    )
                finally:
                    patcher.close()
                scenario_logits[name] = _aggregate_logits(
                    patched_output.logits[:, -1].float(), variant_ids
                ).detach().cpu().numpy()

            for qid in targets:
                qi = qid_index[qid]
                row = group_qids.index(qid)
                arrays["natural_logits"][condition_index, qi] = natural_logits[row]
                for name, values in scenario_logits.items():
                    arrays[name][condition_index, qi] = values[row]
                arrays["has_same_winner_control"][qi] = rows[qid]["same_winner_donor"] is not None
                if condition_index == 0:
                    diff_delta, recipient_norm = _state_norms(
                        recipient_collector.values, different_cache, row, gla_layers
                    )
                    same_delta, _ = _state_norms(
                        recipient_collector.values, same_cache, row, gla_layers
                    )
                    arrays["different_state_delta_norm"][qi] = diff_delta
                    arrays["same_state_delta_norm"][qi] = same_delta
                    arrays["recipient_state_norm"][qi] = recipient_norm

                if not audit_path.exists():
                    mapping_index = int(rows[qid]["different_winner_donor"]["mapping_index"])
                    audit_path.write_text(json.dumps({
                        "question_id": qid,
                        "condition": condition,
                        "historical_group_qids": group_qids,
                        "target_row": row,
                        "different_winner_donor": rows[qid]["different_winner_donor"],
                        "same_winner_donor": rows[qid]["same_winner_donor"],
                        "target_prompt_hash": prompt_hash(target_batch["prompts"][row]),
                        "donor_prompt_hash": prompt_hash(candidate_batches[mapping_index]["prompts"][row]),
                        "boundary_positions": target_batch["spans"][row],
                        "boundary_tokens": tokenizer.convert_ids_to_tokens(
                            target_batch["boundary_token_ids"][row]
                        ),
                        "target_prompt": target_batch["prompts"][row],
                        "donor_prompt": candidate_batches[mapping_index]["prompts"][row],
                    }, indent=2, sort_keys=True) + "\n")

        for qid in targets:
            arrays["completed"][qid_index[qid]] = True
        atomic_save_npz(result_path, **arrays)
        done = int(arrays["completed"].sum())
        if done % 5 == 0 or done == len(qids):
            print(f"first-boundary GLA state transplant: {done}/{len(qids)}", flush=True)

    metadata = {
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "donor_plan_path": str(donor_plan_path),
        "second_mapping_plan_path": str(second_mapping_plan_path),
        "mapping_plan_paths": [str(path) for path in mapping_plan_paths],
        "n_questions": len(qids),
        "gla_layer_indices_zero_based": gla_layers,
        "intervention": (
            "At all 48 GLA layers, replace the accumulated recurrent matrix state "
            "immediately after the first-answer boundary, leaving the recipient's "
            "visible prompt and all post-boundary tokens fixed."
        ),
        "controls": (
            "Recipient-state reinsertion measures segmented-kernel numerical effects; "
            "same-semantic-winner/different-mapping transplantation controls for generic "
            "first-presentation and mapping state."
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
        args.config, args.plan, args.donor_plan, args.second_mapping_plan,
        args.mapping_plans, args.output,
    )


if __name__ == "__main__":
    main()
