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

from . import LETTERS
from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSDPAQuerySourceAttentionAblator
from .io import atomic_save_npz
from .modeling import get_tokenizer, load_model_and_processor, resolve_answer_tokens
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_fixed_a_final_query_edge_ablation import _option_line_positions


ORDINARY_BLOCKS = tuple(range(4, 49, 4))
RANKS = ("W1", "W2", "W3", "W4")


class AllCandidateWriteCollector:
    """Collect compact natural matching-edge metrics for all four candidates."""

    def __init__(
        self,
        parts: Any,
        layer_indices: tuple[int, ...],
        source_positions: list[list[list[int]]],
        query_positions: list[list[list[int]]],
    ) -> None:
        import torch

        self.parts = parts
        self.layer_indices = layer_indices
        self.source_positions = source_positions
        self.query_positions = query_positions
        self.layers = {index: parts.layers[index].self_attn for index in layer_indices}
        self.active: int | None = None
        self.gates: dict[int, Any] = {}
        self.metrics: dict[int, dict[str, Any]] = {}
        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        self.handles: list[Any] = []
        for layer_index, attention in self.layers.items():
            self.handles.extend(
                [
                    attention.register_forward_pre_hook(self._enter(layer_index)),
                    attention.register_forward_hook(self._leave(layer_index)),
                    attention.q_proj.register_forward_hook(self._gate_hook(layer_index)),
                ]
            )
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

    def _gate_hook(self, layer_index: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            head_dim = int(self.layers[layer_index].head_dim)
            shaped = output.view(*output.shape[:-1], -1, 2 * head_dim)
            _query, gate = shaped.chunk(2, dim=-1)
            self.gates[layer_index] = gate.sigmoid().permute(0, 2, 1, 3)

        return capture

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        import torch
        import torch.nn.functional as F

        result = self.original_sdpa(query, key, value, *args, **kwargs)
        layer_index = self.active
        if layer_index not in self.layers:
            return result
        gate = self.gates.get(layer_index)
        if gate is None:
            raise RuntimeError("Qwen output gate was not captured before SDPA")

        key_length = int(key.shape[-2])
        head_dim = int(value.shape[-1])
        weights = torch.empty(
            query.shape[0], query.shape[1], query.shape[2], key_length,
            device=query.device, dtype=value.dtype,
        )
        for start in range(0, key_length, head_dim):
            end = min(start + head_dim, key_length)
            synthetic = torch.zeros_like(value)
            positions = torch.arange(start, end, device=value.device)
            dimensions = torch.arange(end - start, device=value.device)
            synthetic[:, :, positions, dimensions] = 1
            recovered = self.original_sdpa(query, key, synthetic, *args, **kwargs)
            weights[..., start:end] = recovered[..., : end - start]

        if value.shape[1] != query.shape[1]:
            if query.shape[1] % value.shape[1]:
                raise RuntimeError("Query heads are not divisible by KV heads")
            value_for_heads = value.repeat_interleave(
                query.shape[1] // value.shape[1], dim=1
            )
        else:
            value_for_heads = value

        batch = int(query.shape[0])
        attention_mass = torch.zeros(batch, 4, device=query.device, dtype=torch.float32)
        context_norm = torch.zeros_like(attention_mass)
        projected_write_norm = torch.zeros_like(attention_mass)
        mean_gate = torch.zeros_like(attention_mass)
        projection = self.layers[layer_index].o_proj
        for row in range(batch):
            for rank in range(4):
                sources = self.source_positions[row][rank]
                queries = self.query_positions[row][rank]
                if not sources or not queries:
                    raise RuntimeError("Empty candidate source or query line")
                source_index = torch.as_tensor(sources, device=weights.device)
                query_index = torch.as_tensor(queries, device=weights.device)
                local_weights = weights[row].index_select(1, query_index).index_select(
                    2, source_index
                )
                attention_mass[row, rank] = local_weights.float().sum(-1).mean()
                local_values = value_for_heads[row].index_select(1, source_index)
                context = torch.einsum("hqk,hkd->hqd", local_weights, local_values)
                context_norm[row, rank] = context.float().norm(dim=-1).mean()
                local_gate = gate[row].index_select(1, query_index)
                mean_gate[row, rank] = local_gate.float().mean()
                gated = context * local_gate
                flat = gated.permute(1, 0, 2).flatten(1).float()
                write = F.linear(flat, projection.weight.float(), None)
                projected_write_norm[row, rank] = write.mean(0).norm()
        self.metrics[layer_index] = {
            "attention_mass": attention_mass.detach().to("cpu"),
            "context_norm": context_norm.detach().to("cpu"),
            "projected_write_norm": projected_write_norm.detach().to("cpu"),
            "mean_gate": mean_gate.detach().to("cpu"),
        }
        return result

    def close(self) -> None:
        import torch

        torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.active = None

    def arrays(self) -> dict[str, np.ndarray]:
        import torch

        missing = [index for index in self.layer_indices if index not in self.metrics]
        if missing:
            raise RuntimeError(f"Missing source-write metrics at layers {missing}")
        return {
            name: torch.stack(
                [self.metrics[index][name] for index in self.layer_indices], dim=0
            ).to("cpu").numpy()
            for name in self.metrics[self.layer_indices[0]]
        }


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses a different question order")
        return arrays
    n = len(qids)
    layer_count = len(ORDINARY_BLOCKS)
    arrays: dict[str, np.ndarray] = {
        "question_ids": np.asarray(qids),
        "completed": np.zeros(n, dtype=bool),
        "rank_contents": np.full((n, 4), "", dtype="<U1"),
        "baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "matched_logits": np.full((2, 4, n, 4), np.nan, dtype=np.float32),
        "control_logits": np.full((2, 4, n, 4), np.nan, dtype=np.float32),
        "joint_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "source_position_counts": np.zeros((n, 4), dtype=np.int16),
        "control_position_counts": np.zeros((n, 4), dtype=np.int16),
        "query_position_counts": np.zeros((n, 4), dtype=np.int16),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
    }
    for name in ("attention_mass", "context_norm", "projected_write_norm", "mean_gate"):
        arrays[name] = np.full((2, layer_count, n, 4), np.nan, dtype=np.float32)
    return arrays


def _specs(
    layer_indices: tuple[int, ...],
    source_positions: list[list[list[int]]],
    query_positions: list[list[list[int]]],
    ranks: tuple[int, ...],
    controls: bool,
) -> dict[int, dict[int, dict[int, list[int]]]]:
    specs: dict[int, dict[int, dict[int, list[int]]]] = {}
    for layer_index in layer_indices:
        rows: dict[int, dict[int, list[int]]] = {}
        for row in range(len(source_positions)):
            row_specs: dict[int, list[int]] = {}
            for rank in ranks:
                source_rank = (rank + 1) % 4 if controls else rank
                sources = source_positions[row][source_rank]
                for query in query_positions[row][rank]:
                    row_specs[int(query)] = [int(value) for value in sources]
            rows[row] = row_specs
        specs[layer_index] = rows
    return specs


def run(
    config_path: Path,
    remapping_plan_path: Path,
    baseline_path: Path,
    trusted_game_path: Path,
    trusted_neutral_path: Path,
    output_dir: Path,
    max_cohorts: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires exact historical batch-size-4 SDPA execution")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires action-matched incorrect/lost feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw_qwen_chatml serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    if max_cohorts is not None:
        qids = qids[: int(max_cohorts) * config.batch_size]
    mappings = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    baseline = json.loads(baseline_path.read_text())["results"]
    trusted = [
        json.loads(trusted_game_path.read_text())["results"],
        json.loads(trusted_neutral_path.read_text())["results"],
    ]

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    layer_indices = tuple(
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None and index + 1 <= 48
    )
    if tuple(index + 1 for index in layer_indices) != ORDINARY_BLOCKS:
        raise RuntimeError(f"Unexpected ordinary-attention layers: {layer_indices}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = output_dir / "prompt_audit.json"
    started = time.monotonic()
    durations: list[float] = []

    for start in range(0, len(qids), config.batch_size):
        cohort = qids[start : start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if np.all(arrays["completed"][indices]):
            continue
        cohort_started = time.monotonic()
        batches = [
            _build_batch(config, processor, tokenizer, questions, mappings, cohort, condition)
            for condition in CONDITIONS
        ]
        cohort_audit: dict[str, Any] = {"question_ids": cohort, "conditions": {}}

        for row, qid in enumerate(cohort):
            logits = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=float)
            order = np.argsort(-logits, kind="stable")
            qi = qid_index[qid]
            arrays["baseline_logits"][qi] = logits
            arrays["rank_contents"][qi] = np.asarray([LETTERS[int(i)] for i in order])

        for ci, (condition, batch) in enumerate(zip(CONDITIONS, batches)):
            width = int(batch["input_ids"].shape[1])
            source_physical: list[list[list[int]]] = []
            query_physical: list[list[list[int]]] = []
            control_physical: list[list[list[int]]] = []
            audits: list[dict[str, Any]] = []
            for row, qid in enumerate(cohort):
                qi = qid_index[qid]
                ids = batch["token_rows"][row]
                left_pad = width - len(ids)
                remapped_question = {
                    **questions[qid],
                    "options": {
                        new: questions[qid]["options"][old]
                        for new, old in mappings[qid]["new_to_original"].items()
                    },
                }
                first_positions, first_audit = _option_line_positions(
                    tokenizer, batch["prompts"][row], questions[qid]
                )
                second_positions, second_audit = _option_line_positions(
                    tokenizer, batch["prompts"][row], remapped_question
                )
                ranks = arrays["rank_contents"][qi].astype(str).tolist()
                row_sources: list[list[int]] = []
                row_queries: list[list[int]] = []
                for rank, content in enumerate(ranks):
                    second_letter = mappings[qid]["original_to_new"][content]
                    sources = [left_pad + value for value in first_positions[content]]
                    queries = [left_pad + value for value in second_positions[second_letter]]
                    if not sources or not queries or max(sources) >= min(queries):
                        raise RuntimeError("Invalid matching source/query option lines")
                    row_sources.append(sources)
                    row_queries.append(queries)
                    arrays["source_position_counts"][qi, rank] = len(sources)
                    arrays["query_position_counts"][qi, rank] = len(queries)
                source_physical.append(row_sources)
                query_physical.append(row_queries)
                control_physical.append([row_sources[(rank + 1) % 4] for rank in range(4)])
                for rank in range(4):
                    arrays["control_position_counts"][qi, rank] = len(
                        row_sources[(rank + 1) % 4]
                    )
                arrays["prompt_hashes"][ci, qi] = _hash_prompt(batch["prompts"][row])
                audits.append({"first": first_audit, "second": second_audit, "ranks": ranks})

            collector = AllCandidateWriteCollector(
                parts, layer_indices, source_physical, query_physical
            )
            try:
                natural_output = _forward(
                    model, parts, batch["input_ids"], batch["attention_mask"]
                )
                observational = collector.arrays()
            finally:
                collector.close()
            natural = _aggregate_logits(natural_output, variant_ids)
            if not np.all(np.isfinite(natural)):
                raise RuntimeError("Non-finite natural logits")
            arrays["natural_logits"][ci, indices] = natural
            for row, qid in enumerate(cohort):
                qi = qid_index[qid]
                arrays["trusted_natural_logits"][ci, qi] = np.asarray(
                    trusted[ci][qid]["aggregated_ad_logits"], dtype=np.float32
                )
                for name, values in observational.items():
                    arrays[name][ci, :, qi] = values[:, row]

            for rank in range(4):
                matched_specs = _specs(
                    layer_indices, source_physical, query_physical, (rank,), False
                )
                with BatchedSDPAQuerySourceAttentionAblator(parts, matched_specs):
                    output = _aggregate_logits(
                        _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                        variant_ids,
                    )
                if not np.all(np.isfinite(output)):
                    raise RuntimeError(f"Non-finite matched intervention for {RANKS[rank]}")
                arrays["matched_logits"][ci, rank, indices] = output

                control_specs = _specs(
                    layer_indices, source_physical, query_physical, (rank,), True
                )
                with BatchedSDPAQuerySourceAttentionAblator(parts, control_specs):
                    output = _aggregate_logits(
                        _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                        variant_ids,
                    )
                if not np.all(np.isfinite(output)):
                    raise RuntimeError(f"Non-finite control intervention for {RANKS[rank]}")
                arrays["control_logits"][ci, rank, indices] = output

            joint_specs = _specs(
                layer_indices, source_physical, query_physical, tuple(range(4)), False
            )
            with BatchedSDPAQuerySourceAttentionAblator(parts, joint_specs):
                joint = _aggregate_logits(
                    _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                    variant_ids,
                )
            if not np.all(np.isfinite(joint)):
                raise RuntimeError("Non-finite joint intervention")
            arrays["joint_logits"][ci, indices] = joint

            cohort_audit["conditions"][condition] = {
                "rendered_prompt": batch["prompts"][0],
                "prompt_hash": arrays["prompt_hashes"][ci, indices[0]].item(),
                "rank_contents": audits[0]["ranks"],
                "first_option_lines": audits[0]["first"],
                "second_option_lines": audits[0]["second"],
                "cyclic_control": "For target rank Wr, use the original source line at rank W(r+1), wrapping W4 to W1.",
            }

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"all-candidate relay: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.2f}",
            flush=True,
        )
        if not audit_path.exists():
            audit_path.write_text(
                json.dumps(cohort_audit, indent=2, ensure_ascii=False) + "\n"
            )

    metadata = {
        "experiment": "canonical remapped all-candidate matched semantic relay factorial",
        "config": config.as_dict(),
        "n_questions": len(qids),
        "conditions": list(CONDITIONS),
        "ranks": list(RANKS),
        "ordinary_blocks_one_based": list(ORDINARY_BLOCKS),
        "complete_model_forwards_per_cohort": 20,
        "complete_model_work": (
            "Per condition: one natural forward with compact layerwise source-write collection, "
            "four matching-edge lesions, four cyclic nonmatching-edge controls, and one joint matching-edge lesion."
        ),
        "observational_metrics": [
            "attention_mass", "context_norm", "projected_write_norm", "mean_gate"
        ],
        "elapsed_seconds_after_load": time.monotonic() - started,
        "cohort_seconds": durations,
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
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.remapping_plan,
        args.baseline,
        args.trusted_game,
        args.trusted_neutral,
        args.output_dir,
        args.max_cohorts,
    )


if __name__ == "__main__":
    main()
