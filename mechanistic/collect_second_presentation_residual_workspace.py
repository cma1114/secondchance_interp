from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor, model_input_device, resolve_answer_tokens
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_second_presentation_attention_distribution import (
    ORDINARY_LAYERS,
    SOURCE_BINS,
    _source_bins_and_queries,
)
from .sublayer import _hidden, mixer_module


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _select_ragged(hidden: Any, positions: list[list[int]]) -> tuple[Any, Any]:
    """Select variable row-wise positions into a padded native-dtype tensor."""
    import torch

    maximum = max(len(row) for row in positions)
    if maximum <= 0:
        raise ValueError("Cannot select an empty position collection")
    index = torch.zeros((len(positions), maximum), device=hidden.device, dtype=torch.long)
    mask = torch.zeros((len(positions), maximum), device=hidden.device, dtype=torch.bool)
    for row, values in enumerate(positions):
        if not values:
            continue
        index[row, : len(values)] = torch.as_tensor(values, device=hidden.device)
        mask[row, : len(values)] = True
    rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    selected = hidden[rows, index]
    broadcast_mask = mask.view(mask.shape + (1,) * (selected.ndim - 2))
    selected = selected.masked_fill(~broadcast_mask, 0)
    return selected, mask


class WorkspaceStateCollector:
    """Capture full residual readouts and component writes at ragged positions."""

    def __init__(
        self,
        parts: Any,
        residual_positions: list[list[int]],
        receiver_positions: list[list[int]],
    ) -> None:
        self.residual_positions = residual_positions
        self.receiver_positions = receiver_positions
        self.residuals: list[Any] = [None] * (len(parts.layers) + 1)
        self.mixer_outputs: list[Any] = [None] * len(parts.layers)
        self.mlp_outputs: list[Any] = [None] * len(parts.layers)
        self.residual_mask = None
        self.receiver_mask = None
        self.handles = [parts.embedding.register_forward_hook(self._residual_hook(0))]
        for index, layer in enumerate(parts.layers):
            self.handles.append(layer.register_forward_hook(self._residual_hook(index + 1)))
            self.handles.append(
                mixer_module(layer).register_forward_hook(self._component_hook(index, "mixer"))
            )
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                raise RuntimeError(f"Layer {index} has no MLP")
            self.handles.append(mlp.register_forward_hook(self._component_hook(index, "mlp")))

    @staticmethod
    def _cpu_native(tensor: Any) -> Any:
        # Qwen runs in bfloat16. Preserve those exact 16-bit values rather than
        # silently narrowing them through IEEE float16.
        return tensor.detach().to(device="cpu", non_blocking=False).contiguous()

    def _residual_hook(self, index: int) -> Callable:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            selected, mask = _select_ragged(_hidden(output), self.residual_positions)
            self.residuals[index] = self._cpu_native(selected)
            if self.residual_mask is None:
                self.residual_mask = mask.detach().cpu()
        return capture

    def _component_hook(self, index: int, kind: str) -> Callable:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            selected, mask = _select_ragged(_hidden(output), self.receiver_positions)
            target = self.mixer_outputs if kind == "mixer" else self.mlp_outputs
            target[index] = self._cpu_native(selected)
            if self.receiver_mask is None:
                self.receiver_mask = mask.detach().cpu()
        return capture

    def arrays(self) -> dict[str, Any]:
        import torch

        missing_residual = [i for i, value in enumerate(self.residuals) if value is None]
        missing_mixer = [i for i, value in enumerate(self.mixer_outputs) if value is None]
        missing_mlp = [i for i, value in enumerate(self.mlp_outputs) if value is None]
        if missing_residual or missing_mixer or missing_mlp:
            raise RuntimeError(
                f"Missing workspace hooks: residual={missing_residual}, "
                f"mixer={missing_mixer}, mlp={missing_mlp}"
            )
        return {
            "residuals": torch.stack(self.residuals, dim=1),
            "mixer_outputs": torch.stack(self.mixer_outputs, dim=1),
            "mlp_outputs": torch.stack(self.mlp_outputs, dim=1),
            "residual_mask": self.residual_mask,
            "receiver_mask": self.receiver_mask,
        }

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


class WorkspaceAttentionCollector:
    """Cache exact SDPA weights, unexpanded values, and query output gates."""

    def __init__(
        self,
        parts: Any,
        layer_indices: tuple[int, ...],
        receiver_positions: list[list[int]],
    ) -> None:
        import torch

        self.layer_indices = layer_indices
        self.receiver_positions = receiver_positions
        self.layers = {index: parts.layers[index].self_attn for index in layer_indices}
        self.active: int | None = None
        self.gates: dict[int, Any] = {}
        self.weights: dict[int, Any] = {}
        self.values: dict[int, Any] = {}
        self.query_masks: dict[int, Any] = {}
        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        self.handles: list[Any] = []
        for index, attention in self.layers.items():
            self.handles.extend(
                [
                    attention.register_forward_pre_hook(self._enter(index)),
                    attention.register_forward_hook(self._leave(index)),
                    attention.q_proj.register_forward_hook(self._gate_hook(index, attention)),
                ]
            )
        torch.nn.functional.scaled_dot_product_attention = self._wrapped_sdpa

    def _enter(self, index: int) -> Callable:
        def enter(_module: Any, _inputs: Any) -> None:
            if self.active is not None:
                raise RuntimeError("Nested ordinary-attention calls are unsupported")
            self.active = index
        return enter

    def _leave(self, index: int) -> Callable:
        def leave(_module: Any, _inputs: Any, _output: Any) -> None:
            if self.active != index:
                raise RuntimeError("Ordinary-attention layer stack became inconsistent")
            self.active = None
        return leave

    def _gate_hook(self, index: int, attention: Any) -> Callable:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            head_dim = int(attention.head_dim)
            shaped = output.view(*output.shape[:-1], -1, head_dim * 2)
            _query, gate = shaped.chunk(2, dim=-1)
            selected, mask = _select_ragged(gate.sigmoid(), self.receiver_positions)
            self.gates[index] = selected.detach().to("cpu").contiguous()
            self.query_masks[index] = mask.detach().cpu()
        return capture

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        import torch

        result = self.original_sdpa(query, key, value, *args, **kwargs)
        index = self.active
        if index not in self.layers:
            return result

        key_length = int(key.shape[-2])
        head_dim = int(value.shape[-1])
        recovered = torch.empty(
            query.shape[0], query.shape[1], query.shape[2], key_length,
            device=query.device, dtype=value.dtype,
        )
        # This reproduces the exact configured SDPA implementation rather than
        # assuming a particular masking, scaling, or GQA convention.
        for start in range(0, key_length, head_dim):
            stop = min(start + head_dim, key_length)
            synthetic = torch.zeros_like(value)
            positions = torch.arange(start, stop, device=value.device)
            dimensions = torch.arange(stop - start, device=value.device)
            synthetic[:, :, positions, dimensions] = 1
            encoded = self.original_sdpa(query, key, synthetic, *args, **kwargs)
            recovered[..., start:stop] = encoded[..., : stop - start]

        selected, mask = _select_ragged(
            recovered.transpose(1, 2), self.receiver_positions
        )
        # Store heads before queries for convenient reconstruction.
        self.weights[index] = selected.permute(0, 2, 1, 3).detach().to("cpu").contiguous()
        self.values[index] = value.detach().to("cpu").contiguous()
        self.query_masks[index] = mask.detach().cpu()
        return result

    def arrays(self) -> dict[str, Any]:
        import torch

        missing = [
            index for index in self.layer_indices
            if index not in self.weights or index not in self.values or index not in self.gates
        ]
        if missing:
            raise RuntimeError(f"Missing ordinary-attention workspace layers: {missing}")
        return {
            "ordinary_layer_indices": torch.as_tensor(self.layer_indices, dtype=torch.int16),
            "attention_weights": torch.stack([self.weights[i] for i in self.layer_indices], dim=1),
            "attention_values": torch.stack([self.values[i] for i in self.layer_indices], dim=1),
            "attention_gates": torch.stack([self.gates[i] for i in self.layer_indices], dim=1),
            "attention_query_mask": torch.stack([self.query_masks[i] for i in self.layer_indices], dim=1),
        }

    def close(self) -> None:
        import torch

        torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.active = None


def _physical_workspace_positions(
    tokenizer: Any,
    prompt: str,
    first_question: dict[str, Any],
    second_question: dict[str, Any],
    condition: str,
    rank_letters: list[str],
    original_to_new: dict[str, str],
    padded_width: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    bins, queries, audit = _source_bins_and_queries(
        tokenizer,
        prompt,
        first_question,
        second_question,
        condition,
        rank_letters,
        original_to_new,
    )
    encoded = tokenizer(prompt, add_special_tokens=False)
    ids = [int(value) for value in encoded["input_ids"]]
    left_pad = padded_width - len(ids)
    last_option = max(value for row in queries for value in row)
    tail = [value for value in bins[-1] if value > last_option]
    residual_unpadded = sorted(
        set(
            value
            for bin_index in range(3, 10)
            for value in bins[bin_index]
        )
        | set(value for row in queries for value in row)
        | set(tail)
    )
    receiver_unpadded = sorted(set(value for row in queries for value in row) | set(tail))
    if not residual_unpadded or not receiver_unpadded:
        raise RuntimeError("Workspace position selection unexpectedly became empty")

    role_for_position: dict[int, str] = {}
    for bin_index in range(3, 10):
        for value in bins[bin_index]:
            role_for_position[value] = SOURCE_BINS[bin_index]
    for rank_index, row in enumerate(queries):
        for value in row:
            role_for_position[value] = f"second_{rank_letters[rank_index]}_match_line"
    for value in tail:
        role_for_position[value] = "post_second_options"
    audit["workspace"] = {
        "residual_positions": residual_unpadded,
        "receiver_positions": receiver_unpadded,
        "residual_tokens": tokenizer.convert_ids_to_tokens([ids[i] for i in residual_unpadded]),
        "receiver_tokens": tokenizer.convert_ids_to_tokens([ids[i] for i in receiver_unpadded]),
        "residual_roles": [role_for_position[i] for i in residual_unpadded],
        "receiver_roles": [role_for_position[i] for i in receiver_unpadded],
        "left_pad": left_pad,
        "unpadded_length": len(ids),
    }
    return (
        [left_pad + value for value in residual_unpadded],
        [left_pad + value for value in receiver_unpadded],
        audit,
    )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _reconstruction_checks(
    parts: Any,
    state: dict[str, Any],
    attention: dict[str, Any],
) -> dict[str, float]:
    import torch

    residuals = state["residuals"].float()
    mixer = state["mixer_outputs"].float()
    mlp = state["mlp_outputs"].float()
    mask = state["receiver_mask"]

    # Receiver positions are an ordered subset of residual positions. The
    # caller supplies the corresponding indices for this identity check.
    receiver_in_residual = state["receiver_in_residual"].long()
    receiver_residual = torch.stack(
        [residuals[row, :, receiver_in_residual[row]] for row in range(residuals.shape[0])]
    )
    predicted = receiver_residual[:, :-1] + mixer + mlp
    valid = mask[:, None, :, None]
    residual_error = float(
        (predicted - receiver_residual[:, 1:]).masked_select(valid).abs().max().item()
    )

    layer_indices = attention["ordinary_layer_indices"].long().tolist()
    attention_error = 0.0
    for stored, layer_index in enumerate(layer_indices):
        module = parts.layers[layer_index].self_attn
        weights = attention["attention_weights"][:, stored].float()
        values = attention["attention_values"][:, stored].float()
        if values.shape[1] != weights.shape[1]:
            groups = weights.shape[1] // values.shape[1]
            values = values.repeat_interleave(groups, dim=1)
        context = torch.einsum("bhqk,bhkd->bqhd", weights, values)
        gate = attention["attention_gates"][:, stored].float()
        projected = module.o_proj(
            (context * gate).reshape(context.shape[0], context.shape[1], -1).to(
                device=module.o_proj.weight.device,
                dtype=module.o_proj.weight.dtype,
            )
        ).detach().float().cpu()
        target = state["mixer_outputs"][:, layer_index].float()
        local_mask = mask[..., None]
        error = float((projected - target).masked_select(local_mask).abs().max().item())
        attention_error = max(attention_error, error)
    return {
        "max_residual_identity_error": residual_error,
        "max_attention_reconstruction_error": attention_error,
    }


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
        raise ValueError("Requires exact raw Qwen ChatML serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
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
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if tuple(index + 1 for index in layer_indices) != ORDINARY_LAYERS:
        raise RuntimeError(f"Unexpected ordinary-attention layers: {layer_indices}")

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    completed_path = output_dir / "completed.npy"
    completed = np.load(completed_path) if completed_path.exists() else np.zeros(len(qids), bool)
    natural_path = output_dir / "natural_logits.npy"
    natural_logits = (
        np.lib.format.open_memmap(natural_path, mode="r+")
        if natural_path.exists()
        else np.lib.format.open_memmap(natural_path, mode="w+", dtype=np.float32, shape=(2, len(qids), 4))
    )
    audit_path = output_dir / "prompt_audit.json"
    audit: dict[str, Any] = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    durations: list[float] = []
    sizes: list[int] = []
    started = time.monotonic()
    processed = 0

    for cohort_index, start in enumerate(range(0, len(qids), config.batch_size)):
        cohort = qids[start : start + config.batch_size]
        indices = list(range(start, start + len(cohort)))
        shard = shard_dir / f"cohort_{cohort_index:04d}.pt"
        if all(completed[index] for index in indices):
            if not shard.exists():
                raise RuntimeError(f"Completion bit set but shard is missing: {shard}")
            continue
        if any(completed[index] for index in indices):
            raise RuntimeError("Partially completed cohort")
        if max_cohorts is not None and processed >= max_cohorts:
            break

        cohort_started = time.monotonic()
        condition_payloads: dict[str, Any] = {}
        rank_letters: list[list[str]] = []
        for qid in cohort:
            logits = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=float)
            order = np.argsort(-logits, kind="stable")
            rank_letters.append([LETTERS[int(value)] for value in order])

        for condition_index, condition in enumerate(CONDITIONS):
            batch = _build_batch(config, processor, tokenizer, questions, mappings, cohort, condition)
            padded_width = int(batch["input_ids"].shape[1])
            residual_positions: list[list[int]] = []
            receiver_positions: list[list[int]] = []
            row_audits: list[dict[str, Any]] = []
            for row, qid in enumerate(cohort):
                second_question = {
                    **questions[qid],
                    "options": {
                        new: questions[qid]["options"][old]
                        for new, old in mappings[qid]["new_to_original"].items()
                    },
                }
                residual_row, receiver_row, row_audit = _physical_workspace_positions(
                    tokenizer,
                    batch["prompts"][row],
                    questions[qid],
                    second_question,
                    condition,
                    rank_letters[row],
                    mappings[qid]["original_to_new"],
                    padded_width,
                )
                residual_positions.append(residual_row)
                receiver_positions.append(receiver_row)
                row_audit["prompt_hash"] = _hash_prompt(batch["prompts"][row])
                row_audits.append(row_audit)

            state_collector = WorkspaceStateCollector(parts, residual_positions, receiver_positions)
            attention_collector = WorkspaceAttentionCollector(parts, layer_indices, receiver_positions)
            try:
                natural_output = _forward(model, parts, batch["input_ids"], batch["attention_mask"])
                state = state_collector.arrays()
                attention = attention_collector.arrays()
            finally:
                attention_collector.close()
                state_collector.close()

            natural = _aggregate_logits(natural_output, variant_ids)
            expected = np.stack([
                np.asarray(trusted[condition_index][qid]["aggregated_ad_logits"], dtype=np.float32)
                for qid in cohort
            ])
            natural_error = float(np.max(np.abs(natural - expected)))
            if natural_error != 0.0:
                raise RuntimeError(f"Natural A-D logits failed exact reproduction: {natural_error}")

            max_residual = int(state["residuals"].shape[2])
            receiver_in_residual = torch.zeros(
                (len(cohort), int(state["receiver_mask"].shape[1])), dtype=torch.int16
            )
            for row in range(len(cohort)):
                lookup = {value: index for index, value in enumerate(residual_positions[row])}
                for column, value in enumerate(receiver_positions[row]):
                    receiver_in_residual[row, column] = lookup[value]
            state["receiver_in_residual"] = receiver_in_residual
            checks = _reconstruction_checks(parts, state, attention)

            condition_payloads[condition] = {
                "input_ids": batch["input_ids"].cpu(),
                "attention_mask": batch["attention_mask"].cpu(),
                "residual_positions": torch.as_tensor(
                    [row + [0] * (max_residual - len(row)) for row in residual_positions],
                    dtype=torch.int32,
                ),
                "receiver_positions": torch.as_tensor(
                    [
                        row + [0] * (int(state["receiver_mask"].shape[1]) - len(row))
                        for row in receiver_positions
                    ],
                    dtype=torch.int32,
                ),
                "natural_logits": torch.from_numpy(natural),
                "trusted_logits": torch.from_numpy(expected),
                **state,
                **attention,
                "validation": checks,
            }
            natural_logits[condition_index, indices] = natural
            if condition not in audit:
                audit[condition] = row_audits

        payload = {
            "format_version": 1,
            "question_ids": cohort,
            "conditions": list(CONDITIONS),
            "rank_letters": rank_letters,
            "model_width": int(parts.embedding.weight.shape[-1]),
            "payloads": condition_payloads,
        }
        _atomic_torch_save(payload, shard)
        sizes.append(shard.stat().st_size)
        completed[indices] = True
        natural_logits.flush()
        np.save(completed_path, completed)
        if not audit_path.exists():
            audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        processed += 1
        free = os.statvfs(output_dir)
        free_bytes = int(free.f_bavail * free.f_frsize)
        print(
            f"2P residual workspace: {int(completed.sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.3f}; shard_gib={sizes[-1] / 2**30:.3f}; "
            f"free_gib={free_bytes / 2**30:.1f}",
            flush=True,
        )

    metadata = {
        "experiment": "canonical remapped second-presentation residual workspace",
        "config": config.as_dict(),
        "conditions": list(CONDITIONS),
        "ordinary_layers_one_based": list(ORDINARY_LAYERS),
        "completed_questions": int(completed.sum()),
        "total_questions": len(qids),
        "complete_model_forwards_per_cohort": 2,
        "elapsed_seconds_after_load": time.monotonic() - started,
        "new_cohort_seconds": durations,
        "new_shard_bytes": sizes,
        "projected_cache_bytes": (
            int(np.mean(sizes) * (len(qids) / config.batch_size)) if sizes else None
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
