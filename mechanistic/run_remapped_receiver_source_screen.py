from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .io import atomic_save_npz
from .modeling import get_tokenizer, load_model_and_processor, resolve_answer_tokens
from .receiver_path_utils import ROLE_NAMES, locate_receiver_roles
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward


MAX_POSITIONS = 768


class ReceiverSourceWriteCollector:
    """Recover exact attention weights and source-line writes at every query."""

    def __init__(
        self,
        parts: Any,
        layer_indices: tuple[int, ...],
        selected_positions: list[list[int]],
        control_positions: list[list[int]],
    ) -> None:
        import torch

        self.parts = parts
        self.layer_indices = layer_indices
        self.selected_positions = selected_positions
        self.control_positions = control_positions
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
        selected_mask = torch.zeros(
            batch, key_length, device=query.device, dtype=weights.dtype
        )
        control_mask = torch.zeros_like(selected_mask)
        for row in range(batch):
            selected_mask[row, self.selected_positions[row]] = 1
            control_mask[row, self.control_positions[row]] = 1

        selected_weights = weights * selected_mask[:, None, None, :]
        control_weights = weights * control_mask[:, None, None, :]
        selected_context = torch.einsum(
            "bhqk,bhkd->bhqd", selected_weights, value_for_heads
        ) * gate
        control_context = torch.einsum(
            "bhqk,bhkd->bhqd", control_weights, value_for_heads
        ) * gate
        selected_flat = selected_context.permute(0, 2, 1, 3).flatten(2).float()
        control_flat = control_context.permute(0, 2, 1, 3).flatten(2).float()
        projection = self.layers[layer_index].o_proj
        selected_write = F.linear(selected_flat, projection.weight.float(), None)
        control_write = F.linear(control_flat, projection.weight.float(), None)
        self.metrics[layer_index] = {
            "selected_attention_mass": selected_weights.float().sum(-1).mean(1).to(
                "cpu", dtype=torch.float16
            ),
            "control_attention_mass": control_weights.float().sum(-1).mean(1).to(
                "cpu", dtype=torch.float16
            ),
            "selected_write_norm": selected_write.norm(dim=-1).to(
                "cpu", dtype=torch.float16
            ),
            "control_write_norm": control_write.norm(dim=-1).to(
                "cpu", dtype=torch.float16
            ),
            "difference_write_norm": (selected_write - control_write).norm(dim=-1).to(
                "cpu", dtype=torch.float16
            ),
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
            raise RuntimeError(f"Missing receiver source writes at layers {missing}")
        return {
            name: torch.stack(
                [self.metrics[index][name] for index in self.layer_indices], dim=0
            ).numpy()
            for name in self.metrics[self.layer_indices[0]]
        }


def _initialize(
    path: Path,
    all_qids: list[str],
    screen_qids: list[str],
    layer_indices: tuple[int, ...],
) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["all_question_ids"].astype(str).tolist() != all_qids:
            raise ValueError("All question IDs changed")
        if arrays["screen_question_ids"].astype(str).tolist() != screen_qids:
            raise ValueError("Screen question IDs changed")
        return arrays
    n_all = len(all_qids)
    n_screen = len(screen_qids)
    shape = (2, len(layer_indices), n_screen, MAX_POSITIONS)
    arrays: dict[str, np.ndarray] = {
        "all_question_ids": np.asarray(all_qids),
        "screen_question_ids": np.asarray(screen_qids),
        "ordinary_layer_indices": np.asarray(layer_indices, dtype=np.int16),
        "ordinary_blocks": np.asarray([index + 1 for index in layer_indices], dtype=np.int16),
        "role_names": np.asarray(ROLE_NAMES),
        "completed": np.zeros(n_all, dtype=bool),
        "trusted_natural_logits": np.full((2, n_all, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n_all, 4), np.nan, dtype=np.float32),
        "lengths": np.zeros((2, n_screen), dtype=np.int16),
        "role_codes": np.full((2, n_screen, MAX_POSITIONS), -1, dtype=np.int16),
        "token_ids": np.full((2, n_screen, MAX_POSITIONS), -1, dtype=np.int32),
        "control_letters": np.full((2, n_screen), "", dtype="<U1"),
    }
    for name in (
        "selected_attention_mass",
        "control_attention_mass",
        "selected_write_norm",
        "control_write_norm",
        "difference_write_norm",
    ):
        arrays[name] = np.full(shape, np.nan, dtype=np.float16)
    return arrays


def run(
    config_path: Path,
    remapping_plan_path: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
    trusted_game_path: Path,
    trusted_neutral_path: Path,
    discovery_plan_path: Path,
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
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw_qwen_chatml serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    all_qids = [row["id"] for row in manifest["questions"]]
    if max_cohorts is not None:
        all_qids = all_qids[: int(max_cohorts) * config.batch_size]
    mappings = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped_baseline = json.loads(remapped_baseline_path.read_text())["results"]
    trusted = [
        json.loads(trusted_game_path.read_text())["results"],
        json.loads(trusted_neutral_path.read_text())["results"],
    ]
    discovery = set(json.loads(discovery_plan_path.read_text())["question_ids"])
    screen_qids = [qid for qid in all_qids if qid in discovery]

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
        if getattr(layer, "self_attn", None) is not None
    )
    if tuple(index + 1 for index in layer_indices) != tuple(range(4, 65, 4)):
        raise RuntimeError(f"Unexpected ordinary-attention layers: {layer_indices}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, all_qids, screen_qids, layer_indices)
    all_index = {qid: index for index, qid in enumerate(all_qids)}
    screen_index = {qid: index for index, qid in enumerate(screen_qids)}
    audit_path = output_dir / "prompt_audit.json"
    started = time.monotonic()

    for start in range(0, len(all_qids), config.batch_size):
        cohort = all_qids[start : start + config.batch_size]
        indices = [all_index[qid] for qid in cohort]
        if np.all(arrays["completed"][indices]):
            continue
        cohort_started = time.monotonic()
        batches = [
            _build_batch(config, processor, tokenizer, questions, mappings, cohort, condition)
            for condition in CONDITIONS
        ]
        cohort_audit: dict[str, Any] = {"question_ids": cohort, "conditions": {}}
        for ci, (condition, batch) in enumerate(zip(CONDITIONS, batches)):
            width = int(batch["input_ids"].shape[1])
            row_data: list[dict[str, Any]] = []
            selected_physical: list[list[int]] = []
            control_physical: list[list[int]] = []
            for row, qid in enumerate(cohort):
                remapped_question = {
                    **questions[qid],
                    "options": {
                        new: questions[qid]["options"][old]
                        for new, old in mappings[qid]["new_to_original"].items()
                    },
                }
                w1 = baseline[qid].get("answer", baseline[qid].get("subject_answer"))
                w2 = remapped_baseline[qid]["answer_original_content"]
                located = locate_receiver_roles(
                    tokenizer,
                    batch["prompts"][row],
                    batch["messages"][row],
                    questions[qid],
                    remapped_question,
                    condition,
                    w1,
                    w2,
                    mappings[qid],
                )
                ids = batch["token_rows"][row]
                if ids != located["ids"]:
                    raise RuntimeError("Receiver-role tokenization changed")
                if len(ids) > MAX_POSITIONS:
                    raise RuntimeError(
                        f"Prompt length {len(ids)} exceeds MAX_POSITIONS={MAX_POSITIONS}"
                    )
                left_pad = width - len(ids)
                selected_physical.append(
                    [left_pad + position for position in located["selected_positions"]]
                )
                control_physical.append(
                    [left_pad + position for position in located["control_positions"]]
                )
                row_data.append(located)

            collector = ReceiverSourceWriteCollector(
                parts, layer_indices, selected_physical, control_physical
            )
            try:
                output = _forward(
                    model, parts, batch["input_ids"], batch["attention_mask"]
                )
                metrics = collector.arrays()
            finally:
                collector.close()
            natural = _aggregate_logits(output, variant_ids)
            for row, qid in enumerate(cohort):
                qi = all_index[qid]
                arrays["same_batch_natural_logits"][ci, qi] = natural[row]
                arrays["trusted_natural_logits"][ci, qi] = np.asarray(
                    trusted[ci][qid]["aggregated_ad_logits"], dtype=np.float32
                )
                if qid not in screen_index:
                    continue
                si = screen_index[qid]
                ids = row_data[row]["ids"]
                length = len(ids)
                left_pad = width - length
                arrays["lengths"][ci, si] = length
                arrays["role_codes"][ci, si, :length] = row_data[row]["roles"]
                arrays["token_ids"][ci, si, :length] = ids
                arrays["control_letters"][ci, si] = row_data[row]["control_letter"]
                for name, values in metrics.items():
                    arrays[name][ci, :, si, :length] = values[:, row, left_pad:width]

            cohort_audit["conditions"][condition] = {
                "rendered_prompt": batch["prompts"][0],
                "selected_line": row_data[0]["selected_audit"],
                "control_line": row_data[0]["control_audit"],
                "control_letter": row_data[0]["control_letter"],
                "role_tokens": [
                    {
                        "position": position,
                        "role": ROLE_NAMES[int(row_data[0]["roles"][position])],
                        "token": tokenizer.decode([row_data[0]["ids"][position]]),
                    }
                    for position in range(len(row_data[0]["ids"]))
                    if int(row_data[0]["roles"][position]) != 0
                ],
            }

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        print(
            f"receiver source screen: {int(arrays['completed'].sum())}/{len(all_qids)}; "
            f"cohort_seconds={time.monotonic() - cohort_started:.1f}",
            flush=True,
        )
        if not audit_path.exists():
            audit_path.write_text(
                json.dumps(cohort_audit, indent=2, ensure_ascii=False) + "\n"
            )

    metadata = {
        "experiment": "canonical remapped downstream receiver source-write screen",
        "config": config.as_dict(),
        "n_historical_questions": len(all_qids),
        "n_discovery_screen_questions": len(screen_qids),
        "conditions": list(CONDITIONS),
        "ordinary_blocks": [index + 1 for index in layer_indices],
        "max_positions": MAX_POSITIONS,
        "metrics": [
            "selected_attention_mass",
            "control_attention_mass",
            "selected_write_norm",
            "control_write_norm",
            "difference_write_norm",
        ],
        "complete_model_forwards_per_cohort": 2,
        "extra_sdpa_calls_per_ordinary_block": "ceil(prompt_width / head_dim), normally 2",
        "elapsed_seconds_after_load": time.monotonic() - started,
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
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.remapping_plan,
        args.baseline,
        args.remapped_baseline,
        args.trusted_game,
        args.trusted_neutral,
        args.discovery_plan,
        args.output_dir,
        args.max_cohorts,
    )


if __name__ == "__main__":
    main()

