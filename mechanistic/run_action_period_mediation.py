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
from .collect_action_matched_period_jlens import BatchPositionCollector, _period_positions
from .collect_remapped_feedback_factorial import _messages, _remap_question
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
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_first_boundary_gla_state_transplant import (
    CachedGLAStatePatcher,
    GLAStateCollector,
)


CONDITIONS = ("incorrect_again", "lost_again")
SCENARIOS = ("residual_trajectory", "gla_state", "joint")


def _hidden(output: Any):
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_hidden(output: Any, hidden: Any):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + list(output[1:])
    return hidden


class ActionResidualTrajectoryPatcher:
    """Replace the action-period post-block state at every transformer block."""

    def __init__(self, parts: Any, positions: list[int], source: Any) -> None:
        self.positions = positions
        self.source = source
        self.handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(parts.layers)
        ]

    def _hook(self, layer_index: int):
        def patch(_module: Any, _inputs: Any, output: Any):
            import torch

            hidden = _hidden(output)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            cols = torch.as_tensor(self.positions, device=hidden.device)
            updated = hidden.clone()
            updated[rows, cols] = self.source[:, layer_index].to(
                device=hidden.device, dtype=hidden.dtype
            )
            return _replace_hidden(output, updated)

        return patch

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _build_batch(
    config: ExperimentConfig,
    processor: Any,
    tokenizer: Any,
    questions: dict[str, dict[str, Any]],
    mappings: dict[str, dict[str, Any]],
    qids: list[str],
    condition: str,
) -> dict[str, Any]:
    prompts = []
    messages = []
    token_rows = []
    unpadded_positions = []
    audits = []
    for qid in qids:
        question = questions[qid]
        remapped = _remap_question(question, mappings[qid]["new_to_original"])
        row_messages = _messages(config, question, remapped, condition)
        prompt = render_chat(
            processor, row_messages, config.disable_thinking, config.chat_serialization
        )
        periods, audit = _period_positions(tokenizer, prompt, condition)
        ids = [int(value) for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
        prompts.append(prompt)
        messages.append(row_messages)
        token_rows.append(ids)
        unpadded_positions.append(int(periods[1]))
        audits.append(audit)

    input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
    width = int(input_ids.shape[1])
    positions = []
    for row, (ids, position) in enumerate(zip(token_rows, unpadded_positions)):
        left_pad = width - len(ids)
        if input_ids[row, left_pad:].tolist() != ids:
            raise RuntimeError("Exact historical-cohort tokenization changed")
        positions.append(left_pad + position)
    return {
        "prompts": prompts,
        "messages": messages,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "positions": positions,
        "token_rows": token_rows,
        "audits": audits,
    }


def _initialize(path: Path, qids: list[str], gla_layers: list[int]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Question IDs changed")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "gla_layer_indices": np.asarray(gla_layers, dtype=np.int16),
        "completed": np.zeros(n, dtype=bool),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "identity_state_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "patched_logits": np.full((2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32),
        "state_delta_norm": np.full((2, n, len(gla_layers)), np.nan, dtype=np.float32),
        "residual_delta_norm": np.full((2, n, 64), np.nan, dtype=np.float32),
    }


def _cache_for_rows(values: dict[int, dict[int, np.ndarray]], rows: int):
    return {
        layer: {row: layer_values[row] for row in range(rows)}
        for layer, layer_values in values.items()
    }


def _state_delta_norm(target, donor, batch_size: int, gla_layers: list[int]) -> np.ndarray:
    out = np.empty((batch_size, len(gla_layers)), dtype=np.float32)
    for row in range(batch_size):
        for li, layer in enumerate(gla_layers):
            out[row, li] = np.linalg.norm(
                donor[layer][row].astype(np.float64) - target[layer][row].astype(np.float64)
            )
    return out


def run(
    config_path: Path,
    remapping_plan_path: Path,
    trusted_evaluation_path: Path,
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
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    all_qids = [row["id"] for row in manifest["questions"]]
    if max_cohorts is not None:
        all_qids = all_qids[: int(max_cohorts) * config.batch_size]
    mappings = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    trusted_payloads = [
        json.loads(trusted_evaluation_path.read_text())["results"],
        json.loads(trusted_neutral_path.read_text())["results"],
    ]

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

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, all_qids, gla_layers)
    qid_index = {qid: index for index, qid in enumerate(all_qids)}
    audit_path = output_dir / "prompt_audit.json"
    run_start = time.monotonic()

    for cohort_index, start in enumerate(range(0, len(all_qids), config.batch_size)):
        cohort_qids = all_qids[start : start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort_qids]
        if np.all(arrays["completed"][indices]):
            continue
        cohort_start = time.monotonic()
        batches = [
            _build_batch(config, processor, tokenizer, questions, mappings, cohort_qids, condition)
            for condition in CONDITIONS
        ]
        if batches[0]["positions"] != batches[1]["positions"]:
            raise RuntimeError("Action-closing periods are not physically aligned")

        natural_logits = []
        residuals = []
        states = []
        for ci, batch in enumerate(batches):
            positions_by_row = {row: [position] for row, position in enumerate(batch["positions"])}
            state_collector = GLAStateCollector(parts, gla_layers, positions_by_row)
            residual_collector = BatchPositionCollector(
                parts.layers, [[position] for position in batch["positions"]]
            )
            try:
                natural_output = _forward(
                    model, parts, batch["input_ids"], batch["attention_mask"]
                )
            finally:
                residual_collector.close()
                state_collector.close()
            raw_logits = _aggregate_logits(natural_output, variant_ids)
            residual = residual_collector.stacked()[:, :, 0].contiguous()
            natural_logits.append(raw_logits)
            residuals.append(residual)
            states.append(_cache_for_rows(state_collector.values, len(cohort_qids)))
            for local, qid in enumerate(cohort_qids):
                qi = qid_index[qid]
                arrays["same_batch_natural_logits"][ci, qi] = raw_logits[local]
                arrays["trusted_natural_logits"][ci, qi] = np.asarray(
                    trusted_payloads[ci][qid]["aggregated_ad_logits"], dtype=np.float32
                )

        for target_ci, source_ci in ((0, 1), (1, 0)):
            batch = batches[target_ci]
            positions_by_row = {row: [position] for row, position in enumerate(batch["positions"])}
            identity_patcher = CachedGLAStatePatcher(parts, states[target_ci], positions_by_row)
            try:
                identity_raw = _aggregate_logits(
                    _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                    variant_ids,
                )
            finally:
                identity_patcher.close()

            residual_patcher = ActionResidualTrajectoryPatcher(
                parts, batch["positions"], residuals[source_ci]
            )
            try:
                residual_raw = _aggregate_logits(
                    _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                    variant_ids,
                )
            finally:
                residual_patcher.close()

            state_patcher = CachedGLAStatePatcher(parts, states[source_ci], positions_by_row)
            try:
                state_raw = _aggregate_logits(
                    _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                    variant_ids,
                )
            finally:
                state_patcher.close()

            state_patcher = CachedGLAStatePatcher(parts, states[source_ci], positions_by_row)
            residual_patcher = ActionResidualTrajectoryPatcher(
                parts, batch["positions"], residuals[source_ci]
            )
            try:
                joint_raw = _aggregate_logits(
                    _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                    variant_ids,
                )
            finally:
                residual_patcher.close()
                state_patcher.close()

            state_norms = _state_delta_norm(
                states[target_ci], states[source_ci], len(cohort_qids), gla_layers
            )
            residual_norms = torch.linalg.vector_norm(
                residuals[source_ci].float() - residuals[target_ci].float(), dim=-1
            ).numpy()
            for local, qid in enumerate(cohort_qids):
                qi = qid_index[qid]
                trusted = arrays["trusted_natural_logits"][target_ci, qi]
                arrays["identity_state_logits"][target_ci, qi] = identity_raw[local]
                arrays["patched_logits"][target_ci, 0, qi] = (
                    trusted + residual_raw[local] - natural_logits[target_ci][local]
                )
                arrays["patched_logits"][target_ci, 1, qi] = (
                    trusted + state_raw[local] - identity_raw[local]
                )
                arrays["patched_logits"][target_ci, 2, qi] = (
                    trusted + joint_raw[local] - identity_raw[local]
                )
                arrays["state_delta_norm"][target_ci, qi] = state_norms[local]
                arrays["residual_delta_norm"][target_ci, qi] = residual_norms[local]

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        elapsed = time.monotonic() - cohort_start
        done = int(arrays["completed"].sum())
        print(
            f"action-period mediation: {done}/{len(all_qids)} questions; "
            f"cohort_seconds={elapsed:.1f}",
            flush=True,
        )
        if not audit_path.exists():
            audit_path.write_text(json.dumps({
                "question_ids": cohort_qids,
                "conditions": {
                    condition: {
                        "feedback": batches[ci]["audits"][0]["feedback"],
                        "action_period_token": batches[ci]["audits"][0]["decoded_tokens"][1],
                        "action_period_position_zero_based_unpadded": batches[ci]["audits"][0]["unpadded_token_positions_zero_based"][1],
                        "prompt_hash": prompt_hash(batches[ci]["prompts"][0]),
                        "rendered_prompt": batches[ci]["prompts"][0],
                    }
                    for ci, condition in enumerate(CONDITIONS)
                },
            }, indent=2, ensure_ascii=False) + "\n")

    metadata = {
        "config": config.as_dict(),
        "remapping_plan": str(remapping_plan_path),
        "trusted_evaluation": str(trusted_evaluation_path),
        "trusted_neutral": str(trusted_neutral_path),
        "n_questions": len(all_qids),
        "historical_batch_size": config.batch_size,
        "conditions": list(CONDITIONS),
        "scenarios": list(SCENARIOS),
        "complete_model_forwards_per_cohort": 10,
        "gla_layers_zero_based": gla_layers,
        "interventions": {
            "residual_trajectory": "Replace the action-closing period's post-block residual at all 64 blocks with the paired other-condition state.",
            "gla_state": "Replace all 48 accumulated GLA recurrent matrix states immediately after the action-closing period with paired other-condition states.",
            "joint": "Apply both replacements while keeping the target prompt and every later token fixed.",
        },
        "corrections": {
            "residual": "trusted natural + residual-patched same-batch - untouched same-batch natural",
            "state_and_joint": "trusted natural + donor-state segmented pass - recipient-state segmented identity pass",
        },
        "elapsed_seconds_after_load": time.monotonic() - run_start,
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
    parser = argparse.ArgumentParser(description="Test action-period mediation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--trusted-evaluation", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(
        args.config, args.remapping_plan, args.trusted_evaluation,
        args.trusted_neutral, args.output_dir, args.max_cohorts,
    )


if __name__ == "__main__":
    main()
