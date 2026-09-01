from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .attention_spans import attention_span_indices
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import load_trials, prompt_hash
from .run_baseline_mixer_function import SelectedBlockOutputCollector, _forward, _render
from .sublayer_config import SublayerExperimentConfig


WINDOWS = {
    "L1_16": tuple(range(0, 16)),
    "L17_32": tuple(range(16, 32)),
    "L33_40": tuple(range(32, 40)),
    "L41_48": tuple(range(40, 48)),
    "L49_64": tuple(range(48, 64)),
    "all_layers": tuple(range(64)),
}
WINDOW_NAMES = tuple(WINDOWS)
CONDITIONS = ("incorrect", "neutral")


def _hidden(output: Any):
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_hidden(output: Any, hidden: Any):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + list(output[1:])
    return hidden


class BatchedFeedbackResidualPatcher:
    """Replace one token's complete post-block residual in selected batch rows."""

    def __init__(
        self,
        parts: Any,
        position: int,
        source_residuals: dict[int, Any],
    ) -> None:
        self.position = int(position)
        self.source_residuals = source_residuals
        self.rows_by_layer = {
            layer: [row for row, name in enumerate(WINDOW_NAMES) if layer in WINDOWS[name]]
            for layer in range(64)
        }
        self.handles = [
            parts.layers[layer].register_forward_hook(self._hook(layer))
            for layer in range(64)
        ]

    def _hook(self, layer: int):
        def patch(_module: Any, _inputs: Any, output: Any):
            rows = self.rows_by_layer[layer]
            if not rows:
                return None
            hidden = _hidden(output)
            updated = hidden.clone()
            source = self.source_residuals[layer].to(
                device=updated.device, dtype=updated.dtype
            )
            updated[rows, self.position] = source
            return _replace_hidden(output, updated)

        return patch

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _prompt_inputs(
    processor: Any,
    tokenizer: Any,
    config: SublayerExperimentConfig,
    trial: Any,
    condition: str,
):
    prompt = _render(processor, config, trial, condition)
    annotated_ids, spans = attention_span_indices(
        tokenizer, prompt, condition, trial.question
    )
    feedback_position = int(spans["feedback_sentence"][-1])
    input_ids, attention_mask, _ = tokenize_batch(tokenizer, [prompt])
    if annotated_ids != input_ids[0].tolist():
        raise RuntimeError("Offset-aware and model tokenizations disagree")
    return prompt, annotated_ids, feedback_position, input_ids, attention_mask


def _natural_forward(
    model: Any,
    processor: Any,
    parts: Any,
    tokenizer: Any,
    config: SublayerExperimentConfig,
    trial: Any,
    condition: str,
    answer_ids: list[int],
):
    prompt, token_ids, position, input_ids, attention_mask = _prompt_inputs(
        processor, tokenizer, config, trial, condition
    )
    collector = SelectedBlockOutputCollector(parts, tuple(range(64)), position)
    try:
        result = _forward(model, parts, input_ids, attention_mask)
    finally:
        collector.close()
    logits = result.logits.detach().float().cpu()[0, -1, answer_ids].numpy()
    audit = {
        "prompt_hash": prompt_hash(prompt),
        "feedback_position": position,
        "feedback_token_id": int(token_ids[position]),
        "feedback_token": tokenizer.decode([token_ids[position]]),
        "prompt": prompt,
    }
    return logits, collector.values, audit


def _patched_forward(
    model: Any,
    processor: Any,
    parts: Any,
    tokenizer: Any,
    config: SublayerExperimentConfig,
    trial: Any,
    target_condition: str,
    source_residuals: dict[int, Any],
    natural_logits: np.ndarray,
    answer_ids: list[int],
):
    prompt, _, position, _, _ = _prompt_inputs(
        processor, tokenizer, config, trial, target_condition
    )
    # Six intervention rows followed by one unpatched numerical control row.
    input_ids, attention_mask, _ = tokenize_batch(
        tokenizer, [prompt] * (len(WINDOW_NAMES) + 1)
    )
    patcher = BatchedFeedbackResidualPatcher(parts, position, source_residuals)
    try:
        result = _forward(model, parts, input_ids, attention_mask)
    finally:
        patcher.close()
    raw = result.logits.detach().float().cpu()[:, -1, answer_ids].numpy()
    return natural_logits[None, :] + (raw[:-1] - raw[-1][None, :])


def _initialize(path: Path, question_ids: list[str]):
    if path.exists():
        arrays = dict(np.load(path, allow_pickle=False))
        if arrays["question_ids"].astype(str).tolist() != question_ids:
            raise ValueError("Question IDs changed")
        return arrays
    n = len(question_ids)
    return {
        "question_ids": np.asarray(question_ids),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "patched_logits": np.full(
            (2, len(WINDOW_NAMES), n, 4), np.nan, dtype=np.float32
        ),
    }


def run(
    config_path: Path,
    plan_path: Path,
    output: Path,
) -> None:
    import torch
    import transformers

    config = SublayerExperimentConfig.load(config_path)
    if (
        config.prompt_mode != "baseline_matched_empty_history"
        or config.chat_serialization != "raw_qwen_chatml"
    ):
        raise ValueError("Canonical empty-history raw ChatML prompt required")
    question_ids = json.loads(plan_path.read_text())["question_ids"]
    trials = load_trials(
        config.manifest_path, config.baseline_results_path, question_ids, None
    )
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "feedback_end_residual_replacement_results.npz"
    arrays = _initialize(result_path, question_ids)

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    answer_ids = [resolved[letter][0][1] for letter in "ABCD"]

    for question_index, trial in enumerate(trials):
        if bool(arrays["completed"][question_index]):
            continue
        natural: dict[str, np.ndarray] = {}
        residuals: dict[str, dict[int, Any]] = {}
        audits = {}
        for condition_index, condition in enumerate(CONDITIONS):
            logits, values, audit = _natural_forward(
                model, processor, parts, tokenizer, config, trial, condition, answer_ids
            )
            natural[condition] = logits
            residuals[condition] = values
            audits[condition] = audit
            arrays["natural_logits"][condition_index, question_index] = logits

        arrays["patched_logits"][0, :, question_index] = _patched_forward(
            model, processor, parts, tokenizer, config, trial, "incorrect",
            residuals["neutral"], natural["incorrect"], answer_ids,
        )
        arrays["patched_logits"][1, :, question_index] = _patched_forward(
            model, processor, parts, tokenizer, config, trial, "neutral",
            residuals["incorrect"], natural["neutral"], answer_ids,
        )
        arrays["completed"][question_index] = True

        if question_index == 0 and not (output / "prompt_audit.json").exists():
            (output / "prompt_audit.json").write_text(
                json.dumps(
                    {"question_id": trial.question_id, **audits},
                    indent=2,
                    sort_keys=True,
                )
            )
        done = int(np.sum(arrays["completed"]))
        if done == 1 or done % 5 == 0 or done == len(trials):
            atomic_save_npz(result_path, **arrays)
            print(f"feedback-end residual replacement: {done}/{len(trials)}", flush=True)

    metadata = {
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "n_questions": len(question_ids),
        "conditions": list(CONDITIONS),
        "windows": {name: [layer + 1 for layer in layers] for name, layers in WINDOWS.items()},
        "intervention": (
            "At the period ending the feedback sentence, replace the complete "
            "post-block residual with the paired same-question residual from the "
            "other condition at every layer in the specified window."
        ),
        "batch_control": (
            "Six intervention rows plus an unpatched matched row; saved logits "
            "equal single-trial natural logits plus patched-minus-control logits."
        ),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.plan, args.output)


if __name__ == "__main__":
    main()
