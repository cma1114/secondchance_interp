from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .collect_remapped_feedback_factorial import _messages, _remap_question
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
from .prompts import FACTORIAL_FEEDBACK, prompt_hash
from .run_evaluation_update_transplant import _locate_evaluation
from .sublayer import _hidden


CONDITIONS = ("incorrect_again", "lost_again")
POSITIONS = ("evaluation_period", "final_decision")


def _chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


class GLAOutputCollector:
    """Capture the actual projected GLA output at selected row-wise positions."""

    def __init__(self, parts: Any, layers: list[int], positions: list[list[int]]):
        self.positions = positions
        self.values: dict[int, Any] = {}
        self.handles = [
            parts.layers[layer].linear_attn.register_forward_hook(self._hook(layer))
            for layer in layers
        ]

    def _hook(self, layer: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = _hidden(output)
            selected = torch.stack([
                hidden[row, torch.as_tensor(row_positions, device=hidden.device)]
                for row, row_positions in enumerate(self.positions)
            ])
            self.values[layer] = selected.detach().to("cpu", dtype=torch.float16)

        return capture

    def stacked(self, layers: list[int]):
        import torch

        missing = [layer for layer in layers if layer not in self.values]
        if missing:
            raise RuntimeError(f"Missing GLA outputs for layers: {missing}")
        return torch.stack([self.values[layer] for layer in layers], dim=1)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _aggregate_logits(output: Any, variant_ids: dict[str, list[int]]) -> np.ndarray:
    import torch

    logits = output.logits.detach().float()
    final = logits[:, 0] if logits.shape[1] == 1 else logits[:, -1]
    return torch.stack([
        torch.logsumexp(final[:, variant_ids[letter]], dim=-1)
        for letter in LETTERS
    ], dim=-1).cpu().numpy()


def _initialize(
    path: Path,
    question_ids: list[str],
    gla_layers: list[int],
    width: int,
) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != question_ids:
            raise ValueError("Existing checkpoint uses different question IDs")
        if arrays["gla_layers_zero_based"].tolist() != gla_layers:
            raise ValueError("Existing checkpoint uses different GLA layers")
        return arrays
    n = len(question_ids)
    n_layers = len(gla_layers)
    return {
        "question_ids": np.asarray(question_ids),
        "gla_layers_zero_based": np.asarray(gla_layers, dtype=np.int16),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "output_norm": np.full((2, n, n_layers, 2), np.nan, dtype=np.float32),
        "canonical_ad_write": np.full(
            (2, n, n_layers, 2, 4), np.nan, dtype=np.float32
        ),
        "paired_cosine": np.full((n, n_layers, 2), np.nan, dtype=np.float32),
        "paired_delta_norm": np.full((n, n_layers, 2), np.nan, dtype=np.float32),
        "mean_output_sum": np.zeros((2, n_layers, 2, width), dtype=np.float32),
        "w1_output_sum": np.zeros((2, 4, n_layers, 2, width), dtype=np.float32),
        "w1_counts": np.zeros((2, 4), dtype=np.int32),
    }


def collect(
    config_path: Path,
    remapping_plan_path: Path,
    baseline_path: Path,
    output: Path,
    max_questions: int | None,
    checkpoint_every_cohorts: int,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml" or config.attn_implementation != "sdpa":
        raise ValueError("Requires exact raw ChatML + SDPA regime")
    if config.batch_size != 4:
        raise ValueError("Exact historical collection requires batch_size=4")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    question_ids = [row["id"] for row in manifest["questions"]]
    if max_questions is not None:
        question_ids = question_ids[: int(max_questions)]
    remapping = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    baseline = json.loads(baseline_path.read_text())["results"]
    required = set(question_ids)
    for name, values in (("remapping", remapping), ("baseline", baseline)):
        if not required <= set(values):
            raise ValueError(f"{name} is missing requested questions")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    canonical_ids = [resolved[letter][0][1] for letter in LETTERS]
    unembedding = parts.output_head.weight.detach()[canonical_ids].float().cpu().numpy()
    gla_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if len(gla_layers) != 48:
        raise RuntimeError(f"Expected 48 GLA layers, found {len(gla_layers)}")
    width = int(parts.embedding.weight.shape[-1])
    arrays = _initialize(output, question_ids, gla_layers, width)
    output.parent.mkdir(parents=True, exist_ok=True)
    qid_to_index = {qid: index for index, qid in enumerate(question_ids)}
    audit = None
    device = model_input_device(parts)

    cohorts = list(_chunks(question_ids, config.batch_size))
    for cohort_index, cohort in enumerate(cohorts):
        indices = [qid_to_index[qid] for qid in cohort]
        if all(arrays["completed"][index] for index in indices):
            continue
        if any(arrays["completed"][index] for index in indices):
            raise RuntimeError("Checkpoint contains a partially completed cohort")
        captured_by_condition = []
        for condition_index, condition in enumerate(CONDITIONS):
            prompts, position_rows, prompt_rows = [], [], []
            for qid in cohort:
                question = questions[qid]
                mapping = remapping[qid]
                remapped = _remap_question(question, mapping["new_to_original"])
                messages = _messages(config, question, remapped, condition)
                prompt = render_chat(
                    processor, messages, config.disable_thinking, config.chat_serialization
                )
                located = _locate_evaluation(tokenizer, prompt, condition)
                prompts.append(prompt)
                prompt_rows.append((messages, prompt, located))

            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            padded_length = int(input_ids.shape[1])
            for row, (_messages_row, _prompt, located) in enumerate(prompt_rows):
                pad = padded_length - len(located["ids"])
                evaluation_period = int(located["period_position"] + pad)
                final_decision = int(last_indices[row])
                position_rows.append([evaluation_period, final_decision])

            collector = GLAOutputCollector(parts, gla_layers, position_rows)
            try:
                with torch.inference_mode():
                    kwargs = {
                        "input_ids": input_ids.to(device),
                        "attention_mask": attention_mask.to(device),
                        "use_cache": False,
                        "return_dict": True,
                    }
                    try:
                        result = model(**kwargs, logits_to_keep=1)
                    except TypeError:
                        result = model(**kwargs)
                captured = collector.stacked(gla_layers).float().numpy()
            finally:
                collector.close()
            # captured: batch, GLA layer, semantic position, residual width.
            captured_by_condition.append(captured)
            batch_indices = np.asarray(indices)
            arrays["natural_logits"][condition_index, batch_indices] = _aggregate_logits(
                result, variant_ids
            )
            arrays["output_norm"][condition_index, batch_indices] = np.linalg.norm(
                captured, axis=-1
            )
            arrays["canonical_ad_write"][condition_index, batch_indices] = np.einsum(
                "blph,ah->blpa", captured, unembedding, optimize=True
            )
            arrays["mean_output_sum"][condition_index] += captured.sum(axis=0)
            for row, qid in enumerate(cohort):
                w1_index = LETTERS.index(str(baseline[qid]["answer"]))
                arrays["w1_output_sum"][condition_index, w1_index] += captured[row]
                arrays["w1_counts"][condition_index, w1_index] += 1

            if audit is None:
                messages, prompt, located = prompt_rows[0]
                audit = {
                    "question_id": cohort[0],
                    "condition": condition,
                    "prompt_hash": prompt_hash(prompt),
                    "prompt": prompt,
                    "messages": messages,
                    "evaluation_word_token": located["word_token"],
                    "evaluation_period_token": located["period_token"],
                    "unpadded_evaluation_period_zero_based": located["period_position"],
                    "unpadded_final_decision_zero_based": len(located["ids"]) - 1,
                }

        evaluation, neutral = captured_by_condition
        dot = np.sum(evaluation * neutral, axis=-1)
        denominator = np.maximum(
            np.linalg.norm(evaluation, axis=-1) * np.linalg.norm(neutral, axis=-1),
            1e-12,
        )
        arrays["paired_cosine"][indices] = dot / denominator
        arrays["paired_delta_norm"][indices] = np.linalg.norm(evaluation - neutral, axis=-1)
        arrays["completed"][indices] = True

        done_cohorts = cohort_index + 1
        if (
            done_cohorts == 1
            or done_cohorts % checkpoint_every_cohorts == 0
            or done_cohorts == len(cohorts)
        ):
            atomic_save_npz(output, **arrays)
            print(
                f"GLA residual writes: {int(arrays['completed'].sum())}/{len(question_ids)} "
                f"questions ({done_cohorts}/{len(cohorts)} cohorts)",
                flush=True,
            )

    metadata = {
        "config": config.as_dict(),
        "remapping_plan": str(remapping_plan_path),
        "baseline": str(baseline_path),
        "n_questions": len(question_ids),
        "conditions": list(CONDITIONS),
        "positions": list(POSITIONS),
        "gla_layers_zero_based": gla_layers,
        "complete_model_forward_passes": len(cohorts) * len(CONDITIONS),
        "batch_rows_per_forward": config.batch_size,
        "capture": (
            "Actual post-output-projection GLA module vector added to the residual stream "
            "at the evaluation-closing period and final decision position."
        ),
        "prompt_audit": audit,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect action-matched GLA residual-stream output writes"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--checkpoint-every-cohorts", type=int, default=5)
    args = parser.parse_args()
    collect(
        args.config,
        args.remapping_plan,
        args.baseline,
        args.output,
        args.max_questions,
        args.checkpoint_every_cohorts,
    )


if __name__ == "__main__":
    main()
