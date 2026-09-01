from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .attention_intervention import BatchedAttentionEdgeAblator, EdgeTarget
from .config import ExperimentConfig
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import load_trials
from .run_feedback_token_trajectory_swap import _render_and_locate


CONDITIONS = ("incorrect", "neutral")


def _targets(rows: list[dict]) -> list[EdgeTarget]:
    layers = [int(row["layer"]) for row in rows]
    if len(layers) != len(set(layers)):
        raise ValueError("Targets for one scenario must be grouped to one entry per layer")
    return [
        EdgeTarget(int(row["layer"]), tuple(int(head) for head in row["heads"]))
        for row in rows
    ]


def _final_logits(output, answer_ids: list[int]) -> np.ndarray:
    logits = output.logits
    final = logits[:, 0] if logits.shape[1] == 1 else logits[:, -1]
    return final[:, answer_ids].detach().float().cpu().numpy()


def _forward(model, parts, input_ids, attention_mask, answer_ids):
    import torch

    with torch.inference_mode():
        kwargs = {
            "input_ids": input_ids.to(model_input_device(parts)),
            "attention_mask": attention_mask.to(model_input_device(parts)),
            "use_cache": False,
            "return_dict": True,
        }
        try:
            output = model(**kwargs, logits_to_keep=1)
        except TypeError:
            output = model(**kwargs)
    return _final_logits(output, answer_ids)


def _chunks(values: list[dict], size: int):
    for start in range(0, len(values), size):
        yield start, values[start:start + size]


def run(
    config_path: Path,
    plan_path: Path,
    output: Path,
    scenario_batch_size: int,
    max_questions: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires the isolated token_matched_test prompt variant")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw_qwen_chatml serialization")
    if config.attn_implementation != "eager":
        raise ValueError("Exact edge ablation requires eager attention")
    plan = json.loads(plan_path.read_text())
    question_ids = plan["question_ids"]
    if max_questions is not None:
        question_ids = question_ids[:max_questions]
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        question_ids,
        None,
    )
    scenarios = plan["scenarios"]
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    answer_ids = [resolved[letter][0][1] for letter in "ABCD"]
    n_chunks = (len(scenarios) + scenario_batch_size - 1) // scenario_batch_size

    output.parent.mkdir(parents=True, exist_ok=True)
    qids = np.asarray([trial.question_id for trial in trials])
    if output.exists():
        with np.load(output, allow_pickle=False) as existing:
            arrays = {key: existing[key] for key in existing.files}
        if arrays["question_ids"].astype(str).tolist() != qids.astype(str).tolist():
            raise ValueError("Existing output uses different question IDs")
    else:
        arrays = {
            "question_ids": qids,
            "scenario_ids": np.asarray([row["id"] for row in scenarios]),
            "completed": np.zeros((2, len(trials)), dtype=bool),
            "natural_logits": np.full((2, len(trials), 4), np.nan, dtype=np.float32),
            "intervened_logits": np.full(
                (2, len(scenarios), len(trials), 4), np.nan, dtype=np.float32
            ),
            "batch_control_minus_natural": np.full(
                (2, len(trials), n_chunks, 4), np.nan, dtype=np.float32
            ),
        }

    audit = None
    for qi, trial in enumerate(trials):
        for ci, condition in enumerate(CONDITIONS):
            if arrays["completed"][ci, qi]:
                continue
            prompt_data = _render_and_locate(
                processor, tokenizer, config, trial, condition
            )
            natural = _forward(
                model,
                parts,
                prompt_data["input_ids"],
                prompt_data["attention_mask"],
                answer_ids,
            )[0]
            arrays["natural_logits"][ci, qi] = natural
            if audit is None:
                audit = {
                    "question_id": trial.question_id,
                    "condition": condition,
                    "feedback_tokens": list(prompt_data["tokens"]),
                    "feedback_positions_zero_based": prompt_data["positions"],
                    "final_query_position_zero_based": len(prompt_data["ids"]) - 1,
                    "final_query_token": tokenizer.decode([prompt_data["ids"][-1]]),
                }

            for chunk_index, (scenario_start, chunk) in enumerate(
                _chunks(scenarios, scenario_batch_size)
            ):
                prompts = [prompt_data["prompt"]] * (len(chunk) + 1)
                input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
                targets = [_targets(row["targets"]) for row in chunk] + [[]]
                source_positions = [
                    [
                        prompt_data["positions"][
                            int(row["source_token_index_within_feedback_zero_based"])
                        ]
                    ]
                    for row in chunk
                ]
                # The final batch row is an unmodified same-batch control. Its
                # nonempty key list is required by the hook but it has no targets.
                source_positions.append([prompt_data["positions"][3]])
                ablator = BatchedAttentionEdgeAblator(
                    parts, targets, last_indices, source_positions
                )
                try:
                    raw = _forward(
                        model, parts, input_ids, attention_mask, answer_ids
                    )
                finally:
                    ablator.close()
                control = raw[-1]
                corrected = natural[None, :] + raw[:-1] - control[None, :]
                stop = scenario_start + len(chunk)
                arrays["intervened_logits"][ci, scenario_start:stop, qi] = corrected
                arrays["batch_control_minus_natural"][ci, qi, chunk_index] = (
                    control - natural
                )
            arrays["completed"][ci, qi] = True
            atomic_save_npz(output, **arrays)
        if qi == 0 or (qi + 1) % 10 == 0 or qi + 1 == len(trials):
            print(
                f"Feedback-token edge ablation: {int(arrays['completed'].sum())}/"
                f"{2 * len(trials)} condition-trials",
                flush=True,
            )

    metadata = {
        "config": config.as_dict(),
        "plan": str(plan_path),
        "n_questions": len(trials),
        "n_scenarios": len(scenarios),
        "scenario_batch_size": scenario_batch_size,
        "intervention": (
            "set selected exact final-query-to-source-token attention logits to "
            "negative infinity before softmax; remaining attention renormalizes"
        ),
        "batch_drift_correction": (
            "corrected intervention = natural single-row logits + intervention-row "
            "logits minus unmodified same-batch control logits"
        ),
        "first_prompt_audit": audit,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-batch-size", type=int, default=10)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.plan,
        args.output,
        args.scenario_batch_size,
        args.max_questions,
    )


if __name__ == "__main__":
    main()
