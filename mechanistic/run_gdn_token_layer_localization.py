from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSelectiveGDNSourceWriteAblator
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


DEFAULT_CONDITIONS = ("incorrect", "neutral")


def _final_logits(output, answer_ids: list[int]) -> np.ndarray:
    logits = output.logits
    final = logits[:, 0] if logits.shape[1] == 1 else logits[:, -1]
    return final[:, answer_ids].detach().float().cpu().numpy()


def _forward(model, parts, input_ids, attention_mask, answer_ids):
    import torch

    device = model_input_device(parts)
    with torch.inference_mode():
        kwargs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
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
        yield start, values[start : start + size]


def run(
    config_path: Path,
    plan_path: Path,
    output: Path,
    scenario_batch_size: int,
    max_questions: int | None,
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")
    plan = json.loads(plan_path.read_text())
    scenarios = plan["scenarios"]
    if not scenarios:
        raise ValueError("Localization plan contains no scenarios")
    if not conditions or not set(conditions) <= set(DEFAULT_CONDITIONS):
        raise ValueError(f"Conditions must be drawn from {DEFAULT_CONDITIONS}")
    question_ids = plan["question_ids"]
    if max_questions is not None:
        question_ids = question_ids[: int(max_questions)]
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        question_ids,
        None,
    )
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    answer_ids = [resolved[letter][0][1] for letter in "ABCD"]
    gdn_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if len(gdn_layers) != 48:
        raise RuntimeError(f"Expected 48 GLA layers, found {len(gdn_layers)}")
    for scenario in scenarios:
        invalid = set(scenario["layers_zero_based"]) - set(gdn_layers)
        if invalid:
            raise ValueError(f"Scenario {scenario['id']} selects non-GLA layers: {invalid}")
        if not scenario["feedback_token_indices_zero_based"]:
            raise ValueError(f"Scenario {scenario['id']} has no feedback tokens")

    output.parent.mkdir(parents=True, exist_ok=True)
    qids = np.asarray([trial.question_id for trial in trials])
    scenario_ids = np.asarray([row["id"] for row in scenarios])
    n_chunks = (len(scenarios) + scenario_batch_size - 1) // scenario_batch_size
    if output.exists():
        with np.load(output, allow_pickle=False) as existing:
            arrays = {key: existing[key] for key in existing.files}
        if arrays["question_ids"].astype(str).tolist() != qids.astype(str).tolist():
            raise ValueError("Existing output uses different question IDs")
        if arrays["scenario_ids"].astype(str).tolist() != scenario_ids.astype(str).tolist():
            raise ValueError("Existing output uses different scenarios")
    else:
        arrays = {
            "question_ids": qids,
            "scenario_ids": scenario_ids,
            "completed": np.zeros((len(conditions), len(trials)), dtype=bool),
            "natural_logits": np.full(
                (len(conditions), len(trials), 4), np.nan, dtype=np.float32
            ),
            "intervened_logits": np.full(
                (len(conditions), len(scenarios), len(trials), 4),
                np.nan,
                dtype=np.float32,
            ),
            "batch_control_minus_natural": np.full(
                (len(conditions), len(trials), n_chunks, 4),
                np.nan,
                dtype=np.float32,
            ),
        }

    audit = None
    for question_index, trial in enumerate(trials):
        for condition_index, condition in enumerate(conditions):
            if arrays["completed"][condition_index, question_index]:
                continue
            prompt_data = _render_and_locate(processor, tokenizer, config, trial, condition)
            natural = _forward(
                model,
                parts,
                prompt_data["input_ids"],
                prompt_data["attention_mask"],
                answer_ids,
            )[0]
            arrays["natural_logits"][condition_index, question_index] = natural
            for chunk_index, (scenario_start, chunk) in enumerate(
                _chunks(scenarios, scenario_batch_size)
            ):
                prompts = [prompt_data["prompt"]] * (len(chunk) + 1)
                input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
                specs = {}
                for row, scenario in enumerate(chunk):
                    positions = [
                        prompt_data["positions"][int(index)]
                        for index in scenario["feedback_token_indices_zero_based"]
                    ]
                    specs[row] = (positions, scenario["layers_zero_based"])
                ablator = BatchedSelectiveGDNSourceWriteAblator(parts, specs)
                try:
                    raw = _forward(model, parts, input_ids, attention_mask, answer_ids)
                    ablator.assert_fired()
                finally:
                    ablator.close()
                control = raw[-1]
                corrected = natural[None, :] + raw[:-1] - control[None, :]
                stop = scenario_start + len(chunk)
                arrays["intervened_logits"][
                    condition_index, scenario_start:stop, question_index
                ] = corrected
                arrays["batch_control_minus_natural"][
                    condition_index, question_index, chunk_index
                ] = control - natural
            arrays["completed"][condition_index, question_index] = True
            atomic_save_npz(output, **arrays)
            if audit is None:
                audit = {
                    "question_id": trial.question_id,
                    "condition": condition,
                    "feedback_tokens": list(prompt_data["tokens"]),
                    "feedback_positions_zero_based": prompt_data["positions"],
                }
        if question_index == 0 or (question_index + 1) % 5 == 0 or question_index + 1 == len(trials):
            print(
                f"GLA token-layer localization: {int(arrays['completed'].sum())}/"
                f"{len(conditions) * len(trials)} condition-trials",
                flush=True,
            )

    metadata = {
        "config": config.as_dict(),
        "plan": str(plan_path),
        "n_questions": len(trials),
        "n_scenarios": len(scenarios),
        "scenario_batch_size": scenario_batch_size,
        "conditions": list(conditions),
        "gdn_layers_zero_based": gdn_layers,
        "intervention": (
            "set beta=0 in every GLA value head for the selected exact feedback-token "
            "positions and selected model layers"
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
    parser = argparse.ArgumentParser(description="Localize GLA feedback writes by token and layer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-batch-size", type=int, default=10)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=DEFAULT_CONDITIONS,
        default=list(DEFAULT_CONDITIONS),
    )
    args = parser.parse_args()
    run(
        args.config,
        args.plan,
        args.output,
        args.scenario_batch_size,
        args.max_questions,
        tuple(args.conditions),
    )


if __name__ == "__main__":
    main()
