from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .collect_remapped_feedback_factorial import _messages, _remap_question
from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSelectiveGDNSourceWritePatcher
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


CONDITIONS = ("incorrect_again", "lost_again")
DIRECTIONS = ("evaluation_into_neutral", "neutral_into_evaluation")


def _locate_evaluation(tokenizer: Any, prompt: str, condition: str) -> dict[str, Any]:
    feedback = FACTORIAL_FEEDBACK[condition]
    start = prompt.find(feedback)
    if start < 0 or prompt.find(feedback, start + 1) >= 0:
        raise RuntimeError(f"Expected exactly one feedback sentence for {condition}")
    evaluation_word = "incorrect" if condition.startswith("incorrect_") else "lost"
    word_start = prompt.find(evaluation_word, start, start + len(feedback))
    if word_start < 0:
        raise RuntimeError(f"Could not locate {evaluation_word!r}")
    word_end = word_start + len(evaluation_word)
    period_start = prompt.find(".", word_end, start + len(feedback))
    if period_start < 0:
        raise RuntimeError("Could not locate evaluation-closing period")
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]

    def overlapping(left: int, right: int) -> list[int]:
        return [
            index for index, (a, b) in enumerate(offsets)
            if b > a and a < right and b > left
        ]

    word_positions = overlapping(word_start, word_end)
    period_positions = overlapping(period_start, period_start + 1)
    if len(word_positions) != 1 or len(period_positions) != 1:
        raise RuntimeError(
            f"Expected one evaluation word and period token; got "
            f"{word_positions}, {period_positions}"
        )
    return {
        "ids": ids,
        "word_position": word_positions[0],
        "period_position": period_positions[0],
        "word_token": tokenizer.decode([ids[word_positions[0]]]),
        "period_token": tokenizer.decode([ids[period_positions[0]]]),
    }


def _scenarios(
    stage: str,
    gla_layers: list[int],
    blocks: list[int],
    band_starts: list[int],
) -> list[dict]:
    if stage == "gate":
        return [{"id": "evaluation_period_all_gla", "layers_zero_based": gla_layers}]
    if stage == "bands":
        result = []
        starts = [value - 1 for value in band_starts] if band_starts else range(0, 64, 8)
        for start in starts:
            if start < 0 or start >= 64 or start % 8:
                raise ValueError(
                    "Band starts use human block numbers and must be one of "
                    "1, 9, 17, 25, 33, 41, 49, or 57"
                )
            layers = [layer for layer in gla_layers if start <= layer < start + 8]
            if layers:
                result.append({
                    "id": f"blocks_{start + 1:02d}_{start + 8:02d}",
                    "layers_zero_based": layers,
                })
        return result
    if stage == "blocks":
        if not blocks:
            raise ValueError("Block localization requires --blocks")
        selected = []
        for human_block in blocks:
            layer = int(human_block) - 1
            if layer not in gla_layers:
                raise ValueError(f"Human block {human_block} is not a GLA block")
            selected.append(layer)
        result = []
        for layer in selected:
            result.extend([
                {
                    "id": f"block_{layer + 1:02d}",
                    "layers_zero_based": [layer],
                },
                {
                    "id": f"all_gla_except_block_{layer + 1:02d}",
                    "layers_zero_based": [value for value in gla_layers if value != layer],
                },
            ])
        return result
    raise ValueError(f"Unknown stage: {stage}")


def _aggregate_logits(output: Any, variant_ids: dict[str, list[int]]) -> np.ndarray:
    import torch

    logits = output.logits.detach().float()
    final = logits[:, 0] if logits.shape[1] == 1 else logits[:, -1]
    return torch.stack([
        torch.logsumexp(final[:, variant_ids[letter]], dim=-1)
        for letter in LETTERS
    ], dim=-1).cpu().numpy()


def _forward(model: Any, parts: Any, input_ids: Any, attention_mask: Any):
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
            return model(**kwargs, logits_to_keep=1)
        except TypeError:
            return model(**kwargs)


def _initialize(
    path: Path,
    question_ids: list[str],
    scenarios: list[dict],
) -> dict[str, np.ndarray]:
    scenario_ids = [row["id"] for row in scenarios]
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != question_ids:
            raise ValueError("Existing output uses different question IDs")
        if arrays["scenario_ids"].astype(str).tolist() != scenario_ids:
            raise ValueError("Existing output uses different scenarios")
        return arrays
    n = len(question_ids)
    return {
        "question_ids": np.asarray(question_ids),
        "scenario_ids": np.asarray(scenario_ids),
        "completed": np.zeros(n, dtype=bool),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "patched_logits": np.full(
            (2, len(scenarios), n, 4), np.nan, dtype=np.float32
        ),
        "same_batch_control_minus_trusted": np.full(
            (2, n, 4), np.nan, dtype=np.float32
        ),
    }


def run(
    config_path: Path,
    remapping_plan_path: Path,
    split_plan_path: Path,
    evaluation_results_path: Path,
    neutral_results_path: Path,
    output: Path,
    stage: str,
    scenario_batch_size: int,
    blocks: list[int],
    band_starts: list[int],
    max_questions: int | None,
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

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    remapping_plan = json.loads(remapping_plan_path.read_text())
    mappings = {row["question_id"]: row for row in remapping_plan["rows"]}
    split_plan = json.loads(split_plan_path.read_text())
    question_ids = list(split_plan["question_ids"])
    if max_questions is not None:
        question_ids = question_ids[: int(max_questions)]
    evaluation_results = json.loads(evaluation_results_path.read_text())["results"]
    neutral_results = json.loads(neutral_results_path.read_text())["results"]
    required = set(question_ids)
    for name, rows in (
        ("manifest", questions), ("remapping plan", mappings),
        ("evaluation results", evaluation_results), ("neutral results", neutral_results),
    ):
        if not required <= set(rows):
            raise ValueError(f"{name} is missing requested questions")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    gla_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if len(gla_layers) != 48:
        raise RuntimeError(f"Expected 48 GLA layers, found {len(gla_layers)}")
    scenarios = _scenarios(stage, gla_layers, blocks, band_starts)
    arrays = _initialize(output, question_ids, scenarios)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = None

    for question_index, qid in enumerate(question_ids):
        if arrays["completed"][question_index]:
            continue
        question = questions[qid]
        mapping = mappings[qid]
        remapped = _remap_question(question, mapping["new_to_original"])
        condition_data = {}
        for condition in CONDITIONS:
            messages = _messages(config, question, remapped, condition)
            prompt = render_chat(
                processor, messages, config.disable_thinking, config.chat_serialization
            )
            located = _locate_evaluation(tokenizer, prompt, condition)
            condition_data[condition] = {
                "messages": messages,
                "prompt": prompt,
                "prompt_hash": prompt_hash(prompt),
                **located,
            }
        evaluation = condition_data["incorrect_again"]
        neutral = condition_data["lost_again"]
        if len(evaluation["ids"]) != len(neutral["ids"]):
            raise RuntimeError("Action-matched prompts are not token-length matched")
        if evaluation["period_position"] != neutral["period_position"]:
            raise RuntimeError("Evaluation periods are not position aligned")
        if evaluation["word_position"] != neutral["word_position"]:
            raise RuntimeError("Evaluation words are not position aligned")
        period_position = evaluation["period_position"]

        trusted_evaluation = np.asarray(
            evaluation_results[qid]["aggregated_ad_logits"], dtype=np.float32
        )
        trusted_neutral = np.asarray(
            neutral_results[qid]["aggregated_ad_logits"], dtype=np.float32
        )
        arrays["trusted_natural_logits"][0, question_index] = trusted_evaluation
        arrays["trusted_natural_logits"][1, question_index] = trusted_neutral

        for scenario_start in range(0, len(scenarios), scenario_batch_size):
            chunk = scenarios[scenario_start : scenario_start + scenario_batch_size]
            k = len(chunk)
            prompts = (
                [neutral["prompt"]] * k
                + [evaluation["prompt"]] * k
                + [neutral["prompt"], evaluation["prompt"]]
            )
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
            neutral_control_row = 2 * k
            evaluation_control_row = 2 * k + 1
            specs = {}
            for index, scenario in enumerate(chunk):
                layers = scenario["layers_zero_based"]
                specs[index] = (
                    evaluation_control_row, [period_position], layers
                )
                specs[k + index] = (
                    neutral_control_row, [period_position], layers
                )
            patcher = BatchedSelectiveGDNSourceWritePatcher(
                parts, specs, preserve_source_output=True
            )
            try:
                raw = _aggregate_logits(
                    _forward(model, parts, input_ids, attention_mask), variant_ids
                )
                patcher.assert_fired()
            finally:
                patcher.close()
            neutral_control = raw[neutral_control_row]
            evaluation_control = raw[evaluation_control_row]
            stop = scenario_start + k
            arrays["patched_logits"][
                0, scenario_start:stop, question_index
            ] = trusted_neutral[None, :] + raw[:k] - neutral_control[None, :]
            arrays["patched_logits"][
                1, scenario_start:stop, question_index
            ] = trusted_evaluation[None, :] + raw[k : 2 * k] - evaluation_control[None, :]
            arrays["same_batch_control_minus_trusted"][0, question_index] = (
                evaluation_control - trusted_evaluation
            )
            arrays["same_batch_control_minus_trusted"][1, question_index] = (
                neutral_control - trusted_neutral
            )

        arrays["completed"][question_index] = True
        atomic_save_npz(output, **arrays)
        if audit is None:
            audit = {
                "question_id": qid,
                "evaluation_prompt_hash": evaluation["prompt_hash"],
                "neutral_prompt_hash": neutral["prompt_hash"],
                "evaluation_word_token": evaluation["word_token"],
                "neutral_word_token": neutral["word_token"],
                "evaluation_period_token": evaluation["period_token"],
                "word_position_zero_based": evaluation["word_position"],
                "period_position_zero_based": period_position,
                "prompt_token_count": len(evaluation["ids"]),
                "evaluation_feedback": FACTORIAL_FEEDBACK["incorrect_again"],
                "neutral_feedback": FACTORIAL_FEEDBACK["lost_again"],
                "historical_assistant_content": evaluation["messages"][-2]["content"],
            }
        done = int(arrays["completed"].sum())
        if done == 1 or done % 5 == 0 or done == len(question_ids):
            print(
                f"Evaluation-update transplant ({stage}): {done}/{len(question_ids)}",
                flush=True,
            )

    metadata = {
        "config": config.as_dict(),
        "remapping_plan": str(remapping_plan_path),
        "split_plan": str(split_plan_path),
        "evaluation_results": str(evaluation_results_path),
        "neutral_results": str(neutral_results_path),
        "n_questions": len(question_ids),
        "stage": stage,
        "scenario_batch_size": scenario_batch_size,
        "scenarios": scenarios,
        "directions": list(DIRECTIONS),
        "array_axes": {
            "trusted_natural_logits": ["condition", "question", "displayed_answer"],
            "patched_logits": ["direction", "scenario", "question", "displayed_answer"],
            "same_batch_control_minus_trusted": ["condition", "question", "displayed_answer"],
        },
        "condition_axis": ["Evaluation", "Matched Neutral"],
        "gla_layers_zero_based": gla_layers,
        "intervention": (
            "At the evaluation-closing period, copy the paired source row's GLA key, "
            "value, decay gate g, and write strength beta into the target row at selected "
            "GLA layers. The period token's own output remains target-condition natural; "
            "only causally later recurrent outputs receive the donor write. Query and "
            "every other token remain target-condition natural."
        ),
        "preserve_source_output": True,
        "batch_drift_correction": (
            "corrected patched logits = trusted exact natural logits + patched same-batch "
            "logits - untouched same-batch target-control logits"
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
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transplant action-matched evaluation-period GLA updates"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--evaluation-results", type=Path, required=True)
    parser.add_argument("--neutral-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("gate", "bands", "blocks"), required=True)
    parser.add_argument("--scenario-batch-size", type=int, default=3)
    parser.add_argument("--blocks", type=int, nargs="*", default=[])
    parser.add_argument("--band-starts", type=int, nargs="*", default=[])
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.remapping_plan,
        args.split_plan,
        args.evaluation_results,
        args.neutral_results,
        args.output,
        args.stage,
        args.scenario_batch_size,
        args.blocks,
        args.band_starts,
        args.max_questions,
    )


if __name__ == "__main__":
    main()
