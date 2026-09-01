from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSelectiveGDNSourceWritePatcher
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


ALL_SCENARIOS = (
    {
        "id": "evaluation_period_all_gla",
        "feedback_token_indices_zero_based": [4],
    },
    {
        "id": "action_clause_all_gla",
        "feedback_token_indices_zero_based": [5, 6, 7, 8, 9],
    },
    {
        "id": "evaluation_plus_action_all_gla",
        "feedback_token_indices_zero_based": [4, 5, 6, 7, 8, 9],
    },
)

EVALUATION_SELECTED_LAYERS = [17, 24, 25, 28, 34, 40, 45, 46]
ACTION_SELECTED_LAYERS = [4, 17, 21, 25, 32, 33, 44, 45, 46, 49, 50, 53, 54]
SELECTED_SCENARIOS = (
    {
        "id": "evaluation_period_selected_8_gla",
        "patch_groups": [
            {
                "feedback_token_indices_zero_based": [4],
                "layers_zero_based": EVALUATION_SELECTED_LAYERS,
            }
        ],
    },
    {
        "id": "action_clause_selected_13_gla",
        "patch_groups": [
            {
                "feedback_token_indices_zero_based": [5, 6, 7, 8, 9],
                "layers_zero_based": ACTION_SELECTED_LAYERS,
            }
        ],
    },
    {
        "id": "evaluation_8_plus_action_13_gla",
        "patch_groups": [
            {
                "feedback_token_indices_zero_based": [4],
                "layers_zero_based": EVALUATION_SELECTED_LAYERS,
            },
            {
                "feedback_token_indices_zero_based": [5, 6, 7, 8, 9],
                "layers_zero_based": ACTION_SELECTED_LAYERS,
            },
        ],
    },
)


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


def _initialize(
    path: Path,
    question_ids: list[str],
    scenarios: tuple[dict, ...],
) -> dict[str, np.ndarray]:
    scenario_ids = [row["id"] for row in scenarios]
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != question_ids:
            raise ValueError("Existing result uses different question IDs")
        if arrays["scenario_ids"].astype(str).tolist() != scenario_ids:
            raise ValueError("Existing result uses different scenarios")
        return arrays
    n = len(question_ids)
    return {
        "question_ids": np.asarray(question_ids),
        "scenario_ids": np.asarray(scenario_ids),
        "baseline_answer_indices": np.full(n, -1, dtype=np.int8),
        "baseline_correct": np.zeros(n, dtype=bool),
        "completed": np.zeros(n, dtype=bool),
        "natural_game_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "natural_neutral_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "patched_neutral_logits": np.full(
            (len(scenarios), n, 4), np.nan, dtype=np.float32
        ),
        "neutral_batch_control_minus_natural": np.full(
            (n, 4), np.nan, dtype=np.float32
        ),
        "game_batch_control_minus_natural": np.full(
            (n, 4), np.nan, dtype=np.float32
        ),
    }


def run(
    config_path: Path,
    confirmation_plan_path: Path,
    output: Path,
    max_questions: int | None,
    scope: str,
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

    plan = json.loads(confirmation_plan_path.read_text())
    question_ids = list(plan["question_ids"])
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
    gla_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if len(gla_layers) != 48:
        raise RuntimeError(f"Expected 48 GLA layers, found {len(gla_layers)}")

    scenarios = SELECTED_SCENARIOS if scope == "selected" else ALL_SCENARIOS
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = _initialize(output, [trial.question_id for trial in trials], scenarios)
    audit = None
    for question_index, trial in enumerate(trials):
        if arrays["completed"][question_index]:
            continue
        game = _render_and_locate(processor, tokenizer, config, trial, "incorrect")
        neutral = _render_and_locate(processor, tokenizer, config, trial, "neutral")
        if game["positions"] != neutral["positions"]:
            raise RuntimeError("Game and Neutral feedback positions are not token aligned")
        if len(game["ids"]) != len(neutral["ids"]):
            raise RuntimeError("Game and Neutral prompt lengths differ")

        game_natural = _forward(
            model, parts, game["input_ids"], game["attention_mask"], answer_ids
        )[0]
        neutral_natural = _forward(
            model, parts, neutral["input_ids"], neutral["attention_mask"], answer_ids
        )[0]
        arrays["natural_game_logits"][question_index] = game_natural
        arrays["natural_neutral_logits"][question_index] = neutral_natural
        arrays["baseline_answer_indices"][question_index] = "ABCD".index(
            trial.baseline_answer
        )
        arrays["baseline_correct"][question_index] = trial.baseline_correct

        # Rows 0..2 are patched Neutral scenarios, row 3 is an untouched
        # same-batch Neutral control, and row 4 is the live paired Game donor.
        prompts = [neutral["prompt"]] * (len(scenarios) + 1) + [game["prompt"]]
        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        donor_row = len(scenarios) + 1
        specs = {}
        for target_row, scenario in enumerate(scenarios):
            patch_groups = scenario.get("patch_groups")
            if patch_groups is None:
                patch_groups = [{
                    "feedback_token_indices_zero_based": scenario[
                        "feedback_token_indices_zero_based"
                    ],
                    "layers_zero_based": gla_layers,
                }]
            specs[target_row] = [
                (
                    donor_row,
                    [
                        neutral["positions"][index]
                        for index in group["feedback_token_indices_zero_based"]
                    ],
                    group["layers_zero_based"],
                )
                for group in patch_groups
            ]
        patcher = BatchedSelectiveGDNSourceWritePatcher(parts, specs)
        try:
            raw = _forward(model, parts, input_ids, attention_mask, answer_ids)
            patcher.assert_fired()
        finally:
            patcher.close()
        neutral_control = raw[len(scenarios)]
        game_control = raw[donor_row]
        arrays["patched_neutral_logits"][:, question_index] = (
            neutral_natural[None, :] + raw[: len(scenarios)] - neutral_control[None, :]
        )
        arrays["neutral_batch_control_minus_natural"][question_index] = (
            neutral_control - neutral_natural
        )
        arrays["game_batch_control_minus_natural"][question_index] = (
            game_control - game_natural
        )
        arrays["completed"][question_index] = True
        atomic_save_npz(output, **arrays)
        if audit is None:
            audit = {
                "question_id": trial.question_id,
                "game_feedback_tokens": list(game["tokens"]),
                "neutral_feedback_tokens": list(neutral["tokens"]),
                "feedback_positions_zero_based": game["positions"],
                "game_prompt_hash": game["prompt_hash"],
                "neutral_prompt_hash": neutral["prompt_hash"],
            }
        done = int(arrays["completed"].sum())
        if done == 1 or done % 5 == 0 or done == len(trials):
            print(f"Game->Neutral GLA write patching: {done}/{len(trials)}", flush=True)

    metadata = {
        "config": config.as_dict(),
        "confirmation_plan": str(confirmation_plan_path),
        "n_questions": len(trials),
        "scope": scope,
        "scenarios": list(scenarios),
        "gla_layers_zero_based": gla_layers,
        "intervention": (
            "For the selected feedback positions in all 48 GLA layers, copy the paired "
            "Game row's key, value, decay gate g, and write strength beta into Neutral. "
            "These tensors determine the recurrent memory write; query remains Neutral."
        ),
        "batch_drift_correction": (
            "corrected patched Neutral = natural two-row Neutral logits + patched-row "
            "logits minus untouched same-batch Neutral-control logits"
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
    parser = argparse.ArgumentParser(
        description="Patch Game GLA feedback-source memory writes into Neutral"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirmation-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--scope", choices=("all", "selected"), default="all")
    args = parser.parse_args()
    run(args.config, args.confirmation_plan, args.output, args.max_questions, args.scope)


if __name__ == "__main__":
    main()
