from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import (
    TOKEN_MATCHED_TEST_GAME_FEEDBACK,
    TOKEN_MATCHED_TEST_NEUTRAL_FEEDBACK,
    build_messages,
    load_trials,
    prompt_hash,
)
from .run_baseline_mixer_function import _forward


CONDITIONS = ("incorrect", "neutral")
FEEDBACKS = {
    "incorrect": TOKEN_MATCHED_TEST_GAME_FEEDBACK,
    "neutral": TOKEN_MATCHED_TEST_NEUTRAL_FEEDBACK,
}
EXPECTED_TOKENS = {
    "incorrect": ("Your", "answer", "was", "incorrect", ".", "Choose", "a", "different", "answer", "."),
    "neutral": ("Your", "answer", "was", "lost", ".", "Choose", "the", "answer", "again", "."),
}
SCENARIOS = tuple([f"token_{index:02d}" for index in range(1, 11)]) + (
    "evaluation_clause",
    "action_clause",
    "full_feedback",
)
SCENARIO_TOKEN_INDICES = tuple([(index,) for index in range(10)]) + (
    tuple(range(5)),
    tuple(range(5, 10)),
    tuple(range(10)),
)


def _hidden(value: Any):
    return value[0] if isinstance(value, (tuple, list)) else value


class LayerInputTrajectoryCollector:
    """Capture selected token states immediately before every model block."""

    def __init__(self, parts: Any, positions: list[int]) -> None:
        self.positions = positions
        self.values: dict[int, Any] = {}
        self.handles = [
            layer.register_forward_pre_hook(self._hook(index))
            for index, layer in enumerate(parts.layers)
        ]

    def _hook(self, layer: int):
        def capture(_module: Any, inputs: tuple[Any, ...]):
            hidden = inputs[0]
            if hidden.shape[0] != 1:
                raise ValueError("Trajectory collection expects batch size one")
            self.values[layer] = hidden[0, self.positions].detach().clone()

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class BatchedTokenTrajectoryPatcher:
    """Clamp selected target positions to paired source states at every block."""

    def __init__(
        self,
        parts: Any,
        target_positions: list[int],
    ) -> None:
        self.target_positions = target_positions
        self.handles = [
            layer.register_forward_pre_hook(self._hook(index))
            for index, layer in enumerate(parts.layers)
        ]

    def _hook(self, layer: int):
        def patch(_module: Any, inputs: tuple[Any, ...]):
            hidden = inputs[0]
            expected = len(SCENARIOS) + 2
            if hidden.shape[0] != expected:
                raise ValueError(f"Expected {expected} batch rows, got {hidden.shape[0]}")
            updated = hidden.clone()
            # The final row is the paired natural source prompt in this same
            # physical batch. Use its live layer-input states, avoiding the
            # substantial batch-composition drift of Qwen's GLA kernels.
            source = hidden[-1, self.target_positions]
            for row, selected in enumerate(SCENARIO_TOKEN_INDICES):
                positions = [self.target_positions[index] for index in selected]
                updated[row, positions] = source[list(selected)]
            return (updated, *inputs[1:])

        return patch

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _final_logits(output: Any, answer_ids: list[int]):
    logits = output.logits
    final = logits[:, 0] if logits.shape[1] == 1 else logits[:, -1]
    return final[:, answer_ids].detach().float().cpu().numpy()


def _render_and_locate(
    processor: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    trial: Any,
    condition: str,
):
    messages = build_messages(
        trial.question,
        condition,
        config.prompt_mode,
        config.feedback_variant,
    )
    prompt = render_chat(
        processor, messages, config.disable_thinking, config.chat_serialization
    )
    feedback = FEEDBACKS[condition]
    start = prompt.find(feedback)
    if start < 0 or prompt.find(feedback, start + 1) >= 0:
        raise RuntimeError(f"Expected exactly one feedback occurrence for {condition}")
    end = start + len(feedback)
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    ]
    tokens = tuple(tokenizer.decode([ids[position]]).strip() for position in positions)
    if tokens != EXPECTED_TOKENS[condition]:
        raise RuntimeError(
            f"Unexpected feedback tokenization for {condition}: {tokens!r}"
        )
    input_ids, attention_mask, _ = tokenize_batch(tokenizer, [prompt])
    if input_ids[0].tolist() != ids:
        raise RuntimeError("Offset-aware and model tokenizations disagree")
    return {
        "messages": messages,
        "prompt": prompt,
        "prompt_hash": prompt_hash(prompt),
        "ids": ids,
        "positions": positions,
        "tokens": tokens,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def _natural_forward(model, parts, prompt_data, answer_ids):
    collector = LayerInputTrajectoryCollector(parts, prompt_data["positions"])
    try:
        output = _forward(
            model,
            parts,
            prompt_data["input_ids"],
            prompt_data["attention_mask"],
        )
    finally:
        collector.close()
    if len(collector.values) != len(parts.layers):
        raise RuntimeError("Not every layer input was captured")
    return _final_logits(output, answer_ids)[0], collector.values


def _patched_forward(
    model,
    parts,
    tokenizer,
    target_prompt,
    source_prompt,
    natural_logits,
    source_natural_logits,
    answer_ids,
):
    # Thirteen intervention rows, an untouched target control, and an
    # untouched source control.  The full-feedback row and source control are
    # identical at the input to block 1, so their same-batch logits provide an
    # exact intervention-identity check even for batch-sensitive GLA kernels.
    prompts = (
        [target_prompt["prompt"]] * (len(SCENARIOS) + 1)
        + [source_prompt["prompt"]]
    )
    input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
    patcher = BatchedTokenTrajectoryPatcher(
        parts,
        target_prompt["positions"],
    )
    try:
        output = _forward(model, parts, input_ids, attention_mask)
    finally:
        patcher.close()
    raw = _final_logits(output, answer_ids)
    target_control = raw[len(SCENARIOS)]
    source_control = raw[len(SCENARIOS) + 1]
    corrected = natural_logits[None, :] + raw[: len(SCENARIOS)] - target_control
    full_index = SCENARIOS.index("full_feedback")
    full_raw_error = raw[full_index] - source_control
    # The complete swap makes the target sequence exactly the source sequence
    # before block 1. Anchor its saved absolute logits to the paired natural
    # source; the same-batch error records and verifies that identity directly.
    corrected[full_index] = source_natural_logits + full_raw_error
    return corrected, full_raw_error


def _initialize(path: Path, question_ids: list[str], n_layers: int):
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != question_ids:
            raise ValueError("Question IDs changed")
        return arrays
    n = len(question_ids)
    return {
        "question_ids": np.asarray(question_ids),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "patched_logits": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
        "game_neutral_token_state_distance": np.full(
            (n, n_layers, 10), np.nan, dtype=np.float32
        ),
        "full_swap_same_batch_logit_error": np.full(
            (2, n, 4), np.nan, dtype=np.float32
        ),
    }


def run(config_path: Path, output: Path, max_questions: int | None) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("This test requires baseline_matched_empty_history")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("This test requires explicit raw_qwen_chatml")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("This test requires feedback_variant=token_matched_test")
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        max_questions=max_questions,
    )
    question_ids = [trial.question_id for trial in trials]
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "feedback_token_trajectory_swap_results.npz"

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    answer_ids = [resolved[letter][0][1] for letter in "ABCD"]
    arrays = _initialize(result_path, question_ids, len(parts.layers))

    for question_index, trial in enumerate(trials):
        if bool(arrays["completed"][question_index]):
            continue
        prompt_data = {
            condition: _render_and_locate(
                processor, tokenizer, config, trial, condition
            )
            for condition in CONDITIONS
        }
        game = prompt_data["incorrect"]
        neutral = prompt_data["neutral"]
        if len(game["ids"]) != len(neutral["ids"]):
            raise RuntimeError("Game and Neutral prompt lengths differ")
        if game["positions"] != neutral["positions"]:
            raise RuntimeError("Feedback token positions are not aligned")
        feedback_positions = set(game["positions"])
        mismatches = {
            index
            for index, (a, b) in enumerate(zip(game["ids"], neutral["ids"]))
            if a != b
        }
        if not mismatches <= feedback_positions:
            raise RuntimeError(f"Prompts differ outside feedback: {sorted(mismatches)}")

        natural: dict[str, np.ndarray] = {}
        trajectories: dict[str, dict[int, Any]] = {}
        for condition_index, condition in enumerate(CONDITIONS):
            logits, trajectory = _natural_forward(
                model, parts, prompt_data[condition], answer_ids
            )
            natural[condition] = logits
            trajectories[condition] = trajectory
            arrays["natural_logits"][condition_index, question_index] = logits

        arrays["game_neutral_token_state_distance"][question_index] = np.stack(
            [
                torch.linalg.vector_norm(
                    trajectories["incorrect"][layer].float()
                    - trajectories["neutral"][layer].float(),
                    dim=1,
                ).cpu().numpy()
                for layer in range(len(parts.layers))
            ]
        )
        patched, error = _patched_forward(
            model,
            parts,
            tokenizer,
            game,
            neutral,
            natural["incorrect"],
            natural["neutral"],
            answer_ids,
        )
        arrays["patched_logits"][0, :, question_index] = patched
        arrays["full_swap_same_batch_logit_error"][0, question_index] = error
        patched, error = _patched_forward(
            model,
            parts,
            tokenizer,
            neutral,
            game,
            natural["neutral"],
            natural["incorrect"],
            answer_ids,
        )
        arrays["patched_logits"][1, :, question_index] = patched
        arrays["full_swap_same_batch_logit_error"][1, question_index] = error
        arrays["completed"][question_index] = True

        if question_index == 0 and not (output / "prompt_and_token_audit.json").exists():
            audit = {
                "question_id": trial.question_id,
                "feedback_variant": config.feedback_variant,
                "conditions": {
                    condition: {
                        "feedback": FEEDBACKS[condition],
                        "tokens": list(prompt_data[condition]["tokens"]),
                        "token_ids": [
                            prompt_data[condition]["ids"][position]
                            for position in prompt_data[condition]["positions"]
                        ],
                        "absolute_positions": prompt_data[condition]["positions"],
                        "prompt_hash": prompt_data[condition]["prompt_hash"],
                        "messages": prompt_data[condition]["messages"],
                        "rendered_prompt": prompt_data[condition]["prompt"],
                    }
                    for condition in CONDITIONS
                },
                "scenario_token_indices_zero_based": {
                    name: list(indices)
                    for name, indices in zip(SCENARIOS, SCENARIO_TOKEN_INDICES)
                },
            }
            (output / "prompt_and_token_audit.json").write_text(
                json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
            )

        done = int(arrays["completed"].sum())
        if done == 1 or done % 5 == 0 or done == len(trials):
            atomic_save_npz(result_path, **arrays)
            print(f"feedback token trajectory swap: {done}/{len(trials)}", flush=True)
        del trajectories

    metadata = {
        "status": "prompt_variant_test_only",
        "config": config.as_dict(),
        "n_questions": len(trials),
        "conditions": list(CONDITIONS),
        "directions": ["Neutral into Game", "Game into Neutral"],
        "feedbacks": FEEDBACKS,
        "tokens": {key: list(value) for key, value in EXPECTED_TOKENS.items()},
        "scenarios": {
            name: list(indices)
            for name, indices in zip(SCENARIOS, SCENARIO_TOKEN_INDICES)
        },
        "intervention": (
            "At every model block input, clamp the selected aligned feedback-token "
            "positions to the paired same-question states from the other condition."
        ),
        "batch_control": (
            "Thirteen intervention rows, one unpatched target control, and one "
            "unpatched source control. Partial-swap logits are natural target logits "
            "plus intervention-minus-target-control logits. The complete swap is "
            "anchored to paired natural source logits and verified against the "
            "same-batch source control."
        ),
        "resolved_answer_tokens": resolved,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    run(args.config, args.output, args.max_questions)


if __name__ == "__main__":
    main()
