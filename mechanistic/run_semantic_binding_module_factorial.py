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
from .prompts import (
    FACTORIAL_FEEDBACK,
    build_factorial_messages,
    prompt_hash,
    repeated_question_turn,
)
from .sublayer import _hidden, _replace_hidden


CELLS = ("evaluation_x", "neutral_x", "evaluation_y", "neutral_y")
CONDITIONS = ("incorrect_again", "lost_again", "incorrect_again", "lost_again")
TARGETS = ("x", "y")
ANCHORS = ("evaluation_period", "repeated_candidate", "decision")
MODULE_MODES = ("attention", "mlp", "both")
SOURCE_ANCHORS = ("evaluation_period", "repeated_x", "repeated_y", "decision")


def _remap_question(question: dict, new_to_original: dict[str, str]) -> dict:
    remapped = copy.deepcopy(question)
    remapped["options"] = {
        new: question["options"][original]
        for new, original in new_to_original.items()
    }
    inverse = {original: new for new, original in new_to_original.items()}
    remapped["correct_answer"] = inverse[question["correct_answer"]]
    return remapped


def _messages(
    config: ExperimentConfig,
    first_question: dict,
    second_question: dict,
    condition: str,
) -> list[dict]:
    messages = build_factorial_messages(
        first_question, condition, config.prompt_mode
    )
    original_repeat = repeated_question_turn(first_question)
    if not messages[-1]["content"].endswith(original_repeat):
        raise RuntimeError("Could not locate the first question's repeated turn")
    messages[-1]["content"] = (
        messages[-1]["content"][: -len(original_repeat)]
        + repeated_question_turn(second_question)
    )
    return messages


def _overlap_position(
    offsets: list[tuple[int, int]], start: int, end: int
) -> int:
    matches = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one token overlapping {start}:{end}, got {matches}"
        )
    return matches[0]


def _positions(
    tokenizer: Any,
    prompt: str,
    condition: str,
    second_question: dict,
    x_second_letter: str,
    y_second_letter: str,
) -> tuple[list[int], dict[str, Any]]:
    encoded = tokenizer(
        prompt, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]

    feedback = FACTORIAL_FEEDBACK[condition]
    feedback_start = prompt.find(feedback)
    if feedback_start < 0 or prompt.find(feedback, feedback_start + 1) >= 0:
        raise RuntimeError(f"Expected one exact feedback sentence: {feedback!r}")
    evaluation_word = "incorrect" if condition.startswith("incorrect") else "lost"
    word_start = prompt.find(
        evaluation_word, feedback_start, feedback_start + len(feedback)
    )
    period_start = prompt.find(
        ".", word_start + len(evaluation_word), feedback_start + len(feedback)
    )
    if word_start < 0 or period_start < 0:
        raise RuntimeError("Could not locate evaluation word and closing period")
    evaluation_period = _overlap_position(
        offsets, period_start, period_start + 1
    )

    def option_newline(letter: str) -> int:
        line = f"  {letter}: {second_question['options'][letter]}\n"
        line_start = prompt.rfind(line)
        if line_start < 0:
            raise RuntimeError(f"Could not locate repeated option line {line!r}")
        newline = line_start + len(line) - 1
        return _overlap_position(offsets, newline, newline + 1)

    repeated_x = option_newline(x_second_letter)
    repeated_y = option_newline(y_second_letter)
    positions = [evaluation_period, repeated_x, repeated_y, len(ids) - 1]
    audit = {
        "length": len(ids),
        "tokens": {
            name: tokenizer.decode([ids[position]])
            for name, position in zip(SOURCE_ANCHORS, positions)
        },
        "positions": dict(zip(SOURCE_ANCHORS, positions)),
        "prompt_hash": prompt_hash(prompt),
    }
    return positions, audit


def _component_modules(parts: Any):
    ordinary_attention = {}
    mlps = {}
    for layer_index, layer in enumerate(parts.layers):
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            ordinary_attention[f"attention_l{layer_index}"] = attention
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise RuntimeError(f"Layer {layer_index} has no MLP")
        mlps[f"mlp_l{layer_index}"] = mlp
    if len(ordinary_attention) != 16 or len(mlps) != 64:
        raise RuntimeError(
            f"Expected 16 ordinary-attention modules and 64 MLPs, found "
            f"{len(ordinary_attention)} and {len(mlps)}"
        )
    return ordinary_attention, mlps


class NaturalComponentCollector:
    def __init__(
        self,
        modules: dict[str, Any],
        padded_positions: list[list[int]],
    ) -> None:
        import torch

        self.positions = torch.as_tensor(padded_positions, dtype=torch.long)
        self.values: dict[str, Any] = {}
        self.handles = [
            module.register_forward_hook(self._hook(key))
            for key, module in modules.items()
        ]

    def _hook(self, key: str):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = _hidden(output)
            positions = self.positions.to(hidden.device)
            rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
            self.values[key] = hidden[rows, positions].detach().to(
                "cpu", dtype=torch.float16
            )

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class NoInteractionPatcher:
    """Patch three intervention rows and leave the fourth as a batch control."""

    def __init__(
        self,
        attention: dict[str, Any],
        mlps: dict[str, Any],
        source: dict[str, Any],
        source_anchor_index: int,
        target_cell: int,
        padded_target_position: int,
    ) -> None:
        import torch

        if target_cell == 0:
            # E/X with its factorial interaction removed.
            cell_weights = (0.0, 1.0, 1.0, -1.0)
        elif target_cell == 2:
            # E/Y with its factorial interaction removed.
            cell_weights = (1.0, -1.0, 0.0, 1.0)
        else:
            raise ValueError("Only Evaluation/X and Evaluation/Y are patched")
        self.position = int(padded_target_position)
        self.replacements: dict[str, Any] = {}
        for key, values in source.items():
            selected = values[:, source_anchor_index].float()
            replacement = sum(
                weight * selected[cell]
                for cell, weight in enumerate(cell_weights)
                if weight
            )
            self.replacements[key] = replacement.to(dtype=torch.float16)
        self.handles = []
        for key, module in {**attention, **mlps}.items():
            is_attention = key.startswith("attention_")
            rows = []
            if is_attention:
                rows.extend((0, 2))  # attention-only and both
            else:
                rows.extend((1, 2))  # MLP-only and both
            self.handles.append(
                module.register_forward_hook(self._hook(key, rows))
            )

    def _hook(self, key: str, rows: list[int]):
        def patch(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = _hidden(output)
            updated = hidden.clone()
            replacement = self.replacements[key].to(
                device=hidden.device, dtype=hidden.dtype
            )
            for row in rows:
                updated[row, self.position] = replacement
            return _replace_hidden(output, updated)

        return patch

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _forward(model: Any, parts: Any, input_ids: Any, attention_mask: Any):
    import torch

    with torch.inference_mode():
        kwargs = {
            "input_ids": input_ids.to(model_input_device(parts)),
            "attention_mask": attention_mask.to(model_input_device(parts)),
            "use_cache": False,
            "return_dict": True,
        }
        try:
            return model(**kwargs, logits_to_keep=1)
        except TypeError:
            return model(**kwargs)


def _aggregate_logits(output: Any, variant_ids: dict[str, list[int]]) -> np.ndarray:
    import torch

    logits = output.logits.detach().float()
    final = logits[:, 0] if logits.shape[1] == 1 else logits[:, -1]
    return torch.stack(
        [
            torch.logsumexp(final[:, variant_ids[letter]], dim=-1)
            for letter in LETTERS
        ],
        dim=-1,
    ).cpu().numpy()


def _initialize(path: Path, rows: list[dict], n_attn: int, n_mlp: int) -> dict:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != [
            row["question_id"] for row in rows
        ]:
            raise ValueError("Existing output uses different questions")
        return arrays
    n = len(rows)
    return {
        "question_ids": np.asarray([row["question_id"] for row in rows]),
        "x_content": np.asarray([row["x_content_original_letter"] for row in rows]),
        "y_content": np.asarray([row["y_content_original_letter"] for row in rows]),
        "x_second_letter": np.asarray([row["x_second_letter"] for row in rows]),
        "y_second_letter": np.asarray([row["y_second_letter"] for row in rows]),
        "natural_completed": np.zeros(n, dtype=bool),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((len(CELLS), n, 4), np.nan, np.float32),
        "patched_logits": np.full(
            (len(TARGETS), len(ANCHORS), len(MODULE_MODES), n, 4),
            np.nan,
            np.float32,
        ),
        "batch_control_minus_natural": np.full(
            (len(TARGETS), len(ANCHORS), n, 4), np.nan, np.float32
        ),
        "attention_interaction_relative_norm": np.full(
            (n, len(SOURCE_ANCHORS), n_attn), np.nan, np.float32
        ),
        "mlp_interaction_relative_norm": np.full(
            (n, len(SOURCE_ANCHORS), n_mlp), np.nan, np.float32
        ),
    }


def _interaction_relative_norm(values: Any) -> np.ndarray:
    interaction = values[0].float() - values[1].float() - values[2].float() + values[3].float()
    denominator = values.float().norm(dim=-1).mean(dim=0).clamp_min(1e-8)
    return (interaction.norm(dim=-1) / denominator).numpy().astype(np.float32)


def run(
    config_path: Path,
    cohort_path: Path,
    output: Path,
    split: str,
    stage: str,
    max_questions: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires exact raw_qwen_chatml serialization")
    if config.attn_implementation != "sdpa" or config.batch_size != 4:
        raise ValueError("Requires the established batch-4 SDPA regime")
    if split not in {"discovery", "confirmation"}:
        raise ValueError("split must be discovery or confirmation")
    if stage not in {"natural", "full"}:
        raise ValueError("stage must be natural or full")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    cohort = json.loads(cohort_path.read_text())
    rows = [row for row in cohort["rows"] if row["split"] == split]
    if max_questions is not None:
        rows = rows[: int(max_questions)]
    if not rows:
        raise ValueError("No eligible questions")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    attention, mlps = _component_modules(parts)
    modules = {**attention, **mlps}
    arrays = _initialize(output, rows, len(attention), len(mlps))
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = output.with_name(output.stem + "_prompt_audit.json")
    metadata_path = output.with_name(output.stem + "_metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "config": config.as_dict(),
                "cohort_path": str(cohort_path),
                "split": split,
                "stage": stage,
                "cells": CELLS,
                "targets": TARGETS,
                "anchors": ANCHORS,
                "module_modes": MODULE_MODES,
                "ordinary_attention_blocks_one_based": [
                    int(key.rsplit("l", 1)[1]) + 1 for key in attention
                ],
                "mlp_blocks_one_based": list(range(1, 65)),
                "complete_forward_passes_per_question": 1 if stage == "natural" else 7,
                "software": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                },
                "platform": platform.platform(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    for question_index, row in enumerate(rows):
        if stage == "natural" and arrays["natural_completed"][question_index]:
            continue
        if stage == "full" and arrays["completed"][question_index]:
            continue
        question = questions[row["question_id"]]
        first_x = _remap_question(question, row["first_x_new_to_original"])
        first_y = _remap_question(question, row["first_y_new_to_original"])
        second = _remap_question(question, row["second_new_to_original"])
        prompts = []
        positions = []
        audits = []
        for first, condition in zip((first_x, first_x, first_y, first_y), CONDITIONS):
            messages = _messages(config, first, second, condition)
            prompt = render_chat(
                processor, messages, config.disable_thinking, config.chat_serialization
            )
            found, audit = _positions(
                tokenizer,
                prompt,
                condition,
                second,
                row["x_second_letter"],
                row["y_second_letter"],
            )
            prompts.append(prompt)
            positions.append(found)
            audits.append(audit)
        # Within each feedback condition, everything from the feedback action
        # clause onward must be identical across the X/Y histories.
        for left, right in ((0, 2), (1, 3)):
            marker = FACTORIAL_FEEDBACK[CONDITIONS[left]]
            if prompts[left][prompts[left].find(marker) :] != prompts[right][prompts[right].find(marker) :]:
                raise RuntimeError("Second-presentation suffix differs across semantic histories")

        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        padded_width = int(input_ids.shape[1])
        padded_positions = [
            [position + padded_width - audit["length"] for position in found]
            for found, audit in zip(positions, audits)
        ]
        collector = NaturalComponentCollector(modules, padded_positions) if stage == "full" else None
        try:
            natural_output = _forward(model, parts, input_ids, attention_mask)
        finally:
            if collector is not None:
                collector.close()
        natural_logits = _aggregate_logits(natural_output, variant_ids)
        arrays["natural_logits"][:, question_index] = natural_logits
        arrays["natural_completed"][question_index] = True

        if not audit_path.exists():
            audit_path.write_text(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "x_content_original_letter": row["x_content_original_letter"],
                        "y_content_original_letter": row["y_content_original_letter"],
                        "x_second_letter": row["x_second_letter"],
                        "y_second_letter": row["y_second_letter"],
                        "cells": {
                            cell: {**audit, "rendered_prompt": prompt}
                            for cell, audit, prompt in zip(CELLS, audits, prompts)
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

        if stage == "full":
            source = collector.values
            attn_keys = list(attention)
            mlp_keys = list(mlps)
            arrays["attention_interaction_relative_norm"][question_index] = np.stack(
                [_interaction_relative_norm(source[key]) for key in attn_keys], axis=-1
            )
            arrays["mlp_interaction_relative_norm"][question_index] = np.stack(
                [_interaction_relative_norm(source[key]) for key in mlp_keys], axis=-1
            )
            for target_index, target in enumerate(TARGETS):
                target_cell = 0 if target == "x" else 2
                source_anchor_by_test = {
                    "evaluation_period": 0,
                    "repeated_candidate": 1 if target == "x" else 2,
                    "decision": 3,
                }
                target_prompt = prompts[target_cell]
                target_length = audits[target_cell]["length"]
                for anchor_index, anchor in enumerate(ANCHORS):
                    source_anchor = source_anchor_by_test[anchor]
                    target_position = positions[target_cell][source_anchor]
                    repeated_prompts = [target_prompt] * 4
                    patched_ids, patched_mask, _ = tokenize_batch(
                        tokenizer, repeated_prompts
                    )
                    padded_target = target_position + int(patched_ids.shape[1]) - target_length
                    patcher = NoInteractionPatcher(
                        attention,
                        mlps,
                        source,
                        source_anchor,
                        target_cell,
                        padded_target,
                    )
                    try:
                        patched_output = _forward(
                            model, parts, patched_ids, patched_mask
                        )
                    finally:
                        patcher.close()
                    raw = _aggregate_logits(patched_output, variant_ids)
                    control = raw[3]
                    corrected = natural_logits[target_cell][None, :] + (
                        raw[:3] - control[None, :]
                    )
                    arrays["patched_logits"][
                        target_index, anchor_index, :, question_index
                    ] = corrected
                    arrays["batch_control_minus_natural"][
                        target_index, anchor_index, question_index
                    ] = control - natural_logits[target_cell]
            arrays["completed"][question_index] = True

        atomic_save_npz(output, **arrays)
        done = int(
            arrays["natural_completed"].sum()
            if stage == "natural"
            else arrays["completed"].sum()
        )
        print(
            f"semantic-binding {split} {stage}: {done}/{len(rows)} questions",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--stage", choices=("natural", "full"), required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.cohort,
        args.output,
        args.split,
        args.stage,
        args.max_questions,
    )


if __name__ == "__main__":
    main()
