from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .jlens_collect import _token_offsets
from .modeling import (
    QWEN_EMPTY_THINKING,
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    tokenize_batch,
)
from .prompts import build_messages


ANCHORS = tuple(
    [f"content_end_{letter}" for letter in "ABCD"]
    + [f"line_end_{letter}" for letter in "ABCD"]
    + ["first_answer_decision"]
)


class BatchedPositionCollector:
    """Collect different token positions for every example in a prompt batch."""

    def __init__(self, layers: Any, positions: list[list[int]]):
        import torch

        if not positions or not all(len(row) == len(positions[0]) for row in positions):
            raise ValueError("Batched positions must be a nonempty rectangular array")
        self.positions = torch.as_tensor(positions, dtype=torch.long)
        self.values: list[Any] = [None] * len(layers)
        self.handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(layers)
        ]

    def _hook(self, index: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.shape[0] != self.positions.shape[0]:
                raise RuntimeError(
                    f"Hook batch {hidden.shape[0]} differs from position batch "
                    f"{self.positions.shape[0]}"
                )
            minimum = int(self.positions.min().item())
            maximum = int(self.positions.max().item())
            if minimum < 0 or maximum >= hidden.shape[1]:
                raise RuntimeError(
                    f"Layer {index} requested token positions "
                    f"{self.positions.tolist()} outside hidden shape "
                    f"{tuple(hidden.shape)}"
                )
            indices = self.positions.to(hidden.device)
            rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
            self.values[index] = hidden[rows, indices].detach().to(
                "cpu", dtype=torch.float16
            )

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def stacked(self):
        import torch

        if any(value is None for value in self.values):
            raise RuntimeError("Failed to collect one or more residual layers")
        return torch.stack(self.values, dim=0)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _question_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    values = payload.get("question_ids", payload.get("discovery_question_ids", []))
    if not values:
        raise ValueError("Question plan contains no question IDs")
    return list(values)


def _remap_question(question: dict[str, Any], new_to_original: dict[str, str]) -> dict[str, Any]:
    remapped = copy.deepcopy(question)
    remapped["options"] = {
        new: question["options"][original] for new, original in new_to_original.items()
    }
    original_to_new = {original: new for new, original in new_to_original.items()}
    remapped["correct_answer"] = original_to_new[question["correct_answer"]]
    return remapped


def _overlapping_token(
    offsets: list[tuple[int, int]], start: int, end: int, *, last: bool = True
) -> int:
    matches = [
        index for index, (left, right) in enumerate(offsets)
        if right > start and left < end
    ]
    if not matches:
        raise RuntimeError(f"No tokenizer token overlaps character span {start}:{end}")
    return matches[-1] if last else matches[0]


def _positions(
    tokenizer: Any, prompt: str, question: dict[str, Any]
) -> tuple[list[int], dict[str, Any]]:
    offsets = _token_offsets(tokenizer, prompt)
    values: dict[str, int] = {}
    audit: dict[str, Any] = {"options": {}}
    search_start = 0
    for letter in "ABCD":
        prefix = f"  {letter}: "
        text = question["options"][letter]
        line = prefix + text + "\n"
        line_start = prompt.find(line, search_start)
        if line_start < 0:
            raise RuntimeError(f"Could not locate exact option line {line!r}")
        content_start = line_start + len(prefix)
        content_end = content_start + len(text)
        newline_start = content_end
        if prompt[newline_start : newline_start + 1] != "\n":
            raise RuntimeError("Option line does not end in the expected newline")
        content_token = _overlapping_token(offsets, content_end - 1, content_end)
        newline_token = _overlapping_token(offsets, newline_start, newline_start + 1)
        values[f"content_end_{letter}"] = content_token
        values[f"line_end_{letter}"] = newline_token
        audit["options"][letter] = {
            "text": text,
            "line": line,
            "content_character_span": [content_start, content_end],
            "newline_character_span": [newline_start, newline_start + 1],
            "content_end_position": content_token,
            "line_end_position": newline_token,
        }
        search_start = line_start + len(line)

    assistant_header = "<|im_start|>assistant\n"
    assistant_start = prompt.find(assistant_header)
    if assistant_start < 0:
        raise RuntimeError("Could not locate first assistant header")
    scaffold_end = assistant_start + len(assistant_header) + len(QWEN_EMPTY_THINKING)
    scaffold_tokens = [
        index for index, (left, right) in enumerate(offsets)
        if right > left and right <= scaffold_end
    ]
    if not scaffold_tokens:
        raise RuntimeError("Could not locate first empty-thinking scaffold")
    values["first_answer_decision"] = scaffold_tokens[-1]
    positions = [values[anchor] for anchor in ANCHORS]
    audit["positions"] = values
    return positions, audit


def collect(
    config_path: Path,
    question_plan_path: Path,
    output: Path,
    remapping_plan_path: Path | None,
    max_questions: int | None,
    batch_size_override: int | None,
) -> None:
    import torch

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw_qwen_chatml serialization")

    qids = _question_ids(question_plan_path)
    if max_questions is not None:
        qids = qids[: int(max_questions)]
    manifest = json.loads(Path(config.manifest_path).read_text())
    source = {row["id"]: row for row in manifest["questions"]}
    if not set(qids) <= set(source):
        raise ValueError("Question plan contains IDs absent from manifest")

    remapping = None
    if remapping_plan_path is not None:
        payload = json.loads(remapping_plan_path.read_text())
        remapping = {row["question_id"]: row for row in payload["rows"]}
        if not set(qids) <= set(remapping):
            raise ValueError("Remapping plan is missing requested questions")

    questions: list[dict[str, Any]] = []
    mappings: dict[str, Any] = {}
    for qid in qids:
        question = source[qid]
        if remapping is not None:
            row = remapping[qid]
            question = _remap_question(question, row["new_to_original"])
            mappings[qid] = {
                "new_to_original": row["new_to_original"],
                "original_to_new": row["original_to_new"],
            }
        questions.append(question)

    output.mkdir(parents=True, exist_ok=True)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    width = int(parts.embedding.weight.shape[-1])
    shape = (len(questions), len(parts.layers), len(ANCHORS), width)
    residual_path = output / "position_residuals.npy"
    completed_path = output / "completed.npy"
    if residual_path.exists() and completed_path.exists():
        residuals = np.lib.format.open_memmap(residual_path, mode="r+")
        completed = np.load(completed_path)
        if tuple(residuals.shape) != shape or completed.shape != (len(questions),):
            raise ValueError("Existing contextual-option cache has incompatible shape")
    else:
        residuals = np.lib.format.open_memmap(
            residual_path, mode="w+", dtype=np.float16, shape=shape
        )
        completed = np.zeros(len(questions), dtype=bool)

    batch_size = int(batch_size_override or config.batch_size)
    if batch_size < 1:
        raise ValueError("Batch size must be positive")

    # Render and audit every prompt, including already-completed questions. This
    # keeps metadata complete when a checkpointed collection is resumed.
    audits: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for qi, (qid, question) in enumerate(zip(qids, questions)):
        messages = build_messages(
            question, "baseline", config.prompt_mode, config.feedback_variant
        )
        prompt = render_chat(
            processor, messages, config.disable_thinking, config.chat_serialization
        )
        positions, audit = _positions(tokenizer, prompt, question)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for letter in "ABCD":
            row = audit["options"][letter]
            row["content_end_token"] = tokenizer.decode(
                [ids[row["content_end_position"]]]
            )
            row["line_end_token"] = tokenizer.decode([ids[row["line_end_position"]]])
        audit["first_answer_decision"] = {
            "position": positions[-1],
            "token": tokenizer.decode([ids[positions[-1]]]),
        }
        if qi == 0:
            audit["rendered_prompt"] = prompt
        audits[qid] = audit
        records.append(
            {
                "question_index": qi,
                "question_id": qid,
                "prompt": prompt,
                "positions": positions,
                "length": len(ids),
            }
        )

    pending = [record for record in records if not completed[record["question_index"]]]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [record["prompt"] for record in batch]
        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        width_tokens = int(input_ids.shape[1])
        padded_positions = [
            [
                int(position) + width_tokens - int(record["length"])
                for position in record["positions"]
            ]
            for record in batch
        ]
        collector = BatchedPositionCollector(parts.layers, padded_positions)
        try:
            with torch.inference_mode():
                kwargs = {
                    "input_ids": input_ids.to(model_input_device(parts)),
                    "attention_mask": attention_mask.to(model_input_device(parts)),
                    "use_cache": False,
                    "return_dict": True,
                }
                try:
                    model(**kwargs, logits_to_keep=1)
                except TypeError:
                    model(**kwargs)
            batch_values = collector.stacked().numpy().transpose(1, 0, 2, 3)
            batch_indices = [record["question_index"] for record in batch]
            residuals[batch_indices] = batch_values
        finally:
            collector.close()
        completed[batch_indices] = True
        np.save(completed_path, completed)
        print(
            f"contextual option residuals: {int(completed.sum())}/{len(completed)} "
            f"(batch={len(batch)})",
            flush=True,
        )
    residuals.flush()

    metadata = {
        "config": config.as_dict(),
        "question_plan": str(question_plan_path),
        "question_ids": qids,
        "anchors": list(ANCHORS),
        "layers": list(range(1, len(parts.layers) + 1)),
        "width": width,
        "batch_size": batch_size,
        "remapping_plan": str(remapping_plan_path) if remapping_plan_path else None,
        "mappings": mappings,
        "audit": audits,
        "definition": (
            "Contextual post-block residuals at the last option-content token, "
            "the token overlapping the option-closing newline, and the first-answer decision."
        ),
    }
    _write_json(output / "metadata.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--question-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    collect(
        args.config,
        args.question_plan,
        args.output,
        args.remapping_plan,
        args.max_questions,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
