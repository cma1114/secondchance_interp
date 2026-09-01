from __future__ import annotations

import argparse
import copy
import json
import os
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .jlens_collect import PositionCollector, _scope_end_token, _token_offsets
from .modeling import (
    QWEN_EMPTY_THINKING,
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    tokenize_batch,
)
from .prompts import baseline_question_turn, build_messages, present_question


ANCHORS = ("first_question_end", "first_user_end", "first_answer_decision")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _load_question_ids(path: Path) -> list[str]:
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


def _positions(tokenizer: Any, prompt: str, question: dict[str, Any]) -> list[int]:
    offsets = _token_offsets(tokenizer, prompt)
    first_question_end = _scope_end_token(prompt, offsets, present_question(question))
    first_user_end = _scope_end_token(prompt, offsets, baseline_question_turn(question))
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
    return [first_question_end, first_user_end, scaffold_tokens[-1]]


def _content_token_ids(tokenizer: Any, text: str) -> tuple[list[int], list[str]]:
    ids = [int(value) for value in tokenizer.encode(" " + text.strip(), add_special_tokens=False)]
    kept_ids, kept_tokens = [], []
    for token_id in ids:
        token = tokenizer.decode([token_id])
        normalized = unicodedata.normalize("NFKC", token)
        if any(character.isalnum() for character in normalized):
            kept_ids.append(token_id)
            kept_tokens.append(token)
    if not kept_ids:
        raise ValueError(f"Option has no content-bearing tokenizer pieces: {text!r}")
    return kept_ids, kept_tokens


def collect(
    config_path: Path,
    question_plan_path: Path,
    output: Path,
    remapping_plan_path: Path | None,
    embeddings_only: bool,
    max_questions: int | None,
) -> None:
    import torch

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw_qwen_chatml serialization")

    question_ids = _load_question_ids(question_plan_path)
    if max_questions is not None:
        question_ids = question_ids[: int(max_questions)]
    manifest = json.loads(Path(config.manifest_path).read_text())
    source_questions = {row["id"]: row for row in manifest["questions"]}
    if not set(question_ids) <= set(source_questions):
        raise ValueError("Question plan contains IDs absent from the manifest")

    remapping = None
    if remapping_plan_path is not None:
        payload = json.loads(remapping_plan_path.read_text())
        remapping = {row["question_id"]: row for row in payload["rows"]}
        if not set(question_ids) <= set(remapping):
            raise ValueError("Remapping plan is missing requested questions")

    questions = []
    for qid in question_ids:
        question = source_questions[qid]
        if remapping is not None:
            question = _remap_question(question, remapping[qid]["new_to_original"])
        questions.append(question)

    output.mkdir(parents=True, exist_ok=True)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    width = int(parts.embedding.weight.shape[-1])

    option_embeddings = np.empty((len(questions), 4, width), dtype=np.float16)
    option_audit: dict[str, Any] = {}
    with torch.inference_mode():
        for qi, (qid, question) in enumerate(zip(question_ids, questions)):
            option_audit[qid] = {"options": {}}
            for oi, letter in enumerate("ABCD"):
                text = question["options"][letter]
                token_ids, tokens = _content_token_ids(tokenizer, text)
                option_audit[qid]["options"][letter] = {
                    "text": text,
                    "token_ids": token_ids,
                    "tokens": tokens,
                }
                ids = torch.as_tensor(token_ids, device=parts.embedding.weight.device)
                vector = parts.embedding.weight[ids].float().mean(dim=0)
                vector /= vector.norm().clamp_min(1e-12)
                option_embeddings[qi, oi] = vector.cpu().to(torch.float16).numpy()
    np.save(output / "option_embeddings.npy", option_embeddings)
    del option_embeddings

    prompt_audit: dict[str, Any] = {}
    if not embeddings_only:
        shape = (len(questions), len(parts.layers), len(ANCHORS), width)
        residual_path = output / "position_residuals.npy"
        completed_path = output / "completed.npy"
        if residual_path.exists() and completed_path.exists():
            residuals = np.lib.format.open_memmap(residual_path, mode="r+")
            completed = np.load(completed_path)
            if tuple(residuals.shape) != shape or completed.shape != (len(questions),):
                raise ValueError("Existing semantic residual cache has incompatible shape")
        else:
            residuals = np.lib.format.open_memmap(
                residual_path, mode="w+", dtype=np.float16, shape=shape
            )
            completed = np.zeros(len(questions), dtype=bool)

        for qi, (qid, question) in enumerate(zip(question_ids, questions)):
            if completed[qi]:
                continue
            messages = build_messages(
                question, "baseline", config.prompt_mode, config.feedback_variant
            )
            prompt = render_chat(
                processor, messages, config.disable_thinking, config.chat_serialization
            )
            positions = _positions(tokenizer, prompt, question)
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, [prompt])
            collector = PositionCollector(parts.layers, positions)
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
                residuals[qi] = collector.stacked().numpy().transpose(0, 1, 2)
            finally:
                collector.close()
            completed[qi] = True
            np.save(completed_path, completed)
            if qi == 0:
                ids = input_ids[0].tolist()
                prompt_audit[qid] = {
                    "positions": dict(zip(ANCHORS, positions)),
                    "tokens": {
                        anchor: tokenizer.decode([ids[position]])
                        for anchor, position in zip(ANCHORS, positions)
                    },
                    "rendered_prompt": prompt,
                }
            if qi == 0 or (qi + 1) % 10 == 0 or qi + 1 == len(questions):
                print(f"semantic residuals: {int(completed.sum())}/{len(completed)}", flush=True)
        residuals.flush()

    metadata = {
        "config": config.as_dict(),
        "question_plan": str(question_plan_path),
        "question_ids": question_ids,
        "anchors": list(ANCHORS),
        "layers": list(range(1, len(parts.layers) + 1)),
        "width": width,
        "remapping_plan": str(remapping_plan_path) if remapping_plan_path else None,
        "embeddings_only": embeddings_only,
        "option_embedding_definition": (
            "Normalized mean input embedding of all alphanumeric content-bearing "
            "option tokens; punctuation and A-D label tokens are excluded."
        ),
        "option_audit": option_audit,
        "prompt_audit": prompt_audit,
    }
    _write_json(output / "metadata.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--question-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path)
    parser.add_argument("--embeddings-only", action="store_true")
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    collect(
        args.config,
        args.question_plan,
        args.output,
        args.remapping_plan,
        args.embeddings_only,
        args.max_questions,
    )


if __name__ == "__main__":
    main()
