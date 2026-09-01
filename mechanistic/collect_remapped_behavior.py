from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages, prompt_hash, repeated_question_turn


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _remap_question(question: dict, new_to_original: dict[str, str]) -> dict:
    remapped = copy.deepcopy(question)
    remapped["options"] = {
        new: question["options"][original]
        for new, original in new_to_original.items()
    }
    original_to_new = {original: new for new, original in new_to_original.items()}
    remapped["correct_answer"] = original_to_new[question["correct_answer"]]
    return remapped


def _messages(config: ExperimentConfig, question: dict, remapped: dict, condition: str):
    messages = build_messages(
        question, condition, config.prompt_mode, config.feedback_variant
    )
    original_repeat = repeated_question_turn(question)
    if not messages[-1]["content"].endswith(original_repeat):
        raise RuntimeError("Could not locate the original repeated question turn")
    messages[-1]["content"] = (
        messages[-1]["content"][: -len(original_repeat)]
        + repeated_question_turn(remapped)
    )
    return messages


def collect(
    config_path: Path,
    plan_path: Path,
    baseline_path: Path,
    output_dir: Path,
    max_questions: int | None,
) -> None:
    import torch

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {question["id"]: question for question in manifest["questions"]}
    baseline = json.loads(baseline_path.read_text())["results"]
    plan = json.loads(plan_path.read_text())
    plan_rows = {row["question_id"]: row for row in plan["rows"]}
    qids = [question["id"] for question in manifest["questions"]]
    if max_questions is not None:
        qids = qids[: int(max_questions)]
    if not set(qids) <= set(plan_rows) or not set(qids) <= set(baseline):
        raise ValueError("Plan or current Baseline is missing requested questions")

    payloads = {}
    pending = {}
    for condition in ("incorrect", "neutral"):
        path = output_dir / f"{condition}_results.json"
        payload = json.loads(path.read_text()) if path.exists() else {
            "condition": condition,
            "results": {},
        }
        payloads[condition] = (path, payload)
        pending[condition] = [qid for qid in qids if qid not in payload["results"]]
        print(f"{condition}: {len(pending[condition])} pending / {len(qids)} total", flush=True)
    if not any(pending.values()):
        return

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    all_ad_ids = sorted({token_id for ids in variant_ids.values() for token_id in ids})
    device = model_input_device(parts)

    for condition in ("incorrect", "neutral"):
        output_path, payload = payloads[condition]
        results = payload["results"]
        for batch_index, batch_qids in enumerate(
            _chunks(pending[condition], config.batch_size), 1
        ):
            rows = [plan_rows[qid] for qid in batch_qids]
            remapped_questions = [
                _remap_question(questions[qid], row["new_to_original"])
                for qid, row in zip(batch_qids, rows)
            ]
            messages = [
                _messages(config, questions[qid], remapped, condition)
                for qid, remapped in zip(batch_qids, remapped_questions)
            ]
            prompts = [
                render_chat(
                    processor, message, config.disable_thinking,
                    config.chat_serialization,
                )
                for message in messages
            ]
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
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
            logits = output.logits.detach().float().cpu()
            if logits.shape[1] == 1:
                final_logits = logits[:, 0]
            else:
                final_logits = logits[np.arange(len(batch_qids)), last_indices]
            probabilities = torch.softmax(final_logits, dim=-1)
            top_values, top_ids = torch.topk(probabilities, k=10, dim=-1)

            for index, (qid, plan_row, remapped, message, prompt) in enumerate(
                zip(batch_qids, rows, remapped_questions, messages, prompts)
            ):
                aggregated = torch.stack([
                    torch.logsumexp(final_logits[index, variant_ids[letter]], dim=0)
                    for letter in LETTERS
                ])
                aggregated_answer = LETTERS[int(aggregated.argmax())]
                token_ids = [int(value) for value in top_ids[index].tolist()]
                unrestricted_token = tokenizer.decode([token_ids[0]])
                unrestricted_answer = unrestricted_token.strip()
                if unrestricted_answer not in LETTERS:
                    unrestricted_answer = None
                baseline_answer = baseline[qid].get(
                    "answer", baseline[qid].get("subject_answer")
                )
                results[qid] = {
                    "question_id": qid,
                    "baseline_original_letter": baseline_answer,
                    "correct_original_letter": questions[qid]["correct_answer"],
                    "new_to_original": plan_row["new_to_original"],
                    "original_to_new": plan_row["original_to_new"],
                    "baseline_content_new_letter": plan_row["baseline_content_new_letter"],
                    "remapped_correct_letter": remapped["correct_answer"],
                    "answer_new_letter": unrestricted_answer,
                    "answer_original_content": (
                        plan_row["new_to_original"][unrestricted_answer]
                        if unrestricted_answer is not None else None
                    ),
                    "aggregated_ad_answer_new_letter": aggregated_answer,
                    "aggregated_ad_answer_original_content": (
                        plan_row["new_to_original"][aggregated_answer]
                    ),
                    "aggregated_ad_logits": aggregated.tolist(),
                    "full_vocab_top_token_id": token_ids[0],
                    "full_vocab_top_token": unrestricted_token,
                    "full_vocab_top10": [
                        {
                            "rank": rank + 1,
                            "token_id": token_id,
                            "token": tokenizer.decode([token_id]),
                            "probability": float(top_values[index, rank]),
                        }
                        for rank, token_id in enumerate(token_ids)
                    ],
                    "ad_probability_mass": float(
                        probabilities[index, all_ad_ids].sum()
                    ),
                    "prompt_hash": prompt_hash(prompt),
                    "rendered_prompt": prompt,
                    "messages": message,
                }
            payload.update({
                "complete": len(results) == len(qids),
                "n_results": len(results),
                "model_id": config.model_id,
                "model_revision": config.model_revision,
                "config": config.as_dict(),
                "remapping_plan": str(plan_path),
                "current_baseline": str(baseline_path),
            })
            _write_json(output_path, payload)
            print(
                f"{condition}: saved {min(batch_index * config.batch_size, len(pending[condition]))}/"
                f"{len(pending[condition])} pending trials",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect behavior after option remapping")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    collect(args.config, args.plan, args.baseline, args.output_dir, args.max_questions)


if __name__ == "__main__":
    main()
