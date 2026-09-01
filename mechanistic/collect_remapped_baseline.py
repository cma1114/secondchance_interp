from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import LETTERS
from .collect_remapped_behavior import _chunks, _remap_question, _write_json
from .config import ExperimentConfig
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages, prompt_hash


def collect(
    config_path: Path,
    plan_path: Path,
    output_path: Path,
    max_questions: int | None,
) -> None:
    import torch

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test prompt configuration")
    if config.chat_serialization not in {"raw_qwen_chatml", "hf_template"}:
        raise ValueError("Requires a validated raw-Qwen or HF chat serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {question["id"]: question for question in manifest["questions"]}
    plan_payload = json.loads(plan_path.read_text())
    plan = {row["question_id"]: row for row in plan_payload["rows"]}
    qids = [question["id"] for question in manifest["questions"]]
    if max_questions is not None:
        qids = qids[: int(max_questions)]
    if not set(qids) <= set(plan):
        raise ValueError("Remapping plan is missing requested questions")

    payload = json.loads(output_path.read_text()) if output_path.exists() else {
        "condition": "remapped_standalone_baseline",
        "results": {},
    }
    results = payload["results"]
    pending = [qid for qid in qids if qid not in results]
    print(f"remapped baseline: {len(pending)} pending / {len(qids)} total", flush=True)
    if not pending:
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

    for batch_index, batch_qids in enumerate(_chunks(pending, config.batch_size), 1):
        rows = [plan[qid] for qid in batch_qids]
        remapped_questions = [
            _remap_question(questions[qid], row["new_to_original"])
            for qid, row in zip(batch_qids, rows)
        ]
        messages = [
            build_messages(
                question,
                "baseline",
                config.prompt_mode,
                config.feedback_variant,
            )
            for question in remapped_questions
        ]
        prompts = [
            render_chat(
                processor,
                message,
                config.disable_thinking,
                config.chat_serialization,
                config.chat_template_kwargs,
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
        raw_logits = output.logits.detach().float().cpu()
        if raw_logits.shape[1] == 1:
            final_logits = raw_logits[:, 0]
        else:
            final_logits = raw_logits[np.arange(len(batch_qids)), last_indices]
        probabilities = torch.softmax(final_logits, dim=-1)
        top_values, top_ids = torch.topk(probabilities, k=10, dim=-1)

        for index, (qid, row, question, message, prompt) in enumerate(
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
            results[qid] = {
                "question_id": qid,
                "new_to_original": row["new_to_original"],
                "original_to_new": row["original_to_new"],
                "remapped_correct_letter": question["correct_answer"],
                "answer_new_letter": unrestricted_answer,
                "answer_original_content": (
                    row["new_to_original"][unrestricted_answer]
                    if unrestricted_answer is not None else None
                ),
                "aggregated_ad_answer_new_letter": aggregated_answer,
                "aggregated_ad_answer_original_content": (
                    row["new_to_original"][aggregated_answer]
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
                "ad_probability_mass": float(probabilities[index, all_ad_ids].sum()),
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
        })
        _write_json(output_path, payload)
        print(
            f"remapped baseline: saved "
            f"{min(batch_index * config.batch_size, len(pending))}/"
            f"{len(pending)} pending trials",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect a standalone Baseline on frozen remapped options"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    collect(args.config, args.plan, args.output, args.max_questions)


if __name__ == "__main__":
    main()
