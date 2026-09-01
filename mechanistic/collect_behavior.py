from __future__ import annotations

import argparse
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
from .prompts import build_messages, load_trials, prompt_hash


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def collect(config: ExperimentConfig) -> None:
    import torch

    allowed = {"baseline", "incorrect", "neutral", "incorrect_no_system_setup"}
    if not config.conditions or not set(config.conditions) <= allowed:
        raise ValueError(
            "The behavioral collector requires one or more second-chance "
            f"conditions drawn from {sorted(allowed)}"
        )
    output_dir = Path(config.output_dir)
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        config.question_ids,
        config.max_questions,
        config.skip_missing_baseline,
    )
    payloads = {}
    pending_by_condition = {}
    for condition in config.conditions:
        output_path = output_dir / f"{condition}_results.json"
        payload = (
            json.loads(output_path.read_text())
            if output_path.exists()
            else {"condition": condition, "results": {}}
        )
        payloads[condition] = (output_path, payload)
        pending = [
            trial for trial in trials
            if trial.question_id not in payload["results"]
        ]
        pending_by_condition[condition] = pending
        print(f"{condition}: {len(pending)} pending / {len(trials)} total", flush=True)
    if not any(pending_by_condition.values()):
        return

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    all_ad_ids = sorted({token_id for ids in variant_ids.values() for token_id in ids})
    bracket_ids = tokenizer.encode("[", add_special_tokens=False)
    if len(bracket_ids) != 1:
        raise RuntimeError(f"Expected '[' to be one token, found {bracket_ids}")
    bracket_id = int(bracket_ids[0])
    device = model_input_device(parts)

    for condition in config.conditions:
        output_path, payload = payloads[condition]
        results = payload["results"]
        pending = pending_by_condition[condition]
        for batch_index, batch in enumerate(_chunks(pending, config.batch_size), 1):
            messages = [
                build_messages(
                    trial.question,
                    condition,
                    config.prompt_mode,
                    config.feedback_variant,
                )
                for trial in batch
            ]
            prompts = [
                render_chat(
                    processor,
                    message,
                    config.disable_thinking,
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
                rows = np.arange(len(batch))
                final_logits = logits[rows, last_indices]
            probabilities = torch.softmax(final_logits, dim=-1)
            top_values, top_ids = torch.topk(probabilities, k=10, dim=-1)

            for index, trial in enumerate(batch):
                aggregated = torch.stack([
                    torch.logsumexp(final_logits[index, variant_ids[letter]], dim=0)
                    for letter in LETTERS
                ])
                token_ids = [int(value) for value in top_ids[index].tolist()]
                bracket_logit = final_logits[index, bracket_id]
                results[trial.question_id] = {
                    "question_id": trial.question_id,
                    "baseline_answer": trial.baseline_answer,
                    "baseline_correct": trial.baseline_correct,
                    "correct_answer": trial.question["correct_answer"],
                    "answer": LETTERS[int(aggregated.argmax())],
                    "aggregated_ad_logits": aggregated.tolist(),
                    "full_vocab_top_token_id": token_ids[0],
                    "full_vocab_top_token": tokenizer.decode([token_ids[0]]),
                    "full_vocab_top10": [
                        {
                            "rank": rank + 1,
                            "token_id": token_id,
                            "token": tokenizer.decode([token_id]),
                            "probability": float(top_values[index, rank]),
                        }
                        for rank, token_id in enumerate(token_ids)
                    ],
                    "left_bracket_token_id": bracket_id,
                    "left_bracket_rank": int(
                        1 + (final_logits[index] > bracket_logit).sum()
                    ),
                    "left_bracket_probability": float(
                        probabilities[index, bracket_id]
                    ),
                    "ad_probability_mass": float(
                        probabilities[index, all_ad_ids].sum()
                    ),
                    "prompt_hash": prompt_hash(prompts[index]),
                    "rendered_prompt": prompts[index],
                    "messages": messages[index],
                }
            payload["complete"] = len(results) == len(trials)
            payload["n_results"] = len(results)
            payload["model_id"] = config.model_id
            payload["model_revision"] = config.model_revision
            payload["config"] = config.as_dict()
            _write_json(output_path, payload)
            print(
                f"{condition}: saved "
                f"{min(batch_index * config.batch_size, len(pending))}/"
                f"{len(pending)} pending trials",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect final behavioral outputs for a prompt ablation"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    collect(ExperimentConfig.load(args.config))


if __name__ == "__main__":
    main()
