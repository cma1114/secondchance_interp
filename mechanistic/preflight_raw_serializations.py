from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .modeling import (
    QWEN_EMPTY_THINKING,
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    resolve_answer_tokens,
    tokenize_batch,
    variant_layout,
)
from .prompts import build_messages, load_trials


def _render(messages: list[dict[str, str]], mode: str) -> str:
    parts: list[str] = []
    for original in messages:
        message = dict(original)
        role = message["role"]
        content = message["content"]
        if mode == "empty_think" and role == "assistant":
            content = QWEN_EMPTY_THINKING + content
        elif mode == "no_think_directive" and role == "system":
            content = content.rstrip() + "\n/no_think\n"
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    suffix = "<|im_start|>assistant\n"
    if mode == "empty_think":
        suffix += QWEN_EMPTY_THINKING
    return "".join(parts) + suffix


def _select_trials(config: ExperimentConfig, residual_root: Path, n_each: int):
    trials = load_trials(config.manifest_path, config.baseline_results_path)
    bad, good = [], []
    for trial in trials:
        path = residual_root / "shards" / "incorrect" / f"{trial.question_id}.npz"
        with np.load(path, allow_pickle=False) as shard:
            metadata = json.loads(shard["metadata"].item())
        (bad if metadata["full_vocab_top_token"].strip() not in "ABCD" else good).append(trial)
    return bad[:n_each] + good[:n_each]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--n-each", type=int, default=12)
    args = parser.parse_args()

    import torch

    config = ExperimentConfig.load(args.config)
    trials = _select_trials(config, args.residual_root, args.n_each)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    selected_ids, layout = variant_layout(resolved)
    by_letter = {
        letter: [index for index, row in enumerate(layout) if row["letter"] == letter]
        for letter in "ABCD"
    }
    results: dict[str, dict[str, list[dict]]] = {}
    for mode in ("bare", "no_think_directive", "empty_think"):
        results[mode] = {}
        for condition in ("baseline", "incorrect", "neutral"):
            prompts = [_render(build_messages(t.question, condition, config.prompt_mode), mode) for t in trials]
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids.to(model_input_device(parts)),
                    attention_mask=attention_mask.to(model_input_device(parts)),
                    use_cache=False,
                    return_dict=True,
                )
            rows = torch.arange(len(trials))
            logits = output.logits.detach().float().cpu()[rows, last_indices]
            selected = logits[:, selected_ids].numpy()
            records = []
            for index, trial in enumerate(trials):
                top_id = int(logits[index].argmax())
                family_scores = []
                for letter in "ABCD":
                    values = selected[index, by_letter[letter]]
                    maximum = values.max()
                    family_scores.append(maximum + np.log(np.exp(values - maximum).sum()))
                records.append({
                    "question_id": trial.question_id,
                    "full_vocab_top_token": tokenizer.decode([top_id]),
                    "full_vocab_top_token_id": top_id,
                    "ad_choice": "ABCD"[int(np.argmax(family_scores))],
                })
            results[mode][condition] = records
            counts: dict[str, int] = {}
            for row in records:
                token = row["full_vocab_top_token"]
                counts[token] = counts.get(token, 0) + 1
            print(mode, condition, counts, flush=True)

        baseline = results[mode]["baseline"]
        for condition in ("incorrect", "neutral"):
            before_redacted = [
                _render(build_messages(t.question, condition, config.prompt_mode)[:2], mode)
                for t in trials
            ]
            baseline_prompts = [
                _render(build_messages(t.question, "baseline", config.prompt_mode), mode)
                for t in trials
            ]
            print(
                mode,
                condition,
                "prefix_exact",
                all(left == right for left, right in zip(before_redacted, baseline_prompts)),
                flush=True,
            )

    print("RESULT_JSON=" + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
