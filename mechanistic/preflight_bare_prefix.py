from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .config import ExperimentConfig
from .modeling import render_chat
from .prompts import build_messages, load_trials, prompt_hash


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit bare raw-ChatML equality through the first assistant header"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    if config.chat_serialization != "raw_qwen_chatml_bare":
        raise ValueError("Preflight requires raw_qwen_chatml_bare")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
    )
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        config.question_ids,
        config.max_questions,
        config.skip_missing_baseline,
    )
    records = []
    example = None
    for trial in trials:
        rendered = {
            condition: render_chat(
                tokenizer,
                build_messages(trial.question, condition, config.prompt_mode),
                config.disable_thinking,
                config.chat_serialization,
            )
            for condition in ("baseline", "incorrect", "neutral")
        }
        prefixes = {}
        for condition in ("incorrect", "neutral"):
            marker = rendered[condition].find("[redacted]")
            if marker < 0:
                raise RuntimeError(f"Missing [redacted] for {condition}/{trial.question_id}")
            prefixes[condition] = rendered[condition][:marker]
        texts = list(rendered.values())
        no_thinking_tokens = all(
            "<think>" not in text and "</think>" not in text for text in texts
        )
        rendered_prefix_exact = (
            rendered["baseline"] == prefixes["incorrect"] == prefixes["neutral"]
        )
        tokenized = {
            name: tokenizer(text, add_special_tokens=False)["input_ids"]
            for name, text in {
                "baseline": rendered["baseline"],
                "incorrect": prefixes["incorrect"],
                "neutral": prefixes["neutral"],
            }.items()
        }
        token_prefix_exact = (
            tokenized["baseline"] == tokenized["incorrect"] == tokenized["neutral"]
        )
        bare_assistant_boundary = rendered["baseline"].endswith(
            "<|im_start|>assistant\n"
        )
        row = {
            "question_id": trial.question_id,
            "no_thinking_tokens": no_thinking_tokens,
            "rendered_prefix_exact": rendered_prefix_exact,
            "token_prefix_exact": token_prefix_exact,
            "bare_assistant_boundary": bare_assistant_boundary,
            "shared_prefix_tokens": len(tokenized["baseline"]),
            "baseline_prompt_hash": prompt_hash(rendered["baseline"]),
        }
        if not all(
            row[key]
            for key in (
                "no_thinking_tokens",
                "rendered_prefix_exact",
                "token_prefix_exact",
                "bare_assistant_boundary",
            )
        ):
            raise RuntimeError(f"Bare-prefix audit failed: {row}")
        records.append(row)
        if example is None:
            example = {
                "question_id": trial.question_id,
                "baseline": rendered["baseline"],
                "incorrect": rendered["incorrect"],
                "neutral": rendered["neutral"],
            }

    result = {
        "serialization": config.chat_serialization,
        "n": len(records),
        "all_no_thinking_tokens": all(row["no_thinking_tokens"] for row in records),
        "all_rendered_prefixes_exact": all(row["rendered_prefix_exact"] for row in records),
        "all_token_prefixes_exact": all(row["token_prefix_exact"] for row in records),
        "all_bare_assistant_boundaries": all(
            row["bare_assistant_boundary"] for row in records
        ),
        "example": example,
        "trials": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"example", "trials"}}, indent=2))


if __name__ == "__main__":
    main()
