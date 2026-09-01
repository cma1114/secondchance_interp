from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .config import ExperimentConfig
from .modeling import QWEN_EMPTY_THINKING, render_chat
from .prompts import build_messages, load_trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        max_questions=1,
    )
    trial = trials[0]
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
    )
    prompts = {
        condition: render_chat(
            tokenizer,
            build_messages(
                trial.question,
                condition,
                config.prompt_mode,
                config.feedback_variant,
            ),
            config.disable_thinking,
            config.chat_serialization,
        )
        for condition in ("baseline", "incorrect", "neutral")
    }
    baseline = prompts["baseline"]
    historical_close = "<|im_end|>\n"
    checks = {
        "no_redacted_game": "[redacted]" not in prompts["incorrect"],
        "no_redacted_neutral": "[redacted]" not in prompts["neutral"],
        "game_first_boundary_exactly_baseline": prompts["incorrect"].startswith(
            baseline + historical_close
        ),
        "neutral_first_boundary_exactly_baseline": prompts["neutral"].startswith(
            baseline + historical_close
        ),
        "game_final_boundary_same_scaffold": prompts["incorrect"].endswith(
            "<|im_start|>assistant\n" + QWEN_EMPTY_THINKING
        ),
        "neutral_final_boundary_same_scaffold": prompts["neutral"].endswith(
            "<|im_start|>assistant\n" + QWEN_EMPTY_THINKING
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    scaffold_ids = tokenizer.encode(QWEN_EMPTY_THINKING, add_special_tokens=False)
    payload = {
        "question_id": trial.question_id,
        "checks": checks,
        "empty_thinking": repr(QWEN_EMPTY_THINKING),
        "empty_thinking_token_ids": scaffold_ids,
        "empty_thinking_tokens": [
            tokenizer.decode([token_id]) for token_id in scaffold_ids
        ],
        "messages": {
            condition: build_messages(
                trial.question,
                condition,
                config.prompt_mode,
                config.feedback_variant,
            )
            for condition in ("baseline", "incorrect", "neutral")
        },
        "rendered_prompts": prompts,
    }
    output = Path(config.output_dir) / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in payload if key != "rendered_prompts"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
