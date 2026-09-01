from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .prompts import build_messages, load_trials


def main() -> None:
    p = argparse.ArgumentParser(description="Export exact unrendered messages without loading model weights")
    p.add_argument("--config", required=True); p.add_argument("--output")
    args = p.parse_args(); cfg = ExperimentConfig.load(args.config)
    trials = load_trials(cfg.manifest_path, cfg.baseline_results_path, cfg.question_ids, 1)
    trial = trials[0]
    result = {
        "question_id": trial.question_id,
        "prompt_mode": cfg.prompt_mode,
        "conditions": {c: build_messages(trial.question, c, cfg.prompt_mode) for c in cfg.conditions},
        "notes": [
            "Chat-template rendering happens on the GPU host with the checkpoint's own processor.",
            "Neutral setup_text=None is normalized away; neither the literal word 'None' nor a leading blank line is model-visible.",
        ],
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__": main()
