#!/usr/bin/env python3
"""Rerun only Qwen3.6-27B SimpleMC neutral with corrected empty setup text."""

from __future__ import annotations

import copy
from pathlib import Path

import run_simplemc_qwen36_27b_reproduction  # configures the shared runner
import run_triviamc_reproduction as runner


SOURCE_STATE = Path("outputs/reproduction/simplemc_qwen36_27b/run_state.json")
OUTPUT_DIR = Path("outputs/reproduction/simplemc_qwen36_27b_clean_neutral")
STATE_PATH = OUTPUT_DIR / "run_state.json"


def initialize_state() -> dict:
    if STATE_PATH.exists():
        return runner.read_json(STATE_PATH, {})
    source = runner.read_json(SOURCE_STATE, {})
    state = {
        "model": source["model"],
        "dataset": source["dataset"],
        "n_questions": 500,
        "temperature": source["temperature"],
        "seed": source["seed"],
        "top_logprobs": source["top_logprobs"],
        "baseline_work_path": source["baseline_work_path"],
        "baseline_compiled_path": source["baseline_compiled_path"],
        "baseline_complete": True,
        "neutral_setup_text": "",
        "expected_system_message": "Output ONLY the letter of your choice: A, B, C, or D.\n",
        "source_contaminated_state": str(SOURCE_STATE),
        "second_chance": {
            "incorrect_baseline_incorrect": copy.deepcopy(
                source["second_chance"]["incorrect_baseline_incorrect"]
            ),
            "incorrect_baseline_correct": copy.deepcopy(
                source["second_chance"]["incorrect_baseline_correct"]
            ),
        },
        "complete": False,
    }
    runner.write_json(STATE_PATH, state)
    return state


def main() -> None:
    preflight = runner.read_json(runner.PREFLIGHT_PATH, None)
    if not preflight:
        raise RuntimeError(f"Missing existing preflight: {runner.PREFLIGHT_PATH}")
    state = initialize_state()
    # The shared resumability helper writes through its module-level state path.
    runner.STATE_PATH = STATE_PATH
    runner.BaseGameClass._openrouter_request_interval = float(
        preflight.get("request_interval_seconds", 0.0)
    )
    runner.BaseGameClass._openrouter_provider_only = preflight.get("selected_provider")
    top_logprobs = int(preflight["selected_top_logprobs"])
    for stage_name, correct in (
        ("neutral_baseline_incorrect", False),
        ("neutral_baseline_correct", True),
    ):
        runner.run_second_chance_stage(
            stage_name, "_neut", correct, top_logprobs, state
        )
    state["complete"] = True
    runner.save_state(state)
    print(f"Corrected neutral run complete: {STATE_PATH}")


if __name__ == "__main__":
    main()
