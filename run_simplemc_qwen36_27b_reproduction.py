#!/usr/bin/env python3
"""Run Qwen3.6-27B on the exact SimpleMC stimuli used for Qwen3-235B."""

from pathlib import Path

import run_triviamc_reproduction as runner


runner.MODEL = "qwen3.6-27b"
runner.DATASET = "SimpleMC"
runner.N_QUESTIONS = 500
runner.SEED = 42
runner.OUTPUT_DIR = Path("outputs/reproduction/simplemc_qwen36_27b")
runner.PREFLIGHT_PATH = runner.OUTPUT_DIR / "preflight.json"
runner.MANIFEST_PATH = runner.OUTPUT_DIR / "stimulus_manifest.json"
runner.STATE_PATH = runner.OUTPUT_DIR / "run_state.json"
runner.COMPILED_PATH = (
    Path("compiled_results_simplemc_qwen36_27b")
    / f"{runner.MODEL}_phase1_compiled.json"
)
runner.PREFLIGHT_REQUEST_INTERVAL = 0.0
runner.PIN_PREFLIGHT_PROVIDER = True


REFERENCE_BASELINE = Path(
    "compiled_results_smc/qwen3-235b-a22b-2507_phase1_compiled.json"
)


def load_matched_manifest() -> list[dict]:
    """Freeze the exact question objects and A-D assignments from the 235B run."""
    if runner.MANIFEST_PATH.exists():
        return runner.read_json(runner.MANIFEST_PATH, {})["questions"]
    reference = runner.read_json(REFERENCE_BASELINE, {})
    questions = [trial["question"] for trial in reference["results"].values()]
    if len(questions) != runner.N_QUESTIONS:
        raise RuntimeError(
            f"Expected {runner.N_QUESTIONS} reference questions; got {len(questions)}"
        )
    manifest = {
        "dataset": runner.DATASET,
        "source": str(REFERENCE_BASELINE),
        "seed": runner.SEED,
        "n_questions": len(questions),
        "options_preserved_from_reference_run": True,
        "questions": questions,
    }
    runner.write_json(runner.MANIFEST_PATH, manifest)
    return questions


runner.load_manifest = load_matched_manifest


if __name__ == "__main__":
    runner.main()
