#!/usr/bin/env python3
"""Run the frozen TriviaMC replication on Qwen3.6 27B."""

from pathlib import Path

import run_triviamc_reproduction as runner


runner.MODEL = "qwen3.6-27b"
runner.DATASET = "TriviaMC"
runner.N_QUESTIONS = 500
runner.SEED = 42
runner.OUTPUT_DIR = Path("outputs/reproduction/triviamc_qwen36_27b")
runner.PREFLIGHT_PATH = runner.OUTPUT_DIR / "preflight.json"
runner.MANIFEST_PATH = runner.OUTPUT_DIR / "stimulus_manifest.json"
runner.STATE_PATH = runner.OUTPUT_DIR / "run_state.json"
runner.COMPILED_PATH = (
    Path("compiled_results_triviamc_qwen36_27b")
    / f"{runner.MODEL}_phase1_compiled.json"
)

# This paid, smaller-model route is burst-tested in preflight rather than
# inheriting the 18-RPM pacing needed by the 235B Parasail endpoint.
runner.PREFLIGHT_REQUEST_INTERVAL = 0.0
runner.PIN_PREFLIGHT_PROVIDER = True


if __name__ == "__main__":
    runner.main()
