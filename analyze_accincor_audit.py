#!/usr/bin/env python3
"""Audit unconditional versus change-conditional AccIncor definitions."""

from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import binomtest


ROOT = Path(__file__).parent
OUTPUT = ROOT / "outputs/reproduction/ACCINCOR_AUDIT.json"


def read(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def merged(paths: list[str | Path]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in paths:
        result.update(read(path)["results"])
    return result


def summarize(
    baseline: dict[str, dict], condition: dict[str, dict]
) -> dict[str, float | int]:
    incorrect_ids = [qid for qid, trial in baseline.items() if not trial["is_correct"]]
    changed_ids = [qid for qid in incorrect_ids if condition[qid]["answer_changed"]]
    correct = sum(bool(condition[qid]["is_correct"]) for qid in incorrect_ids)
    if correct != sum(bool(condition[qid]["is_correct"]) for qid in changed_ids):
        raise ValueError("An unchanged baseline-incorrect answer became correct")
    return {
        "baseline_incorrect_n": len(incorrect_ids),
        "changed_n": len(changed_ids),
        "correct_n": correct,
        "unconditional_accuracy": correct / len(incorrect_ids),
        "unconditional_p_vs_25pct": float(
            binomtest(correct, len(incorrect_ids), 0.25, alternative="greater").pvalue
        ),
        "conditional_on_change_accuracy": correct / len(changed_ids),
        "conditional_p_vs_25pct": float(
            binomtest(correct, len(changed_ids), 0.25, alternative="greater").pvalue
        ),
        "conditional_p_vs_one_third": float(
            binomtest(correct, len(changed_ids), 1 / 3, alternative="greater").pvalue
        ),
    }


def compare_conditions(
    baseline: dict[str, dict], game: dict[str, dict], neutral: dict[str, dict]
) -> dict[str, float | int]:
    incorrect_ids = [qid for qid, trial in baseline.items() if not trial["is_correct"]]
    game_only = sum(
        bool(game[qid]["is_correct"]) and not bool(neutral[qid]["is_correct"])
        for qid in incorrect_ids
    )
    neutral_only = sum(
        bool(neutral[qid]["is_correct"]) and not bool(game[qid]["is_correct"])
        for qid in incorrect_ids
    )
    both = sum(
        bool(game[qid]["is_correct"]) and bool(neutral[qid]["is_correct"])
        for qid in incorrect_ids
    )
    discordant = game_only + neutral_only
    return {
        "both_correct": both,
        "game_only_correct": game_only,
        "neutral_only_correct": neutral_only,
        "neither_correct": len(incorrect_ids) - both - discordant,
        "game_minus_neutral_accuracy": (
            sum(bool(game[qid]["is_correct"]) for qid in incorrect_ids)
            - sum(bool(neutral[qid]["is_correct"]) for qid in incorrect_ids)
        ) / len(incorrect_ids),
        "mcnemar_exact_p": float(
            binomtest(game_only, discordant, 0.5).pvalue if discordant else 1.0
        ),
    }


def from_state(path: str | Path) -> tuple[dict, dict, dict]:
    state = read(path)
    baseline = merged([state["baseline_compiled_path"]])
    stages = state["second_chance"]
    game = merged([
        stages["incorrect_baseline_incorrect"]["path"],
        stages["incorrect_baseline_correct"]["path"],
    ])
    neutral = merged([
        stages["neutral_baseline_incorrect"]["path"],
        stages["neutral_baseline_correct"]["path"],
    ])
    return baseline, game, neutral


def main() -> None:
    simple_baseline = merged([
        ROOT / "compiled_results_smc/qwen3-235b-a22b-2507_phase1_compiled.json"
    ])
    simple_game = merged([
        ROOT / "secondchance_game_logs/qwen3-235b-a22b-2507_SimpleMC_redacted_temp0.0_1785521299_game_data.json",
        ROOT / "secondchance_game_logs/qwen3-235b-a22b-2507_SimpleMC_redacted_cor_temp0.0_1785522010_game_data.json",
    ])
    simple_neutral = merged([
        ROOT / "secondchance_game_logs/qwen3-235b-a22b-2507_SimpleMC_neut_redacted_temp0.0_1785523104_game_data.json",
        ROOT / "secondchance_game_logs/qwen3-235b-a22b-2507_SimpleMC_neut_redacted_cor_temp0.0_1785523823_game_data.json",
    ])
    datasets = {
        "SimpleMC": (simple_baseline, simple_game, simple_neutral),
        "TriviaMC_Qwen3_235B": from_state(ROOT / "outputs/reproduction/triviamc/run_state.json"),
        "TriviaMC_Qwen3.6_27B": from_state(
            ROOT / "outputs/reproduction/triviamc_qwen36_27b/run_state.json"
        ),
        "SimpleMC_Qwen3.6_27B": from_state(
            ROOT / "outputs/reproduction/simplemc_qwen36_27b/run_state.json"
        ),
        "PopMC": from_state(ROOT / "outputs/reproduction/popmc/run_state.json"),
    }
    output = {
        dataset: {
            "game": summarize(baseline, game),
            "neutral": summarize(baseline, neutral),
            "game_vs_neutral": compare_conditions(baseline, game, neutral),
        }
        for dataset, (baseline, game, neutral) in datasets.items()
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
