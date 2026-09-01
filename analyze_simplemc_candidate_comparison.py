#!/usr/bin/env python3
"""Build a compact comparison of the new SimpleMC candidate screens."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("outputs/reproduction")
CANDIDATES = {
    "Qwen3.5-122B-A10B": ROOT / "simplemc_qwen35_122b_a10b",
    "Gemma 4 26B-A4B IT": ROOT / "simplemc_gemma4_26b_a4b_it",
}
REFERENCE = ROOT / "simplemc_qwen36_27b" / "replication_summary.json"
OUTPUT = ROOT / "SIMPLEMC_CANDIDATE_COMPARISON.md"


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float) -> str:
    return f"{value:.1%}"


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def main() -> None:
    summaries = {
        name: load(directory / "replication_summary.json")
        for name, directory in CANDIDATES.items()
    }
    probabilities = {
        name: load(directory / "PROBABILITY_ENTROPY.json")
        for name, directory in CANDIDATES.items()
    }
    states = {
        name: load(directory / "run_state.json") for name, directory in CANDIDATES.items()
    }
    qwen36 = load(REFERENCE)

    test_rows = []
    for name, summary in summaries.items():
        lift = summary["lift"]
        acc = summary["paper_accincor_changed_baseline_incorrect"]
        runner = summary["runner_up"]["game"]
        entropy = summary["entropy_minus_baseline"]["game"][
            "paper_top4_raw_token_entropy"
        ]
        tests = summary["paper_tests"]
        test_rows.append(
            f"| {name} | {states[name]['n_questions']} | {pct(summary['baseline']['accuracy'])} | "
            f"{pct(lift['game_rate'])} | {pct(lift['neutral_rate'])} | "
            f"{signed(lift['absolute_lift'], 3)} (p={lift['mcnemar_exact_p']:.3g}) | "
            f"{pct(acc['accuracy'])} | {pct(runner['rate'])} | {signed(entropy['mean'])} bits | "
            f"{'✓' if tests['Lift'] else 'X'} / {'✓' if tests['AccIncor'] else 'X'} / "
            f"{'✓' if tests['SecChoice'] else 'X'} / {'✓' if tests['NoEntInc'] else 'X'} |"
        )

    level_rows = []
    condition_names = {
        "baseline": "Baseline",
        "incorrect_feedback": "Game",
        "neutral": "Neutral",
    }
    for name, probability in probabilities.items():
        levels = probability["normalized_A_D_probability_levels"]
        for key in ("baseline", "incorrect_feedback", "neutral"):
            row = levels[key]
            level_rows.append(
                f"| {name} | {condition_names[key]} | {row['n']} | "
                f"{pct(row['first_choice_mean'])} | {pct(row['runner_up_mean'])} | "
                f"{pct(row['mean_rank_3_4'])} |"
            )

    movement_rows = []
    contrast_names = {
        "incorrect_minus_baseline": "Game − baseline",
        "neutral_minus_baseline": "Neutral − baseline",
        "incorrect_minus_neutral": "Game − neutral",
    }
    for name, probability in probabilities.items():
        movements = probability["normalized_A_D_probability_movements"]
        for key in (
            "incorrect_minus_baseline",
            "neutral_minus_baseline",
            "incorrect_minus_neutral",
        ):
            row = movements[key]
            movement_rows.append(
                f"| {name} | {contrast_names[key]} | "
                f"{signed(row['first_choice_probability']['mean'])} | "
                f"{signed(row['runner_up_probability']['mean'])} | "
                f"{signed(row['mean_rank_3_4_probability']['mean'])} |"
            )

    qwen_summary = summaries["Qwen3.5-122B-A10B"]
    gemma_summary = summaries["Gemma 4 26B-A4B IT"]
    report = f"""# SimpleMC candidate behavioral screen

## Main result

**Qwen3.5-122B-A10B reproduces the established Qwen three-of-four profile.** It shows a large, statistically reliable Game-versus-neutral switching lift, above-chance correctness after changing from a baseline-incorrect answer, and strong second-choice selection. It fails entropy preservation decisively: Game entropy rises by {qwen_summary['entropy_minus_baseline']['game']['paper_top4_raw_token_entropy']['mean']:+.3f} bits.

**Gemma 4 26B-A4B IT is not behaviorally successful in the Game.** It switches almost equally often in Game and neutral ({gemma_summary['lift']['game_rate']:.1%} versus {gemma_summary['lift']['neutral_rate']:.1%}; paired p={gemma_summary['lift']['mcnemar_exact_p']:.3g}). Its AccIncor and SecChoice passes therefore do not identify a feedback-specific ability: the neutral redo produces essentially the same redistribution.

## Paper tests

The final column is Lift / changed-trial AccIncor / SecChoice / NoEntInc.

| Model | Paired valid n | Baseline accuracy | Game switch | Neutral switch | Absolute lift | AccIncor | Second choice | Game − baseline entropy | Passes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(test_rows)}

## Where the A–D probability goes

Probabilities are aggregated over letter-token variants and renormalized over A–D. “Lower option” is the mean of the baseline-defined third- and fourth-ranked choices.

| Model | Condition | Covered n | Original choice | Runner-up | Mean lower option |
|---|---|---:|---:|---:|---:|
{chr(10).join(level_rows)}

| Model | Contrast | Original choice Δ | Runner-up Δ | Each lower option Δ |
|---|---|---:|---:|---:|
{chr(10).join(movement_rows)}

For Qwen, incorrect feedback selectively changes the distribution relative to both baseline and neutral: the original choice loses about 17.5 percentage points relative to neutral, while the runner-up gains 5.5 and each lower option gains 6.0. This is targeted original-choice suppression plus broad flattening—not runner-up boosting alone. Neutral is close to baseline.

For Gemma, the large baseline-to-redo redistribution is generic. Game and neutral each lower the original choice by about 20 percentage points, raise the runner-up by about 13, and raise each lower option by about 3.5. Their direct differences are approximately zero. Thus Gemma supplies a useful negative control for the distinction between responding to an unusual redo context and responding strategically to incorrect feedback.

## Relation to Qwen3.6-27B

The 122B Qwen has higher baseline SimpleMC accuracy than Qwen3.6-27B ({qwen_summary['baseline']['accuracy']:.1%} versus {qwen36['baseline']['accuracy']:.1%}) and a similar qualitative signature. Its absolute switching lift is {qwen_summary['lift']['absolute_lift']:.1%}, compared with {qwen36['lift']['absolute_lift']:.1%} for Qwen3.6-27B. Its entropy increase is more extreme: {qwen_summary['entropy_minus_baseline']['game']['paper_top4_raw_token_entropy']['mean']:+.3f} versus {qwen36['entropy_minus_baseline']['game']['paper_top4_raw_token_entropy']['mean']:+.3f} bits.

## Capture and exclusions

- Both models used the exact frozen 500 SimpleMC questions and option assignments, temperature zero, redacted first response, clean neutral prompt, and explicit reasoning-off routing.
- Qwen was pinned to Novita BF16 and returned five top alternatives, with complete A–D coverage on 498/500 baseline trials and all 500 redo trials.
- Gemma was pinned to NextBit BF16 and returned twenty alternatives. Five trials had a non-A–D generated response in at least one condition and were excluded as a paired union from behavioral tests (495 usable). All 500 raw probability captures are retained and used in the separate probability sensitivity file.
- Preflight logs record zero reasoning tokens and no reasoning content for both models.
"""
    OUTPUT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
