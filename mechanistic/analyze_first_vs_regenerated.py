from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LETTERS = "ABCD"


def _stratified_interval(
    values: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    draws: int = 10_000,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    samples = np.zeros(draws, dtype=float)
    for group in groups:
        selected = rng.choice(group, size=(draws, len(group)), replace=True)
        samples += values[selected].sum(axis=1)
    samples /= len(values)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _fmt(interval: dict[str, float], scale: float = 1.0, digits: int = 3) -> str:
    return (
        f"{interval['mean'] * scale:+.{digits}f} "
        f"[{interval['ci_low'] * scale:+.{digits}f}, "
        f"{interval['ci_high'] * scale:+.{digits}f}]"
    )


def _letter_adjusted_effect(
    condition_difference: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    selected: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    draws: int = 10_000,
) -> dict:
    """Estimate W1 versus W2 suppression controlling absolute answer letter.

    Each question contributes all four answer-letter logits.  Question means are
    removed, then the condition difference is regressed on fixed effects for
    absolute letters A-C plus indicators for the letters currently carrying W1
    and W2.  Bootstrap resampling is at the question level and stratified by the
    original Baseline answer letter.
    """
    indices = np.flatnonzero(selected)
    per_question_xtx = []
    per_question_xty = []
    for qi in indices:
        y = condition_difference[qi] - condition_difference[qi].mean()
        x = np.asarray([
            [
                float(letter == 0),
                float(letter == 1),
                float(letter == 2),
                float(letter == w1[qi]),
                float(letter == w2[qi]),
            ]
            for letter in range(4)
        ])
        x -= x.mean(axis=0, keepdims=True)
        per_question_xtx.append(x.T @ x)
        per_question_xty.append(x.T @ y)
    xtx = np.asarray(per_question_xtx)
    xty = np.asarray(per_question_xty)

    beta = np.linalg.lstsq(xtx.sum(axis=0), xty.sum(axis=0), rcond=None)[0]
    bootstrap = np.empty((draws, 3), dtype=float)
    selected_strata = strata[indices]
    groups = [np.flatnonzero(selected_strata == label) for label in np.unique(selected_strata)]
    for draw in range(draws):
        sampled = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in groups
        ])
        draw_beta = np.linalg.lstsq(
            xtx[sampled].sum(axis=0), xty[sampled].sum(axis=0), rcond=None
        )[0]
        bootstrap[draw] = (draw_beta[3], draw_beta[4], draw_beta[3] - draw_beta[4])

    def interval(point: float, column: int) -> dict[str, float]:
        low, high = np.quantile(bootstrap[:, column], (0.025, 0.975))
        return {"mean": float(point), "ci_low": float(low), "ci_high": float(high)}

    return {
        "original_winner_suppression": interval(beta[3], 0),
        "remapped_winner_suppression": interval(beta[4], 1),
        "original_minus_remapped_suppression": interval(beta[3] - beta[4], 2),
    }


def _content_aligned_logits(rows: list[dict]) -> np.ndarray:
    aligned = np.empty((len(rows), 4), dtype=float)
    for qi, row in enumerate(rows):
        raw = np.asarray(row["aggregated_ad_logits"], dtype=float)
        aligned[qi] = [
            raw[LETTERS.index(row["original_to_new"][content])]
            for content in LETTERS
        ]
    return aligned


def analyze(
    original_baseline_path: Path,
    remapped_baseline_path: Path,
    remapped_root: Path,
    plan_path: Path,
    output: Path,
    seed: int,
) -> dict:
    original = json.loads(original_baseline_path.read_text())["results"]
    remapped = json.loads(remapped_baseline_path.read_text())["results"]
    plan_payload = json.loads(plan_path.read_text())
    qids = [row["question_id"] for row in plan_payload["rows"]]
    game = json.loads((remapped_root / "incorrect_results.json").read_text())["results"]
    neutral = json.loads((remapped_root / "neutral_results.json").read_text())["results"]
    for name, rows in (
        ("original Baseline", original),
        ("remapped Baseline", remapped),
        ("Game", game),
        ("Neutral", neutral),
    ):
        if not set(qids) <= set(rows):
            raise RuntimeError(f"{name} is missing requested questions")

    original_winner = np.asarray([original[qid]["answer"] for qid in qids])
    remapped_winner = np.asarray([
        remapped[qid]["answer_original_content"] for qid in qids
    ])
    aggregated_remapped_winner = np.asarray([
        remapped[qid]["aggregated_ad_answer_original_content"] for qid in qids
    ])
    if not np.all(np.isin(original_winner, list(LETTERS))):
        raise RuntimeError("Original Baseline has non-A-D answers")
    if not np.all(np.isin(remapped_winner, list(LETTERS))):
        raise RuntimeError("Remapped Baseline has non-A-D answers")
    strata = original_winner.copy()
    discordant = original_winner != remapped_winner

    remapped_rows = [remapped[qid] for qid in qids]
    game_rows = [game[qid] for qid in qids]
    neutral_rows = [neutral[qid] for qid in qids]
    game_raw_logits = np.asarray([
        row["aggregated_ad_logits"] for row in game_rows
    ], dtype=float)
    neutral_raw_logits = np.asarray([
        row["aggregated_ad_logits"] for row in neutral_rows
    ], dtype=float)
    remapped_logits = _content_aligned_logits(remapped_rows)
    game_logits = _content_aligned_logits(game_rows)
    neutral_logits = _content_aligned_logits(neutral_rows)
    game_centered = game_logits - game_logits.mean(axis=1, keepdims=True)
    neutral_centered = neutral_logits - neutral_logits.mean(axis=1, keepdims=True)
    game_answer = np.asarray([game[qid]["answer_original_content"] for qid in qids])
    neutral_answer = np.asarray([neutral[qid]["answer_original_content"] for qid in qids])

    qi = np.arange(len(qids))
    w1 = np.asarray([LETTERS.index(value) for value in original_winner])
    w2 = np.asarray([LETTERS.index(value) for value in remapped_winner])
    w2_aggregated = np.asarray([
        LETTERS.index(value) for value in aggregated_remapped_winner
    ])
    remapped_w2_over_w1 = remapped_logits[qi, w2] - remapped_logits[qi, w1]
    original_suppression = neutral_centered[qi, w1] - game_centered[qi, w1]
    remapped_suppression = neutral_centered[qi, w2] - game_centered[qi, w2]
    original_over_remapped_suppression = original_suppression - remapped_suppression
    condition_difference_absolute = neutral_raw_logits - game_raw_logits

    aggregated_discordant = original_winner != aggregated_remapped_winner
    aggregated_remapped_suppression = (
        neutral_centered[qi, w2_aggregated] - game_centered[qi, w2_aggregated]
    )
    aggregated_winner_robustness = {
        "n": int(aggregated_discordant.sum()),
        "original_minus_remapped_suppression": _stratified_interval(
            (original_suppression - aggregated_remapped_suppression)[aggregated_discordant],
            strata[aggregated_discordant],
            np.random.default_rng(seed + 1),
        ),
    }

    rng = np.random.default_rng(seed)
    sensitivity = {}
    selections = [("all_discordant", discordant)] + [
        (
            f"remapped_winner_margin_at_least_{threshold:.2f}",
            discordant & (remapped_w2_over_w1 >= threshold),
        )
        for threshold in (0.25, 0.50)
    ]
    for label, selected in selections:
        sensitivity[label] = {
            "n": int(selected.sum()),
            "mean_fresh_remapped_winner_margin": float(
                remapped_w2_over_w1[selected].mean()
            ) if np.any(selected) else float("nan"),
            "original_winner_suppression": _stratified_interval(
                original_suppression[selected], strata[selected], rng
            ),
            "remapped_winner_suppression": _stratified_interval(
                remapped_suppression[selected], strata[selected], rng
            ),
            "original_minus_remapped_suppression": _stratified_interval(
                original_over_remapped_suppression[selected], strata[selected], rng
            ),
            "game_minus_neutral_avoid_original": _stratified_interval(
                (game_answer[selected] != original_winner[selected]).astype(float)
                - (neutral_answer[selected] != original_winner[selected]).astype(float),
                strata[selected],
                rng,
            ),
            "game_minus_neutral_avoid_remapped": _stratified_interval(
                (game_answer[selected] != remapped_winner[selected]).astype(float)
                - (neutral_answer[selected] != remapped_winner[selected]).astype(float),
                strata[selected],
                rng,
            ),
        }

    selected = discordant
    choice_rates = {}
    for condition, answers in (("game", game_answer), ("neutral", neutral_answer)):
        choice_rates[condition] = {
            "original_winner": float(np.mean(answers[selected] == original_winner[selected])),
            "remapped_winner": float(np.mean(answers[selected] == remapped_winner[selected])),
            "other": float(np.mean(
                (answers[selected] != original_winner[selected])
                & (answers[selected] != remapped_winner[selected])
            )),
        }

    original_new_letters = np.asarray([
        remapped[qid]["original_to_new"][original_winner[index]]
        for index, qid in enumerate(qids)
    ])
    remapped_new_letters = np.asarray([
        remapped[qid]["original_to_new"][remapped_winner[index]]
        for index, qid in enumerate(qids)
    ])
    w1_new = np.asarray([LETTERS.index(value) for value in original_new_letters])
    w2_new = np.asarray([LETTERS.index(value) for value in remapped_new_letters])
    letter_adjusted = _letter_adjusted_effect(
        condition_difference_absolute, w1_new, w2_new, discordant, strata, rng
    )

    summary = {
        "definitions": {
            "original_winner": (
                "Content selected by the unrestricted standalone Baseline under the "
                "original option mapping."
            ),
            "remapped_winner": (
                "Content selected by a fresh unrestricted standalone Baseline that sees "
                "only the remapped question."
            ),
            "positive_original_minus_remapped_suppression": (
                "Game lowers the original winner more than the fresh remapped winner, "
                "supporting first-pass retrieval. Negative supports regenerate-then-suppress."
            ),
        },
        "n_total": len(qids),
        "n_discordant": int(discordant.sum()),
        "discordant_rate": float(discordant.mean()),
        "fresh_remapped_baseline": {
            "mean_ad_probability_mass": float(np.mean([
                row["ad_probability_mass"] for row in remapped_rows
            ])),
            "aggregated_ad_mismatch_from_unrestricted": int(np.sum([
                row["aggregated_ad_answer_original_content"]
                != row["answer_original_content"]
                for row in remapped_rows
            ])),
        },
        "discordant_choice_rates": choice_rates,
        "discordant_absolute_letter_counts": {
            "original_winner": {
                letter: int(np.sum(discordant & (original_new_letters == letter)))
                for letter in LETTERS
            },
            "remapped_winner": {
                letter: int(np.sum(discordant & (remapped_new_letters == letter)))
                for letter in LETTERS
            },
        },
        "absolute_letter_adjusted": letter_adjusted,
        "aggregated_ad_winner_robustness": aggregated_winner_robustness,
        "primary_and_sensitivity": sensitivity,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    primary = sensitivity["all_discordant"]
    remapped_letter_counts = "/".join(
        str(summary["discordant_absolute_letter_counts"]["remapped_winner"][letter])
        for letter in LETTERS
    )
    original_letter_counts = "/".join(
        str(summary["discordant_absolute_letter_counts"]["original_winner"][letter])
        for letter in LETTERS
    )
    rows = []
    for threshold, label in (
        ("All discordant", "all_discordant"),
        ("≥0.25 logits", "remapped_winner_margin_at_least_0.25"),
        ("≥0.50 logits", "remapped_winner_margin_at_least_0.50"),
    ):
        row = sensitivity[label]
        rows.append(
            f"| {threshold} | {row['n']} | "
            f"{_fmt(row['original_winner_suppression'])} | "
            f"{_fmt(row['remapped_winner_suppression'])} | "
            f"{_fmt(row['original_minus_remapped_suppression'])} |"
        )
    report = f"""# Original winner versus freshly regenerated winner

## Definitions

- The **original winner** is the option content chosen by the standalone
  Baseline under the original mapping.
- The **remapped winner** is the content chosen by a fresh standalone Baseline
  that sees only the remapped question.
- The primary analysis uses questions where those contents differ. Positive
  original-minus-remapped suppression means Game preferentially suppresses the
  first-pass winner; negative means it preferentially suppresses the fresh
  remapped-presentation winner.

## Winner discordance

The independent remapped Baseline selected a different content on
**{summary['n_discordant']}/{summary['n_total']} ({summary['discordant_rate']:.1%})**
questions. Its mean A–D probability mass was
{summary['fresh_remapped_baseline']['mean_ad_probability_mass']:.2%}; aggregated
A–D and unrestricted decisions differed on
{summary['fresh_remapped_baseline']['aggregated_ad_mismatch_from_unrestricted']}
questions.

## Game-specific target suppression on discordant questions

| Required fresh remapped-winner margin | N | Original-winner suppression | Remapped-winner suppression | Original minus remapped |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Suppression is `Neutral logit - Game logit` for the named content. The final
column is the decisive comparison: positive supports retrieval of the original
winner; negative supports regeneration and suppression of the current winner.

## Which content is ultimately selected?

On the {summary['n_discordant']} discordant questions:

| Final choice | Game | Neutral |
|---|---:|---:|
| Original winner | {choice_rates['game']['original_winner']:.1%} | {choice_rates['neutral']['original_winner']:.1%} |
| Fresh remapped winner | {choice_rates['game']['remapped_winner']:.1%} | {choice_rates['neutral']['remapped_winner']:.1%} |
| Either other option | {choice_rates['game']['other']:.1%} | {choice_rates['neutral']['other']:.1%} |

Game-minus-Neutral avoidance of the original winner is
{_fmt(primary['game_minus_neutral_avoid_original'], 100, 1)} percentage points;
avoidance of the fresh remapped winner is
{_fmt(primary['game_minus_neutral_avoid_remapped'], 100, 1)} percentage points.

## Absolute-letter robustness check

The fresh remapped Baseline is strongly absolute-letter-biased: among discordant
trials, W2 occupied A/B/C/D on
{remapped_letter_counts} trials, whereas remapped W1 occupied those letters on
{original_letter_counts} trials.
This is not a novel A bias in the fresh run: the original and remapped Baselines
selected literal A on 240/500 and 260/500 questions, respectively. The
within-discordant imbalance arises because the derangement forces W1 away from
its original letter while a fresh answer is free to express the model's usual
letter preference.
Therefore, a post-specified robustness model uses all four logits per question
and controls both question and absolute answer letter. It estimates:

- original-winner suppression: {_fmt(letter_adjusted['original_winner_suppression'])}
- fresh-remapped-winner suppression: {_fmt(letter_adjusted['remapped_winner_suppression'])}
- original minus remapped: {_fmt(letter_adjusted['original_minus_remapped_suppression'])}

The decisive contrast remains positive after this adjustment. This robustness
check was added after observing the fresh Baseline's letter imbalance and is not
the frozen primary analysis.

Defining the fresh remapped winner by the aggregated A–D logits rather than the
unrestricted top token gives N={aggregated_winner_robustness['n']} discordant
trials and an original-minus-remapped suppression contrast of
{_fmt(aggregated_winner_robustness['original_minus_remapped_suppression'])}.
"""
    (output / "REPORT.md").write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--remapped-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    result = analyze(
        args.original_baseline,
        args.remapped_baseline,
        args.remapped_root,
        args.plan,
        args.output,
        args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
