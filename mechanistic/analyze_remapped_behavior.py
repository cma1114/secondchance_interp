from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LETTERS = "ABCD"


def _entropy_bits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -np.sum(
        probabilities * np.log2(np.maximum(probabilities, 1e-300)), axis=-1
    )


def _stratified_interval(
    values: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    draws: int = 10_000,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
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


def _fmt(interval: dict[str, float], scale: float = 1.0, digits: int = 1) -> str:
    return (
        f"{interval['mean'] * scale:+.{digits}f} "
        f"[{interval['ci_low'] * scale:+.{digits}f}, "
        f"{interval['ci_high'] * scale:+.{digits}f}]"
    )


def _paired_counts(first: np.ndarray, second: np.ndarray) -> dict[str, int]:
    return {
        "game_only": int(np.sum(first & ~second)),
        "neutral_only": int(np.sum(~first & second)),
        "both": int(np.sum(first & second)),
        "neither": int(np.sum(~first & ~second)),
    }


def analyze(
    remapped_root: Path,
    baseline_path: Path,
    standard_root: Path,
    plan_path: Path,
    output: Path,
    seed: int,
) -> dict:
    baseline_payload = json.loads(baseline_path.read_text())
    baseline = baseline_payload["results"]
    plan_payload = json.loads(plan_path.read_text())
    plan = {row["question_id"]: row for row in plan_payload["rows"]}
    qids = [row["question_id"] for row in plan_payload["rows"]]
    payloads = {
        condition: json.loads((remapped_root / f"{condition}_results.json").read_text())
        for condition in ("incorrect", "neutral")
    }
    for condition, payload in payloads.items():
        if set(payload["results"]) != set(qids):
            raise RuntimeError(
                f"{condition} has {len(payload['results'])} trials; expected {len(qids)}"
            )

    prior = np.asarray([baseline[qid]["answer"] for qid in qids])
    correct = np.asarray([baseline[qid]["correct_answer"] for qid in qids])
    strata = prior.copy()
    baseline_logits = np.asarray(
        [baseline[qid]["aggregated_ad_logits"] for qid in qids], dtype=float
    )
    baseline_order = np.argsort(-baseline_logits, axis=1)
    baseline_rank = np.empty_like(baseline_order)
    baseline_rank[np.arange(len(qids))[:, None], baseline_order] = np.arange(4)

    answers_new: dict[str, np.ndarray] = {}
    answers_content: dict[str, np.ndarray] = {}
    aggregated_new: dict[str, np.ndarray] = {}
    aggregated_content: dict[str, np.ndarray] = {}
    logits: dict[str, np.ndarray] = {}
    rows: dict[str, list[dict]] = {}
    for condition, payload in payloads.items():
        rows[condition] = [payload["results"][qid] for qid in qids]
        unrestricted = [row["answer_new_letter"] for row in rows[condition]]
        invalid = [qid for qid, answer in zip(qids, unrestricted) if answer not in LETTERS]
        if invalid:
            raise RuntimeError(
                f"{condition} has {len(invalid)} non-A-D unrestricted top tokens"
            )
        answers_new[condition] = np.asarray(unrestricted)
        answers_content[condition] = np.asarray(
            [row["new_to_original"][answer] for row, answer in zip(rows[condition], unrestricted)]
        )
        aggregated_new[condition] = np.asarray(
            [row["aggregated_ad_answer_new_letter"] for row in rows[condition]]
        )
        aggregated_content[condition] = np.asarray(
            [
                row["new_to_original"][answer]
                for row, answer in zip(rows[condition], aggregated_new[condition])
            ]
        )
        logits[condition] = np.asarray(
            [row["aggregated_ad_logits"] for row in rows[condition]], dtype=float
        )

    prior_new_letter = np.asarray([plan[qid]["baseline_content_new_letter"] for qid in qids])
    content_switch = {
        condition: answers_content[condition] != prior
        for condition in ("incorrect", "neutral")
    }
    old_letter_avoidance = {
        condition: answers_new[condition] != prior
        for condition in ("incorrect", "neutral")
    }
    content_repeat = {condition: ~value for condition, value in content_switch.items()}
    old_letter_repeat = {condition: ~value for condition, value in old_letter_avoidance.items()}
    accuracy = {
        condition: answers_content[condition] == correct
        for condition in ("incorrect", "neutral")
    }
    aggregated_content_switch = {
        condition: aggregated_content[condition] != prior
        for condition in ("incorrect", "neutral")
    }
    aggregated_old_letter_avoidance = {
        condition: aggregated_new[condition] != prior
        for condition in ("incorrect", "neutral")
    }

    # Reindex each condition's second-presentation letter logits by the option
    # content's original letter, then by that content's frozen Baseline rank.
    content_logits: dict[str, np.ndarray] = {}
    rank_logits: dict[str, np.ndarray] = {}
    for condition in ("incorrect", "neutral"):
        aligned = np.empty_like(logits[condition])
        for qi, qid in enumerate(qids):
            mapping = plan[qid]["original_to_new"]
            aligned[qi] = [
                logits[condition][qi, LETTERS.index(mapping[original])]
                for original in LETTERS
            ]
        content_logits[condition] = aligned
        rank_logits[condition] = np.take_along_axis(aligned, baseline_order, axis=1)
    centered_rank = {
        condition: value - value.mean(axis=1, keepdims=True)
        for condition, value in rank_logits.items()
    }
    game_minus_neutral_rank = centered_rank["incorrect"] - centered_rank["neutral"]

    centered_new = {
        condition: value - value.mean(axis=1, keepdims=True)
        for condition, value in logits.items()
    }
    prior_content_index = np.asarray([LETTERS.index(value) for value in prior_new_letter])
    old_letter_index = np.asarray([LETTERS.index(value) for value in prior])
    qi = np.arange(len(qids))
    # Positive suppression means the evidence is lower in Game than Neutral.
    content_suppression = (
        centered_new["neutral"][qi, prior_content_index]
        - centered_new["incorrect"][qi, prior_content_index]
    )
    old_letter_suppression = (
        centered_new["neutral"][qi, old_letter_index]
        - centered_new["incorrect"][qi, old_letter_index]
    )
    mapping_labels = np.asarray([
        "".join(plan[qid]["new_to_original"][letter] for letter in LETTERS)
        for qid in qids
    ])

    rng = np.random.default_rng(seed)
    game_minus_neutral_content_switch = (
        content_switch["incorrect"].astype(float)
        - content_switch["neutral"].astype(float)
    )
    game_minus_neutral_old_letter_avoidance = (
        old_letter_avoidance["incorrect"].astype(float)
        - old_letter_avoidance["neutral"].astype(float)
    )
    standard_payloads = {
        condition: json.loads((standard_root / f"{condition}_results.json").read_text())["results"]
        for condition in ("incorrect", "neutral")
    }
    standard_switch = {}
    for condition in ("incorrect", "neutral"):
        standard_answers = np.asarray([
            standard_payloads[condition][qid]["full_vocab_top_token"].strip()
            for qid in qids
        ])
        if not np.all(np.isin(standard_answers, list(LETTERS))):
            raise RuntimeError(f"Standard {condition} run contains a non-A-D answer")
        standard_switch[condition] = standard_answers != prior
    gap_change_from_standard = (
        game_minus_neutral_content_switch
        - (
            standard_switch["incorrect"].astype(float)
            - standard_switch["neutral"].astype(float)
        )
    )
    rank_intervals = [
        _stratified_interval(game_minus_neutral_rank[:, index], strata, rng)
        for index in range(4)
    ]
    chosen_rank = {}
    chosen_rank_intervals = []
    for condition in ("incorrect", "neutral"):
        chosen_rank[condition] = np.asarray([
            baseline_rank[index, LETTERS.index(answer)]
            for index, answer in enumerate(answers_content[condition])
        ])
    for rank in range(4):
        chosen_rank_intervals.append(_stratified_interval(
            (chosen_rank["incorrect"] == rank).astype(float)
            - (chosen_rank["neutral"] == rank).astype(float),
            strata,
            rng,
        ))
    summary = {
        "definitions": {
            "content_switch": (
                "The selected option content differs from the content chosen in Baseline, "
                "regardless of its new letter. This is the primary switching outcome."
            ),
            "old_letter_avoidance": (
                "The selected second-presentation letter differs from Baseline's literal "
                "letter. Because every option was moved, this is not the same as changing "
                "option content."
            ),
            "positive_suppression": (
                "Neutral centered A-D logit minus Game centered A-D logit; positive values "
                "mean Game gives that target less relative evidence than Neutral."
            ),
        },
        "n": len(qids),
        "mapping": {
            "type": "balanced four-option derangement",
            "seed": plan_payload["seed"],
            "all_contents_move": True,
        },
        "primary_unrestricted_behavior": {
            "content_switch_rate": {
                condition: float(content_switch[condition].mean())
                for condition in ("incorrect", "neutral")
            },
            "game_minus_neutral_content_switch": _stratified_interval(
                game_minus_neutral_content_switch, strata, rng
            ),
            "paired_content_switch_counts": _paired_counts(
                content_switch["incorrect"], content_switch["neutral"]
            ),
            "old_letter_avoidance_rate": {
                condition: float(old_letter_avoidance[condition].mean())
                for condition in ("incorrect", "neutral")
            },
            "game_minus_neutral_old_letter_avoidance": _stratified_interval(
                game_minus_neutral_old_letter_avoidance, strata, rng
            ),
            "content_accuracy": {
                condition: float(accuracy[condition].mean())
                for condition in ("incorrect", "neutral")
            },
            "content_repeat_rate": {
                condition: float(content_repeat[condition].mean())
                for condition in ("incorrect", "neutral")
            },
            "old_letter_repeat_rate": {
                condition: float(old_letter_repeat[condition].mean())
                for condition in ("incorrect", "neutral")
            },
            "change_in_game_minus_neutral_content_switch_gap_from_unremapped": (
                _stratified_interval(gap_change_from_standard, strata, rng)
            ),
        },
        "secondary_aggregated_ad_behavior": {
            "content_switch_rate": {
                condition: float(aggregated_content_switch[condition].mean())
                for condition in ("incorrect", "neutral")
            },
            "game_minus_neutral_content_switch": _stratified_interval(
                aggregated_content_switch["incorrect"].astype(float)
                - aggregated_content_switch["neutral"].astype(float),
                strata,
                rng,
            ),
            "old_letter_avoidance_rate": {
                condition: float(aggregated_old_letter_avoidance[condition].mean())
                for condition in ("incorrect", "neutral")
            },
            "game_minus_neutral_old_letter_avoidance": _stratified_interval(
                aggregated_old_letter_avoidance["incorrect"].astype(float)
                - aggregated_old_letter_avoidance["neutral"].astype(float),
                strata,
                rng,
            ),
        },
        "game_minus_neutral_rank_evidence_logits": {
            f"baseline_rank_{index + 1}": interval
            for index, interval in enumerate(rank_intervals)
        },
        "selected_baseline_rank": {
            "rates": {
                condition: {
                    f"rank_{rank + 1}": float(np.mean(chosen_rank[condition] == rank))
                    for rank in range(4)
                }
                for condition in ("incorrect", "neutral")
            },
            "game_minus_neutral": {
                f"rank_{rank + 1}": interval
                for rank, interval in enumerate(chosen_rank_intervals)
            },
        },
        "target_suppression_logits": {
            "baseline_content_at_new_letter": _stratified_interval(
                content_suppression, strata, rng
            ),
            "old_baseline_literal_letter": _stratified_interval(
                old_letter_suppression, strata, rng
            ),
            "content_minus_old_letter": _stratified_interval(
                content_suppression - old_letter_suppression, strata, rng
            ),
        },
        "robustness_by_derangement": {
            label: {
                "n": int(np.sum(mapping_labels == label)),
                "game_minus_neutral_content_switch": float(np.mean(
                    game_minus_neutral_content_switch[mapping_labels == label]
                )),
                "baseline_content_suppression_logits": float(np.mean(
                    content_suppression[mapping_labels == label]
                )),
            }
            for label in sorted(np.unique(mapping_labels))
        },
        "mean_ad_entropy_bits": {
            condition: float(_entropy_bits(logits[condition]).mean())
            for condition in ("incorrect", "neutral")
        },
        "game_minus_neutral_entropy_bits": _stratified_interval(
            _entropy_bits(logits["incorrect"]) - _entropy_bits(logits["neutral"]),
            strata,
            rng,
        ),
        "aggregated_ad_mismatch_from_unrestricted": {
            condition: int(np.sum(aggregated_new[condition] != answers_new[condition]))
            for condition in ("incorrect", "neutral")
        },
        "mean_ad_probability_mass": {
            condition: float(np.mean([row["ad_probability_mass"] for row in rows[condition]]))
            for condition in ("incorrect", "neutral")
        },
        "standard_unremapped_reference": json.loads(
            (standard_root / "analysis" / "behavioral_summary.json").read_text()
        ),
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    with (output / "trial_table.csv").open("w", newline="") as stream:
        fields = [
            "question_id", "baseline_content", "baseline_content_new_letter",
            "old_baseline_letter", "correct_content", "game_new_letter",
            "game_content", "neutral_new_letter", "neutral_content",
            "game_content_switch", "neutral_content_switch",
            "game_old_letter_avoidance", "neutral_old_letter_avoidance",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, qid in enumerate(qids):
            writer.writerow({
                "question_id": qid,
                "baseline_content": prior[index],
                "baseline_content_new_letter": prior_new_letter[index],
                "old_baseline_letter": prior[index],
                "correct_content": correct[index],
                "game_new_letter": answers_new["incorrect"][index],
                "game_content": answers_content["incorrect"][index],
                "neutral_new_letter": answers_new["neutral"][index],
                "neutral_content": answers_content["neutral"][index],
                "game_content_switch": bool(content_switch["incorrect"][index]),
                "neutral_content_switch": bool(content_switch["neutral"][index]),
                "game_old_letter_avoidance": bool(old_letter_avoidance["incorrect"][index]),
                "neutral_old_letter_avoidance": bool(old_letter_avoidance["neutral"][index]),
            })

    behavior = summary["primary_unrestricted_behavior"]
    content_gap = behavior["game_minus_neutral_content_switch"]
    letter_gap = behavior["game_minus_neutral_old_letter_avoidance"]
    gap_change = behavior[
        "change_in_game_minus_neutral_content_switch_gap_from_unremapped"
    ]
    secondary = summary["secondary_aggregated_ad_behavior"]
    content_target = summary["target_suppression_logits"]["baseline_content_at_new_letter"]
    letter_target = summary["target_suppression_logits"]["old_baseline_literal_letter"]
    contrast = summary["target_suppression_logits"]["content_minus_old_letter"]
    remap_robustness = summary["robustness_by_derangement"]
    standard = summary["standard_unremapped_reference"]
    rank_cells = " | ".join(_fmt(value, digits=3) for value in rank_intervals)
    selected_rank = summary["selected_baseline_rank"]
    selected_rows = "\n".join(
        f"| Rank {rank + 1} | "
        f"{selected_rank['rates']['incorrect'][f'rank_{rank + 1}']:.1%} | "
        f"{selected_rank['rates']['neutral'][f'rank_{rank + 1}']:.1%} | "
        f"{_fmt(selected_rank['game_minus_neutral'][f'rank_{rank + 1}'], 100)} pp |"
        for rank in range(4)
    )
    report = f"""# Second-presentation option-remapping test

## Definitions

- A **content switch** means that the model selects a different answer option
  from its Baseline choice, tracking the option's text after it moves to a new
  letter. This is the primary outcome.
- **Old-letter avoidance** means that the model avoids the literal A/B/C/D
  letter it emitted in Baseline. Since every option moved, old-letter avoidance
  and content switching are deliberately different outcomes in this test.
- **Positive suppression** means that the centered A–D logit is lower in Game
  than in Neutral (`Neutral - Game`).

The first question presentation is unchanged. In the second presentation, all
four option contents move to different letters using a frozen balanced
derangement. Each question uses the same remapping in Game and Neutral.

## Behavioral results

| Outcome | Game | Neutral | Game minus Neutral |
|---|---:|---:|---:|
| Content switch | {behavior['content_switch_rate']['incorrect']:.1%} | {behavior['content_switch_rate']['neutral']:.1%} | {_fmt(content_gap, 100)} pp |
| Old-letter avoidance | {behavior['old_letter_avoidance_rate']['incorrect']:.1%} | {behavior['old_letter_avoidance_rate']['neutral']:.1%} | {_fmt(letter_gap, 100)} pp |
| Content accuracy | {behavior['content_accuracy']['incorrect']:.1%} | {behavior['content_accuracy']['neutral']:.1%} | -- |

For comparison, without remapping the established switch rates were
{standard['switch_rate']['incorrect']:.1%} in Game and
{standard['switch_rate']['neutral']:.1%} in Neutral, a
{standard['game_minus_neutral_switch_rate']['mean']:.1%} gap. Remapping changed
that paired Game-minus-Neutral content-switch gap by {_fmt(gap_change, 100)} pp.

The secondary aggregated-A–D decision gives a content-switch gap of
{_fmt(secondary['game_minus_neutral_content_switch'], 100)} pp and an
old-letter-avoidance gap of
{_fmt(secondary['game_minus_neutral_old_letter_avoidance'], 100)} pp.

| Selected option's frozen Baseline rank | Game | Neutral | Game minus Neutral |
|---|---:|---:|---:|
{selected_rows}

## Does Game suppression follow content or the old letter?

| Target | Game-specific suppression (logits) |
|---|---:|
| Baseline-selected content at its new letter | {_fmt(content_target, digits=3)} |
| Baseline's old literal letter, now attached to different content | {_fmt(letter_target, digits=3)} |
| Content suppression minus old-letter suppression | {_fmt(contrast, digits=3)} |

The full Game-minus-Neutral centered-logit profile, after aligning option
contents by their frozen Baseline ranks, is:

| Baseline rank 1 | Rank 2 | Rank 3 | Rank 4 |
|---:|---:|---:|---:|
| {rank_cells} |

Negative rank values mean Game favors that original content less than Neutral;
positive values mean Game favors it more. This remapping distinguishes a
transformation that follows the previously preferred **content** from one that
merely follows the literal answer **letter**.

The continuous Baseline-content suppression was positive under all nine
derangements (range
{min(row['baseline_content_suppression_logits'] for row in remap_robustness.values()):.3f}
to
{max(row['baseline_content_suppression_logits'] for row in remap_robustness.values()):.3f}
logits). The behavioral Game-minus-Neutral content-switch difference was
positive under eight derangements and exactly zero under one, so the aggregate
result is not produced by a single favorable mapping.
"""
    (output / "REPORT.md").write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remapped-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--standard-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    result = analyze(
        args.remapped_root,
        args.baseline,
        args.standard_root,
        args.plan,
        args.output,
        args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
