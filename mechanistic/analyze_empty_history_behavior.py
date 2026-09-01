from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .data import load_activation_dataset


LETTERS = "ABCD"


def _entropy_bits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-300)), axis=-1)


def _paired_interval(values: np.ndarray, rng: np.random.Generator, draws: int = 10000):
    selected = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[selected].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def analyze(new_root: Path, baseline_root: Path, output: Path, seed: int) -> dict:
    baseline = load_activation_dataset(baseline_root, ["baseline"])
    qids = baseline.question_ids
    payloads = {
        condition: json.loads((new_root / f"{condition}_results.json").read_text())
        for condition in ("incorrect", "neutral")
    }
    for condition, payload in payloads.items():
        if set(payload["results"]) != set(qids):
            raise RuntimeError(
                f"{condition} question IDs do not match Baseline: "
                f"{len(payload['results'])} versus {len(qids)}"
            )

    baseline_logits = baseline.logits[:, 0, -1]
    logits = {"baseline": baseline_logits}
    aggregated_answers = {"baseline": baseline_logits.argmax(axis=-1)}
    baseline_tokens = [
        baseline.metadata[(qid, "baseline")]["full_vocab_top_token"].strip()
        for qid in qids
    ]
    if any(token not in LETTERS for token in baseline_tokens):
        raise RuntimeError("The reused Baseline contains a non-A-D unrestricted top token")
    answers = {"baseline": np.asarray([LETTERS.index(token) for token in baseline_tokens])}
    rows = {}
    for condition, payload in payloads.items():
        rows[condition] = [payload["results"][qid] for qid in qids]
        logits[condition] = np.asarray(
            [row["aggregated_ad_logits"] for row in rows[condition]], dtype=float
        )
        aggregated_answers[condition] = logits[condition].argmax(axis=-1)
        tokens = [row["full_vocab_top_token"].strip() for row in rows[condition]]
        if any(token not in LETTERS for token in tokens):
            raise RuntimeError(f"{condition} contains a non-A-D unrestricted top token")
        answers[condition] = np.asarray([LETTERS.index(token) for token in tokens])

    prior = answers["baseline"]
    correct = np.asarray([
        LETTERS.index(baseline.metadata[(qid, "baseline")]["correct_answer"])
        for qid in qids
    ])
    switch = {
        condition: answers[condition] != prior
        for condition in ("incorrect", "neutral")
    }
    aggregated_switch = {
        condition: aggregated_answers[condition] != aggregated_answers["baseline"]
        for condition in ("incorrect", "neutral")
    }
    entropy = {condition: _entropy_bits(value) for condition, value in logits.items()}
    rng = np.random.default_rng(seed)

    baseline_runner_logits = baseline_logits.copy()
    baseline_runner_logits[np.arange(len(prior)), prior] = -np.inf
    runner = baseline_runner_logits.argmax(axis=-1)
    game_changed = switch["incorrect"]
    changed_wrong = game_changed & (prior != correct)

    unrestricted = {}
    for condition in ("incorrect", "neutral"):
        condition_rows = rows[condition]
        top_tokens = [row["full_vocab_top_token"] for row in condition_rows]
        unrestricted[condition] = {
            "top_token_is_ad": int(sum(token.strip() in LETTERS for token in top_tokens)),
            "top_token_exact_left_bracket": int(sum(token == "[" for token in top_tokens)),
            "left_bracket_in_literal_top4": int(sum(
                any(item["token_id"] == row["left_bracket_token_id"] for item in row["full_vocab_top10"][:4])
                for row in condition_rows
            )),
            "left_bracket_rank_at_most_4_including_ties": int(sum(
                row["left_bracket_rank"] <= 4 for row in condition_rows
            )),
            "left_bracket_in_top10": int(sum(
                row["left_bracket_rank"] <= 10 for row in condition_rows
            )),
            "mean_left_bracket_probability": float(np.mean([
                row["left_bracket_probability"] for row in condition_rows
            ])),
            "mean_ad_probability_mass": float(np.mean([
                row["ad_probability_mass"] for row in condition_rows
            ])),
            "top_token_counts": dict(Counter(top_tokens).most_common()),
        }

    game_top_bracket = np.asarray([
        row["full_vocab_top_token"] == "[" for row in rows["incorrect"]
    ])
    neutral_top_bracket = np.asarray([
        row["full_vocab_top_token"] == "[" for row in rows["neutral"]
    ])
    game_top4_bracket = np.asarray([
        any(item["token_id"] == row["left_bracket_token_id"] for item in row["full_vocab_top10"][:4])
        for row in rows["incorrect"]
    ])
    neutral_top4_bracket = np.asarray([
        any(item["token_id"] == row["left_bracket_token_id"] for item in row["full_vocab_top10"][:4])
        for row in rows["neutral"]
    ])

    result = {
        "n": len(qids),
        "accuracy": {
            condition: float(np.mean(answers[condition] == correct))
            for condition in ("baseline", "incorrect", "neutral")
        },
        "switch_rate": {
            condition: float(switch[condition].mean())
            for condition in ("incorrect", "neutral")
        },
        "game_minus_neutral_switch_rate": _paired_interval(
            switch["incorrect"].astype(float) - switch["neutral"].astype(float), rng
        ),
        "paired_switch_counts": {
            "game_only": int(np.sum(switch["incorrect"] & ~switch["neutral"])),
            "neutral_only": int(np.sum(~switch["incorrect"] & switch["neutral"])),
            "both": int(np.sum(switch["incorrect"] & switch["neutral"])),
            "neither": int(np.sum(~switch["incorrect"] & ~switch["neutral"])),
        },
        "aggregated_ad_decision_behavior": {
            "note": (
                "Secondary analysis choosing the letter with the largest summed "
                "bare-plus-leading-space A-D probability. Primary behavior above "
                "uses the unrestricted generated top token, matching the paper."
            ),
            "accuracy": {
                condition: float(np.mean(aggregated_answers[condition] == correct))
                for condition in ("baseline", "incorrect", "neutral")
            },
            "switch_rate": {
                condition: float(aggregated_switch[condition].mean())
                for condition in ("incorrect", "neutral")
            },
            "game_minus_neutral_switch_rate": _paired_interval(
                aggregated_switch["incorrect"].astype(float)
                - aggregated_switch["neutral"].astype(float), rng
            ),
            "mismatches_from_unrestricted_top_token": {
                condition: int(np.sum(aggregated_answers[condition] != answers[condition]))
                for condition in ("baseline", "incorrect", "neutral")
            },
        },
        "game_switches_to_baseline_runner": {
            "hits": int(np.sum(game_changed & (answers["incorrect"] == runner))),
            "n": int(game_changed.sum()),
            "rate": float(np.mean(answers["incorrect"][game_changed] == runner[game_changed])),
        },
        "accincor_changed_baseline_wrong": {
            "hits": int(np.sum(changed_wrong & (answers["incorrect"] == correct))),
            "n": int(changed_wrong.sum()),
            "rate": float(np.mean(answers["incorrect"][changed_wrong] == correct[changed_wrong])),
        },
        "mean_ad_entropy_bits": {
            condition: float(entropy[condition].mean())
            for condition in ("baseline", "incorrect", "neutral")
        },
        "entropy_differences_bits": {
            "game_minus_baseline": _paired_interval(entropy["incorrect"] - entropy["baseline"], rng),
            "neutral_minus_baseline": _paired_interval(entropy["neutral"] - entropy["baseline"], rng),
            "game_minus_neutral": _paired_interval(entropy["incorrect"] - entropy["neutral"], rng),
        },
        "unrestricted": unrestricted,
        "paired_bracket_counts": {
            "top_token": {
                "game_only": int(np.sum(game_top_bracket & ~neutral_top_bracket)),
                "neutral_only": int(np.sum(~game_top_bracket & neutral_top_bracket)),
                "both": int(np.sum(game_top_bracket & neutral_top_bracket)),
            },
            "literal_top4": {
                "game_only": int(np.sum(game_top4_bracket & ~neutral_top4_bracket)),
                "neutral_only": int(np.sum(~game_top4_bracket & neutral_top4_bracket)),
                "both": int(np.sum(game_top4_bracket & neutral_top4_bracket)),
            },
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "behavioral_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "trial_table.csv").open("w", newline="") as stream:
        fields = [
            "question_id", "correct_answer", "baseline_answer", "game_answer",
            "neutral_answer", "game_switch", "neutral_switch", "game_entropy_bits",
            "neutral_entropy_bits", "game_left_bracket_probability",
            "neutral_left_bracket_probability", "game_ad_probability_mass",
            "neutral_ad_probability_mass",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, qid in enumerate(qids):
            writer.writerow({
                "question_id": qid,
                "correct_answer": LETTERS[correct[index]],
                "baseline_answer": LETTERS[answers["baseline"][index]],
                "game_answer": LETTERS[answers["incorrect"][index]],
                "neutral_answer": LETTERS[answers["neutral"][index]],
                "game_switch": bool(switch["incorrect"][index]),
                "neutral_switch": bool(switch["neutral"][index]),
                "game_entropy_bits": float(entropy["incorrect"][index]),
                "neutral_entropy_bits": float(entropy["neutral"][index]),
                "game_left_bracket_probability": rows["incorrect"][index]["left_bracket_probability"],
                "neutral_left_bracket_probability": rows["neutral"][index]["left_bracket_probability"],
                "game_ad_probability_mass": rows["incorrect"][index]["ad_probability_mass"],
                "neutral_ad_probability_mass": rows["neutral"][index]["ad_probability_mass"],
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(analyze(args.new_root, args.baseline_root, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
