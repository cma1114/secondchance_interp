from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import decision_letter, load_activation_dataset
from .io import shard_path


LETTERS = "ABCD"


def _labels(data, condition: str) -> np.ndarray:
    values = []
    for qid in data.question_ids:
        token = decision_letter(data.metadata[(qid, condition)])
        if token not in LETTERS:
            raise ValueError(f"Non-A-D output for {condition}/{qid}: {token!r}")
        values.append(LETTERS.index(token))
    return np.asarray(values, dtype=int)


def _paired_interval(values: np.ndarray, rng: np.random.Generator, draws: int = 10000):
    selected = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[selected].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def _entropy_bits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-300)), axis=-1)


def _aggregated_ad_logits(residual_root: Path, data) -> np.ndarray:
    metadata = json.loads((residual_root / "run_metadata.json").read_text())
    layout = metadata["variant_layout"]
    indices = {
        letter: [index for index, row in enumerate(layout) if row["letter"] == letter]
        for letter in LETTERS
    }
    values = np.empty((len(data.question_ids), len(data.conditions), 4), dtype=np.float64)
    for qi, qid in enumerate(data.question_ids):
        for ci, condition in enumerate(data.conditions):
            with np.load(shard_path(residual_root, condition, qid), allow_pickle=False) as shard:
                logits = shard["variant_logits"][-1].astype(np.float64)
            for li, letter in enumerate(LETTERS):
                selected = logits[indices[letter]]
                maximum = selected.max()
                values[qi, ci, li] = maximum + np.log(np.exp(selected - maximum).sum())
    return values


def analyze(residual_root: Path, output: Path, seed: int = 42) -> dict:
    conditions = ["baseline", "incorrect", "neutral"]
    data = load_activation_dataset(residual_root, conditions)
    generated = {condition: _labels(data, condition) for condition in conditions}
    prior = generated["baseline"]
    correct = np.asarray([
        LETTERS.index(data.metadata[(qid, "baseline")]["correct_answer"])
        for qid in data.question_ids
    ])
    switch = {
        condition: generated[condition] != prior
        for condition in ("incorrect", "neutral")
    }
    rng = np.random.default_rng(seed)

    # Aggregate the valid bare and leading-space variants of A--D before
    # renormalizing, matching the established probability/entropy analysis.
    final_logits = _aggregated_ad_logits(residual_root, data)
    entropy = {
        condition: _entropy_bits(final_logits[:, index])
        for index, condition in enumerate(conditions)
    }
    game_neutral_switch = switch["incorrect"].astype(float) - switch["neutral"].astype(float)
    game_neutral_entropy = entropy["incorrect"] - entropy["neutral"]
    game_baseline_entropy = entropy["incorrect"] - entropy["baseline"]
    neutral_baseline_entropy = entropy["neutral"] - entropy["baseline"]

    baseline_options = final_logits[:, 0].copy()
    baseline_options[np.arange(len(prior)), prior] = -np.inf
    runner = baseline_options.argmax(axis=-1)
    game_changed = switch["incorrect"]
    baseline_wrong = prior != correct
    changed_wrong = game_changed & baseline_wrong

    result = {
        "n": len(prior),
        "accuracy": {
            condition: float(np.mean(generated[condition] == correct))
            for condition in conditions
        },
        "switch_rate": {
            "incorrect": float(switch["incorrect"].mean()),
            "neutral": float(switch["neutral"].mean()),
        },
        "game_minus_neutral_switch_rate": dict(zip(
            ("mean", "ci_low", "ci_high"),
            _paired_interval(game_neutral_switch, rng),
        )),
        "changed_game_trials": int(game_changed.sum()),
        "game_switches_to_baseline_runner": {
            "hits": int(np.sum(game_changed & (generated["incorrect"] == runner))),
            "n": int(game_changed.sum()),
            "rate": float(np.mean(generated["incorrect"][game_changed] == runner[game_changed])),
        },
        "accincor_changed_baseline_wrong": {
            "hits": int(np.sum(changed_wrong & (generated["incorrect"] == correct))),
            "n": int(changed_wrong.sum()),
            "rate": float(np.mean(
                generated["incorrect"][changed_wrong] == correct[changed_wrong]
            )),
        },
        "mean_ad_entropy_bits": {
            condition: float(entropy[condition].mean()) for condition in conditions
        },
        "entropy_differences_bits": {
            "game_minus_baseline": dict(zip(
                ("mean", "ci_low", "ci_high"),
                _paired_interval(game_baseline_entropy, rng),
            )),
            "neutral_minus_baseline": dict(zip(
                ("mean", "ci_low", "ci_high"),
                _paired_interval(neutral_baseline_entropy, rng),
            )),
            "game_minus_neutral": dict(zip(
                ("mean", "ci_low", "ci_high"),
                _paired_interval(game_neutral_entropy, rng),
            )),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "behavioral_summary.json").write_text(json.dumps(result, indent=2) + "\n")

    lift = result["game_minus_neutral_switch_rate"]
    report = f"""# Qwen3.6-27B SimpleMC baseline-matched Second Chance run

The first question presentation exactly matches Baseline. In the second user
turn, the answer-only instruction again precedes the repeated question, without
repeating the conversational introduction. Game and Neutral differ only in the
feedback sentence.

| Condition | Accuracy | Switch rate | Mean A-D entropy |
|---|---:|---:|---:|
| Baseline | {result['accuracy']['baseline']:.1%} | -- | {result['mean_ad_entropy_bits']['baseline']:.3f} bits |
| Game | {result['accuracy']['incorrect']:.1%} | {result['switch_rate']['incorrect']:.1%} | {result['mean_ad_entropy_bits']['incorrect']:.3f} bits |
| Neutral | {result['accuracy']['neutral']:.1%} | {result['switch_rate']['neutral']:.1%} | {result['mean_ad_entropy_bits']['neutral']:.3f} bits |

Game-minus-Neutral switching is {lift['mean']:+.1%}
[{lift['ci_low']:+.1%}, {lift['ci_high']:+.1%}]. Among Game switches,
{result['game_switches_to_baseline_runner']['hits']}/{result['game_switches_to_baseline_runner']['n']}
({result['game_switches_to_baseline_runner']['rate']:.1%}) go to the Baseline
runner-up. Among changed Baseline-wrong Game trials,
{result['accincor_changed_baseline_wrong']['hits']}/{result['accincor_changed_baseline_wrong']['n']}
({result['accincor_changed_baseline_wrong']['rate']:.1%}) move to the correct answer.
"""
    (output / "BEHAVIORAL_REPORT.md").write_text(report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(analyze(args.residual_root, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
