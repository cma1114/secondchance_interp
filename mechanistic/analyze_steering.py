from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .io import read_metadata
from .run_steering import steering_shard


METRICS = (
    "switch",
    "winner_probability",
    "winner_margin",
    "rank1_centered_logit",
    "rank2_centered_logit",
    "rank3_centered_logit",
    "rank4_centered_logit",
    "runner_relative_logit",
    "entropy",
    "ad_spread",
    "accuracy",
    "baseline_incorrect_accuracy",
    "valid_ad_top_token",
)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def _macro_mean(values: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean([values[labels == letter].mean() for letter in range(4)]))


def _macro_bootstrap(
    values: np.ndarray,
    labels: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    strata = [np.flatnonzero(labels == letter) for letter in range(4)]
    if any(len(ids) == 0 for ids in strata):
        raise ValueError("Letter-balanced bootstrap requires all four winner-letter strata")
    mean = _macro_mean(values, labels)
    stratum_bootstraps = []
    for ids in strata:
        sampled_ids = rng.choice(ids, size=(repetitions, len(ids)), replace=True)
        stratum_bootstraps.append(values[sampled_ids].mean(axis=1))
    boot = np.stack(stratum_bootstraps, axis=1).mean(axis=1)
    return mean, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _scenario_values(
    output_dir: str | Path,
    scenario: dict,
    qids: list[str],
    baseline_logits: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    logits, metadata = [], []
    for qid in qids:
        with np.load(steering_shard(output_dir, scenario["scenario_id"], qid), allow_pickle=False) as data:
            logits.append(data["canonical_logits"].astype(np.float64))
            metadata.append(read_metadata(data))
    logits = np.asarray(logits)
    probabilities = _softmax(logits)
    baseline_order = np.argsort(-baseline_logits, axis=-1)
    winner = baseline_order[:, 0]
    runner = baseline_order[:, 1]
    lower = baseline_order[:, 2:]
    rows = np.arange(len(qids))
    choices = np.argmax(logits, axis=-1)
    correct = np.asarray(["ABCD".index(item["correct_answer"]) for item in metadata])
    baseline_correct = winner == correct
    winner_logits = logits[rows, winner]
    alternative_logits = np.where(np.eye(4)[winner].astype(bool), -np.inf, logits)
    lower_logits = np.take_along_axis(logits, lower, axis=-1)
    ranked_centered_logits = np.take_along_axis(
        logits - logits.mean(axis=-1, keepdims=True), baseline_order, axis=-1
    )
    values = {
        "switch": (choices != winner).astype(float),
        "winner_probability": probabilities[rows, winner],
        "winner_margin": winner_logits - np.max(alternative_logits, axis=-1),
        "rank1_centered_logit": ranked_centered_logits[:, 0],
        "rank2_centered_logit": ranked_centered_logits[:, 1],
        "rank3_centered_logit": ranked_centered_logits[:, 2],
        "rank4_centered_logit": ranked_centered_logits[:, 3],
        "runner_relative_logit": logits[rows, runner] - lower_logits.mean(axis=-1),
        "entropy": -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=-1),
        "ad_spread": np.std(logits, axis=-1),
        "accuracy": (choices == correct).astype(float),
        "baseline_incorrect_accuracy": np.where(~baseline_correct, (choices == correct).astype(float), np.nan),
        "valid_ad_top_token": np.asarray([item["valid_ad_top_token"] for item in metadata], dtype=float),
    }
    return values, metadata


def analyze(
    run_dir: str | Path,
    directions_path: str | Path,
    output_dir: str | Path,
    bootstrap_repetitions: int,
    seed: int,
) -> dict:
    run_dir = Path(run_dir)
    run_metadata = json.loads((run_dir / "run_metadata.json").read_text())
    schedule = run_metadata["schedule"]
    with np.load(directions_path, allow_pickle=False) as data:
        reference_qids = [str(value) for value in data["reference_question_ids"].tolist()]
        reference_logits = data["reference_canonical_logits"].astype(np.float64)
    reference_lookup = {qid: reference_logits[index] for index, qid in enumerate(reference_qids)}
    complete = []
    for qid in reference_qids:
        if all(steering_shard(run_dir, scenario["scenario_id"], qid).exists() for scenario in schedule):
            complete.append(qid)
    if not complete:
        raise FileNotFoundError("No questions have a complete steering schedule")
    baseline_logits = np.asarray([reference_lookup[qid][0] for qid in complete])
    winner_labels = np.argmax(baseline_logits, axis=-1)
    if set(winner_labels.tolist()) != set(range(4)):
        raise ValueError("Complete results do not contain all four baseline-winner letters")

    scenario_values = {}
    scenario_metadata = {}
    for scenario in schedule:
        values, metadata = _scenario_values(run_dir, scenario, complete, baseline_logits)
        scenario_values[scenario["scenario_id"]] = values
        scenario_metadata[scenario["scenario_id"]] = metadata

    unsteered = {
        scenario["condition"]: scenario_values[scenario["scenario_id"]]
        for scenario in schedule if scenario["direction_kind"] == "none"
    }
    rng = np.random.default_rng(seed)
    rows = []
    for scenario in schedule:
        scenario_id = scenario["scenario_id"]
        for metric in METRICS:
            values = scenario_values[scenario_id][metric]
            if metric == "baseline_incorrect_accuracy":
                mask = np.isfinite(values)
                metric_values = values[mask]
                metric_labels = winner_labels[mask]
            else:
                metric_values = values
                metric_labels = winner_labels
            mean, low, high = _macro_bootstrap(
                metric_values, metric_labels, bootstrap_repetitions, rng
            )
            baseline_values = unsteered[scenario["condition"]][metric]
            difference = values - baseline_values
            if metric == "baseline_incorrect_accuracy":
                difference = difference[np.isfinite(difference)]
            delta_mean, delta_low, delta_high = _macro_bootstrap(
                difference,
                metric_labels,
                bootstrap_repetitions,
                rng,
            )
            rows.append({
                "scenario_id": scenario_id,
                "condition": scenario["condition"],
                "direction_kind": scenario["direction_kind"],
                "readout": scenario["readout"],
                "dose": scenario["dose"],
                "metric": metric,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "paired_delta_vs_unsteered": delta_mean,
                "paired_delta_ci_low": delta_low,
                "paired_delta_ci_high": delta_high,
                "n_questions": len(metric_values),
            })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "steering_effects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lookup = {(row["scenario_id"], row["metric"]): row for row in rows}

    selected = {}
    for scenario in schedule:
        if (
            scenario["direction_kind"] == "none"
            or scenario["readout"] == run_metadata["steering_config"]["detailed_readout"]
        ):
            selected[scenario["scenario_id"]] = {
                metric: {
                    "mean": lookup[(scenario["scenario_id"], metric)]["mean"],
                    "paired_delta_vs_unsteered": lookup[(scenario["scenario_id"], metric)]["paired_delta_vs_unsteered"],
                    "paired_delta_ci": [
                        lookup[(scenario["scenario_id"], metric)]["paired_delta_ci_low"],
                        lookup[(scenario["scenario_id"], metric)]["paired_delta_ci_high"],
                    ],
                }
                for metric in ("switch", "winner_margin", "entropy", "accuracy", "valid_ad_top_token")
            }
    summary = {
        "run_dir": str(run_dir),
        "n_complete_questions": len(complete),
        "bootstrap_repetitions": bootstrap_repetitions,
        "estimand": "Original-baseline-winner-letter-balanced paired effect versus the same condition unsteered.",
        "selected_scenarios": selected,
    }
    (output_dir / "steering_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze causal feedback-direction steering")
    parser.add_argument("--run", required=True)
    parser.add_argument("--directions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.directions, args.output, args.bootstrap, args.seed), indent=2))


if __name__ == "__main__":
    main()
