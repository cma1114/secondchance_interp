from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mechanistic.analyze_contextual_option_representations import (
    LAMBDAS,
    _balanced_accuracy,
    _center_normalize,
    _cv_scores,
    _fit_scores,
    _folds,
    _labels,
    _load,
)


LETTERS = "ABCD"
ANCHORS = ("content_end", "line_end")


def _candidate_layer(
    residuals: np.ndarray, metadata: dict, anchor: str, layer: int
) -> np.ndarray:
    indices = [metadata["anchors"].index(f"{anchor}_{letter}") for letter in LETTERS]
    return _center_normalize(np.asarray(residuals[:, layer, indices]).copy())


def _balanced_bootstrap_ci(
    prediction: np.ndarray,
    labels: np.ndarray,
    seed: int,
    draws: int = 5000,
) -> list[float]:
    correct = prediction == labels
    rng = np.random.default_rng(seed)
    values = np.zeros(draws, dtype=np.float32)
    groups = [np.flatnonzero(labels == label) for label in range(4)]
    for group in groups:
        sampled = rng.choice(group, size=(draws, len(group)), replace=True)
        values += correct[sampled].mean(axis=1) / 4
    return np.quantile(values, (0.025, 0.975)).tolist()


def analyze(
    discovery_original_root: Path,
    confirmation_original_root: Path,
    confirmation_remapped_root: Path,
    baseline_results: Path,
    remapped_baseline_results: Path,
    output: Path,
    seed: int,
) -> dict:
    discovery, discovery_meta = _load(discovery_original_root)
    confirmation, confirmation_meta = _load(confirmation_original_root)
    remapped, remapped_meta = _load(confirmation_remapped_root)
    discovery_qids = list(discovery_meta["question_ids"])
    confirmation_qids = list(confirmation_meta["question_ids"])
    if confirmation_qids != list(remapped_meta["question_ids"]):
        raise ValueError("Original/remapped confirmation orders differ")

    discovery_labels = _labels(baseline_results, discovery_qids, "answer")
    confirmation_labels = _labels(baseline_results, confirmation_qids, "answer")
    remapped_labels = _labels(
        remapped_baseline_results, confirmation_qids, "answer_new_letter"
    )
    folds = _folds(discovery_labels, 5, seed)
    discovery_decision = discovery_meta["anchors"].index("first_answer_decision")
    confirmation_decision = confirmation_meta["anchors"].index("first_answer_decision")
    remapped_decision = remapped_meta["anchors"].index("first_answer_decision")

    results: dict[str, dict[str, list]] = {}
    rows = []
    for anchor_index, anchor in enumerate(ANCHORS):
        result = {
            "discovery_cv_balanced_accuracy": [],
            "selected_lambda": [],
            "confirmation_original_balanced_accuracy": [],
            "confirmation_original_ci": [],
            "confirmation_remapped_balanced_accuracy": [],
            "confirmation_remapped_ci": [],
            "confirmation_original_prediction": [],
            "confirmation_remapped_prediction": [],
        }
        for layer in range(64):
            discovery_candidates = _candidate_layer(
                discovery, discovery_meta, anchor, layer
            )
            confirmation_candidates = _candidate_layer(
                confirmation, confirmation_meta, anchor, layer
            )
            remapped_candidates = _candidate_layer(
                remapped, remapped_meta, anchor, layer
            )
            discovery_query = np.asarray(
                discovery[:, layer, discovery_decision]
            ).copy()
            confirmation_query = np.asarray(
                confirmation[:, layer, confirmation_decision]
            ).copy()
            remapped_query = np.asarray(remapped[:, layer, remapped_decision]).copy()

            _, cv_by_lambda = _cv_scores(
                discovery_query,
                discovery_candidates,
                discovery_labels,
                folds,
            )
            selected = int(np.argmax(cv_by_lambda))
            penalty = float(LAMBDAS[selected])
            original_scores = _fit_scores(
                discovery_query,
                discovery_candidates,
                discovery_labels,
                confirmation_query,
                confirmation_candidates,
                penalty,
            )
            remapped_scores = _fit_scores(
                discovery_query,
                discovery_candidates,
                discovery_labels,
                remapped_query,
                remapped_candidates,
                penalty,
            )
            original_prediction = original_scores.argmax(axis=1)
            remapped_prediction = remapped_scores.argmax(axis=1)
            original_accuracy = _balanced_accuracy(
                original_prediction, confirmation_labels
            )
            remapped_accuracy = _balanced_accuracy(remapped_prediction, remapped_labels)
            original_ci = _balanced_bootstrap_ci(
                original_prediction, confirmation_labels, seed + 1000 * anchor_index + layer
            )
            remapped_ci = _balanced_bootstrap_ci(
                remapped_prediction, remapped_labels, seed + 2000 + 1000 * anchor_index + layer
            )

            result["discovery_cv_balanced_accuracy"].append(float(cv_by_lambda[selected]))
            result["selected_lambda"].append(penalty)
            result["confirmation_original_balanced_accuracy"].append(original_accuracy)
            result["confirmation_original_ci"].append(original_ci)
            result["confirmation_remapped_balanced_accuracy"].append(remapped_accuracy)
            result["confirmation_remapped_ci"].append(remapped_ci)
            result["confirmation_original_prediction"].append(original_prediction.tolist())
            result["confirmation_remapped_prediction"].append(remapped_prediction.tolist())
            rows.append({
                "anchor": anchor,
                "layer": layer + 1,
                "selected_lambda": penalty,
                "discovery_cv_balanced_accuracy": float(cv_by_lambda[selected]),
                "confirmation_original_balanced_accuracy": original_accuracy,
                "confirmation_original_ci_low": original_ci[0],
                "confirmation_original_ci_high": original_ci[1],
                "confirmation_remapped_balanced_accuracy": remapped_accuracy,
                "confirmation_remapped_ci_low": remapped_ci[0],
                "confirmation_remapped_ci_high": remapped_ci[1],
            })
            print(
                f"same-layer contextual match {anchor} {layer + 1}/64: "
                f"original={original_accuracy:.3f} remapped={remapped_accuracy:.3f}",
                flush=True,
            )
        results[anchor] = result

    output.mkdir(parents=True, exist_ok=True)
    with (output / "layerwise_same_layer_match.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "definition": (
            "At each layer L, fit a linear-kernel ridge map from the first-answer "
            "decision residual at L to the contextual residual of the chosen option "
            "at L. Score all four question-specific option candidates and select the "
            "highest. Ridge penalty is chosen by five-fold discovery CV; reported "
            "accuracies use the disjoint confirmation questions."
        ),
        "split": {"discovery": len(discovery_qids), "confirmation": len(confirmation_qids)},
        "chance": 0.25,
        "anchors": {
            "content_end": "Last token overlapping the option text.",
            "line_end": "Identical newline token closing the option line.",
        },
        "results": results,
        "confirmation_original_labels": confirmation_labels.tolist(),
        "confirmation_remapped_labels": remapped_labels.tolist(),
    }
    (output / "layerwise_same_layer_match.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-original", type=Path, required=True)
    parser.add_argument("--confirmation-original", type=Path, required=True)
    parser.add_argument("--confirmation-remapped", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--remapped-baseline-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    analyze(
        args.discovery_original,
        args.confirmation_original,
        args.confirmation_remapped,
        args.baseline_results,
        args.remapped_baseline_results,
        args.output,
        args.seed,
    )


if __name__ == "__main__":
    main()
