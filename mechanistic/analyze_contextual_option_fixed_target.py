from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mechanistic.analyze_contextual_option_representations import (
    LAMBDAS,
    _bootstrap_accuracy,
    _candidate_array,
    _center_normalize,
    _cv_scores,
    _fit_scores,
    _folds,
    _labels,
    _load,
)


def analyze(
    discovery_original_root: Path,
    confirmation_original_root: Path,
    confirmation_remapped_root: Path,
    baseline_results: Path,
    remapped_baseline_results: Path,
    output: Path,
    anchor: str,
    option_layer: int,
    seed: int,
) -> dict:
    discovery, discovery_meta = _load(discovery_original_root)
    confirmation, confirmation_meta = _load(confirmation_original_root)
    confirmation_remapped, confirmation_remapped_meta = _load(
        confirmation_remapped_root
    )
    discovery_qids = list(discovery_meta["question_ids"])
    confirmation_qids = list(confirmation_meta["question_ids"])

    target_layer = option_layer - 1
    discovery_candidates = _center_normalize(
        _candidate_array(discovery, discovery_meta, anchor)[:, target_layer]
    )
    confirmation_candidates = _center_normalize(
        _candidate_array(confirmation, confirmation_meta, anchor)[:, target_layer]
    )
    confirmation_remapped_candidates = _center_normalize(
        _candidate_array(
            confirmation_remapped, confirmation_remapped_meta, anchor
        )[:, target_layer]
    )

    discovery_decision_index = discovery_meta["anchors"].index(
        "first_answer_decision"
    )
    confirmation_decision_index = confirmation_meta["anchors"].index(
        "first_answer_decision"
    )
    remapped_decision_index = confirmation_remapped_meta["anchors"].index(
        "first_answer_decision"
    )
    discovery_queries = np.asarray(
        discovery[:, :, discovery_decision_index]
    ).copy()
    confirmation_queries = np.asarray(
        confirmation[:, :, confirmation_decision_index]
    ).copy()
    remapped_queries = np.asarray(
        confirmation_remapped[:, :, remapped_decision_index]
    ).copy()

    discovery_labels = _labels(baseline_results, discovery_qids, "answer")
    confirmation_labels = _labels(baseline_results, confirmation_qids, "answer")
    remapped_labels = _labels(
        remapped_baseline_results, confirmation_qids, "answer_new_letter"
    )
    folds = _folds(discovery_labels, 5, seed)

    cv_balanced = np.empty(64, dtype=np.float32)
    penalties = np.empty(64, dtype=np.float32)
    for layer in range(64):
        _, balanced = _cv_scores(
            discovery_queries[:, layer],
            discovery_candidates,
            discovery_labels,
            folds,
        )
        best = int(np.argmax(balanced))
        cv_balanced[layer] = balanced[best]
        penalties[layer] = LAMBDAS[best]
        print(
            f"fixed target {anchor} L{option_layer}; "
            f"decision layer {layer + 1}/64 CV={cv_balanced[layer]:.3f}",
            flush=True,
        )

    decision_layer = int(np.argmax(cv_balanced))
    penalty = float(penalties[decision_layer])
    original_scores = _fit_scores(
        discovery_queries[:, decision_layer],
        discovery_candidates,
        discovery_labels,
        confirmation_queries[:, decision_layer],
        confirmation_candidates,
        penalty,
    )
    remapped_scores = _fit_scores(
        discovery_queries[:, decision_layer],
        discovery_candidates,
        discovery_labels,
        remapped_queries[:, decision_layer],
        confirmation_remapped_candidates,
        penalty,
    )
    original_accuracy, original_ci = _bootstrap_accuracy(
        original_scores, confirmation_labels, seed + 1
    )
    remapped_accuracy, remapped_ci = _bootstrap_accuracy(
        remapped_scores, remapped_labels, seed + 2
    )
    summary = {
        "definition": (
            "Kernel-ridge matching from the first-answer decision residual to "
            "the contextual residual at each option's selected target position."
        ),
        "target": {"anchor": anchor, "layer": option_layer},
        "selected_decision_matcher": {
            "layer": decision_layer + 1,
            "lambda": penalty,
            "discovery_cv_balanced_accuracy": float(cv_balanced[decision_layer]),
            "confirmation_original_balanced_accuracy": original_accuracy,
            "confirmation_original_ci": original_ci,
            "confirmation_remapped_balanced_accuracy": remapped_accuracy,
            "confirmation_remapped_ci": remapped_ci,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    stem = f"decision_matcher_{anchor}_l{option_layer}"
    (output / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        output / f"{stem}.npz",
        discovery_cv_balanced_accuracy=cv_balanced,
        selected_lambda=penalties,
        confirmation_original_scores=original_scores,
        confirmation_original_labels=confirmation_labels,
        confirmation_remapped_scores=remapped_scores,
        confirmation_remapped_labels=remapped_labels,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-original", type=Path, required=True)
    parser.add_argument("--confirmation-original", type=Path, required=True)
    parser.add_argument("--confirmation-remapped", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--remapped-baseline-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor", choices=("content_end", "line_end"), required=True)
    parser.add_argument("--option-layer", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    if not 1 <= args.option_layer <= 64:
        parser.error("--option-layer must be between 1 and 64")
    analyze(
        args.discovery_original,
        args.confirmation_original,
        args.confirmation_remapped,
        args.baseline_results,
        args.remapped_baseline_results,
        args.output,
        args.anchor,
        args.option_layer,
        args.seed,
    )


if __name__ == "__main__":
    main()
