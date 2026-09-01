from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LETTERS = "ABCD"
LAMBDAS = np.asarray([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])


def _labels(path: Path, qids: list[str], field: str | None = None) -> np.ndarray:
    rows = json.loads(path.read_text())["results"]
    values = []
    for qid in qids:
        row = rows[qid]
        value = row[field] if field else row.get("answer", row.get("subject_answer"))
        if value not in LETTERS:
            raise ValueError(f"Non-A-D label for {qid}: {value!r}")
        values.append(LETTERS.index(value))
    return np.asarray(values, dtype=np.int64)


def _load_semantic_cache(root: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    metadata = json.loads((root / "metadata.json").read_text())
    qids = list(metadata["question_ids"])
    anchors = list(metadata["anchors"])
    embeddings = np.load(root / "option_embeddings.npy", mmap_mode="r")
    residual_path = root / "position_residuals.npy"
    residuals = np.load(residual_path, mmap_mode="r") if residual_path.exists() else None
    return residuals, embeddings, qids, anchors


def _balanced_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean([
        np.mean(prediction[target == label] == label)
        for label in range(4) if np.any(target == label)
    ]))


def _accuracy_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    draws: int = 10_000,
) -> dict:
    correct = prediction == target
    groups = [np.flatnonzero(target == label) for label in range(4) if np.any(target == label)]
    ordinary = np.empty(draws, dtype=float)
    balanced = np.empty(draws, dtype=float)
    all_indices = np.arange(len(target))
    for draw in range(draws):
        sample = rng.choice(all_indices, size=len(all_indices), replace=True)
        ordinary[draw] = correct[sample].mean()
        balanced[draw] = np.mean([
            correct[rng.choice(group, size=len(group), replace=True)].mean()
            for group in groups
        ])
    return {
        "n": int(len(target)),
        "accuracy": float(correct.mean()),
        "accuracy_ci": np.quantile(ordinary, (0.025, 0.975)).tolist(),
        "balanced_accuracy": _balanced_accuracy(prediction, target),
        "balanced_accuracy_ci": np.quantile(balanced, (0.025, 0.975)).tolist(),
        "per_letter_recall": {
            LETTERS[label]: float(correct[target == label].mean())
            for label in range(4) if np.any(target == label)
        },
        "target_counts": {
            LETTERS[label]: int(np.sum(target == label)) for label in range(4)
        },
    }


def _folds(labels: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    values: list[list[int]] = [[] for _ in range(count)]
    for label in range(4):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            values[offset % count].append(int(index))
    return [np.asarray(sorted(value), dtype=np.int64) for value in values]


def _normalize_queries(values: np.ndarray, mean: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float32) - mean.astype(np.float32)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(norm, 1e-12)


def _ridge_scores(
    train_values: np.ndarray,
    train_targets: np.ndarray,
    test_values: np.ndarray,
    test_candidates: np.ndarray,
    penalty: float,
) -> np.ndarray:
    mean = train_values.astype(np.float32).mean(axis=0)
    x_train = _normalize_queries(train_values, mean)
    x_test = _normalize_queries(test_values, mean)
    gram = x_train @ x_train.T
    cross = x_test @ x_train.T
    semantic = train_targets.astype(np.float32) @ (
        test_candidates.astype(np.float32).reshape(-1, test_candidates.shape[-1]).T
    )
    weights = np.linalg.solve(
        gram + float(penalty) * np.eye(len(gram), dtype=np.float32), semantic
    )
    full = (cross @ weights).reshape(len(test_values), len(test_values), 4)
    indices = np.arange(len(test_values))
    return full[indices, indices]


def _cross_validated_scores(
    values: np.ndarray,
    candidates: np.ndarray,
    labels: np.ndarray,
    folds: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    target_vectors = candidates[np.arange(len(labels)), labels].astype(np.float32)
    predictions = np.empty((len(LAMBDAS), len(labels), 4), dtype=np.float32)
    all_indices = np.arange(len(labels))
    for validation in folds:
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        mean = values[training].astype(np.float32).mean(axis=0)
        x_train = _normalize_queries(values[training], mean)
        x_validation = _normalize_queries(values[validation], mean)
        gram = x_train @ x_train.T
        cross = x_validation @ x_train.T
        semantic = target_vectors[training] @ (
            candidates[validation].astype(np.float32).reshape(-1, candidates.shape[-1]).T
        )
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        projected = eigenvectors.T @ semantic
        for li, penalty in enumerate(LAMBDAS):
            weights = eigenvectors @ (projected / (eigenvalues[:, None] + penalty))
            full = (cross @ weights).reshape(len(validation), len(validation), 4)
            indices = np.arange(len(validation))
            predictions[li, validation] = full[indices, indices]
    balanced = np.asarray([
        _balanced_accuracy(scores.argmax(axis=1), labels) for scores in predictions
    ])
    return predictions, balanced


def _direct_scores(
    discovery_values: np.ndarray,
    values: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    mean = discovery_values.astype(np.float32).mean(axis=0)
    query = _normalize_queries(values, mean)
    return np.einsum("nd,ncd->nc", query, candidates.astype(np.float32))


def _bootstrap_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    seed: int,
    draws: int = 2_000,
) -> tuple[np.ndarray, np.ndarray]:
    # predictions: anchors, layers, questions, candidates
    correct = predictions.argmax(axis=-1) == labels[None, None, :]
    groups = [np.flatnonzero(labels == label) for label in range(4) if np.any(labels == label)]
    rng = np.random.default_rng(seed)
    samples = np.empty((draws,) + correct.shape[:2], dtype=np.float32)
    for draw in range(draws):
        samples[draw] = np.mean([
            correct[..., rng.choice(group, size=len(group), replace=True)].mean(axis=-1)
            for group in groups
        ], axis=0)
    return np.quantile(samples, 0.025, axis=0), np.quantile(samples, 0.975, axis=0)


def analyze(
    discovery_root: Path,
    confirmation_root: Path,
    remapped_confirmation_root: Path,
    baseline_results: Path,
    remapped_baseline_results: Path,
    output: Path,
    seed: int,
) -> dict:
    discovery_residuals, discovery_candidates, discovery_qids, discovery_anchors = (
        _load_semantic_cache(discovery_root)
    )
    confirmation_residuals, confirmation_candidates, confirmation_qids, confirmation_anchors = (
        _load_semantic_cache(confirmation_root)
    )
    remapped_residuals, remapped_candidates, remapped_qids, remapped_anchors = (
        _load_semantic_cache(remapped_confirmation_root)
    )
    anchors = ["first_question_end", "first_user_end", "first_answer_decision"]
    if confirmation_qids != remapped_qids:
        raise ValueError("Original and remapped confirmation question orders differ")
    if confirmation_anchors != anchors or remapped_anchors != anchors:
        raise ValueError("Unexpected confirmation anchors")
    discovery_anchor_indices = [discovery_anchors.index(anchor) for anchor in anchors]

    discovery_labels = _labels(baseline_results, discovery_qids)
    confirmation_labels = _labels(baseline_results, confirmation_qids)
    remapped_labels = _labels(remapped_baseline_results, remapped_qids, "answer_new_letter")
    discovery_target_vectors = np.asarray(discovery_candidates)[
        np.arange(len(discovery_qids)), discovery_labels
    ].astype(np.float32)
    folds = _folds(discovery_labels, 5, seed)

    cv_balanced = np.empty((len(anchors), 64), dtype=np.float32)
    direct_balanced = np.empty_like(cv_balanced)
    selected_lambda = np.empty((len(anchors), 64), dtype=np.float32)
    cv_predictions: dict[tuple[int, int], np.ndarray] = {}

    for ai, discovery_anchor in enumerate(discovery_anchor_indices):
        for layer in range(64):
            values = np.asarray(discovery_residuals[:, layer, discovery_anchor]).copy()
            predictions, balanced = _cross_validated_scores(
                values, np.asarray(discovery_candidates), discovery_labels, folds
            )
            best = int(np.argmax(balanced))
            cv_balanced[ai, layer] = balanced[best]
            selected_lambda[ai, layer] = LAMBDAS[best]
            cv_predictions[(ai, layer)] = predictions[best]
            direct = _direct_scores(values, values, np.asarray(discovery_candidates))
            direct_balanced[ai, layer] = _balanced_accuracy(
                direct.argmax(axis=1), discovery_labels
            )
            print(
                f"semantic matcher: {anchors[ai]} layer {layer + 1}/64 "
                f"CV={cv_balanced[ai, layer]:.3f}", flush=True
            )

    flat_best = int(np.argmax(cv_balanced))
    best_anchor, best_layer = np.unravel_index(flat_best, cv_balanced.shape)
    best_penalty = float(selected_lambda[best_anchor, best_layer])

    original_predictions = np.empty(
        (len(anchors), 64, len(confirmation_qids), 4), dtype=np.float32
    )
    remapped_predictions = np.empty_like(original_predictions)
    confirmation_balanced = np.empty((len(anchors), 64), dtype=np.float32)
    remapped_balanced = np.empty_like(confirmation_balanced)
    for ai, discovery_anchor in enumerate(discovery_anchor_indices):
        for layer in range(64):
            train_values = np.asarray(
                discovery_residuals[:, layer, discovery_anchor]
            ).copy()
            penalty = float(selected_lambda[ai, layer])
            original_predictions[ai, layer] = _ridge_scores(
                train_values,
                discovery_target_vectors,
                np.asarray(confirmation_residuals[:, layer, ai]).copy(),
                np.asarray(confirmation_candidates),
                penalty,
            )
            remapped_predictions[ai, layer] = _ridge_scores(
                train_values,
                discovery_target_vectors,
                np.asarray(remapped_residuals[:, layer, ai]).copy(),
                np.asarray(remapped_candidates),
                penalty,
            )
            confirmation_balanced[ai, layer] = _balanced_accuracy(
                original_predictions[ai, layer].argmax(axis=1), confirmation_labels
            )
            remapped_balanced[ai, layer] = _balanced_accuracy(
                remapped_predictions[ai, layer].argmax(axis=1), remapped_labels
            )

    rng = np.random.default_rng(seed + 1)
    primary_original_prediction = original_predictions[best_anchor, best_layer].argmax(axis=1)
    primary_remapped_prediction = remapped_predictions[best_anchor, best_layer].argmax(axis=1)
    original_summary = _accuracy_summary(
        primary_original_prediction, confirmation_labels, rng
    )
    remapped_summary = _accuracy_summary(
        primary_remapped_prediction, remapped_labels, rng
    )

    permuted = np.empty(2_000, dtype=float)
    primary_scores = remapped_predictions[best_anchor, best_layer]
    for draw in range(len(permuted)):
        shuffled = np.stack([rng.permutation(4) for _ in range(len(remapped_labels))])
        permuted_prediction = np.take_along_axis(primary_scores, shuffled, axis=1).argmax(axis=1)
        permuted[draw] = _balanced_accuracy(permuted_prediction, remapped_labels)

    original_low, original_high = _bootstrap_curve(
        original_predictions, confirmation_labels, seed + 2
    )
    remapped_low, remapped_high = _bootstrap_curve(
        remapped_predictions, remapped_labels, seed + 3
    )

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for ai, anchor in enumerate(anchors):
        for layer in range(64):
            rows.append({
                "anchor": anchor,
                "layer": layer + 1,
                "discovery_direct_balanced_accuracy": float(direct_balanced[ai, layer]),
                "discovery_cv_balanced_accuracy": float(cv_balanced[ai, layer]),
                "selected_lambda": float(selected_lambda[ai, layer]),
                "confirmation_original_balanced_accuracy": float(confirmation_balanced[ai, layer]),
                "confirmation_original_ci_low": float(original_low[ai, layer]),
                "confirmation_original_ci_high": float(original_high[ai, layer]),
                "confirmation_remapped_balanced_accuracy": float(remapped_balanced[ai, layer]),
                "confirmation_remapped_ci_low": float(remapped_low[ai, layer]),
                "confirmation_remapped_ci_high": float(remapped_high[ai, layer]),
            })
    with (output / "layerwise_accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "definitions": {
            "semantic_target": (
                "Normalized mean Qwen input embedding of the selected option's "
                "content-bearing tokens; A-D label excluded."
            ),
            "matcher": (
                "Kernel ridge map from first-presentation residual to selected "
                "content embedding; candidate chosen among the same question's four contents."
            ),
        },
        "split": {
            "discovery": len(discovery_qids),
            "confirmation": len(confirmation_qids),
        },
        "selected_on_discovery": {
            "anchor": anchors[best_anchor],
            "layer": int(best_layer + 1),
            "lambda": best_penalty,
            "cross_validated_balanced_accuracy": float(cv_balanced[best_anchor, best_layer]),
            "direct_similarity_balanced_accuracy": float(direct_balanced[best_anchor, best_layer]),
        },
        "confirmation_original_mapping": original_summary,
        "confirmation_remapped_mapping": remapped_summary,
        "remapped_candidate_permutation_control": {
            "mean_balanced_accuracy": float(permuted.mean()),
            "ci": np.quantile(permuted, (0.025, 0.975)).tolist(),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    try:
        import matplotlib.pyplot as plt

        layers = np.arange(1, 65)
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
        for ai, (axis, anchor) in enumerate(zip(axes, anchors)):
            axis.plot(layers, cv_balanced[ai], color="#777777", linewidth=1.5,
                      label="Discovery cross-validation")
            axis.plot(layers, confirmation_balanced[ai], color="#2f8ef5", linewidth=2,
                      label="Confirmation: original mapping")
            axis.fill_between(layers, original_low[ai], original_high[ai],
                              color="#2f8ef5", alpha=0.15)
            axis.plot(layers, remapped_balanced[ai], color="#ef7d32", linewidth=2,
                      label="Confirmation: remapped mapping")
            axis.fill_between(layers, remapped_low[ai], remapped_high[ai],
                              color="#ef7d32", alpha=0.15)
            axis.axhline(0.25, color="#999999", linestyle="--", linewidth=1)
            if ai == best_anchor:
                axis.axvline(
                    best_layer + 1, color="#222222", linestyle=":", linewidth=1.2
                )
            axis.set_title(anchor.replace("_", " ").title())
            axis.set_xlabel("Residual readout")
            axis.set_xlim(1, 64)
            axis.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("Balanced four-way semantic accuracy")
        axes[0].legend(frameon=False, fontsize=9, loc="upper left")
        fig.suptitle("First-presentation semantic answer matching", fontsize=15)
        fig.tight_layout()
        fig.savefig(output / "semantic_matching_accuracy.png", dpi=180)
        plt.close(fig)
    except ModuleNotFoundError:
        print("matplotlib is unavailable; wrote numerical results without a figure", flush=True)

    def pct(row: dict, key: str) -> str:
        low, high = row[f"{key}_ci"]
        return f"{row[key]:.1%} [{low:.1%}, {high:.1%}]"

    report = f"""# First-presentation semantic-answer matching

## Method

The target for each option is the normalized mean of Qwen's own input
embeddings for the content-bearing tokens in that option. No A--D label is
included. A regularized kernel-ridge map learns, on 251 discovery questions,
to map the residual at a first-presentation position to the embedding of the
content selected by Baseline. The predicted vector is compared with all four
option-content embeddings from the same held-out question.

## Frozen discovery selection

- Position: **{anchors[best_anchor]}**
- Layer: **{best_layer + 1}**
- Ridge penalty: **{best_penalty:g}**
- Five-fold discovery balanced accuracy: **{cv_balanced[best_anchor, best_layer]:.1%}**
- Zero-fit direct-similarity accuracy at that point: **{direct_balanced[best_anchor, best_layer]:.1%}**

## Held-out confirmation

| Evaluation | Ordinary accuracy | Balanced accuracy |
|---|---:|---:|
| Original mapping, W1 target | {pct(original_summary, 'accuracy')} | {pct(original_summary, 'balanced_accuracy')} |
| Remapped mapping, fresh W2 target | {pct(remapped_summary, 'accuracy')} | {pct(remapped_summary, 'balanced_accuracy')} |

Chance is 25%. The remapped candidate-permutation control was
{summary['remapped_candidate_permutation_control']['mean_balanced_accuracy']:.1%}
[{summary['remapped_candidate_permutation_control']['ci'][0]:.1%},
{summary['remapped_candidate_permutation_control']['ci'][1]:.1%}].

Successful remapped confirmation means the readout follows the freshly selected
answer content after it moves to a new A--D label. It cannot be explained by
including the answer label in the target representation, because labels were
excluded before fitting.

![Layerwise semantic matching](semantic_matching_accuracy.png)
"""
    (output / "REPORT.md").write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--remapped-confirmation", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--remapped-baseline-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    result = analyze(
        args.discovery,
        args.confirmation,
        args.remapped_confirmation,
        args.baseline_results,
        args.remapped_baseline_results,
        args.output,
        args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
