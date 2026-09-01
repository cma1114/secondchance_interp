from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .all_trial_figures import CONDITION_COLORS, CONDITION_LABELS, CONDITION_STYLES, _style
from .answer_emergence_figures import Z_975, macro_mean_and_se
from .data import load_activation_dataset
from .io import shard_path
from .probes import stratified_folds


CONDITIONS = ("baseline", "incorrect", "neutral")


def _labels(data, condition: str) -> np.ndarray:
    values = []
    for qid in data.question_ids:
        answer = data.metadata[(qid, condition)]["full_vocab_top_token"].strip()
        if answer not in "ABCD":
            raise ValueError(f"Non-A-D answer for {condition}/{qid}: {answer!r}")
        values.append("ABCD".index(answer))
    return np.asarray(values, dtype=np.int64)


def _load_residuals(root: str | Path, conditions: tuple[str, ...], qids: list[str]) -> np.ndarray:
    with np.load(shard_path(root, conditions[0], qids[0]), allow_pickle=False) as shard:
        shape = shard["residuals"].shape
    values = np.empty((len(conditions), len(qids), *shape), dtype=np.float16)
    for ci, condition in enumerate(conditions):
        for qi, qid in enumerate(qids):
            with np.load(shard_path(root, condition, qid), allow_pickle=False) as shard:
                values[ci, qi] = shard["residuals"]
    return values


def _dataset_folds(simple_labels: np.ndarray, trivia_labels: np.ndarray, k: int, seed: int):
    simple = stratified_folds(simple_labels, k, seed)
    trivia = stratified_folds(trivia_labels, k, seed + 1)
    return list(zip(simple, trivia))


def pooled_centroid_scores(
    simple: np.ndarray,
    trivia: np.ndarray,
    simple_labels: np.ndarray,
    trivia_labels: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted, dataset-centered centroid scores.

    The common letter centroids are the equal-weight mean of within-dataset
    letter centroids. Each held-out question is scored by a probe trained
    without that question. Only Baseline residuals are used for fitting.
    """
    n_conditions, n_simple, n_layers, _ = simple.shape
    n_trivia = len(trivia_labels)
    simple_scores = np.empty((n_conditions, n_simple, n_layers, 4), dtype=np.float32)
    trivia_scores = np.empty((n_trivia, n_layers, 4), dtype=np.float32)
    simple_all = np.arange(n_simple)
    trivia_all = np.arange(n_trivia)

    for li in range(n_layers):
        for simple_test, trivia_test in _dataset_folds(simple_labels, trivia_labels, folds, seed):
            simple_train = np.setdiff1d(simple_all, simple_test, assume_unique=True)
            trivia_train = np.setdiff1d(trivia_all, trivia_test, assume_unique=True)
            xs = simple[0, simple_train, li].astype(np.float32)
            xt = trivia[trivia_train, li].astype(np.float32)
            mean_s = xs.mean(axis=0)
            mean_t = xt.mean(axis=0)
            centered_s = xs - mean_s
            centered_t = xt - mean_t
            scale = np.concatenate((centered_s, centered_t), axis=0).std(axis=0)
            scale[scale < 1e-6] = 1.0
            zs = centered_s / scale
            zt = centered_t / scale
            centers = np.stack([
                0.5 * (
                    zs[simple_labels[simple_train] == letter].mean(axis=0)
                    + zt[trivia_labels[trivia_train] == letter].mean(axis=0)
                )
                for letter in range(4)
            ])
            for ci in range(n_conditions):
                z = (simple[ci, simple_test, li].astype(np.float32) - mean_s) / scale
                simple_scores[ci, simple_test, li] = z @ centers.T
            z = (trivia[trivia_test, li].astype(np.float32) - mean_t) / scale
            trivia_scores[trivia_test, li] = z @ centers.T

        simple_scores[:, :, li] -= simple_scores[:, :, li].mean(axis=-1, keepdims=True)
        trivia_scores[:, li] -= trivia_scores[:, li].mean(axis=-1, keepdims=True)
        pooled_baseline = np.concatenate(
            (simple_scores[0, :, li].reshape(-1), trivia_scores[:, li].reshape(-1))
        )
        dispersion = pooled_baseline.std(ddof=1)
        if dispersion > 1e-8:
            simple_scores[:, :, li] /= dispersion
            trivia_scores[:, li] /= dispersion
        else:
            simple_scores[:, :, li] = 0.0
            trivia_scores[:, li] = 0.0
    return simple_scores, trivia_scores


def _balanced_accuracy(prediction: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean([np.mean(prediction[labels == letter] == letter) for letter in range(4)]))


def _summary(values: np.ndarray, strata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, se = macro_mean_and_se(values, strata)
    return mean, Z_975 * se


def analyze(
    simple_root: str | Path,
    trivia_root: str | Path,
    output_root: str | Path,
    folds: int,
    seed: int,
) -> dict:
    simple_data = load_activation_dataset(simple_root, list(CONDITIONS))
    trivia_data = load_activation_dataset(trivia_root, ["baseline"])
    simple_labels = _labels(simple_data, "baseline")
    trivia_labels = _labels(trivia_data, "baseline")
    simple_residuals = _load_residuals(simple_root, CONDITIONS, simple_data.question_ids)
    trivia_residuals = _load_residuals(trivia_root, ("baseline",), trivia_data.question_ids)[0]
    simple_scores, trivia_scores = pooled_centroid_scores(
        simple_residuals, trivia_residuals, simple_labels, trivia_labels, folds, seed
    )
    del simple_residuals, trivia_residuals

    output = Path(output_root)
    figure_dir = output / "preserved_figures"
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    layers = np.arange(simple_scores.shape[2])
    np.savez_compressed(
        output / "pooled_cross_fitted_probe_scores.npz",
        simple_scores=simple_scores,
        trivia_scores=trivia_scores,
        layers=layers,
        simple_question_ids=np.asarray(simple_data.question_ids),
        trivia_question_ids=np.asarray(trivia_data.question_ids),
        conditions=np.asarray(CONDITIONS),
        simple_baseline_labels=simple_labels,
        trivia_baseline_labels=trivia_labels,
    )

    accuracy_rows = []
    simple_generated = {condition: _labels(simple_data, condition) for condition in CONDITIONS}
    for dataset, scores, labels_by_condition in (
        ("SimpleMC", simple_scores, simple_generated),
        ("TriviaMC", trivia_scores[None], {"baseline": trivia_labels}),
    ):
        for ci, condition in enumerate(CONDITIONS if dataset == "SimpleMC" else ("baseline",)):
            labels = labels_by_condition[condition]
            for li, layer in enumerate(layers):
                prediction = scores[ci, :, li].argmax(axis=-1)
                accuracy_rows.append({
                    "dataset": dataset,
                    "condition": condition,
                    "layer": int(layer),
                    "accuracy": float(np.mean(prediction == labels)),
                    "balanced_accuracy": _balanced_accuracy(prediction, labels),
                })
    with (output / "pooled_probe_accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(accuracy_rows[0]))
        writer.writeheader()
        writer.writerows(accuracy_rows)

    baseline_logits = simple_data.logits[:, 0, -1].copy()
    prior_answer = simple_labels
    baseline_logits[np.arange(len(prior_answer)), prior_answer] = -np.inf
    prior_runner = baseline_logits.argmax(axis=-1)
    row = np.arange(len(prior_answer))
    trajectory_rows = []
    summaries = {}
    for ci, condition in enumerate(CONDITIONS):
        scores = simple_scores[ci]
        competitors = scores.copy()
        competitors[row, :, prior_answer] = -np.inf
        winner_margin = scores[row, :, prior_answer] - competitors.max(axis=-1)
        competitors = scores.copy()
        competitors[row, :, prior_runner] = -np.inf
        runner_margin = scores[row, :, prior_runner] - competitors.max(axis=-1)
        spread = scores.std(axis=-1)
        for metric, values in (
            ("prior_answer_margin", winner_margin),
            ("prior_runner_margin", runner_margin),
            ("ad_spread", spread),
        ):
            mean, half = _summary(values, prior_answer)
            summaries[(condition, metric)] = mean, half
            for layer, value, width in zip(layers, mean, half):
                trajectory_rows.append({
                    "condition": condition,
                    "metric": metric,
                    "layer": int(layer),
                    "mean": float(value),
                    "ci_low": float(value - width),
                    "ci_high": float(value + width),
                })
    with (output / "pooled_probe_trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    _style()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharex=True)
    baseline_axes = axes[0, 0]
    for dataset, scores, labels, color in (
        ("SimpleMC", simple_scores[0], simple_labels, "#0072B2"),
        ("TriviaMC", trivia_scores, trivia_labels, "#D55E00"),
    ):
        values = np.asarray([
            _balanced_accuracy(scores[:, li].argmax(axis=-1), labels) for li in range(len(layers))
        ])
        baseline_axes.plot(layers, values, color=color, linewidth=1.6, label=dataset)
    baseline_axes.axhline(0.25, color="#555555", linewidth=0.7, linestyle=(0, (3, 2)))
    baseline_axes.set_ylim(0, 1)
    baseline_axes.set_ylabel("Letter-balanced held-out accuracy")
    baseline_axes.set_title("A  Baseline probe reliability", loc="left", fontweight="bold")
    baseline_axes.legend(frameon=False, loc="upper left")

    panels = (
        (axes[0, 1], "prior_answer_margin", "B  Prior-answer advantage", "Score minus strongest competitor"),
        (axes[1, 0], "prior_runner_margin", "C  Prior-runner advantage", "Score minus strongest competitor"),
        (axes[1, 1], "ad_spread", "D  Total A-D spread", "Within-question score SD"),
    )
    for axis, metric, title, ylabel in panels:
        for condition in CONDITIONS:
            mean, half = summaries[(condition, metric)]
            color = CONDITION_COLORS[condition]
            axis.fill_between(layers, mean - half, mean + half, color=color, alpha=0.14, linewidth=0)
            axis.plot(
                layers, mean, color=color, linestyle=CONDITION_STYLES[condition],
                linewidth=1.55, label=CONDITION_LABELS[condition],
            )
        if metric != "ad_spread":
            axis.axhline(0, color="#555555", linewidth=0.7)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(f"{ylabel}\n(pooled Baseline-probe SD units)")
    axes[0, 1].legend(frameon=False, loc="upper left")
    for axis in axes.flat:
        axis.set_xlim(0, int(layers[-1]))
        axis.set_xticks(np.arange(0, int(layers[-1]) + 1, 8))
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[1]:
        axis.set_xlabel("Residual readout (0 = embedding; 64 = final block)")
    figure.suptitle(
        "Qwen3.6-27B SimpleMC: cross-fitted probe pooled across datasets",
        fontsize=10.5, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96), w_pad=1.5, h_pad=1.5)
    figure.savefig(figure_dir / "pooled_probe_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "probe": "cross-fitted dataset-centered equal-dataset-weight centroid",
        "folds": folds,
        "training_questions_per_fold_approx": int((len(simple_labels) + len(trivia_labels)) * (folds - 1) / folds),
        "simple_n": len(simple_labels),
        "trivia_n": len(trivia_labels),
        "final_layer": {
            "simple_baseline_balanced_accuracy": _balanced_accuracy(simple_scores[0, :, -1].argmax(-1), simple_labels),
            "trivia_baseline_balanced_accuracy": _balanced_accuracy(trivia_scores[:, -1].argmax(-1), trivia_labels),
            **{
                f"simple_{condition}_eventual_output_balanced_accuracy": _balanced_accuracy(
                    simple_scores[ci, :, -1].argmax(-1), simple_generated[condition]
                )
                for ci, condition in enumerate(CONDITIONS)
            },
        },
    }
    (output / "pooled_probe_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-fitted probe pooled over two Baseline datasets")
    parser.add_argument("--simple", required=True)
    parser.add_argument("--trivia", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(analyze(args.simple, args.trivia, args.output, args.folds, args.seed), indent=2))


if __name__ == "__main__":
    main()
