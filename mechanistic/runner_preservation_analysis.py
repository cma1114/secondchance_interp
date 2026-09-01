from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .cumulative_hypothesis_analysis import _design, _question_weights, _weighted_ridge
from .data import load_activation_dataset
from .perturbation_analysis import _cross_fitted_compression_residuals, _style
from .probes import stratified_folds
from .trajectory_analysis import centered


CONDITIONS = ("incorrect", "neutral")
CONDITION_LABELS = {"incorrect": "Second Chance", "neutral": "Neutral"}
CONDITION_COLORS = {"incorrect": "#0072B2", "neutral": "#D55E00"}
METRICS = ("raw_change", "standard_residual", "leave_runner_out")
METRIC_TITLES = {
    "raw_change": "A  Raw runner-minus-lower change",
    "standard_residual": "B  Residual after standard compression fit",
    "leave_runner_out": "C  Residual after leave-runner-out fit",
}


def _cross_fitted_leave_runner_out(
    baseline: np.ndarray,
    target: np.ndarray,
    winner: np.ndarray,
    runner: np.ndarray,
    folds: list[np.ndarray],
) -> np.ndarray:
    """Fit letter effects and proportional compression without runner observations."""
    n, n_layers, _ = baseline.shape
    residuals = np.empty_like(target)
    weights = _question_weights(winner)
    all_ids = np.arange(n)
    for layer in range(n_layers):
        baseline_layer = baseline[:, layer]
        leader = np.argmax(baseline_layer, axis=-1)
        sorted_baseline = np.sort(baseline_layer, axis=-1)
        margin = sorted_baseline[:, -1] - sorted_baseline[:, -2]
        for test in folds:
            train = np.setdiff1d(all_ids, test)
            x_train, _ = _design(
                "compression", baseline_layer[train], winner[train], leader[train], margin[train], 0.0
            )
            x_test, _ = _design(
                "compression", baseline_layer[test], winner[test], leader[test], margin[test], 0.0
            )
            keep = np.ones((len(train), 4), dtype=bool)
            keep[np.arange(len(train)), runner[train]] = False
            prediction, _ = _weighted_ridge(
                x_train[keep.reshape(-1)],
                target[train, layer].reshape(-1)[keep.reshape(-1)],
                np.repeat(weights[train], 4)[keep.reshape(-1)],
                x_test,
            )
            residuals[test, layer] = target[test, layer] - prediction.reshape(len(test), 4)
    residuals -= residuals.mean(axis=-1, keepdims=True)
    return residuals


def _rank_contrast(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    aligned = np.take_along_axis(values, order[:, None, :], axis=-1)
    return aligned[:, :, 1] - aligned[:, :, 2:].mean(axis=-1)


def _bootstrap(values: np.ndarray, repetitions: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mean = values.mean(axis=0)
    draws = np.empty((repetitions, values.shape[1]))
    for repetition in range(repetitions):
        sample = rng.integers(0, len(values), len(values))
        draws[repetition] = values[sample].mean(axis=0)
    return mean, np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def _plot(rows: list[dict], output: Path, final_layer: int, n: int) -> None:
    import matplotlib.pyplot as plt

    _style()
    lookup = {(row["metric"], row["condition"]): row for row in rows}
    figure, axes = plt.subplots(1, 3, figsize=(9.4, 2.8), sharex=True)
    layers = np.arange(final_layer + 1)
    for axis, metric in zip(axes, METRICS):
        for condition in CONDITIONS:
            row = lookup[(metric, condition)]
            mean = np.asarray(row["mean"])
            low = np.asarray(row["ci_low"])
            high = np.asarray(row["ci_high"])
            axis.plot(layers, mean, color=CONDITION_COLORS[condition], lw=1.5, label=CONDITION_LABELS[condition])
            axis.fill_between(layers, low, high, color=CONDITION_COLORS[condition], alpha=0.13, linewidth=0)
        axis.axhline(0, color="#555555", lw=0.65)
        axis.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlim(0, final_layer)
        axis.set_title(METRIC_TITLES[metric], loc="left", fontweight="bold")
        axis.set_xlabel("Residual readout")
        axis.text(0.02, 0.95, f"both keep: n={n}", transform=axis.transAxes, va="top", fontsize=7.5)
    axes[0].set_ylabel("Runner-up minus ranks 3-4 mean\n(natural-logit units)")
    axes[0].legend(frameon=False, loc="lower left")
    figure.tight_layout(w_pad=1.25)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"runner_preservation_robustness.{suffix}", bbox_inches="tight")
    plt.close(figure)


def analyze(
    input_dir: str | Path,
    output_dir: str | Path,
    folds: int,
    seed: int,
    bootstrap: int,
) -> dict:
    data = load_activation_dataset(input_dir, ["baseline", *CONDITIONS])
    logits = centered(data.logits)
    baseline = logits[:, 0]
    order = np.argsort(-baseline[:, -1], axis=-1)
    winner = order[:, 0]
    runner = order[:, 1]
    game_choice = np.argmax(logits[:, 1, -1], axis=-1)
    neutral_choice = np.argmax(logits[:, 2, -1], axis=-1)
    both_keep = (game_choice == winner) & (neutral_choice == winner)
    split = stratified_folds(winner, folds, seed)

    per_condition: dict[str, dict[str, np.ndarray]] = {}
    baseline_contrast = _rank_contrast(baseline, order)
    for condition_index, condition in enumerate(CONDITIONS, start=1):
        target = logits[:, condition_index] - baseline
        standard, _ = _cross_fitted_compression_residuals(baseline, target, winner, split)
        leave_runner = _cross_fitted_leave_runner_out(baseline, target, winner, runner, split)
        per_condition[condition] = {
            "raw_change": _rank_contrast(logits[:, condition_index], order) - baseline_contrast,
            "standard_residual": _rank_contrast(standard, order),
            "leave_runner_out": _rank_contrast(leave_runner, order),
        }

    rows = []
    summary = {
        "n_questions": len(winner),
        "both_keep_n": int(both_keep.sum()),
        "final_layer": logits.shape[2] - 1,
        "final": {},
    }
    for metric_index, metric in enumerate(METRICS):
        for condition_index, condition in enumerate(CONDITIONS):
            values = per_condition[condition][metric][both_keep]
            mean, low, high = _bootstrap(values, bootstrap, seed + metric_index * 100 + condition_index)
            rows.append({
                "metric": metric,
                "condition": condition,
                "mean": mean.tolist(),
                "ci_low": low.tolist(),
                "ci_high": high.tolist(),
            })
            summary["final"][f"{metric}:{condition}"] = {
                "mean": float(mean[-1]),
                "ci": [float(low[-1]), float(high[-1])],
            }
        paired = (
            per_condition["incorrect"][metric][both_keep]
            - per_condition["neutral"][metric][both_keep]
        )
        mean, low, high = _bootstrap(paired, bootstrap, seed + metric_index * 100 + 50)
        summary["final"][f"{metric}:incorrect_minus_neutral"] = {
            "mean": float(mean[-1]),
            "ci": [float(low[-1]), float(high[-1])],
            "positive_ci_layers": np.flatnonzero(low > 0).astype(int).tolist(),
            "negative_ci_layers": np.flatnonzero(high < 0).astype(int).tolist(),
        }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    long_rows = []
    for row in rows:
        for layer, (mean, low, high) in enumerate(zip(row["mean"], row["ci_low"], row["ci_high"])):
            long_rows.append({
                "metric": row["metric"],
                "condition": row["condition"],
                "layer": layer,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            })
    with (output / "runner_preservation_robustness.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=long_rows[0].keys())
        writer.writeheader()
        writer.writerows(long_rows)
    (output / "runner_preservation_robustness.json").write_text(json.dumps(summary, indent=2))
    _plot(rows, output, logits.shape[2] - 1, int(both_keep.sum()))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Test runner preservation across compression specifications")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, args.folds, args.seed, args.bootstrap), indent=2))


if __name__ == "__main__":
    main()
