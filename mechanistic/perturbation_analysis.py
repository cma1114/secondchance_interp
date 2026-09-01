from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .cumulative_hypothesis_analysis import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_STYLES,
    _design,
    _question_weights,
    _weighted_ridge,
)
from .data import load_activation_dataset
from .probes import stratified_folds
from .trajectory_analysis import centered


CONDITIONS = ("incorrect", "neutral")


def _balanced_bootstrap_ci(
    values: np.ndarray,
    labels: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Letter-balanced mean and question-clustered stratified percentile CI."""
    strata = [np.flatnonzero(labels == letter) for letter in range(4)]
    if any(len(ids) == 0 for ids in strata):
        raise ValueError("All four winner-letter strata are required")
    mean = np.mean(np.stack([values[ids].mean(axis=0) for ids in strata]), axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((repetitions, values.shape[1]))
    for repetition in range(repetitions):
        letter_means = []
        for ids in strata:
            sample = rng.choice(ids, size=len(ids), replace=True)
            letter_means.append(values[sample].mean(axis=0))
        boot[repetition] = np.mean(letter_means, axis=0)
    return mean, np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)


def _cross_fitted_compression_residuals(
    baseline: np.ndarray,
    target: np.ndarray,
    winner: np.ndarray,
    folds: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict condition-baseline from letter effects plus baseline geometry."""
    n, n_layers, _ = baseline.shape
    residuals = np.empty_like(target)
    compression_strength = np.empty((n, n_layers))
    weights = _question_weights(winner)
    all_ids = np.arange(n)
    for layer in range(n_layers):
        baseline_layer = baseline[:, layer]
        leader = np.argmax(baseline_layer, axis=-1)
        sorted_baseline = np.sort(baseline_layer, axis=-1)
        margin = sorted_baseline[:, -1] - sorted_baseline[:, -2]
        for test in folds:
            train = np.setdiff1d(all_ids, test)
            x_train, names = _design(
                "compression",
                baseline_layer[train],
                winner[train],
                leader[train],
                margin[train],
                0.0,
            )
            x_test, _ = _design(
                "compression",
                baseline_layer[test],
                winner[test],
                leader[test],
                margin[test],
                0.0,
            )
            prediction, coefficients = _weighted_ridge(
                x_train,
                target[train, layer].reshape(-1),
                np.repeat(weights[train], 4),
                x_test,
            )
            prediction = prediction.reshape(len(test), 4)
            residuals[test, layer] = target[test, layer] - prediction
            coefficient_map = dict(zip(["intercept", *names], coefficients))
            compression_strength[test, layer] = -coefficient_map["baseline_geometry"]
    residuals -= residuals.mean(axis=-1, keepdims=True)
    return residuals, compression_strength


def _residual_metrics(
    residuals: np.ndarray,
    baseline_order: np.ndarray,
) -> dict[str, np.ndarray]:
    aligned = np.take_along_axis(residuals, baseline_order[:, None, :], axis=-1)
    winner_values = aligned[:, :, 0]
    runner_values = aligned[:, :, 1]
    lower_alternatives = aligned[:, :, 2:].mean(axis=-1)
    signed_projection = (runner_values - winner_values) / np.sqrt(2)
    energy = np.sum(residuals**2, axis=-1)
    energy_fraction = np.divide(
        signed_projection**2,
        energy,
        out=np.zeros_like(signed_projection),
        where=energy > 1e-12,
    )
    return {
        "residual_rms": np.sqrt(np.mean(residuals**2, axis=-1)),
        "runner_vs_winner_projection": signed_projection,
        "winner_runner_energy_fraction": energy_fraction,
        # These two contrasts sum to runner-minus-winner. They distinguish a
        # falling winner from a rising runner using original ranks 3-4 as the
        # common reference after compression and letter effects are removed.
        "leader_suppression": lower_alternatives - winner_values,
        "runner_boost": runner_values - lower_alternatives,
    }


def _aligned_updates(logits: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    direction = baseline[:, :-1]
    norms = np.linalg.norm(direction, axis=-1, keepdims=True)
    unit = np.divide(direction, norms, out=np.zeros_like(direction), where=norms > 1e-12)
    updates = logits[:, 1:] - logits[:, :-1]
    return np.sum(updates * unit, axis=-1)


def _update_categories(baseline_updates: np.ndarray, condition_updates: np.ndarray) -> dict[str, np.ndarray]:
    baseline_accumulates = baseline_updates > 0
    return {
        "active_cancellation_fraction": (baseline_accumulates & (condition_updates < 0)).astype(float),
        "reduced_accumulation_fraction": (
            baseline_accumulates & (condition_updates >= 0) & (condition_updates < baseline_updates)
        ).astype(float),
    }


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "svg.fonttype": "none",
    })


def _finish_axis(axis, final_layer: int) -> None:
    axis.axhline(0, color="#555555", lw=0.65)
    axis.set_xlim(0, final_layer)
    step = max(1, round(final_layer / 8))
    ticks = list(np.arange(0, final_layer + 1, step))
    if ticks[-1] != final_layer:
        ticks.append(final_layer)
    axis.set_xticks(ticks)
    axis.set_xlabel(f"Residual readout (0 = embedding; {final_layer} = final block)")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.7)
    axis.set_axisbelow(True)


def _save_figure(output: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    _style()
    lookup = {(row["condition"], row["metric"]): row for row in rows}
    figure, axes = plt.subplots(2, 2, figsize=(7.6, 5.8))
    layers = np.arange(len(lookup[("incorrect", "residual_rms")]["mean"]))
    for condition in CONDITIONS:
        color = CONDITION_COLORS[condition]
        style = CONDITION_STYLES[condition]
        label = CONDITION_LABELS[condition]
        for axis, metric in zip(
            (axes[0, 0], axes[1, 0]),
            ("residual_rms", "runner_vs_winner_projection"),
        ):
            row = lookup[(condition, metric)]
            mean = np.asarray(row["mean"])
            low = np.asarray(row["ci_low"])
            high = np.asarray(row["ci_high"])
            axis.fill_between(layers, low, high, color=color, alpha=0.16, linewidth=0)
            axis.plot(layers, mean, color=color, ls=style, lw=1.55, label=label)

        row = lookup[(condition, "winner_runner_energy_fraction")]
        mean = np.asarray(row["mean"])
        low = np.asarray(row["ci_low"])
        high = np.asarray(row["ci_high"])
        axes[1, 1].fill_between(layers, low, high, color=color, alpha=0.16, linewidth=0)
        axes[1, 1].plot(layers, mean, color=color, ls=style, lw=1.55, label=label)

    difference = lookup[("incorrect_minus_neutral", "residual_rms")]
    difference_mean = np.asarray(difference["mean"])
    difference_low = np.asarray(difference["ci_low"])
    difference_high = np.asarray(difference["ci_high"])
    axes[0, 1].fill_between(
        layers, difference_low, difference_high, color=CONDITION_COLORS["incorrect"], alpha=0.18, linewidth=0
    )
    axes[0, 1].plot(layers, difference_mean, color=CONDITION_COLORS["incorrect"], lw=1.55)
    axes[1, 1].axhline(1 / 3, color="#777777", ls=":", lw=1.0)
    axes[1, 1].text(2, 1 / 3 + .008, "Isotropic expectation (1/3)", color="#555555", fontsize=8)

    axes[0, 0].set_title("A  Unexplained perturbation", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Cross-fitted residual RMS\n(natural-logit units)")
    axes[0, 1].set_title("B  Extra perturbation in the Game", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("Game-minus-neutral residual RMS\n(natural-logit units)")
    axes[1, 0].set_title("C  Direction toward the runner-up", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("Residual runner-up-minus-winner\nprojection (natural-logit units)")
    axes[1, 1].set_title("D  Winner-runner structure", loc="left", fontweight="bold")
    axes[1, 1].set_ylabel("Fraction of residual energy\nin winner-runner direction")
    for axis in axes.ravel():
        _finish_axis(axis, int(layers[-1]))
    axes[0, 0].legend(frameon=False, loc="upper left")
    figure.tight_layout(w_pad=2.0, h_pad=1.6)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"perturbation_decomposition.{suffix}", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(1, 1, figsize=(5.3, 3.2))
    update_layers = np.arange(len(lookup[("baseline", "evidence_aligned_update")]["mean"]))
    update_colors = {"baseline": "#333333", **CONDITION_COLORS}
    update_styles = {"baseline": "-", **CONDITION_STYLES}
    update_labels = {"baseline": "Baseline", "incorrect": "Second Chance", "neutral": "Neutral"}
    for condition in ("baseline", *CONDITIONS):
        row = lookup[(condition, "evidence_aligned_update")]
        mean = np.asarray(row["mean"])
        low = np.asarray(row["ci_low"])
        high = np.asarray(row["ci_high"])
        axis.fill_between(
            update_layers, low, high, color=update_colors[condition], alpha=0.10, linewidth=0
        )
        axis.plot(
            update_layers,
            mean,
            color=update_colors[condition],
            ls=update_styles[condition],
            lw=1.45,
            label=update_labels[condition],
        )

    axis.set_title("Evidence-aligned block updates", loc="left", fontweight="bold")
    axis.set_ylabel("Update along current baseline answer geometry\n(natural-logit units)")
    _finish_axis(axis, int(update_layers[-1]))
    axis.set_xlabel(f"Transition from residual readout (0 through {int(update_layers[-1])})")
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(output / f"evidence_aligned_updates.{suffix}", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharex=True, sharey=True)
    contrast_styles = {
        "leader_suppression": ("Leader suppression", "#0072B2", "-"),
        "runner_boost": ("Runner-up boost", "#E69F00", "--"),
    }
    for axis, condition in zip(axes, CONDITIONS):
        for metric, (label, color, style) in contrast_styles.items():
            row = lookup[(condition, metric)]
            mean = np.asarray(row["mean"])
            low = np.asarray(row["ci_low"])
            high = np.asarray(row["ci_high"])
            axis.fill_between(layers, low, high, color=color, alpha=0.16, linewidth=0)
            axis.plot(layers, mean, color=color, ls=style, lw=1.55, label=label)
        axis.set_title(CONDITION_LABELS[condition])
        _finish_axis(axis, int(layers[-1]))
    axes[0].set_ylabel("Residual rank contrast after compression\n(natural-logit units)")
    axes[0].legend(frameon=False, loc="upper left")
    figure.tight_layout(w_pad=1.6)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"leader_suppression_vs_runner_boost.{suffix}", bbox_inches="tight")
    plt.close(figure)


def analyze(
    input_dir: str | Path,
    output_dir: str | Path,
    k_folds: int,
    seed: int,
    bootstrap_repetitions: int,
) -> dict:
    data = load_activation_dataset(input_dir, ["baseline", *CONDITIONS])
    logits = centered(data.logits)
    baseline = logits[:, 0]
    baseline_order = np.argsort(-baseline[:, -1], axis=-1)
    winner = baseline_order[:, 0]
    folds = stratified_folds(winner, k_folds, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    per_question_metrics: dict[tuple[str, str], np.ndarray] = {}
    compression_strengths = {}
    for condition_index, condition in enumerate(CONDITIONS, start=1):
        target = logits[:, condition_index] - baseline
        residuals, compression_strength = _cross_fitted_compression_residuals(
            baseline, target, winner, folds
        )
        compression_strengths[condition] = compression_strength
        for metric, values in _residual_metrics(residuals, baseline_order).items():
            per_question_metrics[(condition, metric)] = values

    for condition_index, condition in enumerate(("baseline", *CONDITIONS)):
        per_question_metrics[(condition, "evidence_aligned_update")] = _aligned_updates(
            logits[:, condition_index], baseline
        )
    baseline_updates = per_question_metrics[("baseline", "evidence_aligned_update")]
    for condition in CONDITIONS:
        categories = _update_categories(
            baseline_updates,
            per_question_metrics[(condition, "evidence_aligned_update")],
        )
        for metric, values in categories.items():
            per_question_metrics[(condition, metric)] = values

    # Paired differences answer whether the Game has more residual perturbation
    # than neutral, rather than inferring that from two separate intervals.
    for metric in (
        "residual_rms",
        "runner_vs_winner_projection",
        "winner_runner_energy_fraction",
    ):
        per_question_metrics[("incorrect_minus_neutral", metric)] = (
            per_question_metrics[("incorrect", metric)]
            - per_question_metrics[("neutral", metric)]
        )

    rows = []
    summarized = {}
    for row_index, ((condition, metric), values) in enumerate(per_question_metrics.items()):
        mean, low, high = _balanced_bootstrap_ci(
            values, winner, bootstrap_repetitions, seed + 100 * row_index
        )
        row = {
            "condition": condition,
            "metric": metric,
            "mean": mean.tolist(),
            "ci_low": low.tolist(),
            "ci_high": high.tolist(),
        }
        rows.append(row)
        summarized[(condition, metric)] = (mean, low, high)

    long_rows = []
    for row in rows:
        for layer, (mean, low, high) in enumerate(zip(row["mean"], row["ci_low"], row["ci_high"])):
            long_rows.append({
                "condition": row["condition"],
                "metric": row["metric"],
                "layer": layer,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            })
    with (output / "perturbation_and_accumulation_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=long_rows[0].keys())
        writer.writeheader()
        writer.writerows(long_rows)

    _save_figure(output, rows)
    summary = {
        "n_questions": len(winner),
        "cross_validation": f"{k_folds}-fold, stratified by final baseline-winner letter",
        "bootstrap": f"{bootstrap_repetitions} question-clustered, winner-letter-stratified repetitions",
        "residual_definition": (
            "Condition-minus-baseline centered pseudo-logits after cross-fitted removal "
            "of option-letter effects and proportional baseline-geometry compression."
        ),
        "selected_layers": {},
    }
    final_layer = logits.shape[2] - 1
    selected_layers = sorted(
        set(layer for layer in (16, 24, 30, 32, 40, 48, 56, 60, 63, final_layer) if layer <= final_layer)
    )
    for condition in CONDITIONS:
        summary["selected_layers"][condition] = {}
        for layer in selected_layers:
            summary["selected_layers"][condition][str(layer)] = {
                metric: {
                    "mean": float(summarized[(condition, metric)][0][layer]),
                    "ci": [
                        float(summarized[(condition, metric)][1][layer]),
                        float(summarized[(condition, metric)][2][layer]),
                    ],
                }
                for metric in (
                    "residual_rms",
                    "runner_vs_winner_projection",
                    "winner_runner_energy_fraction",
                    "leader_suppression",
                    "runner_boost",
                )
            }
        summary["selected_layers"][condition]["update_transitions_24_40"] = {
            "mean_evidence_aligned_update": float(
                per_question_metrics[(condition, "evidence_aligned_update")][:, 24:40].mean()
            ),
            "active_cancellation_fraction": float(
                per_question_metrics[(condition, "active_cancellation_fraction")][:, 24:40].mean()
            ),
            "reduced_accumulation_fraction": float(
                per_question_metrics[(condition, "reduced_accumulation_fraction")][:, 24:40].mean()
            ),
        }
    summary["baseline_updates_24_40"] = {
        "mean_evidence_aligned_update": float(baseline_updates[:, 24:40].mean())
    }
    difference_mean, difference_low, difference_high = summarized[
        ("incorrect_minus_neutral", "residual_rms")
    ]
    summary["game_minus_neutral_residual_rms"] = {
        "positive_ci_layers": np.flatnonzero(difference_low > 0).astype(int).tolist(),
        "negative_ci_layers": np.flatnonzero(difference_high < 0).astype(int).tolist(),
        "layer_30": {
            "mean": float(difference_mean[30]),
            "ci": [float(difference_low[30]), float(difference_high[30])],
        },
        f"layer_{final_layer}": {
            "mean": float(difference_mean[final_layer]),
            "ci": [float(difference_low[final_layer]), float(difference_high[final_layer])],
        },
    }
    (output / "perturbation_and_accumulation_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate compression, perturbation, and evidence accumulation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, args.folds, args.seed, args.bootstrap), indent=2))


if __name__ == "__main__":
    main()
