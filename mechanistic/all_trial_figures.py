from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .answer_emergence_figures import (
    RANK_COLORS,
    RANK_LABELS,
    Z_975,
    macro_mean_and_se,
)
from .data import load_activation_dataset


CONDITIONS = ("baseline", "incorrect", "neutral")
CONDITION_LABELS = {"baseline": "Baseline", "incorrect": "Second Chance", "neutral": "Neutral"}
CONDITION_COLORS = {"baseline": "#333333", "incorrect": "#0072B2", "neutral": "#D55E00"}
CONDITION_STYLES = {"baseline": "-", "incorrect": "-", "neutral": "--"}


def rank_summary(
    aligned: np.ndarray, baseline_order: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_layers = aligned.shape[1]
    means = np.empty((n_layers, 4))
    halfwidths = np.empty((n_layers, 4))
    for rank in range(4):
        mean, se = macro_mean_and_se(aligned[:, :, rank], baseline_order[:, rank])
        means[:, rank] = mean
        halfwidths[:, rank] = Z_975 * se
    return means, halfwidths


def metric_summary(values: np.ndarray, winner_letters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, se = macro_mean_and_se(values, winner_letters)
    return mean, Z_975 * se


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


def _finish(ax, max_layer: int) -> None:
    ax.axhline(0, color="#555555", lw=0.65)
    ax.set_xlim(0, max_layer)
    step = max(1, round(max_layer / 8))
    ticks = list(np.arange(0, max_layer + 1, step))
    if ticks[-1] != max_layer:
        ticks.append(max_layer)
    ax.set_xticks(ticks)
    ax.set_xlabel(f"Residual readout (0 = embedding; {max_layer} = final block)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def save_rank_figure(
    output: Path,
    layers: np.ndarray,
    summaries: dict[str, tuple[np.ndarray, np.ndarray]],
    n_trials: int,
) -> None:
    import matplotlib.pyplot as plt

    _style()
    figure, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), sharex=True, sharey=True)
    for ax, condition in zip(axes, CONDITIONS):
        means, halfwidths = summaries[condition]
        for rank, (label, color) in enumerate(zip(RANK_LABELS, RANK_COLORS)):
            mean, half = means[:, rank], halfwidths[:, rank]
            ax.fill_between(layers, mean - half, mean + half, color=color, alpha=0.20, linewidth=0)
            ax.plot(layers, mean, color=color, lw=1.45, label=f"Original {label.lower()}")
        _finish(ax, int(layers[-1]))
        ax.set_title(CONDITION_LABELS[condition])
    axes[0].set_ylabel("Centered pseudo-logit\n(natural-logit units)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.94))
    figure.suptitle(
        f"All {n_trials} questions: trajectories aligned by original baseline rank",
        y=1.01, fontweight="bold", fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.87), w_pad=1.3)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"all_trials_original_rank_trajectories.{suffix}", bbox_inches="tight")
    plt.close(figure)


def save_mechanism_figure(
    output: Path,
    layers: np.ndarray,
    margin: dict[str, tuple[np.ndarray, np.ndarray]],
    spread: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    _style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharex=True)
    for condition in CONDITIONS:
        label = CONDITION_LABELS[condition]
        color = CONDITION_COLORS[condition]
        style = CONDITION_STYLES[condition]
        for ax, values in zip(axes, (margin, spread)):
            mean, half = values[condition]
            ax.fill_between(layers, mean - half, mean + half, color=color, alpha=0.16, linewidth=0)
            ax.plot(layers, mean, color=color, ls=style, lw=1.55, label=label)

    axes[0].set_title("A  Original-winner advantage", loc="left", fontweight="bold")
    axes[0].set_ylabel("Winner minus strongest alternative\n(natural-logit units)")
    axes[1].set_title("B  Total A-D spread", loc="left", fontweight="bold")
    axes[1].set_ylabel("Within-question A-D SD\n(natural-logit units)")
    for ax in axes:
        _finish(ax, int(layers[-1]))
    axes[1].set_ylim(bottom=0)
    axes[1].spines["bottom"].set_position(("data", 0))
    axes[0].legend(frameon=False, loc="upper left")
    figure.tight_layout(w_pad=2.0)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"all_trials_mechanism_summary.{suffix}", bbox_inches="tight")
    plt.close(figure)


def analyze(input_dir: str | Path, output_dir: str | Path) -> None:
    data = load_activation_dataset(input_dir, list(CONDITIONS))
    centered = data.logits - data.logits.mean(axis=-1, keepdims=True)
    baseline_order = np.argsort(-centered[:, 0, -1], axis=-1)
    layers = np.arange(centered.shape[2])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rank_summaries = {}
    margin_summaries = {}
    spread_summaries = {}
    rows = []
    for ci, condition in enumerate(CONDITIONS):
        aligned = np.take_along_axis(centered[:, ci], baseline_order[:, None, :], axis=-1)
        rank_summaries[condition] = rank_summary(aligned, baseline_order)

        winner_advantage = aligned[:, :, 0] - aligned[:, :, 1:].max(axis=-1)
        ad_spread = centered[:, ci].std(axis=-1)
        margin_summaries[condition] = metric_summary(winner_advantage, baseline_order[:, 0])
        spread_summaries[condition] = metric_summary(ad_spread, baseline_order[:, 0])

        for metric, summary in (("original_winner_advantage", margin_summaries[condition]),
                                ("ad_spread", spread_summaries[condition])):
            means, halfwidths = summary
            for layer, (mean, half) in enumerate(zip(means, halfwidths)):
                rows.append({"condition": condition, "metric": metric, "layer": layer,
                             "mean": mean, "ci_low": mean - half, "ci_high": mean + half})

    save_rank_figure(output, layers, rank_summaries, len(data.question_ids))
    save_mechanism_figure(output, layers, margin_summaries, spread_summaries)
    with (output / "all_trials_mechanism_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="All-trial baseline/Game/neutral trajectory figures")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(args.input, args.output)


if __name__ == "__main__":
    main()
