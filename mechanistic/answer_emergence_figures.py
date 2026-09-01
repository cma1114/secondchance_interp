from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .data import load_activation_dataset


LETTERS = "ABCD"
Z_975 = 1.959963984540054
RANK_LABELS = ("Winner", "Runner-up", "Rank 3", "Rank 4")
RANK_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def macro_mean_and_se(values: np.ndarray, strata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal-stratum mean and its independent-strata standard error.

    ``values`` has shape (question, layer), and ``strata`` identifies one of the
    four answer letters for each question.  The estimand gives each letter 1/4
    weight, irrespective of its frequency in the sample.
    """
    means, variances = [], []
    for letter in range(4):
        group = values[strata == letter]
        if len(group) < 2:
            raise ValueError(f"Need at least two questions in letter stratum {LETTERS[letter]}")
        means.append(group.mean(axis=0))
        variances.append(group.var(axis=0, ddof=1) / len(group))
    return np.mean(means, axis=0), np.sqrt(np.sum(variances, axis=0) / 16.0)


def balanced_lens_summary(input_dir: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    data = load_activation_dataset(input_dir, ["baseline"])
    logits = data.logits[:, 0]
    centered = logits - logits.mean(axis=-1, keepdims=True)
    final_order = np.argsort(-centered[:, -1], axis=-1)
    aligned = np.take_along_axis(centered, final_order[:, None, :], axis=-1)

    means, halfwidths = {}, {}
    for rank, label in enumerate(RANK_LABELS):
        mean, se = macro_mean_and_se(aligned[:, :, rank], final_order[:, rank])
        means[label] = mean
        halfwidths[label] = Z_975 * se
    return final_order, means, halfwidths


def balanced_probe_summary(
    probe_csv: str | Path, final_order: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    with Path(probe_csv).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    layers = np.asarray(sorted({int(row["layer"]) for row in rows}), dtype=int)
    means, halfwidths = {}, {}
    for target, label, rank in (("winner", "Winner identity", 0), ("runner_up", "Runner-up identity", 1)):
        target_rows = {int(row["layer"]): row for row in rows if row["target"] == target}
        counts = np.bincount(final_order[:, rank], minlength=4)
        target_means, target_halfwidths = [], []
        for layer in layers:
            recalls = np.asarray([float(target_rows[int(layer)][f"accuracy_{letter}"]) for letter in LETTERS])
            target_means.append(float(recalls.mean()))
            # Conditional normal approximation for the mean of four held-out
            # class recalls. This does not include probe-refitting uncertainty.
            se = np.sqrt(np.sum(recalls * (1.0 - recalls) / counts) / 16.0)
            target_halfwidths.append(float(Z_975 * se))
        means[label] = np.asarray(target_means)
        halfwidths[label] = np.asarray(target_halfwidths)
    return layers, means, halfwidths


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
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 7.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _lens_panel(ax, layers: np.ndarray, means: dict[str, np.ndarray], halfwidths: dict[str, np.ndarray]) -> None:
    for label, color in zip(RANK_LABELS, RANK_COLORS):
        mean, half = means[label], halfwidths[label]
        ax.fill_between(layers, mean - half, mean + half, color=color, alpha=0.22, linewidth=0)
        ax.plot(layers, mean, color=color, lw=1.55, label=label)
    ax.axhline(0, color="#444444", lw=0.65)
    ax.set_xlim(layers[0], layers[-1])
    ax.set_xticks(np.arange(0, layers[-1] + 1, 8))
    ax.set_ylabel("Centered pseudo-logit\n(natural-logit units)")
    ax.set_title("A  Native logit lens", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.7, columnspacing=1.0)


def _probe_panel(
    ax,
    layers: np.ndarray,
    means: dict[str, np.ndarray],
    halfwidths: dict[str, np.ndarray],
) -> None:
    for label, color in zip(("Winner identity", "Runner-up identity"), RANK_COLORS[:2]):
        mean, half = means[label], halfwidths[label]
        ax.fill_between(layers, mean - half, mean + half, color=color, alpha=0.22, linewidth=0)
        ax.plot(layers, mean, color=color, lw=1.55, label=label)
    ax.axhline(0.25, color="#555555", lw=0.85, ls=(0, (3, 2)), label="Chance")
    ax.set_xlim(layers[0], layers[-1])
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, layers[-1] + 1, 8))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("B  Held-out linear probes", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left", handlelength=1.7)


def _finish_axis(ax, final_layer: int) -> None:
    ax.set_xlabel(f"Residual readout (0 = embedding; {final_layer} = final block)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def save_figures(
    output_dir: str | Path,
    lens_means: dict[str, np.ndarray],
    lens_halfwidths: dict[str, np.ndarray],
    probe_layers: np.ndarray,
    probe_means: dict[str, np.ndarray],
    probe_halfwidths: dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    _style()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lens_layers = np.arange(len(next(iter(lens_means.values()))))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    _lens_panel(axes[0], lens_layers, lens_means, lens_halfwidths)
    _probe_panel(axes[1], probe_layers, probe_means, probe_halfwidths)
    for ax in axes:
        _finish_axis(ax, int(lens_layers[-1]))
    fig.tight_layout(w_pad=2.0)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"answer_emergence_combined.{suffix}", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.55, 3.0))
    _lens_panel(ax, lens_layers, lens_means, lens_halfwidths)
    _finish_axis(ax, int(lens_layers[-1]))
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"answer_emergence_logit_lens.{suffix}", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.55, 3.0))
    _probe_panel(ax, probe_layers, probe_means, probe_halfwidths)
    _finish_axis(ax, int(probe_layers[-1]))
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"answer_emergence_probes.{suffix}", bbox_inches="tight")
    plt.close(fig)


def write_values(
    output_dir: str | Path,
    lens_means: dict[str, np.ndarray],
    lens_halfwidths: dict[str, np.ndarray],
    probe_layers: np.ndarray,
    probe_means: dict[str, np.ndarray],
    probe_halfwidths: dict[str, np.ndarray],
) -> None:
    output = Path(output_dir)
    rows = []
    for label in RANK_LABELS:
        for layer, (mean, half) in enumerate(zip(lens_means[label], lens_halfwidths[label])):
            rows.append({"method": "native_logit_lens", "series": label, "layer": layer,
                         "mean": mean, "ci_low": mean - half, "ci_high": mean + half})
    for label in ("Winner identity", "Runner-up identity"):
        for layer, mean, half in zip(probe_layers, probe_means[label], probe_halfwidths[label]):
            rows.append({"method": "held_out_logistic_probe", "series": label, "layer": layer,
                         "mean": mean, "ci_low": mean - half, "ci_high": mean + half})
    with (output / "answer_emergence_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("method", "series", "layer", "mean", "ci_low", "ci_high"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready baseline answer-emergence figures")
    parser.add_argument("--input", required=True, help="Mechanistic run directory containing shards/")
    parser.add_argument("--probe-csv", required=True, help="Held-out logistic-probe results CSV")
    parser.add_argument("--output", required=True, help="Destination directory for PDF/SVG/PNG and values")
    args = parser.parse_args()

    order, lens_means, lens_halfwidths = balanced_lens_summary(args.input)
    probe_layers, probe_means, probe_halfwidths = balanced_probe_summary(args.probe_csv, order)
    save_figures(args.output, lens_means, lens_halfwidths, probe_layers, probe_means, probe_halfwidths)
    write_values(args.output, lens_means, lens_halfwidths, probe_layers, probe_means, probe_halfwidths)


if __name__ == "__main__":
    main()
