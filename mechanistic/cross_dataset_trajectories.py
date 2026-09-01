from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .all_trial_figures import CONDITION_COLORS, CONDITION_LABELS, CONDITION_STYLES, CONDITIONS
from .answer_emergence_figures import Z_975, macro_mean_and_se
from .data import load_activation_dataset


METRICS = ("original_winner_advantage", "ad_spread")
METRIC_LABELS = {
    "original_winner_advantage": "Original-winner advantage\n(natural-logit units)",
    "ad_spread": "Within-question A-D SD\n(natural-logit units)",
}


def _summaries(input_dir: Path) -> tuple[dict, int, int]:
    data = load_activation_dataset(input_dir, list(CONDITIONS))
    centered = data.logits - data.logits.mean(axis=-1, keepdims=True)
    order = np.argsort(-centered[:, 0, -1], axis=-1)
    winner = order[:, 0]
    out = {}
    for ci, condition in enumerate(CONDITIONS):
        aligned = np.take_along_axis(centered[:, ci], order[:, None, :], axis=-1)
        values = {
            "original_winner_advantage": aligned[:, :, 0] - aligned[:, :, 1:].max(axis=-1),
            "ad_spread": centered[:, ci].std(axis=-1),
        }
        for metric, array in values.items():
            mean, se = macro_mean_and_se(array, winner)
            out[(condition, metric)] = (mean, mean - Z_975 * se, mean + Z_975 * se)
    return out, len(data.question_ids), centered.shape[2]


def analyze(
    simple_dir: str | Path,
    trivia_dir: str | Path,
    output_dir: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    datasets = {
        "SimpleMC": _summaries(Path(simple_dir)),
        "TriviaMC": _summaries(Path(trivia_dir)),
    }
    n_readouts = {value[2] for value in datasets.values()}
    if len(n_readouts) != 1:
        raise ValueError(f"Datasets have different readout counts: {sorted(n_readouts)}")
    layers = np.arange(n_readouts.pop())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True, sharey="col")
    for row_index, (dataset, (summary, n_questions, _)) in enumerate(datasets.items()):
        for column_index, metric in enumerate(METRICS):
            axis = axes[row_index, column_index]
            for condition in CONDITIONS:
                mean, low, high = summary[(condition, metric)]
                color = CONDITION_COLORS[condition]
                axis.fill_between(layers, low, high, color=color, alpha=0.14, linewidth=0)
                axis.plot(layers, mean, color=color, linestyle=CONDITION_STYLES[condition],
                          linewidth=1.5, label=CONDITION_LABELS[condition])
                for layer, values in enumerate(zip(mean, low, high)):
                    rows.append({
                        "dataset": dataset,
                        "n_questions": n_questions,
                        "condition": condition,
                        "metric": metric,
                        "layer": layer,
                        "mean": float(values[0]),
                        "ci_low": float(values[1]),
                        "ci_high": float(values[2]),
                    })
            axis.axvline(62, color="#777777", linewidth=0.7, linestyle=":")
            axis.axvline(63, color="#777777", linewidth=0.7, linestyle=":")
            axis.set_xlim(0, layers[-1])
            axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_ylabel(f"{dataset}\n{METRIC_LABELS[metric]}")
            if row_index == 0:
                title = "Original-winner advantage" if column_index == 0 else "Total A-D spread"
                axis.set_title(title, loc="left", fontweight="bold")
            if row_index == 1:
                axis.set_xlabel("Residual readout (0 = embedding; 64 = final block)")
    axes[0, 0].legend(frameon=False, loc="upper left", ncol=1)
    axes[0, 1].text(62.5, 0.98, "Mixer 62 / MLP 63", transform=axes[0, 1].get_xaxis_transform(),
                    ha="right", va="top", fontsize=8, color="#555555")
    figure.tight_layout(w_pad=2.0, h_pad=1.5)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"cross_dataset_mechanism_trajectories.{suffix}", dpi=300,
                       bbox_inches="tight")
    plt.close(figure)

    with (output / "cross_dataset_mechanism_trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched SimpleMC and TriviaMC mechanism trajectories")
    parser.add_argument("--simple", required=True)
    parser.add_argument("--trivia", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(args.simple, args.trivia, args.output)


if __name__ == "__main__":
    main()
