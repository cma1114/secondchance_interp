from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .all_trial_figures import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_STYLES,
    CONDITIONS,
    _style,
)
from .answer_emergence_figures import (
    RANK_COLORS,
    RANK_LABELS,
    Z_975,
    macro_mean_and_se,
)
from .data import decision_letter, load_activation_dataset


METRICS = ("original_winner_advantage", "ad_spread")
METRIC_LABELS = {
    "original_winner_advantage": (
        "Original-winner probe score minus\nstrongest alternative (baseline SD units)"
    ),
    "ad_spread": "Within-question A-D probe-score SD\n(baseline SD units)",
}


def _probe_accuracy_figure(
    scores: np.ndarray,
    layers: np.ndarray,
    generated_answers: np.ndarray,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    predictions = scores.argmax(axis=-1)  # condition, question, layer
    rows = []
    summaries = {}
    for ci, condition in enumerate(CONDITIONS):
        labels = generated_answers[:, ci]
        counts = np.bincount(labels, minlength=4)
        overall, balanced, overall_half, balanced_half = [], [], [], []
        for li, layer in enumerate(layers):
            correct = predictions[ci, :, li] == labels
            recalls = np.asarray([
                np.mean(predictions[ci, labels == letter, li] == letter)
                for letter in range(4)
            ])
            accuracy = float(correct.mean())
            balanced_accuracy = float(recalls.mean())
            accuracy_se = np.sqrt(accuracy * (1.0 - accuracy) / len(labels))
            balanced_se = np.sqrt(np.sum(recalls * (1.0 - recalls) / counts) / 16.0)
            overall.append(accuracy)
            balanced.append(balanced_accuracy)
            overall_half.append(Z_975 * accuracy_se)
            balanced_half.append(Z_975 * balanced_se)
            rows.extend((
                {"condition": condition, "metric": "overall_accuracy", "layer": int(layer),
                 "mean": accuracy, "ci_low": accuracy - Z_975 * accuracy_se,
                 "ci_high": accuracy + Z_975 * accuracy_se},
                {"condition": condition, "metric": "balanced_accuracy", "layer": int(layer),
                 "mean": balanced_accuracy, "ci_low": balanced_accuracy - Z_975 * balanced_se,
                 "ci_high": balanced_accuracy + Z_975 * balanced_se},
            ))
        summaries[condition] = (
            np.asarray(overall), np.asarray(overall_half),
            np.asarray(balanced), np.asarray(balanced_half),
            float(counts.max() / counts.sum()),
        )

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), sharex=True, sharey=True)
    for condition in CONDITIONS:
        overall, overall_half, balanced, balanced_half, majority = summaries[condition]
        color = CONDITION_COLORS[condition]
        style = CONDITION_STYLES[condition]
        axes[0].fill_between(layers, overall - overall_half, overall + overall_half,
                             color=color, alpha=0.14, linewidth=0)
        axes[0].plot(layers, overall, color=color, linestyle=style, linewidth=1.55,
                     label=CONDITION_LABELS[condition])
        axes[0].axhline(majority, color=color, linewidth=0.65, linestyle=(0, (1, 2)), alpha=0.65)
        axes[1].fill_between(layers, balanced - balanced_half, balanced + balanced_half,
                             color=color, alpha=0.14, linewidth=0)
        axes[1].plot(layers, balanced, color=color, linestyle=style, linewidth=1.55,
                     label=CONDITION_LABELS[condition])
    axes[0].set_title("A  Overall accuracy", loc="left", fontweight="bold")
    axes[0].set_ylabel("Probe leader matches eventual\ngenerated answer")
    axes[0].text(1, 0.965, "Dotted: condition majority-letter baseline",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=7.2,
                 color="#555555")
    axes[1].set_title("B  Letter-balanced accuracy", loc="left", fontweight="bold")
    axes[1].axhline(0.25, color="#555555", linewidth=0.8, linestyle=(0, (3, 2)))
    axes[1].text(1, 0.27, "Chance = 25%", transform=axes[1].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=7.2, color="#555555")
    for axis in axes:
        axis.set_xlim(0, int(layers[-1]))
        axis.set_ylim(0, 1)
        axis.set_xticks(np.arange(0, int(layers[-1]) + 1, 8))
        axis.set_yticks(np.arange(0, 1.01, 0.25))
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    figure.suptitle("SimpleMC: reliability of the Baseline-trained candidate probe",
                    fontsize=10.5, fontweight="bold")
    figure.supxlabel("Residual readout (0 = embedding; 64 = final block)", y=0.015)
    figure.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.0)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output / f"simplemc_probe_final_answer_accuracy.{suffix}",
                       dpi=300, bbox_inches="tight")
    plt.close(figure)

    with (output / "simplemc_probe_final_answer_accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(score_file: str | Path, input_dir: str | Path, output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    _style()

    with np.load(score_file, allow_pickle=False) as cached:
        scores = cached["scores"].astype(np.float64)
        layers = cached["layers"].astype(int)
        conditions = cached["conditions"].astype(str).tolist()
        order = cached["baseline_order"].astype(int)
        cached_question_ids = cached["question_ids"].astype(str).tolist()

    if conditions != list(CONDITIONS):
        raise ValueError(f"Expected conditions {CONDITIONS}, found {conditions}")

    winner_letters = order[:, 0]
    summaries: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rank_summaries: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for ci, condition in enumerate(CONDITIONS):
        aligned = np.take_along_axis(scores[ci], order[:, None, :], axis=-1)
        rank_mean = np.empty((len(layers), 4))
        rank_halfwidth = np.empty((len(layers), 4))
        for rank in range(4):
            mean, se = macro_mean_and_se(aligned[:, :, rank], order[:, rank])
            rank_mean[:, rank] = mean
            rank_halfwidth[:, rank] = Z_975 * se
        rank_summaries[condition] = rank_mean, rank_halfwidth
        values = {
            "original_winner_advantage": aligned[:, :, 0] - aligned[:, :, 1:].max(axis=-1),
            "ad_spread": scores[ci].std(axis=-1),
        }
        for metric, array in values.items():
            mean, se = macro_mean_and_se(array, winner_letters)
            low = mean - Z_975 * se
            high = mean + Z_975 * se
            summaries[(condition, metric)] = mean, low, high
            for layer, value, lower, upper in zip(layers, mean, low, high):
                rows.append({
                    "condition": condition,
                    "metric": metric,
                    "layer": int(layer),
                    "mean": float(value),
                    "ci_low": float(lower),
                    "ci_high": float(upper),
                })

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data = load_activation_dataset(input_dir, list(CONDITIONS))
    if data.question_ids != cached_question_ids:
        raise ValueError("Probe-score and activation-dataset question order differs")
    generated_answers = np.empty((len(data.question_ids), len(CONDITIONS)), dtype=int)
    for qi, question_id in enumerate(data.question_ids):
        for ci, condition in enumerate(CONDITIONS):
            answer = decision_letter(data.metadata[(question_id, condition)])
            if answer not in "ABCD":
                raise ValueError(f"Non-A-D generated answer for {condition}/{question_id}: {answer!r}")
            generated_answers[qi, ci] = "ABCD".index(answer)
    _probe_accuracy_figure(scores, layers, generated_answers, output)

    rank_figure, rank_axes = plt.subplots(1, 3, figsize=(9.8, 3.25), sharex=True, sharey=True)
    for axis, condition in zip(rank_axes, CONDITIONS):
        means, halfwidths = rank_summaries[condition]
        for rank, (label, color) in enumerate(zip(RANK_LABELS, RANK_COLORS)):
            mean = means[:, rank]
            halfwidth = halfwidths[:, rank]
            axis.fill_between(
                layers, mean - halfwidth, mean + halfwidth,
                color=color, alpha=0.20, linewidth=0,
            )
            axis.plot(layers, mean, color=color, linewidth=1.45,
                      label=f"Original {label.lower()}")
        axis.axhline(0, color="#555555", linewidth=0.65)
        axis.axvline(62, color="#777777", linewidth=0.7, linestyle=":")
        axis.axvline(63, color="#777777", linewidth=0.7, linestyle=":")
        axis.set_xlim(0, int(layers[-1]))
        axis.set_xticks(np.arange(0, int(layers[-1]) + 1, 8))
        axis.set_title(CONDITION_LABELS[condition])
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    rank_axes[0].set_ylabel("Centered candidate-probe score\n(baseline SD units)")
    handles, labels = rank_axes[0].get_legend_handles_labels()
    rank_figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
                       bbox_to_anchor=(0.5, 0.94))
    rank_figure.suptitle(
        "SimpleMC: probe trajectories aligned by final Baseline rank",
        y=1.01, fontsize=10.5, fontweight="bold",
    )
    rank_figure.supxlabel("Residual readout (0 = embedding; 64 = final block)", y=0.015)
    rank_figure.tight_layout(rect=(0, 0.06, 1, 0.87), w_pad=1.3)
    for suffix in ("png", "svg", "pdf"):
        rank_figure.savefig(output / f"simplemc_probe_rank_trajectories.{suffix}",
                            dpi=300, bbox_inches="tight")
    plt.close(rank_figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.35), sharex=True)
    for column, metric in enumerate(METRICS):
        axis = axes[column]
        for condition in CONDITIONS:
            mean, low, high = summaries[(condition, metric)]
            color = CONDITION_COLORS[condition]
            axis.fill_between(layers, low, high, color=color, alpha=0.15, linewidth=0)
            axis.plot(
                layers,
                mean,
                color=color,
                linestyle=CONDITION_STYLES[condition],
                linewidth=1.55,
                label=CONDITION_LABELS[condition],
            )
        axis.axvline(62, color="#777777", linewidth=0.7, linestyle=":")
        axis.axvline(63, color="#777777", linewidth=0.7, linestyle=":")
        axis.set_xlim(0, int(layers[-1]))
        axis.set_xticks(np.arange(0, int(layers[-1]) + 1, 8))
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel(METRIC_LABELS[metric])
    axes[0].set_title("A  Original-winner advantage", loc="left", fontweight="bold")
    axes[1].set_title("B  Total A-D spread", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].text(
        62.5,
        0.98,
        "Mixer 62 / MLP 63",
        transform=axes[1].get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )
    figure.suptitle(
        "SimpleMC: cross-fitted Baseline-winner probe trajectories",
        fontsize=10.5,
        fontweight="bold",
    )
    figure.supxlabel("Residual readout (0 = embedding; 64 = final block)", y=0.015)
    figure.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.0)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output / f"simplemc_probe_mechanism_trajectories.{suffix}",
                       dpi=300, bbox_inches="tight")
    plt.close(figure)

    with (output / "simplemc_probe_mechanism_trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimpleMC all-trial mechanism trajectories in probe-score coordinates"
    )
    parser.add_argument("--scores", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(args.scores, args.input, args.output)


if __name__ == "__main__":
    main()
