from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCENARIOS = ("residual_trajectory", "gla_state", "joint")
SCENARIO_LABELS = ("Residual\ntrajectory", "GLA recurrent\nstate", "Both")
DIRECTIONS = (
    ("neutral_into_evaluation", "Neutral → Evaluation", "#377eb8", "o"),
    ("evaluation_into_neutral", "Evaluation → Neutral", "#e67e22", "s"),
)


def _row(summary: dict, subset: str, scenario: str, direction: str, metric: str, key: str):
    return summary["subsets"][subset]["scenarios"][scenario][direction][metric][key]


def plot(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    x = np.arange(len(SCENARIOS), dtype=float)
    offsets = (-0.10, 0.10)

    panels = (
        (
            "Conflict trials: W1−W2 margin",
            "conflict_W1_not_equal_W2",
            "w1_minus_w2_margin",
            "fraction_of_natural_gap",
            100.0,
            "Natural Evaluation–Neutral gap transferred (%)",
            100.0,
        ),
        (
            "Conflict trials: W1 choices",
            "conflict_W1_not_equal_W2",
            "w1_selection",
            "signed_transfer",
            100.0,
            "Movement toward donor condition (percentage points)",
            None,
        ),
        (
            "No-conflict trials: W1 choices",
            "no_conflict_W1_equal_W2",
            "w1_selection",
            "signed_transfer",
            100.0,
            "Movement toward donor condition (percentage points)",
            None,
        ),
    )

    for axis, (title, subset, metric, key, scale, ylabel, reference) in zip(axes, panels):
        axis.axhline(0, color="#777777", linewidth=0.8, zorder=0)
        if reference is not None:
            axis.axhline(reference, color="#999999", linewidth=0.9, linestyle="--", zorder=0)
            axis.text(2.34, reference, "full natural gap", color="#666666", va="bottom", ha="right", fontsize=9)
        for di, (direction, label, color, marker) in enumerate(DIRECTIONS):
            means, lows, highs = [], [], []
            for scenario in SCENARIOS:
                row = _row(summary, subset, scenario, direction, metric, key)
                means.append(row["mean"] * scale)
                lows.append(row["ci_low"] * scale)
                highs.append(row["ci_high"] * scale)
            means = np.asarray(means)
            lows = np.asarray(lows)
            highs = np.asarray(highs)
            axis.errorbar(
                x + offsets[di],
                means,
                yerr=np.vstack([means - lows, highs - means]),
                fmt=marker,
                linestyle="none",
                color=color,
                ecolor=color,
                markersize=7,
                capsize=4,
                linewidth=1.8,
                label=label,
                zorder=3,
            )
        axis.set_title(title, loc="left", fontsize=13, fontweight="normal")
        axis.set_xticks(x, SCENARIO_LABELS)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=10)

    axes[0].set_ylim(-15, 118)
    axes[1].set_ylim(-4, 20)
    axes[2].set_ylim(-5, 20)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.18, top=0.77, wspace=0.30)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.89))
    fig.suptitle("The action-period GLA state carries the revision effect", fontsize=16, fontweight="normal", y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.summary, args.output)


if __name__ == "__main__":
    main()
