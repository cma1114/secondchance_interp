from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("normalized_full_stream_variance_ratio", "Full residual"),
    ("answer_contrast_variance_ratio", "A–D contrast"),
    ("orthogonal_complement_variance_ratio", "Everything else"),
)
CONDITIONS = (("game", "Game", "#C44E52"), ("neutral", "Neutral", "#4C72B0"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def plot(simple_path: Path, trivia_path: Path, output: Path) -> None:
    datasets = (("SimpleMC", _load(simple_path)), ("TriviaMC", _load(trivia_path)))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    x = np.arange(len(METRICS), dtype=float)
    width = 0.34
    for ax, (title, summary) in zip(axes, datasets):
        cells = summary["splits"]["all"]["conditions"]
        for offset, (key, label, color) in zip((-width / 2, width / 2), CONDITIONS):
            estimates = np.asarray([cells[key][metric]["estimate"] for metric, _ in METRICS])
            intervals = np.asarray([cells[key][metric]["ci95"] for metric, _ in METRICS])
            errors = np.vstack((estimates - intervals[:, 0], intervals[:, 1] - estimates))
            ax.bar(x + offset, estimates, width, label=label, color=color, alpha=0.9)
            ax.errorbar(
                x + offset,
                estimates,
                yerr=errors,
                fmt="none",
                ecolor="#202020",
                elinewidth=1.5,
                capsize=4,
                capthick=1.5,
                zorder=5,
            )
        ax.axhline(1.0, color="#303030", linestyle="--", linewidth=1.2)
        ax.set_title(f"{title} (n=500)", fontsize=13, weight="bold")
        ax.set_xticks(x, [label for _, label in METRICS])
        ax.tick_params(axis="x", labelrotation=12)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Variance relative to prompt-matched baseline", fontsize=11)
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Cleaned paradigm: final answer-choice variance contracts selectively",
        fontsize=15,
        weight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        "Actual layer-64 final RMSNorm state; bars are equal-baseline-letter estimates, error bars are paired bootstrap 95% CIs",
        ha="center",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", type=Path, required=True)
    parser.add_argument("--trivia", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.simple, args.trivia, args.output)


if __name__ == "__main__":
    main()
