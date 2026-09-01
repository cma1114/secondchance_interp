from __future__ import annotations

import argparse
import json
from pathlib import Path


COLORS = ("#3795f6", "#f0833a", "#52bd72", "#e65eaa")


def plot(source: Path, output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(source.read_text())
    layers = np.asarray(data["layers"])
    figure = plt.figure(figsize=(12, 11.4))
    grid = figure.add_gridspec(2, 2, height_ratios=(1, 1.05), hspace=0.48, wspace=0.18)
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, :]),
    )
    keys = ("game_minus_baseline", "neutral_minus_baseline", "game_minus_neutral")
    titles = ("A  Game minus Baseline", "B  Neutral minus Baseline", "C  Game minus Neutral")

    all_bounds = []
    for contrast in data["contrasts"].values():
        for series in contrast["series"]:
            all_bounds.extend(series["ci_low"])
            all_bounds.extend(series["ci_high"])
    limit = max(1.4, float(np.ceil(max(abs(np.asarray(all_bounds))) * 10) / 10))

    for axis, key, title in zip(axes, keys, titles):
        contrast = data["contrasts"][key]
        axis.axvspan(1, 47.5, color="#d9d9d9", alpha=0.18, linewidth=0)
        axis.axhline(0, color="#808080", linewidth=0.65)
        axis.axvline(48, color="#8f8f8f", linewidth=0.9, linestyle=(0, (4, 3)))
        for color, series in zip(COLORS, contrast["series"]):
            mean = np.asarray(series["mean"])
            low = np.asarray(series["ci_low"])
            high = np.asarray(series["ci_high"])
            axis.fill_between(layers, low, high, color=color, alpha=0.13, linewidth=0)
            axis.plot(layers, mean, color=color, linewidth=1.75, label=series["rank"])
        axis.set_xlim(1, 64)
        axis.set_ylim(-limit, limit)
        axis.set_xticks((1, 8, 16, 24, 32, 40, 48, 56, 64))
        axis.set_yticks((-limit, -limit / 2, 0, limit / 2, limit))
        axis.grid(axis="y", color="#e5e5e5", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
        axis.tick_params(colors="#8c8c8c", labelsize=8, length=0)
        axis.set_title(title, loc="left", fontsize=15, pad=15, fontweight="normal")
        axis.set_xlabel("Residual readout", color="#8c8c8c", fontsize=9)
        axis.set_ylabel("Centered contrast (JLens score units)", color="#8c8c8c", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.988),
        fontsize=12,
        labelcolor="#8c8c8c",
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.25,
    )
    figure.text(
        0.5,
        0.018,
        "JLens; paired within question; fixed Baseline ranks; centered across A-D; "
        "95% confidence intervals. Shading marks low answer-decoding reliability.",
        ha="center",
        va="bottom",
        color="#8c8c8c",
        fontsize=10,
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.075)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.output)


if __name__ == "__main__":
    main()
