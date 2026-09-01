from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input.open()))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    panels = [
        (
            "confirmation_original_balanced_accuracy",
            "confirmation_original_ci_low",
            "confirmation_original_ci_high",
            "A  Original option mapping",
        ),
        (
            "confirmation_remapped_balanced_accuracy",
            "confirmation_remapped_ci_low",
            "confirmation_remapped_ci_high",
            "B  Every option moved to a new letter",
        ),
    ]
    styles = {
        "content_end": ("#2F8FED", "Last option-content token"),
        "line_end": ("#F27F33", "Option-closing newline"),
    }
    for axis, (metric, low_key, high_key, title) in zip(axes, panels):
        for anchor in ("content_end", "line_end"):
            subset = [row for row in rows if row["anchor"] == anchor]
            layers = np.asarray([int(row["layer"]) for row in subset])
            values = np.asarray([float(row[metric]) for row in subset])
            low = np.asarray([float(row[low_key]) for row in subset])
            high = np.asarray([float(row[high_key]) for row in subset])
            color, label = styles[anchor]
            axis.plot(layers, values, color=color, linewidth=2.2, label=label)
            axis.fill_between(layers, low, high, color=color, alpha=0.14, linewidth=0)
            best = int(np.argmax(values))
            axis.scatter([layers[best]], [values[best]], color=color, s=30, zorder=3)
            axis.annotate(
                f"{values[best] * 100:.1f}% at L{layers[best]}",
                (layers[best], values[best]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                color="#333333",
                fontsize=9.5,
            )
        axis.axhline(0.25, color="#777777", linewidth=1.2, linestyle="--")
        axis.set_title(title, loc="left", fontsize=15)
        axis.set_xlabel("Shared residual readout L")
        axis.set_xlim(1, 64)
        axis.set_ylim(0.18, 0.95)
        axis.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("Held-out balanced accuracy")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Does the decision residual select among the four contextual option representations?",
        x=0.055,
        ha="left",
        fontsize=17,
    )
    fig.text(
        0.055,
        0.012,
        "Same-layer ridge matching; n=249 confirmation questions; four-way chance = 25%; shading = 95% stratified bootstrap CI.",
        fontsize=10.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.04, 0.06, 0.99, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
