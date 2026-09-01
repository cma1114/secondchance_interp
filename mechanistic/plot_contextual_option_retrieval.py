from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _curve(path: Path, anchor: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.load(path)[anchor]
    mean = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(5000, len(values)))
    bootstrap = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975), axis=0)
    return mean, low, high


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=True)
    layers = np.arange(1, 65)
    specs = [
        ("content_end", "A  Last token of option content"),
        ("line_end", "B  Option-closing newline"),
    ]
    for panel, (axis, (anchor, title)) in enumerate(zip(axes, specs)):
        for split, path, color, seed in [
            ("Discovery (n=251)", args.discovery, "#2F8FED", 20260810),
            ("Confirmation (n=249)", args.confirmation, "#F27F33", 20260811),
        ]:
            mean, low, high = _curve(path, anchor, seed)
            axis.plot(layers, mean, color=color, linewidth=2.0, label=split)
            axis.fill_between(layers, low, high, color=color, alpha=0.15, linewidth=0)
        axis.axhline(0.25, color="#777777", linewidth=1.2, linestyle="--")
        axis.set_title(title, loc="left", fontsize=15)
        axis.set_xlim(1, 64)
        axis.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
        axis.set_xlabel("Residual readout")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.grid(axis="x", visible=False)
        if panel == 0:
            axis.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("Same-content retrieval accuracy")
    axes[0].set_ylim(0, 1.0)
    fig.suptitle(
        "Can the same option content be identified after its letter and position are remapped?",
        x=0.055,
        ha="left",
        fontsize=17,
    )
    fig.text(
        0.055,
        0.015,
        "Paired original/remapped prompts; four-way chance = 0.25; shading = paired 95% bootstrap CI.",
        fontsize=10.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.04, 0.06, 0.99, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
