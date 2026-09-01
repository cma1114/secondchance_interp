#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LENSES = ("J-lens", "R-lens")
KINDS = (("letter", "2P option letters"), ("semantic", "2P semantic wordpieces"), ("newline", "2P closing newlines"))
CONDITIONS = (("incorrect_again", "Game", "#2878b5"), ("lost_again", "Neutral", "#e07a2f"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    layers = np.asarray(data["layers"])
    arrays = {}
    for lens in LENSES:
        for kind, _ in KINDS:
            for condition, _label, _color in CONDITIONS:
                arrays[(lens, kind, condition)] = np.asarray([
                    np.mean([
                        data["readouts"][lens][str(layer)]["positions"][f"R{rank}_{kind}"][condition]
                        ["signed_policy_coordinate"]
                        for rank in range(1, 5)
                    ])
                    for layer in layers
                ])
    low = min(float(values.min()) for values in arrays.values())
    high = max(float(values.max()) for values in arrays.values())
    margin = 0.04 * (high - low)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), sharex=True, sharey=True)
    fig.suptitle("Raw held-out 2P scores on discovery-frozen policy-token coordinates", fontsize=19)
    fig.text(
        0.5,
        0.925,
        "Positive points toward the evaluation-word incorrect-minus-lost profile. The 32-token coordinate is reselected on discovery at each layer.",
        ha="center",
        fontsize=11.5,
    )
    for row, lens in enumerate(LENSES):
        for column, (kind, title) in enumerate(KINDS):
            ax = axes[row, column]
            for condition, label, color in CONDITIONS:
                ax.plot(layers, arrays[(lens, kind, condition)], label=label, color=color, lw=2)
            ax.axhline(0, color="black", lw=0.8, alpha=0.5)
            ax.set_xlim(1, 64)
            ax.set_ylim(low - margin, high + margin)
            ax.set_xticks(np.arange(4, 65, 4))
            if row == 1:
                ax.set_xlabel("Layer")
            if row == 0:
                ax.set_title(title)
            if column == 0:
                ax.set_ylabel(f"{lens}\nraw signed score")
            ax.grid(alpha=0.2)
    axes[0, 2].legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
