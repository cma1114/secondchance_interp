#!/usr/bin/env python3
"""Plot within-condition J/R-lens alignment between 2P and feedback residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RECEIVERS = (
    "R1_line", "R1_newline", "R2_line", "R2_newline",
    "R3_line", "R3_newline", "R4_line", "R4_newline",
    "choice_cue_space", "final_decision",
)
CONDITIONS = (("incorrect_again", "Game"), ("lost_again", "Neutral"))
LENSES = ("J-lens", "R-lens")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    arrays = {}
    for condition, _ in CONDITIONS:
        for lens in LENSES:
            arrays[(condition, lens)] = np.asarray([
                [
                    data["readouts"][lens][str(layer)]["confirmation"][receiver]
                    ["conditions"][condition]["full_vocab_cosine_to_own_sources"]
                    ["evaluation_word"]
                    for layer in data["layers"]
                ]
                for receiver in RECEIVERS
            ])
    limit = max(float(np.abs(array).max()) for array in arrays.values())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle("Within-task J/R-lens alignment of 2P with its own feedback-word state", fontsize=18)
    for row, (condition, condition_label) in enumerate(CONDITIONS):
        for column, lens in enumerate(LENSES):
            ax = axes[row, column]
            image = ax.imshow(arrays[(condition, lens)], aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-limit, vmax=limit, extent=(0.5, 64.5, len(RECEIVERS)-0.5, -0.5))
            ax.set_title(f"{condition_label}: {lens}")
            ax.set_xticks(np.arange(4, 65, 4))
            if row == 1:
                ax.set_xlabel("Layer")
            ax.set_yticks(range(len(RECEIVERS)))
            ax.set_yticklabels([name.replace("_", " ") for name in RECEIVERS] if column == 0 else [])
    fig.colorbar(image, ax=axes, label="Within-task full-vocabulary cosine")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
