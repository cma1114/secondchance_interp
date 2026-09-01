from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    **{
        f"R{rank}_{kind}": f"R{rank} {label}"
        for rank in range(1, 5)
        for kind, label in (
            ("letter", "letter"),
            ("semantic", "semantic words"),
            ("newline", "newline"),
        )
    },
    "choice_cue_space": "Post-list answer cue",
    "final_decision": "Final decision",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens", choices=("J-lens", "R-lens"), default="J-lens")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    lens_index = ("J-lens", "R-lens").index(args.lens)
    mean = np.asarray(payload["mean"])[:, lens_index]
    lower = np.asarray(payload["lower"])[:, lens_index]
    upper = np.asarray(payload["upper"])[:, lens_index]
    layers = np.asarray(payload["layers"])
    positions = payload["position_names"]

    colors = ("#2878b5", "#d95319")
    figure, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.ravel()
    for position_index, position in enumerate(positions):
        axis = axes[position_index]
        for condition_index, condition in enumerate(("Game", "Neutral")):
            linestyle = "-" if condition == "Game" else "--"
            for family_index, family in enumerate(
                (
                    "incorrect/failed/mistake/wrong",
                    "lost/again/resend/repeat",
                )
            ):
                label = f"{family} · {condition}"
                values = mean[condition_index, :, position_index, family_index]
                lo = lower[condition_index, :, position_index, family_index]
                hi = upper[condition_index, :, position_index, family_index]
                axis.fill_between(layers, lo, hi, color=colors[family_index], alpha=0.08)
                axis.plot(
                    layers,
                    values,
                    color=colors[family_index],
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=label,
                )
        axis.axhline(0, color="0.65", linewidth=0.7)
        axis.set_title(LABELS[position], fontsize=10)
        axis.set_xticks((1, 16, 32, 48, 64))
        axis.grid(axis="y", color="0.88", linewidth=0.6)

    for axis in axes[len(positions) :]:
        axis.axis("off")
    for row in range(4):
        axes[row * 4].set_ylabel("Mean requested-word score")
    for axis in axes[12:14]:
        axis.set_xlabel("Post-layer residual")
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower right",
        bbox_to_anchor=(0.965, 0.065),
        frameon=False,
        ncol=1,
    )
    figure.suptitle(
        f"Requested-word activation in complete 2P destination residuals ({args.lens})",
        fontsize=16,
    )
    figure.text(
        0.5,
        0.018,
        "Solid = Game; dashed = Neutral. Shading: 95% normal-approximation intervals over 249 held-out questions. Shared y-scale.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.03, 0.05, 0.98, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
