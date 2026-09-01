#!/usr/bin/env python3
"""Plot raw source-specific residual-write magnitude for Game and Neutral."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DESTINATION_KINDS = (
    ("letter", "2P option letter"),
    ("content", "2P semantic wordpieces"),
    ("newline", "2P closing newline"),
)


def destination_matches(destination: str, kind: str) -> bool:
    return "_content_" in destination if kind == "content" else destination.endswith(f"_{kind}")


def aggregate(data: dict, condition: str, kind: str) -> np.ndarray:
    layers = data["layers"]
    sources = data["source_tokens"]
    source_index = {source: index for index, source in enumerate(sources)}
    layer_index = {layer: index for index, layer in enumerate(layers)}
    numerator = np.zeros((len(sources), len(layers)), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    field = f"confirmation_mean_per_question_write_rms_{condition}"
    for cell in data["all_cells"]:
        if not destination_matches(cell["destination"], kind):
            continue
        count = cell["confirmation_count"]
        if count <= 0:
            continue
        row = source_index[cell["source"]]
        column = layer_index[cell["layer"]]
        numerator[row, column] += cell[field] * count
        denominator[row, column] += count
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    conditions = (("game", "Game"), ("neutral", "Neutral"))
    arrays = {
        (condition, kind): aggregate(data, condition, kind)
        for condition, _ in conditions
        for kind, _ in DESTINATION_KINDS
    }
    # A single scale is required for comparisons across both tasks and all
    # destination types. Smaller semantic-token writes should remain visibly
    # smaller rather than being independently rescaled.
    global_limit = max(float(array.max()) for array in arrays.values())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.86, wspace=0.24, hspace=0.18)
    fig.suptitle("Raw residual writes from each feedback token into second-presentation tokens", fontsize=20, y=0.98)
    fig.text(
        0.5,
        0.935,
        "Held-out 249 questions; mean per-question write RMS over R1–R4. Game and Neutral are shown separately.",
        ha="center",
        fontsize=12,
    )
    labels_base = [source.split(":", 1)[1].strip() for source in data["source_tokens"]]
    for row, (condition, condition_label) in enumerate(conditions):
        for column, (kind, kind_label) in enumerate(DESTINATION_KINDS):
            ax = axes[row, column]
            image = ax.imshow(arrays[(condition, kind)], aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=global_limit)
            if row == 0:
                ax.set_title(kind_label, fontsize=14)
            if column == 0:
                ax.set_ylabel(condition_label, fontsize=14, fontweight="bold")
            ax.set_xticks(np.arange(len(data["layers"])))
            ax.set_xticklabels(data["layers"], rotation=45, ha="right")
            if row == 1:
                ax.set_xlabel("Ordinary-attention layer")
            labels = labels_base.copy()
            labels[3] = "incorrect" if condition == "game" else "lost"
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels if column == 0 else [])
            if column == 0:
                ax.get_yticklabels()[3].set_fontweight("bold")
                ax.get_yticklabels()[4].set_fontweight("bold")
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
            colorbar.set_label("Mean write RMS")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
