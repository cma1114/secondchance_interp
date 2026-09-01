#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DESTINATIONS = (
    ("letter", "2P option letter"),
    ("content", "2P semantic wordpieces"),
    ("newline", "2P closing newline"),
)


def _matches(destination: str, kind: str) -> bool:
    return "_content_" in destination if kind == "content" else destination.endswith(f"_{kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--writes", type=Path, required=True)
    parser.add_argument("--scale", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    writes = json.loads(args.writes.read_text())
    scale = json.loads(args.scale.read_text())
    layers = writes["layers"]
    sources = writes["source_tokens"]
    layer_slot = {layer: index for index, layer in enumerate(layers)}
    source_slot = {source: index for index, source in enumerate(sources)}
    denominator = {
        (row["condition"], row["layer"], row["destination"]): row
        for row in scale["rows"]
        if row["split"] == "confirmation"
    }
    arrays = {}
    residual_arrays = {}
    for short_condition, condition in (("game", "incorrect_again"), ("neutral", "lost_again")):
        for kind, _label in DESTINATIONS:
            numerator = np.zeros((len(sources), len(layers)), dtype=np.float64)
            mixer = np.zeros_like(numerator)
            residual = np.zeros_like(numerator)
            counts = np.zeros_like(numerator)
            field = f"confirmation_mean_per_question_write_rms_{short_condition}"
            for cell in writes["all_cells"]:
                if not _matches(cell["destination"], kind):
                    continue
                key = (condition, cell["layer"], cell["destination"])
                denom = denominator[key]
                count = int(cell["confirmation_count"])
                row = source_slot[cell["source"]]
                column = layer_slot[cell["layer"]]
                numerator[row, column] += float(cell[field]) * count
                mixer[row, column] += float(denom["mean_complete_attention_write_rms"]) * count
                residual[row, column] += float(denom["mean_receiver_pre_layer_residual_rms"]) * count
                counts[row, column] += count
            arrays[(short_condition, kind)] = np.divide(
                numerator, mixer, out=np.zeros_like(numerator), where=mixer > 0
            )
            residual_arrays[(short_condition, kind)] = np.divide(
                numerator, residual, out=np.zeros_like(numerator), where=residual > 0
            )

    global_limit = max(float(array.max()) for array in arrays.values())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.86, wspace=0.24, hspace=0.18)
    fig.suptitle("Each feedback token's write relative to the receiver's complete attention update", fontsize=20, y=0.98)
    fig.text(
        0.5,
        0.935,
        "Held-out 249 questions; ratio of matched mean per-question RMS values. All panels share one scale.",
        ha="center",
        fontsize=12,
    )
    labels_base = [source.split(":", 1)[1].strip() for source in sources]
    for row, (condition, condition_label) in enumerate((("game", "Game"), ("neutral", "Neutral"))):
        for column, (kind, kind_label) in enumerate(DESTINATIONS):
            ax = axes[row, column]
            image = ax.imshow(
                100 * arrays[(condition, kind)],
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=0,
                vmax=100 * global_limit,
            )
            if row == 0:
                ax.set_title(kind_label, fontsize=14)
            if column == 0:
                ax.set_ylabel(condition_label, fontsize=14, fontweight="bold")
            ax.set_xticks(np.arange(len(layers)))
            ax.set_xticklabels(layers, rotation=45, ha="right")
            if row == 1:
                ax.set_xlabel("Ordinary-attention layer")
            labels = labels_base.copy()
            labels[3] = "incorrect" if condition == "game" else "lost"
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels if column == 0 else [])
            bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
            bar.set_label("Source-write RMS / complete-attention-write RMS (%)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "definition": "Ratio of mean source-specific write RMS to mean complete ordinary-attention output RMS at the same receiver tokens and layer. Contributions are vectors and can cancel, so this is a magnitude ratio, not a variance partition.",
        "maximum_attention_write_ratio": float(global_limit),
        "maximum_residual_ratio": float(max(array.max() for array in residual_arrays.values())),
        "arrays": {
            f"{condition}_{kind}": arrays[(condition, kind)].tolist()
            for condition, _ in (("game", "Game"), ("neutral", "Neutral"))
            for kind, _ in DESTINATIONS
        },
        "residual_arrays": {
            f"{condition}_{kind}": residual_arrays[(condition, kind)].tolist()
            for condition, _ in (("game", "Game"), ("neutral", "Neutral"))
            for kind, _ in DESTINATIONS
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
