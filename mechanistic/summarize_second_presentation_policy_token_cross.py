from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SOURCE_INDICES = (3, 4, 5)
SOURCE_TITLES = (
    "Literal evaluation word: incorrect / lost",
    "Evaluation-closing period",
    "Contextualized later token: Choose",
)
DESTINATIONS = ("letter", "content", "newline")
DESTINATION_LABELS = {
    "letter": "option letter",
    "content": "semantic wordpieces",
    "newline": "closing newline",
}
COLORS = {"letter": "#7B2CBF", "content": "#1982C4", "newline": "#E76F51"}


def destination_kind(label: str) -> str:
    suffix = label.split("_", 1)[1]
    return "content" if suffix.startswith("content_") else suffix


def aggregate(cells: list[dict], source: str, layer: int, destination: str, metric: str) -> float:
    selected = [
        cell
        for cell in cells
        if cell["source"] == source
        and cell["layer"] == layer
        and destination_kind(cell["destination"]) == destination
        and cell["eligible_for_selection"]
    ]
    weights = np.asarray([cell["confirmation_count"] for cell in selected], dtype=float)
    values = np.asarray([cell[metric] for cell in selected], dtype=float)
    if not len(values) or not weights.sum():
        return float("nan")
    return float(np.average(values, weights=weights))


def summarize(input_path: Path, output_json: Path, output_figure: Path) -> None:
    payload = json.loads(input_path.read_text())
    cells = payload["all_cells"]
    layers = payload["layers"]
    sources = [payload["source_tokens"][index] for index in SOURCE_INDICES]
    summary: dict[str, object] = {
        "definition": "Count-weighted mean across first-pass ranks and sufficiently represented relative semantic-wordpiece positions; structural token roles have all questions.",
        "questions": payload["questions"],
        "discovery_questions": payload["discovery_questions"],
        "confirmation_questions": payload["confirmation_questions"],
        "sources": {},
    }

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    for axis, source, title in zip(axes, sources, SOURCE_TITLES):
        source_rows: dict[str, object] = {}
        for destination in DESTINATIONS:
            discovery = np.asarray(
                [aggregate(cells, source, layer, destination, "discovery_write_contrast_rms") for layer in layers]
            )
            confirmation = np.asarray(
                [aggregate(cells, source, layer, destination, "confirmation_write_contrast_rms") for layer in layers]
            )
            # Freeze the descriptive peak on discovery; confirmation is used
            # only to estimate the magnitude at that prespecified layer.
            peak_index = int(np.nanargmax(discovery))
            source_rows[destination] = {
                "discovery_selected_peak_layer": int(layers[peak_index]),
                "discovery_peak_write_contrast_rms": float(discovery[peak_index]),
                "confirmation_write_contrast_rms_at_discovery_peak": float(
                    confirmation[peak_index]
                ),
                "confirmation_trajectory": confirmation.tolist(),
                "discovery_trajectory": discovery.tolist(),
            }
            axis.plot(
                layers,
                discovery,
                color=COLORS[destination],
                linewidth=1.2,
                linestyle="--",
                alpha=0.45,
            )
            axis.plot(
                layers,
                confirmation,
                color=COLORS[destination],
                linewidth=2.4,
                label=DESTINATION_LABELS[destination],
            )
        summary["sources"][source] = source_rows
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("ordinary-attention layer")
        axis.set_xticks(layers[::2])
        axis.grid(alpha=0.18, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Game–Neutral exact-write contrast (RMS)")
    axes[-1].legend(frameon=False, fontsize=9, loc="upper left")
    figure.suptitle(
        "Where feedback tokens write directly into second-presentation option lines",
        fontsize=16,
        y=1.02,
    )
    figure.text(
        0.5,
        -0.015,
        "Solid: held-out confirmation (249 questions). Dashed: discovery (251). Content averages exact per-wordpiece cells; it is not a whole-line average.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout()
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input, args.output_json, args.output_figure)


if __name__ == "__main__":
    main()
