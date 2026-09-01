from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


KINDS = ("indent", "letter", "colon", "semantic", "newline")
TABLE_KINDS = ("letter", "semantic", "newline")
TASKS = ("incorrect_again", "lost_again")
TASK_LABELS = {"incorrect_again": "Game", "lost_again": "Neutral"}


def _clean_source(label: str, task: str | None = None) -> str:
    index, text = label.split(":", 1)
    names = {
        "0": "Your",
        "1": "answer₁",
        "2": "was",
        "3": (
            "incorrect"
            if task == "incorrect_again"
            else "lost"
            if task == "lost_again"
            else "incorrect / lost"
        ),
        "4": "period₁",
        "5": "Choose",
        "6": "the",
        "7": "answer₂",
        "8": "again",
        "9": "period₂",
    }
    return names.get(index, text.strip())


def _consensus_tokens(j_cell: dict[str, Any], r_cell: dict[str, Any], k: int = 4) -> list[str]:
    j = [row["token"].strip() for row in j_cell["confirmation_top"]]
    r = [row["token"].strip() for row in r_cell["confirmation_top"]]
    j_rank = {token.lower(): index for index, token in enumerate(j)}
    r_rank = {token.lower(): index for index, token in enumerate(r)}
    common = sorted(
        set(j_rank) & set(r_rank), key=lambda token: (j_rank[token] + r_rank[token], j_rank[token])
    )
    chosen: list[str] = []
    for key in common:
        token = j[j_rank[key]]
        if token and token.lower() not in {value.lower() for value in chosen}:
            chosen.append(token)
        if len(chosen) == k:
            return chosen
    for token in j + r:
        if token and token.lower() not in {value.lower() for value in chosen}:
            chosen.append(token)
        if len(chosen) == k:
            break
    return chosen


def plot(input_path: Path, output_path: Path) -> None:
    payload = json.loads(input_path.read_text())
    layers = payload["layers"]
    sources = payload["source_tokens"]
    cells = payload["cells"]
    lookup = {
        (cell["lens"], cell["task"], cell["source"], cell["destination_kind"], cell["layer"]): cell
        for cell in cells
    }

    # Raw magnitude is lens-independent; use one copy of each cell.
    matrices: dict[tuple[str, str], np.ndarray] = {}
    for task in TASKS:
        for kind in KINDS:
            matrix = np.empty((len(sources), len(layers)), dtype=float)
            for source_index, source in enumerate(sources):
                for layer_index, layer in enumerate(layers):
                    matrix[source_index, layer_index] = lookup[
                        ("J-lens", task, source, kind, layer)
                    ]["confirmation_mean_per_question_write_rms"]
            matrices[(task, kind)] = matrix
    vmax = max(float(matrix.max()) for matrix in matrices.values())

    figure = plt.figure(figsize=(20, 18), constrained_layout=False)
    grid = GridSpec(4, 5, figure=figure, height_ratios=(1, 1, 1.18, 1.18), hspace=0.38, wspace=0.18)
    images = []
    kind_titles = {
        "indent": "Line-leading space",
        "letter": "Option letter",
        "colon": "Colon",
        "semantic": "Semantic wordpieces",
        "newline": "Closing newline",
    }
    for task_index, task in enumerate(TASKS):
        for kind_index, kind in enumerate(KINDS):
            axis = figure.add_subplot(grid[task_index, kind_index])
            image = axis.imshow(
                matrices[(task, kind)], aspect="auto", interpolation="nearest", vmin=0, vmax=vmax, cmap="viridis"
            )
            images.append(image)
            axis.set_title(kind_titles[kind], fontsize=11)
            axis.set_xticks(range(len(layers)))
            axis.set_xticklabels(layers, rotation=90, fontsize=7)
            if kind_index == 0:
                axis.set_yticks(range(len(sources)))
                axis.set_yticklabels([_clean_source(source, task) for source in sources], fontsize=8)
                axis.set_ylabel(TASK_LABELS[task], fontsize=12, fontweight="bold")
            else:
                axis.set_yticks(range(len(sources)))
                axis.set_yticklabels([])
            if task_index == 1:
                axis.set_xlabel("Ordinary-attention layer", fontsize=9)

    colorbar = figure.colorbar(
        images[0], ax=figure.axes[:10], orientation="horizontal", fraction=0.025, pad=0.055, aspect=55
    )
    colorbar.set_label("Held-out mean per-question RMS of the exact source-specific write (one shared scale)")

    table_titles = {
        "letter": "2P option letter",
        "semantic": "2P semantic wordpieces",
        "newline": "2P closing newline",
    }
    for table_row, task in enumerate(TASKS, start=2):
        axis = figure.add_subplot(grid[table_row, :])
        axis.axis("off")
        rows = []
        # The first three tokens are identical before the prompts diverge and
        # serve as magnitude controls in the heatmaps. Keeping them out of the
        # semantic table leaves the policy-bearing and contextualized sources
        # readable at normal display size.
        for source in sources[3:]:
            values = [_clean_source(source, task)]
            for kind in TABLE_KINDS:
                discovery = [
                    lookup[("J-lens", task, source, kind, layer)][
                        "discovery_mean_per_question_write_rms"
                    ]
                    for layer in layers
                ]
                peak_layer = layers[int(np.argmax(discovery))]
                j_cell = lookup[("J-lens", task, source, kind, peak_layer)]
                r_cell = lookup[("R-lens", task, source, kind, peak_layer)]
                tokens = _consensus_tokens(j_cell, r_cell)
                stability = min(
                    j_cell["discovery_confirmation_readable_vocab_cosine"],
                    r_cell["discovery_confirmation_readable_vocab_cosine"],
                )
                rank_agreement = min(j_cell["confirmation_rank_to_rank_mean_write_cosine"])
                flags = []
                if stability < 0.8:
                    flags.append("split-unstable")
                if rank_agreement < 0.8:
                    flags.append("rank-varied")
                suffix = f" [{', '.join(flags)}]" if flags else ""
                values.append(
                    f"L{peak_layer} · RMS {j_cell['confirmation_mean_per_question_write_rms']:.4f}{suffix}\n"
                    + ", ".join(tokens)
                )
            rows.append(values)
        table = axis.table(
            cellText=rows,
            colLabels=("Evaluation source",) + tuple(table_titles[kind] for kind in TABLE_KINDS),
            loc="center",
            cellLoc="left",
            colLoc="left",
            colWidths=(0.14, 0.286, 0.286, 0.286),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.2)
        table.scale(1, 1.85)
        for (row, column), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#e8eef6")
            elif column == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f4f4f4")
        axis.set_title(
            f"{TASK_LABELS[task]}: J/R-consensus top tokens carried by each source at its discovery-selected peak layer",
            fontsize=12,
            fontweight="bold",
            pad=16,
        )

    figure.suptitle(
        "What each evaluation token writes into each second-presentation position",
        fontsize=20,
        y=0.995,
    )
    figure.text(
        0.5,
        0.974,
        "Top: exact write magnitude at every ordinary-attention layer. Bottom: held-out semantic readout of the exact write—not the complete 2P residual. R1–R4 are averaged only for the displayed semantic readout.",
        ha="center",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.006,
        "Peak layers are selected on 251 discovery questions; magnitudes and token lists use 249 held-out questions. [rank-varied] means at least one rank-specific write has cosine < .8 with the rank mean.",
        ha="center",
        fontsize=8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()
