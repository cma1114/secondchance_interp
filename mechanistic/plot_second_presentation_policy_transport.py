from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RECEIVERS = (
    "R1_line",
    "R1_newline",
    "R2_line",
    "R2_newline",
    "R3_line",
    "R3_newline",
    "R4_line",
    "R4_newline",
    "choice_cue_space",
    "final_decision",
)


def _trajectory(
    data: dict, lens: str, position: str, source: str, split: str = "confirmation"
) -> np.ndarray:
    return np.asarray(
        [
            data["readouts"][lens][str(layer)][split][position][
                "full_vocab_cosine_to_sources"
            ][source]
            for layer in data["layers"]
        ],
        dtype=float,
    )


def plot(
    trajectory_path: Path,
    transport_path: Path,
    source_trajectory_path: Path,
    output_path: Path,
) -> None:
    trajectory = json.loads(trajectory_path.read_text())
    transport = json.loads(transport_path.read_text())
    source_trajectory = json.loads(source_trajectory_path.read_text())
    layers = np.asarray(trajectory["layers"], dtype=int)
    labels = [name.replace("_", " ") for name in RECEIVERS]

    heatmaps = {}
    for lens in ("J-lens", "R-lens"):
        heatmaps[lens] = np.stack(
            [_trajectory(trajectory, lens, position, "evaluation_word") for position in RECEIVERS]
        )
    limit = max(float(np.abs(values).max()) for values in heatmaps.values())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for axis, lens in zip(axes[0], ("J-lens", "R-lens")):
        image = axis.imshow(
            heatmaps[lens],
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            extent=(layers[0] - 0.5, layers[-1] + 0.5, len(RECEIVERS) - 0.5, -0.5),
        )
        axis.set_title(f"{lens}: complete residual alignment with feedback word")
        axis.set_xlabel("Layer")
        axis.set_yticks(range(len(RECEIVERS)), labels)
        axis.set_xticks(np.arange(4, 65, 4))
    fig.colorbar(image, ax=axes[0], label="Full-vocabulary cosine (Game − Neutral)")

    axis = axes[1, 0]
    colors = {"evaluation_word": "#d62728", "evaluation_period": "#9467bd", "action_choose": "#1f77b4"}
    display = {
        "evaluation_word": "incorrect / lost token",
        "evaluation_period": "evaluation-closing period",
        "action_choose": "contextualized ‘Choose’ token",
    }
    newline_roles = ("R1_newline", "R2_newline", "R3_newline", "R4_newline")
    for source in colors:
        curves = np.stack(
            [_trajectory(trajectory, "J-lens", role, source) for role in newline_roles]
        )
        axis.plot(layers, curves.mean(0), color=colors[source], linewidth=2.5, label=display[source])
        axis.fill_between(
            layers, curves.min(0), curves.max(0), color=colors[source], alpha=0.12, linewidth=0
        )
    axis.axhline(0, color="0.35", linewidth=0.8)
    axis.set_title("J-lens: four 2P option newlines (mean and range)")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Full-vocabulary cosine (Game − Neutral)")
    axis.set_xticks(np.arange(4, 65, 4))
    axis.legend(frameon=False, fontsize=9)

    axis = axes[1, 1]
    source_labels = {
        "3: incorrect| lost": "incorrect / lost token",
        "4:.": "evaluation-closing period",
        "5: Choose": "contextualized ‘Choose’ token",
    }
    source_colors = {
        "3: incorrect| lost": colors["evaluation_word"],
        "4:.": colors["evaluation_period"],
        "5: Choose": colors["action_choose"],
    }
    for source, label in source_labels.items():
        rows = [
            row
            for row in source_trajectory["rows"]
            if row["source"] == source and row["receiver_role"] in newline_roles
        ]
        confirmation = np.asarray(
            [
                np.mean(
                    [
                        row["confirmation_contrast_rms"]
                        for row in rows
                        if row["layer"] == layer
                    ]
                )
                for layer in source_trajectory["layers"]
            ]
        )
        discovery = np.asarray(
            [
                np.mean(
                    [row["discovery_contrast_rms"] for row in rows if row["layer"] == layer]
                )
                for layer in source_trajectory["layers"]
            ]
        )
        axis.plot(
            source_trajectory["layers"],
            confirmation,
            color=source_colors[source],
            linewidth=2.5,
            label=label,
        )
        axis.plot(
            source_trajectory["layers"],
            discovery,
            color=source_colors[source],
            linewidth=1.1,
            linestyle="--",
            alpha=0.55,
        )
    axis.axhline(0, color="0.35", linewidth=0.8)
    axis.set_xticks(np.arange(4, 65, 4))
    axis.set_xlabel("Ordinary-attention layer")
    axis.set_title("Exact feedback-token writes into 2P option newlines")
    axis.set_ylabel("Game − Neutral write RMS")
    axis.legend(title="solid: held-out; dashed: discovery", frameon=False, fontsize=8)

    fig.suptitle(
        "Where the incorrect-versus-lost policy appears and how feedback writes reach 2P",
        fontsize=17,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--source-trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.trajectory, args.transport, args.source_trajectory, args.output)


if __name__ == "__main__":
    main()
