from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RANKS = ("R1", "R2", "R3", "R4")
COLORS = {"natural": "#2878B5", "swapped": "#D95F02", "restored": "#5B4B9A"}


def points(ax, rows, x, offset, label, color, marker):
    means = np.asarray([rows[rank]["mean"] for rank in RANKS])
    cis = np.asarray([rows[rank]["ci"] for rank in RANKS])
    ax.errorbar(
        x + offset, means, yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
        fmt=marker, color=color, capsize=3.5, linewidth=1.7, markersize=5.5, label=label,
    )


def main(args: argparse.Namespace) -> None:
    data = json.loads(args.summary.read_text())
    conf = data["splits"]["confirmation"]["conditions"]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), constrained_layout=True)
    x = np.arange(4)

    for ax, condition, swapped_name in [
        (axes[0, 0], "Game", "Neutral policy write"),
        (axes[0, 1], "Neutral", "Game policy write"),
    ]:
        row = conf[condition]
        points(ax, row["natural_matching_specific_rank_effect"], x, -0.09,
               f"Natural {condition} policy", COLORS["natural"], "o")
        points(ax, row["policy_swapped_matching_specific_rank_effect"], x, +0.09,
               swapped_name, COLORS["swapped"], "s")
        ax.axhline(0, color="0.35", linewidth=1)
        ax.set_xticks(x, RANKS)
        ax.set_ylabel("Effect of blocking matching 1P→2P route\n(logits; matching minus control)")
        ax.set_title(f"{'A' if condition == 'Game' else 'B'}  {condition}: how retrieved rank is used")
        ax.grid(axis="y", color="0.88", linewidth=0.8)
        ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    metrics = [
        ("policy_swap_mlp49_bivalent_effect", "MLP 49 write"),
        ("policy_swap_final_bivalent_effect", "Final evidence"),
        ("after_mlp49_restore_final_bivalent_effect", "Final after MLP 49 restore"),
    ]
    for ci, condition in enumerate(("Game", "Neutral")):
        rows = [conf[condition][key] for key, _ in metrics]
        means = np.asarray([row["mean"] for row in rows])
        cis = np.asarray([row["ci"] for row in rows])
        offset = -0.10 if condition == "Game" else +0.10
        color = COLORS["natural"] if condition == "Game" else COLORS["swapped"]
        marker = "o" if condition == "Game" else "s"
        ax.errorbar(
            np.arange(3) + offset, means,
            yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
            fmt=marker, color=color, capsize=3.5, linewidth=1.7, markersize=5.5,
            label=condition,
        )
    ax.axhline(0, color="0.35", linewidth=1)
    ax.set_xticks(np.arange(3), [label for _, label in metrics])
    ax.set_ylabel("Policy-swap change in bivalent rank index\nR4 − mean(R1,R2), logits")
    ax.set_title("C  Policy effect and MLP-49 mediation")
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    scenario_labels = ["Natural", "Policy\nswapped", "Swap + natural\nMLP 49 restored"]
    for condition_index, condition in enumerate(("Game", "Neutral")):
        choice = conf[condition]["conflict_W1_choice"]
        rows = [choice["natural"], choice["policy_swapped"], choice["policy_swapped_mlp49_restored"]]
        means = np.asarray([row["mean"] * 100 for row in rows])
        cis = np.asarray([np.asarray(row["ci"]) * 100 for row in rows])
        offset = -0.10 if condition == "Game" else +0.10
        color = COLORS["natural"] if condition == "Game" else COLORS["swapped"]
        marker = "o" if condition == "Game" else "s"
        ax.errorbar(
            np.arange(3) + offset, means,
            yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
            fmt=marker, color=color, capsize=3.5, linewidth=1.7, markersize=5.5,
            label=condition,
        )
    ax.set_xticks(np.arange(3), scenario_labels)
    ax.set_ylabel("Conflict-trial W1 choice (%)")
    ax.set_title("D  Does the causal chain reach behavior?")
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.legend(frameon=False)

    fig.suptitle("Causal policy × retrieved-rank factorial (frozen confirmation split)", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
