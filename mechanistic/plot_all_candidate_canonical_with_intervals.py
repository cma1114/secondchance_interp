from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANKS = ("W1", "W2", "W3", "W4")
DISPLAY_RANKS = ("R1 (winner)", "R2", "R3", "R4")
CONDITIONS = ("Game", "Neutral")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--observational-results", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--winner-attenuation-figure", type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    with np.load(args.observational_results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    confirmation = np.asarray(
        [qid not in discovery_ids for qid in arrays["question_ids"].astype(str)]
    )
    blocks = arrays["ordinary_blocks_one_based"].astype(int)
    if blocks.tolist() != list(range(4, 65, 4)):
        raise RuntimeError(f"Expected complete L4--L64 inventory, got {blocks.tolist()}")

    rng = np.random.default_rng(args.seed)
    n = int(confirmation.sum())
    bootstrap_indices = rng.integers(0, n, size=(args.draws, n))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    condition_colors = {"Game": "#b13c2e", "Neutral": "#28658c"}
    offsets = {"Game": -0.08, "Neutral": 0.08}

    heldout = summary["subsets"]["confirmation_all"]["ranks"]
    for condition in CONDITIONS:
        rows = [heldout[rank]["advantage"][condition] for rank in RANKS]
        means = np.asarray([row["mean"] for row in rows])
        lows = np.asarray([row["ci"][0] for row in rows])
        highs = np.asarray([row["ci"][1] for row in rows])
        x = np.arange(4) + offsets[condition]
        axes[0, 0].errorbar(
            x, means, yerr=[means - lows, highs - means], fmt="o-",
            color=condition_colors[condition], capsize=3, label=condition,
        )
    axes[0, 0].axhline(0, color="#777", lw=1)
    axes[0, 0].set_xticks(range(4), DISPLAY_RANKS)
    axes[0, 0].set_title("A  Held-out matching-specific causal effects", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Candidate centered-advantage effect (logits)")
    axes[0, 0].legend(frameon=False)

    for split, color, marker in (
        ("discovery_all", "#8ab6d6", "o"),
        ("confirmation_all", "#125a8a", "s"),
    ):
        rows = [summary["subsets"][split]["ranks"][rank]["advantage"]["Game_minus_Neutral"] for rank in RANKS]
        means = np.asarray([row["mean"] for row in rows])
        lows = np.asarray([row["ci"][0] for row in rows])
        highs = np.asarray([row["ci"][1] for row in rows])
        axes[0, 1].errorbar(
            range(4), means, yerr=[means - lows, highs - means],
            fmt=marker + "-", color=color, capsize=3, label=split.split("_")[0].capitalize(),
        )
    axes[0, 1].axhline(0, color="#777", lw=1)
    axes[0, 1].set_xticks(range(4), DISPLAY_RANKS)
    axes[0, 1].set_title("B  Policy interaction by first-pass rank", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("Game minus Neutral effect (logits)")
    axes[0, 1].legend(frameon=False)

    rank_colors = ("#2878b5", "#f28e2b", "#2ca02c", "#d62728")
    for axis, metric, title in (
        (axes[1, 0], "attention_mass", "C  Held-out attention-mass difference"),
        (axes[1, 1], "projected_write_norm", "D  Held-out projected-write norm difference"),
    ):
        values = arrays[metric].astype(float)[:, :, confirmation, :]
        differences = values[0] - values[1]
        for rank, (label, color) in enumerate(zip(DISPLAY_RANKS, rank_colors)):
            rank_values = differences[:, :, rank]
            means = rank_values.mean(1)
            boot = rank_values[:, bootstrap_indices].mean(2)
            lows, highs = np.quantile(boot, (0.025, 0.975), axis=1)
            axis.plot(blocks, means, marker="o", color=color, label=label)
            axis.fill_between(blocks, lows, highs, color=color, alpha=0.11, linewidth=0)
        axis.axhline(0, color="#777", lw=1)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Ordinary-attention block")
        axis.set_ylabel("Game minus Neutral")
    axes[1, 1].legend(frameon=False, ncol=2)

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("All-candidate semantic relay: generic, graded, or winner-specific?", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)

    if args.winner_attenuation_figure is not None:
        attention = arrays["attention_mass"].astype(float)[:, :, confirmation, :]
        game_minus_neutral = attention[0] - attention[1]
        winner_specific = (
            game_minus_neutral[:, :, 0] - game_minus_neutral[:, :, 1:].mean(2)
        )
        means = winner_specific.mean(1) * 100
        boot = winner_specific[:, bootstrap_indices].mean(2) * 100
        lows, highs = np.quantile(boot, (0.025, 0.975), axis=1)

        direct_fig, axis = plt.subplots(figsize=(9.5, 4.8))
        axis.axvspan(48, 64, color="#f4c7b8", alpha=0.24, label="Persistent late effect")
        axis.axhline(0, color="#666", lw=1)
        axis.plot(blocks, means, color="#9f2d20", marker="o", lw=2.2)
        axis.fill_between(blocks, lows, highs, color="#9f2d20", alpha=0.18, linewidth=0)
        axis.set_title(
            "Game selectively attenuates the first-pass winner's matched-line read from L48 onward",
            fontweight="bold",
        )
        axis.set_xlabel("Ordinary-attention block")
        axis.set_ylabel("R1 attenuation relative to R2–R4\n(Game − Neutral, percentage points)")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, loc="lower left")
        direct_fig.tight_layout()
        args.winner_attenuation_figure.parent.mkdir(parents=True, exist_ok=True)
        direct_fig.savefig(args.winner_attenuation_figure, dpi=180)
        plt.close(direct_fig)


if __name__ == "__main__":
    main()
