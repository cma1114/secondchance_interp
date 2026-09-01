from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONDITIONS = ("Game", "Neutral")
RANKS = ("R1 (winner)", "R2", "R3", "R4")


def interval(values: np.ndarray, indices: np.ndarray) -> tuple[float, float, float]:
    means = values[indices].mean(1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=280036)
    args = parser.parse_args()

    with np.load(args.results, allow_pickle=False) as loaded:
        completed = loaded["completed"].astype(bool)
        question_ids = loaded["question_ids"].astype(str)
        if "ordinary_blocks_one_based" not in loaded.files:
            raise RuntimeError("Results do not declare their measured block inventory")
        blocks = loaded["ordinary_blocks_one_based"].astype(int)
        attention = loaded["attention_mass"].astype(float)
    if not completed.all() or len(question_ids) != 500:
        raise RuntimeError("Expected a complete 500-question checkpoint")
    expected_blocks = np.arange(4, 65, 4)
    if not np.array_equal(blocks, expected_blocks):
        raise RuntimeError(
            f"Absolute-attention figure requires complete blocks 4--64; got {blocks.tolist()}"
        )
    if attention.shape != (2, len(blocks), 500, 4):
        raise RuntimeError(f"Unexpected attention-mass shape: {attention.shape}")

    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    confirmation = np.asarray([qid not in discovery_ids for qid in question_ids])
    if int(confirmation.sum()) != 249:
        raise RuntimeError("Held-out confirmation split is not 249 questions")
    values = attention[:, :, confirmation, :]
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0, int(confirmation.sum()), size=(args.draws, int(confirmation.sum()))
    )

    means = np.empty((2, len(blocks), 4), dtype=float)
    lows = np.empty_like(means)
    highs = np.empty_like(means)
    rows: list[list[object]] = []
    for condition in range(2):
        for block in range(len(blocks)):
            for rank in range(4):
                mean, low, high = interval(
                    values[condition, block, :, rank], bootstrap_indices
                )
                means[condition, block, rank] = mean
                lows[condition, block, rank] = low
                highs[condition, block, rank] = high
                rows.append(
                    [
                        RANKS[rank],
                        int(blocks[block]),
                        CONDITIONS[condition],
                        int(confirmation.sum()),
                        mean,
                        low,
                        high,
                    ]
                )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["first_pass_rank", "block", "condition", "n", "mean", "ci_low", "ci_high"])
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Game": "#b13c2e", "Neutral": "#28658c"}
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharex=True, sharey=True)
    for rank, axis in enumerate(axes.flat):
        for condition, label in enumerate(CONDITIONS):
            axis.plot(
                blocks,
                means[condition, :, rank],
                marker="o",
                linewidth=2,
                color=colors[label],
                label=label,
            )
            axis.fill_between(
                blocks,
                lows[condition, :, rank],
                highs[condition, :, rank],
                color=colors[label],
                alpha=0.14,
                linewidth=0,
            )
        axis.axvline(28, color="#888888", linewidth=1, linestyle=":")
        axis.axvline(36, color="#888888", linewidth=1, linestyle=":")
        axis.set_title(RANKS[rank], loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xlabel("Ordinary-attention block")
    axes[1, 1].set_xlabel("Ordinary-attention block")
    axes[0, 0].set_ylabel("Matched-line attention mass")
    axes[1, 0].set_ylabel("Matched-line attention mass")
    fig.suptitle(
        "Absolute attention from repeated options to matching original lines (held-out)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
