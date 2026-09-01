from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASKS = ("incorrect_again", "lost_again")
TASK_LABELS = ("Game", "Neutral")
SPLITS = ("discovery", "confirmation")
LETTERS = tuple("ABCD")


def _bootstrap_mean(values: np.ndarray, seed: int, draws: int = 10_000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for start in range(0, draws, 250):
        stop = min(start + 250, draws)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, (0.025, 0.975))
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def _unit_center(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    return centered / np.maximum(np.linalg.norm(centered, axis=-1, keepdims=True), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.input.read_text())
    qids = result["question_ids"]
    scores = np.asarray(result["scores"], dtype=float)
    split_labels = np.asarray(result["split"])
    trusted = [
        json.loads(args.trusted_game.read_text())["results"],
        json.loads(args.trusted_neutral.read_text())["results"],
    ]

    summary: dict[str, dict] = {}
    for split_index, split in enumerate(SPLITS):
        mask = split_labels == split
        summary[split] = {}
        for task_index, task in enumerate(TASKS):
            cue = scores[task_index, mask, 0]
            final = scores[task_index, mask, 1]
            actual = np.asarray(
                [trusted[task_index][qid]["aggregated_ad_logits"] for qid in np.asarray(qids)[mask]],
                dtype=float,
            )
            cue_winner = cue.argmax(axis=-1)
            final_winner = final.argmax(axis=-1)
            actual_winner = actual.argmax(axis=-1)
            cue_actual = (cue_winner == actual_winner).astype(float)
            final_actual = (final_winner == actual_winner).astype(float)
            cue_prob = np.bincount(cue_winner, minlength=4) / len(cue_winner)
            actual_prob = np.bincount(actual_winner, minlength=4) / len(actual_winner)
            summary[split][task] = {
                **result["summaries"][split][task],
                "cue_winner_vs_actual_answer": _bootstrap_mean(
                    cue_actual, seed=7100 + split_index * 100 + task_index
                ),
                "reconstructed_final_winner_vs_actual_answer": _bootstrap_mean(
                    final_actual, seed=7200 + split_index * 100 + task_index
                ),
                "independent_marginal_winner_agreement": float(np.dot(cue_prob, actual_prob)),
                "maximum_reconstructed_final_score_error": float(np.max(np.abs(final - actual))),
            }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), sharex=True, sharey=True)
    colors = ("#2f6fdb", "#e07a2f", "#2f9d62", "#8b5fbf")
    for split_index, split in enumerate(SPLITS):
        mask = split_labels == split
        for task_index, task in enumerate(TASKS):
            ax = axes[split_index, task_index]
            cue = _unit_center(scores[task_index, mask, 0])
            final = _unit_center(scores[task_index, mask, 1])
            for letter_index, letter in enumerate(LETTERS):
                ax.scatter(
                    cue[:, letter_index],
                    final[:, letter_index],
                    s=10,
                    alpha=0.25,
                    linewidths=0,
                    color=colors[letter_index],
                    label=letter if split_index == 0 and task_index == 0 else None,
                )
            ax.plot((-1, 1), (-1, 1), color="#777777", linewidth=1, alpha=0.7)
            stats = summary[split][task]
            ax.text(
                0.03,
                0.97,
                (
                    f"top letter: {100 * stats['winner_agreement']['mean']:.1f}%\n"
                    f"pair order: {100 * stats['pairwise_order_agreement']['mean']:.1f}%\n"
                    f"centered r: {stats['pooled_centered_pearson']:.3f}"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 3},
            )
            ax.set_title(f"{TASK_LABELS[task_index]} — {split}")
            ax.grid(True, color="#dddddd", linewidth=0.6)
            ax.set_xlim(-1.02, 1.02)
            ax.set_ylim(-1.02, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("Cue-space A–D score (centered and normalized within question)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Final-position A–D score\n(centered and normalized within question)")
    fig.legend(loc="upper center", ncol=4, frameon=False, title="Letter", bbox_to_anchor=(0.5, 0.952))
    fig.suptitle("The answer-cue space already carries much of the final A–D ordering", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
