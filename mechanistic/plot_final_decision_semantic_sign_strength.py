from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LETTERS = "ABCD"
COLORS = {"game": "#348ce8", "neutral": "#ed7d31"}


def _balanced_letter_bootstrap(
    values: np.ndarray,
    letters: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and CI with each observed W1 letter receiving equal weight."""
    groups = [np.flatnonzero(letters == letter) for letter in np.unique(letters)]
    if not groups:
        raise ValueError("No observations in panel")
    per_group = np.stack([values[group].mean(axis=0) for group in groups])
    mean = per_group.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, values.shape[1]), dtype=np.float32)
    for draw in range(draws):
        group_means = []
        for group in groups:
            sampled = group[rng.integers(0, len(group), size=len(group))]
            group_means.append(values[sampled].mean(axis=0))
        boot[draw] = np.stack(group_means).mean(axis=0)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    return mean, low, high


def plot(
    results_path: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
    output_path: Path,
    *,
    draws: int,
    seed: int,
) -> None:
    arrays = dict(np.load(results_path, allow_pickle=False))
    qids = arrays["question_ids"].astype(str).tolist()
    projections = arrays["natural_projection"]
    if projections.shape != (2, len(qids), 64):
        raise ValueError(f"Unexpected projection shape: {projections.shape}")

    baseline = json.loads(baseline_path.read_text())["results"]
    remapped = json.loads(remapped_baseline_path.read_text())["results"]
    w1 = np.asarray([LETTERS.index(baseline[qid]["answer"]) for qid in qids])
    w2_values = [remapped[qid].get("answer_original_content") for qid in qids]
    valid = np.asarray([value in LETTERS for value in w2_values])
    w2 = np.asarray([
        LETTERS.index(value) if value in LETTERS else -1 for value in w2_values
    ])

    conflict = valid & (w1 != w2)
    nonconflict = valid & (w1 == w2)
    panels = (
        (conflict, "positive", "A  Conflict: positive W1 activation"),
        (conflict, "negative", "B  Conflict: negative W1 activation"),
        (nonconflict, "positive", "C  Non-conflict: positive W1 activation"),
        (nonconflict, "negative", "D  Non-conflict: negative W1 activation"),
    )

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    layers = np.arange(1, 65)
    for panel_index, (ax, (mask, sign, title)) in enumerate(zip(axes.flat, panels)):
        for condition_index, condition in enumerate(("game", "neutral")):
            signed = projections[condition_index, mask]
            strength = np.maximum(signed, 0) if sign == "positive" else np.maximum(-signed, 0)
            mean, low, high = _balanced_letter_bootstrap(
                strength,
                w1[mask],
                seed=seed + panel_index * 10 + condition_index,
                draws=draws,
            )
            ax.plot(layers, mean, color=COLORS[condition], linewidth=2.1, label=condition.title())
            ax.fill_between(layers, low, high, color=COLORS[condition], alpha=0.20, linewidth=0)
        ax.set_title(f"{title}  (n={int(mask.sum())})", loc="left")
        ax.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.75)
        ax.legend(frameon=False, loc="upper left")

    axes[0, 0].set_ylabel("Projection strength (residual units)")
    axes[1, 0].set_ylabel("Projection strength (residual units)")
    axes[1, 0].set_xlabel("Post-block residual readout")
    axes[1, 1].set_xlabel("Post-block residual readout")
    fig.suptitle(
        "Positive and negative W1-semantic activation at the final decision position",
        fontsize=17,
    )
    fig.text(
        0.5,
        0.012,
        "Natural confirmation-set executions. Positive strength = max(projection, 0); "
        "negative strength = max(-projection, 0). Shading: 95% within-W1-letter bootstrap CI; "
        "letters weighted equally.",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    plot(
        args.results,
        args.baseline,
        args.remapped_baseline,
        args.output,
        draws=args.draws,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
