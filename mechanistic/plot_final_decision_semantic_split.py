from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/final_decision_semantic_ablation/confirmation/results.npz"
BASELINE = ROOT / "outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/baseline_results.json"
REMAPPED = ROOT / "outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/remapped_baseline_results.json"
OUTPUT = ROOT / "figures/qwen36_27b_simplemc_corrected/final_decision_semantic_ablation.png"

LETTERS = "ABCD"
COLORS = {"conflict": "#348ce8", "agreement": "#ed7d31"}


def letter_standardized_bootstrap(
    values: np.ndarray,
    strata: np.ndarray,
    *,
    seed: int,
    draws: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == value) for value in range(4)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Every W1 letter must occur in the plotted subset")
    means = np.empty((draws, values.shape[1]), dtype=np.float32)
    for draw in range(draws):
        group_means = [
            values[group[rng.integers(0, len(group), size=len(group))]].mean(axis=0)
            for group in groups
        ]
        means[draw] = np.mean(group_means, axis=0)
    return (
        np.mean([values[group].mean(axis=0) for group in groups], axis=0),
        np.quantile(means, 0.025, axis=0),
        np.quantile(means, 0.975, axis=0),
    )


def main() -> None:
    arrays = np.load(RESULTS, allow_pickle=False)
    qids = arrays["question_ids"].astype(str).tolist()
    projections = arrays["natural_projection"]
    baseline = json.loads(BASELINE.read_text())["results"]
    remapped = json.loads(REMAPPED.read_text())["results"]

    w1 = np.asarray([LETTERS.index(baseline[qid]["answer"]) for qid in qids])
    w2_text = [remapped[qid].get("answer_original_content") for qid in qids]
    valid = np.asarray([answer in LETTERS for answer in w2_text])
    w2 = np.asarray(
        [LETTERS.index(answer) if answer in LETTERS else -1 for answer in w2_text]
    )
    masks = {
        "conflict": valid & (w1 != w2),
        "agreement": valid & (w1 == w2),
    }

    layers = np.arange(1, projections.shape[-1] + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey="row")

    for column, (condition_index, condition) in enumerate(
        enumerate(("Game", "Neutral"))
    ):
        for row, (transform, ylabel) in enumerate(
            (
                (lambda x: x, "Signed projection onto W1 direction (residual units)"),
                (np.abs, "Absolute projection onto W1 direction (residual units)"),
            )
        ):
            ax = axes[row, column]
            for split_index, split in enumerate(("conflict", "agreement")):
                mask = masks[split]
                values = transform(projections[condition_index, mask])
                mean, low, high = letter_standardized_bootstrap(
                    values,
                    w1[mask],
                    seed=20260811 + 100 * column + 10 * row + split_index,
                )
                label = (
                    f"W1 ≠ W2 (n={mask.sum()})"
                    if split == "conflict"
                    else f"W1 = W2 (n={mask.sum()})"
                )
                ax.plot(layers, mean, color=COLORS[split], linewidth=2, label=label)
                ax.fill_between(
                    layers, low, high, color=COLORS[split], alpha=0.15, linewidth=0
                )

            if row == 0:
                ax.axhline(0, color="#777777", linewidth=0.8)
            ax.set_title(f"{'ABCD'[row * 2 + column]}  {condition}")
            if column == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
            ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.65)
            ax.legend(frameon=False, loc="upper left")

    axes[1, 0].set_xlabel("Post-block residual readout")
    axes[1, 1].set_xlabel("Post-block residual readout")
    fig.suptitle(
        "Final-decision projection onto the prior semantic answer (confirmation set)",
        fontsize=17,
    )
    fig.text(
        0.5,
        0.012,
        "Each A–D W1 letter receives equal weight; shading is the 95% within-letter bootstrap confidence interval.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)

    for condition_index, condition in enumerate(("Game", "Neutral")):
        for split in ("conflict", "agreement"):
            mask = masks[split]
            signed = projections[condition_index, mask, -1]
            absolute = np.abs(signed)
            print(
                f"{condition} {split}: n={mask.sum()}, "
                f"L64 signed={signed.mean():.4f}, abs={absolute.mean():.4f}"
            )
    print(OUTPUT)


if __name__ == "__main__":
    main()
