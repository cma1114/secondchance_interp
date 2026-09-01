#!/usr/bin/env python3
"""Create the canonical score-source attribution figure from compact outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(values), size=(5000, len(values)))
    sampled = values[draws].mean(axis=1)
    return values.mean(axis=0), np.quantile(sampled, 0.025, axis=0), np.quantile(sampled, 0.975, axis=0)


def source_trajectory(summary: dict, condition: str, source: str) -> tuple[np.ndarray, np.ndarray]:
    rows = summary["summaries"]["content_mean"]["old_unique"]["ordinary_attention_sources"]
    layers = np.asarray(sorted(int(layer) for layer in rows))
    values = np.asarray(
        [rows[str(layer)][source][condition]["confirmation_correlation"] for layer in layers]
    )
    return layers, values


def component_trajectory(summary: dict, condition: str, target: str, component: str) -> np.ndarray:
    rows = summary["summaries"]["content_mean"][target]["components"]
    return np.asarray(
        [rows[str(layer)][component][condition]["confirmation_correlation"] for layer in range(1, 65)]
    )


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    arrays = np.load(args.arrays, allow_pickle=False)
    discovery_qids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    question_ids = arrays["question_ids"].astype(str)
    confirmation = np.asarray([qid not in discovery_qids for qid in question_ids])
    source_values = arrays["derived_source_values"].astype(np.float32)
    component_values = arrays["component_values"].astype(np.float32)
    rng = np.random.default_rng(49052)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.075, top=0.88, wspace=0.24, hspace=0.32)
    fig.suptitle(
        "Where first-pass and fresh candidate scores enter second-presentation semantic states",
        fontsize=20,
        y=0.975,
    )
    fig.text(
        0.5,
        0.935,
        "Held-out 249 questions. Game and Neutral are shown separately before their comparison.",
        ha="center",
        fontsize=12,
    )

    source_specs = (
        ("first_matching_line", "matching 1P line"),
        ("first_other_lines", "other three 1P lines"),
        ("first_answer_boundary", "1P answer boundary"),
        ("second_matching_line", "current 2P line"),
        ("second_other_lines", "other 2P lines"),
    )
    source_colors = plt.get_cmap("tab10").colors
    for column, condition in enumerate(("Game", "Neutral")):
        ax = axes[0, column]
        for index, (source, label) in enumerate(source_specs):
            layers, values = source_trajectory(summary, condition, source)
            ax.plot(layers, values, marker="o", markersize=3.5, linewidth=1.8, label=label, color=source_colors[index])
        ax.axhline(0.0, color="0.45", linewidth=0.8)
        ax.axvline(32, color="0.55", linewidth=0.8, linestyle=":")
        ax.set_title(f"{condition}: exact attention sources of old score")
        ax.set_xlabel("Ordinary-attention layer")
        ax.set_ylabel("Correlation with unique 1P score")
        ax.set_xticks(np.arange(4, 65, 8))
        ax.set_ylim(-0.2, 0.45)
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=9, loc="upper left")

    # Exact matching-line write at the strongest covariance layer.
    ax = axes[0, 2]
    # Dimensions: condition, question, ordinary layer, rank, summary, target, source.
    # L32 -> stored layer index 7; content mean 0; old target 0; first matching source 3.
    for condition_index, (condition, offset, color) in enumerate(
        (("Game", -0.11, source_colors[0]), ("Neutral", 0.11, source_colors[1]))
    ):
        values = source_values[condition_index, confirmation, 7, :, 0, 0, 3]
        mean, low, high = bootstrap_mean(values, rng)
        x = np.arange(4) + offset
        ax.errorbar(x, mean, yerr=[mean - low, high - mean], fmt="o", capsize=4, linewidth=2, label=condition, color=color)
    ax.axhline(0.0, color="0.45", linewidth=0.8)
    ax.set_xticks(np.arange(4), ["R1", "R2", "R3", "R4"])
    ax.set_title("Layer 32: matching 1P line writes old rank")
    ax.set_ylabel("Normalized old-score write")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)

    layers = np.arange(1, 65)
    for column, condition in enumerate(("Game", "Neutral")):
        ax = axes[1, column]
        old_mixer = component_trajectory(summary, condition, "old_unique", "mixer")
        old_mlp = component_trajectory(summary, condition, "old_unique", "mlp")
        fresh_mixer = component_trajectory(summary, condition, "fresh_unique", "mixer")
        fresh_mlp = component_trajectory(summary, condition, "fresh_unique", "mlp")
        ax.plot(layers, old_mixer, label="old score: mixer", color=source_colors[0], linewidth=1.8)
        ax.plot(layers, old_mlp, label="old score: MLP", color=source_colors[1], linewidth=1.8)
        ax.plot(layers, fresh_mixer, label="fresh score: mixer", color=source_colors[0], linewidth=1.5, linestyle="--")
        ax.plot(layers, fresh_mlp, label="fresh score: MLP", color=source_colors[1], linewidth=1.5, linestyle="--")
        ordinary = np.arange(4, 65, 4)
        ax.scatter(ordinary, old_mixer[ordinary - 1], s=14, color=source_colors[0], zorder=3)
        ax.scatter(ordinary, fresh_mixer[ordinary - 1], s=14, facecolors="none", edgecolors=source_colors[0], zorder=3)
        ax.axhline(0.0, color="0.45", linewidth=0.8)
        ax.axvline(49, color="0.55", linewidth=0.8, linestyle=":")
        ax.set_title(f"{condition}: layer writes aligned with each score")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Correlation with score target")
        ax.set_xlim(1, 64)
        ax.set_ylim(-0.2, 0.45)
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=9, loc="upper left")

    # Absolute within-task MLP49 write, with the comparison visually secondary.
    ax = axes[1, 2]
    # L49 -> index 48; last content 1; old target 0; MLP 1.
    for condition_index, (condition, offset, color) in enumerate(
        (("Game", -0.11, source_colors[0]), ("Neutral", 0.11, source_colors[1]))
    ):
        values = component_values[condition_index, confirmation, 48, :, 1, 0, 1]
        mean, low, high = bootstrap_mean(values, rng)
        x = np.arange(4) + offset
        ax.errorbar(x, mean, yerr=[mean - low, high - mean], fmt="o", capsize=4, linewidth=2, label=condition, color=color)
    ax.axhline(0.0, color="0.45", linewidth=0.8)
    ax.set_xticks(np.arange(4), ["R1", "R2", "R3", "R4"])
    ax.set_title("MLP 49: old-rank write in each task")
    ax.set_ylabel("Normalized old-score write")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
