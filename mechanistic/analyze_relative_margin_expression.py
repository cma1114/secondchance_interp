"""Relative margin expression across models.

Descriptive analysis: compares the size of the natural Game-minus-Neutral
old-winner (R1) score adjustment against each model's own decision margins
(top-1 minus top-2 aggregated A-D logit) at the final decision position, on
the canonical non-remapped trajectory runs. The ratio of adjustment to margin
- not the absolute adjustment - is the quantity that orders the three models
the way their behavioral choice-rate effects do.

Evidence class: descriptive/observational. No margin is intervened on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

CELLS = (
    ("Qwen3.6-27B", "SimpleMC",
     "outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/run/simplemc/results.npz"),
    ("Qwen3.6-27B", "TriviaMC",
     "outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/run/triviamc/results.npz"),
    ("Seed-OSS-36B", "SimpleMC",
     "outputs/model_replications/seed_oss_36b_final_position_trajectories/run/simplemc/results.npz"),
    ("Seed-OSS-36B", "TriviaMC",
     "outputs/model_replications/seed_oss_36b_final_position_trajectories/run/triviamc/results.npz"),
    ("Gemma-4-31B", "SimpleMC",
     "outputs/model_replications/gemma4_31b_negative_model_comparison/simplemc/trajectories/run/results.npz"),
    ("Gemma-4-31B", "TriviaMC",
     "outputs/model_replications/gemma4_31b_negative_model_comparison/triviamc/trajectories/run/results.npz"),
)


def cell_quantities(direct_logits: np.ndarray, rank_order: np.ndarray) -> dict[str, np.ndarray]:
    """Per-question margins and R1 effects from one trajectory cell.

    direct_logits: (2, n, 4) natural final-position aggregated A-D logits,
    row 0 Game and row 1 Neutral. rank_order: (n, 4) displayed indices sorted
    by the model's own first-presentation score, best first.
    """
    if direct_logits.ndim != 3 or direct_logits.shape[0] != 2 or direct_logits.shape[2] != 4:
        raise ValueError(f"Unexpected direct_logits shape {direct_logits.shape}")
    if rank_order.shape != (direct_logits.shape[1], 4):
        raise ValueError(f"Unexpected rank_order shape {rank_order.shape}")
    ordered = np.sort(direct_logits, axis=-1)
    margins = ordered[..., -1] - ordered[..., -2]
    centered = direct_logits - direct_logits.mean(axis=-1, keepdims=True)
    by_rank = np.take_along_axis(centered, rank_order[None, :, :], axis=-1)
    r1_effect = by_rank[0, :, 0] - by_rank[1, :, 0]
    return {
        "margin_game": margins[0],
        "margin_neutral": margins[1],
        "r1_effect": r1_effect,
    }


def summarize_cell(q: dict[str, np.ndarray], rng: np.random.Generator, draws: int) -> dict[str, Any]:
    n = len(q["r1_effect"])
    index = rng.integers(0, n, size=(draws, n))

    def interval(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
        }

    adjustment = float(abs(np.mean(q["r1_effect"])))
    boot_adjustment = np.abs(q["r1_effect"][index].mean(axis=1))
    boot_margin_neutral = np.median(q["margin_neutral"][index], axis=1)
    boot_margin_game = np.median(q["margin_game"][index], axis=1)
    boot_ratio = boot_adjustment / boot_margin_neutral
    boot_flippable = (q["margin_neutral"][index] < boot_adjustment[:, None]).mean(axis=1)
    return {
        "n": int(n),
        "median_margin_game": {
            "value": float(np.median(q["margin_game"])), **interval(boot_margin_game)},
        "median_margin_neutral": {
            "value": float(np.median(q["margin_neutral"])), **interval(boot_margin_neutral)},
        "r1_adjustment_abs": {"value": adjustment, **interval(boot_adjustment)},
        "adjustment_over_median_neutral_margin": {
            "value": float(adjustment / np.median(q["margin_neutral"])), **interval(boot_ratio)},
        "fraction_neutral_margin_below_adjustment": {
            "value": float(np.mean(q["margin_neutral"] < adjustment)), **interval(boot_flippable)},
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    summary: dict[str, Any] = {
        "analysis": "relative_margin_expression",
        "evidence_class": "descriptive/observational; no causal margin intervention",
        "definition": {
            "endpoint": "natural non-remapped final-position aggregated A-D logits from each model's canonical trajectory run",
            "margin": "top-1 minus top-2 raw A-D logit per question per condition",
            "r1_adjustment": "absolute mean over questions of the Game-minus-Neutral within-question-centered logit at the model's own old rank 1",
            "ratio": "r1_adjustment divided by the median Neutral-condition margin",
            "flippable_fraction": "share of questions whose Neutral margin is below the mean adjustment; a proxy, since the true per-question adjustment varies",
            "bootstrap": f"question-level percentile bootstrap, seed {args.seed}, {args.draws} draws",
        },
        "cells": [],
    }
    for model, dataset, path in CELLS:
        with np.load(args.root / path, allow_pickle=False) as loaded:
            q = cell_quantities(
                loaded["direct_logits"].astype(np.float64),
                np.asarray(loaded["rank_order"], dtype=np.int64),
            )
        summary["cells"].append(
            {"model": model, "dataset": dataset, "source": path, **summarize_cell(q, rng, args.draws)}
        )
    return summary


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Relative margin expression across models",
        "",
        "Descriptive comparison of each model's natural Game-minus-Neutral R1 score",
        "adjustment against its own final-position decision margins, on the canonical",
        "non-remapped trajectory endpoint. Absolute adjustments are similar across",
        "models; the adjustment-to-margin ratio is what orders the models the way",
        "their behavioral choice-rate effects do. No margin is intervened on.",
        "",
        "| Model | Dataset | Median margin (Neutral) | R1 adjustment | Adjustment / margin | Margin < adjustment |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        mn = cell["median_margin_neutral"]
        adj = cell["r1_adjustment_abs"]
        ratio = cell["adjustment_over_median_neutral_margin"]
        flip = cell["fraction_neutral_margin_below_adjustment"]
        lines.append(
            f"| {cell['model']} | {cell['dataset']} "
            f"| {mn['value']:.2f} [{mn['ci_low']:.2f}, {mn['ci_high']:.2f}] "
            f"| {adj['value']:.2f} [{adj['ci_low']:.2f}, {adj['ci_high']:.2f}] "
            f"| {ratio['value']:.2f} [{ratio['ci_low']:.2f}, {ratio['ci_high']:.2f}] "
            f"| {flip['value']*100:.0f}% [{flip['ci_low']*100:.0f}, {flip['ci_high']*100:.0f}] |"
        )
    lines += [
        "",
        "The flippable fraction is a proxy: it compares each question's Neutral",
        "margin with the mean adjustment, while the true adjustment varies by",
        "question. Behavioral choice-rate gaps for context: Qwen large, Seed",
        "moderate, Gemma null; see the indexed behavioral reports.",
        "",
    ]
    path.write_text("\n".join(lines))


def make_figure(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = summary["cells"]
    labels = [f"{c['model'].split('-')[0]}\n{c['dataset']}" for c in cells]
    x = np.arange(len(cells))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    margins = [c["median_margin_neutral"]["value"] for c in cells]
    adjusts = [c["r1_adjustment_abs"]["value"] for c in cells]
    axes[0].bar(x - 0.2, margins, width=0.4, label="Median top-2 margin (Neutral)")
    axes[0].bar(x + 0.2, adjusts, width=0.4, label="R1 adjustment (|Game-Neutral|)")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("logits")
    axes[0].set_title("Adjustment vs decision margin")
    axes[0].legend()
    ratios = [c["adjustment_over_median_neutral_margin"] for c in cells]
    vals = [r["value"] for r in ratios]
    err = np.array([
        [r["value"] - r["ci_low"] for r in ratios],
        [r["ci_high"] - r["value"] for r in ratios],
    ])
    axes[1].bar(x, vals, yerr=err, capsize=3, color="tab:purple")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("adjustment / median margin")
    axes[1].set_title("Relative expression ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/model_replications/relative_margin_expression/analysis"))
    parser.add_argument(
        "--figure", type=Path,
        default=Path("figures/model_replications/relative_margin_expression.png"))
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--draws", type=int, default=10000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = analyze(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    write_report(summary, args.output_dir / "REPORT.md")
    make_figure(summary, args.figure)
    print(f"Wrote {args.output_dir} and {args.figure}")


if __name__ == "__main__":
    main()
