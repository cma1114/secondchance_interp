from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analyze_baseline_mixer_function import (
    _bootstrap_indices,
    _entropy,
    _spread,
    _summary,
    _winner_advantage,
)


CONDITIONS = ("Baseline", "Game", "Neutral")
RANKS = ("Rank 1", "Rank 2", "Rank 3", "Rank 4")
COLORS = ("#555555", "#1689d8", "#e66b19")


def _rank(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    return np.take_along_axis(centered, order[:, None, :], axis=-1)


def _ci_record(values: np.ndarray, bootstrap: np.ndarray):
    point, low, high = _summary(values, bootstrap)
    return {
        "estimate": float(point),
        "ci": [float(low), float(high)],
    }


def _plot(path: Path, immediate_stats, causal_stats):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    x = np.arange(4)
    for condition, color, stats in zip(CONDITIONS, COLORS, immediate_stats):
        point, low, high = stats
        axes[0].plot(x, point, marker="o", lw=2.5, color=color, label=condition)
        axes[0].fill_between(x, low, high, color=color, alpha=0.18)
    axes[0].set_title("A  Immediate Mixer-56 JLens write", loc="left", weight="bold")
    axes[0].set_ylabel("Centered A–D contribution (score units)")
    axes[0].legend(frameon=False)

    for condition, color, stats in zip(CONDITIONS, COLORS, causal_stats):
        point, low, high = stats
        axes[1].plot(x, point, marker="o", lw=2.5, color=color, label=condition)
        axes[1].fill_between(x, low, high, color=color, alpha=0.18)
    axes[1].set_title("B  Within-condition mean-ablation", loc="left", weight="bold")
    axes[1].set_ylabel("Change in final canonical logit")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.axhline(0, color="#555", lw=1, ls="--")
        axis.set_xticks(x, RANKS)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        -0.01,
        "Ranks are fixed by each question's natural Baseline logits; bands are paired, Baseline-letter-stratified 95% bootstrap CIs.",
        ha="center",
        fontsize=10.5,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def analyze(
    baseline_path: Path,
    conditions_path: Path,
    metadata_path: Path,
    output: Path,
    figure: Path,
    draws: int,
    seed: int,
):
    output.mkdir(parents=True, exist_ok=True)
    with np.load(baseline_path, allow_pickle=False) as loaded:
        baseline_data = {key: loaded[key] for key in loaded.files}
    with np.load(conditions_path, allow_pickle=False) as loaded:
        condition_data = {key: loaded[key] for key in loaded.files}
    if baseline_data["question_ids"].astype(str).tolist() != condition_data["question_ids"].astype(str).tolist():
        raise ValueError("Baseline and follow-up question orders differ")
    if not np.all(condition_data["completed"]):
        raise ValueError("Follow-up run is incomplete")

    baseline_logits = baseline_data["natural_baseline_logits"].astype(float)
    order = np.argsort(-baseline_logits, axis=1)
    baseline_answers = np.argmax(baseline_logits, axis=1)
    bootstrap = _bootstrap_indices(baseline_answers, draws, seed)
    natural = np.stack(
        [
            baseline_logits,
            condition_data["natural_logits"][0].astype(float),
            condition_data["natural_logits"][1].astype(float),
        ],
        axis=1,
    )
    ablated = np.stack(
        [
            baseline_data["baseline_mean_ablation_logits"][:, 0].astype(float),
            condition_data["mean_ablation_logits"][0].astype(float),
            condition_data["mean_ablation_logits"][1].astype(float),
        ],
        axis=1,
    )
    immediate = np.stack(
        [
            baseline_data["baseline_immediate_jlens_write"][:, 0].astype(float),
            condition_data["immediate_jlens_write"][0].astype(float),
            condition_data["immediate_jlens_write"][1].astype(float),
        ],
        axis=1,
    )
    immediate_ranked = _rank(immediate, order)
    causal_ranked = _rank(ablated - natural, order)
    immediate_stats = [_summary(immediate_ranked[:, i], bootstrap) for i in range(3)]
    causal_stats = [_summary(causal_ranked[:, i], bootstrap) for i in range(3)]

    repeated_order = np.repeat(order[:, None, :], 3, axis=1)
    natural_answer = np.argmax(natural, axis=2)
    ablated_answer = np.argmax(ablated, axis=2)
    natural_spread = _spread(natural)
    ablated_spread = _spread(ablated)
    natural_advantage = _winner_advantage(natural, repeated_order)
    ablated_advantage = _winner_advantage(ablated, repeated_order)
    natural_entropy = _entropy(natural)
    ablated_entropy = _entropy(ablated)
    metrics = {
        "answer_change_rate": (ablated_answer != natural_answer).astype(float),
        "spread_change": ablated_spread - natural_spread,
        "winner_advantage_change": ablated_advantage - natural_advantage,
        "entropy_change": ablated_entropy - natural_entropy,
    }
    natural_switch = (natural_answer != baseline_answers[:, None]).astype(float)
    ablated_switch = (ablated_answer != baseline_answers[:, None]).astype(float)
    metrics["switch_rate_change"] = ablated_switch - natural_switch

    summary = {
        "n_confirmation": int(len(baseline_logits)),
        "bootstrap_draws": int(draws),
        "immediate_rank_writes": {},
        "mean_ablation_rank_changes": {},
        "metrics": {},
        "paired_condition_contrasts": {},
        "run_metadata": json.loads(metadata_path.read_text()),
    }
    for condition_index, condition in enumerate(CONDITIONS):
        summary["immediate_rank_writes"][condition] = {
            rank: {
                "estimate": float(immediate_stats[condition_index][0][rank_index]),
                "ci": [
                    float(immediate_stats[condition_index][1][rank_index]),
                    float(immediate_stats[condition_index][2][rank_index]),
                ],
            }
            for rank_index, rank in enumerate(RANKS)
        }
        summary["mean_ablation_rank_changes"][condition] = {
            rank: {
                "estimate": float(causal_stats[condition_index][0][rank_index]),
                "ci": [
                    float(causal_stats[condition_index][1][rank_index]),
                    float(causal_stats[condition_index][2][rank_index]),
                ],
            }
            for rank_index, rank in enumerate(RANKS)
        }
        summary["metrics"][condition] = {
            metric: _ci_record(values[:, condition_index], bootstrap)
            for metric, values in metrics.items()
        }

    # Positive values mean that the natural component sharpens more in the
    # first-named condition, because the recorded ablation changes are negative
    # when removing the component reduces spread or winner advantage.
    for label, first, second in (
        ("Baseline_minus_Game", 0, 1),
        ("Neutral_minus_Game", 2, 1),
        ("Baseline_minus_Neutral", 0, 2),
    ):
        summary["paired_condition_contrasts"][label] = {}
        for metric in ("spread_change", "winner_advantage_change"):
            sharpening_difference = -(metrics[metric][:, first] - metrics[metric][:, second])
            summary["paired_condition_contrasts"][label][f"{metric.removesuffix('_change')}_sharpening"] = _ci_record(
                sharpening_difference, bootstrap
            )

    (output / "mixer56_across_conditions_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    _plot(figure, immediate_stats, causal_stats)

    lines = [
        "# Mixer 56 across Baseline, Game, and Neutral",
        "",
        f"Confirmation questions: **{len(baseline_logits)}**. Condition-specific mean outputs were estimated on the disjoint 251-question discovery set, equal-weighting the natural Baseline winner letters. All prompts use the canonical explicit empty-history ChatML format.",
        "",
        "## Immediate JLens write",
        "",
        "| Condition | Baseline rank 1 | Rank 2 | Rank 3 | Rank 4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition_index, condition in enumerate(CONDITIONS):
        values = immediate_stats[condition_index][0]
        lines.append("| " + condition + " | " + " | ".join(f"{value:+.3f}" for value in values) + " |")
    lines += [
        "",
        "## Within-condition causal mean-ablation",
        "",
        "| Condition | Answers changed | Spread change | Winner-advantage change | Entropy change | Switch-rate change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = summary["metrics"][condition]
        def fmt(name, scale=1.0):
            value = row[name]
            return f"{scale * value['estimate']:+.3f} [{scale * value['ci'][0]:+.3f}, {scale * value['ci'][1]:+.3f}]"
        lines.append(
            f"| {condition} | {fmt('answer_change_rate', 100)} pp | {fmt('spread_change')} | "
            f"{fmt('winner_advantage_change')} | {fmt('entropy_change')} | {fmt('switch_rate_change', 100)} pp |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Mixer 56 is not a sign-reversing component. Removing its question-specific output flattens the A–D distribution in all three conditions, so its natural causal function is sharpening in Baseline, Game, and Neutral. The sharpening is substantially weaker in Game; Neutral is approximately Baseline-like.",
        "",
        "| Paired contrast in natural Mixer-56 sharpening | Spread | Winner advantage |",
        "|---|---:|---:|",
    ]
    for label in ("Baseline_minus_Game", "Neutral_minus_Game", "Baseline_minus_Neutral"):
        values = summary["paired_condition_contrasts"][label]
        def contrast_fmt(name):
            value = values[name]
            return f"{value['estimate']:+.3f} [{value['ci'][0]:+.3f}, {value['ci'][1]:+.3f}]"
        lines.append(
            f"| {label.replace('_minus_', ' minus ')} | {contrast_fmt('spread_sharpening')} | "
            f"{contrast_fmt('winner_advantage_sharpening')} |"
        )
    lines += [
        "",
        "Panel A is an observational JLens finite-difference attribution. Panel B is causal: it replaces the question-specific Mixer-56 output with that condition's discovery-set mean.",
        "",
        f"Figure: `{figure}`",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--condition-results", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(
        args.baseline_results, args.condition_results, args.metadata,
        args.output, args.figure, args.bootstrap, args.seed,
    )


if __name__ == "__main__":
    main()
