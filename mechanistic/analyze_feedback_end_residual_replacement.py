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
from .run_feedback_end_residual_replacement import WINDOW_NAMES


DIRECTIONS = ("Neutral into Game", "Game into Neutral")
COLORS = ("#1689d8", "#e66b19")
WINDOW_LABELS = ("L1–16", "L17–32", "L33–40", "L41–48", "L49–64", "All")


def _ci(values: np.ndarray, bootstrap: np.ndarray, scale: float = 1.0):
    point, low, high = _summary(values, bootstrap)
    return {
        "estimate": float(point * scale),
        "ci": [float(low * scale), float(high * scale)],
    }


def _plot(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.6), sharex=True)
    x = np.arange(len(WINDOW_NAMES))
    panels = (
        (axes[0, 0], "winner_advantage_change", "A  Original-winner advantage", "Change in canonical logits"),
        (axes[0, 1], "spread_change", "B  A–D spread", "Change in canonical-logit SD"),
        (axes[1, 0], "switch_rate_change_pp", "C  Switching", "Change in switch rate (pp)"),
    )
    for axis, metric, title, ylabel in panels:
        for direction, color in zip(DIRECTIONS, COLORS):
            records = [summary["effects"][direction][name][metric] for name in WINDOW_NAMES]
            point = np.asarray([record["estimate"] for record in records])
            low = np.asarray([record["ci"][0] for record in records])
            high = np.asarray([record["ci"][1] for record in records])
            axis.errorbar(
                x, point, yerr=np.stack([point - low, high - point]),
                marker="o", lw=2.2, capsize=3, color=color, label=direction,
            )
        axis.axhline(0, color="#555", lw=1, ls="--")
        axis.set_title(title, loc="left", weight="bold")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False)

    width = 0.36
    for offset, direction, color in zip((-width / 2, width / 2), DIRECTIONS, COLORS):
        values = [
            summary["effects"][direction][name]["choice_transitions"]["total_choice_flips"]
            for name in WINDOW_NAMES
        ]
        axes[1, 1].bar(x + offset, values, width, color=color, label=direction)
    axes[1, 1].set_title("D  Questions whose answer changed", loc="left", weight="bold")
    axes[1, 1].set_ylabel(f"Count out of {summary['n_questions']}")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.set_xticks(x, WINDOW_LABELS)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[1]:
        axis.set_xlabel("Feedback-end residual replacement window")
    fig.suptitle(
        "Complete feedback-end residual replacement — canonical prompt",
        weight="bold",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def analyze(
    results_path: Path,
    baseline_path: Path,
    metadata_path: Path,
    output: Path,
    figure: Path,
    draws: int,
    seed: int,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with np.load(results_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    with np.load(baseline_path, allow_pickle=False) as loaded:
        baseline = {key: loaded[key] for key in loaded.files}
    if not np.all(data["completed"]):
        raise ValueError("Run is incomplete")
    if data["question_ids"].astype(str).tolist() != baseline["question_ids"].astype(str).tolist():
        raise ValueError("Baseline and intervention question orders differ")

    baseline_logits = baseline["natural_baseline_logits"].astype(float)
    baseline_answers = np.argmax(baseline_logits, axis=1)
    order = np.argsort(-baseline_logits, axis=1)
    bootstrap = _bootstrap_indices(baseline_answers, draws, seed)
    natural = data["natural_logits"].astype(float)
    patched = data["patched_logits"].astype(float)
    natural_choice = np.argmax(natural, axis=2)
    patched_choice = np.argmax(patched, axis=3)
    natural_switch = natural_choice != baseline_answers[None, :]
    patched_switch = patched_choice != baseline_answers[None, None, :]

    repeated_order_natural = np.repeat(order[None, :, :], 2, axis=0)
    repeated_order_patched = np.repeat(order[None, None, :, :], 2, axis=0)
    repeated_order_patched = np.repeat(repeated_order_patched, len(WINDOW_NAMES), axis=1)
    metrics = {
        "winner_advantage_change": (
            _winner_advantage(patched, repeated_order_patched)
            - _winner_advantage(natural, repeated_order_natural)[:, None, :]
        ),
        "spread_change": _spread(patched) - _spread(natural)[:, None, :],
        "entropy_change": _entropy(patched) - _entropy(natural)[:, None, :],
        "switch_rate_change_pp": (
            patched_switch.astype(float) - natural_switch[:, None, :].astype(float)
        ) * 100.0,
    }
    natural_advantage = _winner_advantage(natural, repeated_order_natural)
    natural_spread = _spread(natural)
    natural_entropy = _entropy(natural)

    summary = {
        "n_questions": int(len(baseline_answers)),
        "bootstrap_draws": int(draws),
        "natural": {
            "Game_switch_rate": float(np.mean(natural_switch[0])),
            "Neutral_switch_rate": float(np.mean(natural_switch[1])),
            "Game_winner_advantage": float(np.mean(natural_advantage[0])),
            "Neutral_winner_advantage": float(np.mean(natural_advantage[1])),
            "Game_spread": float(np.mean(natural_spread[0])),
            "Neutral_spread": float(np.mean(natural_spread[1])),
            "Game_entropy": float(np.mean(natural_entropy[0])),
            "Neutral_entropy": float(np.mean(natural_entropy[1])),
        },
        "effects": {},
        "run_metadata": json.loads(metadata_path.read_text()),
    }
    for direction_index, direction in enumerate(DIRECTIONS):
        summary["effects"][direction] = {}
        for window_index, window in enumerate(WINDOW_NAMES):
            records = {
                metric: _ci(values[direction_index, window_index], bootstrap)
                for metric, values in metrics.items()
            }
            changed = natural_choice[direction_index] != patched_choice[direction_index, window_index]
            new_switch = ~natural_switch[direction_index] & patched_switch[direction_index, window_index]
            prevented = natural_switch[direction_index] & ~patched_switch[direction_index, window_index]
            other = (
                natural_switch[direction_index]
                & patched_switch[direction_index, window_index]
                & changed
            )
            records["choice_transitions"] = {
                "total_choice_flips": int(np.sum(changed)),
                "new_switches": int(np.sum(new_switch)),
                "prevented_switches": int(np.sum(prevented)),
                "switched_to_different_alternative": int(np.sum(other)),
            }
            delta = patched[direction_index, window_index] - natural[direction_index]
            delta -= delta.mean(axis=1, keepdims=True)
            ranked = np.take_along_axis(delta, order, axis=1)
            point, low, high = _summary(ranked, bootstrap)
            records["rank_logit_changes"] = {
                f"Rank {rank + 1}": {
                    "estimate": float(point[rank]),
                    "ci": [float(low[rank]), float(high[rank])],
                }
                for rank in range(4)
            }
            summary["effects"][direction][window] = records

    expected_moves = {
        "Neutral into Game": {
            "winner_advantage_change": float(np.mean(natural_advantage[1] - natural_advantage[0])),
            "spread_change": float(np.mean(natural_spread[1] - natural_spread[0])),
            "entropy_change": float(np.mean(natural_entropy[1] - natural_entropy[0])),
            "switch_rate_change_pp": float(100 * np.mean(natural_switch[1].astype(float) - natural_switch[0].astype(float))),
        },
        "Game into Neutral": {
            "winner_advantage_change": float(np.mean(natural_advantage[0] - natural_advantage[1])),
            "spread_change": float(np.mean(natural_spread[0] - natural_spread[1])),
            "entropy_change": float(np.mean(natural_entropy[0] - natural_entropy[1])),
            "switch_rate_change_pp": float(100 * np.mean(natural_switch[0].astype(float) - natural_switch[1].astype(float))),
        },
    }
    summary["all_layer_fraction_of_condition_gap"] = {
        direction: {
            metric: float(summary["effects"][direction]["all_layers"][metric]["estimate"] / expected)
            for metric, expected in expected_moves[direction].items()
        }
        for direction in DIRECTIONS
    }

    (output / "feedback_end_residual_replacement_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    _plot(figure, summary)

    lines = [
        "# Complete feedback-end residual replacement — canonical prompt",
        "",
        f"Held-out questions: **{len(baseline_answers)}**. At the period ending the feedback sentence, the complete post-block residual was replaced with the paired same-question residual from the other condition. All prompts use the canonical explicit empty-history raw ChatML format.",
        "",
        f"Natural switching: Game **{100*np.mean(natural_switch[0]):.1f}%**; Neutral **{100*np.mean(natural_switch[1]):.1f}%**.",
        "",
        "## Causal effects",
        "",
        "| Direction | Window | Winner advantage | A–D spread | Entropy | Switch rate | Answers changed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for direction in DIRECTIONS:
        for window, label in zip(WINDOW_NAMES, WINDOW_LABELS):
            record = summary["effects"][direction][window]
            def fmt(metric: str):
                value = record[metric]
                return f"{value['estimate']:+.3f} [{value['ci'][0]:+.3f}, {value['ci'][1]:+.3f}]"
            lines.append(
                f"| {direction} | {label} | {fmt('winner_advantage_change')} | "
                f"{fmt('spread_change')} | {fmt('entropy_change')} | "
                f"{fmt('switch_rate_change_pp')} pp | "
                f"{record['choice_transitions']['total_choice_flips']} |"
            )
    lines += [
        "",
        "## All-layer rank redistribution",
        "",
        "| Direction | Rank 1 | Rank 2 | Rank 3 | Rank 4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for direction in DIRECTIONS:
        ranks = summary["effects"][direction]["all_layers"]["rank_logit_changes"]
        lines.append(
            "| " + direction + " | "
            + " | ".join(f"{ranks[f'Rank {rank}']['estimate']:+.3f}" for rank in range(1, 5))
            + " |"
        )
    neutral_game = summary["all_layer_fraction_of_condition_gap"]["Neutral into Game"]
    game_neutral = summary["all_layer_fraction_of_condition_gap"]["Game into Neutral"]
    lines += [
        "",
        "## Interpretation",
        "",
        "The complete feedback-end residual is a real causal carrier of continuous answer compression, but not of net switching. Replacing all Game feedback-end readouts with Neutral restores "
        f"{100*neutral_game['winner_advantage_change']:.1f}% of the natural winner-advantage gap, "
        f"{100*neutral_game['spread_change']:.1f}% of the spread gap, and "
        f"{100*neutral_game['entropy_change']:.1f}% of the entropy gap. It mediates only "
        f"{100*neutral_game['switch_rate_change_pp']:.1f}% of the switch-rate gap, with a confidence interval spanning zero.",
        "",
        "The effect is asymmetric. Inserting the Game feedback-end state into Neutral moves only "
        f"{100*game_neutral['winner_advantage_change']:.1f}% of the winner-advantage gap, "
        f"{100*game_neutral['spread_change']:.1f}% of the spread gap, and "
        f"{100*game_neutral['entropy_change']:.1f}% of the entropy gap; switching moves slightly in the wrong direction. The strongest isolated source window is L33–40. This is more consistent with the Neutral state supplying sharpening that Game lacks than with a portable feedback-end compression command.",
        "",
        f"Figure: `{figure}`",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(
        args.results, args.baseline_results, args.metadata, args.output,
        args.figure, args.bootstrap, args.seed,
    )


if __name__ == "__main__":
    main()
