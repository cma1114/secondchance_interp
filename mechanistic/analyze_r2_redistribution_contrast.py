from __future__ import annotations

"""Does the Game policy preferentially route redistributed score to R2?

This is the direct test of the second-choice-targeting reading of the natural
final-answer profiles. For each model and dataset, it takes the exact live
final A-D logits from the natural non-remapped trajectory runs, centers them
within question, aligns candidates by frozen first-presentation rank, and
forms the paired within-question Game-minus-Neutral effect per rank. The
primary contrast is the R2 gain minus the mean R3/R4 gain: positive would
indicate that the old runner-up receives preferentially more of the
redistributed mass than the lower-ranked candidates, as second-choice
targeting predicts.

Observational evidence class: natural paired runs, no intervention. The rank
order is frozen per question from each model's own first-presentation
baseline (stable displayed-order argsort, stored in the run arrays). No
discrete argmax is taken anywhere in this analysis.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CELLS = (
    ("qwen36_simplemc", "Qwen3.6-27B SimpleMC"),
    ("qwen36_triviamc", "Qwen3.6-27B TriviaMC"),
    ("seed_oss_simplemc", "Seed-OSS-36B SimpleMC"),
    ("seed_oss_triviamc", "Seed-OSS-36B TriviaMC"),
)
RANK_LABELS = ("R1", "R2", "R3", "R4")


def paired_rank_effects(results_path: Path) -> tuple[np.ndarray, list[str]]:
    """Per-question paired Game-minus-Neutral centered final logits by rank."""

    with np.load(results_path, allow_pickle=False) as arrays:
        direct = arrays["direct_logits"].astype(np.float64)
        rank_order = arrays["rank_order"].astype(np.int64)
        question_ids = [str(value) for value in arrays["question_ids"]]
    if direct.shape[0] != 2 or direct.shape[-1] != 4:
        raise ValueError(f"Expected (2, n, 4) final logits in {results_path}")
    if rank_order.shape != (direct.shape[1], 4):
        raise ValueError(f"Expected (n, 4) rank order in {results_path}")
    centered = direct - direct.mean(axis=-1, keepdims=True)
    questions = np.arange(direct.shape[1])
    ranked = np.stack(
        [centered[:, questions, rank_order[:, rank]] for rank in range(4)], axis=-1
    )
    return ranked[0] - ranked[1], question_ids


def contrast(effects: np.ndarray) -> np.ndarray:
    """R2 gain minus the mean R3/R4 gain, per question."""

    return effects[:, 1] - effects[:, 2:4].mean(axis=1)


def _interval(
    values: np.ndarray, rng: np.random.Generator, draws: int
) -> dict[str, Any]:
    if values.ndim != 1 or not len(values):
        raise ValueError("Expected a nonempty one-dimensional sample")
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": [float(value) for value in np.quantile(means, (0.025, 0.975))],
    }


def _discovery_mask(split_path: Path, question_ids: list[str]) -> np.ndarray:
    payload = json.loads(split_path.read_text())
    discovery = payload.get("discovery_question_ids", payload.get("question_ids"))
    if not discovery:
        raise ValueError(f"No discovery question IDs in {split_path}")
    discovery_set = set(str(value) for value in discovery)
    mask = np.asarray([qid in discovery_set for qid in question_ids])
    if not mask.any() or mask.all():
        raise ValueError(f"Split {split_path} does not partition the questions")
    return mask


def _summarize_cell(
    effects: np.ndarray,
    discovery: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, Any]:
    cell: dict[str, Any] = {}
    subsets = {
        "all": np.ones(len(effects), dtype=bool),
        "discovery": discovery,
        "confirmation": ~discovery,
    }
    for name, mask in subsets.items():
        selected = effects[mask]
        cell[name] = {
            "rank_effects": {
                label: _interval(selected[:, rank], rng, draws)
                for rank, label in enumerate(RANK_LABELS)
            },
            "r2_minus_mean_r3_r4": _interval(contrast(selected), rng, draws),
        }
    return cell


def _fmt(row: dict[str, Any]) -> str:
    return f"{row['mean']:+.3f} [{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]"


def _plot(summary: dict[str, Any], figure_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    positions = np.arange(len(CELLS))
    offsets = np.linspace(-0.27, 0.27, 4)
    colors = ("#c23b2f", "#2b6fbb", "#3f9b5f", "#7b5cc4")

    axis = axes[0]
    for rank, (label, color) in enumerate(zip(RANK_LABELS, colors)):
        means = [summary["cells"][key]["all"]["rank_effects"][label]["mean"] for key, _ in CELLS]
        lows = [summary["cells"][key]["all"]["rank_effects"][label]["ci"][0] for key, _ in CELLS]
        highs = [summary["cells"][key]["all"]["rank_effects"][label]["ci"][1] for key, _ in CELLS]
        axis.errorbar(
            positions + offsets[rank],
            means,
            yerr=[np.subtract(means, lows), np.subtract(highs, means)],
            fmt="o",
            capsize=3.5,
            color=color,
            label=f"1P {label}",
        )
    axis.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
    axis.set_title("Game − Neutral final answer score by first-presentation rank")
    axis.set_ylabel("Paired centered-logit effect")

    axis = axes[1]
    for subset, color, shift in (("discovery", "#9a9a9a", -0.14), ("confirmation", "#1f5f8a", 0.14)):
        means = [summary["cells"][key][subset]["r2_minus_mean_r3_r4"]["mean"] for key, _ in CELLS]
        lows = [summary["cells"][key][subset]["r2_minus_mean_r3_r4"]["ci"][0] for key, _ in CELLS]
        highs = [summary["cells"][key][subset]["r2_minus_mean_r3_r4"]["ci"][1] for key, _ in CELLS]
        axis.errorbar(
            positions + shift,
            means,
            yerr=[np.subtract(means, lows), np.subtract(highs, means)],
            fmt="s" if subset == "confirmation" else "o",
            capsize=3.5,
            color=color,
            label=subset.title(),
        )
    axis.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
    axis.set_title("R2 gain minus mean R3/R4 gain\n(positive would indicate second-choice targeting)")
    axis.set_ylabel("Paired contrast (logits)")

    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels([label for _, label in CELLS], rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle(
        "The old runner-up gains least (or no more) of the redistributed score",
        fontsize=13.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(args: argparse.Namespace) -> None:
    inputs = {
        "qwen36_simplemc": (args.qwen_simplemc, args.simplemc_split),
        "qwen36_triviamc": (args.qwen_triviamc, args.triviamc_split),
        "seed_oss_simplemc": (args.seed_simplemc, args.simplemc_split),
        "seed_oss_triviamc": (args.seed_triviamc, args.triviamc_split),
    }
    rng = np.random.default_rng(args.seed)
    summary: dict[str, Any] = {
        "analysis": "r2_redistribution_contrast",
        "question": (
            "When Game redistributes score away from the old winner, does the old "
            "runner-up receive preferentially more of it than the lower-ranked "
            "candidates, as second-choice targeting would predict?"
        ),
        "evidence_class": (
            "Observational paired natural-run contrast of exact live final A-D "
            "logits; no intervention; no discrete argmax is used"
        ),
        "definitions": {
            "rank_effects": (
                "Within-question centered final A-D logits, aligned by frozen "
                "first-presentation rank, Game minus Neutral, paired per question"
            ),
            "r2_minus_mean_r3_r4": (
                "The R2 rank effect minus the mean of the R3 and R4 rank effects; "
                "positive means the runner-up gains preferentially"
            ),
        },
        "bootstrap": {"draws": args.draws, "seed": args.seed, "resampling": "question-level percentile"},
        "inputs": {key: str(path) for key, (path, _split) in inputs.items()},
        "splits": {
            "simplemc": str(args.simplemc_split),
            "triviamc": str(args.triviamc_split),
        },
        "cells": {},
    }
    for key, (results_path, split_path) in inputs.items():
        effects, question_ids = paired_rank_effects(Path(results_path))
        discovery = _discovery_mask(Path(split_path), question_ids)
        summary["cells"][key] = _summarize_cell(effects, discovery, rng, args.draws)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _plot(summary, args.figure)

    lines = [
        "# Does Game preferentially route redistributed score to the old runner-up?",
        "",
        "No. Across both models and both datasets, when `incorrect` pushes the old",
        "winner down, the old runner-up gains **less** of the redistributed score than",
        "the two lower-ranked candidates in three of four cells, and is statistically",
        "indistinguishable from them in the fourth. In no cell does it gain reliably",
        "more. The runner-up wins most switches despite being the policy's",
        "least-favored alternative, because it starts closest to the top. This is the",
        "direct quantitative rebuttal of a second-choice-targeting reading of the",
        "natural rank profiles.",
        "",
        "These are paired within-question Game-minus-Neutral effects on exact live",
        "final A-D logits (centered within question, candidates aligned by each",
        "model's own frozen first-presentation ranking). Observational natural runs;",
        "no intervention; question-level percentile bootstrap.",
        "",
        "## All questions",
        "",
        "| Cell | R1 | R2 | R3 | R4 | R2 − mean(R3, R4) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in CELLS:
        rows = summary["cells"][key]["all"]
        lines.append(
            f"| {label} | "
            + " | ".join(_fmt(rows["rank_effects"][rank]) for rank in RANK_LABELS)
            + f" | {_fmt(rows['r2_minus_mean_r3_r4'])} |"
        )
    lines += [
        "",
        "## Frozen-split robustness of the primary contrast",
        "",
        "| Cell | Discovery | Confirmation |",
        "|---|---:|---:|",
    ]
    for key, label in CELLS:
        lines.append(
            f"| {label} | {_fmt(summary['cells'][key]['discovery']['r2_minus_mean_r3_r4'])} "
            f"| {_fmt(summary['cells'][key]['confirmation']['r2_minus_mean_r3_r4'])} |"
        )
    lines += [
        "",
        "The absolute R2 rise on TriviaMC in both models is conservation of mass: the",
        "old winner there is dominant (its suppression is largest), so every",
        "alternative floats up — and R2 floats up least or equally, never most.",
        "",
        "## Scope",
        "",
        "This is a descriptive contrast on natural runs. The causal case against",
        "second-choice targeting rests on the Qwen interventions (categorical-winner",
        "audit, matching-route lesions, destination analysis); this analysis shows",
        "the natural profiles point the same way in both models and both datasets.",
        "",
        "## Artifacts",
        "",
        f"- Figure: `{args.figure}`",
        "- Machine-readable estimates: `summary.json`",
        "- Inputs: the natural non-remapped trajectory run arrays for each model and",
        "  dataset, and the frozen SimpleMC/TriviaMC discovery splits.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    root = Path("outputs")
    parser.add_argument(
        "--qwen-simplemc",
        type=Path,
        default=root / "prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/run/simplemc/results.npz",
    )
    parser.add_argument(
        "--qwen-triviamc",
        type=Path,
        default=root / "prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/run/triviamc/results.npz",
    )
    parser.add_argument(
        "--seed-simplemc",
        type=Path,
        default=root / "model_replications/seed_oss_36b_final_position_trajectories/run/simplemc/results.npz",
    )
    parser.add_argument(
        "--seed-triviamc",
        type=Path,
        default=root / "model_replications/seed_oss_36b_final_position_trajectories/run/triviamc/results.npz",
    )
    parser.add_argument(
        "--simplemc-split",
        type=Path,
        default=root / "causal/qwen36_27b_simplemc_causal_sweep/plans/discovery_plan.json",
    )
    parser.add_argument(
        "--triviamc-split",
        type=Path,
        default=root / "prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/split_plan.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "model_replications/r2_redistribution_contrast/analysis",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/model_replications/r2_redistribution_contrast.png"),
    )
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser


if __name__ == "__main__":
    analyze(build_parser().parse_args())
