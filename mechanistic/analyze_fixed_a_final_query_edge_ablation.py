from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .run_fixed_a_final_query_edge_ablation import INTERVENTION_CELLS


def _load(path: Path) -> dict[str, np.ndarray]:
    arrays = dict(np.load(path / "results.npz", allow_pickle=False))
    if not np.all(arrays["completed"]):
        raise ValueError(f"Incomplete output: {path}")
    if arrays["intervention_cells"].astype(str).tolist() != list(INTERVENTION_CELLS):
        raise ValueError(f"Unexpected intervention cells: {path}")
    eligible = arrays["first_decision_valid"].astype(bool)
    if not np.any(eligible):
        raise ValueError(f"No exact-regime questions in {path}")
    return {
        key: value[..., eligible, :] if key in {"natural_logits", "intervention_logits"}
        else value[eligible] if key in {"question_ids", "x_second_letter", "y_second_letter"}
        else value
        for key, value in arrays.items()
    }


def _entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(shifted)
    p /= p.sum(axis=-1, keepdims=True)
    return -(p * np.log2(np.clip(p, 1e-12, None))).sum(axis=-1)


def _metrics(logits: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    q = np.arange(len(x))
    answers = logits.argmax(axis=-1)
    game_w1 = 0.5 * ((answers[0] == x) + (answers[2] == y))
    neutral_w1 = 0.5 * ((answers[1] == x) + (answers[3] == y))
    game_margin = 0.5 * (
        logits[0, q, x] - logits[0, q, y]
        + logits[2, q, y] - logits[2, q, x]
    )
    neutral_margin = 0.5 * (
        logits[1, q, x] - logits[1, q, y]
        + logits[3, q, y] - logits[3, q, x]
    )
    entropy = _entropy(logits)
    return {
        "game_w1_selection": game_w1.astype(float),
        "neutral_w1_selection": neutral_w1.astype(float),
        "w1_avoidance_gap": neutral_w1.astype(float) - game_w1.astype(float),
        "game_w1_margin": game_margin,
        "neutral_w1_margin": neutral_margin,
        "w1_margin_gap": neutral_margin - game_margin,
        "game_entropy": 0.5 * (entropy[0] + entropy[2]),
        "neutral_entropy": 0.5 * (entropy[1] + entropy[3]),
    }


def _interval(values: np.ndarray, bootstrap: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    means = values[bootstrap].mean(axis=1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def _split_summary(path: Path, draws: int, seed: int) -> dict[str, Any]:
    arrays = _load(path)
    x = np.asarray(["ABCD".index(value) for value in arrays["x_second_letter"].astype(str)])
    y = np.asarray(["ABCD".index(value) for value in arrays["y_second_letter"].astype(str)])
    natural = _metrics(arrays["natural_logits"], x, y)
    n = len(x)
    rng = np.random.default_rng(seed)
    bootstrap = rng.integers(0, n, size=(draws, n))
    cells: list[dict[str, Any]] = []
    intervened_metrics: list[dict[str, np.ndarray]] = []
    for index, cell in enumerate(INTERVENTION_CELLS):
        intervened = _metrics(arrays["intervention_logits"][index], x, y)
        intervened_metrics.append(intervened)
        changes = {key: intervened[key] - natural[key] for key in natural}
        cells.append({
            "cell": cell,
            **{key: _interval(value, bootstrap) for key, value in changes.items()},
        })
    source_contrasts = []
    for selected_index, control_index, label in (
        (0, 1, "block_44"),
        (2, 3, "band_36_48"),
        (4, 5, "all_04_48"),
    ):
        differences = {
            key: intervened_metrics[selected_index][key]
            - intervened_metrics[control_index][key]
            for key in natural
        }
        source_contrasts.append({
            "block_set": label,
            **{key: _interval(value, bootstrap) for key, value in differences.items()},
        })
    return {
        "n_historical": int(len(arrays["first_decision_valid"])),
        "n_eligible": n,
        "natural": {key: _interval(value, bootstrap) for key, value in natural.items()},
        "cells": cells,
        "selected_minus_control": source_contrasts,
    }


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    selected_cells = [
        "block_44_selected", "band_36_48_selected", "all_04_48_selected",
        "block_44_matched_control", "band_36_48_matched_control",
        "all_04_48_matched_control",
    ]
    labels = ["B44\nselected", "B36–48\nselected", "B4–48\nselected",
              "B44\ncontrol", "B36–48\ncontrol", "B4–48\ncontrol"]
    panels = [
        ("game_w1_selection", "A  Game chooses W1", "Change (percentage points)", 100.0),
        ("neutral_w1_selection", "B  Neutral chooses W1", "Change (percentage points)", 100.0),
        ("w1_avoidance_gap", "C  Preferential W1-avoidance gap", "Change (percentage points)", 100.0),
        ("game_w1_margin", "D  Game W1-versus-counterfactual margin", "Change (logit units)", 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"discovery": "#8bbdf3", "confirmation": "#1f7fd1"}
    offsets = {"discovery": -0.10, "confirmation": 0.10}
    for axis, (metric, title, ylabel, scale) in zip(axes.flat, panels):
        for split in ("discovery", "confirmation"):
            lookup = {row["cell"]: row for row in summary[split]["cells"]}
            means = np.asarray([lookup[cell][metric]["mean"] for cell in selected_cells]) * scale
            cis = np.asarray([lookup[cell][metric]["ci"] for cell in selected_cells]) * scale
            x = np.arange(len(selected_cells)) + offsets[split]
            axis.errorbar(
                x, means,
                yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
                fmt="o", markersize=6, capsize=4, linewidth=1.5,
                color=colors[split], label=split.capitalize(),
            )
        axis.axhline(0, color="#666666", linewidth=1, linestyle="--")
        axis.set_title(title, loc="left", fontsize=14)
        axis.set_ylabel(ylabel)
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Direct final-query reading of the first selected-option line is not required\n"
        "Points are paired means; bars are 95% question-bootstrap intervals",
        fontsize=17,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _report(summary: dict[str, Any], figure: Path) -> str:
    lines = [
        "# Fixed-A final-query selected-option attention-edge ablation",
        "",
        "## Bottom line",
        "",
        "This report tests whether the final answer decision must directly read the "
        "semantic option line selected on the first presentation. Only final-query "
        "ordinary-attention edges are removed; every earlier query remains intact.",
        "",
        "The predicted result did not occur. Across the prespecified block sets, "
        "removing the selected-line edges did not reliably increase Game's W1 choices, "
        "reduce its preferential avoidance of W1, or change the W1-versus-counterfactual "
        "margin. Selected-line effects were also not reliably different from matched "
        "unselected-line controls. Thus, although the selected option's K/V history is "
        "causally important in the earlier cache-transplant experiment, a clean direct "
        "read from the final query is not the route by which that information affects "
        "preferential semantic switching.",
        "",
    ]
    for split in ("discovery", "confirmation"):
        data = summary[split]
        lines.extend([
            f"## {split.capitalize()} (n={data['n_eligible']} exact-regime questions)",
            "",
            "| Intervention | Δ Game W1 choice | Δ Neutral W1 choice | Δ preferential W1 avoidance | Δ Game W1 margin |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in data["cells"]:
            def fmt(key: str, scale: float = 1.0) -> str:
                value = row[key]
                return f"{value['mean']*scale:+.2f} [{value['ci'][0]*scale:+.2f}, {value['ci'][1]*scale:+.2f}]"
            lines.append(
                f"| `{row['cell']}` | {fmt('game_w1_selection',100)} pp | "
                f"{fmt('neutral_w1_selection',100)} pp | {fmt('w1_avoidance_gap',100)} pp | "
                f"{fmt('game_w1_margin')} |"
            )
        lines.append("")
        lines.extend([
            "Selected-line effect minus matched-control effect:",
            "",
            "| Block set | Δ Game W1 choice | Δ preferential W1 avoidance | Δ Game W1 margin |",
            "|---|---:|---:|---:|",
        ])
        for row in data["selected_minus_control"]:
            def contrast_fmt(key: str, scale: float = 1.0) -> str:
                value = row[key]
                return f"{value['mean']*scale:+.2f} [{value['ci'][0]*scale:+.2f}, {value['ci'][1]*scale:+.2f}]"
            lines.append(
                f"| `{row['block_set']}` | {contrast_fmt('game_w1_selection',100)} pp | "
                f"{contrast_fmt('w1_avoidance_gap',100)} pp | "
                f"{contrast_fmt('game_w1_margin')} |"
            )
        lines.append("")
    lines.extend([
        "## Definitions",
        "",
        "- **W1** is the semantic content chosen as literal `A` on the first presentation. "
        "X and Y histories have the same second presentation but different W1 content.",
        "- **Preferential W1 avoidance** is Neutral's W1-choice rate minus Game's W1-choice rate. "
        "A negative intervention change means the lesion erased part of Game's preferential avoidance.",
        "- **Game W1 margin** is W1's A-D logit minus the counterfactual first-answer content's logit, "
        "averaged symmetrically over X and Y histories.",
        "- **Matched control** is the unselected first-presentation option line with the nearest token count to the selected A line.",
        "",
        f"Canonical figure: `{figure}`.",
    ])
    return "\n".join(lines) + "\n"


def analyze(
    discovery: Path,
    confirmation: Path,
    output_dir: Path,
    figure: Path,
    draws: int,
    seed: int,
) -> None:
    summary = {
        "discovery": _split_summary(discovery, draws, seed),
        "confirmation": _split_summary(confirmation, draws, seed + 1),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _plot(summary, figure)
    (output_dir / "REPORT.md").write_text(_report(summary, figure))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    analyze(args.discovery, args.confirmation, args.output, args.figure, args.draws, args.seed)


if __name__ == "__main__":
    main()
