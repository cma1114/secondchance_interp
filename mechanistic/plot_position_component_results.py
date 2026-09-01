from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


RANK_COLORS = ("#2E91E5", "#E15F33", "#1CA71C", "#FB0D98")
ANCHOR_LABELS = {
    "feedback_end": "Feedback-end period",
    "second_user_end": "Repeated-user final token",
    "decision": "Final decision position",
}


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _fraction_interval(row: dict) -> tuple[float, float, float]:
    gap = float(row["natural_game_minus_neutral_gap"])
    direction = row["direction"]
    multiplier = -1.0 / gap if direction == "neutral_into_game" else 1.0 / gap
    values = multiplier * np.asarray(
        [float(row["effect_mean"]), float(row["effect_ci_low"]), float(row["effect_ci_high"])]
    )
    return float(values[0]), float(min(values[1:])), float(max(values[1:]))


def _rank_trajectory(
    discovery_rows: list[dict], selected: set[str], output: Path
) -> None:
    import matplotlib.pyplot as plt

    primary = [
        row
        for row in discovery_rows
        if row["aggregation"] == "dataset"
        and row["direction"] == "neutral_into_game"
        and int(row["n_targets"]) == 1
    ]
    anchors = ("feedback_end", "second_user_end", "decision")
    kinds = ("mixer", "mlp")
    fig, axes = plt.subplots(
        3, 2, figsize=(16, 13), sharex=True, constrained_layout=True
    )
    limits = {
        "feedback_end": (-0.045, 0.045),
        "second_user_end": (-0.032, 0.032),
        "decision": (-0.10, 0.10),
    }
    for row_index, anchor in enumerate(anchors):
        for column_index, kind in enumerate(kinds):
            axis = axes[row_index, column_index]
            for rank, color in zip(range(1, 5), RANK_COLORS):
                subset = sorted(
                    (
                        row
                        for row in primary
                        if row["anchor"] == anchor
                        and row["kind"] == kind
                        and row["metric"] == f"causal_rank_write_{rank}"
                    ),
                    key=lambda row: int(row["layer"]),
                )
                x = np.asarray([int(row["layer"]) + 1 for row in subset])
                mean = np.asarray([float(row["effect_mean"]) for row in subset])
                low = np.asarray([float(row["effect_ci_low"]) for row in subset])
                high = np.asarray([float(row["effect_ci_high"]) for row in subset])
                axis.plot(
                    x,
                    mean,
                    color=color,
                    linewidth=1.35,
                    label=f"Baseline rank {rank}",
                )
                axis.fill_between(x, low, high, color=color, alpha=0.09, linewidth=0)
                if anchor == "decision" and kind == "mixer" and len(mean):
                    last_value = mean[-1]
                    lower, upper = limits[anchor]
                    if last_value < lower or last_value > upper:
                        boundary = upper * 0.94 if last_value > 0 else lower * 0.94
                        axis.annotate(
                            f"{last_value:+.2f}",
                            xy=(64, boundary),
                            xytext=(61.0, boundary * 0.92),
                            color=color,
                            fontsize=8,
                            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.8},
                        )
            lower, upper = limits[anchor]
            axis.set_ylim(lower, upper)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.grid(axis="y", alpha=0.18)
            selected_layers = sorted({
                int(row["layer"]) + 1
                for row in primary
                if row["anchor"] == anchor
                and row["kind"] == kind
                and row["component"] in selected
            })
            for layer in selected_layers:
                axis.plot(
                    [layer],
                    [upper * 0.91],
                    marker="v",
                    color="black",
                    markersize=5,
                    clip_on=False,
                )
            if row_index == 0:
                axis.set_title("Mixer outputs" if kind == "mixer" else "MLP outputs")
            if column_index == 0:
                axis.set_ylabel(
                    f"{ANCHOR_LABELS[anchor]}\nGame-specific causal write (logits)"
                )
            if row_index == 2:
                axis.set_xlabel("Transformer block")
            axis.set_xlim(1, 64)
            axis.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
    axes[0, 0].legend(frameon=False, ncol=2, loc="lower left")
    fig.suptitle(
        "Component-level causal writes to Baseline-ranked answer evidence\n"
        "Black triangles mark discovery-selected components",
        fontsize=15,
    )
    fig.savefig(output / "position_component_rank_writes.png", dpi=240)
    fig.savefig(output / "position_component_rank_writes.svg")
    plt.close(fig)


def _group_mediation(confirmation_rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = (
        ("ad_entropy", "A-D entropy"),
        ("ad_spread", "A-D spread"),
        ("winner_advantage", "Original-winner advantage"),
        ("switch", "Game-Neutral switch-rate gap"),
        ("rank_opposed_slope", "Ordered rank redistribution"),
    )
    directions = (
        ("neutral_into_game", "Remove Game outputs", "#2E91E5"),
        ("game_into_neutral", "Insert into Neutral", "#E15F33"),
    )
    selected = [
        row
        for row in confirmation_rows
        if row["aggregation"] == "dataset"
        and row["scenario"].endswith("all_candidates")
    ]
    y = np.arange(len(metrics), dtype=float)
    fig, axis = plt.subplots(figsize=(11, 6.3), constrained_layout=True)
    for offset, (direction, label, color) in zip((-0.13, 0.13), directions):
        means, low, high = [], [], []
        for metric, _ in metrics:
            row = next(
                row
                for row in selected
                if row["direction"] == direction and row["metric"] == metric
            )
            mean, left, right = _fraction_interval(row)
            means.append(mean)
            low.append(left)
            high.append(right)
        means = np.asarray(means)
        low = np.asarray(low)
        high = np.asarray(high)
        axis.errorbar(
            means,
            y + offset,
            xerr=np.vstack([means - low, high - means]),
            fmt="o",
            color=color,
            capsize=3,
            linewidth=1.4,
            markersize=6,
            label=label,
        )
        for value, yy in zip(means, y + offset):
            axis.text(value + 0.025, yy, f"{100 * value:.0f}%", va="center", fontsize=9)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.axvline(1, color="gray", linewidth=0.9, linestyle="--")
    axis.set_yticks(y, [label for _, label in metrics])
    axis.invert_yaxis()
    axis.set_xlim(-0.12, 1.33)
    axis.set_xlabel("Fraction of natural Game-Neutral gap causally mediated")
    axis.set_title("Eight-component group: held-out reciprocal mediation")
    axis.grid(axis="x", alpha=0.18)
    axis.legend(frameon=False, loc="lower right")
    fig.savefig(output / "heldout_group_mediation.png", dpi=240)
    fig.savefig(output / "heldout_group_mediation.svg")
    plt.close(fig)


def _individual_forest(
    confirmation_rows: list[dict], plan: dict, output: Path
) -> None:
    import matplotlib.pyplot as plt

    selected = [row["component"] for row in plan["overall_ranking"][:8]]
    labels = []
    for component in selected:
        anchor, remainder = component.split("__", 1)
        kind, layer = remainder.rsplit("_l", 1)
        block = int(layer) + 1
        labels.append(
            f"{ANCHOR_LABELS[anchor]} — {'MLP' if kind == 'mlp' else 'Mixer'} {block}"
        )
    rows = [
        row
        for row in confirmation_rows
        if row["aggregation"] == "dataset"
        and row["metric"] == "rank_opposed_slope"
        and row["component"] in selected
    ]
    directions = (
        ("neutral_into_game", "Remove Game output", "#2E91E5"),
        ("game_into_neutral", "Insert into Neutral", "#E15F33"),
    )
    y = np.arange(len(selected), dtype=float)
    fig, axis = plt.subplots(figsize=(12, 7.2), constrained_layout=True)
    for offset, (direction, label, color) in zip((-0.13, 0.13), directions):
        means, low, high = [], [], []
        for component in selected:
            row = next(
                row
                for row in rows
                if row["direction"] == direction and row["component"] == component
            )
            mean, left, right = _fraction_interval(row)
            means.append(mean)
            low.append(left)
            high.append(right)
        means = np.asarray(means)
        low = np.asarray(low)
        high = np.asarray(high)
        axis.errorbar(
            means,
            y + offset,
            xerr=np.vstack([means - low, high - means]),
            fmt="o",
            color=color,
            capsize=3,
            linewidth=1.3,
            markersize=5,
            label=label,
        )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Fraction of ordered rank-redistribution gap mediated")
    axis.set_title("Selected components: held-out reciprocal effects")
    axis.grid(axis="x", alpha=0.18)
    axis.legend(frameon=False, loc="lower right")
    fig.savefig(output / "heldout_individual_rank_mediation.png", dpi=240)
    fig.savefig(output / "heldout_individual_rank_mediation.svg")
    plt.close(fig)


def plot(discovery: Path, confirmation: Path, plan_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    discovery_rows = _rows(discovery)
    confirmation_rows = _rows(confirmation)
    plan = json.loads(plan_path.read_text())
    selected = {row["component"] for row in plan["overall_ranking"][:8]}
    _rank_trajectory(discovery_rows, selected, output)
    _group_mediation(confirmation_rows, output)
    _individual_forest(confirmation_rows, plan, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot position-resolved component causal results"
    )
    parser.add_argument("--discovery-effects", required=True)
    parser.add_argument("--confirmation-effects", required=True)
    parser.add_argument("--confirmation-plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(
        Path(args.discovery_effects),
        Path(args.confirmation_effects),
        Path(args.confirmation_plan),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
