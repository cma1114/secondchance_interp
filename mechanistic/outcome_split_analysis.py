from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import load_activation_dataset
from .perturbation_analysis import _cross_fitted_compression_residuals, _style
from .probes import stratified_folds
from .trajectory_analysis import centered


CONDITIONS = ("incorrect", "neutral")
CONDITION_LABELS = {
    "incorrect": "Second Chance - baseline",
    "neutral": "Neutral - baseline (same trials)",
}
GROUP_LABELS = {
    "keep": "Game keeps baseline winner",
    "lower": "Game switches to baseline rank 3 or 4",
}
SERIES_COLORS = {
    "baseline_winner": "#0072B2",
    "baseline_runner": "#D55E00",
    "lower_mean": "#666666",
    "game_chosen_lower": "#009E73",
    "other_lower": "#CC79A7",
}
SERIES_LABELS = {
    "baseline_winner": "Baseline winner",
    "baseline_runner": "Baseline runner-up",
    "lower_mean": "Ranks 3-4 mean",
    "game_chosen_lower": "Game-chosen lower option",
    "other_lower": "Other lower option",
}


def _bootstrap(values: np.ndarray, repetitions: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Question-clustered percentile interval within an outcome-defined group."""
    rng = np.random.default_rng(seed)
    mean = values.mean(axis=0)
    draws = np.empty((repetitions, values.shape[1]))
    for repetition in range(repetitions):
        sample = rng.integers(0, len(values), size=len(values))
        draws[repetition] = values[sample].mean(axis=0)
    return mean, np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def _series(
    residuals: np.ndarray,
    baseline_order: np.ndarray,
    game_choice: np.ndarray,
    group: str,
) -> dict[str, np.ndarray]:
    aligned = np.take_along_axis(residuals, baseline_order[:, None, :], axis=-1)
    if group == "keep":
        return {
            "baseline_winner": aligned[:, :, 0],
            "baseline_runner": aligned[:, :, 1],
            "lower_mean": aligned[:, :, 2:].mean(axis=-1),
        }
    if group != "lower":
        raise ValueError(group)

    row = np.arange(len(residuals))
    chosen_lower = residuals[row, :, game_choice]
    lower_letters = baseline_order[:, 2:]
    other_letter = np.where(lower_letters[:, 0] == game_choice, lower_letters[:, 1], lower_letters[:, 0])
    other_lower = residuals[row, :, other_letter]
    return {
        "baseline_winner": aligned[:, :, 0],
        "baseline_runner": aligned[:, :, 1],
        "game_chosen_lower": chosen_lower,
        "other_lower": other_lower,
    }


def _plot(rows: list[dict], output: Path, group_counts: dict[str, int], final_layer: int) -> None:
    import matplotlib.pyplot as plt

    _style()
    lookup = {(row["condition"], row["group"], row["series"]): row for row in rows}
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, sharey=True)
    layers = np.arange(final_layer + 1)
    groups = ("keep", "lower")
    for row_index, condition in enumerate(CONDITIONS):
        for column_index, group in enumerate(groups):
            axis = axes[row_index, column_index]
            names = (
                ("baseline_winner", "baseline_runner", "lower_mean")
                if group == "keep"
                else ("baseline_winner", "baseline_runner", "game_chosen_lower", "other_lower")
            )
            for name in names:
                result = lookup[(condition, group, name)]
                mean = np.asarray(result["mean"])
                low = np.asarray(result["ci_low"])
                high = np.asarray(result["ci_high"])
                axis.plot(layers, mean, color=SERIES_COLORS[name], lw=1.45, label=SERIES_LABELS[name])
                axis.fill_between(layers, low, high, color=SERIES_COLORS[name], alpha=0.13, linewidth=0)
            axis.axhline(0, color="#555555", lw=0.65)
            axis.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.7)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_xlim(0, final_layer)
            if row_index == 0:
                axis.set_title(f"{GROUP_LABELS[group]} (n={group_counts[group]})", fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(f"{CONDITION_LABELS[condition]}\nresidual pseudo-logit")
            if row_index == 1:
                axis.set_xlabel(f"Residual readout (0 = embedding; {final_layer} = final block)")
            axis.legend(frameon=False, loc="best", fontsize=7)
    figure.tight_layout(w_pad=1.25, h_pad=1.1)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"outcome_split_residual_trajectories.{suffix}", bbox_inches="tight")
    plt.close(figure)


def analyze(
    input_dir: str | Path,
    output_dir: str | Path,
    folds: int,
    seed: int,
    bootstrap: int,
) -> dict:
    data = load_activation_dataset(input_dir, ["baseline", *CONDITIONS])
    logits = centered(data.logits)
    baseline = logits[:, 0]
    baseline_order = np.argsort(-baseline[:, -1], axis=-1)
    winner = baseline_order[:, 0]
    game_choice = np.argmax(logits[:, 1, -1], axis=-1)
    game_rank = np.asarray([
        int(np.flatnonzero(baseline_order[index] == game_choice[index])[0]) + 1
        for index in range(len(game_choice))
    ])
    masks = {"keep": game_rank == 1, "lower": game_rank >= 3}
    split = stratified_folds(winner, folds, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    residual_by_condition = {}
    for condition_index, condition in enumerate(CONDITIONS, start=1):
        residual_by_condition[condition], _ = _cross_fitted_compression_residuals(
            baseline,
            logits[:, condition_index] - baseline,
            winner,
            split,
        )

    rows = []
    long_rows = []
    for condition_index, condition in enumerate(CONDITIONS):
        for group_index, (group, mask) in enumerate(masks.items()):
            group_series = _series(
                residual_by_condition[condition][mask],
                baseline_order[mask],
                game_choice[mask],
                group,
            )
            for series_index, (series, values) in enumerate(group_series.items()):
                mean, low, high = _bootstrap(
                    values,
                    bootstrap,
                    seed + 1000 * condition_index + 100 * group_index + series_index,
                )
                row = {
                    "condition": condition,
                    "group": group,
                    "series": series,
                    "mean": mean.tolist(),
                    "ci_low": low.tolist(),
                    "ci_high": high.tolist(),
                }
                rows.append(row)
                for layer in range(values.shape[1]):
                    long_rows.append({
                        "condition": condition,
                        "group": group,
                        "series": series,
                        "layer": layer,
                        "mean": mean[layer],
                        "ci_low": low[layer],
                        "ci_high": high[layer],
                    })

    with (output / "outcome_split_residual_trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=long_rows[0].keys())
        writer.writeheader()
        writer.writerows(long_rows)

    final_layer = logits.shape[2] - 1
    group_counts = {group: int(mask.sum()) for group, mask in masks.items()}
    _plot(rows, output, group_counts, final_layer)
    summary = {
        "n_questions": len(winner),
        "group_counts": group_counts,
        "excluded_switch_to_runner_count": int((game_rank == 2).sum()),
        "residual_definition": (
            "Condition-minus-baseline centered pseudo-logits after cross-fitted removal of "
            "option-letter effects and proportional baseline-geometry compression."
        ),
        "groups_are_defined_by": "Final Game choice rank under the final Baseline A-D ordering.",
        "intervals": f"{bootstrap}-draw question-clustered percentile bootstrap within each outcome group.",
        "final_layer": final_layer,
        "final_values": {
            f"{row['condition']}:{row['group']}:{row['series']}": {
                "mean": row["mean"][final_layer],
                "ci": [row["ci_low"][final_layer], row["ci_high"][final_layer]],
            }
            for row in rows
        },
    }
    (output / "outcome_split_residual_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Outcome-split residual A-D trajectories")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, args.folds, args.seed, args.bootstrap), indent=2))


if __name__ == "__main__":
    main()
