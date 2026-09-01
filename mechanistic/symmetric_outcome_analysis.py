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
CONDITION_LABELS = {"incorrect": "Second Chance", "neutral": "Neutral"}
CONDITION_COLORS = {"incorrect": "#0072B2", "neutral": "#D55E00"}
PANELS = ("both_keep", "game_lower", "neutral_lower")
PANEL_TITLES = {
    "both_keep": "A  Both conditions keep baseline winner",
    "game_lower": "B  Game selects baseline rank 3 or 4",
    "neutral_lower": "C  Neutral selects baseline rank 3 or 4",
}
PANEL_YLABELS = {
    "both_keep": "Baseline runner-up minus\nranks 3-4 mean",
    "game_lower": "Game-selected option minus\nbaseline runner-up",
    "neutral_lower": "Neutral-selected option minus\nbaseline runner-up",
}


def _ranks(order: np.ndarray, choices: np.ndarray) -> np.ndarray:
    return np.asarray([
        int(np.flatnonzero(order[index] == choices[index])[0]) + 1
        for index in range(len(choices))
    ])


def _bootstrap(
    values: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mean = values.mean(axis=0)
    draws = np.empty((repetitions, values.shape[1]))
    for repetition in range(repetitions):
        sample = rng.integers(0, len(values), len(values))
        draws[repetition] = values[sample].mean(axis=0)
    return mean, np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def _metric(
    residuals: np.ndarray,
    order: np.ndarray,
    selected_choice: np.ndarray | None,
) -> np.ndarray:
    aligned = np.take_along_axis(residuals, order[:, None, :], axis=-1)
    if selected_choice is None:
        return aligned[:, :, 1] - aligned[:, :, 2:].mean(axis=-1)
    row = np.arange(len(residuals))
    selected = residuals[row, :, selected_choice]
    return selected - aligned[:, :, 1]


def _analyze_model(
    model: str,
    input_dir: str | Path,
    folds: int,
    seed: int,
    bootstrap: int,
) -> tuple[list[dict], dict]:
    data = load_activation_dataset(input_dir, ["baseline", *CONDITIONS])
    logits = centered(data.logits)
    baseline = logits[:, 0]
    order = np.argsort(-baseline[:, -1], axis=-1)
    winner = order[:, 0]
    game_choice = np.argmax(logits[:, 1, -1], axis=-1)
    neutral_choice = np.argmax(logits[:, 2, -1], axis=-1)
    game_rank = _ranks(order, game_choice)
    neutral_rank = _ranks(order, neutral_choice)
    masks = {
        "both_keep": (game_rank == 1) & (neutral_rank == 1),
        "game_lower": game_rank >= 3,
        "neutral_lower": neutral_rank >= 3,
    }
    selected = {
        "both_keep": None,
        "game_lower": game_choice,
        "neutral_lower": neutral_choice,
    }
    split = stratified_folds(winner, folds, seed)
    residuals = {}
    for condition_index, condition in enumerate(CONDITIONS, start=1):
        residuals[condition], _ = _cross_fitted_compression_residuals(
            baseline,
            logits[:, condition_index] - baseline,
            winner,
            split,
        )

    rows: list[dict] = []
    paired_summary = {}
    for panel_index, panel in enumerate(PANELS):
        mask = masks[panel]
        panel_values = {}
        for condition_index, condition in enumerate(CONDITIONS):
            choice = None if selected[panel] is None else selected[panel][mask]
            values = _metric(residuals[condition][mask], order[mask], choice)
            panel_values[condition] = values
            mean, low, high = _bootstrap(
                values,
                bootstrap,
                seed + panel_index * 100 + condition_index,
            )
            rows.append({
                "model": model,
                "panel": panel,
                "condition": condition,
                "n": int(mask.sum()),
                "mean": mean.tolist(),
                "ci_low": low.tolist(),
                "ci_high": high.tolist(),
            })

        primary, control = (
            ("neutral", "incorrect") if panel == "neutral_lower" else ("incorrect", "neutral")
        )
        paired = panel_values[primary] - panel_values[control]
        mean, low, high = _bootstrap(paired, bootstrap, seed + panel_index * 100 + 50)
        paired_summary[panel] = {
            "n": int(mask.sum()),
            "contrast": f"{CONDITION_LABELS[primary]} minus {CONDITION_LABELS[control]}",
            "final": {
                "mean": float(mean[-1]),
                "ci": [float(low[-1]), float(high[-1])],
            },
            "positive_ci_layers": np.flatnonzero(low > 0).astype(int).tolist(),
            "negative_ci_layers": np.flatnonzero(high < 0).astype(int).tolist(),
        }

    summary = {
        "model": model,
        "n_questions": len(winner),
        "final_layer": logits.shape[2] - 1,
        "group_counts": {panel: int(mask.sum()) for panel, mask in masks.items()},
        "paired_contrasts": paired_summary,
    }
    return rows, summary


def _plot(rows: list[dict], summaries: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    _style()
    models = [summary["model"] for summary in summaries]
    final_layers = {summary["model"]: summary["final_layer"] for summary in summaries}
    lookup = {(row["model"], row["panel"], row["condition"]): row for row in rows}
    figure, axes = plt.subplots(len(models), len(PANELS), figsize=(10.2, 5.2), squeeze=False)

    for row_index, model in enumerate(models):
        final_layer = final_layers[model]
        x = np.arange(final_layer + 1) / final_layer * 100
        for column_index, panel in enumerate(PANELS):
            axis = axes[row_index, column_index]
            for condition in CONDITIONS:
                result = lookup[(model, panel, condition)]
                mean = np.asarray(result["mean"])
                low = np.asarray(result["ci_low"])
                high = np.asarray(result["ci_high"])
                axis.plot(x, mean, color=CONDITION_COLORS[condition], lw=1.5, label=CONDITION_LABELS[condition])
                axis.fill_between(x, low, high, color=CONDITION_COLORS[condition], alpha=0.13, linewidth=0)
            axis.axhline(0, color="#555555", lw=0.65)
            axis.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.7)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_xlim(0, 100)
            panel_note = f"n={result['n']}"
            if column_index == 0:
                panel_note = f"{model}\n{panel_note}"
            axis.text(
                0.02,
                0.95,
                panel_note,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                fontweight="bold" if column_index == 0 else "normal",
            )
            if row_index == 0:
                axis.set_title(PANEL_TITLES[panel], loc="left", fontweight="bold")
            if row_index == len(models) - 1:
                axis.set_xlabel("Model depth (%)")
            axis.set_ylabel(PANEL_YLABELS[panel])
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, loc="lower left")

    # Share y scales down each column while retaining the sensitivity needed for
    # the much smaller both-keep contrast.
    for column_index in range(len(PANELS)):
        limits = [axes[row_index, column_index].get_ylim() for row_index in range(len(models))]
        low = min(value[0] for value in limits)
        high = max(value[1] for value in limits)
        for row_index in range(len(models)):
            axes[row_index, column_index].set_ylim(low, high)

    figure.tight_layout(w_pad=1.25, h_pad=1.1)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"symmetric_outcome_comparison.{suffix}", bbox_inches="tight")
    plt.close(figure)


def analyze(
    model_inputs: list[tuple[str, str]],
    output_dir: str | Path,
    folds: int,
    seed: int,
    bootstrap: int,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    summaries = []
    for index, (model, input_dir) in enumerate(model_inputs):
        model_rows, summary = _analyze_model(model, input_dir, folds, seed + index * 10000, bootstrap)
        rows.extend(model_rows)
        summaries.append(summary)

    long_rows = []
    for row in rows:
        for layer, (mean, low, high) in enumerate(zip(row["mean"], row["ci_low"], row["ci_high"])):
            long_rows.append({
                "model": row["model"],
                "panel": row["panel"],
                "condition": row["condition"],
                "n": row["n"],
                "layer": layer,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            })
    with (output / "symmetric_outcome_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=long_rows[0].keys())
        writer.writeheader()
        writer.writerows(long_rows)

    _plot(rows, summaries, output)
    summary = {
        "models": summaries,
        "residual_definition": (
            "Condition-minus-baseline centered pseudo-logits after cross-fitted removal of "
            "option-letter effects and proportional baseline-geometry compression."
        ),
        "intervals": f"{bootstrap}-draw question-clustered percentile bootstrap within each outcome-defined group.",
    }
    (output / "symmetric_outcome_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Symmetric Game- and Neutral-defined outcome analysis")
    parser.add_argument(
        "--model-input",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat once per model.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    model_inputs = []
    for item in args.model_input:
        label, separator, path = item.partition("=")
        if not separator:
            raise ValueError(f"Expected LABEL=PATH, got {item!r}")
        model_inputs.append((label, path))
    print(json.dumps(analyze(model_inputs, args.output, args.folds, args.seed, args.bootstrap), indent=2))


if __name__ == "__main__":
    main()
