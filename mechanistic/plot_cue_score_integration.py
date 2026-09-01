from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TARGETS = ("old_unique", "fresh_unique")
LABELS = {"old_unique": "Old 1P evidence", "fresh_unique": "Fresh 2P evidence"}
CONDITIONS = ("Game", "Neutral")
COLORS = {"Game": "#2478d4", "Neutral": "#e8792e", "Shared": "#333333"}


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = _center(left.astype(np.float64)).reshape(-1)
    right = _center(right.astype(np.float64)).reshape(-1)
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else float("nan")


def _bootstrap_correlation(
    prediction: np.ndarray,
    target: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        rows = rng.integers(0, len(target), size=len(target))
        values[index] = _correlation(prediction[rows], target[rows])
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "mean": _correlation(prediction, target),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _bootstrap_rank_profile(
    ranked: np.ndarray, seed: int, draws: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, 4), dtype=np.float64)
    for start in range(0, draws, 250):
        stop = min(start + 250, draws)
        rows = rng.integers(0, len(ranked), size=(stop - start, len(ranked)))
        samples[start:stop] = ranked[rows].mean(axis=1)
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    return ranked.mean(axis=0), intervals[0], intervals[1]


def _bootstrap_bivalent(
    values: np.ndarray, seed: int, draws: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """values: question x layer x rank, already Game minus Neutral."""
    bivalent = values[:, :, 3] - values[:, :, :2].mean(axis=2)
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, bivalent.shape[1]), dtype=np.float64)
    for start in range(0, draws, 200):
        stop = min(start + 200, draws)
        rows = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = bivalent[rows].mean(axis=1)
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    return bivalent.mean(axis=0), intervals[0], intervals[1]


def plot(args: argparse.Namespace) -> None:
    report = json.loads(args.result.read_text())
    arrays = np.load(args.projections)
    confirmation = ~arrays["discovery"].astype(bool)
    predictions = arrays["predictions"].astype(np.float32)[:, confirmation]
    rank_order = arrays["old_rank_order"].astype(int)[confirmation]
    targets = {
        "old_unique": arrays["old_unique"].astype(np.float32)[confirmation],
        "fresh_unique": arrays["fresh_unique"].astype(np.float32)[confirmation],
    }
    layers = np.arange(1, 65)

    summary: dict[str, Any] = {
        "evidence_label": "Held-out activation decoding; not a causal intervention.",
        "confirmation_questions": int(confirmation.sum()),
        "selected": {},
        "bivalent_trajectory": {},
    }
    for target_index, target_name in enumerate(TARGETS):
        selected_layer = int(report["selected"][target_name]["layer"]) - 1
        selected_predictions = predictions[:, :, selected_layer, :, target_index]
        rows: dict[str, Any] = {}
        for condition_index, condition in enumerate(CONDITIONS):
            rows[condition] = _bootstrap_correlation(
                selected_predictions[condition_index],
                targets[target_name],
                args.seed + 100 * target_index + condition_index,
                args.bootstrap_draws,
            )
        rows["Shared"] = _bootstrap_correlation(
            selected_predictions.mean(axis=0),
            targets[target_name],
            args.seed + 100 * target_index + 3,
            args.bootstrap_draws,
        )
        ranked = np.take_along_axis(
            _center(selected_predictions), rank_order[None, :, :], axis=2
        )
        rank_rows = {}
        for condition_index, condition in enumerate(CONDITIONS):
            mean, low, high = _bootstrap_rank_profile(
                ranked[condition_index],
                args.seed + 1000 * target_index + condition_index,
                args.bootstrap_draws,
            )
            rank_rows[condition] = {
                f"R{rank + 1}": {
                    "mean": float(mean[rank]),
                    "ci_low": float(low[rank]),
                    "ci_high": float(high[rank]),
                }
                for rank in range(4)
            }
        summary["selected"][target_name] = {
            "layer": selected_layer + 1,
            "correlations": rows,
            "decoded_profile_by_old_rank": rank_rows,
            "Game_minus_Neutral": report["selected"][target_name][
                "Game_minus_Neutral_by_old_rank"
            ],
            "Game_minus_Neutral_bivalent": report["selected"][target_name][
                "Game_minus_Neutral_bivalent"
            ],
        }

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    for target_index, target_name in enumerate(TARGETS):
        axis = axes[0, target_index]
        shared = []
        for condition in CONDITIONS:
            values = [
                report["trajectory"][str(layer)][target_name]["tasks"][condition][
                    "confirmation_correlation"
                ]
                for layer in layers
            ]
            axis.plot(layers, values, color=COLORS[condition], label=condition, linewidth=2)
        shared = [
            report["trajectory"][str(layer)][target_name][
                "shared_confirmation_correlation"
            ]
            for layer in layers
        ]
        axis.plot(layers, shared, color=COLORS["Shared"], label="Task mean", linewidth=1.8)
        selected_layer = int(report["selected"][target_name]["layer"])
        axis.axvline(selected_layer, color="#777777", linestyle="--", linewidth=1)
        axis.axhline(0, color="#999999", linewidth=0.8)
        axis.set_title(f"{chr(65 + target_index)}  Held-out {LABELS[target_name].lower()} decoding")
        axis.set_xlabel("Layer")
        axis.set_ylabel("Pooled candidate-score correlation")
        axis.set_xlim(1, 64)
        axis.legend(frameon=False, loc="upper left")

    old_layer = int(report["selected"]["old_unique"]["layer"]) - 1
    old_ranked = np.take_along_axis(
        _center(predictions[:, :, old_layer, :, 0]), rank_order[None, :, :], axis=2
    )
    axis = axes[1, 0]
    x = np.arange(4)
    for condition_index, condition in enumerate(CONDITIONS):
        mean, low, high = _bootstrap_rank_profile(
            old_ranked[condition_index],
            args.seed + 5000 + condition_index,
            args.bootstrap_draws,
        )
        offset = -0.10 if condition_index == 0 else 0.10
        axis.errorbar(
            x + offset,
            mean,
            yerr=np.stack([mean - low, high - mean]),
            marker="o",
            capsize=4,
            linewidth=2,
            color=COLORS[condition],
            label=condition,
        )
    axis.axhline(0, color="#999999", linewidth=0.8)
    axis.set_xticks(x, ["R1", "R2", "R3", "R4"])
    axis.set_title(f"C  Decoded old evidence at frozen layer {old_layer + 1}")
    axis.set_xlabel("First-presentation rank")
    axis.set_ylabel("Decoded unique old evidence")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    for target_index, target_name in enumerate(TARGETS):
        ranked_all = np.take_along_axis(
            _center(predictions[:, :, :, :, target_index]),
            rank_order[None, :, None, :],
            axis=3,
        )
        difference = ranked_all[0] - ranked_all[1]
        mean, low, high = _bootstrap_bivalent(
            difference,
            args.seed + 6000 + target_index,
            args.bootstrap_draws,
        )
        color = "#6a3d9a" if target_index == 0 else "#1b9e77"
        axis.plot(layers, mean, color=color, linewidth=2, label=LABELS[target_name])
        axis.fill_between(layers, low, high, color=color, alpha=0.20)
        summary["bivalent_trajectory"][target_name] = {
            "definition": "Game minus Neutral: R4 minus mean(R1,R2)",
            "mean": mean.tolist(),
            "ci_low": low.tolist(),
            "ci_high": high.tolist(),
            "layers_with_ci_excluding_zero": [
                int(layer)
                for layer, lower, upper in zip(layers, low, high)
                if lower > 0 or upper < 0
            ],
        }
    axis.axhline(0, color="#777777", linewidth=0.9)
    axis.set_xlim(1, 64)
    axis.set_title("D  Game–Neutral lower-rank redistribution")
    axis.set_xlabel("Layer")
    axis.set_ylabel("R4 − mean(R1,R2), decoded units")
    axis.legend(frameon=False)

    fig.suptitle("Old evidence, fresh evidence, and rank policy at the post-list choice cue", fontsize=16)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--projections", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=48333966)
    plot(parser.parse_args())


if __name__ == "__main__":
    main()
