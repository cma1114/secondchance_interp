from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SUMMARIES = ("line_mean", "content_mean", "last_content", "newline")
TARGETS = ("old_unique", "fresh_unique")


def _ranked(values: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    return np.stack(
        [values[np.arange(len(values)), ranks[:, rank]] for rank in range(4)],
        axis=1,
    )


def _bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty((draws,) + values.shape[1:], dtype=np.float64)
    for start in range(0, draws, 250):
        stop = min(start + 250, draws)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(1)
    return values.mean(0), np.quantile(means, [0.025, 0.975], axis=0)


def _record(point: np.ndarray, interval: np.ndarray) -> Any:
    if np.ndim(point) == 0:
        return {
            "mean": float(point),
            "ci": [float(interval[0]), float(interval[1])],
        }
    return [
        {
            "mean": float(point[index]),
            "ci": [float(interval[0, index]), float(interval[1, index])],
        }
        for index in range(len(point))
    ]


def summarize(args: argparse.Namespace) -> None:
    trajectory = json.loads(args.trajectory.read_text())
    with np.load(args.projections, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    projections = arrays["projections"].astype(np.float64)
    discovery = arrays["discovery"].astype(bool)
    confirmation = ~discovery
    ranks = arrays["rank_indices"].astype(int)
    rng = np.random.default_rng(args.seed)

    normalized = np.empty_like(projections)
    for layer in range(64):
        for summary in range(len(SUMMARIES)):
            for target in range(len(TARGETS)):
                shared = (
                    projections[0, :, layer, :, summary, target]
                    + projections[1, :, layer, :, summary, target]
                ) / 2.0
                scale = shared[discovery].std()
                if not np.isfinite(scale) or scale <= 0:
                    raise RuntimeError("Invalid discovery projection scale")
                normalized[:, :, layer, :, summary, target] = (
                    projections[:, :, layer, :, summary, target] / scale
                )

    result: dict[str, Any] = {
        "definition": {
            "units": "Each layer/summary/target is divided by the discovery SD of the condition-mean score projection.",
            "rank_effect": "Game minus Neutral projection after mapping semantic candidates back to first-pass ranks R1--R4.",
            "bivalent_contrast": "R4 minus the mean of R1 and R2; selected on discovery and evaluated on confirmation.",
        },
        "selected": {},
        "rank_trajectory": {},
    }

    layers = np.arange(1, 65)
    rank_points: dict[tuple[int, int], np.ndarray] = {}
    rank_intervals: dict[tuple[int, int], np.ndarray] = {}
    for summary_index, summary_name in enumerate(SUMMARIES):
        summary_rows: dict[str, Any] = {}
        for target_index, target_name in enumerate(TARGETS):
            split_rows: dict[str, Any] = {}
            for split_name, mask in (
                ("discovery", discovery),
                ("confirmation", confirmation),
            ):
                points = np.empty((64, 4), dtype=np.float64)
                intervals = np.empty((2, 64, 4), dtype=np.float64)
                bivalent = np.empty(64, dtype=np.float64)
                bivalent_interval = np.empty((2, 64), dtype=np.float64)
                for layer in range(64):
                    delta = (
                        normalized[0, :, layer, :, summary_index, target_index]
                        - normalized[1, :, layer, :, summary_index, target_index]
                    )
                    rank_values = _ranked(delta, ranks)[mask]
                    point, interval = _bootstrap_mean(
                        rank_values, rng, args.bootstrap_draws
                    )
                    points[layer] = point
                    intervals[:, layer] = interval
                    candidate = rank_values[:, 3] - rank_values[:, :2].mean(1)
                    candidate_point, candidate_interval = _bootstrap_mean(
                        candidate, rng, args.bootstrap_draws
                    )
                    bivalent[layer] = candidate_point
                    bivalent_interval[:, layer] = candidate_interval
                split_rows[split_name] = {
                    "rank_means": points.tolist(),
                    "rank_ci_low": intervals[0].tolist(),
                    "rank_ci_high": intervals[1].tolist(),
                    "bivalent": bivalent.tolist(),
                    "bivalent_ci_low": bivalent_interval[0].tolist(),
                    "bivalent_ci_high": bivalent_interval[1].tolist(),
                }
                if split_name == "confirmation":
                    rank_points[(summary_index, target_index)] = points
                    rank_intervals[(summary_index, target_index)] = intervals
            discovery_bivalent = np.asarray(split_rows["discovery"]["bivalent"])
            selected_layer = int(np.argmax(discovery_bivalent))
            result["selected"][f"{summary_name}:{target_name}"] = {
                "layer": selected_layer + 1,
                "discovery_bivalent": _record(
                    np.asarray(split_rows["discovery"]["bivalent"])[selected_layer],
                    np.asarray(
                        [
                            split_rows["discovery"]["bivalent_ci_low"],
                            split_rows["discovery"]["bivalent_ci_high"],
                        ]
                    )[:, selected_layer],
                ),
                "confirmation_bivalent": _record(
                    np.asarray(split_rows["confirmation"]["bivalent"])[selected_layer],
                    np.asarray(
                        [
                            split_rows["confirmation"]["bivalent_ci_low"],
                            split_rows["confirmation"]["bivalent_ci_high"],
                        ]
                    )[:, selected_layer],
                ),
                "confirmation_rank_means": _record(
                    np.asarray(split_rows["confirmation"]["rank_means"])[selected_layer],
                    np.asarray(
                        [
                            split_rows["confirmation"]["rank_ci_low"],
                            split_rows["confirmation"]["rank_ci_high"],
                        ]
                    )[:, selected_layer],
                ),
                "shared_correlations": trajectory["trajectory"][
                    str(selected_layer + 1)
                ][summary_name][target_name],
            }
            summary_rows[target_name] = split_rows
        result["rank_trajectory"][summary_name] = summary_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "score_residual_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    colors = ("#2f70b7", "#d1495b", "#2a9d6f", "#8d63b8")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    for target_index, (target_name, title) in enumerate(
        (("old_unique", "Unique old 1P score"), ("fresh_unique", "Unique fresh 2P score"))
    ):
        ax = axes[0, target_index]
        for summary_index, summary_name in enumerate(SUMMARIES):
            values = [
                trajectory["trajectory"][str(layer)][summary_name][target_name][
                    "shared_confirmation_correlation"
                ]
                for layer in layers
            ]
            ax.plot(layers, values, label=summary_name.replace("_", " "), color=colors[summary_index])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{title}: held-out decoding")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Correlation")
        ax.legend(frameon=False, fontsize=9)

    for panel, (summary_index, target_index, title) in enumerate(
        (
            (2, 0, "Game − Neutral along old-score direction\n(final semantic token)"),
            (0, 1, "Game − Neutral along fresh-score direction\n(complete 2P line)"),
        )
    ):
        ax = axes[1, panel]
        points = rank_points[(summary_index, target_index)]
        intervals = rank_intervals[(summary_index, target_index)]
        for rank in range(4):
            ax.plot(layers, points[:, rank], label=f"R{rank + 1}", color=colors[rank])
            ax.fill_between(
                layers,
                intervals[0, :, rank],
                intervals[1, :, rank],
                color=colors[rank],
                alpha=0.16,
                linewidth=0,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(48, color="0.4", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Discovery-SD units")
        ax.legend(frameon=False, ncol=2, fontsize=9)

    fig.suptitle(
        "Old and fresh candidate evidence are jointly readable in 2P; the policy-dependent rank pattern emerges in semantic content",
        fontsize=15,
    )
    fig.savefig(args.figure, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"complete": True, "figure": str(args.figure)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--projections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=48333965)
    summarize(parser.parse_args())


if __name__ == "__main__":
    main()
