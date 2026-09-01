from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .component_causal_metrics import RANK_AXIS, RANK_AXIS_DENOMINATOR, center
from .io import shard_path


def _load(root: Path, group: str, qids: list[str]) -> np.ndarray:
    rows = []
    for qid in qids:
        with np.load(shard_path(root, group, qid), allow_pickle=False) as data:
            rows.append(data["final_canonical_logits"])
    return np.asarray(rows, dtype=np.float64)


def _metrics(
    logits: np.ndarray, baseline: np.ndarray, winners: np.ndarray
) -> dict[str, np.ndarray]:
    centered = center(logits)
    baseline_centered = center(baseline)
    order = np.argsort(-baseline_centered, axis=-1)
    delta = centered - baseline_centered
    aligned = np.take_along_axis(delta, order, axis=-1)
    return {
        "switch": (np.argmax(centered, axis=-1) != winners).astype(np.float64),
        "rank_redistribution": (
            np.sum(aligned * RANK_AXIS, axis=-1) / RANK_AXIS_DENOMINATOR
        ),
    }


def _bootstrap_mean(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        stop = min(samples, start + 1000)
        index = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[index].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _bootstrap_profile(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    draws = np.empty((samples, values.shape[1]), dtype=np.float64)
    for start in range(0, samples, 1000):
        stop = min(samples, start + 1000)
        index = rng.integers(0, len(values), size=(stop - start, len(values)))
        draws[start:stop] = values[index].mean(axis=1)
    return (
        values.mean(axis=0),
        np.quantile(draws, 0.025, axis=0),
        np.quantile(draws, 0.975, axis=0),
    )


def analyze(
    patch_root: Path,
    plan_path: Path,
    output: Path,
    samples: int,
    seed: int,
) -> list[dict]:
    plan = json.loads(plan_path.read_text())
    metadata = json.loads((patch_root / "run_metadata.json").read_text())
    routes = [
        {"id": route_id, **values}
        for route_id, values in metadata["routes"].items()
    ]
    required = [
        "natural_baseline",
        "natural_game",
        "natural_neutral",
        *[
            f"{direction}__{suffix}"
            for direction in ("game_into_neutral", "neutral_into_game")
            for suffix in ("full", *[route["id"] for route in routes])
        ],
    ]
    qids = [
        qid
        for qid in plan["question_ids"]
        if all(shard_path(patch_root, group, qid).exists() for group in required)
    ]
    if not qids:
        raise FileNotFoundError("No complete held-out source-route questions")

    baseline = _load(patch_root, "natural_baseline", qids)
    natural_game = _load(patch_root, "natural_game", qids)
    natural_neutral = _load(patch_root, "natural_neutral", qids)
    winners = np.argmax(center(baseline), axis=-1)
    order = np.argsort(-center(baseline), axis=-1)
    natural_metrics = {
        "game": _metrics(natural_game, baseline, winners),
        "neutral": _metrics(natural_neutral, baseline, winners),
    }
    gaps = {
        metric: natural_metrics["game"][metric] - natural_metrics["neutral"][metric]
        for metric in natural_metrics["game"]
    }
    rank_gap = float(gaps["rank_redistribution"].mean())
    switch_gap = float(gaps["switch"].mean())

    direction_spec = {
        "game_into_neutral": ("neutral", 1.0),
        "neutral_into_game": ("game", -1.0),
    }
    rng = np.random.default_rng(seed)
    rows = []
    profiles: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    full_summary = {}
    for direction, (target, sign) in direction_spec.items():
        full_logits = _load(patch_root, f"{direction}__full", qids)
        full_metrics = _metrics(full_logits, baseline, winners)
        full_effect = {
            metric: sign * (full_metrics[metric] - natural_metrics[target][metric])
            for metric in full_metrics
        }
        full_rank = _bootstrap_mean(full_effect["rank_redistribution"], samples, rng)
        full_switch = _bootstrap_mean(full_effect["switch"], samples, rng)
        full_summary[direction] = {
            "rank_fraction": full_rank[0] / rank_gap,
            "rank_fraction_ci": [full_rank[1] / rank_gap, full_rank[2] / rank_gap],
            "switch_pp": 100.0 * full_switch[0],
            "switch_pp_ci": [100.0 * full_switch[1], 100.0 * full_switch[2]],
        }
        for route in routes:
            ablated_logits = _load(
                patch_root, f"{direction}__{route['id']}", qids
            )
            ablated_metrics = _metrics(ablated_logits, baseline, winners)
            ablated_effect = {
                metric: sign
                * (ablated_metrics[metric] - natural_metrics[target][metric])
                for metric in ablated_metrics
            }
            contribution = {
                metric: full_effect[metric] - ablated_effect[metric]
                for metric in full_effect
            }
            rank = _bootstrap_mean(
                contribution["rank_redistribution"], samples, rng
            )
            switch = _bootstrap_mean(contribution["switch"], samples, rng)

            # The logit-space route contribution uses the same Game-like sign
            # convention. Align and center it by each question's Baseline ranks.
            logit_contribution = sign * (
                center(full_logits) - center(ablated_logits)
            )
            aligned = np.take_along_axis(logit_contribution, order, axis=-1)
            profile = _bootstrap_profile(aligned, samples, rng)
            profiles[(direction, route["id"])] = profile
            rows.append(
                {
                    **route,
                    "direction": direction,
                    "n_questions": len(qids),
                    "rank_fraction": rank[0] / rank_gap,
                    "rank_fraction_ci_low": rank[1] / rank_gap,
                    "rank_fraction_ci_high": rank[2] / rank_gap,
                    "switch_pp": 100.0 * switch[0],
                    "switch_pp_ci_low": 100.0 * switch[1],
                    "switch_pp_ci_high": 100.0 * switch[2],
                    **{
                        f"rank_{index + 1}_logit": float(profile[0][index])
                        for index in range(4)
                    },
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "conditional_source_route_effects.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "complete": len(qids) == len(plan["question_ids"]),
        "n_questions": len(qids),
        "natural_rank_redistribution_gap": rank_gap,
        "natural_switch_gap_pp": 100.0 * switch_gap,
        "full_eight_component_effect": full_summary,
    }
    (output / "conditional_source_route_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    _plot(rows, profiles, output)
    _report(rows, summary, output)
    return rows


def _route_label(row: dict) -> str:
    source = row["source"]
    if source.startswith("repeated_option_"):
        source = f"option {source[-1]}"
    elif source == "second_choice_cue":
        source = "choice cue"
    elif source == "final_assistant_prefix":
        source = "assistant prefix"
    else:
        source = source.replace("_", " ")
    return f"H{row['head']}\n{source}"


def _plot(
    rows: list[dict],
    profiles: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    directions = ("game_into_neutral", "neutral_into_game")
    colors = {"game_into_neutral": "#E15F33", "neutral_into_game": "#2E91E5"}
    labels = {
        "game_into_neutral": "Insert Game-like computation into Neutral",
        "neutral_into_game": "Remove Game-like computation from Game",
    }
    reference = {row["id"]: row for row in rows if row["direction"] == directions[0]}
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), constrained_layout=True)
    width = 0.38
    for row_index, (kind, component_title) in enumerate(
        (("attention", "Mixer 56 attention routes"), ("gdn", "Mixer 63 DeltaNet routes"))
    ):
        route_order = [
            row["id"]
            for row in rows
            if row["direction"] == directions[0] and row["kind"] == kind
        ]
        x = np.arange(len(route_order))
        for column, (metric, scale, metric_title, ylabel) in enumerate(
            (
                (
                    "rank_fraction",
                    100.0,
                    "Ordered answer-rank redistribution",
                    "Percent of natural Game−Neutral gap",
                ),
                (
                    "switch_pp",
                    1.0,
                    "Switching away from Baseline winner",
                    "Percentage points",
                ),
            )
        ):
            axis = axes[row_index, column]
            for offset, direction in (
                (-width / 2, directions[0]),
                (width / 2, directions[1]),
            ):
                lookup = {
                    row["id"]: row
                    for row in rows
                    if row["direction"] == direction
                }
                values = scale * np.asarray(
                    [lookup[route][metric] for route in route_order]
                )
                low = scale * np.asarray(
                    [lookup[route][f"{metric}_ci_low"] for route in route_order]
                )
                high = scale * np.asarray(
                    [lookup[route][f"{metric}_ci_high"] for route in route_order]
                )
                axis.bar(
                    x + offset,
                    values,
                    width,
                    color=colors[direction],
                    label=labels[direction],
                )
                axis.errorbar(
                    x + offset,
                    values,
                    yerr=np.vstack([values - low, high - values]),
                    fmt="none",
                    color="black",
                    linewidth=0.8,
                    capsize=2,
                )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(f"{component_title}: {metric_title}", loc="left")
            axis.set_ylabel(ylabel)
            axis.set_xticks(
                x, [_route_label(reference[route]) for route in route_order]
            )
            axis.grid(axis="y", alpha=0.2)
            if row_index == 0:
                axis.legend(frameon=False, fontsize=9)
    fig.savefig(output / "conditional_source_route_effects.png", dpi=220)
    fig.savefig(output / "conditional_source_route_effects.svg")
    plt.close(fig)


def _report(rows: list[dict], summary: dict, output: Path) -> None:
    lines = [
        "# Held-out conditional source-route results",
        "",
        f"Confirmation questions: {summary['n_questions']}.",
        "",
        "Each row removes one selected source/head route from its Mixer 56 or Mixer 63 source output while the other seven selected components and every other route remain patched. Positive estimates mean that route contributes to the Game-like transformation.",
        "",
        "| Direction | Route | Source span | Ordered-rank gap | Switching |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {direction} | {kind} H{head} | {source} | {rank:.1f}% "
            "[{rank_low:.1f}, {rank_high:.1f}] | {switch:.2f} pp "
            "[{switch_low:.2f}, {switch_high:.2f}] |".format(
                direction=row["direction"],
                kind="Mixer 56" if row["kind"] == "attention" else "Mixer 63",
                head=row["head"],
                source=row["source"],
                rank=100.0 * row["rank_fraction"],
                rank_low=100.0 * row["rank_fraction_ci_low"],
                rank_high=100.0 * row["rank_fraction_ci_high"],
                switch=row["switch_pp"],
                switch_low=row["switch_pp_ci_low"],
                switch_high=row["switch_pp_ci_high"],
            )
        )
    (output / "CONDITIONAL_SOURCE_ROUTE_RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze held-out conditional source/head route interventions"
    )
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = analyze(
        Path(args.patch_root),
        Path(args.plan),
        Path(args.output),
        args.bootstrap_samples,
        args.seed,
    )
    print(f"Analyzed {len(rows)} direction-specific source-route effects")


if __name__ == "__main__":
    main()
