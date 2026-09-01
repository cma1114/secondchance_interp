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
            key = (
                "final_canonical_logits"
                if "final_canonical_logits" in data
                else "canonical_logits"
            )
            values = data[key]
            if values.ndim == 2:
                values = values[-1]
            rows.append(values)
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
        stop = min(start + 1000, samples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _label(component: str) -> str:
    anchor, kind, layer = component.split("__")[0], component.split("__")[1].split("_")[0], component.rsplit("_l", 1)[1]
    block = int(layer) + 1
    prefix = "Feedback" if anchor == "feedback_end" else "Decision"
    return f"{prefix} {kind.upper()} {block}"


def analyze(
    natural_root: Path,
    patch_root: Path,
    plan_path: Path,
    output: Path,
    samples: int,
    seed: int,
) -> list[dict]:
    plan = json.loads(plan_path.read_text())
    groups = [scenario["id"] for scenario in plan["scenarios"]]
    qids = [
        qid
        for qid in plan["question_ids"]
        if all(shard_path(patch_root, group, qid).exists() for group in groups)
        and shard_path(patch_root, "natural_game", qid).exists()
        and shard_path(patch_root, "natural_neutral", qid).exists()
        and shard_path(natural_root, "baseline", qid).exists()
    ]
    if not qids:
        raise FileNotFoundError("No complete interaction questions")

    baseline = _load(natural_root, "baseline", qids)
    winners = np.argmax(center(baseline), axis=-1)
    natural = {
        "neutral_into_game": _load(patch_root, "natural_game", qids),
        "game_into_neutral": _load(patch_root, "natural_neutral", qids),
    }
    natural_metrics = {
        direction: _metrics(values, baseline, winners)
        for direction, values in natural.items()
    }
    game_metrics = _metrics(_load(patch_root, "natural_game", qids), baseline, winners)
    neutral_metrics = _metrics(_load(patch_root, "natural_neutral", qids), baseline, winners)
    natural_gap = {
        metric: game_metrics[metric] - neutral_metrics[metric]
        for metric in game_metrics
    }

    scenario_lookup = {scenario["id"]: scenario for scenario in plan["scenarios"]}
    game_like: dict[str, dict[str, np.ndarray]] = {}
    for scenario_id, scenario in scenario_lookup.items():
        direction = (
            "neutral_into_game"
            if scenario["target_condition"] == "incorrect"
            else "game_into_neutral"
        )
        patched_metrics = _metrics(_load(patch_root, scenario_id, qids), baseline, winners)
        sign = -1.0 if direction == "neutral_into_game" else 1.0
        game_like[scenario_id] = {
            metric: sign * (patched_metrics[metric] - natural_metrics[direction][metric])
            for metric in patched_metrics
        }

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    ordered = plan["ordered_components"]
    for direction in ("neutral_into_game", "game_into_neutral"):
        full_id = f"{direction}__cumulative_08"
        full = game_like[full_id]
        for count, component in enumerate(ordered, 1):
            scenario_id = f"{direction}__cumulative_{count:02d}"
            effects = game_like[scenario_id]
            rank_mean, rank_low, rank_high = _bootstrap_mean(
                effects["rank_redistribution"], samples, rng
            )
            switch_mean, switch_low, switch_high = _bootstrap_mean(
                effects["switch"], samples, rng
            )
            rank_gap = float(natural_gap["rank_redistribution"].mean())
            rows.append(
                {
                    "diagnostic": "cumulative",
                    "direction": direction,
                    "step": count,
                    "component": component,
                    "component_label": _label(component),
                    "n_questions": len(qids),
                    "rank_fraction": rank_mean / rank_gap,
                    "rank_fraction_ci_low": rank_low / rank_gap,
                    "rank_fraction_ci_high": rank_high / rank_gap,
                    "switch_pp": 100.0 * switch_mean,
                    "switch_pp_ci_low": 100.0 * switch_low,
                    "switch_pp_ci_high": 100.0 * switch_high,
                }
            )

        for omitted_index, component in enumerate(ordered, 1):
            scenario_id = f"{direction}__leave_out_{omitted_index:02d}"
            conditional = {
                metric: full[metric] - game_like[scenario_id][metric]
                for metric in full
            }
            rank_mean, rank_low, rank_high = _bootstrap_mean(
                conditional["rank_redistribution"], samples, rng
            )
            switch_mean, switch_low, switch_high = _bootstrap_mean(
                conditional["switch"], samples, rng
            )
            rank_gap = float(natural_gap["rank_redistribution"].mean())
            rows.append(
                {
                    "diagnostic": "leave_one_out",
                    "direction": direction,
                    "step": omitted_index,
                    "component": component,
                    "component_label": _label(component),
                    "n_questions": len(qids),
                    "rank_fraction": rank_mean / rank_gap,
                    "rank_fraction_ci_low": rank_low / rank_gap,
                    "rank_fraction_ci_high": rank_high / rank_gap,
                    "switch_pp": 100.0 * switch_mean,
                    "switch_pp_ci_low": 100.0 * switch_low,
                    "switch_pp_ci_high": 100.0 * switch_high,
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "interaction_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _plot(rows, output)
    summary = {
        "n_questions": len(qids),
        "complete": len(qids) == len(plan["question_ids"]),
        "natural_switch_gap_pp": 100.0 * float(natural_gap["switch"].mean()),
        "natural_rank_redistribution_gap": float(
            natural_gap["rank_redistribution"].mean()
        ),
        "ordered_components": ordered,
    }
    (output / "interaction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return rows


def _plot(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"neutral_into_game": "#2E91E5", "game_into_neutral": "#E15F33"}
    labels = {"neutral_into_game": "Remove from Game", "game_into_neutral": "Insert into Neutral"}
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for col, (metric, title, scale) in enumerate(
        (("rank_fraction", "Ordered answer-rank redistribution", 100.0), ("switch_pp", "Switching", 1.0))
    ):
        axis = axes[0, col]
        for direction in colors:
            subset = sorted(
                (row for row in rows if row["diagnostic"] == "cumulative" and row["direction"] == direction),
                key=lambda row: row["step"],
            )
            x = np.asarray([row["step"] for row in subset])
            y = scale * np.asarray([row[metric] for row in subset])
            low = scale * np.asarray([row[f"{metric}_ci_low"] for row in subset])
            high = scale * np.asarray([row[f"{metric}_ci_high"] for row in subset])
            axis.plot(x, y, marker="o", color=colors[direction], label=labels[direction])
            axis.fill_between(x, low, high, color=colors[direction], alpha=0.12)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"Cumulative: {title}", loc="left")
        axis.set_xlabel("Components patched in causal order")
        axis.set_xticks(range(1, 9))
        axis.set_ylabel("Percent of Game-Neutral gap" if metric == "rank_fraction" else "Percentage points")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)

        axis = axes[1, col]
        directions = list(colors)
        subset0 = sorted(
            (row for row in rows if row["diagnostic"] == "leave_one_out" and row["direction"] == directions[0]),
            key=lambda row: row["step"],
        )
        x = np.arange(len(subset0))
        width = 0.38
        for offset, direction in ((-width / 2, directions[0]), (width / 2, directions[1])):
            subset = sorted(
                (row for row in rows if row["diagnostic"] == "leave_one_out" and row["direction"] == direction),
                key=lambda row: row["step"],
            )
            y = scale * np.asarray([row[metric] for row in subset])
            low = scale * np.asarray([row[f"{metric}_ci_low"] for row in subset])
            high = scale * np.asarray([row[f"{metric}_ci_high"] for row in subset])
            axis.bar(x + offset, y, width, color=colors[direction], alpha=0.85, label=labels[direction])
            axis.errorbar(x + offset, y, yerr=np.vstack([y - low, high - y]), fmt="none", color="black", linewidth=0.8, capsize=2)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"Conditional contribution: {title}", loc="left")
        axis.set_xticks(x, [row["component_label"].replace(" ", "\n", 1) for row in subset0], fontsize=8)
        axis.set_ylabel("Percent of Game-Neutral gap" if metric == "rank_fraction" else "Percentage points")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    fig.savefig(output / "component_interactions.png", dpi=220)
    fig.savefig(output / "component_interactions.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze cumulative and leave-one-out component patches"
    )
    parser.add_argument("--natural-root", required=True)
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = analyze(
        Path(args.natural_root),
        Path(args.patch_root),
        Path(args.plan),
        Path(args.output),
        args.bootstrap_samples,
        args.seed,
    )
    print(f"Analyzed {len(rows)} cumulative/conditional effects")


if __name__ == "__main__":
    main()
