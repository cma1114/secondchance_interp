from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .component_causal_metrics import (
    aggregate_mean,
    bootstrap,
    causal_geometry,
    center,
    outcome_metrics,
)
from .io import read_metadata, shard_path


def _load(root: Path, group: str, qids: list[str], key: str) -> np.ndarray:
    rows = []
    for qid in qids:
        with np.load(shard_path(root, group, qid), allow_pickle=False) as data:
            resolved_key = key
            if key not in data and key == "final_canonical_logits" and "canonical_logits" in data:
                resolved_key = "canonical_logits"
            values = data[resolved_key]
            if resolved_key == "canonical_logits" and values.ndim == 2:
                values = values[-1]
            rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def _plot(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    primary = [
        row for row in rows
        if row["aggregation"] == "dataset"
        and row["direction"] == "neutral_into_game"
        and row["n_targets"] == 1
    ]
    if not primary:
        return
    metric_specs = [
        ("ad_entropy", "A-D entropy", 1.0),
        ("ad_spread", "A-D logit spread", 1.0),
        ("winner_advantage", "Original-winner advantage", 1.0),
        ("switch", "Switch probability (pp)", 100.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, constrained_layout=True)
    for axis, (metric, title, scale) in zip(axes.ravel(), metric_specs):
        selected = sorted((row for row in primary if row["metric"] == metric), key=lambda row: (row["kind"], row["layer"]))
        for kind, color, marker in (("mixer", "#0072B2", "o"), ("mlp", "#009E73", "s")):
            subset = [row for row in selected if row["kind"] == kind]
            x = np.asarray([row["layer"] for row in subset], dtype=float)
            mean = scale * np.asarray([row["effect_mean"] for row in subset])
            low = scale * np.asarray([row["effect_ci_low"] for row in subset])
            high = scale * np.asarray([row["effect_ci_high"] for row in subset])
            axis.vlines(x, low, high, color=color, alpha=0.35, linewidth=1)
            axis.scatter(x, mean, color=color, marker=marker, s=24, label=kind)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel("Neutral-output patch - natural Game")
        axis.grid(axis="y", alpha=0.2)
    axes[-1, 0].set_xlabel("Zero-indexed layer")
    axes[-1, 1].set_xlabel("Zero-indexed layer")
    axes[0, 0].legend(frameon=False)
    fig.savefig(output / "causal_outcome_sweep.png", dpi=220)
    fig.savefig(output / "causal_outcome_sweep.svg")
    plt.close(fig)

    geometry_specs = [
        ("causal_total_l1", "Total absolute centered logit movement"),
        ("causal_baseline_coefficient", "Baseline-aligned coefficient"),
        ("causal_orthogonal_l2", "Orthogonal logit movement"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharex=True, constrained_layout=True)
    for axis, (metric, title) in zip(axes, geometry_specs):
        selected = sorted((row for row in primary if row["metric"] == metric), key=lambda row: (row["kind"], row["layer"]))
        for kind, color, marker in (("mixer", "#0072B2", "o"), ("mlp", "#009E73", "s")):
            subset = [row for row in selected if row["kind"] == kind]
            axis.scatter([row["layer"] for row in subset], [row["effect_mean"] for row in subset], color=color, marker=marker, s=24, label=kind)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Zero-indexed layer")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.savefig(output / "causal_geometry_sweep.png", dpi=220)
    fig.savefig(output / "causal_geometry_sweep.svg")
    plt.close(fig)

    rank_colors = ("#2E91E5", "#E15F33", "#1CA71C", "#FB0D98")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), sharex=True, sharey=True, constrained_layout=True)
    for axis, kind in zip(axes, ("mixer", "mlp")):
        for rank, color in zip(range(1, 5), rank_colors):
            subset = sorted(
                (
                    row for row in primary
                    if row["kind"] == kind
                    and row["metric"] == f"causal_rank_write_{rank}"
                ),
                key=lambda row: row["layer"],
            )
            x = np.asarray([row["layer"] + 1 for row in subset], dtype=float)
            mean = np.asarray([row["effect_mean"] for row in subset])
            low = np.asarray([row["effect_ci_low"] for row in subset])
            high = np.asarray([row["effect_ci_high"] for row in subset])
            axis.plot(x, mean, color=color, linewidth=1.5, label=f"Baseline rank {rank}")
            axis.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title("Mixer outputs" if kind == "mixer" else "MLP outputs")
        axis.set_xlabel("Layer")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Game-like causal write to final centered A-D logits")
    axes[0].legend(frameon=False, ncol=2)
    fig.savefig(output / "causal_rank_write_sweep.png", dpi=220)
    fig.savefig(output / "causal_rank_write_sweep.svg")
    plt.close(fig)


def _write_report(rows: list[dict], summary: dict, output: Path) -> None:
    lines = [
        "# Component causal sweep",
        "",
        f"Complete questions: {summary['n_questions']}/{summary['planned_questions']}.",
        "",
        "Effects are patch minus natural target condition. Dataset-weighted estimates explain the observed sample; equal-letter estimates test answer-letter generality.",
        "",
    ]
    group_rows = [
        row for row in rows
        if row["aggregation"] == "dataset"
        and row["metric"] in {"ad_entropy", "ad_spread", "winner_advantage", "switch"}
    ]
    scenarios = list(dict.fromkeys(row["scenario"] for row in group_rows))
    if len(scenarios) <= 30:
        lines.extend([
            "| Scenario | Direction | Entropy | Spread | Winner advantage | Switch |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for scenario in scenarios:
            values = {row["metric"]: row for row in group_rows if row["scenario"] == scenario}
            if len(values) != 4:
                continue
            def fmt(metric: str, scale: float = 1.0) -> str:
                row = values[metric]
                return f"{scale*row['effect_mean']:+.4f} [{scale*row['effect_ci_low']:+.4f}, {scale*row['effect_ci_high']:+.4f}]"
            direction = next(row["direction"] for row in group_rows if row["scenario"] == scenario)
            lines.append(f"| {scenario} | {direction} | {fmt('ad_entropy')} | {fmt('ad_spread')} | {fmt('winner_advantage')} | {fmt('switch', 100)} pp |")
    else:
        lines.append("The exhaustive individual-component results are in `component_causal_effects.csv`; figures show their layerwise structure.")
    lines.extend([
        "",
        "Geometry descriptors are causal changes in the final A-D logits. `causal_total_l1` is total absolute centered movement; `causal_baseline_coefficient` is movement along the Baseline evidence vector; `causal_orthogonal_l2` is the residual reorganization perpendicular to that vector.",
    ])
    (output / "COMPONENT_CAUSAL_REPORT.md").write_text("\n".join(lines))


def analyze(
    natural_root: Path,
    patch_root: Path,
    plan_path: Path,
    output: Path,
    samples: int,
    seed: int,
) -> dict:
    plan = json.loads(plan_path.read_text())
    planned_qids = plan.get("question_ids", plan.get("confirmation_question_ids", []))
    qids = [
        qid
        for qid in planned_qids
        if all(
            shard_path(patch_root, group, qid).exists()
            for group in ["natural_game", "natural_neutral", *[scenario["id"] for scenario in plan["scenarios"]]]
        )
    ]
    if not qids:
        raise FileNotFoundError("No questions have a complete set of planned causal shards")

    baseline = _load(natural_root, "baseline", qids, "final_canonical_logits")
    natural = {
        "incorrect": _load(patch_root, "natural_game", qids, "final_canonical_logits"),
        "neutral": _load(patch_root, "natural_neutral", qids, "final_canonical_logits"),
    }
    metadata = []
    for qid in qids:
        with np.load(shard_path(natural_root, "baseline", qid), allow_pickle=False) as data:
            metadata.append(read_metadata(data))
    winners = np.argmax(baseline, axis=-1)
    correct = np.asarray(["ABCD".index(row["correct_answer"]) for row in metadata])
    aggregations = ["dataset"]
    if all(np.any(winners == label) for label in range(4)):
        aggregations.append("letter_macro")
    natural_metrics = {condition: outcome_metrics(values, baseline, winners, correct) for condition, values in natural.items()}
    natural_gap = {
        metric: natural_metrics["incorrect"][metric] - natural_metrics["neutral"][metric]
        for metric in natural_metrics["incorrect"]
    }

    rows: list[dict] = []
    letter_rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for scenario in plan["scenarios"]:
        values = _load(patch_root, scenario["id"], qids, "final_canonical_logits")
        target = scenario["target_condition"]
        metrics = outcome_metrics(values, baseline, winners, correct)
        effects = {metric: result - natural_metrics[target][metric] for metric, result in metrics.items()}
        effects.update(causal_geometry(values, natural[target], baseline))
        direction = "neutral_into_game" if target == "incorrect" else "game_into_neutral"
        game_like_sign = -1.0 if direction == "neutral_into_game" else 1.0
        baseline_order = np.argsort(-center(baseline), axis=-1)
        game_like_delta = game_like_sign * (
            center(values) - center(natural[target])
        )
        aligned_game_like_delta = np.take_along_axis(
            game_like_delta, baseline_order, axis=-1
        )
        for rank in range(4):
            effects[f"causal_rank_write_{rank + 1}"] = aligned_game_like_delta[:, rank]
        target_info = scenario["targets"][0] if len(scenario["targets"]) == 1 else None
        for metric, effect in effects.items():
            is_outcome = metric in natural_gap
            for aggregation in aggregations:
                mean, low, high = bootstrap(effect, winners, aggregation, samples, rng)
                gap_mean = aggregate_mean(natural_gap[metric], winners, aggregation) if is_outcome else None
                if is_outcome and abs(gap_mean) > 1e-12:
                    fraction = (-mean / gap_mean) if target == "incorrect" else (mean / gap_mean)
                else:
                    fraction = None
                rows.append({
                    "scenario": scenario["id"],
                    "direction": direction,
                    "component": None if target_info is None else target_info["component"],
                    "kind": None if target_info is None else target_info["kind"],
                    "layer": None if target_info is None else target_info["layer"],
                    "anchor": None if target_info is None else target_info.get("anchor", "decision"),
                    "n_targets": len(scenario["targets"]),
                    "aggregation": aggregation,
                    "metric": metric,
                    "n_questions": len(qids),
                    "effect_mean": mean,
                    "effect_ci_low": low,
                    "effect_ci_high": high,
                    "natural_game_minus_neutral_gap": gap_mean,
                    "fraction_gap_mediated": fraction,
                })
            for label in range(4):
                mask = winners == label
                letter_rows.append({
                    "scenario": scenario["id"],
                    "direction": direction,
                    "component": None if target_info is None else target_info["component"],
                    "anchor": None if target_info is None else target_info.get("anchor", "decision"),
                    "metric": metric,
                    "baseline_winner": "ABCD"[label],
                    "n_questions": int(mask.sum()),
                    "effect_mean": float(effect[mask].mean()),
                    "natural_game_minus_neutral_gap": (
                        float(natural_gap[metric][mask].mean()) if is_outcome else None
                    ),
                })

    output.mkdir(parents=True, exist_ok=True)
    effects_path = output / "component_causal_effects.csv"
    with effects_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "component_causal_effects_by_letter.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(letter_rows[0]))
        writer.writeheader()
        writer.writerows(letter_rows)
    summary = {
        "n_questions": len(qids),
        "planned_questions": len(planned_qids),
        "complete": len(qids) == len(planned_qids),
        "baseline_winner_counts": {"ABCD"[label]: int(np.sum(winners == label)) for label in range(4)},
        "natural_dataset": {
            condition: {metric: float(values.mean()) for metric, values in metrics.items()}
            for condition, metrics in natural_metrics.items()
        },
        "natural_letter_macro": (
            {
                condition: {metric: aggregate_mean(values, winners, "letter_macro") for metric, values in metrics.items()}
                for condition, metrics in natural_metrics.items()
            }
            if "letter_macro" in aggregations else None
        ),
        "metrics": {
            "causal_total_l1": "Sum of absolute centered final A-D logit changes caused by the patch.",
            "causal_baseline_coefficient": "Coefficient of the patch-induced final-logit change along the same-question Baseline evidence vector.",
            "causal_orthogonal_l2": "L2 magnitude of the patch-induced final-logit change orthogonal to the Baseline evidence vector.",
            "ad_entropy": "Patch-minus-natural change in final A-D entropy.",
            "ad_spread": "Patch-minus-natural change in final centered A-D logit standard deviation.",
            "rank_opposed_slope": "Patch-minus-natural change in the final Baseline-rank opposition slope.",
            "causal_rank_write_1_to_4": "Game-like causal final-logit write aligned by each question's Baseline A-D ranking.",
        },
    }
    (output / "component_causal_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    individual_anchors = sorted({
        row["anchor"] for row in rows
        if row["anchor"] is not None and row["n_targets"] == 1
    })
    if len(individual_anchors) == 1:
        _plot(rows, output)
    else:
        for anchor in individual_anchors:
            anchor_output = output / anchor
            anchor_output.mkdir(parents=True, exist_ok=True)
            _plot([row for row in rows if row["anchor"] == anchor], anchor_output)
    _write_report(rows, summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an exhaustive or confirmatory component causal sweep")
    parser.add_argument("--natural-root", required=True)
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(Path(args.natural_root), Path(args.patch_root), Path(args.plan), Path(args.output), args.bootstrap_samples, args.seed)


if __name__ == "__main__":
    main()
