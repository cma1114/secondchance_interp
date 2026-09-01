from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_rank_opposition import _baseline_order, _load, _rank_slope
from .component_causal_metrics import bootstrap, center, entropy
from .io import shard_path


def _load_reference(
    root: Path,
    group: str,
    qids: list[str],
    field: str,
    fallback: np.ndarray,
) -> tuple[np.ndarray, int]:
    values = []
    matched = 0
    for index, qid in enumerate(qids):
        with np.load(shard_path(root, group, qid), allow_pickle=False) as data:
            if field in data:
                values.append(data[field])
                matched += 1
            else:
                values.append(fallback[index])
    return np.asarray(values, dtype=np.float64), matched


def _macro(values: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean([values[labels == label].mean() for label in range(4)]))


def _mean(values: np.ndarray, labels: np.ndarray, aggregation: str) -> float:
    return float(values.mean()) if aggregation == "dataset" else _macro(values, labels)


def _metrics(values: np.ndarray, winners: np.ndarray) -> dict[str, np.ndarray]:
    centered = center(values)
    row = np.arange(len(values))
    winner = centered[row, winners]
    return {
        "switch": (np.argmax(centered, axis=-1) != winners).astype(float),
        "entropy": entropy(centered),
        "spread": centered.std(axis=-1),
        "winner_advantage": winner - (centered.sum(axis=-1) - winner) / 3.0,
    }


def analyze(
    patch_root: Path,
    plan_path: Path,
    ranking_root: Path,
    output: Path,
    samples: int = 10_000,
    seed: int = 42,
) -> dict:
    plan = json.loads(plan_path.read_text())
    qids = [
        qid for qid in plan["question_ids"]
        if all(
            shard_path(patch_root, group, qid).exists()
            for group in ["natural_game", "natural_neutral", *[row["id"] for row in plan["scenarios"]]]
        )
    ]
    if not qids:
        raise FileNotFoundError(f"No complete questions in {patch_root}")
    order, winners = _baseline_order(ranking_root, qids)
    natural = {
        "incorrect": _load(patch_root, "natural_game", qids),
        "neutral": _load(patch_root, "natural_neutral", qids),
    }
    natural_metrics = {condition: _metrics(values, winners) for condition, values in natural.items()}
    natural_rank = {
        condition: _rank_slope(np.take_along_axis(center(values), order, axis=-1))
        for condition, values in natural.items()
    }
    natural_rank_gap = natural_rank["incorrect"] - natural_rank["neutral"]
    rng = np.random.default_rng(seed)
    rows = []
    summaries = []
    for scenario in plan["scenarios"]:
        values = _load(patch_root, scenario["id"], qids)
        target = scenario["target_condition"]
        direction = "neutral_into_game" if target == "incorrect" else "game_into_neutral"
        sign = -1.0 if direction == "neutral_into_game" else 1.0
        reference, matched_count = _load_reference(
            patch_root,
            scenario["id"],
            qids,
            "matched_natural_logits",
            natural[target],
        )
        game_like = sign * (center(values) - center(reference))
        aligned = np.take_along_axis(game_like, order, axis=-1)
        reference_metrics = _metrics(reference, winners)
        effects = {
            "rank_slope": _rank_slope(aligned),
            **{
                metric: sign * (metric_values - reference_metrics[metric])
                for metric, metric_values in _metrics(values, winners).items()
            },
        }
        vector = np.mean(
            np.stack([aligned[winners == label].mean(axis=0) for label in range(4)]), axis=0
        )
        entry = {
            "scenario": scenario["id"],
            "direction": direction,
            "targets": scenario["targets"],
            "n_questions": len(qids),
            "rank_vector": vector.tolist(),
            "batch_matched_reference_count": matched_count,
        }
        for metric, effect in effects.items():
            entry[metric] = {}
            for aggregation in ("dataset", "letter_macro"):
                mean, low, high = bootstrap(effect, winners, aggregation, samples, rng)
                natural_gap = (
                    _mean(natural_rank_gap, winners, aggregation)
                    if metric == "rank_slope"
                    else _mean(
                        natural_metrics["incorrect"][metric]
                        - natural_metrics["neutral"][metric],
                        winners,
                        aggregation,
                    )
                )
                result = {
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "natural_game_minus_neutral_gap": natural_gap,
                    "fraction_gap_mediated": mean / natural_gap if abs(natural_gap) > 1e-12 else None,
                }
                entry[metric][aggregation] = result
                rows.append({
                    "scenario": scenario["id"],
                    "direction": direction,
                    "targets": "+".join(row["component"] for row in scenario["targets"]),
                    "metric": metric,
                    "aggregation": aggregation,
                    **result,
                })
        summaries.append(entry)

    payload = {
        "stage": plan.get("stage"),
        "n_questions": len(qids),
        "natural_rank_gap": {
            "dataset": float(natural_rank_gap.mean()),
            "letter_macro": _macro(natural_rank_gap, winners),
        },
        "scenarios": summaries,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "onset_patching_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    with (output / "onset_patching_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def select_heads(
    discovery_summary: Path,
    confirmation_split: Path,
    output: Path,
    per_metric: int = 3,
) -> dict:
    summary = json.loads(discovery_summary.read_text())
    split = json.loads(confirmation_split.read_text())
    candidates = []
    for row in summary["scenarios"]:
        target = row["targets"][0]
        candidates.append({
            **target,
            "rank_score": row["rank_slope"]["letter_macro"]["mean"],
            "switch_score": row["switch"]["dataset"]["mean"],
        })
    selected = []
    ranking = {}
    for layer in sorted({row["layer"] for row in candidates}):
        layer_rows = [row for row in candidates if row["layer"] == layer]
        rank_top = sorted(layer_rows, key=lambda row: row["rank_score"], reverse=True)[:per_metric]
        switch_top = sorted(layer_rows, key=lambda row: row["switch_score"], reverse=True)[:per_metric]
        union = {row["component"]: row for row in [*rank_top, *switch_top]}
        selected.extend(union.values())
        ranking[str(layer)] = {
            "rank_top": rank_top,
            "switch_top": switch_top,
            "selected": list(union.values()),
        }

    scenarios = []
    for target in selected:
        for source, target_condition, direction in (
            ("neutral", "incorrect", "neutral_into_game"),
            ("incorrect", "neutral", "game_into_neutral"),
        ):
            scenarios.append({
                "id": f"{direction}__{target['component']}",
                "source_condition": source,
                "target_condition": target_condition,
                "targets": [{key: target[key] for key in ("component", "layer", "heads")}],
            })
    groups = {}
    for layer in sorted({row["layer"] for row in selected}):
        groups[f"mixer_l{layer}_selected_heads"] = [
            {key: row[key] for key in ("component", "layer", "heads")}
            for row in selected if row["layer"] == layer
        ]
    groups["both_mixers_selected_heads"] = [
        {key: row[key] for key in ("component", "layer", "heads")} for row in selected
    ]
    for group_id, targets in groups.items():
        for source, target_condition, direction in (
            ("neutral", "incorrect", "neutral_into_game"),
            ("incorrect", "neutral", "game_into_neutral"),
        ):
            scenarios.append({
                "id": f"{direction}__{group_id}",
                "source_condition": source,
                "target_condition": target_condition,
                "targets": targets,
            })
    targets = [
        {key: row[key] for key in ("component", "layer", "heads")} for row in selected
    ]
    plan = {
        "stage": "onset_head_confirmation",
        "question_ids": split["question_ids"],
        "targets": targets,
        "groups": groups,
        "scenarios": scenarios,
        "discovery_ranking": ranking,
        "selection_rule": f"Union of top {per_metric} rank and top {per_metric} switch mediators per mixer",
        "discovery_summary": str(discovery_summary),
        "split_source": str(confirmation_split),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and select onset-circuit head patches")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--patch-root", required=True)
    analyze_parser.add_argument("--plan", required=True)
    analyze_parser.add_argument("--ranking-root", required=True)
    analyze_parser.add_argument("--output", required=True)
    analyze_parser.add_argument("--samples", type=int, default=10_000)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--discovery-summary", required=True)
    select_parser.add_argument("--confirmation-split", required=True)
    select_parser.add_argument("--output", required=True)
    select_parser.add_argument("--per-metric", type=int, default=3)
    args = parser.parse_args()
    if args.command == "analyze":
        analyze(Path(args.patch_root), Path(args.plan), Path(args.ranking_root), Path(args.output), args.samples)
    else:
        select_heads(Path(args.discovery_summary), Path(args.confirmation_split), Path(args.output), args.per_metric)


if __name__ == "__main__":
    main()
