from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_attention_source_ablation import SOURCE_SELECTORS


def _select_heads(
    confirmation_summary: Path,
    heads_per_layer: int = 2,
) -> tuple[list[dict], dict]:
    summary = json.loads(confirmation_summary.read_text())
    individual = {}
    for row in summary["scenarios"]:
        if len(row["targets"]) != 1:
            continue
        target = row["targets"][0]
        component = target["component"]
        individual.setdefault(component, {"target": target})[row["direction"]] = row
    selected = []
    ranking = {}
    for layer in (47, 51):
        rows = []
        for component, entry in individual.items():
            if entry["target"]["layer"] != layer:
                continue
            if not {"neutral_into_game", "game_into_neutral"}.issubset(entry):
                continue
            forward = entry["neutral_into_game"]["rank_slope"]["letter_macro"]["mean"]
            reverse = entry["game_into_neutral"]["rank_slope"]["letter_macro"]["mean"]
            rows.append({
                **entry["target"],
                "forward_rank": forward,
                "reverse_rank": reverse,
                "reciprocal_score": min(forward, reverse),
            })
        rows.sort(key=lambda row: row["reciprocal_score"], reverse=True)
        ranking[str(layer)] = rows
        selected.extend(rows[:heads_per_layer])
    return [
        {key: row[key] for key in ("component", "layer", "heads")} for row in selected
    ], ranking


def _source_scenarios(targets: list[dict], sources: tuple[str, ...], individual: bool) -> list[dict]:
    scenarios = []
    target_groups = [[target] for target in targets] if individual else []
    target_groups.append(targets)
    for group in target_groups:
        group_id = group[0]["component"] if len(group) == 1 else "selected_joint"
        for condition, label in (("incorrect", "game"), ("neutral", "neutral")):
            for source in sources:
                # Neutral has no condition-specific system prefix. The span is
                # empty there and is therefore a Game-only necessity test.
                if condition == "neutral" and source == "system_condition":
                    continue
                scenarios.append({
                    "id": f"ablate_{label}__{group_id}__{source}",
                    "target_condition": condition,
                    "source": source,
                    "targets": group,
                })
    return scenarios


def prepare_discovery(
    confirmation_summary: Path,
    discovery_plan: Path,
    output: Path,
    heads_per_layer: int = 2,
) -> dict:
    targets, ranking = _select_heads(confirmation_summary, heads_per_layer)
    split = json.loads(discovery_plan.read_text())
    plan = {
        "stage": "onset_attention_source_discovery",
        "question_ids": split["question_ids"],
        "targets": targets,
        "scenarios": _source_scenarios(targets, SOURCE_SELECTORS, individual=False),
        "source_selectors": list(SOURCE_SELECTORS),
        "selection_rule": (
            f"Top {heads_per_layer} reciprocal rank mediators per mixer are fixed; "
            "screen all source spans jointly on discovery questions."
        ),
        "confirmation_ranking": ranking,
        "confirmation_summary": str(confirmation_summary),
        "split_source": str(discovery_plan),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def prepare_confirmation(
    discovery_source_summary: Path,
    discovery_plan: Path,
    confirmation_plan: Path,
    output: Path,
    max_sources: int = 3,
) -> dict:
    discovery = json.loads(discovery_plan.read_text())
    summary = json.loads(discovery_source_summary.read_text())
    split = json.loads(confirmation_plan.read_text())
    candidates = []
    for row in summary["rows"]:
        if row["metric"] != "rank_slope":
            continue
        if row["contrast"] == "game_minus_neutral_removal":
            candidates.append(row)
        elif row["contrast"] == "game_removal" and row["source"] == "system_condition":
            candidates.append(row)
    candidates.sort(key=lambda row: row["mean"], reverse=True)
    sources = tuple(dict.fromkeys(row["source"] for row in candidates[:max_sources]))
    plan = {
        "stage": "onset_attention_source_confirmation",
        "question_ids": split["question_ids"],
        "targets": discovery["targets"],
        "scenarios": _source_scenarios(discovery["targets"], sources, individual=True),
        "selected_sources": list(sources),
        "source_ranking": candidates,
        "selection_rule": f"Top {max_sources} discovery source spans by rank-slope removal",
        "discovery_summary": str(discovery_source_summary),
        "split_source": str(confirmation_plan),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare semantic source-edge tests")
    sub = parser.add_subparsers(dest="command", required=True)
    discovery = sub.add_parser("discovery")
    discovery.add_argument("--confirmation-summary", required=True)
    discovery.add_argument("--discovery-plan", required=True)
    discovery.add_argument("--output", required=True)
    discovery.add_argument("--heads-per-layer", type=int, default=2)
    confirmation = sub.add_parser("confirmation")
    confirmation.add_argument("--discovery-source-summary", required=True)
    confirmation.add_argument("--discovery-plan", required=True)
    confirmation.add_argument("--confirmation-plan", required=True)
    confirmation.add_argument("--output", required=True)
    confirmation.add_argument("--max-sources", type=int, default=3)
    args = parser.parse_args()
    if args.command == "discovery":
        prepare_discovery(
            Path(args.confirmation_summary), Path(args.discovery_plan),
            Path(args.output), args.heads_per_layer,
        )
    else:
        prepare_confirmation(
            Path(args.discovery_source_summary), Path(args.discovery_plan),
            Path(args.confirmation_plan), Path(args.output), args.max_sources,
        )


if __name__ == "__main__":
    main()
