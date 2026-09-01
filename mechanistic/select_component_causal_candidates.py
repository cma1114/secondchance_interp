from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str) -> float:
    return float(row[key])


def select(
    effects_path: Path,
    discovery_plan_path: Path,
    split_plan_path: Path,
    output: Path,
    max_candidates: int,
) -> dict:
    discovery_plan = json.loads(discovery_plan_path.read_text())
    split_plan = json.loads(split_plan_path.read_text())
    rows = [
        row for row in _rows(effects_path)
        if row["aggregation"] == "dataset" and row["direction"] == "neutral_into_game"
    ]
    lookup = {(row["component"], row["metric"]): row for row in rows}
    components = sorted({row["component"] for row in rows if row["component"]})

    compression_rank = []
    switching_rank = []
    rank_redistribution_rank = []
    for component in components:
        entropy = lookup[(component, "ad_entropy")]
        spread = lookup[(component, "ad_spread")]
        switch = lookup[(component, "switch")]
        winner = lookup[(component, "winner_advantage")]
        rank_opposition = lookup.get((component, "rank_opposed_slope"))
        entropy_fraction = _float(entropy, "fraction_gap_mediated")
        spread_fraction = _float(spread, "fraction_gap_mediated")
        switch_fraction = _float(switch, "fraction_gap_mediated")
        winner_fraction = _float(winner, "fraction_gap_mediated")
        compression_rank.append({
            "component": component,
            "score": 0.5 * (entropy_fraction + spread_fraction),
            "entropy_fraction": entropy_fraction,
            "spread_fraction": spread_fraction,
            "eligible": _float(entropy, "effect_mean") < 0 and _float(spread, "effect_mean") > 0,
        })
        switching_rank.append({
            "component": component,
            "score": 0.5 * (switch_fraction + winner_fraction),
            "switch_fraction": switch_fraction,
            "winner_fraction": winner_fraction,
            "eligible": _float(switch, "effect_mean") < 0 and _float(winner, "effect_mean") > 0,
        })
        if rank_opposition is not None:
            rank_fraction = _float(rank_opposition, "fraction_gap_mediated")
            rank_redistribution_rank.append({
                "component": component,
                "score": rank_fraction,
                "rank_opposition_fraction": rank_fraction,
                "eligible": _float(rank_opposition, "effect_mean") < 0,
            })
    compression_rank.sort(key=lambda row: row["score"], reverse=True)
    switching_rank.sort(key=lambda row: row["score"], reverse=True)
    rank_redistribution_rank.sort(key=lambda row: row["score"], reverse=True)
    compression_lookup = {row["component"]: row for row in compression_rank}
    switching_lookup = {row["component"]: row for row in switching_rank}
    rank_redistribution_lookup = {
        row["component"]: row for row in rank_redistribution_rank
    }
    overall_rank = []
    for component in components:
        compression_row = compression_lookup[component]
        switching_row = switching_lookup[component]
        rank_row = rank_redistribution_lookup.get(component)
        eligible_scores = {}
        if compression_row["eligible"]:
            eligible_scores["literal_flattening"] = compression_row["score"]
        if switching_row["eligible"]:
            eligible_scores["switching"] = switching_row["score"]
        if rank_row is not None and rank_row["eligible"]:
            eligible_scores["rank_redistribution"] = rank_row["score"]
        if not eligible_scores:
            continue
        overall_rank.append({
            "component": component,
            "selection_score": max(eligible_scores.values()),
            "primary_family": max(eligible_scores, key=eligible_scores.get),
            "eligible_family_scores": eligible_scores,
        })
    overall_rank.sort(key=lambda row: row["selection_score"], reverse=True)
    selected_names = [row["component"] for row in overall_rank[:max_candidates]]
    compression = [name for name in selected_names if compression_lookup[name]["eligible"]]
    switching = [name for name in selected_names if switching_lookup[name]["eligible"]]
    rank_redistribution = [
        name for name in selected_names
        if name in rank_redistribution_lookup
        and rank_redistribution_lookup[name]["eligible"]
    ]
    target_lookup = {target["component"]: target for target in discovery_plan["targets"]}
    targets = [target_lookup[name] for name in selected_names]
    scenarios = []
    for target in targets:
        scenarios.extend([
            {"id": f"neutral_into_game__{target['component']}", "source_condition": "neutral", "target_condition": "incorrect", "targets": [target]},
            {"id": f"game_into_neutral__{target['component']}", "source_condition": "incorrect", "target_condition": "neutral", "targets": [target]},
        ])
    for label, names in (
        ("compression_candidates", compression),
        ("switching_candidates", switching),
        ("rank_redistribution_candidates", rank_redistribution),
        ("all_candidates", selected_names),
    ):
        group = [target_lookup[name] for name in names]
        if not group:
            continue
        scenarios.extend([
            {"id": f"neutral_into_game__{label}", "source_condition": "neutral", "target_condition": "incorrect", "targets": group},
            {"id": f"game_into_neutral__{label}", "source_condition": "incorrect", "target_condition": "neutral", "targets": group},
        ])
    plan = {
        "stage": "heldout_causal_confirmation",
        "question_ids": split_plan["confirmation_question_ids"],
        "targets": targets,
        "scenarios": scenarios,
        "selection_rule": {
            "aggregation": "dataset-weighted",
            "compression_score": "Mean fraction of A-D entropy and spread gaps removed; both signs must indicate less flattening.",
            "switching_score": "Mean fraction of switch and winner-advantage gaps removed; both signs must indicate less switching/restored winner advantage.",
            "rank_redistribution_score": "Fraction of the final Baseline-rank opposition gap removed; the patch must reduce rank opposition.",
            "overall": f"For every component, take the larger eligible family score and select the top {max_candidates} overall. This imposes no quota by mechanism family.",
        },
        "selected_compression": compression,
        "selected_switching": switching,
        "selected_rank_redistribution": rank_redistribution,
        "compression_ranking": compression_rank,
        "switching_ranking": switching_rank,
        "rank_redistribution_ranking": rank_redistribution_rank,
        "overall_ranking": overall_rank,
        "discovery_effects": str(effects_path),
        "split_source": str(split_plan_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Select causal compression and switching candidates for held-out confirmation")
    parser.add_argument("--effects", required=True)
    parser.add_argument("--discovery-plan", required=True)
    parser.add_argument("--split-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args()
    select(Path(args.effects), Path(args.discovery_plan), Path(args.split_plan), Path(args.output), args.max_candidates)


if __name__ == "__main__":
    main()
