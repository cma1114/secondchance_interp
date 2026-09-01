from __future__ import annotations

import argparse
import json
from pathlib import Path


CANDIDATES = (
    "mixer_l45",
    "mixer_l47",
    "mlp_l49",
    "mlp_l54",
    "mixer_l55",
    "mlp_l61",
    "mlp_l62",
)

GROUPS = {
    "onset_candidates": ("mixer_l45", "mixer_l47", "mlp_l49"),
    "late_candidates": ("mlp_l54", "mixer_l55", "mlp_l61", "mlp_l62"),
    "all_rank_candidates": CANDIDATES,
}


def prepare(discovery_plan: Path, heldout_plan: Path, output: Path) -> dict:
    discovery = json.loads(discovery_plan.read_text())
    heldout = json.loads(heldout_plan.read_text())
    lookup = {
        target["component"]: target
        for target in discovery["targets"]
    }
    missing = sorted(set(CANDIDATES) - set(lookup))
    if missing:
        raise ValueError(f"Candidates absent from exhaustive plan: {missing}")

    targets = [lookup[name] for name in CANDIDATES]
    scenarios = []
    for name in CANDIDATES:
        target = lookup[name]
        scenarios.extend([
            {
                "id": f"neutral_into_game__{name}",
                "source_condition": "neutral",
                "target_condition": "incorrect",
                "targets": [target],
            },
            {
                "id": f"game_into_neutral__{name}",
                "source_condition": "incorrect",
                "target_condition": "neutral",
                "targets": [target],
            },
        ])
    for label, names in GROUPS.items():
        group = [lookup[name] for name in names]
        scenarios.extend([
            {
                "id": f"neutral_into_game__{label}",
                "source_condition": "neutral",
                "target_condition": "incorrect",
                "targets": group,
            },
            {
                "id": f"game_into_neutral__{label}",
                "source_condition": "incorrect",
                "target_condition": "neutral",
                "targets": group,
            },
        ])

    plan = {
        "stage": "heldout_rank_opposition_confirmation",
        "question_ids": heldout["question_ids"],
        "targets": targets,
        "scenarios": scenarios,
        "selection_data_policy": (
            "Candidates were selected from the untouched exhaustive discovery half using the "
            "rank-opposed final-logit endpoint. This plan uses the previously untouched confirmation half "
            "and tests both patch directions."
        ),
        "discovery_plan": str(discovery_plan),
        "heldout_split_source": str(heldout_plan),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare held-out rank-opposition component confirmation")
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--heldout-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.discovery_plan, args.heldout_plan, args.output)


if __name__ == "__main__":
    main()
