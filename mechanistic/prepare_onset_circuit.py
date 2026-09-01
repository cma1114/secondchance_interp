from __future__ import annotations

import argparse
import json
from pathlib import Path


COMPONENTS = (
    {"component": "mixer_l47", "kind": "mixer", "layer": 47},
    {"component": "mlp_l49", "kind": "mlp", "layer": 49},
    {"component": "mixer_l50", "kind": "mixer", "layer": 50},
    {"component": "mixer_l51", "kind": "mixer", "layer": 51},
)
ATTENTION_LAYERS = (47, 51)
N_HEADS = 24


def _reciprocal(component_id: str, targets: list[dict]) -> list[dict]:
    return [
        {
            "id": f"neutral_into_game__{component_id}",
            "source_condition": "neutral",
            "target_condition": "incorrect",
            "targets": targets,
        },
        {
            "id": f"game_into_neutral__{component_id}",
            "source_condition": "incorrect",
            "target_condition": "neutral",
            "targets": targets,
        },
    ]


def prepare_component_plan(confirmation_plan: Path, output: Path) -> dict:
    source = json.loads(confirmation_plan.read_text())
    qids = source["question_ids"]
    groups = {
        "onset_all4": list(COMPONENTS),
        "onset_early_pair": list(COMPONENTS[:2]),
        "onset_late_pair": list(COMPONENTS[2:]),
    }
    for omitted in COMPONENTS:
        groups[f"onset_without_{omitted['component']}"] = [
            target for target in COMPONENTS if target != omitted
        ]
    scenarios = []
    for group_id, targets in groups.items():
        scenarios.extend(_reciprocal(group_id, list(targets)))
    plan = {
        "stage": "onset_circuit_joint_confirmation",
        "question_ids": qids,
        "targets": list(COMPONENTS),
        "groups": groups,
        "scenarios": scenarios,
        "selection_data_policy": (
            "The four components and every group were frozen from prior held-out reciprocal "
            "rank-opposition results before this joint experiment."
        ),
        "split_source": str(confirmation_plan),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def prepare_head_discovery_plan(discovery_plan: Path, output: Path) -> dict:
    source = json.loads(discovery_plan.read_text())
    qids = source["question_ids"]
    targets = [
        {"component": f"mixer_l{layer}_h{head}", "layer": layer, "heads": [head]}
        for layer in ATTENTION_LAYERS
        for head in range(N_HEADS)
    ]
    scenarios = [
        {
            "id": f"neutral_into_game__{target['component']}",
            "source_condition": "neutral",
            "target_condition": "incorrect",
            "targets": [target],
        }
        for target in targets
    ]
    plan = {
        "stage": "onset_head_discovery",
        "question_ids": qids,
        "targets": targets,
        "scenarios": scenarios,
        "selection_rule": (
            "Within each mixer, take the union of the three largest ordered-rank "
            "mediators and three largest switching mediators."
        ),
        "split_source": str(discovery_plan),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare frozen onset-circuit plans")
    parser.add_argument("--confirmation-plan", required=True)
    parser.add_argument("--discovery-plan", required=True)
    parser.add_argument("--component-output", required=True)
    parser.add_argument("--head-output", required=True)
    args = parser.parse_args()
    prepare_component_plan(Path(args.confirmation_plan), Path(args.component_output))
    prepare_head_discovery_plan(Path(args.discovery_plan), Path(args.head_output))


if __name__ == "__main__":
    main()
