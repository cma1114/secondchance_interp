from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(confirmation_plan_path: Path, output: Path) -> dict:
    confirmation = json.loads(confirmation_plan_path.read_text())
    target_lookup = {
        target["component"]: target for target in confirmation["targets"]
    }

    # True causal order: the feedback-period state is computed before the final
    # decision position; within a position, blocks are ordered shallow to deep.
    ordered_names = [
        "feedback_end__mlp_l31",
        "feedback_end__mlp_l43",
        "decision__mixer_l49",
        "decision__mixer_l51",
        "decision__mixer_l55",
        "decision__mixer_l59",
        "decision__mixer_l60",
        "decision__mixer_l62",
    ]
    missing = [name for name in ordered_names if name not in target_lookup]
    if missing:
        raise KeyError(f"Selected targets missing from confirmation plan: {missing}")
    ordered_targets = [target_lookup[name] for name in ordered_names]

    scenarios: list[dict] = []
    for direction, source, target in (
        ("neutral_into_game", "neutral", "incorrect"),
        ("game_into_neutral", "incorrect", "neutral"),
    ):
        for count in range(1, len(ordered_targets) + 1):
            scenarios.append(
                {
                    "id": f"{direction}__cumulative_{count:02d}",
                    "source_condition": source,
                    "target_condition": target,
                    "targets": ordered_targets[:count],
                    "diagnostic": "cumulative",
                    "count": count,
                    "added_component": ordered_names[count - 1],
                }
            )
        for omitted_index, omitted_name in enumerate(ordered_names):
            scenarios.append(
                {
                    "id": f"{direction}__leave_out_{omitted_index + 1:02d}",
                    "source_condition": source,
                    "target_condition": target,
                    "targets": [
                        target_value
                        for index, target_value in enumerate(ordered_targets)
                        if index != omitted_index
                    ],
                    "diagnostic": "leave_one_out",
                    "omitted_component": omitted_name,
                }
            )

    plan = {
        "stage": "heldout_position_component_interactions",
        "question_ids": confirmation["question_ids"],
        "targets": ordered_targets,
        "scenarios": scenarios,
        "ordered_components": ordered_names,
        "source_confirmation_plan": str(confirmation_plan_path),
        "interpretation": {
            "cumulative": (
                "Add paired source-condition component outputs in causal order. "
                "The eighth step is the full eight-component intervention."
            ),
            "leave_one_out": (
                "Patch seven of eight outputs. Compare against the full cumulative "
                "step to measure each component's conditional contribution."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare cumulative and leave-one-out position-component patches"
    )
    parser.add_argument("--confirmation-plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = prepare(Path(args.confirmation_plan), Path(args.output))
    print(
        f"Wrote {len(plan['scenarios'])} scenarios over "
        f"{len(plan['question_ids'])} held-out questions to {args.output}"
    )


if __name__ == "__main__":
    main()
