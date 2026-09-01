from __future__ import annotations

import argparse
import json
from pathlib import Path


SELECTED_IDS = [
    # Replicate the two source-level effects.
    "evaluation_period__all_gla",
    "action_clause__all_gla",
    # Predefined leave-one-token family; primary interaction hypothesis is 06 (a/the).
    "action_clause__all_tokens_except_05",
    "action_clause__all_tokens_except_06",
    "action_clause__all_tokens_except_07",
    "action_clause__all_tokens_except_08",
    "action_clause__all_tokens_except_09",
    # Necessary-window hypotheses and discovery controls.
    "evaluation_period__all_except_blocks_01_08",
    "evaluation_period__all_except_blocks_25_32",
    "evaluation_period__all_except_blocks_33_40",
    "action_clause__all_except_blocks_01_08",
    "action_clause__all_except_blocks_17_24",
    "action_clause__all_except_blocks_25_32",
    "action_clause__all_except_blocks_49_56",
    # Cumulative onset for action-clause routing.
    "action_clause__prefix_through_block_08",
    "action_clause__prefix_through_block_16",
    "action_clause__prefix_through_block_24",
    "action_clause__prefix_through_block_32",
    "action_clause__prefix_through_block_40",
    "action_clause__suffix_from_block_09",
    "action_clause__suffix_from_block_17",
    "action_clause__suffix_from_block_25",
    "action_clause__suffix_from_block_33",
    "action_clause__suffix_from_block_41",
    # Cumulative onset for evaluation-boundary routing.
    "evaluation_period__prefix_through_block_24",
    "evaluation_period__prefix_through_block_32",
    "evaluation_period__prefix_through_block_40",
    "evaluation_period__prefix_through_block_48",
    "evaluation_period__suffix_from_block_17",
    "evaluation_period__suffix_from_block_25",
    "evaluation_period__suffix_from_block_33",
    # Largest/opposing exact-layer discovery effects; not selected as a one-layer claim.
    "action_clause__block_09",
    "action_clause__block_14",
    "action_clause__block_18",
    "action_clause__block_31",
    "evaluation_period__block_25",
    "evaluation_period__block_31",
]


def build(discovery_plan: Path, confirmation_source: Path, output: Path) -> None:
    discovery = json.loads(discovery_plan.read_text())
    confirmation = json.loads(confirmation_source.read_text())
    by_id = {row["id"]: row for row in discovery["scenarios"]}
    missing = [scenario_id for scenario_id in SELECTED_IDS if scenario_id not in by_id]
    if missing:
        raise ValueError(f"Selected scenarios missing from discovery plan: {missing}")
    scenarios = [by_id[scenario_id] for scenario_id in SELECTED_IDS]
    payload = {
        "status": "frozen_source_layer_confirmation",
        "question_ids": confirmation["question_ids"],
        "scenarios": scenarios,
        "selection_source": str(discovery_plan),
        "selection_notes": {
            "primary": [
                "action clause leave a/the unablated",
                "action prefix transition through blocks 24 and 32",
                "evaluation period all-except blocks 25-32",
            ],
            "controls": (
                "Complete token family, adjacent cumulative steps, counter-direction "
                "windows, and largest/opposing exact-layer discovery effects."
            ),
            "full_feedback_interventions": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze source-layer confirmation plan")
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--confirmation-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.discovery_plan, args.confirmation_source, args.output)


if __name__ == "__main__":
    main()
