from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def prepare(
    source_plan_path: Path,
    manifest_path: Path,
    output: Path,
    max_questions: int | None = None,
) -> dict:
    source = json.loads(source_plan_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    question_ids = [question["id"] for question in manifest["questions"]]
    if max_questions is not None:
        question_ids = question_ids[:max_questions]
    if not question_ids:
        raise ValueError("Transfer manifest contains no questions")
    if not source.get("targets") or not source.get("scenarios"):
        raise ValueError("Source confirmation plan has no fixed targets or scenarios")

    plan = {
        "stage": "cross_dataset_component_transfer",
        "question_ids": question_ids,
        "collect_baseline": True,
        "targets": copy.deepcopy(source["targets"]),
        "scenarios": copy.deepcopy(source["scenarios"]),
        "selected_compression": copy.deepcopy(source.get("selected_compression", [])),
        "selected_switching": copy.deepcopy(source.get("selected_switching", [])),
        "source_selection_plan": str(source_plan_path),
        "source_selection_dataset": "SimpleMC",
        "transfer_dataset": manifest.get("dataset", "TriviaMC"),
        "selection_data_policy": (
            "All targets and grouped scenarios were frozen from the SimpleMC discovery and "
            "confirmation run. No TriviaMC outcome was used for target selection."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a frozen cross-dataset component-transfer plan")
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    prepare(Path(args.source_plan), Path(args.manifest), Path(args.output), args.max_questions)


if __name__ == "__main__":
    main()
