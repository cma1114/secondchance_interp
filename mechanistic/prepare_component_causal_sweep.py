from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(
    split_plan_path: Path,
    natural_root: Path,
    output: Path,
    max_questions: int | None = None,
    max_components: int | None = None,
) -> dict:
    split_plan = json.loads(split_plan_path.read_text())
    question_ids = split_plan.get("discovery_question_ids")
    if not question_ids:
        raise ValueError("Split plan has no discovery_question_ids")
    run_metadata = json.loads((natural_root / "run_metadata.json").read_text())
    n_layers = int(run_metadata["n_layers"])
    targets = [
        {"component": f"{kind}_l{layer}", "kind": kind, "layer": layer}
        for layer in range(n_layers)
        for kind in ("mixer", "mlp")
    ]
    if max_questions is not None:
        question_ids = question_ids[:max_questions]
    if max_components is not None:
        targets = targets[:max_components]
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
        "stage": "causal_discovery",
        "question_ids": question_ids,
        "targets": targets,
        "scenarios": scenarios,
        "selection_data_policy": (
            "All components are causally swept Neutral-into-Game on these questions. "
            "Candidate selection uses only these causal outcomes."
        ),
        "split_source": str(split_plan_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an exhaustive component causal-discovery sweep")
    parser.add_argument("--split-plan", required=True)
    parser.add_argument("--natural-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-components", type=int)
    args = parser.parse_args()
    prepare(
        Path(args.split_plan), Path(args.natural_root), Path(args.output),
        args.max_questions, args.max_components,
    )


if __name__ == "__main__":
    main()
