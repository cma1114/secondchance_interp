from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(
    split_plan_path: Path,
    natural_root: Path,
    output: Path,
    anchors: list[str],
    max_questions: int | None = None,
    max_targets: int | None = None,
    batch_size: int = 4,
) -> dict:
    split_plan = json.loads(split_plan_path.read_text())
    question_ids = list(split_plan.get("discovery_question_ids", []))
    if not question_ids:
        raise ValueError("Split plan has no discovery_question_ids")
    run_metadata = json.loads((natural_root / "run_metadata.json").read_text())
    n_layers_value = run_metadata.get("n_layers", run_metadata.get("n_text_layers"))
    if n_layers_value is None:
        raise KeyError("Natural run metadata has neither n_layers nor n_text_layers")
    n_layers = int(n_layers_value)
    if not anchors or len(set(anchors)) != len(anchors):
        raise ValueError("Anchors must be a non-empty list without duplicates")
    targets = [
        {
            "component": f"{anchor}__{kind}_l{layer}",
            "anchor": anchor,
            "kind": kind,
            "layer": layer,
        }
        for anchor in anchors
        for layer in range(n_layers)
        for kind in ("mixer", "mlp")
    ]
    if max_questions is not None:
        question_ids = question_ids[:max_questions]
    if max_targets is not None:
        targets = targets[:max_targets]
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
        "stage": "position_component_causal_discovery",
        "question_ids": question_ids,
        "anchors": anchors,
        "targets": targets,
        "scenarios": scenarios,
        "batch_size": int(batch_size),
        "selection_data_policy": (
            "Every anchored mixer/MLP target is causally swept Neutral-into-Game "
            "on the fixed discovery questions. Candidate selection uses only these outcomes."
        ),
        "split_source": str(split_plan_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an exhaustive semantically anchored component sweep"
    )
    parser.add_argument("--split-plan", required=True)
    parser.add_argument("--natural-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchors", nargs="+", required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    prepare(
        Path(args.split_plan),
        Path(args.natural_root),
        Path(args.output),
        args.anchors,
        args.max_questions,
        args.max_targets,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
