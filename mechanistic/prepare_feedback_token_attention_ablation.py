from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _group_targets(rows: list[dict]) -> list[dict]:
    by_layer: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["block"]) - 1].append(int(row["head"]) - 1)
    return [
        {"layer": layer, "heads": sorted(set(heads))}
        for layer, heads in sorted(by_layer.items())
    ]


def prepare(summary_path: Path, confirmation_plan: Path, output: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    confirmation = json.loads(confirmation_plan.read_text())
    candidates = summary["candidates"]
    scenarios = []
    for row in candidates:
        source = row["source"]
        block = int(row["block"])
        head = int(row["head"])
        scenarios.append({
            "id": f"{source}_B{block}_H{head}",
            "label": f"B{block}/H{head} {source}",
            "source": source,
            "source_token_index_within_feedback_zero_based": 3 if source == "evaluation" else 8,
            "targets": [{"layer": block - 1, "heads": [head - 1]}],
            "selection": row["nominated_by"],
        })

    for source in ("evaluation", "action"):
        rows = [
            row for row in candidates
            if row["source"] == source
            and (
                "game" in row["nominated_by"]
                or (
                    "contrast" in row["nominated_by"]
                    and row["discovery_game_minus_neutral"] > 0
                )
            )
        ]
        scenarios.append({
            "id": f"{source}_game_reader_joint",
            "label": f"{source} Game-reader joint",
            "source": source,
            "source_token_index_within_feedback_zero_based": 3 if source == "evaluation" else 8,
            "targets": _group_targets(rows),
            "selection": [
                f"B{row['block']}/H{row['head']}" for row in rows
            ],
        })

    plan = {
        "stage": "held_out_causal_confirmation",
        "split_source": str(confirmation_plan),
        "question_ids": confirmation["question_ids"],
        "candidate_source": str(summary_path),
        "conditions": ["incorrect", "neutral"],
        "intervention": (
            "mask one exact final-query-to-feedback-token attention logit before "
            "softmax and renormalize the remaining attention"
        ),
        "indexing": {
            "displayed_blocks_and_heads": "one-based",
            "stored_target_layers_and_heads": "zero-based",
        },
        "scenarios": scenarios,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--confirmation-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = prepare(args.summary, args.confirmation_plan, args.output)
    print(json.dumps({
        "n_questions": len(plan["question_ids"]),
        "n_scenarios": len(plan["scenarios"]),
        "scenario_ids": [row["id"] for row in plan["scenarios"]],
    }, indent=2))


if __name__ == "__main__":
    main()
