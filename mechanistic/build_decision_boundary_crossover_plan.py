from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


LETTERS = "ABCD"


def build(
    eligible_path: Path,
    screen_plan_path: Path,
    screen_results_path: Path,
    second_mapping_path: Path,
    output_path: Path,
) -> None:
    eligible = json.loads(eligible_path.read_text())["rows"]
    screen_plan = json.loads(screen_plan_path.read_text())
    plan_by_qid = {row["question_id"]: row for row in screen_plan["rows"]}
    second_by_qid = {
        row["question_id"]: row
        for row in json.loads(second_mapping_path.read_text())["rows"]
    }
    with np.load(screen_results_path, allow_pickle=False) as loaded:
        qids = loaded["question_ids"].astype(str).tolist()
        logits = loaded["aggregated_ad_logits"].astype(float)
        completed = loaded["completed"].astype(bool)
    qi_by_qid = {qid: index for index, qid in enumerate(qids)}

    rows = []
    for source in eligible:
        qid = source["question_id"]
        qi = qi_by_qid[qid]
        chosen_index = int(source["chosen_mapping_index"])
        unchosen_index = int(source["unchosen_mapping_index"])
        if not completed[chosen_index, qi] or not completed[unchosen_index, qi]:
            raise ValueError(f"{qid}: incomplete screen result")
        mappings = {
            int(row["mapping_index"]): row for row in plan_by_qid[qid]["mappings"]
        }
        x_mapping = mappings[chosen_index]
        y_mapping = mappings[unchosen_index]
        x_letter = LETTERS[int(np.argmax(logits[chosen_index, qi]))]
        y_letter = LETTERS[int(np.argmax(logits[unchosen_index, qi]))]
        x_content = x_mapping["new_to_original"][x_letter]
        y_content = y_mapping["new_to_original"][y_letter]
        if x_content != source["w1_original_content"]:
            raise ValueError(f"{qid}: chosen mapping does not choose frozen W1")
        if y_content == x_content:
            raise ValueError(f"{qid}: unchosen mapping retained the same semantic winner")
        second = second_by_qid[qid]
        rows.append(
            {
                "question_id": qid,
                "split": source["split"],
                "x_mapping_index": chosen_index,
                "y_mapping_index": unchosen_index,
                "x_first_new_to_original": x_mapping["new_to_original"],
                "y_first_new_to_original": y_mapping["new_to_original"],
                "x_screen_winner_first_letter": x_letter,
                "y_screen_winner_first_letter": y_letter,
                "x_screen_winner_original_content": x_content,
                "y_screen_winner_original_content": y_content,
                "second_new_to_original": second["new_to_original"],
                "second_original_to_new": second["original_to_new"],
                "x_screen_winner_second_letter": second["original_to_new"][x_content],
                "y_screen_winner_second_letter": second["original_to_new"][y_content],
                "x_winner_fixed_displayed_letter": source["w1_displayed_letter"],
            }
        )

    counts = {
        "split": dict(Counter(row["split"] for row in rows)),
        "x_first_letter": dict(
            Counter(row["x_screen_winner_first_letter"] for row in rows)
        ),
        "y_first_letter": dict(
            Counter(row["y_screen_winner_first_letter"] for row in rows)
        ),
    }
    payload = {
        "status": "frozen_before_causal_run",
        "definition": (
            "For each question, history X is the frozen six-permutation presentation "
            "in which the fixed semantic option wins; history Y keeps that option at "
            "the same displayed letter and semantic position but permutes distractors "
            "so a different semantic option wins. The canonical second presentation "
            "is identical for X and Y."
        ),
        "n_rows": len(rows),
        "counts": counts,
        "source_artifacts": {
            "eligible_pairs": str(eligible_path),
            "screen_plan": str(screen_plan_path),
            "screen_results": str(screen_results_path),
            "second_mapping": str(second_mapping_path),
        },
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_rows": len(rows), "counts": counts}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--screen-plan", type=Path, required=True)
    parser.add_argument("--screen-results", type=Path, required=True)
    parser.add_argument("--second-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(
        args.eligible,
        args.screen_plan,
        args.screen_results,
        args.second_mapping,
        args.output,
    )


if __name__ == "__main__":
    main()
