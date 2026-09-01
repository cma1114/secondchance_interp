from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LETTERS = "ABCD"


def prepare(
    eligible_path: Path,
    permutation_plan_path: Path,
    permutation_results_path: Path,
    second_mapping_path: Path,
    output_path: Path,
) -> dict:
    eligible = json.loads(eligible_path.read_text())["rows"]
    permutation_plan = json.loads(permutation_plan_path.read_text())
    plan_rows = {row["question_id"]: row for row in permutation_plan["rows"]}
    second_rows = {
        row["question_id"]: row
        for row in json.loads(second_mapping_path.read_text())["rows"]
    }
    run = np.load(permutation_results_path, allow_pickle=False)
    qids = [str(value) for value in run["question_ids"]]
    qindex = {qid: index for index, qid in enumerate(qids)}
    logits = np.asarray(run["aggregated_ad_logits"], dtype=float)

    rows = []
    for source in eligible:
        if source["w1_displayed_letter"] != "A":
            continue
        qid = source["question_id"]
        if qid not in plan_rows or qid not in second_rows or qid not in qindex:
            raise ValueError(f"Missing plan entry for {qid}")
        chosen_index = int(source["chosen_mapping_index"])
        recipient_index = int(source["unchosen_mapping_index"])
        if chosen_index != 0:
            raise ValueError(f"{qid}: chosen donor is not identity mapping")
        mappings = plan_rows[qid]["mappings"]
        donor_mapping = mappings[chosen_index]["new_to_original"]
        recipient_mapping = mappings[recipient_index]["new_to_original"]
        if donor_mapping != {letter: letter for letter in LETTERS}:
            raise ValueError(f"{qid}: donor mapping is not identity")
        if donor_mapping["A"] != "A" or recipient_mapping["A"] != "A":
            raise ValueError(f"{qid}: semantic A is not fixed at displayed A")

        qi = qindex[qid]
        donor_answer = LETTERS[int(np.argmax(logits[chosen_index, qi]))]
        recipient_answer = LETTERS[int(np.argmax(logits[recipient_index, qi]))]
        if donor_answer != "A" or recipient_answer == "A":
            raise ValueError(
                f"{qid}: frozen screen answers are {donor_answer}/{recipient_answer}"
            )
        recipient_content = recipient_mapping[recipient_answer]
        second = second_rows[qid]
        original_to_new = second["original_to_new"]
        rows.append(
            {
                "question_id": qid,
                "split": source["split"],
                "donor_first_new_to_original": donor_mapping,
                "recipient_first_new_to_original": recipient_mapping,
                "recipient_mapping_index": recipient_index,
                "screen_donor_answer_letter": donor_answer,
                "screen_recipient_answer_letter": recipient_answer,
                "screen_recipient_answer_original_content": recipient_content,
                "second_new_to_original": second["new_to_original"],
                "second_original_to_new": original_to_new,
                "target_original_content": "A",
                "target_second_letter": original_to_new["A"],
                "screen_recipient_winner_second_letter": original_to_new[
                    recipient_content
                ],
                "screen_target_centered_logit_donor": float(
                    source["w1_centered_logit_chosen"]
                ),
                "screen_target_centered_logit_recipient": float(
                    source["w1_centered_logit_unchosen"]
                ),
            }
        )

    rows.sort(key=lambda row: (row["split"], row["question_id"]))
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("discovery", "confirmation")
    }
    if counts != {"discovery": 41, "confirmation": 36}:
        raise RuntimeError(f"Unexpected frozen counts: {counts}")
    payload = {
        "definition": (
            "Semantic A remains at displayed A. The identity first presentation "
            "selects A; a frozen permutation of only B-D does not. The second "
            "presentation is the canonical per-question derangement and is identical "
            "for donor and recipient histories."
        ),
        "counts": counts,
        "n_total": len(rows),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--permutation-plan", type=Path, required=True)
    parser.add_argument("--permutation-results", type=Path, required=True)
    parser.add_argument("--second-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(
        args.eligible,
        args.permutation_plan,
        args.permutation_results,
        args.second_mapping,
        args.output,
    )
    print(json.dumps(payload["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
