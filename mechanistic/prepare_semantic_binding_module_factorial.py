from __future__ import annotations

import argparse
import json
from pathlib import Path


LETTERS = "ABCD"


def _rows(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    return {row["question_id"]: row for row in payload["rows"]}


def prepare(
    baseline_path: Path,
    remapped_baseline_path: Path,
    remapping_path: Path,
    second_mapping_path: Path,
    discovery_path: Path,
    confirmation_path: Path,
    output: Path,
) -> dict:
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped_baseline = json.loads(remapped_baseline_path.read_text())["results"]
    first_y = _rows(remapping_path)
    second = _rows(second_mapping_path)
    split_lookup: dict[str, str] = {}
    for split, path in (
        ("discovery", discovery_path),
        ("confirmation", confirmation_path),
    ):
        payload = json.loads(path.read_text())
        for qid in payload["question_ids"]:
            if qid in split_lookup:
                raise ValueError(f"Question {qid} occurs in both splits")
            split_lookup[qid] = split

    identity = {letter: letter for letter in LETTERS}
    rows = []
    for qid, base_row in baseline.items():
        if qid not in remapped_baseline or qid not in first_y or qid not in second:
            continue
        x_letter = base_row.get("answer", base_row.get("subject_answer"))
        y_row = remapped_baseline[qid]
        y_answer_letter = y_row.get("answer_new_letter")
        y_content = y_row.get("answer_original_content")
        if x_letter != "A" or y_answer_letter != "A" or y_content == x_letter:
            continue
        if split_lookup.get(qid) not in {"discovery", "confirmation"}:
            continue
        first_y_row = first_y[qid]
        if first_y_row["new_to_original"]["A"] != y_content:
            raise ValueError(f"Remapped Baseline content disagrees with plan for {qid}")
        second_row = second[qid]
        second_original_to_new = {
            original: new for new, original in second_row["new_to_original"].items()
        }
        rows.append(
            {
                "question_id": qid,
                "split": split_lookup[qid],
                "literal_first_answer": "A",
                "x_content_original_letter": x_letter,
                "y_content_original_letter": y_content,
                "first_x_new_to_original": identity,
                "first_y_new_to_original": first_y_row["new_to_original"],
                "second_new_to_original": second_row["new_to_original"],
                "second_original_to_new": second_original_to_new,
                "x_second_letter": second_original_to_new[x_letter],
                "y_second_letter": second_original_to_new[y_content],
            }
        )
    rows.sort(key=lambda row: (row["split"], row["question_id"]))
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("discovery", "confirmation")
    }
    if counts != {"discovery": 64, "confirmation": 73}:
        raise RuntimeError(f"Unexpected eligible cohort counts: {counts}")
    payload = {
        "definition": (
            "Two first-presentation mappings both yield literal A but different "
            "semantic winners; the second presentation uses one fixed third mapping."
        ),
        "counts": counts,
        "n_total": len(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--remapping", type=Path, required=True)
    parser.add_argument("--second-mapping", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(
        args.baseline,
        args.remapped_baseline,
        args.remapping,
        args.second_mapping,
        args.discovery,
        args.confirmation,
        args.output,
    )
    print(json.dumps(payload["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
