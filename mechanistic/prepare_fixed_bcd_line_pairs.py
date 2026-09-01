from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


LETTERS = "ABCD"


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    return {row["question_id"]: row for row in json.loads(path.read_text())["rows"]}


def _centered(logits: np.ndarray, target: int) -> float:
    return float(logits[target] - (logits.sum() - logits[target]) / 3.0)


def prepare(
    screen_path: Path,
    permutation_plan_path: Path,
    second_mapping_path: Path,
    discovery_path: Path,
    output_path: Path,
    cap_per_letter_split: int,
    seed: int,
    letters: str = "BCD",
) -> dict[str, Any]:
    letters = "".join(dict.fromkeys(letters.upper()))
    if not letters or any(letter not in LETTERS for letter in letters):
        raise ValueError("letters must be a nonempty subset of ABCD")
    screen = np.load(screen_path, allow_pickle=False)
    if not np.all(screen["completed"].astype(bool)):
        raise RuntimeError("Permutation screen is incomplete")
    qids = screen["question_ids"].astype(str).tolist()
    logits = screen["aggregated_ad_logits"].astype(float)
    starts = screen["option_line_starts"].astype(int)
    ends = screen["option_line_ends"].astype(int)
    lengths = screen["prompt_lengths"].astype(int)
    plan = json.loads(permutation_plan_path.read_text())
    if qids != plan["question_ids"]:
        raise RuntimeError("Screen and permutation-plan question orders differ")
    global_mappings = plan.get("mappings")
    row_mappings = {
        row["question_id"]: list(row["mappings"])
        for row in plan.get("rows", [])
    }
    second = _rows(second_mapping_path)
    discovery = set(json.loads(discovery_path.read_text())["question_ids"])

    candidates: list[dict[str, Any]] = []
    for qi, qid in enumerate(qids):
        mappings = global_mappings if global_mappings is not None else row_mappings[qid]
        split = "discovery" if qid in discovery else "confirmation"
        for literal in letters:
            li = LETTERS.index(literal)
            selected = [mi for mi in range(len(mappings)) if int(logits[mi, qi].argmax()) == li]
            possible = []
            for x, y in itertools.combinations(selected, 2):
                x_content = mappings[x]["new_to_original"][literal]
                y_content = mappings[y]["new_to_original"][literal]
                if x_content == y_content:
                    continue
                aligned = (
                    starts[x, qi, li] == starts[y, qi, li]
                    and ends[x, qi, li] == ends[y, qi, li]
                    and lengths[x, qi] == lengths[y, qi]
                )
                if not aligned:
                    continue
                x_margin = _centered(logits[x, qi], li)
                y_margin = _centered(logits[y, qi], li)
                possible.append((min(x_margin, y_margin), x, y, x_margin, y_margin))
            if not possible:
                continue
            _, x, y, x_margin, y_margin = max(
                possible, key=lambda row: (row[0], -row[1], -row[2])
            )
            x_map = mappings[x]["new_to_original"]
            y_map = mappings[y]["new_to_original"]
            second_map = second[qid]["new_to_original"]
            second_inverse = {original: new for new, original in second_map.items()}
            x_content = x_map[literal]
            y_content = y_map[literal]
            stable = hashlib.sha256(
                f"{seed}|{split}|{literal}|{qid}|{x}|{y}".encode()
            ).hexdigest()
            candidates.append({
                "question_id": qid,
                "split": split,
                "literal_first_answer": literal,
                "x_mapping_index": x,
                "y_mapping_index": y,
                "x_content_original_letter": x_content,
                "y_content_original_letter": y_content,
                "first_x_new_to_original": x_map,
                "first_y_new_to_original": y_map,
                "second_new_to_original": second_map,
                "second_original_to_new": second_inverse,
                "x_second_letter": second_inverse[x_content],
                "y_second_letter": second_inverse[y_content],
                "selected_line_start": int(starts[x, qi, li]),
                "selected_line_end": int(ends[x, qi, li]),
                "screen_x_centered_margin": x_margin,
                "screen_y_centered_margin": y_margin,
                "sampling_hash": stable,
            })

    selected_rows: list[dict[str, Any]] = []
    available = Counter((row["split"], row["literal_first_answer"]) for row in candidates)
    for split in ("discovery", "confirmation"):
        for literal in letters:
            cell = [
                row for row in candidates
                if row["split"] == split and row["literal_first_answer"] == literal
            ]
            cell.sort(key=lambda row: row["sampling_hash"])
            selected_rows.extend(cell[:cap_per_letter_split])
    selected_rows.sort(
        key=lambda row: (row["split"], row["literal_first_answer"], row["question_id"])
    )
    counts = Counter((row["split"], row["literal_first_answer"]) for row in selected_rows)
    payload = {
        "status": "frozen_after_first-decision-only_screen",
        "definition": (
            "Same question and literal selected letter, different semantic content, "
            "with exactly aligned complete selected-line token spans."
        ),
        "cap_per_letter_split": cap_per_letter_split,
        "sampling_seed": seed,
        "available_counts": {f"{split}_{letter}": available[(split, letter)]
                             for split in ("discovery", "confirmation") for letter in letters},
        "selected_counts": {f"{split}_{letter}": counts[(split, letter)]
                            for split in ("discovery", "confirmation") for letter in letters},
        "letters": list(letters),
        "rows": selected_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--permutation-plan", type=Path, required=True)
    parser.add_argument("--second-mapping", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap-per-letter-split", type=int, default=70)
    parser.add_argument("--seed", type=int, default=8202026)
    parser.add_argument("--letters", default="BCD")
    args = parser.parse_args()
    payload = prepare(
        args.screen,
        args.permutation_plan,
        args.second_mapping,
        args.discovery,
        args.output,
        args.cap_per_letter_split,
        args.seed,
        args.letters,
    )
    print(json.dumps({
        "available_counts": payload["available_counts"],
        "selected_counts": payload["selected_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
