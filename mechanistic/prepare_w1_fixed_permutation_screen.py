from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from . import LETTERS


def prepare(manifest_path: Path, baseline_path: Path, output_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    baseline = json.loads(baseline_path.read_text())["results"]
    qids = [row["id"] for row in manifest["questions"]]
    rows = []
    for qid in qids:
        w1 = baseline[qid]["answer"]
        other_letters = [letter for letter in LETTERS if letter != w1]
        permutations = list(itertools.permutations(other_letters))
        identity = tuple(other_letters)
        permutations = [identity] + [value for value in permutations if value != identity]
        mappings = []
        for mapping_index, permutation in enumerate(permutations):
            new_to_original = {w1: w1}
            new_to_original.update(dict(zip(other_letters, permutation)))
            mappings.append({
                "mapping_index": mapping_index,
                "new_to_original": new_to_original,
                "original_to_new": {
                    original: new for new, original in new_to_original.items()
                },
                "identity": mapping_index == 0,
            })
        rows.append({
            "question_id": qid,
            "w1_original_content": w1,
            "w1_displayed_letter": w1,
            "mappings": mappings,
        })
    payload = {
        "status": "frozen",
        "definition": (
            "For each question, keep the original Baseline winner W1 at its original "
            "displayed letter and permute the other three semantic options through all "
            "six arrangements. Mapping 0 is identity."
        ),
        "n_questions": len(rows),
        "n_mappings_per_question": 6,
        "question_ids": qids,
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.baseline, args.output)


if __name__ == "__main__":
    main()
