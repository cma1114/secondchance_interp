from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


LETTERS = "ABCD"


def prepare(manifest_path: Path, baseline_path: Path, output: Path, seed: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    baseline = json.loads(baseline_path.read_text())["results"]
    qids = [question["id"] for question in manifest["questions"]]
    if set(qids) != set(baseline):
        raise ValueError("Manifest and current Baseline question IDs differ")
    derangements = [
        permutation
        for permutation in itertools.permutations(LETTERS)
        if all(permutation[index] != LETTERS[index] for index in range(4))
    ]
    if len(derangements) != 9:
        raise RuntimeError(f"Expected 9 derangements, found {len(derangements)}")

    rng = np.random.default_rng(seed)
    assignments: dict[str, tuple[str, ...]] = {}
    for baseline_letter in LETTERS:
        group = [
            qid for qid in qids
            if baseline[qid].get("answer", baseline[qid].get("subject_answer"))
            == baseline_letter
        ]
        rng.shuffle(group)
        offset = int(rng.integers(0, len(derangements)))
        for index, qid in enumerate(group):
            assignments[qid] = derangements[(index + offset) % len(derangements)]

    rows = []
    for qid in qids:
        permutation = assignments[qid]
        new_to_original = dict(zip(LETTERS, permutation))
        original_to_new = {original: new for new, original in new_to_original.items()}
        baseline_letter = baseline[qid].get("answer", baseline[qid].get("subject_answer"))
        rows.append({
            "question_id": qid,
            "baseline_original_letter": baseline_letter,
            "new_to_original": new_to_original,
            "original_to_new": original_to_new,
            "baseline_content_new_letter": original_to_new[baseline_letter],
        })
    payload = {
        "status": "frozen_balanced_derangement_plan",
        "seed": seed,
        "n_questions": len(rows),
        "derangements": ["".join(value) for value in derangements],
        "mapping_definition": (
            "new_to_original[new second-presentation letter] = original first-presentation "
            "letter whose option content is displayed there"
        ),
        "balance": (
            "Question IDs are shuffled within current Baseline-answer letter, then cycled "
            "through all nine four-option derangements. No option content retains its letter."
        ),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze balanced option remappings")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    payload = prepare(args.manifest, args.baseline, args.output, args.seed)
    print(json.dumps({key: payload[key] for key in ("status", "seed", "n_questions")}, indent=2))


if __name__ == "__main__":
    main()
