from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


LETTERS = "ABCD"


def _inverse(new_to_original: dict[str, str]) -> dict[str, str]:
    return {original: new for new, original in new_to_original.items()}


def _complementary_mappings(
    existing_new_to_original: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Complete identity + an existing derangement to a four-map Latin square."""
    existing_original_to_new = _inverse(existing_new_to_original)
    used = {
        original: {original, existing_original_to_new[original]}
        for original in LETTERS
    }
    candidates: list[dict[str, str]] = []
    for permutation in itertools.permutations(LETTERS):
        mapping = dict(zip(LETTERS, permutation))
        inverse = _inverse(mapping)
        if all(inverse[original] not in used[original] for original in LETTERS):
            candidates.append(mapping)
    if not candidates:
        raise RuntimeError("Could not find a complementary third mapping")
    third = candidates[0]
    third_inverse = _inverse(third)
    fourth_by_position: dict[str, str] = {}
    for original in LETTERS:
        remaining = set(LETTERS) - used[original] - {third_inverse[original]}
        if len(remaining) != 1:
            raise RuntimeError("Complementary mapping is not uniquely determined")
        fourth_by_position[remaining.pop()] = original
    if set(fourth_by_position) != set(LETTERS) or set(fourth_by_position.values()) != set(LETTERS):
        raise RuntimeError("Fourth mapping is not a permutation")
    # Prompt construction deliberately rejects option dictionaries whose key
    # insertion order is not A, B, C, D. Canonicalize after solving by content.
    fourth = {letter: fourth_by_position[letter] for letter in LETTERS}
    return third, fourth


def prepare(existing_plan: Path, output_two: Path, output_three: Path) -> None:
    payload = json.loads(existing_plan.read_text())
    rows_two: list[dict] = []
    rows_three: list[dict] = []
    for row in payload["rows"]:
        qid = row["question_id"]
        mapping_two, mapping_three = _complementary_mappings(row["new_to_original"])
        for target, mapping, index in (
            (rows_two, mapping_two, 2),
            (rows_three, mapping_three, 3),
        ):
            target.append(
                {
                    "question_id": qid,
                    "mapping_index": index,
                    "new_to_original": mapping,
                    "original_to_new": _inverse(mapping),
                }
            )

        all_maps = [
            {letter: letter for letter in LETTERS},
            row["new_to_original"],
            mapping_two,
            mapping_three,
        ]
        for original in LETTERS:
            positions = [_inverse(mapping)[original] for mapping in all_maps]
            if set(positions) != set(LETTERS):
                raise RuntimeError(f"{qid}: {original} does not occupy A-D exactly once")

    shared = {
        "status": "frozen",
        "source_existing_plan": str(existing_plan),
        "n_questions": len(rows_two),
        "mapping_definition": (
            "new_to_original[new letter] is the original content identity. Together "
            "with the identity prompt and source derangement, mappings 2 and 3 place "
            "every option content at A, B, C, and D exactly once."
        ),
    }
    for path, rows, index in (
        (output_two, rows_two, 2),
        (output_three, rows_three, 3),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**shared, "mapping_index": index, "rows": rows}, indent=2)
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-plan", type=Path, required=True)
    parser.add_argument("--output-two", type=Path, required=True)
    parser.add_argument("--output-three", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.existing_plan, args.output_two, args.output_three)


if __name__ == "__main__":
    main()
