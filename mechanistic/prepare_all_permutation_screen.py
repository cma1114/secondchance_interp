from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from . import LETTERS


def prepare(manifest_path: Path, output_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    qids = [row["id"] for row in manifest["questions"]]
    permutations = list(itertools.permutations(LETTERS))
    identity = tuple(LETTERS)
    permutations = [identity] + [value for value in permutations if value != identity]
    mappings = []
    for index, values in enumerate(permutations):
        new_to_original = dict(zip(LETTERS, values))
        mappings.append({
            "mapping_index": index,
            "new_to_original": new_to_original,
            "original_to_new": {original: new for new, original in new_to_original.items()},
            "identity": index == 0,
        })
    payload = {
        "status": "frozen_before_screen",
        "definition": "All 24 permutations of the four semantic options.",
        "n_questions": len(qids),
        "n_mappings_per_question": len(mappings),
        "question_ids": qids,
        "mappings": mappings,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.output)


if __name__ == "__main__":
    main()
