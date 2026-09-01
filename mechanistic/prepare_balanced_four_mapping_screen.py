from __future__ import annotations

import argparse
import json
from pathlib import Path


LETTERS = "ABCD"


def _by_qid(path: Path) -> dict[str, dict]:
    return {row["question_id"]: row for row in json.loads(path.read_text())["rows"]}


def prepare(
    manifest_path: Path,
    canonical_path: Path,
    mapping_2_path: Path,
    mapping_3_path: Path,
    output_path: Path,
) -> None:
    manifest = json.loads(manifest_path.read_text())
    qids = [row["id"] for row in manifest["questions"]]
    canonical = _by_qid(canonical_path)
    mapping_2 = _by_qid(mapping_2_path)
    mapping_3 = _by_qid(mapping_3_path)
    identity = {letter: letter for letter in LETTERS}
    rows = []
    for qid in qids:
        mappings = [
            {"mapping_index": 0, "new_to_original": identity},
            {"mapping_index": 1, "new_to_original": canonical[qid]["new_to_original"]},
            {"mapping_index": 2, "new_to_original": mapping_2[qid]["new_to_original"]},
            {"mapping_index": 3, "new_to_original": mapping_3[qid]["new_to_original"]},
        ]
        for original in LETTERS:
            observed = [row["new_to_original"] for row in mappings]
            positions = [
                next(new for new in LETTERS if mapping[new] == original)
                for mapping in observed
            ]
            if set(positions) != set(LETTERS):
                raise RuntimeError(f"{qid}: {original} does not occupy A-D exactly once")
        rows.append({"question_id": qid, "mappings": mappings})
    payload = {
        "status": "frozen_balanced_four_mapping_screen",
        "definition": (
            "Identity, canonical remap, and two complementary mappings; every "
            "semantic option occupies each literal letter exactly once per question."
        ),
        "question_ids": qids,
        "n_questions": len(qids),
        "n_mappings_per_question": 4,
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--mapping-2", type=Path, required=True)
    parser.add_argument("--mapping-3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.canonical, args.mapping_2, args.mapping_3, args.output)


if __name__ == "__main__":
    main()
