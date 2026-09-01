from __future__ import annotations

import argparse
import json
from pathlib import Path


LETTERS = "ABCD"


def prepare(manifest_path: Path, baseline_path: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    baseline = json.loads(baseline_path.read_text())["results"]
    identity = {letter: letter for letter in LETTERS}
    rows = []
    for question in manifest["questions"]:
        qid = question["id"]
        if qid not in baseline:
            raise KeyError(f"Missing Baseline result for {qid}")
        answer = baseline[qid].get(
            "answer", baseline[qid].get("subject_answer")
        )
        if answer not in LETTERS:
            raise ValueError(f"Non-A-D Baseline answer for {qid}: {answer!r}")
        rows.append(
            {
                "question_id": qid,
                "baseline_original_letter": answer,
                "baseline_content_new_letter": answer,
                "new_to_original": identity,
                "original_to_new": identity,
            }
        )
    payload = {
        "status": "frozen",
        "definition": (
            "Identity option mapping for both presentations; used to collect the "
            "non-remapped incorrect-again condition."
        ),
        "n_questions": len(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(args.manifest, args.baseline, args.output)
    print(json.dumps({"n_questions": payload["n_questions"]}))


if __name__ == "__main__":
    main()
