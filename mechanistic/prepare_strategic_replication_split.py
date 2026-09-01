from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def prepare(manifest_path: Path, output_path: Path, seed: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    qids = [str(row["id"]) for row in manifest["questions"]]
    if len(qids) != len(set(qids)):
        raise ValueError("Manifest question IDs are not unique")
    rng = np.random.default_rng(seed)
    shuffled = list(qids)
    rng.shuffle(shuffled)
    midpoint = len(shuffled) // 2
    payload = {
        "status": "frozen_result_independent_split",
        "seed": int(seed),
        "manifest": str(manifest_path),
        "n_questions": len(qids),
        "discovery_question_ids": shuffled[:midpoint],
        "confirmation_question_ids": shuffled[midpoint:],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a result-independent replication split")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    payload = prepare(args.manifest, args.output, args.seed)
    print(json.dumps({
        "status": payload["status"],
        "n_questions": payload["n_questions"],
        "discovery": len(payload["discovery_question_ids"]),
        "confirmation": len(payload["confirmation_question_ids"]),
    }, indent=2))


if __name__ == "__main__":
    main()
