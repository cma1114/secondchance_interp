from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io import atomic_save_npz
from .run_remapped_repeated_w1_relay import INTERVENTION_CELLS


QUESTION_AXES = {
    "completed": 0,
    "trusted_natural_logits": 1,
    "same_batch_natural_logits": 1,
    "intervention_logits": 2,
    "query_position_counts": 2,
    "source_position_counts": 2,
    "w1_displayed_letters": 0,
    "control_displayed_letters": 0,
}


def merge(inputs: list[Path], output: Path) -> None:
    if len(inputs) < 2:
        raise ValueError("At least two shard inputs are required")
    shards = []
    for path in inputs:
        with np.load(path, allow_pickle=False) as loaded:
            shards.append({key: loaded[key] for key in loaded.files})
    expected_cells = [cell["id"] for cell in INTERVENTION_CELLS]
    for shard in shards:
        if shard["intervention_cells"].astype(str).tolist() != expected_cells:
            raise ValueError("Intervention cells differ across shards")
        if not np.all(shard["completed"]):
            raise ValueError("A shard is incomplete")
    merged = {
        "question_ids": np.concatenate([shard["question_ids"] for shard in shards]),
        "intervention_cells": shards[0]["intervention_cells"],
    }
    for key, axis in QUESTION_AXES.items():
        merged[key] = np.concatenate([shard[key] for shard in shards], axis=axis)
    if len(set(merged["question_ids"].astype(str).tolist())) != len(
        merged["question_ids"]
    ):
        raise ValueError("Merged shards contain duplicate question IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(output, **merged)
    metadata = {
        "experiment": "merged canonical remapped repeated-W1 downstream relay localization",
        "inputs": [str(path) for path in inputs],
        "n_questions": int(len(merged["question_ids"])),
        "intervention_cells": expected_cells,
        "all_completed": bool(np.all(merged["completed"])),
        "all_logits_finite": bool(
            np.isfinite(merged["same_batch_natural_logits"]).all()
            and np.isfinite(merged["intervention_logits"]).all()
        ),
    }
    output.with_name("merge_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge(args.inputs, args.output)


if __name__ == "__main__":
    main()
