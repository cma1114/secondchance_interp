from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io import atomic_save_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--natural-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_qids = [
        row["id"] for row in json.loads(args.manifest.read_text())["questions"]
    ]
    pieces = [dict(np.load(path, allow_pickle=False)) for path in args.shards]
    source = dict(np.load(args.natural_source, allow_pickle=False))
    seen: dict[str, tuple[int, int]] = {}
    for shard_index, piece in enumerate(pieces):
        if not np.all(piece["completed"]):
            raise ValueError(f"Incomplete shard: {args.shards[shard_index]}")
        for row, qid in enumerate(piece["question_ids"].astype(str)):
            if qid in seen:
                raise ValueError(f"Duplicate question across shards: {qid}")
            seen[qid] = (shard_index, row)
    if set(seen) != set(manifest_qids):
        raise ValueError(
            f"Shard coverage mismatch: got {len(seen)} of {len(manifest_qids)}"
        )

    n = len(manifest_qids)
    merged: dict[str, np.ndarray] = {
        "question_ids": np.asarray(manifest_qids),
        "completed": np.ones(n, dtype=bool),
    }
    for key in pieces[0]:
        if key in merged:
            continue
        sample = pieces[0][key]
        if sample.ndim < 2:
            raise ValueError(f"Unexpected shard array: {key} {sample.shape}")
        shape = list(sample.shape)
        shape[1] = n
        values = np.empty(shape, dtype=sample.dtype)
        for target_row, qid in enumerate(manifest_qids):
            shard_index, source_row = seen[qid]
            values[:, target_row] = pieces[shard_index][key][:, source_row]
        merged[key] = values

    source_qids = source["question_ids"].astype(str).tolist()
    if source_qids != manifest_qids:
        raise ValueError("Natural source order differs from manifest")
    validation = {}
    for key in ("natural_logits", "natural_projection", "natural_residual_norm"):
        difference = float(np.max(np.abs(merged[key] - source[key])))
        validation[key] = difference
        if difference != 0.0:
            raise ValueError(f"Natural companion mismatch for {key}: {difference}")

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(args.output / "results.npz", **merged)
    (args.output / "merge_validation.json").write_text(
        json.dumps(
            {
                "shards": [str(path) for path in args.shards],
                "n_questions": n,
                "natural_max_abs_difference": validation,
            },
            indent=2,
        ) + "\n"
    )


if __name__ == "__main__":
    main()
