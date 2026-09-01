from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .io import atomic_save_npz


def merge(inputs: list[Path], output_dir: Path) -> None:
    payloads = []
    for root in inputs:
        arrays = dict(np.load(root / "results.npz", allow_pickle=False))
        if not np.all(arrays["completed"]):
            raise ValueError(f"Incomplete shard: {root}")
        payloads.append(arrays)
    fixed = ("gla_layer_indices",)
    for key in fixed:
        for arrays in payloads[1:]:
            if not np.array_equal(payloads[0][key], arrays[key]):
                raise ValueError(f"Shard mismatch: {key}")
    question_axis = {
        "question_ids": 0,
        "x_second_letter": 0,
        "y_second_letter": 0,
        "completed": 0,
        "natural_logits": 1,
        "identity_state_logits": 1,
        "cross_state_logits": 2,
        "cross_state_delta_norm": 1,
        "recipient_state_norm": 1,
    }
    merged = {"gla_layer_indices": payloads[0]["gla_layer_indices"]}
    for key, axis in question_axis.items():
        merged[key] = np.concatenate([arrays[key] for arrays in payloads], axis=axis)
    order = np.argsort(merged["question_ids"].astype(str), kind="stable")
    for key, axis in question_axis.items():
        merged[key] = np.take(merged[key], order, axis=axis)
    if len(np.unique(merged["question_ids"].astype(str))) != len(order):
        raise ValueError("Duplicate question IDs after shard merge")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(output_dir / "results.npz", **merged)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    merge(args.inputs, args.output_dir)


if __name__ == "__main__":
    main()
