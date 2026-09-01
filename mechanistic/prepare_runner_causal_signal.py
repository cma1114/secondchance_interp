from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import load_activation_dataset


def prepare(mechanistic_dir: str, bridge_path: str, output_path: str, readout: int) -> None:
    data = load_activation_dataset(mechanistic_dir, ["baseline", "incorrect", "neutral"])
    with np.load(bridge_path, allow_pickle=False) as source:
        signal = source["signal"]
        order = source["order"]
    if signal.shape[0] != len(data.question_ids) or order.shape != (len(data.question_ids), 4):
        raise ValueError("Bridge artifact does not align with the activation dataset")
    observed_order = np.argsort(-data.logits[:, 0, -1], axis=-1)
    if not np.array_equal(order, observed_order):
        raise ValueError("Bridge order no longer matches the natural baseline ordering")
    values = signal[:, readout]
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite candidate signal at readout {readout}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        question_ids=np.asarray(data.question_ids),
        signal=values.astype(np.float32),
        order=order.astype(np.int8),
        readout=np.asarray(readout),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the runner residual used by the causal intervention")
    parser.add_argument("--mechanistic-dir", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--readout", type=int, default=49)
    args = parser.parse_args()
    prepare(args.mechanistic_dir, args.bridge, args.output, args.readout)


if __name__ == "__main__":
    main()
