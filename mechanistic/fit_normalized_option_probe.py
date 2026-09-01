from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_option_newline_choice_probe import _score_layer


# Input residuals immediately preceding every ordinary-attention block.
TARGET_READOUTS = tuple(range(3, 64, 4))


def _rms_unit(values: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=-1)).astype(
        np.float32
    )
    return values / np.maximum(rms[..., None], 1e-6)


def fit(cache_dir: Path, discovery_plan: Path, output: Path) -> None:
    residuals = np.load(cache_dir / "option_newline_residuals.npy", mmap_mode="r")
    with np.load(cache_dir / "results.npz", allow_pickle=False) as loaded:
        qids = loaded["question_ids"].astype(str).tolist()
        choices = loaded["aggregated_ad_logits"].argmax(axis=-1)
        if not loaded["completed"].all():
            raise ValueError("Option residual cache is incomplete")
    discovery_ids = set(json.loads(discovery_plan.read_text())["question_ids"])
    train = np.asarray([i for i, qid in enumerate(qids) if qid in discovery_ids])
    test = np.asarray([i for i, qid in enumerate(qids) if qid not in discovery_ids])
    if len(train) != 251 or len(test) != 249:
        raise ValueError("Expected frozen 251/249 split")

    n_layers, width = residuals.shape[2], residuals.shape[-1]
    weights = np.zeros((n_layers, width), dtype=np.float32)
    means = np.zeros((n_layers, 4, width), dtype=np.float32)
    scales = np.ones((n_layers, width), dtype=np.float32)
    accuracy = np.full(n_layers, np.nan, dtype=np.float32)
    train_choice = choices[:, train].reshape(-1)
    test_choice = choices[:, test].reshape(-1)
    for readout in TARGET_READOUTS:
        layer = readout - 1
        train_values = _rms_unit(
            np.asarray(residuals[:, train, layer], dtype=np.float32).reshape(
                -1, 4, width
            )
        )
        test_values = _rms_unit(
            np.asarray(residuals[:, test, layer], dtype=np.float32).reshape(
                -1, 4, width
            )
        )
        scores, weight, letter_mean, scale = _score_layer(
            train_values, train_choice, test_values
        )
        weights[layer] = weight
        means[layer] = letter_mean
        scales[layer] = scale
        accuracy[layer] = np.mean(scores.argmax(axis=-1) == test_choice)
        print(
            f"normalized option probe readout {readout}: accuracy={accuracy[layer]:.4f}",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        weights=weights,
        letter_means=means,
        scales=scales,
        accuracy=accuracy,
        target_readouts=np.asarray(TARGET_READOUTS),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit(args.cache_dir, args.discovery_plan, args.output)


if __name__ == "__main__":
    main()
