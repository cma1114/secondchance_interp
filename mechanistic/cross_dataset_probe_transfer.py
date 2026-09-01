from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .all_trial_figures import _style
from .data import load_activation_dataset
from .io import shard_path
from .probes import stratified_folds


LETTERS = "ABCD"
COLORS = {
    "simple_within": "#333333",
    "trivia_within": "#777777",
    "simple_to_trivia": "#0072B2",
    "trivia_to_simple": "#D55E00",
}


def _labels(data) -> np.ndarray:
    values = []
    for question_id in data.question_ids:
        token = data.metadata[(question_id, "baseline")]["full_vocab_top_token"].strip()
        if token not in LETTERS:
            raise ValueError(f"Non-A-D native output for {question_id}: {token!r}")
        values.append(LETTERS.index(token))
    return np.asarray(values, dtype=int)


def _load_residuals(root: str | Path, question_ids: list[str]) -> np.ndarray:
    with np.load(shard_path(root, "baseline", question_ids[0]), allow_pickle=False) as shard:
        shape = shard["residuals"].shape
    values = np.empty((len(question_ids), *shape), dtype=np.float16)
    for index, question_id in enumerate(question_ids):
        with np.load(shard_path(root, "baseline", question_id), allow_pickle=False) as shard:
            values[index] = shard["residuals"]
    return values


def _fit(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    centers = np.stack([z[labels == letter].mean(axis=0) for letter in range(4)])
    return mean, scale, centers


def _predict(model: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    mean, scale, centers = model
    return np.argmax(((x - mean) / scale) @ centers.T, axis=-1)


def _balanced_accuracy(prediction: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean([
        np.mean(prediction[labels == letter] == letter) for letter in range(4)
    ]))


def _within_cv(x: np.ndarray, labels: np.ndarray, folds: int, seed: int) -> float:
    prediction = np.empty_like(labels)
    all_indices = np.arange(len(labels))
    for test in stratified_folds(labels, folds, seed):
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        prediction[test] = _predict(_fit(x[train], labels[train]), x[test])
    return _balanced_accuracy(prediction, labels)


def _raw_directions(model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    _, scale, centers = model
    weights = centers / scale[None, :]
    return weights - weights.mean(axis=0, keepdims=True)


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def _stratified_halves(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left, right = [], []
    for letter in range(4):
        indices = np.flatnonzero(labels == letter)
        rng.shuffle(indices)
        midpoint = len(indices) // 2
        left.extend(indices[:midpoint])
        right.extend(indices[midpoint:])
    return np.asarray(sorted(left)), np.asarray(sorted(right))


def _split_half_similarity(
    x: np.ndarray, labels: np.ndarray, repeats: int, seed: int
) -> np.ndarray:
    values = []
    for repeat in range(repeats):
        left, right = _stratified_halves(labels, seed + repeat)
        left_direction = _raw_directions(_fit(x[left], labels[left]))
        right_direction = _raw_directions(_fit(x[right], labels[right]))
        values.append(_cosine_rows(left_direction, right_direction))
    return np.mean(values, axis=0)


def analyze(
    simple_dir: str | Path,
    trivia_dir: str | Path,
    output_dir: str | Path,
    folds: int,
    seed: int,
    reliability_repeats: int,
) -> None:
    import matplotlib.pyplot as plt

    simple_data = load_activation_dataset(simple_dir, ["baseline"])
    trivia_data = load_activation_dataset(trivia_dir, ["baseline"])
    simple_labels = _labels(simple_data)
    trivia_labels = _labels(trivia_data)
    simple_residuals = _load_residuals(simple_dir, simple_data.question_ids)
    trivia_residuals = _load_residuals(trivia_dir, trivia_data.question_ids)
    if simple_residuals.shape[1:] != trivia_residuals.shape[1:]:
        raise ValueError(
            f"Residual shapes differ: {simple_residuals.shape} versus {trivia_residuals.shape}"
        )

    n_layers = simple_residuals.shape[1]
    layers = np.arange(n_layers)
    accuracy = {key: np.empty(n_layers) for key in COLORS}
    matched_cosine = np.empty((n_layers, 4))
    mismatched_cosine = np.empty((n_layers, 4))
    simple_reliability = np.empty((n_layers, 4))
    trivia_reliability = np.empty((n_layers, 4))

    for layer in layers:
        simple_x = simple_residuals[:, layer].astype(np.float32)
        trivia_x = trivia_residuals[:, layer].astype(np.float32)
        simple_model = _fit(simple_x, simple_labels)
        trivia_model = _fit(trivia_x, trivia_labels)

        accuracy["simple_within"][layer] = _within_cv(simple_x, simple_labels, folds, seed)
        accuracy["trivia_within"][layer] = _within_cv(trivia_x, trivia_labels, folds, seed)
        accuracy["simple_to_trivia"][layer] = _balanced_accuracy(
            _predict(simple_model, trivia_x), trivia_labels
        )
        accuracy["trivia_to_simple"][layer] = _balanced_accuracy(
            _predict(trivia_model, simple_x), simple_labels
        )

        simple_direction = _raw_directions(simple_model)
        trivia_direction = _raw_directions(trivia_model)
        matched_cosine[layer] = _cosine_rows(simple_direction, trivia_direction)
        for letter in range(4):
            mismatched_cosine[layer, letter] = np.mean([
                _cosine_rows(simple_direction[letter : letter + 1],
                             trivia_direction[other : other + 1])[0]
                for other in range(4) if other != letter
            ])
        simple_reliability[layer] = _split_half_similarity(
            simple_x, simple_labels, reliability_repeats, seed
        )
        trivia_reliability[layer] = _split_half_similarity(
            trivia_x, trivia_labels, reliability_repeats, seed + 1000
        )
        print(f"layer {layer}/{n_layers - 1}", flush=True)

    output = Path(output_dir)
    figure_dir = output / "preserved_figures"
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer in layers:
        for metric, values in accuracy.items():
            rows.append({"layer": int(layer), "family": "accuracy", "metric": metric,
                         "letter": "", "value": float(values[layer])})
        for letter, name in enumerate(LETTERS):
            rows.extend((
                {"layer": int(layer), "family": "direction", "metric": "matched_cross_dataset_cosine",
                 "letter": name, "value": float(matched_cosine[layer, letter])},
                {"layer": int(layer), "family": "direction", "metric": "mismatched_cross_dataset_cosine",
                 "letter": name, "value": float(mismatched_cosine[layer, letter])},
                {"layer": int(layer), "family": "reliability", "metric": "simple_split_half",
                 "letter": name, "value": float(simple_reliability[layer, letter])},
                {"layer": int(layer), "family": "reliability", "metric": "trivia_split_half",
                 "letter": name, "value": float(trivia_reliability[layer, letter])},
            ))
    with (output / "cross_dataset_probe_transfer.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "probe": "standardized class-centroid linear probe",
        "target": "native full-vocabulary Baseline output letter",
        "n_simple": len(simple_labels),
        "n_trivia": len(trivia_labels),
        "folds": folds,
        "direction_definition": "class weight minus mean of all four class weights, in raw residual coordinates",
        "selected_layers": {},
    }
    for layer in (32, 40, 44, 48, 52, 56, 60, 64):
        summary["selected_layers"][str(layer)] = {
            **{key: float(values[layer]) for key, values in accuracy.items()},
            "matched_direction_cosine": {
                letter: float(matched_cosine[layer, index]) for index, letter in enumerate(LETTERS)
            },
            "mean_matched_direction_cosine": float(matched_cosine[layer].mean()),
            "mean_mismatched_direction_cosine": float(mismatched_cosine[layer].mean()),
            "simple_split_half": float(simple_reliability[layer].mean()),
            "trivia_split_half": float(trivia_reliability[layer].mean()),
        }
    (output / "cross_dataset_probe_transfer_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    _style()
    figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.3), sharex=True)
    labels = {
        "simple_within": "SimpleMC within-dataset",
        "trivia_within": "TriviaMC within-dataset",
        "simple_to_trivia": "SimpleMC → TriviaMC",
        "trivia_to_simple": "TriviaMC → SimpleMC",
    }
    styles = {
        "simple_within": "-", "trivia_within": "--",
        "simple_to_trivia": "-", "trivia_to_simple": "--",
    }
    for key, values in accuracy.items():
        axes[0].plot(layers, values, color=COLORS[key], linestyle=styles[key],
                     linewidth=1.5, label=labels[key])
    axes[0].axhline(0.25, color="#555555", linewidth=0.8, linestyle=(0, (3, 2)))
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Letter-balanced accuracy")
    axes[0].set_title("A  Cross-dataset prediction", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7.2, loc="upper left")

    letter_colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for index, (letter, color) in enumerate(zip(LETTERS, letter_colors)):
        axes[1].plot(layers, matched_cosine[:, index], color=color, linewidth=1.35,
                     label=f"{letter} vs rest")
    axes[1].plot(layers, matched_cosine.mean(axis=1), color="#111111", linewidth=2.0,
                 label="Matched-letter mean")
    axes[1].plot(layers, mismatched_cosine.mean(axis=1), color="#888888", linewidth=1.2,
                 linestyle=":", label="Mismatched-letter mean")
    simple_full_reliability = np.clip(
        2 * simple_reliability / np.maximum(1 + simple_reliability, 1e-8), 0, 1
    )
    trivia_full_reliability = np.clip(
        2 * trivia_reliability / np.maximum(1 + trivia_reliability, 1e-8), 0, 1
    )
    ceiling = np.sqrt(simple_full_reliability * trivia_full_reliability).mean(axis=1)
    axes[1].plot(layers, ceiling, color="#555555", linewidth=1.0, linestyle="--",
                 label="Noise ceiling from split halves")
    axes[1].axhline(0, color="#555555", linewidth=0.65)
    axes[1].set_ylim(-0.4, 1)
    axes[1].set_ylabel("Cosine similarity in raw residual space")
    axes[1].set_title("B  Are corresponding letter directions shared?", loc="left",
                      fontweight="bold")
    axes[1].legend(frameon=False, fontsize=7.0, loc="upper left", ncol=2)

    for axis in axes:
        axis.set_xlim(0, n_layers - 1)
        axis.set_xticks(np.arange(0, n_layers, 8))
        axis.set_xlabel(f"Residual readout (0 = embedding; {n_layers - 1} = final block)")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Qwen3.6-27B: SimpleMC–TriviaMC answer-probe transfer",
                    fontsize=10.5, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.0)
    figure.savefig(figure_dir / "cross_dataset_probe_transfer.png", dpi=300,
                   bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bidirectional cross-dataset answer-probe transfer")
    parser.add_argument("--simple", required=True)
    parser.add_argument("--trivia", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reliability-repeats", type=int, default=5)
    args = parser.parse_args()
    analyze(args.simple, args.trivia, args.output, args.folds, args.seed,
            args.reliability_repeats)


if __name__ == "__main__":
    main()
