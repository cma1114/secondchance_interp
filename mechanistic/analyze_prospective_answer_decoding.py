from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = ("game", "neutral")
DECODER_TRAINING = ("shared", "game", "neutral")
RIDGE_MULTIPLIERS = np.asarray(
    [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0],
    dtype=np.float64,
)


def _center_candidates(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values - values.mean(axis=-1, keepdims=True)


def _rms_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    scale = np.sqrt(np.maximum(np.mean(values * values, axis=-1, keepdims=True), 1e-12))
    return values / scale


def _pooled_correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = _center_candidates(prediction).reshape(-1).astype(np.float64)
    target = _center_candidates(target).reshape(-1).astype(np.float64)
    prediction -= prediction.mean()
    target -= target.mean()
    denominator = np.linalg.norm(prediction) * np.linalg.norm(target)
    return float(prediction @ target / denominator) if denominator else float("nan")


def _r2(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = _center_candidates(prediction).astype(np.float64)
    target = _center_candidates(target).astype(np.float64)
    denominator = float(np.sum(target * target))
    if denominator == 0:
        return float("nan")
    return float(1.0 - np.sum((prediction - target) ** 2) / denominator)


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "pooled_correlation": _pooled_correlation(prediction, target),
        "r2": _r2(prediction, target),
        "winner_accuracy": float(
            np.mean(np.argmax(prediction, axis=-1) == np.argmax(target, axis=-1))
        ),
    }


def _discovery_mask(path: Path, qids: list[str]) -> np.ndarray:
    payload = json.loads(path.read_text())
    identifiers = payload.get("discovery_question_ids", payload.get("question_ids"))
    if identifiers is None:
        raise ValueError(f"No discovery question IDs in {path}")
    selected = {str(value) for value in identifiers}
    mask = np.asarray([qid in selected for qid in qids], dtype=bool)
    if not mask.any() or mask.all():
        raise ValueError(f"Discovery split in {path} must leave a nonempty confirmation split")
    return mask


def _paired_folds(indices: np.ndarray, labels: np.ndarray, seed: int, count: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(count)]
    for label in np.unique(labels[indices]):
        rows = indices[labels[indices] == label].copy()
        rng.shuffle(rows)
        for offset, row in enumerate(rows):
            buckets[offset % count].append(int(row))
    folds = [np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets]
    if any(len(fold) == 0 for fold in folds):
        raise RuntimeError("A decoder cross-validation fold is empty")
    return folds


def _prepare_training(
    states: np.ndarray,
    targets: np.ndarray,
    question_indices: np.ndarray,
    training: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Center each condition separately, then return the requested training rows."""
    x_means = np.stack(
        [states[ci, question_indices].mean(axis=0) for ci in range(2)], axis=0
    ).astype(np.float32)
    y_means = np.stack(
        [targets[ci, question_indices].mean(axis=0) for ci in range(2)], axis=0
    ).astype(np.float32)
    if training == "shared":
        x = np.concatenate(
            [states[ci, question_indices] - x_means[ci] for ci in range(2)], axis=0
        )
        y = np.concatenate(
            [targets[ci, question_indices] - y_means[ci] for ci in range(2)], axis=0
        )
    else:
        ci = CONDITIONS.index(training)
        x = states[ci, question_indices] - x_means[ci]
        y = targets[ci, question_indices] - y_means[ci]
    return x.astype(np.float32), y.astype(np.float32), x_means, y_means


def _ridge_coefficients(
    x: np.ndarray, y: np.ndarray, multiplier: float
) -> tuple[np.ndarray, float]:
    kernel = x @ x.T
    scale = float(np.trace(kernel) / max(len(x), 1))
    regularization = max(float(multiplier) * scale, 1e-8)
    dual = np.linalg.solve(
        kernel + regularization * np.eye(len(x), dtype=np.float32), y
    )
    return (x.T @ dual).astype(np.float32), regularization


def _ridge_coefficient_path(
    x: np.ndarray, y: np.ndarray, multipliers: np.ndarray
) -> list[tuple[np.ndarray, float]]:
    """Solve a complete ridge path with one eigendecomposition of the dual kernel."""
    kernel = x @ x.T
    kernel = (kernel + kernel.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ y
    scale = float(np.trace(kernel) / max(len(x), 1))
    fits: list[tuple[np.ndarray, float]] = []
    for multiplier in multipliers:
        regularization = max(float(multiplier) * scale, 1e-8)
        dual = eigenvectors @ (projected / (eigenvalues[:, None] + regularization))
        fits.append(((x.T @ dual).astype(np.float32), regularization))
    return fits


def _predict_all_conditions(
    states: np.ndarray,
    coefficients: np.ndarray,
    x_means: np.ndarray,
    y_means: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            (states[ci] - x_means[ci]) @ coefficients + y_means[ci]
            for ci in range(2)
        ],
        axis=0,
    ).astype(np.float32)


def _select_ridge(
    states: np.ndarray,
    targets: np.ndarray,
    discovery_indices: np.ndarray,
    folds: list[np.ndarray],
    training: str,
) -> tuple[float, list[dict[str, Any]]]:
    predictions = np.full(
        (len(RIDGE_MULTIPLIERS), 2, len(discovery_indices), 4),
        np.nan,
        dtype=np.float32,
    )
    discovery_lookup = {int(row): offset for offset, row in enumerate(discovery_indices)}
    for heldout in folds:
        train = discovery_indices[~np.isin(discovery_indices, heldout)]
        x, y, x_means, y_means = _prepare_training(states, targets, train, training)
        for mi, (coefficients, _regularization) in enumerate(
            _ridge_coefficient_path(x, y, RIDGE_MULTIPLIERS)
        ):
            for ci in range(2):
                offsets = [discovery_lookup[int(row)] for row in heldout]
                predictions[mi, ci, offsets] = (
                    (states[ci, heldout] - x_means[ci]) @ coefficients + y_means[ci]
                )
    target = targets[:, discovery_indices]
    records = []
    for mi, multiplier in enumerate(RIDGE_MULTIPLIERS):
        if not np.isfinite(predictions[mi]).all():
            raise RuntimeError("Decoder cross-validation prediction is incomplete")
        aggregate = _metrics(predictions[mi], target)
        records.append(
            {
                "multiplier": float(multiplier),
                **aggregate,
                "by_condition": {
                    condition: _metrics(predictions[mi, ci], target[ci])
                    for ci, condition in enumerate(CONDITIONS)
                },
            }
        )
    scores = np.asarray([record["pooled_correlation"] for record in records])
    if not np.isfinite(scores).all():
        raise RuntimeError("Decoder cross-validation returned a non-finite score")
    return float(RIDGE_MULTIPLIERS[int(np.argmax(scores))]), records


def analyze_dataset(spec: dict[str, Any], max_layers: int | None = None) -> None:
    results = np.load(spec["results"], allow_pickle=False)
    qids = [str(value) for value in results["question_ids"].tolist()]
    conditions = tuple(str(value) for value in results["conditions"].tolist())
    if conditions != CONDITIONS:
        raise ValueError(f"Unexpected condition order {conditions}")
    targets = _center_candidates(results["direct_logits"])
    discovery = _discovery_mask(Path(spec["discovery_plan"]), qids)
    discovery_indices = np.flatnonzero(discovery)
    confirmation_indices = np.flatnonzero(~discovery)
    paired_labels = 4 * np.argmax(targets[0], axis=-1) + np.argmax(targets[1], axis=-1)
    folds = _paired_folds(discovery_indices, paired_labels, int(spec.get("seed", 0)))

    residual_path = Path(spec.get("residuals", Path(spec["output"]) / "decision_residuals.npy"))
    residuals = np.load(residual_path, mmap_mode="r")
    if residuals.ndim != 4 or residuals.shape[:2] != (2, len(qids)):
        raise ValueError(f"Unexpected residual array shape {residuals.shape}")
    available_layers = int(residuals.shape[2])
    layer_count = (
        available_layers
        if max_layers is None
        else min(int(max_layers), available_layers)
    )
    width = int(residuals.shape[-1])
    predictions = np.full(
        (3, 2, len(qids), layer_count, 4), np.nan, dtype=np.float16
    )
    coefficients_out = np.empty((3, layer_count, width, 4), dtype=np.float16)
    input_means_out = np.empty((3, layer_count, 2, width), dtype=np.float16)
    output_means_out = np.empty((3, layer_count, 2, 4), dtype=np.float32)
    regularizations = np.empty((3, layer_count), dtype=np.float64)
    summary: dict[str, Any] = {
        "dataset": spec["name"],
        "n_questions": len(qids),
        "n_discovery": int(discovery.sum()),
        "n_confirmation": int((~discovery).sum()),
        "target": "exact final A-D logits centered within question",
        "input": "final-decision-position post-block residual, RMS-normalized within row",
        "centering": (
            "Input and output means are estimated on discovery questions separately for "
            "Game and Neutral. Cross-condition transfer changes only the fitted coefficient basis."
        ),
        "decoder_training": list(DECODER_TRAINING),
        "layers": [],
    }

    for layer in range(layer_count):
        states = _rms_normalize(np.asarray(residuals[:, :, layer], dtype=np.float32))
        layer_record: dict[str, Any] = {"layer": layer + 1, "decoders": {}}
        for model_index, training in enumerate(DECODER_TRAINING):
            multiplier, cv_records = _select_ridge(
                states, targets, discovery_indices, folds, training
            )
            x, y, x_means, y_means = _prepare_training(
                states, targets, discovery_indices, training
            )
            coefficients, regularization = _ridge_coefficients(x, y, multiplier)
            predicted = _predict_all_conditions(
                states, coefficients, x_means, y_means
            )
            predictions[model_index, :, :, layer] = predicted.astype(np.float16)
            coefficients_out[model_index, layer] = coefficients.astype(np.float16)
            input_means_out[model_index, layer] = x_means.astype(np.float16)
            output_means_out[model_index, layer] = y_means
            regularizations[model_index, layer] = regularization
            layer_record["decoders"][training] = {
                "selected_multiplier": multiplier,
                "regularization": regularization,
                "cross_validation": cv_records,
                "confirmation": {
                    condition: _metrics(
                        predicted[ci, confirmation_indices],
                        targets[ci, confirmation_indices],
                    )
                    for ci, condition in enumerate(CONDITIONS)
                },
            }
        summary["layers"].append(layer_record)
        print(f"{spec['name']} prospective decoder: {layer + 1}/{layer_count}", flush=True)

    output = Path(spec["analysis_output"])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz",
        question_ids=np.asarray(qids),
        conditions=np.asarray(CONDITIONS),
        decoder_training=np.asarray(DECODER_TRAINING),
        discovery=discovery,
        rank_order=results["rank_order"],
        exact_final_scores=targets.astype(np.float32),
        predictions=predictions,
    )
    np.savez_compressed(
        output / "decoders.npz",
        decoder_training=np.asarray(DECODER_TRAINING),
        coefficients=coefficients_out,
        input_means=input_means_out,
        output_means=output_means_out,
        regularizations=regularizations,
    )
    (output / "decoder_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--max-layers", type=int)
    args = parser.parse_args()
    specs = json.loads(args.specs.read_text())["datasets"]
    for spec in specs:
        analyze_dataset(spec, args.max_layers)


if __name__ == "__main__":
    main()
