from __future__ import annotations

import numpy as np


RANK_AXIS = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
RANK_AXIS_DENOMINATOR = float(np.sum(RANK_AXIS**2))


def center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=-1, keepdims=True)
    return -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=-1)


def outcome_metrics(values: np.ndarray, baseline: np.ndarray, winners: np.ndarray, correct: np.ndarray) -> dict[str, np.ndarray]:
    values = center(values)
    baseline = center(baseline)
    row = np.arange(len(values))
    winner_score = values[row, winners]
    denominator = np.maximum(np.sum(baseline * baseline, axis=-1), 1e-12)
    choice = np.argmax(values, axis=-1)
    order = np.argsort(-baseline, axis=-1)
    condition_minus_baseline = values - baseline
    aligned_delta = np.take_along_axis(condition_minus_baseline, order, axis=-1)
    return {
        "baseline_alignment": np.sum(values * baseline, axis=-1) / denominator,
        "winner_advantage": winner_score - (values.sum(axis=-1) - winner_score) / 3.0,
        "ad_entropy": entropy(values),
        "ad_spread": values.std(axis=-1),
        "switch": (choice != winners).astype(float),
        "accuracy": (choice == correct).astype(float),
        "rank_opposed_slope": (
            np.sum(aligned_delta * RANK_AXIS, axis=-1) / RANK_AXIS_DENOMINATOR
        ),
    }


def causal_geometry(intervened: np.ndarray, natural: np.ndarray, baseline: np.ndarray) -> dict[str, np.ndarray]:
    delta = center(intervened) - center(natural)
    baseline = center(baseline)
    denominator = np.maximum(np.sum(baseline * baseline, axis=-1), 1e-12)
    coefficient = np.sum(delta * baseline, axis=-1) / denominator
    parallel = coefficient[:, None] * baseline
    orthogonal = delta - parallel
    total_l2 = np.linalg.norm(delta, axis=-1)
    return {
        "causal_total_l1": np.sum(np.abs(delta), axis=-1),
        "causal_total_l2": total_l2,
        "causal_baseline_coefficient": coefficient,
        "causal_parallel_l2": np.linalg.norm(parallel, axis=-1),
        "causal_orthogonal_l2": np.linalg.norm(orthogonal, axis=-1),
        "causal_orthogonal_fraction": np.linalg.norm(orthogonal, axis=-1) / np.maximum(total_l2, 1e-12),
    }


def aggregate_mean(values: np.ndarray, labels: np.ndarray, aggregation: str) -> float:
    if aggregation == "dataset":
        return float(values.mean())
    if aggregation == "letter_macro":
        return float(np.mean([values[labels == label].mean() for label in range(4)]))
    raise ValueError(f"Unknown aggregation: {aggregation}")


def bootstrap(
    values: np.ndarray,
    labels: np.ndarray,
    aggregation: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    means = np.empty(samples)
    if aggregation == "dataset":
        for start in range(0, samples, 1000):
            stop = min(samples, start + 1000)
            index = rng.integers(0, len(values), size=(stop - start, len(values)))
            means[start:stop] = values[index].mean(axis=1)
    elif aggregation == "letter_macro":
        groups = [values[labels == label] for label in range(4)]
        for start in range(0, samples, 1000):
            stop = min(samples, start + 1000)
            group_means = []
            for group in groups:
                index = rng.integers(0, len(group), size=(stop - start, len(group)))
                group_means.append(group[index].mean(axis=1))
            means[start:stop] = np.mean(group_means, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    return (
        aggregate_mean(values, labels, aggregation),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )
