from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


LETTERS = "ABCD"


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _candidate_advantage(logits: np.ndarray, indices: np.ndarray) -> np.ndarray:
    rows = np.arange(len(indices))
    target = logits[rows, indices]
    return target - (logits.sum(-1) - target) / 3.0


def _align(values: np.ndarray, qids: list[str], mappings: dict[str, dict[str, Any]]) -> np.ndarray:
    output = np.empty_like(values)
    for qi, qid in enumerate(qids):
        for original_index, original in enumerate(LETTERS):
            new_letter = mappings[qid]["original_to_new"][original]
            output[..., qi, original_index] = values[..., qi, LETTERS.index(new_letter)]
    return output


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=1, keepdims=True)


def _standardize(values: np.ndarray, discovery: np.ndarray) -> tuple[np.ndarray, float, float]:
    flat = values[discovery].reshape(-1)
    mean = float(flat.mean())
    scale = float(flat.std())
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Degenerate feature scale")
    return (values - mean) / scale, mean, scale


def _truncated_cubic(values: np.ndarray, knots: np.ndarray) -> list[np.ndarray]:
    return [np.maximum(values - knot, 0.0) ** 3 for knot in knots]


def _features(
    score: np.ndarray,
    gap: np.ndarray,
    first_positions: np.ndarray,
    second_positions: np.ndarray,
    discovery: np.ndarray,
    model: str,
    frozen: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if frozen is None:
        score_z, score_mean, score_scale = _standardize(score, discovery)
        gap_z, gap_mean, gap_scale = _standardize(gap, discovery)
        score_knots = np.quantile(score_z[discovery], [0.1, 0.25, 0.5, 0.75, 0.9])
        gap_knots = np.quantile(gap_z[discovery], [0.1, 0.25, 0.5, 0.75, 0.9])
        frozen = {
            "score_mean": score_mean,
            "score_scale": score_scale,
            "gap_mean": gap_mean,
            "gap_scale": gap_scale,
            "score_knots": score_knots.tolist(),
            "gap_knots": gap_knots.tolist(),
        }
    else:
        score_z = (score - frozen["score_mean"]) / frozen["score_scale"]
        gap_z = (gap - frozen["gap_mean"]) / frozen["gap_scale"]
        score_knots = np.asarray(frozen["score_knots"])
        gap_knots = np.asarray(frozen["gap_knots"])

    columns: list[np.ndarray] = [score_z]
    if model in {"score_cubic", "score_gap_spline"}:
        columns.extend([score_z**2, score_z**3])
    if model == "score_gap_spline":
        columns.extend(_truncated_cubic(score_z, np.asarray(score_knots)))
        columns.extend([gap_z, gap_z**2, gap_z**3])
        columns.extend(_truncated_cubic(gap_z, np.asarray(gap_knots)))
    for position in range(3):
        columns.append((first_positions == position).astype(float))
        columns.append((second_positions == position).astype(float))
    design = np.stack([_center(column) for column in columns], axis=-1)
    return design, frozen


def _fit(y: np.ndarray, design: np.ndarray, mask: np.ndarray, include_w1: bool) -> tuple[np.ndarray, np.ndarray]:
    w1 = np.zeros_like(y)
    w1[:, 0] = 1.0
    columns = design
    if include_w1:
        columns = np.concatenate([columns, _center(w1)[..., None]], axis=-1)
    x = columns[mask].reshape(-1, columns.shape[-1])
    target = _center(y)[mask].reshape(-1)
    coefficients = np.linalg.lstsq(x, target, rcond=1e-9)[0]
    prediction = np.einsum("qcf,f->qc", columns, coefficients)
    return coefficients, prediction


def _bootstrap_coefficient(
    y: np.ndarray,
    design: np.ndarray,
    mask: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    rows = np.flatnonzero(mask)
    w1 = np.zeros_like(y)
    w1[:, 0] = 1.0
    columns = np.concatenate([design, _center(w1)[..., None]], axis=-1)

    def estimate(selected: np.ndarray) -> float:
        x = columns[selected].reshape(-1, columns.shape[-1])
        target = _center(y)[selected].reshape(-1)
        return float(np.linalg.lstsq(x, target, rcond=1e-9)[0][-1])

    point = estimate(rows)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        samples[draw] = estimate(rng.choice(rows, size=len(rows), replace=True))
    return {
        "mean": point,
        "ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "n_questions": int(len(rows)),
    }


def _bootstrap_mean(values: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(1)
    return {
        "mean": float(values.mean()),
        "ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "n": int(len(values)),
    }


def analyze(args: argparse.Namespace) -> None:
    arrays = _load(args.results)
    qids = arrays["question_ids"].astype(str).tolist()
    if len(qids) != 500 or not arrays["completed"].all():
        raise RuntimeError("Expected one complete 500-question causal run")
    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    confirmation = ~discovery

    rank_contents = arrays["rank_contents"].astype(str)
    rank_indices = np.asarray(
        [[LETTERS.index(value) for value in row] for row in rank_contents], dtype=int
    )
    baseline = arrays["baseline_logits"].astype(float)
    score = np.stack(
        [baseline[np.arange(len(qids)), rank_indices[:, rank]] for rank in range(4)],
        axis=1,
    )
    score = _center(score)
    gap = np.empty_like(score)
    for rank in range(4):
        others = [value for value in range(4) if value != rank]
        gap[:, rank] = score[:, rank] - score[:, others].max(axis=1)
    first_positions = rank_indices.copy()
    second_positions = np.empty_like(first_positions)
    for qi, qid in enumerate(qids):
        for rank in range(4):
            original = LETTERS[first_positions[qi, rank]]
            second_positions[qi, rank] = LETTERS.index(mappings[qid]["original_to_new"][original])

    matched = _align(arrays["matched_logits"].astype(float), qids, mappings)
    control = _align(arrays["control_logits"].astype(float), qids, mappings)
    effect = np.empty((2, len(qids), 4), dtype=float)
    for condition in range(2):
        for rank in range(4):
            indices = rank_indices[:, rank]
            effect[condition, :, rank] = (
                _candidate_advantage(matched[condition, rank], indices)
                - _candidate_advantage(control[condition, rank], indices)
            )
    interaction = effect[0] - effect[1]

    output: dict[str, Any] = {
        "definition": {
            "outcome": "Game minus Neutral matching-specific candidate-advantage lesion effect",
            "gap": "candidate 1P score minus the maximum 1P score among the other three candidates",
            "W1_term": "question-centered W1 indicator after simultaneous flexible score, gap, and display-position controls",
        },
        "validation": {
            "questions": len(qids),
            "discovery": int(discovery.sum()),
            "confirmation": int(confirmation.sum()),
        },
        "models": {},
        "near_ties": {},
    }
    frozen: dict[str, Any] | None = None
    for model_index, model in enumerate(("linear_score", "score_cubic", "score_gap_spline")):
        design, current_frozen = _features(
            score,
            gap,
            first_positions,
            second_positions,
            discovery,
            model,
            frozen if model == "score_gap_spline" else None,
        )
        if model == "score_gap_spline":
            frozen = current_frozen
        no_w1_coef, no_w1_prediction = _fit(interaction, design, discovery, False)
        with_w1_coef, with_w1_prediction = _fit(interaction, design, discovery, True)
        record: dict[str, Any] = {
            "features": int(design.shape[-1]),
            "discovery_W1_term": _bootstrap_coefficient(
                interaction, design, discovery, args.seed + model_index * 10000, args.draws
            ),
            "confirmation_W1_term": _bootstrap_coefficient(
                interaction, design, confirmation, args.seed + model_index * 10000 + 1, args.draws
            ),
            "confirmation_mse_without_W1": float(
                np.mean((_center(interaction)[confirmation] - no_w1_prediction[confirmation]) ** 2)
            ),
            "confirmation_mse_with_W1": float(
                np.mean((_center(interaction)[confirmation] - with_w1_prediction[confirmation]) ** 2)
            ),
        }
        record["confirmation_mse_improvement_fraction"] = float(
            1.0 - record["confirmation_mse_with_W1"] / record["confirmation_mse_without_W1"]
        )
        output["models"][model] = record

    flexible_design, frozen = _features(
        score,
        gap,
        first_positions,
        second_positions,
        discovery,
        "score_gap_spline",
        frozen,
    )
    _, flexible_prediction = _fit(interaction, flexible_design, discovery, False)
    residual = _center(interaction) - flexible_prediction
    top_gap = score[:, 0] - score[:, 1]
    for threshold_index, threshold in enumerate((0.25, 0.5, 1.0)):
        mask = confirmation & (top_gap <= threshold)
        if mask.sum() < 20:
            output["near_ties"][str(threshold)] = {"n": int(mask.sum()), "status": "insufficient"}
            continue
        contrast = residual[mask, 0] - residual[mask, 1]
        output["near_ties"][str(threshold)] = _bootstrap_mean(
            contrast, args.seed + 50000 + threshold_index, args.draws
        )
    output["frozen_feature_parameters"] = frozen

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "categorical_winner_nonlinearity.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--draws", type=int, default=2000)
    return parser


if __name__ == "__main__":
    analyze(build_parser().parse_args())
