from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CONDITIONS = ("baseline", "game", "neutral")


def _weights(strata: np.ndarray, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.zeros((draws + 1, len(strata)), dtype=np.float64)
    for label in np.unique(strata):
        indices = np.flatnonzero(strata == label)
        result[0, indices] = 0.25 / len(indices)
        counts = rng.multinomial(
            len(indices), np.full(len(indices), 1 / len(indices)), size=draws
        )
        result[1:, indices] = counts * (0.25 / len(indices))
    return result


def _variance(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    norms = np.sum(values * values, axis=1)
    means = weights @ values
    return np.maximum(weights @ norms - np.sum(means * means, axis=1), 0)


def _summary(values: np.ndarray) -> dict:
    return {
        "estimate": float(values[0]),
        "ci95": np.quantile(values[1:], [0.025, 0.975]).astype(float).tolist(),
    }


def analyze(
    results_path: Path,
    output_dir: Path,
    dataset: str,
    split_path: Path | None,
    draws: int,
    seed: int,
) -> dict:
    with np.load(results_path, allow_pickle=False) as data:
        qids = data["question_ids"].astype(str)
        conditions = tuple(data["conditions"].astype(str))
        residuals = data["normalized_residuals"].astype(np.float64)
        logits = data["aggregated_logits"].astype(np.float64)
        rows = data["mean_answer_rows"].astype(np.float64)
        mappings = data["original_to_new"].astype(int)
        labels = data["baseline_answer"].astype(str)
    if conditions != CONDITIONS or not np.isfinite(residuals).all():
        raise ValueError("Unexpected conditions or non-finite residuals")

    centered_rows = rows - rows.mean(axis=0, keepdims=True)
    _, singular, right = np.linalg.svd(centered_rows, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-5))
    if rank != 3:
        raise ValueError(f"Expected rank-three answer contrast space, got {rank}")
    basis = right[:rank].T
    answer_coordinates = residuals @ basis
    total_dimensions = residuals.shape[-1]

    split_masks = {"all": np.ones(len(qids), dtype=bool)}
    if split_path is not None:
        split = json.loads(split_path.read_text())
        for name in ("discovery", "confirmation"):
            wanted = set(split[f"{name}_question_ids"])
            split_masks[name] = np.asarray([qid in wanted for qid in qids])

    result_splits = {}
    for split_name, mask in split_masks.items():
        local_labels = labels[mask]
        weights = _weights(local_labels, draws, seed)
        totals = [_variance(residuals[ci, mask], weights) for ci in range(3)]
        answers = [_variance(answer_coordinates[ci, mask], weights) for ci in range(3)]
        complements = [np.maximum(totals[ci] - answers[ci], 0) for ci in range(3)]

        # A directly inspectable supporting check. Center the four answer scores
        # within question, then compare their across-question variance in both
        # displayed-letter and original-semantic coordinates.
        displayed = logits[:, mask] - logits[:, mask].mean(axis=-1, keepdims=True)
        semantic = displayed.copy()
        local_mappings = mappings[mask]
        for ci in (1, 2):
            semantic[ci] = np.take_along_axis(displayed[ci], local_mappings, axis=1)
        displayed_var = [_variance(displayed[ci], weights) for ci in range(3)]
        semantic_var = [_variance(semantic[ci], weights) for ci in range(3)]

        cells = {}
        for ci, condition in enumerate(CONDITIONS):
            answer_ratio = answers[ci] / answers[0]
            complement_ratio = complements[ci] / complements[0]
            cells[condition] = {
                "normalized_full_stream_variance_ratio": _summary(totals[ci] / totals[0]),
                "answer_contrast_variance_ratio": _summary(answer_ratio),
                "orthogonal_complement_variance_ratio": _summary(complement_ratio),
                "answer_to_complement_relative_variance_ratio": _summary(
                    answer_ratio / complement_ratio
                ),
                "displayed_aggregated_logit_contrast_variance_ratio": _summary(
                    displayed_var[ci] / displayed_var[0]
                ),
                "semantic_aggregated_logit_contrast_variance_ratio": _summary(
                    semantic_var[ci] / semantic_var[0]
                ),
            }
        result_splits[split_name] = {
            "n": int(mask.sum()),
            "conditions": cells,
            "game_vs_neutral": {
                "answer_contrast_variance_ratio": _summary(answers[1] / answers[2]),
                "orthogonal_complement_variance_ratio": _summary(
                    complements[1] / complements[2]
                ),
                "normalized_full_stream_variance_ratio": _summary(totals[1] / totals[2]),
            },
        }

    summary = {
        "dataset": dataset,
        "n_questions": len(qids),
        "representation": "actual final RMSNorm output immediately before lm_head",
        "answer_subspace_rank": rank,
        "residual_width": total_dimensions,
        "centering": "condition-specific equal-Baseline-letter-weighted mean across questions",
        "bootstrap_draws": draws,
        "splits": result_splits,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    print(json.dumps(analyze(
        args.results, args.output, args.dataset, args.split, args.draws, args.seed
    ), indent=2))


if __name__ == "__main__":
    main()
