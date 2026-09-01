from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy.stats import rankdata, spearmanr

from .data import load_activation_dataset
from .io import atomic_save_npz, shard_path


GROUP_NAMES = ("neither", "game_only", "neutral_only", "both")
VARIANTS = ("raw", "nuisance_adjusted", "answer_orthogonal")


def _unit(value: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(value))
    return (value / norm if norm > 1e-12 else np.zeros_like(value)), norm


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if not n1 or not n0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _stratified_folds(strata: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(k)]
    for value in np.unique(strata):
        ids = np.flatnonzero(strata == value)
        rng.shuffle(ids)
        for offset, index in enumerate(ids):
            buckets[offset % k].append(int(index))
    return [np.asarray(sorted(bucket), dtype=int) for bucket in buckets]


def _letter_balanced_mean(values: np.ndarray, mask: np.ndarray, letters: np.ndarray) -> np.ndarray:
    means = []
    for letter in range(4):
        selected = mask & (letters == letter)
        if not np.any(selected):
            raise ValueError(f"No observations for original-winner letter {'ABCD'[letter]}")
        means.append(values[selected].mean(axis=0))
    return np.mean(means, axis=0)


def _nuisance_design(
    baseline_logits: np.ndarray,
    original: np.ndarray,
    baseline_correct: np.ndarray,
) -> np.ndarray:
    centered = baseline_logits - baseline_logits.mean(axis=-1, keepdims=True)
    order = np.argsort(-baseline_logits, axis=-1)
    margin = baseline_logits[np.arange(len(order)), order[:, 0]] - baseline_logits[
        np.arange(len(order)), order[:, 1]
    ]
    spread = centered.std(axis=-1)
    probabilities = softmax(baseline_logits, axis=-1)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30)), axis=-1)

    columns = [np.ones(len(original))]
    for values in (margin, spread, entropy):
        scale = values.std()
        columns.append((values - values.mean()) / (scale if scale > 1e-12 else 1.0))
    columns.append(baseline_correct.astype(float))
    columns.extend((original == letter).astype(float) for letter in range(3))
    return np.column_stack(columns)


def _remove_nuisance(values: np.ndarray, design: np.ndarray, fit: np.ndarray) -> np.ndarray:
    coefficients = np.linalg.pinv(design[fit]) @ values[fit]
    return values - design @ coefficients


def _answer_basis(baseline: np.ndarray, original: np.ndarray, fit: np.ndarray) -> np.ndarray:
    centroids = np.stack([baseline[fit & (original == letter)].mean(axis=0) for letter in range(4)])
    centroids -= centroids.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centroids, full_matrices=False)
    rank = int(np.sum(singular > singular.max() * 1e-6)) if singular.size else 0
    return vt[:rank]


def _fit_directions(
    delta: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    original: np.ndarray,
    design: np.ndarray,
    fit: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    game_only = fit & (groups == 1)
    neither = fit & (groups == 0)
    raw, raw_gap = _unit(
        _letter_balanced_mean(delta, game_only, original)
        - _letter_balanced_mean(delta, neither, original)
    )
    adjusted_values = _remove_nuisance(delta, design, fit & np.isin(groups, (0, 1)))
    adjusted, adjusted_gap = _unit(
        _letter_balanced_mean(adjusted_values, game_only, original)
        - _letter_balanced_mean(adjusted_values, neither, original)
    )
    basis = _answer_basis(baseline, original, fit)
    orthogonal_value = adjusted.copy()
    if len(basis):
        orthogonal_value -= basis.T @ (basis @ orthogonal_value)
    orthogonal, orthogonal_gap = _unit(orthogonal_value)
    return (
        {"raw": raw, "nuisance_adjusted": adjusted, "answer_orthogonal": orthogonal},
        {"raw": raw_gap, "nuisance_adjusted": adjusted_gap, "answer_orthogonal": orthogonal_gap},
    )


def _compression(game: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    game = game - game.mean(axis=-1, keepdims=True)
    neutral = neutral - neutral.mean(axis=-1, keepdims=True)
    delta = game - neutral
    denominator = np.sum(neutral * neutral, axis=-1)
    return np.divide(
        -np.sum(delta * neutral, axis=-1),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 1e-10,
    )


def _load_all_residuals(
    input_dir: str | Path, conditions: tuple[str, ...], question_ids: list[str]
) -> np.ndarray:
    """Read each compressed shard once; repeated layerwise NPZ reads are much slower."""
    with np.load(shard_path(input_dir, conditions[0], question_ids[0]), allow_pickle=False) as data:
        shape = data["residuals"].shape
        dtype = data["residuals"].dtype
    values = np.empty((len(conditions), len(question_ids), *shape), dtype=dtype)
    for condition_index, condition in enumerate(conditions):
        for question_index, qid in enumerate(question_ids):
            with np.load(shard_path(input_dir, condition, qid), allow_pickle=False) as data:
                values[condition_index, question_index] = data["residuals"]
        print(f"loaded residuals: {condition}", flush=True)
    return values


def analyze(input_dir: str | Path, output_dir: str | Path, k: int, seed: int) -> dict:
    data = load_activation_dataset(input_dir, ["baseline", "incorrect", "neutral"])
    baseline_logits = data.condition("baseline")
    game_logits = data.condition("incorrect")
    neutral_logits = data.condition("neutral")
    original = np.argmax(baseline_logits[:, -1], axis=-1)
    game_choice = np.argmax(game_logits[:, -1], axis=-1)
    neutral_choice = np.argmax(neutral_logits[:, -1], axis=-1)
    game_switch = game_choice != original
    neutral_switch = neutral_choice != original
    groups = game_switch.astype(int) + 2 * neutral_switch.astype(int)
    baseline_correct = np.asarray(
        [bool(data.metadata[(qid, "baseline")]["baseline_correct"]) for qid in data.question_ids]
    )
    design = _nuisance_design(baseline_logits[:, -1], original, baseline_correct)
    folds = _stratified_folds(groups * 4 + original, k, seed)
    fold_of = np.empty(len(groups), dtype=int)
    for fold_index, fold in enumerate(folds):
        fold_of[fold] = fold_index

    compression = _compression(game_logits, neutral_logits)
    late_start = max(0, compression.shape[1] - 12)
    late_compression = np.nanmean(compression[:, late_start:], axis=1)
    primary = np.isin(groups, (0, 1))
    n_layers = baseline_logits.shape[1]
    residuals = _load_all_residuals(
        input_dir, ("baseline", "incorrect", "neutral"), data.question_ids
    )
    hidden_size = residuals.shape[-1]
    fold_directions = np.zeros((len(VARIANTS), k, n_layers, hidden_size), dtype=np.float32)
    production = np.zeros((len(VARIANTS), n_layers, hidden_size), dtype=np.float32)
    mean_gaps = np.zeros((len(VARIANTS), n_layers), dtype=np.float32)
    general_direction = np.zeros((n_layers, hidden_size), dtype=np.float32)
    game_switch_mean = np.zeros_like(general_direction)
    scores = np.full((len(VARIANTS), len(groups), n_layers), np.nan, dtype=np.float32)
    rows: list[dict] = []

    for layer in range(n_layers):
        baseline = residuals[0, :, layer].astype(np.float64)
        game = residuals[1, :, layer].astype(np.float64)
        neutral = residuals[2, :, layer].astype(np.float64)
        delta = game - neutral
        general_direction[layer], _ = _unit(delta.mean(axis=0))
        game_switch_mean[layer], _ = _unit(
            _letter_balanced_mean(delta, game_switch, original)
        )
        all_fit = np.ones(len(groups), dtype=bool)
        directions, gaps = _fit_directions(delta, baseline, groups, original, design, all_fit)
        for variant_index, variant in enumerate(VARIANTS):
            production[variant_index, layer] = directions[variant]
            mean_gaps[variant_index, layer] = gaps[variant]

        for fold_index, test in enumerate(folds):
            fit = np.ones(len(groups), dtype=bool)
            fit[test] = False
            directions, _ = _fit_directions(delta, baseline, groups, original, design, fit)
            for variant_index, variant in enumerate(VARIANTS):
                direction = directions[variant]
                fold_directions[variant_index, fold_index, layer] = direction
                scores[variant_index, test, layer] = delta[test] @ direction

        for variant_index, variant in enumerate(VARIANTS):
            directions = fold_directions[variant_index, :, layer].astype(np.float64)
            cosines = [
                float(directions[a] @ directions[b])
                for a in range(k)
                for b in range(a + 1, k)
            ]
            selected_scores = scores[variant_index, primary, layer]
            labels = groups[primary] == 1
            if np.nanstd(selected_scores) < 1e-12 or np.nanstd(late_compression[primary]) < 1e-12:
                correlation_statistic, correlation_p = float("nan"), float("nan")
            else:
                correlation = spearmanr(selected_scores, late_compression[primary], nan_policy="omit")
                correlation_statistic = float(correlation.statistic)
                correlation_p = float(correlation.pvalue)
            rows.append(
                {
                    "variant": variant,
                    "layer": layer,
                    "heldout_auc_game_only_vs_neither": _auc(labels, selected_scores),
                    "mean_fold_cosine": float(np.mean(cosines)),
                    "spearman_with_late_compression": correlation_statistic,
                    "spearman_p": correlation_p,
                    "game_only_mean_compression": float(np.nanmean(compression[groups == 1, layer])),
                    "neither_mean_compression": float(np.nanmean(compression[groups == 0, layer])),
                    "mean_gap": float(mean_gaps[variant_index, layer]),
                    "cosine_with_general_feedback": float(
                        production[variant_index, layer].astype(np.float64)
                        @ general_direction[layer].astype(np.float64)
                    ),
                    "cosine_with_game_switch_mean": float(
                        production[variant_index, layer].astype(np.float64)
                        @ game_switch_mean[layer].astype(np.float64)
                    ),
                }
            )
        print(f"layer {layer}/{n_layers - 1}", flush=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "switch_direction_layers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    atomic_save_npz(
        output_dir / "switch_direction.npz",
        directions=production,
        direction_variants=np.asarray(VARIANTS),
        mean_gaps=mean_gaps,
        fold_directions=fold_directions,
        cross_fitted_scores=scores,
        general_feedback_directions=general_direction,
        game_switch_mean_directions=game_switch_mean,
        compression=compression.astype(np.float32),
        groups=groups.astype(np.int8),
        question_ids=np.asarray(data.question_ids),
        metadata=np.asarray(json.dumps({"input_dir": str(input_dir), "folds": k, "seed": seed})),
    )
    atomic_save_npz(
        output_dir / "switch_direction_answer_orthogonal.npz",
        directions=production[VARIANTS.index("answer_orthogonal")],
        mean_gap=mean_gaps[VARIANTS.index("answer_orthogonal")],
        metadata=np.asarray(
            json.dumps(
                {
                    "input_dir": str(input_dir),
                    "folds": k,
                    "seed": seed,
                    "direction_sign": "Game-only-switch difference-in-differences",
                    "nuisance_adjusted": True,
                    "answer_subspace_removed": True,
                },
                sort_keys=True,
            )
        ),
    )

    primary_rows = [row for row in rows if row["variant"] == "answer_orthogonal"]
    summary = {
        "n_questions": len(groups),
        "groups": {name: int(np.sum(groups == index)) for index, name in enumerate(GROUP_NAMES)},
        "definition": (
            "letter-balanced mean(Game-Neutral | Game-only switch) minus "
            "mean(Game-Neutral | neither switches), with baseline nuisance adjustment "
            "and answer-subspace removal"
        ),
        "late_compression_layers": [late_start, n_layers - 1],
        "peak_heldout_auc": max(primary_rows, key=lambda row: row["heldout_auc_game_only_vs_neither"]),
        "peak_fold_stability": max(primary_rows, key=lambda row: row["mean_fold_cosine"]),
        "peak_late_compression_correlation": max(
            (row for row in primary_rows if np.isfinite(row["spearman_with_late_compression"])),
            key=lambda row: abs(row["spearman_with_late_compression"]),
        ),
    }
    (output_dir / "switch_direction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover a switch-specific Game-minus-neutral direction")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, args.folds, args.seed), indent=2))


if __name__ == "__main__":
    main()
