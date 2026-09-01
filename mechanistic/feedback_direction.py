from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import load_activation_dataset
from .io import atomic_save_npz, shard_path
from .probes import stratified_folds


def _residuals(path: str | Path, condition: str, question_id: str) -> np.ndarray:
    with np.load(shard_path(path, condition, question_id), allow_pickle=False) as data:
        if "residuals" not in data.files:
            raise KeyError(f"Residuals were not saved for {condition}/{question_id}")
        return data["residuals"].astype(np.float32)


def _unit_rows(values: np.ndarray, allow_zero: bool = False) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=-1)
    zero = norms < 1e-12
    if np.any(zero) and not allow_zero:
        bad = np.flatnonzero(norms < 1e-12).tolist()
        raise ValueError(f"Zero-length direction at readouts {bad}")
    safe = np.where(zero, 1.0, norms)
    return values / safe[:, None], norms


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    # Mann-Whitney AUC with average ranks for ties.
    from scipy.stats import rankdata

    labels = np.asarray(labels, dtype=bool)
    positive = int(labels.sum())
    negative = len(labels) - positive
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _balanced_intervention_ids(
    qids: list[str],
    labels: np.ndarray,
    test: np.ndarray,
    per_letter: int,
    seed: int,
) -> list[str]:
    rng = np.random.default_rng(seed)
    selected = []
    for letter in range(4):
        ids = test[labels[test] == letter].copy()
        rng.shuffle(ids)
        if len(ids) < per_letter:
            raise ValueError(
                f"Held-out fold has only {len(ids)} questions with baseline-winner letter "
                f"{'ABCD'[letter]}; requested {per_letter}"
            )
        selected.extend(int(i) for i in ids[:per_letter])
    rng.shuffle(selected)
    return [qids[i] for i in selected]


def discover(
    input_dir: str | Path,
    output_path: str | Path,
    k_folds: int,
    test_fold: int,
    intervention_per_letter: int,
    seed: int,
) -> dict:
    dataset = load_activation_dataset(input_dir, ["baseline", "incorrect", "neutral"])
    qids = dataset.question_ids
    labels = np.argmax(dataset.logits[:, 0, -1], axis=-1)
    folds = stratified_folds(labels, k_folds, seed)
    if not 0 <= test_fold < len(folds):
        raise ValueError(f"test_fold must be between 0 and {len(folds) - 1}")
    fold_of = np.empty(len(qids), dtype=int)
    for fold_index, ids in enumerate(folds):
        fold_of[ids] = fold_index

    first = _residuals(input_dir, "incorrect", qids[0])
    n_readouts, hidden_size = first.shape
    game_sums = np.zeros((k_folds, n_readouts, hidden_size), dtype=np.float64)
    neutral_sums = np.zeros_like(game_sums)
    counts = np.zeros(k_folds, dtype=int)
    rng = np.random.default_rng(seed + 991)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(qids))
    signed_train_sum = np.zeros((n_readouts, hidden_size), dtype=np.float64)

    for question_index, qid in enumerate(qids):
        game = _residuals(input_dir, "incorrect", qid)
        neutral = _residuals(input_dir, "neutral", qid)
        if game.shape != first.shape or neutral.shape != first.shape:
            raise ValueError(f"Residual shape mismatch for {qid}")
        fold_index = fold_of[question_index]
        game_sums[fold_index] += game
        neutral_sums[fold_index] += neutral
        counts[fold_index] += 1
        if fold_index != test_fold:
            signed_train_sum += signs[question_index] * (game - neutral)

    total_game = game_sums.sum(axis=0)
    total_neutral = neutral_sums.sum(axis=0)
    total_count = int(counts.sum())
    fold_directions = np.empty((k_folds, n_readouts, hidden_size), dtype=np.float32)
    fold_thresholds = np.empty((k_folds, n_readouts), dtype=np.float64)
    fold_gaps = np.empty((k_folds, n_readouts), dtype=np.float64)
    for fold_index in range(k_folds):
        train_count = total_count - counts[fold_index]
        game_mean = (total_game - game_sums[fold_index]) / train_count
        neutral_mean = (total_neutral - neutral_sums[fold_index]) / train_count
        direction, gap = _unit_rows(game_mean - neutral_mean, allow_zero=True)
        fold_directions[fold_index] = direction.astype(np.float32)
        fold_gaps[fold_index] = gap
        fold_thresholds[fold_index] = 0.5 * np.sum((game_mean + neutral_mean) * direction, axis=-1)

    production_direction = fold_directions[test_fold].astype(np.float64)
    production_gap = fold_gaps[test_fold]
    train_count = total_count - counts[test_fold]
    game_train_mean = (total_game - game_sums[test_fold]) / train_count
    neutral_train_mean = (total_neutral - neutral_sums[test_fold]) / train_count
    game_projection_mean = np.sum(game_train_mean * production_direction, axis=-1)
    neutral_projection_mean = np.sum(neutral_train_mean * production_direction, axis=-1)

    control = signed_train_sum / train_count
    control -= np.sum(control * production_direction, axis=-1, keepdims=True) * production_direction
    control, _ = _unit_rows(control, allow_zero=True)

    scores = np.empty((len(qids), 2, n_readouts), dtype=np.float64)
    for question_index, qid in enumerate(qids):
        direction = fold_directions[fold_of[question_index]].astype(np.float64)
        threshold = fold_thresholds[fold_of[question_index]]
        game = _residuals(input_dir, "incorrect", qid)
        neutral = _residuals(input_dir, "neutral", qid)
        scores[question_index, 0] = np.sum(game * direction, axis=-1) - threshold
        scores[question_index, 1] = np.sum(neutral * direction, axis=-1) - threshold

    accuracy = np.mean(
        np.concatenate([scores[:, 0] > 0, scores[:, 1] <= 0], axis=0), axis=0
    )
    auc = np.empty(n_readouts)
    auc_labels = np.r_[np.ones(len(qids), dtype=int), np.zeros(len(qids), dtype=int)]
    for layer in range(n_readouts):
        auc[layer] = _auc(auc_labels, np.r_[scores[:, 0, layer], scores[:, 1, layer]])

    cosines = []
    for first_fold in range(k_folds):
        for second_fold in range(first_fold + 1, k_folds):
            cosines.append(np.sum(
                fold_directions[first_fold].astype(np.float64)
                * fold_directions[second_fold].astype(np.float64),
                axis=-1,
            ))
    mean_fold_cosine = np.mean(cosines, axis=0)
    test_ids = [qids[int(i)] for i in folds[test_fold]]
    intervention_ids = _balanced_intervention_ids(
        qids, labels, folds[test_fold], intervention_per_letter, seed + 31
    )
    question_index = {qid: index for index, qid in enumerate(qids)}
    intervention_indices = np.asarray([question_index[qid] for qid in intervention_ids], dtype=int)

    metadata = {
        "input_dir": str(input_dir),
        "n_questions": len(qids),
        "n_readouts": n_readouts,
        "hidden_size": hidden_size,
        "k_folds": k_folds,
        "test_fold": test_fold,
        "seed": seed,
        "direction_sign": "incorrect_minus_neutral",
        "scale": "training mean projection gap",
        "probe": "paired mean-difference linear classifier with midpoint threshold",
        "control": "randomly signed paired differences, orthogonalized to feedback direction",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(
        output_path,
        directions=production_direction.astype(np.float32),
        control_directions=control.astype(np.float32),
        mean_gap=production_gap.astype(np.float32),
        game_projection_mean=game_projection_mean.astype(np.float32),
        neutral_projection_mean=neutral_projection_mean.astype(np.float32),
        fold_directions=fold_directions,
        probe_accuracy=accuracy.astype(np.float32),
        probe_auc=auc.astype(np.float32),
        mean_fold_cosine=mean_fold_cosine.astype(np.float32),
        heldout_question_ids=np.asarray(test_ids),
        intervention_question_ids=np.asarray(intervention_ids),
        reference_question_ids=np.asarray(intervention_ids),
        reference_canonical_logits=dataset.logits[intervention_indices, :, -1].astype(np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    rows = [
        {
            "readout": layer,
            "probe_accuracy": float(accuracy[layer]),
            "probe_auc": float(auc[layer]),
            "mean_fold_cosine": float(mean_fold_cosine[layer]),
            "mean_gap": float(production_gap[layer]),
            "game_projection_mean": float(game_projection_mean[layer]),
            "neutral_projection_mean": float(neutral_projection_mean[layer]),
        }
        for layer in range(n_readouts)
    ]
    with output_path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        **metadata,
        "output_path": str(output_path),
        "heldout_questions": len(test_ids),
        "intervention_questions": len(intervention_ids),
        "selected_readouts": {
            str(layer): rows[layer] for layer in (0, 8, 16, 24, 30, 36, 48, 56, 64)
        },
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover a held-out Game-versus-neutral residual direction")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--intervention-per-letter", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(discover(
        args.input,
        args.output,
        args.folds,
        args.test_fold,
        args.intervention_per_letter,
        args.seed,
    ), indent=2))


if __name__ == "__main__":
    main()
