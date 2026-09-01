from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LETTERS = "ABCD"
OPTION_ANCHORS = ("content_end", "line_end")
LAMBDAS = np.asarray([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])


def _load(root: Path):
    metadata = json.loads((root / "metadata.json").read_text())
    residuals = np.load(root / "position_residuals.npy", mmap_mode="r")
    return residuals, metadata


def _labels(path: Path, qids: list[str], field: str) -> np.ndarray:
    rows = json.loads(path.read_text())["results"]
    labels = []
    for qid in qids:
        value = rows[qid][field]
        if value not in LETTERS:
            raise ValueError(f"Non-A-D label for {qid}: {value!r}")
        labels.append(LETTERS.index(value))
    return np.asarray(labels, dtype=np.int64)


def _center_normalize(candidates: np.ndarray) -> np.ndarray:
    values = candidates.astype(np.float32)
    values -= values.mean(axis=-2, keepdims=True)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def _aligned_remapped_indices(metadata: dict, qids: list[str]) -> np.ndarray:
    values = np.empty((len(qids), 4), dtype=np.int64)
    for qi, qid in enumerate(qids):
        mapping = metadata["mappings"][qid]["original_to_new"]
        values[qi] = [LETTERS.index(mapping[letter]) for letter in LETTERS]
    return values


def _candidate_array(
    residuals: np.ndarray, metadata: dict, anchor: str
) -> np.ndarray:
    indices = [metadata["anchors"].index(f"{anchor}_{letter}") for letter in LETTERS]
    return np.asarray(residuals[:, :, indices]).copy()


def _retrieval_curve(
    original: np.ndarray,
    remapped: np.ndarray,
    remapped_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Align the remapped candidate axis to original semantic content.
    aligned = np.take_along_axis(
        remapped,
        remapped_indices[:, None, :, None],
        axis=2,
    )
    original = _center_normalize(original)
    aligned = _center_normalize(aligned)
    forward = np.einsum("qlod,qlpd->qlop", original, aligned)
    reverse = np.einsum("qlod,qlpd->qlop", aligned, original)
    target = np.arange(4)[None, None, :]
    forward_correct = forward.argmax(axis=-1) == target
    reverse_correct = reverse.argmax(axis=-1) == target
    per_question = 0.5 * (
        forward_correct.mean(axis=-1) + reverse_correct.mean(axis=-1)
    )
    return per_question.mean(axis=0), per_question


def _balanced_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean([
        np.mean(prediction[target == label] == label)
        for label in range(4) if np.any(target == label)
    ]))


def _folds(labels: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    values: list[list[int]] = [[] for _ in range(count)]
    for label in range(4):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            values[offset % count].append(int(index))
    return [np.asarray(sorted(value), dtype=np.int64) for value in values]


def _normalize_queries(values: np.ndarray, mean: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float32) - mean.astype(np.float32)
    return centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)


def _cv_scores(
    queries: np.ndarray,
    candidates: np.ndarray,
    labels: np.ndarray,
    folds: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    targets = candidates[np.arange(len(labels)), labels].astype(np.float32)
    predictions = np.empty((len(LAMBDAS), len(labels), 4), dtype=np.float32)
    all_indices = np.arange(len(labels))
    for validation in folds:
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        mean = queries[training].astype(np.float32).mean(axis=0)
        x_train = _normalize_queries(queries[training], mean)
        x_validation = _normalize_queries(queries[validation], mean)
        gram = x_train @ x_train.T
        cross = x_validation @ x_train.T
        semantic = targets[training] @ candidates[validation].reshape(-1, candidates.shape[-1]).T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        projected = eigenvectors.T @ semantic
        for li, penalty in enumerate(LAMBDAS):
            weights = eigenvectors @ (projected / (eigenvalues[:, None] + penalty))
            full = (cross @ weights).reshape(len(validation), len(validation), 4)
            local = np.arange(len(validation))
            predictions[li, validation] = full[local, local]
    balanced = np.asarray([
        _balanced_accuracy(row.argmax(axis=1), labels) for row in predictions
    ])
    return predictions, balanced


def _fit_scores(
    train_queries: np.ndarray,
    train_candidates: np.ndarray,
    train_labels: np.ndarray,
    test_queries: np.ndarray,
    test_candidates: np.ndarray,
    penalty: float,
) -> np.ndarray:
    targets = train_candidates[np.arange(len(train_labels)), train_labels].astype(np.float32)
    mean = train_queries.astype(np.float32).mean(axis=0)
    x_train = _normalize_queries(train_queries, mean)
    x_test = _normalize_queries(test_queries, mean)
    gram = x_train @ x_train.T
    cross = x_test @ x_train.T
    semantic = targets @ test_candidates.reshape(-1, test_candidates.shape[-1]).T
    weights = np.linalg.solve(
        gram + float(penalty) * np.eye(len(gram), dtype=np.float32), semantic
    )
    full = (cross @ weights).reshape(len(test_queries), len(test_queries), 4)
    local = np.arange(len(test_queries))
    return full[local, local]


def _bootstrap_accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    seed: int,
    draws: int = 5_000,
) -> tuple[float, list[float]]:
    correct = scores.argmax(axis=1) == labels
    groups = [np.flatnonzero(labels == label) for label in range(4) if np.any(labels == label)]
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        values[draw] = np.mean([
            correct[rng.choice(group, len(group), replace=True)].mean()
            for group in groups
        ])
    return _balanced_accuracy(scores.argmax(axis=1), labels), np.quantile(values, (0.025, 0.975)).tolist()


def _bootstrap_retrieval(
    values: np.ndarray, layer: int, seed: int, draws: int = 5_000
) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.choice(n, n, replace=True)
        samples[draw] = values[indices, layer].mean()
    return np.quantile(samples, (0.025, 0.975)).tolist()


def analyze(
    discovery_original_root: Path,
    discovery_remapped_root: Path,
    confirmation_original_root: Path,
    confirmation_remapped_root: Path,
    baseline_results: Path,
    remapped_baseline_results: Path,
    output: Path,
    seed: int,
) -> dict:
    do, do_meta = _load(discovery_original_root)
    dr, dr_meta = _load(discovery_remapped_root)
    co, co_meta = _load(confirmation_original_root)
    cr, cr_meta = _load(confirmation_remapped_root)
    discovery_qids = list(do_meta["question_ids"])
    confirmation_qids = list(co_meta["question_ids"])
    if discovery_qids != list(dr_meta["question_ids"]):
        raise ValueError("Discovery original/remapped question orders differ")
    if confirmation_qids != list(cr_meta["question_ids"]):
        raise ValueError("Confirmation original/remapped question orders differ")

    discovery_retrieval: dict[str, np.ndarray] = {}
    discovery_per_question: dict[str, np.ndarray] = {}
    confirmation_retrieval: dict[str, np.ndarray] = {}
    confirmation_per_question: dict[str, np.ndarray] = {}
    discovery_candidates: dict[str, np.ndarray] = {}
    confirmation_candidates: dict[str, np.ndarray] = {}
    remapped_confirmation_candidates: dict[str, np.ndarray] = {}
    for anchor in OPTION_ANCHORS:
        d_original = _candidate_array(do, do_meta, anchor)
        d_remapped = _candidate_array(dr, dr_meta, anchor)
        c_original = _candidate_array(co, co_meta, anchor)
        c_remapped = _candidate_array(cr, cr_meta, anchor)
        discovery_candidates[anchor] = d_original
        confirmation_candidates[anchor] = c_original
        remapped_confirmation_candidates[anchor] = c_remapped
        discovery_retrieval[anchor], discovery_per_question[anchor] = _retrieval_curve(
            d_original,
            d_remapped,
            _aligned_remapped_indices(dr_meta, discovery_qids),
        )
        confirmation_retrieval[anchor], confirmation_per_question[anchor] = _retrieval_curve(
            c_original,
            c_remapped,
            _aligned_remapped_indices(cr_meta, confirmation_qids),
        )

    best_anchor, best_option_layer = max(
        (
            (anchor, layer)
            for anchor in OPTION_ANCHORS
            for layer in range(64)
        ),
        key=lambda item: float(discovery_retrieval[item[0]][item[1]]),
    )
    option_confirmation_ci = _bootstrap_retrieval(
        confirmation_per_question[best_anchor], best_option_layer, seed + 1
    )

    d_candidates = _center_normalize(
        discovery_candidates[best_anchor][:, best_option_layer]
    )
    c_candidates = _center_normalize(
        confirmation_candidates[best_anchor][:, best_option_layer]
    )
    rc_candidates = _center_normalize(
        remapped_confirmation_candidates[best_anchor][:, best_option_layer]
    )
    decision_index = do_meta["anchors"].index("first_answer_decision")
    d_queries = np.asarray(do[:, :, decision_index]).copy()
    c_queries = np.asarray(co[:, :, decision_index]).copy()
    rc_queries = np.asarray(cr[:, :, decision_index]).copy()
    d_labels = _labels(baseline_results, discovery_qids, "answer")
    c_labels = _labels(baseline_results, confirmation_qids, "answer")
    rc_labels = _labels(remapped_baseline_results, confirmation_qids, "answer_new_letter")
    folds = _folds(d_labels, 5, seed)

    cv_balanced = np.empty(64, dtype=np.float32)
    penalties = np.empty(64, dtype=np.float32)
    cv_predictions: list[np.ndarray] = []
    for layer in range(64):
        predictions, balanced = _cv_scores(
            d_queries[:, layer], d_candidates, d_labels, folds
        )
        best = int(np.argmax(balanced))
        cv_balanced[layer] = balanced[best]
        penalties[layer] = LAMBDAS[best]
        cv_predictions.append(predictions[best])
        print(
            f"contextual decision matcher layer {layer + 1}/64 "
            f"CV={cv_balanced[layer]:.3f}",
            flush=True,
        )
    decision_layer = int(np.argmax(cv_balanced))
    penalty = float(penalties[decision_layer])
    original_scores = _fit_scores(
        d_queries[:, decision_layer], d_candidates, d_labels,
        c_queries[:, decision_layer], c_candidates, penalty,
    )
    remapped_scores = _fit_scores(
        d_queries[:, decision_layer], d_candidates, d_labels,
        rc_queries[:, decision_layer], rc_candidates, penalty,
    )
    original_accuracy, original_ci = _bootstrap_accuracy(
        original_scores, c_labels, seed + 2
    )
    remapped_accuracy, remapped_ci = _bootstrap_accuracy(
        remapped_scores, rc_labels, seed + 3
    )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "retrieval_trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "anchor", "layer", "discovery_accuracy", "confirmation_accuracy"
        ])
        writer.writeheader()
        for anchor in OPTION_ANCHORS:
            for layer in range(64):
                writer.writerow({
                    "anchor": anchor,
                    "layer": layer + 1,
                    "discovery_accuracy": float(discovery_retrieval[anchor][layer]),
                    "confirmation_accuracy": float(confirmation_retrieval[anchor][layer]),
                })
    summary = {
        "definitions": {
            "content_end": "Contextual residual at the last token overlapping the option text.",
            "line_end": "Contextual residual at the token overlapping the option-closing newline.",
            "retrieval": (
                "Symmetric same-content top-1 retrieval across original and deranged mappings "
                "after within-question centering of the four option residuals."
            ),
        },
        "split": {"discovery": len(discovery_qids), "confirmation": len(confirmation_qids)},
        "selected_option_representation": {
            "anchor": best_anchor,
            "layer": best_option_layer + 1,
            "discovery_retrieval_accuracy": float(discovery_retrieval[best_anchor][best_option_layer]),
            "confirmation_retrieval_accuracy": float(confirmation_retrieval[best_anchor][best_option_layer]),
            "confirmation_ci": option_confirmation_ci,
        },
        "selected_decision_matcher": {
            "layer": decision_layer + 1,
            "lambda": penalty,
            "discovery_cv_balanced_accuracy": float(cv_balanced[decision_layer]),
            "confirmation_original_balanced_accuracy": original_accuracy,
            "confirmation_original_ci": original_ci,
            "confirmation_remapped_balanced_accuracy": remapped_accuracy,
            "confirmation_remapped_ci": remapped_ci,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        output / "decision_matcher_scores.npz",
        discovery_cv_balanced_accuracy=cv_balanced,
        selected_lambda=penalties,
        confirmation_original_scores=original_scores,
        confirmation_original_labels=c_labels,
        confirmation_remapped_scores=remapped_scores,
        confirmation_remapped_labels=rc_labels,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-original", type=Path, required=True)
    parser.add_argument("--discovery-remapped", type=Path, required=True)
    parser.add_argument("--confirmation-original", type=Path, required=True)
    parser.add_argument("--confirmation-remapped", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--remapped-baseline-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    analyze(
        args.discovery_original,
        args.discovery_remapped,
        args.confirmation_original,
        args.confirmation_remapped,
        args.baseline_results,
        args.remapped_baseline_results,
        args.output,
        args.seed,
    )


if __name__ == "__main__":
    main()
