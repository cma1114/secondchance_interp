from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .data import load_activation_dataset, load_residual_layer


def stratified_folds(labels: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed); buckets = [[] for _ in range(k)]
    for label in range(4):
        ids = np.flatnonzero(labels == label); rng.shuffle(ids)
        for i, qid in enumerate(ids): buckets[i % k].append(int(qid))
    return [np.asarray(sorted(x), dtype=int) for x in buckets]


def centroid_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray) -> np.ndarray:
    mean = xtr.mean(axis=0); scale = xtr.std(axis=0); scale[scale < 1e-6] = 1
    a = (xtr - mean) / scale; b = (xte - mean) / scale
    centers = np.stack([a[ytr == label].mean(axis=0) for label in range(4)])
    return np.argmax(b @ centers.T, axis=1)


def logistic_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, c: float) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(); a = scaler.fit_transform(xtr); b = scaler.transform(xte)
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, solver="lbfgs")
    return clf.fit(a, ytr).predict(b)


def run(input_dir: str, output_dir: str, probe: str, step: int, k: int, seed: int, c: float) -> None:
    data = load_activation_dataset(input_dir, ["baseline"])
    final = data.logits[:, 0, -1]
    order = np.argsort(-final, axis=-1)
    targets = {"winner": order[:, 0], "runner_up": order[:, 1]}
    n_layers = data.logits.shape[2]
    layers = sorted(set(range(0, n_layers, step)) | {n_layers - 1})
    rows = []
    for target_name, labels in targets.items():
        split = stratified_folds(labels, k, seed)
        for layer in layers:
            x = load_residual_layer(input_dir, "baseline", data.question_ids, layer)
            pred = np.empty_like(labels)
            for test in split:
                train = np.setdiff1d(np.arange(len(labels)), test)
                pred[test] = centroid_predict(x[train], labels[train], x[test]) if probe == "centroid" else logistic_predict(x[train], labels[train], x[test], c)
            row = {"probe": probe, "target": target_name, "layer": layer, "accuracy": float(np.mean(pred == labels))}
            for label, letter in enumerate("ABCD"):
                mask = labels == label; row[f"accuracy_{letter}"] = float(np.mean(pred[mask] == label)) if mask.any() else np.nan
            rows.append(row); print(row, flush=True)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / f"{probe}_probe_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Held-out winner and runner-up residual-stream probes")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--probe", choices=("centroid", "logistic"), default="centroid")
    p.add_argument("--layer-step", type=int, default=1); p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--c", type=float, default=.01)
    a = p.parse_args(); run(a.input, a.output, a.probe, a.layer_step, a.folds, a.seed, a.c)


if __name__ == "__main__": main()

