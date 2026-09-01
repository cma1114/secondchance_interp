from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import load_activation_dataset
from .trajectory_analysis import centered


MODELS = ("null", "compression", "targeted_prior_winner", "threshold_current_leader", "threshold_compression")


def folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed); order = rng.permutation(n)
    return [x for x in np.array_split(order, min(k, n)) if len(x)]


def _design(model: str, x: np.ndarray, prior: np.ndarray, leader: np.ndarray, margin: np.ndarray, tau: float) -> np.ndarray:
    # Inputs have shape questions x options. Center every option-level predictor
    # within question, because a common logit shift is not identifiable.
    def c(v): return v - v.mean(axis=1, keepdims=True)
    one_leader = np.eye(4)[leader]
    one_prior = np.eye(4)[prior]
    gate = (margin > tau)[:, None]
    hinge = np.maximum(margin - tau, 0)[:, None]
    if model == "null": cols = []
    elif model == "compression": cols = [c(x)]
    elif model == "targeted_prior_winner": cols = [c(one_prior)]
    elif model == "threshold_current_leader": cols = [c(one_leader * gate), c(one_leader * hinge)]
    elif model == "threshold_compression": cols = [c(x), c(x * gate)]
    else: raise ValueError(model)
    return np.stack(cols, axis=-1).reshape(-1, len(cols)) if cols else np.empty((x.size, 0))


def _ridge_fit_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, alpha: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    if xtr.shape[1] == 0:
        return np.full(len(xte), ytr.mean()), np.asarray([ytr.mean()])
    mean, scale = xtr.mean(axis=0), xtr.std(axis=0)
    scale[scale < 1e-10] = 1
    a = (xtr - mean) / scale; b = (xte - mean) / scale
    a = np.column_stack([np.ones(len(a)), a]); b = np.column_stack([np.ones(len(b)), b])
    penalty = np.eye(a.shape[1]) * alpha; penalty[0, 0] = 0
    coef = np.linalg.solve(a.T @ a + penalty, a.T @ ytr)
    # Return slopes on the original predictor scale for interpretation.
    raw = np.r_[coef[0] - np.sum(coef[1:] * mean / scale), coef[1:] / scale]
    return b @ coef, raw


def _best_tau(model: str, x: np.ndarray, prior: np.ndarray, leader: np.ndarray, margin: np.ndarray, y: np.ndarray, seed: int) -> float:
    if not model.startswith("threshold_"):
        return 0.0
    candidates = np.unique(np.quantile(margin, np.linspace(.1, .9, 17)))
    inner = folds(len(x), 3, seed)
    scores = []
    for tau in candidates:
        errors = []
        for test in inner:
            train = np.setdiff1d(np.arange(len(x)), test)
            tr = _design(model, x[train], prior[train], leader[train], margin[train], tau)
            te = _design(model, x[test], prior[test], leader[test], margin[test], tau)
            pred, _ = _ridge_fit_predict(tr, y[train].reshape(-1), te)
            errors.append(np.mean((pred - y[test].reshape(-1)) ** 2))
        scores.append(np.mean(errors))
    return float(candidates[int(np.argmin(scores))])


def analyze(input_dir: str, output_dir: str, k_folds: int, seed: int) -> dict:
    conditions = ["baseline", "incorrect", "neutral"]
    data = load_activation_dataset(input_dir, conditions)
    z = centered(data.logits)
    prior = np.argmax(z[:, 0, -1], axis=-1)
    n, _, n_layers, _ = z.shape
    outer = folds(n, k_folds, seed)
    rows = []
    # Baseline is primary; neutral asks whether the same change is feedback-specific.
    for reference_name, reference_i in (("baseline", 0), ("neutral", 2)):
        game = z[:, 1]
        ref = z[:, reference_i]
        for layer in range(n_layers - 1):
            x = game[:, layer]
            leader = np.argmax(x, axis=-1)
            sorted_x = np.sort(x, axis=-1)
            margin = sorted_x[:, -1] - sorted_x[:, -2]
            y = centered((game[:, layer + 1] - game[:, layer]) - (ref[:, layer + 1] - ref[:, layer]))
            for model_name in MODELS:
                predictions = np.empty_like(y)
                fold_coefficients, fold_taus = [], []
                for fold_i, test in enumerate(outer):
                    train = np.setdiff1d(np.arange(n), test)
                    tau = _best_tau(model_name, x[train], prior[train], leader[train], margin[train], y[train], seed + fold_i)
                    xtr = _design(model_name, x[train], prior[train], leader[train], margin[train], tau)
                    xte = _design(model_name, x[test], prior[test], leader[test], margin[test], tau)
                    pred, coef = _ridge_fit_predict(xtr, y[train].reshape(-1), xte)
                    predictions[test] = pred.reshape(len(test), 4)
                    fold_coefficients.append(coef.tolist()); fold_taus.append(tau)
                mse = float(np.mean((predictions - y) ** 2))
                null_mse = float(np.mean(y ** 2))
                max_coef_len = max(map(len, fold_coefficients))
                mean_coefs = np.nanmean(
                    np.asarray([c + [np.nan] * (max_coef_len - len(c)) for c in fold_coefficients]), axis=0
                ).tolist()
                rows.append({
                    "reference": reference_name, "transition_from": layer, "transition_to": layer + 1,
                    "model": model_name, "oos_mse": mse,
                    "oos_r2_vs_zero": 1 - mse / null_mse if null_mse else 0.0,
                    "mean_tau": float(np.mean(fold_taus)),
                    "mean_coefficients_json": json.dumps(mean_coefs),
                    "fold_coefficients_json": json.dumps(fold_coefficients),
                })
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "hypothesis_fits.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    best, best_by_model = {}, {}
    for ref in ("baseline", "neutral"):
        subset = [r for r in rows if r["reference"] == ref and r["model"] != "null"]
        winner = max(subset, key=lambda r: r["oos_r2_vs_zero"])
        fields = ("model", "transition_from", "transition_to", "oos_mse", "oos_r2_vs_zero", "mean_tau", "mean_coefficients_json")
        best[ref] = {k: winner[k] for k in fields}
        best_by_model[ref] = {}
        for name in MODELS[1:]:
            candidate = max((r for r in subset if r["model"] == name), key=lambda r: r["oos_r2_vs_zero"])
            best_by_model[ref][name] = {k: candidate[k] for k in fields if k != "model"}
    summary = {
        "n_questions": n, "n_readouts": n_layers,
        "best_single_transition_fit": best,
        "best_transition_by_hypothesis": best_by_model,
        "interpretation": {
            "threshold_current_leader": "Supports H1 only if it wins out of sample and its gated-leader slopes are negative.",
            "targeted_prior_winner": "Supports H2 only if its prior-winner slope is negative.",
            "compression": "Supports H3 when the current-logit slope is negative; unexplained residual variance is the perturbation diagnostic.",
            "warning": "These are observational signatures. They distinguish trajectory shapes but do not establish a causal circuit.",
        },
    }
    (out / "hypothesis_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Compare observational signatures of Second Chance hypotheses")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--folds", type=int, default=5); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); print(json.dumps(analyze(args.input, args.output, args.folds, args.seed), indent=2))


if __name__ == "__main__": main()
