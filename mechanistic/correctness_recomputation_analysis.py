from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import load_activation_dataset
from .probes import stratified_folds
from .trajectory_analysis import centered


def _weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=4).astype(float)
    return len(labels) / (4 * counts[labels])


def _design(baseline: np.ndarray, winner: np.ndarray, correct: np.ndarray, include_correct: bool) -> np.ndarray:
    n = len(baseline)
    letter = np.broadcast_to(
        np.eye(4)[:, :3] - np.eye(4)[:, :3].mean(axis=0, keepdims=True),
        (n, 4, 3),
    )
    winner_col = np.eye(4)[winner]; winner_col -= winner_col.mean(axis=1, keepdims=True)
    cols = [letter, baseline[:, :, None], winner_col[:, :, None]]
    if include_correct:
        correct_col = np.eye(4)[correct]; correct_col -= correct_col.mean(axis=1, keepdims=True)
        cols.append(correct_col[:, :, None])
    return np.concatenate(cols, axis=-1).reshape(n * 4, -1)


def _ridge(xtr, ytr, weights, xte, alpha=1e-3):
    mean = np.average(xtr, axis=0, weights=weights)
    variance = np.average((xtr - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance); scale[scale < 1e-10] = 1
    a = np.column_stack([np.ones(len(xtr)), (xtr - mean) / scale])
    b = np.column_stack([np.ones(len(xte)), (xte - mean) / scale])
    root = np.sqrt(weights)
    penalty = np.eye(a.shape[1]) * alpha; penalty[0, 0] = 0
    coef = np.linalg.solve((a * root[:, None]).T @ (a * root[:, None]) + penalty, (a * root[:, None]).T @ (ytr * root))
    raw = np.r_[coef[0] - np.sum(coef[1:] * mean / scale), coef[1:] / scale]
    return b @ coef, raw


def _weighted_mse(y, pred, question_weights):
    w = np.repeat(question_weights, 4)
    return float(np.sum(w * (y.reshape(-1) - pred.reshape(-1)) ** 2) / np.sum(w))


def _correct_strength(z: np.ndarray, correct: np.ndarray) -> np.ndarray:
    selected = z[np.arange(len(z)), :, correct]
    return selected - (z.sum(axis=-1) - selected) / 3.0


def analyze(input_dir: str, output_dir: str, folds: int, bootstrap: int, seed: int) -> dict:
    data = load_activation_dataset(input_dir, ["baseline", "incorrect", "neutral"])
    all_z = centered(data.logits)
    keep = np.asarray([not data.metadata[(q, "baseline")]["baseline_correct"] for q in data.question_ids])
    z = all_z[keep]
    qids = [q for q, use in zip(data.question_ids, keep) if use]
    correct = np.asarray(["ABCD".index(data.metadata[(q, "baseline")]["correct_answer"]) for q in qids])
    winner = np.argmax(z[:, 0, -1], axis=-1)
    n, _, layers, _ = z.shape
    question_weights = _weights(correct)
    fold_ids = stratified_folds(correct, folds, seed)
    all_ids = np.arange(n)
    target = z[:, 1] - z[:, 2]
    rng = np.random.default_rng(seed)
    rows = []

    for layer in range(layers):
        predictions = {False: np.empty_like(target[:, layer]), True: np.empty_like(target[:, layer])}
        fold_coefs = []
        for test in fold_ids:
            train = np.setdiff1d(all_ids, test)
            for include_correct in (False, True):
                xtr = _design(z[train, 0, layer], winner[train], correct[train], include_correct)
                xte = _design(z[test, 0, layer], winner[test], correct[test], include_correct)
                pred, coef = _ridge(
                    xtr,
                    target[train, layer].reshape(-1),
                    np.repeat(question_weights[train], 4),
                    xte,
                )
                predictions[include_correct][test] = pred.reshape(len(test), 4)
                if include_correct:
                    fold_coefs.append(float(coef[-1]))
        base_mse = _weighted_mse(target[:, layer], predictions[False], question_weights)
        full_mse = _weighted_mse(target[:, layer], predictions[True], question_weights)

        # Clustered, correct-letter-stratified coefficient bootstrap.
        boot_coef = np.empty(bootstrap)
        strata = [np.flatnonzero(correct == letter) for letter in range(4)]
        for b in range(bootstrap):
            sample = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in strata])
            design = _design(z[sample, 0, layer], winner[sample], correct[sample], True)
            _, coef = _ridge(
                design,
                target[sample, layer].reshape(-1),
                np.repeat(_weights(correct[sample]), 4),
                design,
            )
            boot_coef[b] = coef[-1]
        rows.append({
            "readout": layer,
            "correct_option_coefficient": float(np.mean(fold_coefs)),
            "coefficient_ci_low": float(np.quantile(boot_coef, .025)),
            "coefficient_ci_high": float(np.quantile(boot_coef, .975)),
            "oos_mse_without_correct": base_mse,
            "oos_mse_with_correct": full_mse,
            "incremental_oos_r2": 1 - full_mse / base_mse if base_mse else 0.0,
        })

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "correctness_recomputation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

    strength = np.stack([_correct_strength(z[:, ci], correct) for ci in range(3)], axis=1)
    game_neutral = strength[:, 1] - strength[:, 2]
    final_boot = game_neutral[:, -1][rng.integers(0, n, (5000, n))].mean(axis=1)
    final = rows[-1]
    selected = np.unique(np.rint(np.asarray([.5, .75, .875, .95, 1.0]) * (layers - 1)).astype(int))
    summary = {
        "n_baseline_incorrect_questions": n,
        "estimand": "Game-minus-neutral centered A-D pseudo-logits on baseline-incorrect trials.",
        "final_correct_option_advantage_game_minus_neutral": float(game_neutral[:, -1].mean()),
        "final_correct_option_advantage_ci": np.quantile(final_boot, [.025, .975]).tolist(),
        "final_correct_option_coefficient_controlling_baseline_geometry_and_winner": final["correct_option_coefficient"],
        "final_coefficient_ci": [final["coefficient_ci_low"], final["coefficient_ci_high"]],
        "final_incremental_oos_r2": final["incremental_oos_r2"],
        "selected_readouts": {str(layer): rows[layer] for layer in selected},
        "interpretation": "A positive held-out coefficient means the correct option gains beyond broad compression, original-winner identity, and fixed option-letter effects.",
    }
    (out / "correctness_recomputation_summary.json").write_text(json.dumps(summary, indent=2))
    _plot(rows, out)
    return summary


def _plot(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    x = np.asarray([r["readout"] for r in rows])
    mean = np.asarray([r["correct_option_coefficient"] for r in rows])
    lo = np.asarray([r["coefficient_ci_low"] for r in rows]); hi = np.asarray([r["coefficient_ci_high"] for r in rows])
    r2 = np.asarray([r["incremental_oos_r2"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].fill_between(x, lo, hi, color="#762a83", alpha=.18)
    axes[0].plot(x, mean, color="#762a83")
    axes[0].axhline(0, color="black", lw=.7)
    axes[0].set(title="Correct-option coefficient", xlabel="Residual readout", ylabel="Game − neutral pseudo-logits")
    axes[1].plot(x, r2, color="#1b7837")
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].set(title="Added held-out value of correctness", xlabel="Residual readout", ylabel="Incremental out-of-sample $R^2$")
    fig.tight_layout(); fig.savefig(out / "correctness_recomputation.png", dpi=220); fig.savefig(out / "correctness_recomputation.svg")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Test correctness-directed recomputation on baseline-incorrect trials")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--folds", type=int, default=5); p.add_argument("--bootstrap", type=int, default=500); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); print(json.dumps(analyze(args.input, args.output, args.folds, args.bootstrap, args.seed), indent=2))


if __name__ == "__main__":
    main()
