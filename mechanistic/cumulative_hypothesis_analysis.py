from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import load_activation_dataset
from .probes import stratified_folds
from .trajectory_analysis import centered


CONDITIONS = ("incorrect", "neutral", "incorrect_minus_neutral")
CONDITION_LABELS = {
    "incorrect": "Second Chance - baseline",
    "neutral": "Neutral - baseline",
    "incorrect_minus_neutral": "Second Chance - neutral",
}
CONDITION_COLORS = {
    "incorrect": "#0072B2",
    "neutral": "#D55E00",
    "incorrect_minus_neutral": "#009E73",
}
CONDITION_STYLES = {"incorrect": "-", "neutral": "--", "incorrect_minus_neutral": "-."}
MODELS = (
    "letter_only",
    "compression",
    "winner",
    "compression_winner",
    "threshold_leader",
    "full",
)


def _question_weights(labels: np.ndarray) -> np.ndarray:
    """Give each original-winner letter equal total weight."""
    counts = np.bincount(labels, minlength=4).astype(float)
    if np.any(counts == 0):
        raise ValueError("Every original-winner letter must occur at least once")
    return len(labels) / (4 * counts[labels])


def _weighted_mse(y: np.ndarray, pred: np.ndarray, question_weights: np.ndarray) -> float:
    weights = np.repeat(question_weights, 4)
    return float(np.sum(weights * (y.reshape(-1) - pred.reshape(-1)) ** 2) / np.sum(weights))


def _weighted_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    weights_train = np.asarray(weights_train, dtype=float)
    weight_sum = weights_train.sum()
    mean = (x_train * weights_train[:, None]).sum(axis=0) / weight_sum
    centered_x = x_train - mean
    variance = (weights_train[:, None] * centered_x**2).sum(axis=0) / weight_sum
    scale = np.sqrt(variance)
    scale[scale < 1e-10] = 1

    train_standard = centered_x / scale
    test_standard = (x_test - mean) / scale
    train_design = np.column_stack([np.ones(len(train_standard)), train_standard])
    test_design = np.column_stack([np.ones(len(test_standard)), test_standard])
    root_weight = np.sqrt(weights_train)
    weighted_design = train_design * root_weight[:, None]
    weighted_y = y_train * root_weight
    penalty = np.eye(train_design.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(weighted_design.T @ weighted_design + penalty, weighted_design.T @ weighted_y)
    raw = np.r_[coef[0] - np.sum(coef[1:] * mean / scale), coef[1:] / scale]
    return test_design @ coef, raw


def _design(
    model: str,
    baseline: np.ndarray,
    winner: np.ndarray,
    leader: np.ndarray,
    margin: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, list[str]]:
    n = len(baseline)
    letter_basis = np.eye(4)[:, :3] - np.eye(4)[:, :3].mean(axis=0, keepdims=True)
    columns = [np.broadcast_to(letter_basis, (n, 4, 3))]
    names = ["letter_A", "letter_B", "letter_C"]

    winner_column = np.eye(4)[winner]
    winner_column -= winner_column.mean(axis=1, keepdims=True)
    leader_column = np.eye(4)[leader]
    leader_column -= leader_column.mean(axis=1, keepdims=True)
    gate = (margin > tau).astype(float)
    hinge = np.maximum(margin - tau, 0)

    if model in ("compression", "compression_winner", "full"):
        columns.append(baseline[:, :, None])
        names.append("baseline_geometry")
    if model in ("winner", "compression_winner", "full"):
        columns.append(winner_column[:, :, None])
        names.append("original_winner")
    if model in ("threshold_leader", "full"):
        columns.append((leader_column * gate[:, None])[:, :, None])
        columns.append((leader_column * hinge[:, None])[:, :, None])
        names.extend(("threshold_gate", "threshold_hinge"))
    if model not in MODELS:
        raise ValueError(model)
    return np.concatenate(columns, axis=-1).reshape(n * 4, -1), names


def _choose_tau(
    model: str,
    baseline: np.ndarray,
    winner: np.ndarray,
    leader: np.ndarray,
    margin: np.ndarray,
    target: np.ndarray,
    question_weights: np.ndarray,
    seed: int,
) -> float:
    if model not in ("threshold_leader", "full"):
        return 0.0
    candidates = np.unique(np.r_[0.0, np.quantile(margin, np.linspace(0.1, 0.9, 17))])
    inner_folds = stratified_folds(winner, 3, seed)
    scores = []
    all_ids = np.arange(len(baseline))
    for tau in candidates:
        fold_errors = []
        for test in inner_folds:
            train = np.setdiff1d(all_ids, test)
            x_train, _ = _design(model, baseline[train], winner[train], leader[train], margin[train], tau)
            x_test, _ = _design(model, baseline[test], winner[test], leader[test], margin[test], tau)
            pred, _ = _weighted_ridge(
                x_train,
                target[train].reshape(-1),
                np.repeat(question_weights[train], 4),
                x_test,
            )
            fold_errors.append(_weighted_mse(target[test], pred.reshape(len(test), 4), question_weights[test]))
        scores.append(float(np.mean(fold_errors)))
    return float(candidates[int(np.argmin(scores))])


def _cross_validated_models(
    baseline: np.ndarray,
    winner: np.ndarray,
    target: np.ndarray,
    question_weights: np.ndarray,
    outer_folds: list[np.ndarray],
    seed: int,
) -> dict[str, dict]:
    n = len(baseline)
    leader = np.argmax(baseline, axis=-1)
    sorted_baseline = np.sort(baseline, axis=-1)
    margin = sorted_baseline[:, -1] - sorted_baseline[:, -2]
    all_ids = np.arange(n)
    results = {}
    for model in MODELS:
        predictions = np.empty_like(target)
        coefficients = []
        taus = []
        names: list[str] = []
        for fold_index, test in enumerate(outer_folds):
            train = np.setdiff1d(all_ids, test)
            tau = _choose_tau(
                model,
                baseline[train],
                winner[train],
                leader[train],
                margin[train],
                target[train],
                question_weights[train],
                seed + fold_index,
            )
            x_train, names = _design(model, baseline[train], winner[train], leader[train], margin[train], tau)
            x_test, _ = _design(model, baseline[test], winner[test], leader[test], margin[test], tau)
            pred, coef = _weighted_ridge(
                x_train,
                target[train].reshape(-1),
                np.repeat(question_weights[train], 4),
                x_test,
            )
            predictions[test] = pred.reshape(len(test), 4)
            coefficients.append(coef.tolist())
            taus.append(tau)
        results[model] = {
            "prediction": predictions,
            "mse": _weighted_mse(target, predictions, question_weights),
            "coefficient_names": ["intercept", *names],
            "fold_coefficients": coefficients,
            "taus": taus,
        }
    zero_mse = _weighted_mse(target, np.zeros_like(target), question_weights)
    letter_mse = results["letter_only"]["mse"]
    for result in results.values():
        result["r2_zero"] = 1 - result["mse"] / zero_mse if zero_mse else 0.0
        result["r2_letter"] = 1 - result["mse"] / letter_mse if letter_mse else 0.0
    return results


def _fit_decomposition(
    baseline: np.ndarray,
    winner: np.ndarray,
    target: np.ndarray,
    question_weights: np.ndarray,
) -> tuple[float, float]:
    leader = np.argmax(baseline, axis=-1)
    sorted_baseline = np.sort(baseline, axis=-1)
    margin = sorted_baseline[:, -1] - sorted_baseline[:, -2]
    design, names = _design("compression_winner", baseline, winner, leader, margin, 0.0)
    _, coef = _weighted_ridge(
        design,
        target.reshape(-1),
        np.repeat(question_weights, 4),
        design,
    )
    mapping = dict(zip(["intercept", *names], coef))
    return -float(mapping["baseline_geometry"]), -float(mapping["original_winner"])


def _bootstrap_decomposition(
    baseline_by_layer: np.ndarray,
    winner: np.ndarray,
    target_by_layer: np.ndarray,
    seed: int,
    repetitions: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(winner == label) for label in range(4)]
    n_layers = baseline_by_layer.shape[1]
    compression = np.empty((repetitions, n_layers))
    winner_penalty = np.empty((repetitions, n_layers))
    for repetition in range(repetitions):
        sample = np.concatenate([rng.choice(ids, size=len(ids), replace=True) for ids in strata])
        sampled_winner = winner[sample]
        sampled_weights = _question_weights(sampled_winner)
        for layer in range(n_layers):
            compression[repetition, layer], winner_penalty[repetition, layer] = _fit_decomposition(
                baseline_by_layer[sample, layer],
                sampled_winner,
                target_by_layer[sample, layer],
                sampled_weights,
            )
    return compression, winner_penalty


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "svg.fonttype": "none",
    })


def _finish_axis(axis, n_layers: int) -> None:
    axis.axhline(0, color="#555555", lw=0.65)
    final = n_layers - 1
    step = max(1, int(round(final / 8)))
    axis.set_xlim(0, final)
    axis.set_xticks(np.unique(np.r_[np.arange(0, final + 1, step), final]))
    axis.set_xlabel(f"Residual readout (0 = embedding; {final} = final block)")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.7)
    axis.set_axisbelow(True)


def _save_figures(output: Path, coefficient_rows: list[dict], fit_rows: list[dict], n_layers: int) -> None:
    import matplotlib.pyplot as plt

    _style()
    layers = np.arange(n_layers)
    by_condition = {
        condition: sorted((row for row in coefficient_rows if row["condition"] == condition), key=lambda row: row["layer"])
        for condition in CONDITIONS
    }
    fit_lookup = {(row["condition"], row["layer"], row["model"]): row for row in fit_rows}

    figure, axes = plt.subplots(2, 2, figsize=(7.6, 5.8), sharex=True)
    for condition in CONDITIONS:
        rows = by_condition[condition]
        color = CONDITION_COLORS[condition]
        style = CONDITION_STYLES[condition]
        label = CONDITION_LABELS[condition]
        for axis, prefix in zip(axes[0], ("compression", "winner_penalty")):
            mean = np.asarray([row[prefix] for row in rows])
            low = np.asarray([row[f"{prefix}_ci_low"] for row in rows])
            high = np.asarray([row[f"{prefix}_ci_high"] for row in rows])
            axis.fill_between(layers, low, high, color=color, alpha=0.16, linewidth=0)
            axis.plot(layers, mean, color=color, ls=style, lw=1.55, label=label)

        winner_increment = np.asarray([
            fit_lookup[(condition, layer, "compression_winner")]["oos_r2_vs_letter_only"]
            - fit_lookup[(condition, layer, "compression")]["oos_r2_vs_letter_only"]
            for layer in layers
        ])
        threshold_increment = np.asarray([
            fit_lookup[(condition, layer, "full")]["oos_r2_vs_letter_only"]
            - fit_lookup[(condition, layer, "compression_winner")]["oos_r2_vs_letter_only"]
            for layer in layers
        ])
        axes[1, 0].plot(layers, winner_increment, color=color, ls=style, lw=1.55, label=label)
        axes[1, 1].plot(layers, threshold_increment, color=color, ls=style, lw=1.55, label=label)

    axes[0, 0].set_title("A  Broad compression", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Compression strength\n(0 = none; 1 = erase baseline geometry)")
    axes[0, 1].set_title("B  Original-winner penalty", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("Winner-relative penalty\n(natural-logit units)")
    axes[1, 0].set_title("C  Added value of winner identity", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("Increment in held-out $R^2$\nbeyond compression")
    axes[1, 1].set_title("D  Added value of a threshold rule", loc="left", fontweight="bold")
    axes[1, 1].set_ylabel("Increment in held-out $R^2$\nbeyond compression + winner")
    for axis in axes.ravel():
        _finish_axis(axis, n_layers)
    axes[0, 0].legend(frameon=False, loc="upper left")
    figure.tight_layout(w_pad=2.1, h_pad=1.6)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"cumulative_hypothesis_decomposition.{suffix}", bbox_inches="tight")
    plt.close(figure)


def plot_saved_results(output_dir: str | Path) -> None:
    output = Path(output_dir)
    with (output / "cumulative_hypothesis_coefficients.csv").open(newline="") as stream:
        coefficient_rows = list(csv.DictReader(stream))
    with (output / "cumulative_hypothesis_fits.csv").open(newline="") as stream:
        fit_rows = list(csv.DictReader(stream))
    for row in coefficient_rows:
        row["layer"] = int(row["layer"])
        for key in (
            "compression",
            "compression_ci_low",
            "compression_ci_high",
            "winner_penalty",
            "winner_penalty_ci_low",
            "winner_penalty_ci_high",
        ):
            row[key] = float(row[key])
    for row in fit_rows:
        row["layer"] = int(row["layer"])
        for key in ("oos_mse", "oos_r2_vs_zero", "oos_r2_vs_letter_only", "mean_tau"):
            row[key] = float(row[key])
    n_layers = max(row["layer"] for row in coefficient_rows) + 1
    _save_figures(output, coefficient_rows, fit_rows, n_layers)

    import matplotlib.pyplot as plt

    _style()
    layers = np.arange(n_layers)
    fit_lookup = {(row["condition"], row["layer"], row["model"]): row for row in fit_rows}

    model_labels = {
        "letter_only": "Letter-only nuisance",
        "compression": "Compression",
        "winner": "Original winner",
        "compression_winner": "Compression + winner",
        "threshold_leader": "Thresholded current leader",
        "full": "Compression + winner + threshold",
    }
    model_colors = {
        "letter_only": "#777777",
        "compression": "#0072B2",
        "winner": "#E69F00",
        "compression_winner": "#009E73",
        "threshold_leader": "#CC79A7",
        "full": "#6A3D9A",
    }
    figure, axes = plt.subplots(1, len(CONDITIONS), figsize=(11.5, 3.25), sharex=True, sharey=True)
    for axis, condition in zip(axes, CONDITIONS):
        for model in MODELS:
            values = [fit_lookup[(condition, layer, model)]["oos_r2_vs_letter_only"] for layer in layers]
            axis.plot(layers, values, lw=1.35, color=model_colors[model], label=model_labels[model])
        axis.set_title(CONDITION_LABELS[condition])
        _finish_axis(axis, n_layers)
    axes[0].set_ylabel("Held-out $R^2$ beyond option-letter effects")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    figure.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.5)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"cumulative_hypothesis_model_fits.{suffix}", bbox_inches="tight")
    plt.close(figure)


def analyze(
    input_dir: str | Path,
    output_dir: str | Path,
    k_folds: int,
    seed: int,
    bootstrap_repetitions: int,
) -> dict:
    data = load_activation_dataset(input_dir, ["baseline", "incorrect", "neutral"])
    logits = centered(data.logits)
    baseline = logits[:, 0]
    winner = np.argmax(baseline[:, -1], axis=-1)
    question_weights = _question_weights(winner)
    outer_folds = stratified_folds(winner, k_folds, seed)
    n, n_layers = baseline.shape[:2]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fit_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    coefficient_bootstraps = {}
    targets = {}

    target_map = {
        "incorrect": logits[:, 1] - baseline,
        "neutral": logits[:, 2] - baseline,
        "incorrect_minus_neutral": logits[:, 1] - logits[:, 2],
    }
    for condition_index, condition in enumerate(CONDITIONS, start=1):
        target_by_layer = target_map[condition]
        targets[condition] = target_by_layer
        bootstrap_compression, bootstrap_winner = _bootstrap_decomposition(
            baseline,
            winner,
            target_by_layer,
            seed + 1000 * condition_index,
            bootstrap_repetitions,
        )
        coefficient_bootstraps[condition] = (bootstrap_compression, bootstrap_winner)

        for layer in range(n_layers):
            model_results = _cross_validated_models(
                baseline[:, layer],
                winner,
                target_by_layer[:, layer],
                question_weights,
                outer_folds,
                seed + layer * 10,
            )
            for model, result in model_results.items():
                fit_rows.append({
                    "condition": condition,
                    "reference": "baseline",
                    "layer": layer,
                    "model": model,
                    "oos_mse": result["mse"],
                    "oos_r2_vs_zero": result["r2_zero"],
                    "oos_r2_vs_letter_only": result["r2_letter"],
                    "mean_tau": float(np.mean(result["taus"])),
                    "coefficient_names_json": json.dumps(result["coefficient_names"]),
                    "fold_coefficients_json": json.dumps(result["fold_coefficients"]),
                    "fold_taus_json": json.dumps(result["taus"]),
                })

            compression_strength, winner_penalty = _fit_decomposition(
                baseline[:, layer],
                winner,
                target_by_layer[:, layer],
                question_weights,
            )
            coefficient_rows.append({
                "condition": condition,
                "reference": "baseline",
                "layer": layer,
                "compression": compression_strength,
                "compression_ci_low": float(np.quantile(bootstrap_compression[:, layer], 0.025)),
                "compression_ci_high": float(np.quantile(bootstrap_compression[:, layer], 0.975)),
                "winner_penalty": winner_penalty,
                "winner_penalty_ci_low": float(np.quantile(bootstrap_winner[:, layer], 0.025)),
                "winner_penalty_ci_high": float(np.quantile(bootstrap_winner[:, layer], 0.975)),
            })

    with (output / "cumulative_hypothesis_fits.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fit_rows[0].keys())
        writer.writeheader()
        writer.writerows(fit_rows)
    with (output / "cumulative_hypothesis_coefficients.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=coefficient_rows[0].keys())
        writer.writeheader()
        writer.writerows(coefficient_rows)

    fit_lookup = {(row["condition"], row["layer"], row["model"]): row for row in fit_rows}
    coefficient_lookup = {(row["condition"], row["layer"]): row for row in coefficient_rows}
    summary = {
        "n_questions": n,
        "n_readouts": n_layers,
        "estimand": "Letter-balanced cumulative condition minus baseline centered pseudo-logits.",
        "decomposition": (
            "At each readout, condition-baseline is regressed on the same baseline readout geometry "
            "and a centered indicator for the final baseline winner, with option-letter nuisance effects."
        ),
        "cross_validation": f"{k_folds}-fold, stratified by final baseline-winner letter; threshold selected within training folds.",
        "bootstrap": f"{bootstrap_repetitions} question-clustered, winner-letter-stratified repetitions.",
        "selected_layers": {},
    }
    for condition in CONDITIONS:
        summary["selected_layers"][condition] = {}
        selected = np.unique(np.rint(np.asarray([0, .25, .5, .75, .875, .95, 1.0]) * (n_layers - 1)).astype(int))
        for layer in selected:
            coef = coefficient_lookup[(condition, layer)]
            summary["selected_layers"][condition][str(layer)] = {
                "compression_strength": coef["compression"],
                "compression_ci": [coef["compression_ci_low"], coef["compression_ci_high"]],
                "winner_penalty": coef["winner_penalty"],
                "winner_penalty_ci": [coef["winner_penalty_ci_low"], coef["winner_penalty_ci_high"]],
                "oos_r2_compression": fit_lookup[(condition, layer, "compression")]["oos_r2_vs_zero"],
                "oos_r2_winner": fit_lookup[(condition, layer, "winner")]["oos_r2_vs_zero"],
                "oos_r2_combined": fit_lookup[(condition, layer, "compression_winner")]["oos_r2_vs_zero"],
                "oos_r2_threshold": fit_lookup[(condition, layer, "threshold_leader")]["oos_r2_vs_zero"],
                "oos_r2_full": fit_lookup[(condition, layer, "full")]["oos_r2_vs_zero"],
            }
    (output / "cumulative_hypothesis_summary.json").write_text(json.dumps(summary, indent=2))
    _save_figures(output, coefficient_rows, fit_rows, n_layers)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Cumulative Game-baseline and neutral-baseline hypothesis analysis")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot_saved_results(args.output)
        return
    summary = analyze(args.input, args.output, args.folds, args.seed, args.bootstrap)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
