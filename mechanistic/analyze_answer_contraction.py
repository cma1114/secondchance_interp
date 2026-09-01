from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_jlens_answer_content import answer_letter_scores, baseline_rank_order
from .answer_emergence_figures import RANK_COLORS, Z_975, macro_mean_and_se
from .data import decision_letter, load_activation_dataset


CONDITIONS = ("incorrect", "neutral")
LABELS = {"incorrect": "Game", "neutral": "Neutral"}
RANKS = ("Original winner", "Original runner-up", "Original rank 3", "Original rank 4")
SELECTED_LAYERS = (32, 40, 48, 52, 56, 60, 64)


def _labels(data, condition: str) -> np.ndarray:
    result = []
    for qid in data.question_ids:
        letter = decision_letter(data.metadata[(qid, condition)])
        if letter not in "ABCD":
            raise ValueError(f"Non-A-D output for {qid}/{condition}: {letter!r}")
        result.append("ABCD".index(letter))
    return np.asarray(result, dtype=np.int64)


def _weights(strata: np.ndarray) -> np.ndarray:
    """Question weights giving each Baseline answer letter one quarter weight."""
    weights = np.zeros(len(strata), dtype=np.float64)
    for letter in range(4):
        count = int(np.sum(strata == letter))
        if not count:
            raise ValueError(f"No questions in Baseline-answer stratum {letter}")
        weights[strata == letter] = 0.25 / count
    return weights


def _fit_alpha(baseline: np.ndarray, condition: np.ndarray, strata: np.ndarray) -> np.ndarray:
    weights = _weights(strata)[:, None, None]
    numerator = np.sum(weights * baseline * condition, axis=(0, 2))
    denominator = np.sum(weights * baseline * baseline, axis=(0, 2))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def _energy_metrics(
    baseline: np.ndarray,
    condition: np.ndarray,
    fitted_alpha: np.ndarray,
    strata: np.ndarray,
) -> dict[str, np.ndarray]:
    weights = _weights(strata)[:, None, None]
    delta = condition - baseline
    fitted_delta = (fitted_alpha[None, :, None] - 1.0) * baseline
    residual = delta - fitted_delta
    delta_energy = np.sum(weights * delta * delta, axis=(0, 2))
    residual_energy = np.sum(weights * residual * residual, axis=(0, 2))
    condition_energy = np.sum(weights * condition * condition, axis=(0, 2))
    baseline_energy = np.sum(weights * baseline * baseline, axis=(0, 2))
    state_error = condition - fitted_alpha[None, :, None] * baseline
    state_error_energy = np.sum(weights * state_error * state_error, axis=(0, 2))
    return {
        "transformation_fraction_explained": np.divide(
            delta_energy - residual_energy,
            delta_energy,
            out=np.full_like(delta_energy, np.nan),
            where=delta_energy > 0,
        ),
        "state_fraction_explained": np.divide(
            condition_energy - state_error_energy,
            condition_energy,
            out=np.full_like(condition_energy, np.nan),
            where=condition_energy > 0,
        ),
        "norm_ratio": np.sqrt(np.divide(condition_energy, baseline_energy)),
        "residual_rms": np.sqrt(np.sum(weights * residual * residual, axis=(0, 2))),
    }


def _stratified_folds(strata: np.ndarray, folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(strata), dtype=np.int64)
    for letter in range(4):
        indices = np.flatnonzero(strata == letter)
        rng.shuffle(indices)
        assignment[indices] = np.arange(len(indices)) % folds
    return assignment


def _cross_fit(
    baseline: np.ndarray,
    condition: np.ndarray,
    strata: np.ndarray,
    fold_assignment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    alpha_by_question = np.empty((len(strata), baseline.shape[1]), dtype=np.float64)
    for fold in np.unique(fold_assignment):
        held_out = fold_assignment == fold
        alpha = _fit_alpha(baseline[~held_out], condition[~held_out], strata[~held_out])
        alpha_by_question[held_out] = alpha
    delta = condition - baseline
    fitted_delta = (alpha_by_question[:, :, None] - 1.0) * baseline
    residual = delta - fitted_delta

    weights = _weights(strata)[:, None, None]
    delta_energy = np.sum(weights * delta * delta, axis=(0, 2))
    residual_energy = np.sum(weights * residual * residual, axis=(0, 2))
    condition_energy = np.sum(weights * condition * condition, axis=(0, 2))
    state_error = condition - alpha_by_question[:, :, None] * baseline
    state_error_energy = np.sum(weights * state_error * state_error, axis=(0, 2))
    metrics = {
        "transformation_fraction_explained": (delta_energy - residual_energy) / delta_energy,
        "state_fraction_explained": (condition_energy - state_error_energy) / condition_energy,
    }
    return alpha_by_question, residual, metrics


def _balanced_bootstrap_indices(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = [np.flatnonzero(strata == letter) for letter in range(4)]
    return np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])


def _bootstrap_curves(
    baseline: np.ndarray,
    condition: np.ndarray,
    strata: np.ndarray,
    fold_assignment: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    alpha_draws = np.empty((draws, baseline.shape[1]), dtype=np.float64)
    fraction_draws = np.empty_like(alpha_draws)
    for draw in range(draws):
        sampled = _balanced_bootstrap_indices(strata, rng)
        alpha = _fit_alpha(baseline[sampled], condition[sampled], strata[sampled])
        alpha_draws[draw] = alpha
        sampled_baseline = baseline[sampled]
        sampled_condition = condition[sampled]
        sampled_strata = strata[sampled]
        sampled_folds = fold_assignment[sampled]
        alpha_by_question, _, cross_metrics = _cross_fit(
            sampled_baseline, sampled_condition, sampled_strata, sampled_folds
        )
        # Duplicated bootstrap copies retain their original question's fold, so
        # a question can never appear in both the fit and evaluation split.
        fraction_draws[draw] = cross_metrics["transformation_fraction_explained"]
    return {
        "alpha_low": np.quantile(alpha_draws, 0.025, axis=0),
        "alpha_high": np.quantile(alpha_draws, 0.975, axis=0),
        "fraction_low": np.quantile(fraction_draws, 0.025, axis=0),
        "fraction_high": np.quantile(fraction_draws, 0.975, axis=0),
    }


def _subset_mean_and_se(
    values: np.ndarray, strata: np.ndarray, subset: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    """Use a letter macro when identified; otherwise report the question mixture."""
    group = values[subset]
    if len(group) < 2:
        raise ValueError("Need at least two questions in an outcome subset")
    counts = np.asarray([np.sum(subset & (strata == letter)) for letter in range(4)])
    if np.all(counts >= 2):
        mean, se = macro_mean_and_se(group, strata[subset])
        return mean, se, "equal Baseline-answer-letter macro-average"
    return (
        group.mean(axis=0),
        group.std(axis=0, ddof=1) / np.sqrt(len(group)),
        "question average (letter macro unidentified because a stratum has fewer than two cases)",
    )


def _rank_summary(values: np.ndarray, order: np.ndarray, strata: np.ndarray) -> dict[str, np.ndarray]:
    aligned = np.take_along_axis(values, order[:, None, :], axis=-1)
    means, halfwidths = [], []
    for rank in range(4):
        mean, se = macro_mean_and_se(aligned[:, :, rank], strata)
        means.append(mean)
        halfwidths.append(Z_975 * se)
    return {"mean": np.stack(means), "halfwidth": np.stack(halfwidths)}


def _rank_slope(rank_values: np.ndarray) -> np.ndarray:
    """Slope on an increasing-rank axis; positive means higher boosts for lower Baseline ranks."""
    axis = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    return np.sum(rank_values * axis[:, None], axis=0) / np.sum(axis * axis)


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
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def _finish(ax, layers: np.ndarray, ylabel: str) -> None:
    ax.set_xlim(layers[0], layers[-1])
    ax.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
    ax.set_xlabel("Residual readout")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def _plot_summary(output: Path, layers: np.ndarray, results: dict) -> None:
    import matplotlib.pyplot as plt

    _style()
    colors = {"incorrect": "#0072B2", "neutral": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.4))
    for condition in CONDITIONS:
        result = results[condition]
        color = colors[condition]
        axes[0, 0].fill_between(
            layers, result["bootstrap"]["alpha_low"], result["bootstrap"]["alpha_high"],
            color=color, alpha=0.25, linewidth=0,
        )
        axes[0, 0].plot(
            layers, result["bootstrap"]["alpha_low"], color=color, lw=0.6, alpha=0.65,
        )
        axes[0, 0].plot(
            layers, result["bootstrap"]["alpha_high"], color=color, lw=0.6, alpha=0.65,
        )
        axes[0, 0].plot(layers, result["alpha"], color=color, lw=1.55, label=LABELS[condition])
        axes[0, 1].plot(
            layers, result["cross_fit"]["transformation_fraction_explained"],
            color=color, lw=1.55, label=LABELS[condition],
        )
        axes[0, 1].fill_between(
            layers,
            result["bootstrap"]["fraction_low"],
            result["bootstrap"]["fraction_high"],
            color=color, alpha=0.25, linewidth=0,
        )
        axes[0, 1].plot(
            layers, result["bootstrap"]["fraction_low"], color=color, lw=0.6, alpha=0.65,
        )
        axes[0, 1].plot(
            layers, result["bootstrap"]["fraction_high"], color=color, lw=0.6, alpha=0.65,
        )
        axes[1, 0].plot(layers, result["oof_residual_rms_mean"], color=color, lw=1.55, label=LABELS[condition])
        axes[1, 0].fill_between(
            layers,
            result["oof_residual_rms_mean"] - result["oof_residual_rms_halfwidth"],
            result["oof_residual_rms_mean"] + result["oof_residual_rms_halfwidth"],
            color=color, alpha=0.18, linewidth=0,
        )

    axes[0, 0].axhline(1, color="#555555", ls=(0, (3, 2)), lw=0.8)
    axes[0, 0].set_title("A  Fitted gain on Baseline evidence", loc="left", fontweight="bold")
    _finish(axes[0, 0], layers, r"Gain $\alpha$")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].axhline(0, color="#555555", lw=0.7)
    axes[0, 1].set_title("B  Cross-validated contraction account", loc="left", fontweight="bold")
    _finish(axes[0, 1], layers, "Fraction of condition change explained")

    axes[1, 0].set_title("C  Structure left after contraction", loc="left", fontweight="bold")
    _finish(axes[1, 0], layers, "Cross-fitted residual RMS")

    game = results["incorrect"]
    for group, color in (("repeat", "#009E73"), ("switch", "#CC79A7")):
        axes[1, 1].plot(layers, game[f"{group}_rms_mean"], color=color, lw=1.55, label=group.title())
        axes[1, 1].fill_between(
            layers,
            game[f"{group}_rms_mean"] - game[f"{group}_rms_halfwidth"],
            game[f"{group}_rms_mean"] + game[f"{group}_rms_halfwidth"],
            color=color, alpha=0.18, linewidth=0,
        )
    axes[1, 1].set_title("D  Game residual by behavioral outcome", loc="left", fontweight="bold")
    _finish(axes[1, 1], layers, "Cross-fitted residual RMS")
    axes[1, 1].legend(frameon=False)

    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    fig.text(
        0.5, -0.005,
        "Shading and boundary lines: 95% answer-letter-stratified, question-clustered intervals.",
        ha="center", va="top", fontsize=7.2, color="#555555",
    )
    fig.savefig(output / "answer_contraction_summary.png", bbox_inches="tight")
    plt.close(fig)


def _plot_rank_decomposition(output: Path, layers: np.ndarray, rank: dict) -> None:
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.15), sharey=True)
    panels = (
        ("total", "A  Observed Game minus Baseline"),
        ("fitted", "B  Scalar contraction predicts"),
        ("residual", "C  Left after contraction"),
    )
    for ax, (key, title) in zip(axes, panels):
        values = rank[key]
        for index, (label, color) in enumerate(zip(RANKS, RANK_COLORS)):
            ax.fill_between(
                layers,
                values["mean"][index] - values["halfwidth"][index],
                values["mean"][index] + values["halfwidth"][index],
                color=color, alpha=0.17, linewidth=0,
            )
            ax.plot(layers, values["mean"][index], color=color, lw=1.45, label=label)
        ax.axhline(0, color="#555555", lw=0.7)
        ax.set_title(title, loc="left", fontweight="bold")
        _finish(ax, layers, "Centered JLens-score change")
    axes[0].legend(frameon=False, fontsize=7.0, handlelength=1.4)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(output / "game_rank_contraction_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def analyze(
    jlens_root: Path,
    baseline_root: Path,
    second_chance_root: Path,
    output: Path,
    folds: int,
    bootstrap: int,
    seed: int,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    layout = json.loads((jlens_root / "selected_token_layout.json").read_text())
    with np.load(jlens_root / "jlens_scores.npz", allow_pickle=False) as cached:
        scores = answer_letter_scores(cached["final_scores"].astype(np.float64), layout)
        qids = cached["question_ids"].astype(str).tolist()
    scores -= scores.mean(axis=-1, keepdims=True)
    if scores.shape[0] != 3:
        raise ValueError(f"Expected Baseline/Game/Neutral axis, got {scores.shape}")

    baseline_data = load_activation_dataset(baseline_root, ["baseline"])
    second_data = load_activation_dataset(second_chance_root, list(CONDITIONS))
    if baseline_data.question_ids != qids or second_data.question_ids != qids:
        raise ValueError("Question order differs between JLens scores and activation shards")
    order, prior = baseline_rank_order(baseline_data)
    baseline_output = _labels(baseline_data, "baseline")
    if not np.array_equal(prior, baseline_output):
        raise ValueError("Baseline rank winner and generated Baseline answer differ")

    condition_axis = {"incorrect": 1, "neutral": 2}
    layers = np.arange(1, scores.shape[2] + 1)
    baseline = scores[0]
    results: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {"layers": layers, "question_ids": np.asarray(qids), "prior": prior}

    for offset, condition in enumerate(CONDITIONS):
        current = scores[condition_axis[condition]]
        generated = _labels(second_data, condition)
        switched = generated != prior
        alpha = _fit_alpha(baseline, current, prior)
        in_sample = _energy_metrics(baseline, current, alpha, prior)
        fold_assignment = _stratified_folds(prior, folds, seed + offset * 100)
        alpha_by_question, residual, cross_metrics = _cross_fit(
            baseline, current, prior, fold_assignment
        )
        residual_rms = np.sqrt(np.mean(residual * residual, axis=-1))
        residual_mean, residual_se = macro_mean_and_se(residual_rms, prior)
        repeat_mean, repeat_se, repeat_aggregation = _subset_mean_and_se(
            residual_rms, prior, ~switched
        )
        switch_mean, switch_se, switch_aggregation = _subset_mean_and_se(
            residual_rms, prior, switched
        )
        boot = _bootstrap_curves(
            baseline, current, prior, fold_assignment, bootstrap, seed + offset * 1000
        )
        denominator = np.sum(baseline * baseline, axis=-1)
        question_alpha = np.divide(
            np.sum(baseline * current, axis=-1),
            denominator,
            out=np.full_like(denominator, np.nan),
            where=denominator > 1e-12,
        )
        results[condition] = {
            "alpha": alpha,
            "in_sample": in_sample,
            "cross_fit": cross_metrics,
            "bootstrap": boot,
            "oof_residual_rms_mean": residual_mean,
            "oof_residual_rms_halfwidth": Z_975 * residual_se,
            "repeat_rms_mean": repeat_mean,
            "repeat_rms_halfwidth": Z_975 * repeat_se,
            "switch_rms_mean": switch_mean,
            "switch_rms_halfwidth": Z_975 * switch_se,
            "switch_rate": float(np.mean(switched)),
            "switch_n": int(np.sum(switched)),
            "switch_counts_by_baseline_letter": {
                letter: int(np.sum(switched & (prior == index)))
                for index, letter in enumerate("ABCD")
            },
            "outcome_aggregation": {
                "switch": switch_aggregation,
                "repeat": repeat_aggregation,
            },
        }
        arrays.update({
            f"{condition}_alpha": alpha,
            f"{condition}_alpha_ci_low": boot["alpha_low"],
            f"{condition}_alpha_ci_high": boot["alpha_high"],
            f"{condition}_transformation_fraction_ci_low": boot["fraction_low"],
            f"{condition}_transformation_fraction_ci_high": boot["fraction_high"],
            f"{condition}_alpha_by_question": alpha_by_question,
            f"{condition}_question_alpha": question_alpha,
            f"{condition}_oof_residual": residual,
            f"{condition}_oof_residual_rms": residual_rms,
            f"{condition}_switched": switched,
            f"{condition}_crossfit_transformation_fraction": cross_metrics["transformation_fraction_explained"],
            f"{condition}_crossfit_state_fraction": cross_metrics["state_fraction_explained"],
        })

    game = scores[1]
    game_delta = game - baseline
    game_alpha_by_q = results["incorrect"]["alpha"]
    fitted_delta = (game_alpha_by_q[None, :, None] - 1.0) * baseline
    residual_delta = game_delta - fitted_delta
    rank = {
        "total": _rank_summary(game_delta, order, prior),
        "fitted": _rank_summary(fitted_delta, order, prior),
        "residual": _rank_summary(residual_delta, order, prior),
    }
    total_slope = _rank_slope(rank["total"]["mean"])
    fitted_slope = _rank_slope(rank["fitted"]["mean"])
    residual_slope = _rank_slope(rank["residual"]["mean"])
    slope_fraction = np.divide(
        fitted_slope, total_slope, out=np.full_like(fitted_slope, np.nan), where=np.abs(total_slope) > 1e-12
    )
    arrays.update({
        "game_rank_total_mean": rank["total"]["mean"],
        "game_rank_fitted_mean": rank["fitted"]["mean"],
        "game_rank_residual_mean": rank["residual"]["mean"],
        "game_rank_total_halfwidth": rank["total"]["halfwidth"],
        "game_rank_fitted_halfwidth": rank["fitted"]["halfwidth"],
        "game_rank_residual_halfwidth": rank["residual"]["halfwidth"],
        "game_rank_total_slope": total_slope,
        "game_rank_fitted_slope": fitted_slope,
        "game_rank_residual_slope": residual_slope,
        "game_rank_slope_fraction_explained": slope_fraction,
    })
    np.savez_compressed(output / "answer_contraction_arrays.npz", **arrays)

    csv_rows = []
    for condition in CONDITIONS:
        result = results[condition]
        for layer_index, layer in enumerate(layers):
            csv_rows.append({
                "condition": LABELS[condition],
                "layer": int(layer),
                "alpha": float(result["alpha"][layer_index]),
                "alpha_ci_low": float(result["bootstrap"]["alpha_low"][layer_index]),
                "alpha_ci_high": float(result["bootstrap"]["alpha_high"][layer_index]),
                "crossfit_transformation_fraction_explained": float(
                    result["cross_fit"]["transformation_fraction_explained"][layer_index]
                ),
                "crossfit_state_fraction_explained": float(
                    result["cross_fit"]["state_fraction_explained"][layer_index]
                ),
                "oof_residual_rms": float(result["oof_residual_rms_mean"][layer_index]),
                "switch_oof_residual_rms": float(result["switch_rms_mean"][layer_index]),
                "repeat_oof_residual_rms": float(result["repeat_rms_mean"][layer_index]),
            })
    with (output / "answer_contraction_layerwise.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    selected = {}
    for layer in SELECTED_LAYERS:
        index = layer - 1
        selected[str(layer)] = {
            LABELS[condition]: {
                "alpha": float(results[condition]["alpha"][index]),
                "alpha_ci": [
                    float(results[condition]["bootstrap"]["alpha_low"][index]),
                    float(results[condition]["bootstrap"]["alpha_high"][index]),
                ],
                "crossfit_transformation_fraction_explained": float(
                    results[condition]["cross_fit"]["transformation_fraction_explained"][index]
                ),
                "crossfit_state_fraction_explained": float(
                    results[condition]["cross_fit"]["state_fraction_explained"][index]
                ),
                "oof_residual_rms": float(results[condition]["oof_residual_rms_mean"][index]),
                "switch_oof_residual_rms": float(results[condition]["switch_rms_mean"][index]),
                "repeat_oof_residual_rms": float(results[condition]["repeat_rms_mean"][index]),
            }
            for condition in CONDITIONS
        }
        selected[str(layer)]["Game_rank_slope"] = {
            "observed": float(total_slope[index]),
            "scalar_contraction": float(fitted_slope[index]),
            "residual": float(residual_slope[index]),
            "fraction_explained": float(slope_fraction[index]),
        }

    summary = {
        "n_questions": len(qids),
        "folds": folds,
        "bootstrap_draws": bootstrap,
        "model": "centered condition evidence = alpha(layer) * centered Baseline evidence + residual",
        "balancing": "Every fit and aggregate gives each generated Baseline answer letter equal weight.",
        "cross_validation": "Alpha is fitted on four folds and evaluated on the held-out fifth fold.",
        "switch_counts": {
            LABELS[condition]: {
                "n": results[condition]["switch_n"],
                "rate": results[condition]["switch_rate"],
                "by_baseline_letter": results[condition]["switch_counts_by_baseline_letter"],
                "outcome_aggregation": results[condition]["outcome_aggregation"],
            }
            for condition in CONDITIONS
        },
        "selected_layers": selected,
    }
    (output / "answer_contraction_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Scalar contraction test on corrected-prompt JLens evidence",
        "",
        f"Questions: {len(qids)}. Fits are balanced across the generated Baseline answer letter and evaluated with {folds}-fold cross-fitting.",
        "",
        "At each layer, the centered four-answer JLens vector in Game or Neutral is modeled as `condition = alpha * Baseline + residual`. Alpha below one is literal contraction. The cross-validated transformation fraction asks how much of the condition-minus-Baseline change is predicted by that one scalar; the residual is everything not predicted by contraction.",
        "",
        "| Layer | Game alpha | Neutral alpha | Game change explained | Neutral change explained | Game inverse-rank slope explained |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in SELECTED_LAYERS:
        row = selected[str(layer)]
        lines.append(
            f"| {layer} | {row['Game']['alpha']:.3f} | {row['Neutral']['alpha']:.3f} | "
            f"{row['Game']['crossfit_transformation_fraction_explained']:.1%} | "
            f"{row['Neutral']['crossfit_transformation_fraction_explained']:.1%} | "
            f"{row['Game_rank_slope']['fraction_explained']:.1%} |"
        )
    lines.extend([
        "",
        "Interpretation rule: high change-explained values and a small residual rank slope support global gain reduction. A substantial residual inverse-rank slope means the transformation is more specifically rank-structured than scalar contraction predicts. Switching necessarily depends on the residual because positive scalar contraction alone cannot change the answer ordering.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")

    _plot_summary(output, layers, results)
    _plot_rank_decomposition(output, layers, rank)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jlens-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--second-chance-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = analyze(
        args.jlens_root, args.baseline_root, args.second_chance_root,
        args.output, args.folds, args.bootstrap, args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
