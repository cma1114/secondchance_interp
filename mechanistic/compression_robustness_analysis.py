from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .data import load_activation_dataset
from .perturbation_analysis import _style
from .probes import stratified_folds
from .trajectory_analysis import centered


RANK_LABELS = ("Winner", "Runner-up", "Rank 3", "Rank 4")
COLORS = {"incorrect": "#0072B2", "neutral": "#D55E00", "difference": "#009E73"}


def _rank_of_option(order: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(order)
    ranks[np.arange(len(order))[:, None], order] = np.arange(4)[None, :]
    return ranks


def _question_geometry(baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(baseline, axis=-1)
    margin = ordered[:, -1] - ordered[:, -2]
    spread = baseline.std(axis=-1)
    return margin, spread


class FlexibleRankAgnosticModel:
    """Nonlinear score model with no answer-rank features."""

    def __init__(self, alpha: float = 1.0, n_knots: int = 6):
        self.alpha = alpha
        self.spline = SplineTransformer(
            n_knots=n_knots,
            degree=3,
            include_bias=False,
            knots="quantile",
            extrapolation="linear",
        )
        self.scaler = StandardScaler()
        self.ridge = Ridge(alpha=alpha)
        self.margin_mean = 0.0
        self.margin_scale = 1.0
        self.spread_mean = 0.0
        self.spread_scale = 1.0

    def _raw_features(
        self,
        scores: np.ndarray,
        margins: np.ndarray,
        spreads: np.ndarray,
        letters: np.ndarray,
        fit: bool,
    ) -> np.ndarray:
        score_column = scores[:, None]
        if fit:
            score_basis = self.spline.fit_transform(score_column)
            self.margin_mean, self.margin_scale = float(margins.mean()), float(margins.std())
            self.spread_mean, self.spread_scale = float(spreads.mean()), float(spreads.std())
            self.margin_scale = max(self.margin_scale, 1e-8)
            self.spread_scale = max(self.spread_scale, 1e-8)
        else:
            score_basis = self.spline.transform(score_column)
        margin_z = (margins - self.margin_mean) / self.margin_scale
        spread_z = (spreads - self.spread_mean) / self.spread_scale
        letter_basis = np.eye(4)[letters, :3]
        return np.column_stack(
            [
                score_basis,
                score_basis * margin_z[:, None],
                score_basis * spread_z[:, None],
                letter_basis,
            ]
        )

    def fit(
        self,
        scores: np.ndarray,
        margins: np.ndarray,
        spreads: np.ndarray,
        letters: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
    ) -> "FlexibleRankAgnosticModel":
        raw = self._raw_features(scores, margins, spreads, letters, fit=True)
        features = self.scaler.fit_transform(raw)
        self.ridge.fit(features, target, sample_weight=weights)
        return self

    def predict(
        self,
        scores: np.ndarray,
        margins: np.ndarray,
        spreads: np.ndarray,
        letters: np.ndarray,
    ) -> np.ndarray:
        raw = self._raw_features(scores, margins, spreads, letters, fit=False)
        return self.ridge.predict(self.scaler.transform(raw))


def _question_weights(winner: np.ndarray) -> np.ndarray:
    counts = np.bincount(winner, minlength=4).astype(float)
    return len(winner) / (4.0 * counts[winner])


def _cross_fitted_flexible_residuals(
    baseline: np.ndarray,
    target: np.ndarray,
    order: np.ndarray,
    folds: list[np.ndarray],
    excluded_rank: int | None,
    alpha: float,
    n_knots: int,
) -> np.ndarray:
    n = len(baseline)
    ranks = _rank_of_option(order)
    winner = order[:, 0]
    weights = _question_weights(winner)
    margin, spread = _question_geometry(baseline)
    letters = np.broadcast_to(np.arange(4), (n, 4))
    prediction = np.empty_like(target)
    all_questions = np.arange(n)
    for test in folds:
        train = np.setdiff1d(all_questions, test)
        train_mask = np.ones((len(train), 4), dtype=bool)
        if excluded_rank is not None:
            train_mask &= ranks[train] != excluded_rank
        train_questions = np.broadcast_to(train[:, None], (len(train), 4))[train_mask]
        train_letters = letters[train][train_mask]
        model = FlexibleRankAgnosticModel(alpha=alpha, n_knots=n_knots).fit(
            baseline[train][train_mask],
            margin[train_questions],
            spread[train_questions],
            train_letters,
            target[train][train_mask],
            weights[train_questions],
        )
        test_questions = np.repeat(test, 4)
        pred = model.predict(
            baseline[test].reshape(-1),
            margin[test_questions],
            spread[test_questions],
            letters[test].reshape(-1),
        ).reshape(len(test), 4)
        # The measured A-D updates are centered within question. Remove any
        # question-constant prediction before defining option-specific residuals.
        prediction[test] = pred - pred.mean(axis=-1, keepdims=True)
    residual = target - prediction
    residual -= residual.mean(axis=-1, keepdims=True)
    return residual


def _aligned_rank_contrasts(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    aligned = np.take_along_axis(values, order, axis=-1)
    contrasts = np.empty_like(aligned)
    for rank in range(4):
        other = [x for x in range(4) if x != rank]
        contrasts[:, rank] = aligned[:, rank] - aligned[:, other].mean(axis=-1)
    return contrasts


def _bootstrap_vector(
    values: np.ndarray,
    winner: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strata = [np.flatnonzero(winner == letter) for letter in range(4)]
    mean = np.mean(np.stack([values[idx].mean(axis=0) for idx in strata]), axis=0)
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, values.shape[1]))
    for repetition in range(repetitions):
        draws[repetition] = np.mean(
            np.stack([values[rng.choice(idx, len(idx), replace=True)].mean(axis=0) for idx in strata]),
            axis=0,
        )
    return mean, np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def _score_match(
    baseline: np.ndarray,
    contrast: np.ndarray,
    order: np.ndarray,
    mask: np.ndarray,
    multivariate: bool,
    k: int = 5,
    caliper_sd: float = 0.25,
) -> tuple[np.ndarray, dict]:
    ids = np.flatnonzero(mask)
    margin, spread = _question_geometry(baseline)
    option_sd = float(baseline[ids].std())
    margin_sd = max(float(margin[ids].std()), 1e-8)
    spread_sd = max(float(spread[ids].std()), 1e-8)
    matched = []
    distances = []
    for q in ids:
        runner_letter = int(order[q, 1])
        candidates = []
        for donor_q in ids:
            if donor_q == q:
                continue
            for donor_rank in (2, 3):
                if int(order[donor_q, donor_rank]) != runner_letter:
                    continue
                score_distance = abs(baseline[q, runner_letter] - baseline[donor_q, runner_letter]) / option_sd
                if score_distance > caliper_sd:
                    continue
                if multivariate:
                    distance = np.sqrt(
                        score_distance**2
                        + ((margin[q] - margin[donor_q]) / margin_sd) ** 2
                        + ((spread[q] - spread[donor_q]) / spread_sd) ** 2
                    )
                else:
                    distance = score_distance
                candidates.append((distance, score_distance, donor_q, runner_letter))
        if len(candidates) < k:
            continue
        selected = sorted(candidates, key=lambda row: row[0])[:k]
        donor_value = np.mean([contrast[dq, letter] for _, _, dq, letter in selected])
        matched.append(contrast[q, runner_letter] - donor_value)
        distances.append(np.mean([score_distance for _, score_distance, _, _ in selected]))
    return np.asarray(matched), {
        "n": len(matched),
        "k": k,
        "score_caliper_sd": caliper_sd,
        "mean_score_distance_sd": float(np.mean(distances)),
        "median_score_distance_sd": float(np.median(distances)),
    }


def _bootstrap_scalar(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for repetition in range(repetitions):
        draws[repetition] = rng.choice(values, len(values), replace=True).mean()
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _plot(summary: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.0))
    x = np.arange(4)
    width = 0.34
    all_rank = summary["flexible_all_rank_fit"]
    for offset, condition in ((-width / 2, "incorrect"), (width / 2, "neutral")):
        row = all_rank[condition]
        mean, low, high = (np.asarray(row[key]) for key in ("mean", "ci_low", "ci_high"))
        axes[0].bar(x + offset, mean, width, color=COLORS[condition], label="Second Chance" if condition == "incorrect" else "Neutral")
        axes[0].errorbar(x + offset, mean, yerr=np.vstack([mean - low, high - mean]), fmt="none", ecolor="#333333", capsize=2, lw=0.8)
    axes[0].set_title("A  Flexible rank-agnostic fit", loc="left", fontweight="bold")
    axes[0].set_ylabel("Focal rank residual vs other ranks\n(natural-logit units)")
    axes[0].legend(frameon=False, fontsize=7.5)

    leave = summary["leave_each_rank_out"]["incorrect_minus_neutral"]
    mean, low, high = (np.asarray(leave[key])[1:] for key in ("mean", "ci_low", "ci_high"))
    alternative_x = np.arange(3)
    axes[1].bar(alternative_x, mean, 0.58, color=COLORS["difference"])
    axes[1].errorbar(alternative_x, mean, yerr=np.vstack([mean - low, high - mean]), fmt="none", ecolor="#333333", capsize=2, lw=0.8)
    axes[1].set_title("B  Leave each alternative rank out", loc="left", fontweight="bold")
    axes[1].set_ylabel("Second Chance minus Neutral residual")

    methods = ("score_only", "score_margin_spread")
    labels = ("Score only", "Score + geometry")
    means = np.asarray([summary["baseline_score_matching"][m]["mean"] for m in methods])
    lows = np.asarray([summary["baseline_score_matching"][m]["ci"][0] for m in methods])
    highs = np.asarray([summary["baseline_score_matching"][m]["ci"][1] for m in methods])
    axes[2].bar(np.arange(2), means, 0.52, color=COLORS["difference"])
    axes[2].errorbar(np.arange(2), means, yerr=np.vstack([means - lows, highs - means]), fmt="none", ecolor="#333333", capsize=2, lw=0.8)
    axes[2].set_xticks(np.arange(2), labels)
    axes[2].set_title("C  Matched runner vs lower options", loc="left", fontweight="bold")
    axes[2].set_ylabel("Second Chance minus Neutral difference")
    for idx, method in enumerate(methods):
        axes[2].text(idx, means[idx], f"n={summary['baseline_score_matching'][method]['n']}", ha="center", va="bottom", fontsize=7.5)

    for axis in axes:
        axis.axhline(0, color="#555555", lw=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.7)
        axis.set_axisbelow(True)
    axes[0].set_xticks(x, RANK_LABELS, rotation=20, ha="right")
    axes[1].set_xticks(alternative_x, RANK_LABELS[1:], rotation=20, ha="right")
    fig.tight_layout(w_pad=1.25)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"compression_robustness.{suffix}", bbox_inches="tight")
    plt.close(fig)


def analyze(input_dir: str | Path, output_dir: str | Path, bootstrap: int, seed: int) -> dict:
    data = load_activation_dataset(input_dir, ["baseline", "incorrect", "neutral"])
    logits = centered(data.logits)[:, :, -1]
    baseline = logits[:, 0]
    targets = {"incorrect": logits[:, 1] - baseline, "neutral": logits[:, 2] - baseline}
    order = np.argsort(-baseline, axis=-1)
    winner = order[:, 0]
    choices = np.argmax(logits, axis=-1)
    both_keep = (choices[:, 1] == winner) & (choices[:, 2] == winner)
    folds = stratified_folds(winner, 5, seed)

    all_rank_residuals = {
        condition: _cross_fitted_flexible_residuals(
            baseline, target, order, folds, None, alpha=1.0, n_knots=6
        )
        for condition, target in targets.items()
    }
    summary: dict = {
        "n_questions": len(baseline),
        "both_keep_n": int(both_keep.sum()),
        "model": {
            "rank_features": False,
            "score_spline_knots": 6,
            "score_spline_degree": 3,
            "interactions": ["score_spline_x_winner_margin", "score_spline_x_A-D_spread"],
            "option_letter_effects": True,
            "ridge_alpha": 1.0,
            "cross_folds": 5,
        },
        "flexible_all_rank_fit": {},
        "leave_each_rank_out": {},
        "baseline_score_matching": {},
        "alpha_sensitivity_runner_game_minus_neutral": {},
        "knot_sensitivity_runner_game_minus_neutral": {},
        "matching_sensitivity": {},
    }
    for condition, residual in all_rank_residuals.items():
        values = _aligned_rank_contrasts(residual, order)[both_keep]
        mean, low, high = _bootstrap_vector(values, winner[both_keep], bootstrap, seed + (0 if condition == "incorrect" else 1))
        summary["flexible_all_rank_fit"][condition] = {"mean": mean.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist()}
    difference_values = (
        _aligned_rank_contrasts(all_rank_residuals["incorrect"], order)
        - _aligned_rank_contrasts(all_rank_residuals["neutral"], order)
    )[both_keep]
    mean, low, high = _bootstrap_vector(difference_values, winner[both_keep], bootstrap, seed + 2)
    summary["flexible_all_rank_fit"]["incorrect_minus_neutral"] = {"mean": mean.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist()}

    leave_condition = {condition: np.empty((len(baseline), 4)) for condition in targets}
    for focal_rank in range(4):
        for condition, target in targets.items():
            residual = _cross_fitted_flexible_residuals(
                baseline, target, order, folds, focal_rank, alpha=1.0, n_knots=6
            )
            leave_condition[condition][:, focal_rank] = _aligned_rank_contrasts(residual, order)[:, focal_rank]
    for condition, values in leave_condition.items():
        mean, low, high = _bootstrap_vector(values[both_keep], winner[both_keep], bootstrap, seed + 10 + (0 if condition == "incorrect" else 1))
        summary["leave_each_rank_out"][condition] = {"mean": mean.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist()}
    leave_difference = (leave_condition["incorrect"] - leave_condition["neutral"])[both_keep]
    mean, low, high = _bootstrap_vector(leave_difference, winner[both_keep], bootstrap, seed + 12)
    summary["leave_each_rank_out"]["incorrect_minus_neutral"] = {"mean": mean.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist()}

    for alpha in (0.1, 1.0, 10.0):
        residuals = {
            condition: _cross_fitted_flexible_residuals(
                baseline, target, order, folds, 1, alpha=alpha, n_knots=6
            )
            for condition, target in targets.items()
        }
        runner = (
            _aligned_rank_contrasts(residuals["incorrect"], order)[:, 1]
            - _aligned_rank_contrasts(residuals["neutral"], order)[:, 1]
        )[both_keep]
        estimate, ci_low, ci_high = _bootstrap_scalar(runner, bootstrap, seed + 100 + int(alpha * 10))
        summary["alpha_sensitivity_runner_game_minus_neutral"][str(alpha)] = {
            "mean": estimate,
            "ci": [ci_low, ci_high],
        }

    for n_knots in (4, 6, 8):
        residuals = {
            condition: _cross_fitted_flexible_residuals(
                baseline, target, order, folds, 1, alpha=1.0, n_knots=n_knots
            )
            for condition, target in targets.items()
        }
        runner = (
            _aligned_rank_contrasts(residuals["incorrect"], order)[:, 1]
            - _aligned_rank_contrasts(residuals["neutral"], order)[:, 1]
        )[both_keep]
        estimate, ci_low, ci_high = _bootstrap_scalar(runner, bootstrap, seed + 150 + n_knots)
        summary["knot_sensitivity_runner_game_minus_neutral"][str(n_knots)] = {
            "mean": estimate,
            "ci": [ci_low, ci_high],
        }

    game_minus_neutral = logits[:, 1] - logits[:, 2]
    for name, multivariate in (("score_only", False), ("score_margin_spread", True)):
        values, metadata = _score_match(
            baseline, game_minus_neutral, order, both_keep, multivariate=multivariate
        )
        estimate, ci_low, ci_high = _bootstrap_scalar(values, bootstrap, seed + 200 + int(multivariate))
        summary["baseline_score_matching"][name] = {
            **metadata,
            "mean": estimate,
            "ci": [ci_low, ci_high],
        }
    for caliper in (0.15, 0.25, 0.40):
        for k in (3, 5, 10):
            values, metadata = _score_match(
                baseline,
                game_minus_neutral,
                order,
                both_keep,
                multivariate=True,
                k=k,
                caliper_sd=caliper,
            )
            estimate, ci_low, ci_high = _bootstrap_scalar(
                values, bootstrap, seed + 300 + int(caliper * 100) + k
            )
            summary["matching_sensitivity"][f"caliper_{caliper:.2f}_k_{k}"] = {
                **metadata,
                "mean": estimate,
                "ci": [ci_low, ci_high],
            }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "compression_robustness.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = []
    for family in ("flexible_all_rank_fit", "leave_each_rank_out"):
        for condition, result in summary[family].items():
            for rank in range(4):
                rows.append({
                    "family": family,
                    "condition": condition,
                    "rank": rank + 1,
                    "rank_label": RANK_LABELS[rank],
                    "mean": result["mean"][rank],
                    "ci_low": result["ci_low"][rank],
                    "ci_high": result["ci_high"][rank],
                })
    with (output / "compression_robustness.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _plot(summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether runner preservation survives flexible nonlinear compression controls")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, args.bootstrap, args.seed), indent=2))


if __name__ == "__main__":
    main()
