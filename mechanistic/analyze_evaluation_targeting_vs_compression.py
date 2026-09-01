from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_existing_period_transmission import _align_content, _load_source
from .semantic_mapping import displayed_argmax_to_semantic_indices


LETTERS = "ABCD"
CONDITIONS = ("Game", "Matched Neutral")


def _stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    return np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])


def _fit_compression(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best per-question non-negative scalar contraction y = -lambda*x."""
    denom = np.sum(x * x, axis=1)
    lam = np.maximum(0.0, -np.sum(x * y, axis=1) / np.maximum(denom, 1e-12))
    return lam, -lam[:, None] * x


def _fit_compression_plus_w1(
    x: np.ndarray, y: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit y = -lambda_q*x + gamma*target with lambda_q >= 0."""
    gamma = 0.0
    denom_x = np.sum(x * x, axis=1)
    denom_t = float(np.sum(target * target))
    for _ in range(200):
        lam = np.maximum(
            0.0,
            -np.sum(x * (y - gamma * target), axis=1) / np.maximum(denom_x, 1e-12),
        )
        updated = float(np.sum(target * (y + lam[:, None] * x)) / denom_t)
        if abs(updated - gamma) < 1e-12:
            gamma = updated
            break
        gamma = updated
    prediction = -lam[:, None] * x + gamma * target
    return gamma, lam, prediction


def _fit_unrestricted_gamma(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> float:
    """Sensitivity fit allowing either compression or sharpening per question."""
    denom = np.sum(x * x, axis=1)
    target_resid = target - (
        np.sum(x * target, axis=1) / np.maximum(denom, 1e-12)
    )[:, None] * x
    return float(np.sum(target_resid * y) / np.sum(target_resid * target_resid))


def _model_metrics(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> dict[str, float]:
    lam0, pred0 = _fit_compression(x, y)
    gamma, lam1, pred1 = _fit_compression_plus_w1(x, y, target)
    sst = float(np.sum(y * y))
    sse0 = float(np.sum((y - pred0) ** 2))
    sse1 = float(np.sum((y - pred1) ** 2))
    return {
        "gamma": gamma,
        "gamma_unrestricted_sensitivity": _fit_unrestricted_gamma(x, y, target),
        "compression_r2": 1.0 - sse0 / sst,
        "compression_plus_w1_r2": 1.0 - sse1 / sst,
        "incremental_r2": (sse0 - sse1) / sst,
        "mean_lambda_compression_only": float(np.mean(lam0)),
        "mean_lambda_full_model": float(np.mean(lam1)),
        "fraction_positive_lambda": float(np.mean(lam0 > 0)),
    }


def _bootstrap_model(
    x: np.ndarray,
    y: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, dict[str, float]]:
    point = _model_metrics(x, y, target)
    rng = np.random.default_rng(seed)
    keys = ("gamma", "compression_r2", "compression_plus_w1_r2", "incremental_r2")
    samples = {key: np.empty(draws, dtype=float) for key in keys}
    for draw in range(draws):
        selected = _stratified_indices(labels, rng)
        values = _model_metrics(x[selected], y[selected], target[selected])
        for key in keys:
            samples[key][draw] = values[key]
    result = {}
    for key, value in point.items():
        result[key] = {"mean": float(value)}
        if key in samples:
            low, high = np.quantile(samples[key], (0.025, 0.975))
            result[key].update(ci_low=float(low), ci_high=float(high))
    return result


def _paired_gamma_difference(
    x: np.ndarray,
    y: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, float]:
    def estimate(selected):
        game = _fit_compression_plus_w1(x[0, selected], y[0, selected], target[selected])[0]
        neutral = _fit_compression_plus_w1(x[1, selected], y[1, selected], target[selected])[0]
        return game - neutral

    all_indices = np.arange(x.shape[1])
    point = estimate(all_indices)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        samples[draw] = estimate(_stratified_indices(labels, rng))
    low, high = np.quantile(samples, (0.025, 0.975))
    return {"mean": float(point), "ci_low": float(low), "ci_high": float(high)}


def _bootstrap_mean(values: np.ndarray, labels: np.ndarray, seed: int, draws: int):
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        samples[draw] = np.mean(values[_stratified_indices(labels, rng)])
    low, high = np.quantile(samples, (0.025, 0.975))
    return {"mean": float(np.mean(values)), "ci_low": float(low), "ci_high": float(high)}


def _bootstrap_paired_difference(
    first: np.ndarray,
    second: np.ndarray,
    labels: np.ndarray,
    seed: int,
    draws: int,
):
    """Bootstrap mean(first - second), preserving paired questions."""
    values = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    return _bootstrap_mean(values, labels, seed, draws)


def _fmt(interval: dict[str, float]) -> str:
    return f"{interval['mean']:+.3f} [{interval['ci_low']:+.3f}, {interval['ci_high']:+.3f}]"


def analyze(args):
    import matplotlib.pyplot as plt

    source = _load_source(args.source_run)
    qids = source["qids"]
    mappings = {row["question_id"]: row for row in json.loads(args.remapping_plan.read_text())["rows"]}
    baseline = json.loads(args.baseline.read_text())["results"]
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])

    w1 = np.asarray([baseline[qid]["answer"] for qid in qids])
    w2 = np.asarray([remapped[qid]["answer_original_content"] for qid in qids])
    w1_index = np.asarray([LETTERS.index(letter) for letter in w1])
    w2_index = np.asarray([LETTERS.index(letter) for letter in w2])
    conflict = w1 != w2
    discovery = np.asarray([qid in discovery_ids for qid in qids])

    natural = _align_content(source["natural"], qids, mappings).astype(float)
    ablated = _align_content(source["ablated"], qids, mappings).astype(float)
    mapping_rows = [mappings[qid] for qid in qids]
    natural_choice = displayed_argmax_to_semantic_indices(
        source["natural"], mapping_rows
    )
    ablated_choice = displayed_argmax_to_semantic_indices(
        source["ablated"], mapping_rows
    )
    natural -= natural.mean(axis=-1, keepdims=True)
    ablated -= ablated.mean(axis=-1, keepdims=True)
    effect = natural - ablated

    target = np.full((len(qids), 4), -0.25, dtype=float)
    target[np.arange(len(qids)), w1_index] = 0.75

    masks = {
        "all": np.ones(len(qids), dtype=bool),
        "conflict": conflict,
        "no_conflict": ~conflict,
    }
    summary = {
        "definitions": {
            "counterfactual_evidence": "Final centered A-D logits after deleting the evaluation-period write from all 48 GLAs.",
            "causal_update": "Natural minus globally ablated final centered A-D logits.",
            "generic_compression": "Per-question non-negative scalar contraction of the counterfactual A-D evidence vector.",
            "gamma": "Additional W1-versus-other-three update after fitting the best generic contraction; negative means extra W1 suppression.",
        },
        "validation": {
            "questions": len(qids),
            "conflict": int(conflict.sum()),
            "no_conflict": int((~conflict).sum()),
            "discovery": int(discovery.sum()),
            "confirmation": int((~discovery).sum()),
            "new_model_forwards": 0,
        },
        "models": {},
        "split_stability": {},
        "conflict_letter_stratification": {},
        "no_conflict_letter_stratification": {},
        "non_A_condition_comparison": {},
        "condition_gap_mediation": {},
        "decisive_order_reversal": {},
    }

    for subset, mask in masks.items():
        summary["models"][subset] = {}
        for ci, condition in enumerate(CONDITIONS):
            summary["models"][subset][condition] = _bootstrap_model(
                ablated[ci, mask], effect[ci, mask], target[mask], w1[mask],
                1000 + ci * 100 + len(summary["models"]), args.bootstrap_draws,
            )
        summary["models"][subset]["Game_minus_Neutral_gamma"] = _paired_gamma_difference(
            ablated[:, mask], effect[:, mask], target[mask], w1[mask],
            2000 + len(summary["models"]), args.bootstrap_draws,
        )

    for split_name, split_mask in (("discovery", discovery), ("confirmation", ~discovery)):
        summary["split_stability"][split_name] = {}
        for subset, subset_mask in (("conflict", conflict), ("no_conflict", ~conflict)):
            mask = split_mask & subset_mask
            summary["split_stability"][split_name][subset] = {}
            for ci, condition in enumerate(CONDITIONS):
                summary["split_stability"][split_name][subset][condition] = _bootstrap_model(
                    ablated[ci, mask], effect[ci, mask], target[mask], w1[mask],
                    3000 + ci * 100 + int(split_name == "confirmation") * 10,
                    args.bootstrap_draws,
                )["gamma"]

    displayed_w1 = np.asarray([
        mappings[qid]["original_to_new"][w1[qi]] for qi, qid in enumerate(qids)
    ])
    for subset_name, subset_mask, output_key in (
        ("conflict", conflict, "conflict_letter_stratification"),
        ("no_conflict", ~conflict, "no_conflict_letter_stratification"),
    ):
        letter_groups = {
            "W1_A": subset_mask & (w1 == "A"),
            "W1_non_A": subset_mask & (w1 != "A"),
        }
        letter_groups.update({
            f"W1_{letter}": subset_mask & (w1 == letter) for letter in LETTERS
        })
        for group_name, mask in letter_groups.items():
            group = {
                "n": int(mask.sum()),
                "displayed_W1_counts": {
                    letter: int(np.sum(mask & (displayed_w1 == letter))) for letter in LETTERS
                },
            }
            for ci, condition in enumerate(CONDITIONS):
                model = _bootstrap_model(
                    ablated[ci, mask], effect[ci, mask], target[mask], w1[mask],
                    3500 + ci * 100 + sum(ord(char) for char in group_name)
                    + (500 if subset_name == "no_conflict" else 0),
                    args.bootstrap_draws,
                )
                chosen = natural_choice[ci, mask]
                w1_chosen = chosen == w1_index[mask]
                group[condition] = {
                    "gamma": model["gamma"],
                    "gamma_unrestricted_sensitivity": model["gamma_unrestricted_sensitivity"],
                    "natural_W1_selection_rate": _bootstrap_mean(
                        w1_chosen.astype(float), w1[mask],
                        3600 + ci * 100 + sum(ord(char) for char in group_name)
                        + (500 if subset_name == "no_conflict" else 0),
                        args.bootstrap_draws,
                    ),
                }
            summary[output_key][group_name] = group

    causal_w1 = effect[:, np.arange(len(qids)), w1_index]
    natural_w1 = natural[:, np.arange(len(qids)), w1_index]
    ablated_w1 = ablated[:, np.arange(len(qids)), w1_index]
    for subset_name, subset_mask in (("conflict", conflict), ("no_conflict", ~conflict)):
        natural_logit_contrast = float(np.mean(natural_w1[0, subset_mask] - natural_w1[1, subset_mask]))
        ablated_logit_contrast = float(np.mean(ablated_w1[0, subset_mask] - ablated_w1[1, subset_mask]))
        natural_selection_gap = float(
            np.mean(natural_choice[1, subset_mask] == w1_index[subset_mask])
            - np.mean(natural_choice[0, subset_mask] == w1_index[subset_mask])
        )
        ablated_selection_gap = float(
            np.mean(ablated_choice[1, subset_mask] == w1_index[subset_mask])
            - np.mean(ablated_choice[0, subset_mask] == w1_index[subset_mask])
        )
        summary["condition_gap_mediation"][subset_name] = {
            "n": int(subset_mask.sum()),
            "natural_Game_minus_Neutral_centered_W1": natural_logit_contrast,
            "ablated_Game_minus_Neutral_centered_W1": ablated_logit_contrast,
            "fraction_centered_W1_contrast_removed": (
                (natural_logit_contrast - ablated_logit_contrast) / natural_logit_contrast
            ),
            "natural_Neutral_minus_Game_W1_selection": natural_selection_gap,
            "ablated_Neutral_minus_Game_W1_selection": ablated_selection_gap,
            "fraction_W1_selection_gap_removed": (
                (natural_selection_gap - ablated_selection_gap) / natural_selection_gap
            ),
        }
    for subset_name, subset_mask in (("conflict", conflict), ("no_conflict", ~conflict)):
        mask = subset_mask & (w1 != "A")
        game_selected = natural_choice[0, mask] == w1_index[mask]
        neutral_selected = natural_choice[1, mask] == w1_index[mask]
        summary["non_A_condition_comparison"][subset_name] = {
            "n": int(mask.sum()),
            "Game_W1_selection": _bootstrap_mean(
                game_selected.astype(float), w1[mask], 3700, args.bootstrap_draws
            ),
            "Matched_Neutral_W1_selection": _bootstrap_mean(
                neutral_selected.astype(float), w1[mask], 3701, args.bootstrap_draws
            ),
            "Neutral_minus_Game_W1_selection": _bootstrap_paired_difference(
                neutral_selected, game_selected, w1[mask], 3702, args.bootstrap_draws
            ),
            "Game_W1_causal_update": _bootstrap_mean(
                causal_w1[0, mask], w1[mask], 3703, args.bootstrap_draws
            ),
            "Matched_Neutral_W1_causal_update": _bootstrap_mean(
                causal_w1[1, mask], w1[mask], 3704, args.bootstrap_draws
            ),
            "Game_minus_Neutral_W1_causal_update": _bootstrap_paired_difference(
                causal_w1[0, mask], causal_w1[1, mask], w1[mask], 3705,
                args.bootstrap_draws,
            ),
        }

    rows = np.arange(len(qids))
    game_gap = ablated[0, rows, w1_index] - ablated[0, rows, w2_index]
    w1_below_w2 = conflict & (game_gap < 0)
    w1_above_w2 = conflict & ~w1_below_w2
    reversal_groups = (
        ("W1_below_W2", w1_below_w2),
        ("W1_below_W2_and_W1_A", w1_below_w2 & (w1 == "A")),
        ("W1_below_W2_and_W1_non_A", w1_below_w2 & (w1 != "A")),
        ("W1_at_or_above_W2", w1_above_w2),
    )
    for label, mask in reversal_groups:
        summary["decisive_order_reversal"][label] = {"n": int(mask.sum())}
        for ci, condition in enumerate(CONDITIONS):
            values = (
                effect[ci, rows, w1_index] - effect[ci, rows, w2_index]
            )[mask]
            summary["decisive_order_reversal"][label][condition] = _bootstrap_mean(
                values, w1[mask], 4000 + ci + int(label.startswith("W1_at")) * 10,
                args.bootstrap_draws,
            )

    # Equal-count conflict bins based on Game's counterfactual W1-minus-W2 evidence.
    conflict_indices = np.flatnonzero(conflict)
    order = conflict_indices[np.argsort(game_gap[conflict])]
    bins = np.array_split(order, 5)
    bin_rows = []
    for bin_index, selected in enumerate(bins, 1):
        for ci, condition in enumerate(CONDITIONS):
            lam, prediction = _fit_compression(ablated[ci, selected], effect[ci, selected])
            local = np.arange(len(selected))
            observed = (
                effect[ci, selected, w1_index[selected]]
                - effect[ci, selected, w2_index[selected]]
            )
            predicted = (
                prediction[local, w1_index[selected]]
                - prediction[local, w2_index[selected]]
            )
            bin_rows.append({
                "bin": bin_index,
                "condition": condition,
                "n": len(selected),
                "mean_counterfactual_w1_minus_w2": float(np.mean(
                    ablated[ci, selected, w1_index[selected]]
                    - ablated[ci, selected, w2_index[selected]]
                )),
                "observed": _bootstrap_mean(observed, w1[selected], 5000 + 20 * ci + bin_index, args.bootstrap_draws),
                "compression_prediction": _bootstrap_mean(predicted, w1[selected], 5100 + 20 * ci + bin_index, args.bootstrap_draws),
            })
    summary["conflict_bins"] = bin_rows

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "bin", "condition", "n", "mean_counterfactual_w1_minus_w2",
            "observed_mean", "observed_ci_low", "observed_ci_high",
            "compression_mean", "compression_ci_low", "compression_ci_high",
        ])
        writer.writeheader()
        for row in bin_rows:
            writer.writerow({
                "bin": row["bin"], "condition": row["condition"], "n": row["n"],
                "mean_counterfactual_w1_minus_w2": row["mean_counterfactual_w1_minus_w2"],
                "observed_mean": row["observed"]["mean"],
                "observed_ci_low": row["observed"]["ci_low"],
                "observed_ci_high": row["observed"]["ci_high"],
                "compression_mean": row["compression_prediction"]["mean"],
                "compression_ci_low": row["compression_prediction"]["ci_low"],
                "compression_ci_high": row["compression_prediction"]["ci_high"],
            })

    # Canonical PNG: two condition-specific binned checks plus residual W1 coefficient.
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    condition_colors = {"Game": "#2b83ba", "Matched Neutral": "#f07c2b"}
    prediction_color = "#6f6f6f"
    for ax, condition, panel in zip(axes[:2], CONDITIONS, ("A", "B")):
        rows_c = [row for row in bin_rows if row["condition"] == condition]
        xvals = np.asarray([row["mean_counterfactual_w1_minus_w2"] for row in rows_c])
        for key, label, color, marker, offset in (
            ("observed", "Observed causal update", condition_colors[condition], "o", -0.035),
            ("compression_prediction", "Best generic compression", prediction_color, "s", 0.035),
        ):
            means = np.asarray([row[key]["mean"] for row in rows_c])
            lows = np.asarray([row[key]["ci_low"] for row in rows_c])
            highs = np.asarray([row[key]["ci_high"] for row in rows_c])
            ax.errorbar(
                xvals + offset, means, yerr=[means - lows, highs - means],
                fmt=marker, color=color, capsize=3, label=label, markersize=6,
            )
        ax.axhline(0, color="0.45", lw=0.9)
        ax.axvline(0, color="0.45", lw=0.9)
        ax.set(
            title=f"{panel}  {condition}: conflict trials",
            xlabel="Counterfactual W1 − W2 evidence (logits)",
            ylabel="Causal W1 − W2 update (logits)" if panel == "A" else "",
        )
        ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    entries = [
        ("conflict — Game", "Game", summary["models"]["conflict"]["Game"]["gamma"]),
        ("conflict — Matched Neutral", "Matched Neutral", summary["models"]["conflict"]["Matched Neutral"]["gamma"]),
        ("conflict, W1=A — Game", "Game", summary["conflict_letter_stratification"]["W1_A"]["Game"]["gamma"]),
        ("conflict, W1=B–D — Game", "Game", summary["conflict_letter_stratification"]["W1_non_A"]["Game"]["gamma"]),
        ("no conflict — Game", "Game", summary["models"]["no_conflict"]["Game"]["gamma"]),
        ("no conflict — Matched Neutral", "Matched Neutral", summary["models"]["no_conflict"]["Matched Neutral"]["gamma"]),
        ("no conflict, W1=A — Game", "Game", summary["no_conflict_letter_stratification"]["W1_A"]["Game"]["gamma"]),
        ("no conflict, W1=B–D — Game", "Game", summary["no_conflict_letter_stratification"]["W1_non_A"]["Game"]["gamma"]),
    ]
    y_positions = np.arange(len(entries))[::-1]
    for y_pos, (label, condition, interval) in zip(y_positions, entries):
        color = condition_colors[condition]
        ax.errorbar(
            interval["mean"], y_pos,
            xerr=[[interval["mean"] - interval["ci_low"]], [interval["ci_high"] - interval["mean"]]],
            fmt="o", color=color, capsize=3,
        )
    ax.axvline(0, color="0.45", lw=0.9)
    ax.set_yticks(y_positions, [label for label, _, _ in entries])
    ax.set(
        title="C  Extra W1 effect beyond compression",
        xlabel="Additional W1-vs-others update, γ (logits)",
    )

    fig.suptitle("Does the evaluation-period update target W1 beyond generic compression?", fontsize=15)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=190, bbox_inches="tight")
    plt.close(fig)

    game_conflict = summary["models"]["conflict"]["Game"]
    neutral_conflict = summary["models"]["conflict"]["Matched Neutral"]
    decisive = summary["decisive_order_reversal"]["W1_below_W2"]
    print(json.dumps({
        "game_conflict_gamma": game_conflict["gamma"],
        "neutral_conflict_gamma": neutral_conflict["gamma"],
        "game_minus_neutral_gamma": summary["models"]["conflict"]["Game_minus_Neutral_gamma"],
        "game_conflict_r2": {
            "compression": game_conflict["compression_r2"],
            "plus_w1": game_conflict["compression_plus_w1_r2"],
        },
        "w1_below_w2": decisive,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=3000)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
