from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .semantic_mapping import displayed_argmax_to_semantic_indices


LETTERS = "ABCD"
CONDITIONS = ("game", "neutral")
CARRIER_READOUTS = np.asarray(list(range(3, 64, 4)))
NATURAL_INDEX = 0
IDENTITY_INDEX = 1
PROJECT_INDEX = 2


def _entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=-1, keepdims=True)
    return -(probability * np.log(probability + 1e-30)).sum(axis=-1)


def _interval(values: np.ndarray, boot: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    sampled = values[boot].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci": [float(low), float(high)],
        "n": int(len(values)),
    }


def _load(
    results_path: Path,
    second_mapping_path: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
) -> dict[str, Any]:
    with np.load(results_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not arrays["completed"].all():
        raise ValueError("All-four projection result is incomplete")
    qids = arrays["question_ids"].astype(str).tolist()
    second_rows = {
        row["question_id"]: row
        for row in json.loads(second_mapping_path.read_text())["rows"]
    }
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped_baseline = json.loads(remapped_baseline_path.read_text())["results"]
    displayed = arrays["logits"].astype(np.float64)
    if displayed.shape[:2] != (2, 3):
        raise ValueError(
            "Corrected analysis requires natural, identity_kv, and "
            f"project_centered modes; got logits shape {displayed.shape}"
        )
    semantic = np.empty_like(displayed)
    for qi, qid in enumerate(qids):
        mapping = second_rows[qid]["new_to_original"]
        for new_index, new_letter in enumerate(LETTERS):
            original_index = LETTERS.index(mapping[new_letter])
            semantic[:, :, qi, original_index] = displayed[:, :, qi, new_index]
    w1 = np.asarray([LETTERS.index(baseline[qid]["answer"]) for qid in qids])
    w2 = np.asarray(
        [
            LETTERS.index(remapped_baseline[qid]["answer_original_content"])
            for qid in qids
        ]
    )
    rows = np.arange(len(qids))
    choice = displayed_argmax_to_semantic_indices(
        displayed, [second_rows[qid] for qid in qids]
    )
    w1_logit = semantic[:, :, rows, w1]
    w2_logit = semantic[:, :, rows, w2]
    natural_choice = choice[:, NATURAL_INDEX]
    return {
        "arrays": arrays,
        "qids": qids,
        "split": arrays["split"].astype(str),
        "semantic": semantic,
        "choice": choice,
        "w1": w1,
        "w2": w2,
        "conflict": w1 != w2,
        "w1_choice": choice == w1[None, None, :],
        "w2_choice": choice == w2[None, None, :],
        "switch": choice != w1[None, None, :],
        "answer_change": choice != natural_choice[:, None, :],
        "margin": w1_logit - w2_logit,
        "centered_w1": w1_logit - semantic.mean(axis=-1),
        "entropy": _entropy(semantic),
        "spread": semantic.max(axis=-1) - semantic.min(axis=-1),
    }


def _metrics(data: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "w1_choice_pp": data["w1_choice"].astype(float) * 100,
        "w2_choice_pp": data["w2_choice"].astype(float) * 100,
        "switch_pp": data["switch"].astype(float) * 100,
        "answer_change_pp": data["answer_change"].astype(float) * 100,
        "w1_minus_w2_margin": data["margin"],
        "w1_centered_evidence": data["centered_w1"],
        "ad_entropy": data["entropy"],
        "ad_spread": data["spread"],
    }


def _summarize(
    data: dict[str, Any], split_name: str, seed: int, draws: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    split_mask = data["split"] == split_name
    subsets = {
        "all": split_mask,
        "conflict": split_mask & data["conflict"],
        "no_conflict": split_mask & ~data["conflict"],
    }
    metrics = _metrics(data)
    result: dict[str, Any] = {}
    for subset_name, mask in subsets.items():
        indices = np.flatnonzero(mask)
        boot = rng.integers(0, len(indices), size=(draws, len(indices)))
        subset: dict[str, Any] = {"n": int(len(indices)), "natural": {}, "effect": {}}
        for metric_name, values in metrics.items():
            subset["natural"][metric_name] = {
                condition: _interval(values[ci, 0, indices], boot)
                for ci, condition in enumerate(CONDITIONS)
            }
            game = values[0, PROJECT_INDEX, indices] - values[0, NATURAL_INDEX, indices]
            neutral = values[1, PROJECT_INDEX, indices] - values[1, NATURAL_INDEX, indices]
            subset["effect"][metric_name] = {
                "game": _interval(game, boot),
                "neutral": _interval(neutral, boot),
                "game_minus_neutral_interaction": _interval(game - neutral, boot),
            }
        result[subset_name] = subset
    return result


def _dose(data: dict[str, Any], split_name: str, seed: int, draws: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    arrays = data["arrays"]
    mask = data["split"] == split_name
    layers = CARRIER_READOUTS - 1
    result: dict[str, Any] = {}
    for ci, condition in enumerate(CONDITIONS):
        pre = arrays["pre_score"][ci, PROJECT_INDEX, mask][:, layers, :].astype(np.float64)
        post = arrays["post_score"][ci, PROJECT_INDEX, mask][:, layers, :].astype(np.float64)
        dose = arrays["dose_l2"][ci, PROJECT_INDEX, mask][:, layers, :].astype(np.float64)
        norm = arrays["residual_norm"][ci, PROJECT_INDEX, mask][:, layers, :].astype(np.float64)
        relative = dose / np.maximum(norm, 1e-12) * 100
        # Average options within question, then bootstrap questions.
        relative_question = relative.mean(axis=-1)
        pre_abs_question = np.abs(pre).mean(axis=-1)
        post_abs_question = np.abs(post).mean(axis=-1)
        boot = rng.integers(0, len(relative_question), size=(draws, len(relative_question)))
        relative_boot = relative_question[boot].mean(axis=1)
        result[condition] = {
            "readouts": CARRIER_READOUTS.tolist(),
            "relative_l2_dose_percent_mean": relative_question.mean(axis=0).tolist(),
            "relative_l2_dose_percent_ci_low": np.quantile(relative_boot, 0.025, axis=0).tolist(),
            "relative_l2_dose_percent_ci_high": np.quantile(relative_boot, 0.975, axis=0).tolist(),
            "mean_absolute_pre_score": pre_abs_question.mean(axis=0).tolist(),
            "mean_absolute_post_score": post_abs_question.mean(axis=0).tolist(),
            "max_mean_relative_l2_dose_percent": float(relative_question.mean(axis=0).max()),
            "max_absolute_post_score": float(np.abs(post).max()),
            "mean_absolute_post_score_overall": float(np.abs(post).mean()),
        }
    return result


def _identity_validation(data: dict[str, Any]) -> dict[str, Any]:
    arrays = data["arrays"]
    logits = arrays["logits"].astype(np.float64)
    delta = logits[:, IDENTITY_INDEX] - logits[:, NATURAL_INDEX]
    natural_choice = logits[:, NATURAL_INDEX].argmax(axis=-1)
    identity_choice = logits[:, IDENTITY_INDEX].argmax(axis=-1)
    result: dict[str, Any] = {
        "max_absolute_ad_logit_difference": float(np.max(np.abs(delta))),
        "mean_absolute_ad_logit_difference": float(np.mean(np.abs(delta))),
        "overall_choice_change_rate": float(np.mean(natural_choice != identity_choice)),
    }
    for split_name in ("discovery", "confirmation"):
        mask = data["split"] == split_name
        result[split_name] = {
            condition: {
                "max_absolute_ad_logit_difference": float(
                    np.max(np.abs(delta[ci, mask]))
                ),
                "choice_change_rate": float(
                    np.mean(natural_choice[ci, mask] != identity_choice[ci, mask])
                ),
            }
            for ci, condition in enumerate(CONDITIONS)
        }
    return result


def _plot(
    path: Path,
    discovery: dict[str, Any],
    confirmation: dict[str, Any],
    confirmation_dose: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    colors = {"game": "#2F8EF4", "neutral": "#F08032"}
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))

    continuous = (
        ("w1_minus_w2_margin", "W1−W2 margin"),
        ("w1_centered_evidence", "Centered W1"),
        ("ad_spread", "A–D spread"),
    )
    for ci, condition in enumerate(CONDITIONS):
        means, lows, highs = [], [], []
        for metric, _label in continuous:
            value = confirmation["conflict"]["effect"][metric][condition]
            means.append(value["mean"])
            lows.append(value["mean"] - value["ci"][0])
            highs.append(value["ci"][1] - value["mean"])
        x = np.arange(len(continuous)) + (-0.08 if ci == 0 else 0.08)
        axes[0, 0].errorbar(x, means, yerr=[lows, highs], fmt="o", capsize=4,
                            color=colors[condition], label=condition.title())
    axes[0, 0].axhline(0, color="#888888", linestyle="--", linewidth=1)
    axes[0, 0].set_xticks(np.arange(len(continuous)), [x[1] for x in continuous])
    axes[0, 0].set_ylabel("Change from natural (logit units)")
    axes[0, 0].set_title("A  Held-out conflict trials: continuous effects", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", alpha=0.18)

    behavioral = (
        ("w1_choice_pp", "Choose W1"),
        ("w2_choice_pp", "Choose W2"),
        ("switch_pp", "Switch from W1"),
        ("answer_change_pp", "Any answer change"),
    )
    for ci, condition in enumerate(CONDITIONS):
        means, lows, highs = [], [], []
        for metric, _label in behavioral:
            value = confirmation["conflict"]["effect"][metric][condition]
            means.append(value["mean"])
            lows.append(value["mean"] - value["ci"][0])
            highs.append(value["ci"][1] - value["mean"])
        x = np.arange(len(behavioral)) + (-0.08 if ci == 0 else 0.08)
        axes[0, 1].errorbar(x, means, yerr=[lows, highs], fmt="o", capsize=4,
                            color=colors[condition], label=condition.title())
    axes[0, 1].axhline(0, color="#888888", linestyle="--", linewidth=1)
    axes[0, 1].set_xticks(np.arange(len(behavioral)), [x[1] for x in behavioral], rotation=12)
    axes[0, 1].set_ylabel("Change from natural (percentage points)")
    axes[0, 1].set_title("B  Held-out conflict trials: behavioral effects", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(axis="y", alpha=0.18)

    replication_metrics = (
        ("w1_minus_w2_margin", "W1−W2 margin", 1.0),
        ("w1_choice_pp", "W1 choice ÷ 100", 0.01),
        ("switch_pp", "Switching ÷ 100", 0.01),
    )
    for si, (split_name, values) in enumerate(
        (("Discovery", discovery), ("Confirmation", confirmation))
    ):
        means, lows, highs = [], [], []
        for metric, _label, scale in replication_metrics:
            value = values["conflict"]["effect"][metric]["game_minus_neutral_interaction"]
            means.append(value["mean"] * scale)
            lows.append((value["mean"] - value["ci"][0]) * scale)
            highs.append((value["ci"][1] - value["mean"]) * scale)
        x = np.arange(len(replication_metrics)) + (-0.08 if si == 0 else 0.08)
        axes[1, 0].errorbar(x, means, yerr=[lows, highs], fmt="o", capsize=4,
                            label=split_name)
    axes[1, 0].axhline(0, color="#888888", linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(np.arange(len(replication_metrics)), [x[1] for x in replication_metrics])
    axes[1, 0].set_ylabel("Game-minus-Neutral interaction")
    axes[1, 0].set_title("C  Conflict-trial interaction across frozen splits", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.18)

    readouts = np.asarray(confirmation_dose["game"]["readouts"])
    # The edit occurs before Game and Neutral diverge, so these trajectories
    # should overlap. Plot both to make that identity visible and auditable.
    for condition in CONDITIONS:
        values = confirmation_dose[condition]
        mean = np.asarray(values["relative_l2_dose_percent_mean"])
        low = np.asarray(values["relative_l2_dose_percent_ci_low"])
        high = np.asarray(values["relative_l2_dose_percent_ci_high"])
        axes[1, 1].plot(readouts, mean, color=colors[condition], label=condition.title())
        axes[1, 1].fill_between(readouts, low, high, color=colors[condition], alpha=0.16)
    axes[1, 1].set_xlabel("Post-block residual readout")
    axes[1, 1].set_ylabel("Mean residual L2 changed (%)")
    axes[1, 1].set_title("D  Size of the four-option K/V-carrier projection", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", alpha=0.18)

    fig.suptitle("Centered candidate-value projection from all four option K/V memories", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(
    results_path: Path,
    second_mapping_path: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
    output_dir: Path,
    figure_path: Path,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    data = _load(results_path, second_mapping_path, baseline_path, remapped_baseline_path)
    discovery = _summarize(data, "discovery", seed, draws)
    confirmation = _summarize(data, "confirmation", seed + 1, draws)
    dose = {
        "discovery": _dose(data, "discovery", seed + 2, draws),
        "confirmation": _dose(data, "confirmation", seed + 3, draws),
    }
    identity = _identity_validation(data)
    primary = confirmation["conflict"]["effect"]["w1_choice_pp"]
    summary = {
        "design": (
            "After subtracting each displayed letter's discovery-set mean, "
            "project the candidate-value direction out of all four first-presentation "
            "option-newline ordinary-attention K/V carrier states immediately before "
            "every ordinary-attention block (blocks 4, 8, ..., 64)."
        ),
        "discovery": discovery,
        "confirmation": confirmation,
        "dose": dose,
        "identity_kv_validation": identity,
        "primary_confirmation_conflict_w1_choice": primary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (output_dir / "effects.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "subset", "metric", "effect", "mean", "ci_low", "ci_high", "n"])
        for split_name, split_values in (("discovery", discovery), ("confirmation", confirmation)):
            for subset_name, subset in split_values.items():
                for metric, values in subset["effect"].items():
                    for effect, value in values.items():
                        writer.writerow([split_name, subset_name, metric, effect, value["mean"], value["ci"][0], value["ci"][1], value["n"]])
    _plot(figure_path, discovery, confirmation, dose["confirmation"])

    pm = confirmation["conflict"]["effect"]["w1_minus_w2_margin"]
    ps = confirmation["conflict"]["effect"]["switch_pp"]
    pe = confirmation["conflict"]["effect"]["ad_entropy"]
    report = [
        "# Corrected centered all-four option-newline candidate-value K/V-carrier projection",
        "",
        "## Bottom line",
        "",
        "The earlier paradoxical all-four result was an intervention artifact. With the affine probe geometry corrected, all 16 ordinary-attention blocks included, and an exact zero-delta identity control, removing this one-dimensional option-value coordinate does not robustly explain preferential Game avoidance of W1. The held-out logit effect is weakly consistent with W1 reinstatement in Game, but its Game-minus-Neutral interaction includes zero and the effect is much smaller in discovery.",
        "",
        "## Design",
        "",
        summary["design"],
        "Natural, identity-K/V, and centered-projection executions used all 500 canonical remapped questions, the frozen 251/249 split, and exact historical physical batches.",
        "",
        "## Identity-path validation",
        "",
        f"The same hook path with an exactly zero projected-minus-unprojected K/V delta changed choices on {100 * identity['overall_choice_change_rate']:.3f}% of condition-question trials; maximum absolute A--D logit change was {identity['max_absolute_ad_logit_difference']:.6f}.",
        "",
        "## Held-out conflict result",
        "",
        f"On {confirmation['conflict']['n']} confirmation conflict questions, W1 choice changed by {primary['game']['mean']:+.1f} [{primary['game']['ci'][0]:+.1f}, {primary['game']['ci'][1]:+.1f}] points in Game and {primary['neutral']['mean']:+.1f} [{primary['neutral']['ci'][0]:+.1f}, {primary['neutral']['ci'][1]:+.1f}] in Neutral. The Game-minus-Neutral interaction was {primary['game_minus_neutral_interaction']['mean']:+.1f} [{primary['game_minus_neutral_interaction']['ci'][0]:+.1f}, {primary['game_minus_neutral_interaction']['ci'][1]:+.1f}] points.",
        "",
        f"The W1-minus-W2 margin changed by {pm['game']['mean']:+.3f} [{pm['game']['ci'][0]:+.3f}, {pm['game']['ci'][1]:+.3f}] logits in Game and {pm['neutral']['mean']:+.3f} [{pm['neutral']['ci'][0]:+.3f}, {pm['neutral']['ci'][1]:+.3f}] in Neutral; interaction {pm['game_minus_neutral_interaction']['mean']:+.3f} [{pm['game_minus_neutral_interaction']['ci'][0]:+.3f}, {pm['game_minus_neutral_interaction']['ci'][1]:+.3f}].",
        "",
        f"Switching changed by {ps['game']['mean']:+.1f} [{ps['game']['ci'][0]:+.1f}, {ps['game']['ci'][1]:+.1f}] points in Game and {ps['neutral']['mean']:+.1f} [{ps['neutral']['ci'][0]:+.1f}, {ps['neutral']['ci'][1]:+.1f}] in Neutral. A--D entropy changed by {pe['game']['mean']:+.3f} in Game and {pe['neutral']['mean']:+.3f} in Neutral.",
        "",
        "## Interpretation",
        "",
        "The held-out Game W1-minus-W2 increase is in the direction predicted if the coordinate contributes to suppressing W1, but it is only +0.020 logits, its condition interaction includes zero, and discovery's corresponding interaction is near zero. Discrete W1 choice also has a wide interval. The intervention changes some answers, so the coordinate participates in ordinary option scoring; it is not established as the semantic selectedness-binding mechanism responsible for Game's preferential revision.",
        "",
        "The previous large, counterintuitive W1 effects are withdrawn: that runner removed static displayed-letter structure and absolute cached K/V reconstruction error as well as the intended centered score. Neither contamination exists in this corrected run.",
        "",
        "## Validation and scope",
        "",
        f"The maximum held-out mean residual dose at any readout was {dose['confirmation']['game']['max_mean_relative_l2_dose_percent']:.2f}% of residual L2. Mean absolute post-projection probe score was {dose['confirmation']['game']['mean_absolute_post_score_overall']:.4f}; the maximum absolute post-score was {dose['confirmation']['game']['max_absolute_post_score']:.4f}.",
        "",
        "Discovery replication, no-conflict results, natural rates, and all secondary metrics are in `summary.json` and `effects.csv`.",
        "",
        f"Canonical figure: `{figure_path}`.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--second-mapping", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    analyze(args.results, args.second_mapping, args.baseline, args.remapped_baseline, args.output_dir, args.figure, args.draws, args.seed)


if __name__ == "__main__":
    main()
