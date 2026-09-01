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
MODES = (
    "natural",
    "identity_kv",
    "score_only",
    "decision_letter_only",
    "joint_score_and_letter",
)
NATURAL = 0
IDENTITY = 1
SCORE = 2
LETTER = 3
JOINT = 4


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def _interval(values: np.ndarray, boot: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    samples = values[boot].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
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
        raise ValueError("Joint factorial result is incomplete")
    if arrays["logits"].shape[:2] != (2, 5):
        raise ValueError(f"Unexpected logits shape: {arrays['logits'].shape}")
    qids = arrays["question_ids"].astype(str).tolist()
    second_rows = {
        row["question_id"]: row
        for row in json.loads(second_mapping_path.read_text())["rows"]
    }
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped = json.loads(remapped_baseline_path.read_text())["results"]
    displayed = arrays["logits"].astype(np.float64)
    semantic = np.empty_like(displayed)
    for qi, qid in enumerate(qids):
        mapping = second_rows[qid]["new_to_original"]
        for new_index, new_letter in enumerate(LETTERS):
            original_index = LETTERS.index(mapping[new_letter])
            semantic[:, :, qi, original_index] = displayed[:, :, qi, new_index]
    w1 = np.asarray([LETTERS.index(baseline[qid]["answer"]) for qid in qids])
    w2 = np.asarray(
        [
            LETTERS.index(remapped[qid]["answer_original_content"])
            for qid in qids
        ]
    )
    rows = np.arange(len(qids))
    probability = _softmax(semantic)
    choice = displayed_argmax_to_semantic_indices(
        displayed, [second_rows[qid] for qid in qids]
    )
    w1_logit = semantic[:, :, rows, w1]
    w2_logit = semantic[:, :, rows, w2]
    w1_probability = probability[:, :, rows, w1]
    w2_probability = probability[:, :, rows, w2]
    entropy = -(probability * np.log2(probability + 1e-30)).sum(axis=-1)
    return {
        "arrays": arrays,
        "qids": qids,
        "split": arrays["split"].astype(str),
        "semantic": semantic,
        "probability": probability,
        "choice": choice,
        "w1": w1,
        "w2": w2,
        "conflict": w1 != w2,
        "metrics": {
            "w1_choice_pp": (choice == w1[None, None, :]).astype(float) * 100,
            "w2_choice_pp": (choice == w2[None, None, :]).astype(float) * 100,
            "switch_pp": (choice != w1[None, None, :]).astype(float) * 100,
            "w1_minus_w2_margin": w1_logit - w2_logit,
            "w1_centered_evidence": w1_logit - semantic.mean(axis=-1),
            "w1_probability": w1_probability,
            "w2_probability": w2_probability,
            "ad_entropy_bits": entropy,
            "ad_spread": semantic.max(axis=-1) - semantic.min(axis=-1),
        },
    }


def _summarize(
    data: dict[str, Any], split_name: str, draws: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    split = data["split"] == split_name
    subsets = {
        "all": split,
        "conflict": split & data["conflict"],
        "no_conflict": split & ~data["conflict"],
    }
    result: dict[str, Any] = {}
    for subset_name, mask in subsets.items():
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        boot = rng.integers(0, len(indices), size=(draws, len(indices)))
        subset: dict[str, Any] = {"n": int(len(indices)), "metrics": {}}
        for metric_name, values in data["metrics"].items():
            metric: dict[str, Any] = {"natural": {}, "effects": {}}
            for ci, condition in enumerate(CONDITIONS):
                metric["natural"][condition] = _interval(
                    values[ci, NATURAL, indices], boot
                )
            effects: dict[int, np.ndarray] = {}
            for mode_index, mode in ((SCORE, "score_only"), (LETTER, "decision_letter_only"), (JOINT, "joint_score_and_letter")):
                game = values[0, mode_index, indices] - values[0, IDENTITY, indices]
                neutral = values[1, mode_index, indices] - values[1, IDENTITY, indices]
                effects[mode_index] = np.stack([game, neutral])
                metric["effects"][mode] = {
                    "game": _interval(game, boot),
                    "neutral": _interval(neutral, boot),
                    "game_minus_neutral": _interval(game - neutral, boot),
                }
            score = effects[SCORE]
            letter = effects[LETTER]
            joint = effects[JOINT]
            metric["factorial_interaction"] = {
                "game": _interval(joint[0] - score[0] - letter[0], boot),
                "neutral": _interval(joint[1] - score[1] - letter[1], boot),
                "game_minus_neutral": _interval(
                    (joint[0] - score[0] - letter[0])
                    - (joint[1] - score[1] - letter[1]),
                    boot,
                ),
            }
            subset["metrics"][metric_name] = metric
        result[subset_name] = subset
    return result


def _validation(data: dict[str, Any], prior_score_path: Path | None) -> dict[str, Any]:
    arrays = data["arrays"]
    logits = arrays["logits"].astype(np.float64)
    identity_delta = logits[:, IDENTITY] - logits[:, NATURAL]
    target = np.asarray(list(range(2, 64, 4)))
    score_post = arrays["post_score"][:, [SCORE, JOINT]][:, :, :, target]
    letter_post = arrays["decision_post_ad_norm"][:, [LETTER, JOINT]][:, :, :, target]
    result: dict[str, Any] = {
        "identity_max_abs_ad_logit_error": float(np.max(np.abs(identity_delta))),
        "identity_choice_changes": int(
            np.sum(logits[:, IDENTITY].argmax(axis=-1) != logits[:, NATURAL].argmax(axis=-1))
        ),
        "natural_trusted_max_abs_ad_logit_error": float(
            np.nanmax(arrays["trusted_max_abs_error"])
        ),
        "natural_trusted_choice_agreement": float(
            arrays["trusted_choice_match"].mean()
        ),
        "first_decision_baseline_choice_agreement": float(
            arrays["first_decision_matches_baseline"].mean()
        ),
        "max_abs_option_score_after_score_lesions": float(np.max(np.abs(score_post))),
        "max_decision_ad_norm_after_letter_lesions": float(np.max(letter_post)),
        "all_outputs_finite": bool(
            np.all(np.isfinite(logits))
            and np.all(np.isfinite(score_post))
            and np.all(np.isfinite(letter_post))
        ),
    }
    if prior_score_path is not None:
        with np.load(prior_score_path, allow_pickle=False) as loaded:
            old = {key: loaded[key] for key in loaded.files}
        if old["question_ids"].astype(str).tolist() != data["qids"]:
            raise ValueError("Prior score-only result uses another question order")
        natural_delta = logits[:, NATURAL] - old["logits"][:, 0].astype(np.float64)
        result["prior_natural_max_abs_ad_logit_error"] = float(
            np.max(np.abs(natural_delta))
        )
        result["prior_natural_choice_changes"] = int(
            np.sum(
                logits[:, NATURAL].argmax(axis=-1)
                != old["logits"][:, 0].argmax(axis=-1)
            )
        )
        delta = logits[:, SCORE] - old["logits"][:, 2].astype(np.float64)
        result["prior_score_only_max_abs_ad_logit_error"] = float(
            np.max(np.abs(delta))
        )
        result["prior_score_only_choice_changes"] = int(
            np.sum(
                logits[:, SCORE].argmax(axis=-1)
                != old["logits"][:, 2].argmax(axis=-1)
            )
        )
    return result


def _matched_first_decision_sensitivity(
    data: dict[str, Any], draws: int, seed: int
) -> dict[str, Any]:
    matched = data["arrays"]["first_decision_matches_baseline"].astype(bool)
    result: dict[str, Any] = {}
    for split_offset, split_name in enumerate(("discovery", "confirmation")):
        mask = (data["split"] == split_name) & data["conflict"] & matched
        indices = np.flatnonzero(mask)
        rng = np.random.default_rng(seed + split_offset)
        boot = rng.integers(0, len(indices), size=(draws, len(indices)))
        split: dict[str, Any] = {"n": int(len(indices)), "metrics": {}}
        for metric_name in ("w1_choice_pp", "w1_minus_w2_margin"):
            values = data["metrics"][metric_name]
            metric: dict[str, Any] = {}
            for mode_index, mode in (
                (SCORE, "score_only"),
                (LETTER, "decision_letter_only"),
                (JOINT, "joint_score_and_letter"),
            ):
                game = values[0, mode_index, indices] - values[0, IDENTITY, indices]
                neutral = values[1, mode_index, indices] - values[1, IDENTITY, indices]
                metric[mode] = _interval(game - neutral, boot)
            split["metrics"][metric_name] = metric
        result[split_name] = split
    return result


def _plot(
    path: Path,
    discovery: dict[str, Any],
    confirmation: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    modes = ("score_only", "decision_letter_only", "joint_score_and_letter")
    labels = ("Score only", "Decision letter only", "Both")
    colors = {"game": "#2b8cbe", "neutral": "#f17c32"}
    metrics = (
        ("w1_choice_pp", "Effect on choosing W1 (percentage points)"),
        ("w1_minus_w2_margin", "Effect on W1 − W2 logit margin"),
    )
    splits = (("Discovery", discovery), ("Confirmation", confirmation))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    x = np.arange(len(modes), dtype=float)
    offsets = {"game": -0.10, "neutral": 0.10}
    for row, (split_label, split) in enumerate(splits):
        for column, (metric_name, ylabel) in enumerate(metrics):
            axis = axes[row, column]
            metric = split["conflict"]["metrics"][metric_name]["effects"]
            for condition in CONDITIONS:
                means = np.asarray([metric[mode][condition]["mean"] for mode in modes])
                lows = np.asarray([metric[mode][condition]["ci"][0] for mode in modes])
                highs = np.asarray([metric[mode][condition]["ci"][1] for mode in modes])
                axis.errorbar(
                    x + offsets[condition],
                    means,
                    yerr=np.vstack([means - lows, highs - means]),
                    fmt="o",
                    markersize=7,
                    capsize=5,
                    linewidth=1.8,
                    color=colors[condition],
                    label=condition.title(),
                )
            axis.axhline(0, color="#666666", linewidth=1, linestyle="--")
            axis.set_xticks(x, labels, rotation=12, ha="right")
            axis.set_ylabel(ylabel)
            axis.set_title(f"{split_label} conflict trials (n={split['conflict']['n']})")
            axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Removing candidate score, first-decision letter identity, or both"
    )
    fig.tight_layout()
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
    prior_score_path: Path | None,
    draws: int,
    seed: int,
) -> None:
    data = _load(results_path, second_mapping_path, baseline_path, remapped_baseline_path)
    discovery = _summarize(data, "discovery", draws, seed)
    confirmation = _summarize(data, "confirmation", draws, seed + 1)
    validation = _validation(data, prior_score_path)
    matched_sensitivity = _matched_first_decision_sensitivity(data, draws, seed + 100)
    summary = {
        "definitions": {
            "effect": "Intervention minus same-path zero-delta identity K/V control.",
            "game_minus_neutral": "Game causal effect minus Neutral causal effect.",
            "factorial_interaction": "Joint effect minus score-only effect minus decision-letter-only effect.",
            "conflict": "W1 differs from the fresh remapped Baseline winner W2.",
        },
        "validation": validation,
        "matched_first_decision_sensitivity": matched_sensitivity,
        "discovery": discovery,
        "confirmation": confirmation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    rows: list[dict[str, Any]] = []
    for split_name, split_summary in (("discovery", discovery), ("confirmation", confirmation)):
        for subset_name, subset in split_summary.items():
            for metric_name, metric in subset["metrics"].items():
                for mode in ("score_only", "decision_letter_only", "joint_score_and_letter"):
                    for contrast in ("game", "neutral", "game_minus_neutral"):
                        item = metric["effects"][mode][contrast]
                        rows.append(
                            {
                                "split": split_name,
                                "subset": subset_name,
                                "n": subset["n"],
                                "metric": metric_name,
                                "mode": mode,
                                "contrast": contrast,
                                "mean": item["mean"],
                                "ci_low": item["ci"][0],
                                "ci_high": item["ci"][1],
                            }
                        )
    with (output_dir / "effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def cell(split: dict[str, Any], subset: str, metric: str, mode: str, contrast: str) -> str:
        item = split[subset]["metrics"][metric]["effects"][mode][contrast]
        return f"{item['mean']:+.3f} [{item['ci'][0]:+.3f}, {item['ci'][1]:+.3f}]"

    def interaction_cell(
        split: dict[str, Any], subset: str, metric: str, contrast: str
    ) -> str:
        item = split[subset]["metrics"][metric]["factorial_interaction"][contrast]
        return f"{item['mean']:+.3f} [{item['ci'][0]:+.3f}, {item['ci'][1]:+.3f}]"

    lines = [
        "# Joint candidate-score and first-decision-letter lesion",
        "",
        "## Bottom line",
        "",
        "The redundancy hypothesis is **not supported**. Removing the candidate-score coordinate and the first-decision letter-identity subspace together did not produce a larger or more reliable recovery of W1 than either lesion alone.",
        "",
        f"On held-out conflict trials, decision-letter removal produced the predicted discrete interaction ({cell(confirmation, 'conflict', 'w1_choice_pp', 'decision_letter_only', 'game_minus_neutral')} percentage points), driven by Game choosing W1 more often. But the discovery interaction was {cell(discovery, 'conflict', 'w1_choice_pp', 'decision_letter_only', 'game_minus_neutral')}, and the held-out continuous W1-minus-W2 interaction was only {cell(confirmation, 'conflict', 'w1_minus_w2_margin', 'decision_letter_only', 'game_minus_neutral')} logits. Thus the attractive held-out choice effect does not replicate across the frozen splits or in the continuous margin.",
        "",
        f"For scale, natural held-out W1 choice was {confirmation['conflict']['metrics']['w1_choice_pp']['natural']['game']['mean']:.1f}% in Game and {confirmation['conflict']['metrics']['w1_choice_pp']['natural']['neutral']['mean']:.1f}% in Neutral, a {confirmation['conflict']['metrics']['w1_choice_pp']['natural']['game']['mean'] - confirmation['conflict']['metrics']['w1_choice_pp']['natural']['neutral']['mean']:.1f}-point gap. The letter-only intervention closes only {cell(confirmation, 'conflict', 'w1_choice_pp', 'decision_letter_only', 'game_minus_neutral')} points of that gap (about 17% at the point estimate).",
        "",
        f"The joint lesion was weaker, not stronger: its held-out conflict-trial Game-minus-Neutral W1-choice effect was {cell(confirmation, 'conflict', 'w1_choice_pp', 'joint_score_and_letter', 'game_minus_neutral')} points, and Game itself changed by {cell(confirmation, 'conflict', 'w1_choice_pp', 'joint_score_and_letter', 'game')} points. The Game factorial interaction was antagonistic ({interaction_cell(confirmation, 'conflict', 'w1_choice_pp', 'game')} points), rather than the positive synergy expected if the two coordinates were redundant routes whose joint removal exposed the mechanism.",
        "",
        "The experiment therefore leaves the core binding problem unresolved. A one-dimensional option-value coordinate is readable at the option newline, and centered A-D identity is present at the first-decision position, but jointly removing those two decoded coordinates does not causally account for preferential Game revision.",
        "",
        "Effects are measured against the exact zero-delta identity-K/V path. The primary endpoint is held-out conflict-trial recovery of W1 in Game relative to Neutral.",
        "",
        "## Validation",
        "",
        f"- Natural-versus-identity maximum A–D logit difference: {validation['identity_max_abs_ad_logit_error']:.6g}.",
        f"- Natural-versus-identity choice changes: {validation['identity_choice_changes']}.",
        f"- Natural trusted-choice agreement: {validation['natural_trusted_choice_agreement']:.2%}.",
        f"- First-decision Baseline-choice agreement: {validation['first_decision_baseline_choice_agreement']:.2%}.",
        f"- Maximum residual candidate score after score removal: {validation['max_abs_option_score_after_score_lesions']:.6g}.",
        f"- Maximum residual A–D norm after decision-letter removal: {validation['max_decision_ad_norm_after_letter_lesions']:.6g}.",
        f"- Same-host prior natural maximum A–D logit difference: {validation.get('prior_natural_max_abs_ad_logit_error', float('nan')):.6g}; choice changes: {validation.get('prior_natural_choice_changes', -1)}.",
        f"- Same-host prior score-only maximum A–D logit difference: {validation.get('prior_score_only_max_abs_ad_logit_error', float('nan')):.6g}; choice changes: {validation.get('prior_score_only_choice_changes', -1)}.",
        "",
        "The 98.8% trusted-choice figure compares against an older run from another host and reflects known BF16 host drift. The matched same-host natural and score-only controls reproduced exactly and are the relevant numerical validation for the causal contrasts.",
        "",
        f"Excluding the eight questions whose current first decision differed from the older cross-host Baseline does not change the conclusion. Among {matched_sensitivity['confirmation']['n']} held-out matched conflict questions, the letter-only W1-choice interaction is {matched_sensitivity['confirmation']['metrics']['w1_choice_pp']['decision_letter_only']['mean']:+.3f} [{matched_sensitivity['confirmation']['metrics']['w1_choice_pp']['decision_letter_only']['ci'][0]:+.3f}, {matched_sensitivity['confirmation']['metrics']['w1_choice_pp']['decision_letter_only']['ci'][1]:+.3f}] points, versus {matched_sensitivity['discovery']['metrics']['w1_choice_pp']['decision_letter_only']['mean']:+.3f} [{matched_sensitivity['discovery']['metrics']['w1_choice_pp']['decision_letter_only']['ci'][0]:+.3f}, {matched_sensitivity['discovery']['metrics']['w1_choice_pp']['decision_letter_only']['ci'][1]:+.3f}] in discovery.",
        "",
        "## Held-out confirmation",
        "",
        f"Conflict questions: **{confirmation['conflict']['n']}**; no-conflict questions: **{confirmation['no_conflict']['n']}**.",
        "",
        "| Intervention | Game W1 choice | Neutral W1 choice | Game−Neutral W1 choice | Game−Neutral W1−W2 margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, label in (("score_only", "Score only"), ("decision_letter_only", "Decision letter only"), ("joint_score_and_letter", "Both")):
        lines.append(
            f"| {label} | {cell(confirmation, 'conflict', 'w1_choice_pp', mode, 'game')} | {cell(confirmation, 'conflict', 'w1_choice_pp', mode, 'neutral')} | {cell(confirmation, 'conflict', 'w1_choice_pp', mode, 'game_minus_neutral')} | {cell(confirmation, 'conflict', 'w1_minus_w2_margin', mode, 'game_minus_neutral')} |"
        )
    lines.extend(
        [
            "",
            "The complete machine-readable summary contains all/conflict/no-conflict results for switching, W1 and W2 choice, probabilities, margins, entropy, spread, and the factorial interaction on both frozen splits.",
            "",
            f"Canonical figure: `{figure_path}`.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    _plot(figure_path, discovery, confirmation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--second-mapping", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--prior-score", type=Path)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(
        args.results,
        args.second_mapping,
        args.baseline,
        args.remapped_baseline,
        args.output_dir,
        args.figure,
        args.prior_score,
        args.draws,
        args.seed,
    )


if __name__ == "__main__":
    main()
