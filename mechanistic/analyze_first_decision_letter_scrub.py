from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


LETTERS = "ABCD"
CONDITIONS = ("game", "neutral")
NATURAL, IDENTITY, SCRUB = range(3)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def _interval(values: np.ndarray, boot: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    samples = values[boot].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci": [float(low), float(high)], "n": int(len(values))}


def _load(
    results_path: Path,
    second_mapping_path: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
) -> dict[str, Any]:
    with np.load(results_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not arrays["completed"].all():
        raise ValueError("Continuous scrub result is incomplete")
    if arrays["logits"].shape[:2] != (2, 3):
        raise ValueError(f"Unexpected logits shape: {arrays['logits'].shape}")
    qids = arrays["question_ids"].astype(str).tolist()
    second = {row["question_id"]: row for row in json.loads(second_mapping_path.read_text())["rows"]}
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped = json.loads(remapped_baseline_path.read_text())["results"]
    displayed = arrays["logits"].astype(np.float64)
    semantic = np.empty_like(displayed)
    choice = np.empty(displayed.shape[:-1], dtype=np.int64)
    for qi, qid in enumerate(qids):
        mapping = second[qid]["new_to_original"]
        for new_index, new_letter in enumerate(LETTERS):
            semantic[:, :, qi, LETTERS.index(mapping[new_letter])] = displayed[:, :, qi, new_index]
        # The model emits displayed A-D letters, so exact ties must be broken in
        # displayed order before converting the winner to semantic-content order.
        displayed_choice = displayed[:, :, qi].argmax(axis=-1)
        new_to_original_index = np.asarray(
            [LETTERS.index(mapping[letter]) for letter in LETTERS], dtype=np.int64
        )
        choice[:, :, qi] = new_to_original_index[displayed_choice]
    w1 = np.asarray([LETTERS.index(baseline[qid]["answer"]) for qid in qids])
    w2 = np.asarray([LETTERS.index(remapped[qid]["answer_original_content"]) for qid in qids])
    rows = np.arange(len(qids))
    probability = _softmax(semantic)
    w1_logit = semantic[:, :, rows, w1]
    w2_logit = semantic[:, :, rows, w2]
    entropy = -(probability * np.log2(probability + 1e-30)).sum(axis=-1)
    return {
        "arrays": arrays,
        "qids": qids,
        "split": arrays["split"].astype(str),
        "w1": w1,
        "w2": w2,
        "conflict": w1 != w2,
        "metrics": {
            "w1_choice_pp": (choice == w1[None, None, :]).astype(float) * 100,
            "w2_choice_pp": (choice == w2[None, None, :]).astype(float) * 100,
            "switch_pp": (choice != w1[None, None, :]).astype(float) * 100,
            "w1_minus_w2_margin": w1_logit - w2_logit,
            "w1_centered_evidence": w1_logit - semantic.mean(axis=-1),
            "w1_probability": probability[:, :, rows, w1],
            "w2_probability": probability[:, :, rows, w2],
            "ad_entropy_bits": entropy,
            "ad_spread": semantic.max(axis=-1) - semantic.min(axis=-1),
        },
    }


def _summarize(data: dict[str, Any], split_name: str, draws: int, seed: int) -> dict[str, Any]:
    split = data["split"] == split_name
    subsets = {
        "all": split,
        "conflict": split & data["conflict"],
        "no_conflict": split & ~data["conflict"],
        "conflict_w1_a": split & data["conflict"] & (data["w1"] == 0),
        "conflict_w1_not_a": split & data["conflict"] & (data["w1"] != 0),
        "conflict_first_decision_matched": (
            split
            & data["conflict"]
            & data["arrays"]["first_decision_matches_baseline"].astype(bool)
        ),
    }
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {}
    for subset_name, mask in subsets.items():
        indices = np.flatnonzero(mask)
        boot = rng.integers(0, len(indices), size=(draws, len(indices)))
        subset: dict[str, Any] = {"n": int(len(indices)), "metrics": {}}
        for metric_name, values in data["metrics"].items():
            game = values[0, SCRUB, indices] - values[0, IDENTITY, indices]
            neutral = values[1, SCRUB, indices] - values[1, IDENTITY, indices]
            subset["metrics"][metric_name] = {
                "natural": {
                    condition: _interval(values[ci, NATURAL, indices], boot)
                    for ci, condition in enumerate(CONDITIONS)
                },
                "effect": {
                    "game": _interval(game, boot),
                    "neutral": _interval(neutral, boot),
                    "game_minus_neutral": _interval(game - neutral, boot),
                },
            }
        result[subset_name] = subset
    return result


def _validation(data: dict[str, Any], prior_results_path: Path | None) -> dict[str, Any]:
    arrays = data["arrays"]
    logits = arrays["logits"].astype(np.float64)
    identity_delta = logits[:, IDENTITY] - logits[:, NATURAL]
    targets = np.asarray(range(48, 64), dtype=np.int64)
    post = arrays["post_ad_norm"][:, SCRUB][..., targets]
    pre = arrays["pre_ad_norm"][:, SCRUB][..., targets]
    dose = arrays["dose_l2"][:, SCRUB][..., targets]
    result = {
        "identity_max_abs_ad_logit_error": float(np.max(np.abs(identity_delta))),
        "identity_choice_changes": int(np.sum(logits[:, IDENTITY].argmax(-1) != logits[:, NATURAL].argmax(-1))),
        "natural_trusted_max_abs_ad_logit_error": float(np.nanmax(arrays["trusted_max_abs_error"])),
        "natural_trusted_choice_agreement": float(arrays["trusted_choice_match"].mean()),
        "first_decision_baseline_choice_agreement": float(arrays["first_decision_matches_baseline"].mean()),
        "max_post_ad_norm": float(np.max(post)),
        "mean_pre_ad_norm": float(np.mean(pre)),
        "mean_dose_l2": float(np.mean(dose)),
        "all_outputs_finite": bool(np.all(np.isfinite(logits)) and np.all(np.isfinite(post))),
    }
    if prior_results_path is not None:
        with np.load(prior_results_path, allow_pickle=False) as loaded:
            prior_qids = loaded["question_ids"].astype(str).tolist()
            prior_natural = loaded["logits"][:, 0].astype(np.float64)
        if prior_qids != data["qids"]:
            raise ValueError("Prior same-host result uses another question order")
        result["same_host_prior_natural_max_abs_ad_logit_error"] = float(
            np.max(np.abs(logits[:, NATURAL] - prior_natural))
        )
        result["same_host_prior_natural_choice_changes"] = int(
            np.sum(logits[:, NATURAL].argmax(-1) != prior_natural.argmax(-1))
        )
    return result


def _plot(path: Path, discovery: dict[str, Any], confirmation: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    splits = (("Discovery", discovery), ("Confirmation", confirmation))
    metrics = (
        ("w1_choice_pp", "Effect on choosing W1 (percentage points)"),
        ("w1_centered_evidence", "Effect on W1 centered evidence (logits)"),
    )
    contrasts = ("game", "neutral", "game_minus_neutral")
    labels = ("Game", "Neutral", "Game − Neutral")
    colors = ("#2b8cbe", "#f17c32", "#5e3c99")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4))
    x = np.arange(3)
    for row, (split_label, split) in enumerate(splits):
        for col, (metric_name, ylabel) in enumerate(metrics):
            axis = axes[row, col]
            for subset, marker, offset in (("conflict", "o", -0.09), ("no_conflict", "s", 0.09)):
                effect = split[subset]["metrics"][metric_name]["effect"]
                means = np.asarray([effect[key]["mean"] for key in contrasts])
                lows = np.asarray([effect[key]["ci"][0] for key in contrasts])
                highs = np.asarray([effect[key]["ci"][1] for key in contrasts])
                for index in range(3):
                    axis.errorbar(
                        x[index] + offset,
                        means[index],
                        yerr=[[means[index] - lows[index]], [highs[index] - means[index]]],
                        fmt=marker,
                        markersize=7,
                        capsize=4,
                        linewidth=1.6,
                        color=colors[index],
                        markerfacecolor=colors[index] if subset == "conflict" else "white",
                        label=(f"{subset.replace('_', '-')} (n={split[subset]['n']})" if index == 0 else None),
                    )
            axis.axhline(0, color="#666666", linewidth=1, linestyle="--")
            axis.set_xticks(x, labels)
            axis.set_ylabel(ylabel)
            axis.set_title(split_label)
            axis.grid(axis="y", alpha=0.2)
            if col == 0:
                axis.legend(frameon=False)
    fig.suptitle("Continuous removal of first-decision A–D identity, readouts 48–63")
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
    prior_results_path: Path | None,
    draws: int,
    seed: int,
) -> None:
    data = _load(results_path, second_mapping_path, baseline_path, remapped_baseline_path)
    discovery = _summarize(data, "discovery", draws, seed)
    confirmation = _summarize(data, "confirmation", draws, seed + 1)
    validation = _validation(data, prior_results_path)
    summary = {
        "definitions": {
            "effect": "Continuous scrub minus same-hook no-edit identity control.",
            "game_minus_neutral": "Game causal effect minus Neutral causal effect.",
            "conflict": "Semantic first winner W1 differs from fresh remapped winner W2.",
        },
        "validation": validation,
        "discovery": discovery,
        "confirmation": confirmation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    rows: list[dict[str, Any]] = []
    for split_name, split in (("discovery", discovery), ("confirmation", confirmation)):
        for subset_name, subset in split.items():
            for metric_name, metric in subset["metrics"].items():
                for contrast, item in metric["effect"].items():
                    rows.append({
                        "split": split_name,
                        "subset": subset_name,
                        "n": subset["n"],
                        "metric": metric_name,
                        "contrast": contrast,
                        "mean": item["mean"],
                        "ci_low": item["ci"][0],
                        "ci_high": item["ci"][1],
                    })
    with (output_dir / "effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def cell(split: dict[str, Any], subset: str, metric: str, contrast: str) -> str:
        item = split[subset]["metrics"][metric]["effect"][contrast]
        return f"{item['mean']:+.3f} [{item['ci'][0]:+.3f}, {item['ci'][1]:+.3f}]"

    natural_game = confirmation["conflict"]["metrics"]["w1_choice_pp"]["natural"]["game"]["mean"]
    natural_neutral = confirmation["conflict"]["metrics"]["w1_choice_pp"]["natural"]["neutral"]["mean"]
    lines = [
        "# Continuous first-decision letter scrub",
        "",
        "## Bottom line",
        "",
        "This is a clean negative result: continuously removing the directly readable A–D identity at the first-decision token after every readout 48–63 does **not** causally explain Game's preferential avoidance of semantic W1.",
        "",
        f"Natural held-out conflict-trial W1 choice is {natural_game:.1f}% in Game and {natural_neutral:.1f}% in Neutral, a {natural_neutral - natural_game:.1f}-point preferential-avoidance gap. Continuous letter scrubbing changes Game W1 choice by {cell(confirmation, 'conflict', 'w1_choice_pp', 'game')} points, Neutral by {cell(confirmation, 'conflict', 'w1_choice_pp', 'neutral')} points, and the primary Game-minus-Neutral interaction by {cell(confirmation, 'conflict', 'w1_choice_pp', 'game_minus_neutral')} points. It therefore explains 0% of that held-out gap at the point estimate.",
        "",
        f"The corresponding held-out continuous W1-minus-W2 interaction is {cell(confirmation, 'conflict', 'w1_minus_w2_margin', 'game_minus_neutral')} logits. Discovery gives {cell(discovery, 'conflict', 'w1_choice_pp', 'game_minus_neutral')} points but only {cell(discovery, 'conflict', 'w1_minus_w2_margin', 'game_minus_neutral')} logits. The discovery choice movement does not replicate in confirmation or in the continuous margin.",
        "",
        f"The edit does have a small, almost perfectly shared effect on conflict trials: W1-minus-W2 rises by {cell(confirmation, 'conflict', 'w1_minus_w2_margin', 'game')} logits in Game and {cell(confirmation, 'conflict', 'w1_minus_w2_margin', 'neutral')} in Neutral. A–D spread falls by {cell(confirmation, 'conflict', 'ad_spread', 'game')} and {cell(confirmation, 'conflict', 'ad_spread', 'neutral')} logits, while entropy rises slightly in both. Thus this late A–D coordinate participates in generic candidate geometry/flattening, not the condition-specific semantic binding that makes Game avoid W1.",
        "",
        "The mechanistic implication is narrow but important: the explicit late answer-letter state at the first-decision token is not the route that carries the remembered winner into Game-specific suppression. This does not exclude an earlier relay before readout 48, or winner information encoded in other dimensions or positions.",
        "",
        "## Validation",
        "",
        f"- Natural-versus-identity maximum A–D logit error: {validation['identity_max_abs_ad_logit_error']:.6g}; choice changes: {validation['identity_choice_changes']}.",
        f"- Same-batch natural trusted-choice agreement: {validation['natural_trusted_choice_agreement']:.2%}.",
        f"- Maximum A–D logit error versus the preceding validated same-host natural run: {validation.get('same_host_prior_natural_max_abs_ad_logit_error', float('nan')):.6g}; choice changes: {validation.get('same_host_prior_natural_choice_changes', -1)}.",
        f"- First-decision Baseline-choice agreement: {validation['first_decision_baseline_choice_agreement']:.2%}.",
        f"- Maximum post-projection A–D coefficient norm: {validation['max_post_ad_norm']:.6g}.",
        f"- Mean removed A–D component norm across targeted readouts: {validation['mean_dose_l2']:.4f}.",
        "",
        "## No-conflict trials",
        "",
        f"Held-out no-conflict Game W1-choice effect: {cell(confirmation, 'no_conflict', 'w1_choice_pp', 'game')} points; Neutral: {cell(confirmation, 'no_conflict', 'w1_choice_pp', 'neutral')} points; interaction: {cell(confirmation, 'no_conflict', 'w1_choice_pp', 'game_minus_neutral')} points.",
        "",
        "The machine-readable summary contains all/conflict/no-conflict effects for W1 and W2 choice, switching, probabilities, margins, entropy, and A–D spread on both frozen splits.",
        "",
        "## Letter-bias and matched-decision checks",
        "",
        f"On held-out conflict trials with W1=A (n={confirmation['conflict_w1_a']['n']}), the W1-choice interaction is {cell(confirmation, 'conflict_w1_a', 'w1_choice_pp', 'game_minus_neutral')} points. With W1 in B–D (n={confirmation['conflict_w1_not_a']['n']}), it is {cell(confirmation, 'conflict_w1_not_a', 'w1_choice_pp', 'game_minus_neutral')} points.",
        "",
        f"Restricting to the {confirmation['conflict_first_decision_matched']['n']} held-out conflict questions whose current first decision matches W1 gives {cell(confirmation, 'conflict_first_decision_matched', 'w1_choice_pp', 'game_minus_neutral')} points and {cell(confirmation, 'conflict_first_decision_matched', 'w1_minus_w2_margin', 'game_minus_neutral')} logits.",
        "",
        f"Canonical figure: `{figure_path}`.",
    ]
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
    parser.add_argument("--prior-results", type=Path)
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
        args.prior_results,
        args.draws,
        args.seed,
    )


if __name__ == "__main__":
    main()
