from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .run_remapped_final_query_repeated_option_ablation import SOURCE_ROLES
from .semantic_mapping import displayed_argmax_to_semantic_indices


CONDITIONS = ("Game", "Neutral")


def _align(values: np.ndarray, qids: list[str], mappings: dict[str, dict]) -> np.ndarray:
    out = np.empty_like(values)
    for qi, qid in enumerate(qids):
        original_to_new = mappings[qid]["original_to_new"]
        for oi, original in enumerate(LETTERS):
            out[..., qi, oi] = values[..., qi, LETTERS.index(original_to_new[original])]
    return out


def _entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -(probabilities * np.log2(np.clip(probabilities, 1e-12, None))).sum(axis=-1)


def _metrics(
    logits: np.ndarray, w1i: np.ndarray, w2i: np.ndarray, choices: np.ndarray
) -> dict[str, np.ndarray]:
    rows = np.arange(len(w1i))
    centered = logits - logits.mean(axis=-1, keepdims=True)
    return {
        "w1_selection": (choices == w1i).astype(float),
        "switch_away_from_w1": (choices != w1i).astype(float),
        "w2_selection": (choices == w2i).astype(float),
        "w1_minus_w2_margin": logits[rows, w1i] - logits[rows, w2i],
        "w1_centered_advantage": 4.0 / 3.0 * centered[rows, w1i],
        "entropy_bits": _entropy(logits),
    }


def _interval(values: np.ndarray, labels: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = np.zeros(draws)
    for label in np.unique(labels):
        group = values[labels == label]
        sampled += rng.choice(group, size=(draws, len(group)), replace=True).sum(axis=1)
    sampled /= len(values)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(sampled, (0.025, 0.975)).tolist(),
    }


def _fmt(row: dict[str, Any], scale: float = 1.0) -> str:
    return f"{row['mean']*scale:+.2f} [{row['ci'][0]*scale:+.2f}, {row['ci'][1]*scale:+.2f}]"


def analyze(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"]):
        raise RuntimeError("Run is incomplete")
    if arrays["source_roles"].astype(str).tolist() != list(SOURCE_ROLES):
        raise RuntimeError("Unexpected source roles")
    qids = arrays["question_ids"].astype(str).tolist()
    mappings = {row["question_id"]: row for row in json.loads(args.remapping_plan.read_text())["rows"]}
    baseline = json.loads(args.baseline.read_text())["results"]
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    discovery = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    w1 = np.asarray([baseline[qid]["answer"] for qid in qids])
    w2 = np.asarray([remapped[qid]["answer_original_content"] for qid in qids])
    w1i = np.asarray([LETTERS.index(value) for value in w1])
    w2i = np.asarray([LETTERS.index(value) for value in w2])
    conflict = w1 != w2
    discovery_mask = np.asarray([qid in discovery for qid in qids])

    max_error = float(np.max(np.abs(arrays["same_batch_natural_logits"] - arrays["trusted_natural_logits"])))
    same_batch_choices = arrays["same_batch_natural_logits"].argmax(axis=-1)
    trusted_choices = arrays["trusted_natural_logits"].argmax(axis=-1)
    natural_choice_match = same_batch_choices == trusted_choices
    # The new A100 host uses a different NVIDIA driver from the host that
    # produced the trusted run.  Therefore the causal estimand must be paired
    # against the natural companion computed in the *same batch*, while the
    # trusted run is retained as a behavioral-reproduction check.
    natural = _align(arrays["same_batch_natural_logits"].astype(float), qids, mappings)
    intervened = _align(arrays["intervention_logits"].astype(float), qids, mappings)
    mapping_rows = [mappings[qid] for qid in qids]
    natural_choices = displayed_argmax_to_semantic_indices(
        arrays["same_batch_natural_logits"], mapping_rows
    )
    intervened_choices = displayed_argmax_to_semantic_indices(
        arrays["intervention_logits"], mapping_rows
    )
    natural_metrics = [
        _metrics(natural[ci], w1i, w2i, natural_choices[ci]) for ci in range(2)
    ]
    source_metrics = [
        [
            _metrics(
                intervened[ci, source], w1i, w2i,
                intervened_choices[ci, source],
            )
            for source in range(4)
        ]
        for ci in range(2)
    ]
    masks = {
        "discovery_conflict": discovery_mask & conflict,
        "confirmation_conflict": (~discovery_mask) & conflict,
        "discovery_no_conflict": discovery_mask & (~conflict),
        "confirmation_no_conflict": (~discovery_mask) & (~conflict),
        "all_conflict": conflict,
        "all_no_conflict": ~conflict,
    }

    summary: dict[str, Any] = {
        "definitions": {
            "W1": "Semantic answer selected by the original Baseline.",
            "W2": "Semantic answer selected by a fresh Baseline under the remapped presentation.",
            "source_effect": "Final-query source-line lesion minus natural within condition.",
            "w1_specific_contrast": "W1-line lesion effect minus the per-question mean effect of lesioning each of the other three repeated option lines.",
            "gap_reduction": "Game W1-specific contrast minus Neutral W1-specific contrast; positive means the Game-Neutral W1-avoidance gap shrinks.",
        },
        "validation": {
            "questions": len(qids),
            "conflict": int(conflict.sum()),
            "no_conflict": int((~conflict).sum()),
            "natural_logits_max_abs_error": max_error,
            "natural_choice_matches": int(natural_choice_match.sum()),
            "natural_choice_total": int(natural_choice_match.size),
            "natural_choice_match_rate": float(natural_choice_match.mean()),
            "natural_choice_match_rate_by_condition": {
                condition: float(natural_choice_match[ci].mean())
                for ci, condition in enumerate(CONDITIONS)
            },
            "causal_reference": "same-batch natural companion",
            "trusted_reproduction_role": "behavioral agreement check only",
            "max_abs_intervention_logit_change": float(np.max(np.abs(arrays["intervention_logits"] - arrays["same_batch_natural_logits"][:, None]))),
        },
        "subsets": {},
    }

    for subset_index, (subset, mask) in enumerate(masks.items()):
        labels = w1[mask]
        record: dict[str, Any] = {"n": int(mask.sum()), "natural": {}, "conditions": {}, "game_minus_neutral": {}}
        contrasts: dict[str, dict[str, np.ndarray]] = {}
        for ci, condition in enumerate(CONDITIONS):
            record["natural"][condition] = {
                metric: _interval(values[mask], labels, args.seed + subset_index*1000 + ci*100 + mi, args.draws)
                for mi, (metric, values) in enumerate(natural_metrics[ci].items())
            }
            condition_record: dict[str, Any] = {"source_effects": {}, "w1_specific_contrast": {}}
            effects: list[dict[str, np.ndarray]] = []
            for source_index, source_role in enumerate(SOURCE_ROLES):
                source_effect = {
                    metric: source_metrics[ci][source_index][metric] - natural_metrics[ci][metric]
                    for metric in natural_metrics[ci]
                }
                effects.append(source_effect)
                condition_record["source_effects"][source_role] = {
                    metric: _interval(values[mask], labels, args.seed + subset_index*10000 + ci*1000 + source_index*100 + mi, args.draws)
                    for mi, (metric, values) in enumerate(source_effect.items())
                }
            contrasts[condition] = {}
            for mi, metric in enumerate(natural_metrics[ci]):
                contrast = effects[0][metric] - np.mean([effects[index][metric] for index in (1, 2, 3)], axis=0)
                contrasts[condition][metric] = contrast
                condition_record["w1_specific_contrast"][metric] = _interval(
                    contrast[mask], labels,
                    args.seed + subset_index*10000 + ci*1000 + 700 + mi,
                    args.draws,
                )
            record["conditions"][condition] = condition_record
        for mi, metric in enumerate(natural_metrics[0]):
            difference = contrasts["Game"][metric] - contrasts["Neutral"][metric]
            record["game_minus_neutral"][metric] = _interval(
                difference[mask], labels,
                args.seed + subset_index*10000 + 9000 + mi,
                args.draws,
            )
        summary["subsets"][subset] = record

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"Game": "#2f8df3", "Neutral": "#f07f32"}
    split_colors = {"discovery": "#9ac8ff", "confirmation": "#1769aa"}

    # Panel A: held-out effects of lesioning each repeated option line.
    source_labels = ("W1 line", "Other line 1", "Other line 2", "Other line 3")
    y = np.arange(4)
    for offset, condition in zip((-0.1, 0.1), CONDITIONS):
        rows = [summary["subsets"]["confirmation_conflict"]["conditions"][condition]["source_effects"][role]["w1_selection"] for role in SOURCE_ROLES]
        means = np.asarray([row["mean"] for row in rows]) * 100
        cis = np.asarray([row["ci"] for row in rows]) * 100
        axes[0, 0].errorbar(means, y + offset, xerr=np.vstack([means-cis[:, 0], cis[:, 1]-means]), fmt="o", color=colors[condition], capsize=4, label=condition)
    axes[0, 0].set_yticks(y, source_labels)
    axes[0, 0].set_title("A  Held-out conflict: source-line lesion effect", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("Change in W1 choice (percentage points)")
    axes[0, 0].legend(frameon=False)

    # Panels B-D: W1 line minus mean of all three other-line controls.
    panel_specs = (
        (axes[0, 1], "conflict", "w1_selection", 100.0, "B  Conflict: W1-specific final read", "W1 choice (percentage points)"),
        (axes[1, 0], "conflict", "w1_minus_w2_margin", 1.0, "C  Conflict: W1-specific final read", "W1−W2 margin (logits)"),
        (axes[1, 1], "no_conflict", "w1_selection", 100.0, "D  No conflict: W1-specific final read", "W1 choice (percentage points)"),
    )
    x_positions = np.arange(2)
    for axis, subset_kind, metric, scale, title, xlabel in panel_specs:
        for offset, condition in zip((-0.1, 0.1), CONDITIONS):
            rows = [summary["subsets"][f"{split}_{subset_kind}"]["conditions"][condition]["w1_specific_contrast"][metric] for split in ("discovery", "confirmation")]
            means = np.asarray([row["mean"] for row in rows]) * scale
            cis = np.asarray([row["ci"] for row in rows]) * scale
            axis.errorbar(means, x_positions + offset, xerr=np.vstack([means-cis[:, 0], cis[:, 1]-means]), fmt="o", color=colors[condition], capsize=4, label=condition)
        axis.set_yticks(x_positions, ("Discovery", "Held-out confirmation"))
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel(xlabel)
    for axis in axes.flat:
        axis.axvline(0, color="#777777", linestyle="--", linewidth=1)
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Does the final decision directly read the repeated W1 option line?", fontsize=16, fontweight="bold")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    conflict_record = summary["subsets"]["confirmation_conflict"]
    no_conflict_record = summary["subsets"]["confirmation_no_conflict"]
    all_conflict = summary["subsets"]["all_conflict"]["natural"]
    natural_gap = all_conflict["Neutral"]["w1_selection"]["mean"] - all_conflict["Game"]["w1_selection"]["mean"]
    game_direct = conflict_record["conditions"]["Game"]["source_effects"]["w1"]["w1_selection"]
    game_specific = conflict_record["conditions"]["Game"]["w1_specific_contrast"]["w1_selection"]
    neutral_specific = conflict_record["conditions"]["Neutral"]["w1_specific_contrast"]["w1_selection"]
    gap = conflict_record["game_minus_neutral"]["w1_selection"]
    discovery_gap_choice = summary["subsets"]["discovery_conflict"]["game_minus_neutral"]["w1_selection"]
    discovery_gap_margin = summary["subsets"]["discovery_conflict"]["game_minus_neutral"]["w1_minus_w2_margin"]
    confirmation_gap_margin = conflict_record["game_minus_neutral"]["w1_minus_w2_margin"]
    lines = [
        "# Final decision → repeated-option causal edge test",
        "",
        "## Bottom line",
        "",
        f"The natural conflict-trial W1-avoidance difference is {natural_gap*100:.1f} percentage points (Neutral minus Game W1 choice).",
        "",
        "The final query does directly use the repeated W1 option line, but the net read is **pro-W1, not suppressive**: blocking that line lowers W1 relative to the control option lines in both conditions. The dependence is much stronger in Neutral, so the lesion shrinks the Game--Neutral W1-avoidance gap by disrupting Neutral reinstatement more than Game.",
        "",
        f"The condition difference is consistent across splits in the continuous W1−W2 margin: discovery {_fmt(discovery_gap_margin)} logits; held-out confirmation {_fmt(confirmation_gap_margin)} logits. On discrete W1 choice it is weak in discovery ({_fmt(discovery_gap_choice, 100)} points) but clear in confirmation ({_fmt(gap, 100)} points).",
        "",
        "At only the final decision query, the intervention blocks attention to one complete second-presentation option line across every ordinary-attention block 4--64. The W1 line is compared with the per-question mean of separately blocking each of the other three option lines.",
        "",
        "## Held-out conflict trials",
        "",
        f"- Direct W1-line lesion effect in Game W1 choice: {_fmt(game_direct, 100)} percentage points.",
        f"- Direct W1-line lesion effect in Neutral W1 choice: {_fmt(conflict_record['conditions']['Neutral']['source_effects']['w1']['w1_selection'], 100)} percentage points.",
        f"- Game W1-line effect minus mean other-line effect: {_fmt(game_specific, 100)} percentage points.",
        f"- Neutral W1-line effect minus mean other-line effect: {_fmt(neutral_specific, 100)} percentage points.",
        f"- Reduction in the Game-Neutral W1-avoidance gap: {_fmt(gap, 100)} percentage points.",
        f"- Game W1−W2 margin contrast: {_fmt(conflict_record['conditions']['Game']['w1_specific_contrast']['w1_minus_w2_margin'])} logits.",
        f"- Neutral W1−W2 margin contrast: {_fmt(conflict_record['conditions']['Neutral']['w1_specific_contrast']['w1_minus_w2_margin'])} logits.",
        f"- Game-minus-Neutral margin contrast: {_fmt(confirmation_gap_margin)} logits.",
        "",
        "## Held-out no-conflict trials",
        "",
        f"- Game W1-choice contrast: {_fmt(no_conflict_record['conditions']['Game']['w1_specific_contrast']['w1_selection'], 100)} percentage points.",
        f"- Neutral W1-choice contrast: {_fmt(no_conflict_record['conditions']['Neutral']['w1_specific_contrast']['w1_selection'], 100)} percentage points.",
        "",
        "## Mechanistic interpretation",
        "",
        "This rejects the simple last-hop story in which the final decision suppresses W1 by directly reading the repeated W1 line. The direct final read instead reinforces W1, especially in Neutral. Therefore the previously established Game-specific causal influence from the original W1 line into the repeated W1 line must affect Game through an earlier downstream relay or state update before the final query, not through a suppressive final-query attention edge.",
        "",
        "## Validation",
        "",
        f"Same-batch natural choices matched the trusted run on `{int(natural_choice_match.sum())}/{int(natural_choice_match.size)}` condition-question outputs ({natural_choice_match.mean()*100:.1f}%).",
        f"The maximum trusted-logit discrepancy was `{max_error}` logits, consistent with the changed NVIDIA-driver numerical regime. All causal contrasts use the same-batch natural companion, not the trusted logits.",
        f"Maximum absolute intervention-induced A-D logit change: `{summary['validation']['max_abs_intervention_logit_change']:.6f}`.",
        "",
        f"Canonical figure: `{args.figure}`.",
        "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
