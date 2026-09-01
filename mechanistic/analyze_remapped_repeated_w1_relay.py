from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .run_remapped_repeated_w1_relay import BLOCK_BANDS, INTERVENTION_CELLS
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
    if not len(values):
        return {"n": 0, "mean": None, "ci": [None, None]}
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
    if row["mean"] is None:
        return "not estimable"
    return f"{row['mean']*scale:+.2f} [{row['ci'][0]*scale:+.2f}, {row['ci'][1]*scale:+.2f}]"


def analyze(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"]):
        raise RuntimeError("Run is incomplete")
    cell_ids = arrays["intervention_cells"].astype(str).tolist()
    known_ids = {cell["id"] for cell in INTERVENTION_CELLS}
    if not cell_ids or not set(cell_ids).issubset(known_ids):
        raise RuntimeError("Unexpected intervention cells")
    active_cells = [
        cell for cell in INTERVENTION_CELLS if cell["id"] in set(cell_ids)
    ]
    qids = arrays["question_ids"].astype(str).tolist()
    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    baseline = json.loads(args.baseline.read_text())["results"]
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    discovery = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    w1 = np.asarray([
        baseline[qid]["answer"]
        for qid in qids
    ])
    w2 = np.asarray([remapped[qid]["answer_original_content"] for qid in qids])
    w1i = np.asarray([LETTERS.index(value) for value in w1])
    w2i = np.asarray([LETTERS.index(value) for value in w2])
    conflict = w1 != w2
    discovery_mask = np.asarray([qid in discovery for qid in qids])
    displayed = arrays["w1_displayed_letters"].astype(str)

    max_error = float(np.max(np.abs(
        arrays["same_batch_natural_logits"] - arrays["trusted_natural_logits"]
    )))
    same_choices = arrays["same_batch_natural_logits"].argmax(axis=-1)
    trusted_choices = arrays["trusted_natural_logits"].argmax(axis=-1)
    choice_match = same_choices == trusted_choices

    natural = _align(
        arrays["same_batch_natural_logits"].astype(float), qids, mappings
    )
    intervened = _align(
        arrays["intervention_logits"].astype(float), qids, mappings
    )
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
    cell_metrics = [
        [
            _metrics(
                intervened[ci, cell_index], w1i, w2i,
                intervened_choices[ci, cell_index],
            )
            for cell_index in range(len(cell_ids))
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
            "cell_effect": "Intervention minus same-batch natural within condition.",
            "game_minus_neutral_effect": "Game cell effect minus Neutral cell effect; positive W1 selection or W1-W2 margin means the lesion preferentially recovers W1 in Game.",
            "all_later_pre_final": "Every query after the repeated W1 line and before the already-tested final decision query.",
            "later_options": "Only subsequent second-presentation option-line tokens after the W1 line.",
            "post_options_pre_final": "Every query after all four repeated options and before the final decision query.",
        },
        "validation": {
            "questions": len(qids),
            "conflict": int(conflict.sum()),
            "no_conflict": int((~conflict).sum()),
            "natural_logits_max_abs_error": max_error,
            "natural_choice_matches": int(choice_match.sum()),
            "natural_choice_total": int(choice_match.size),
            "natural_choice_match_rate": float(choice_match.mean()),
            "causal_reference": "same-batch natural companion",
            "max_abs_intervention_logit_change": float(np.max(np.abs(
                arrays["intervention_logits"]
                - arrays["same_batch_natural_logits"][:, None]
            ))),
        },
        "cells": active_cells,
        "subsets": {},
        "displayed_w1_strata": {},
    }

    effects_by_condition: list[list[dict[str, np.ndarray]]] = [[], []]
    for ci in range(2):
        for cell_index in range(len(cell_ids)):
            effects_by_condition[ci].append({
                metric: cell_metrics[ci][cell_index][metric] - natural_metrics[ci][metric]
                for metric in natural_metrics[ci]
            })

    for subset_index, (subset, mask) in enumerate(masks.items()):
        labels = w1[mask]
        record: dict[str, Any] = {"n": int(mask.sum()), "natural": {}, "conditions": {}, "game_minus_neutral": {}}
        for ci, condition in enumerate(CONDITIONS):
            record["natural"][condition] = {
                metric: _interval(values[mask], labels, args.seed + subset_index*1000 + ci*100 + mi, args.draws)
                for mi, (metric, values) in enumerate(natural_metrics[ci].items())
            }
            record["conditions"][condition] = {}
            for cell_index, cell_id in enumerate(cell_ids):
                record["conditions"][condition][cell_id] = {
                    metric: _interval(
                        values[mask], labels,
                        args.seed + subset_index*100000 + ci*10000 + cell_index*100 + mi,
                        args.draws,
                    )
                    for mi, (metric, values) in enumerate(effects_by_condition[ci][cell_index].items())
                }
        for cell_index, cell_id in enumerate(cell_ids):
            record["game_minus_neutral"][cell_id] = {}
            for mi, metric in enumerate(natural_metrics[0]):
                difference = (
                    effects_by_condition[0][cell_index][metric]
                    - effects_by_condition[1][cell_index][metric]
                )
                record["game_minus_neutral"][cell_id][metric] = _interval(
                    difference[mask], labels,
                    args.seed + subset_index*100000 + 90000 + cell_index*100 + mi,
                    args.draws,
                )
        post_index = cell_ids.index("w1_post_options_pre_final__all_blocks")
        control_index = cell_ids.index(
            "matched_control_post_options_pre_final__all_blocks"
        )
        record["w1_minus_matched_control_post_options"] = {}
        post_control_contrasts: dict[str, dict[str, np.ndarray]] = {}
        for ci, condition in enumerate(CONDITIONS):
            post_control_contrasts[condition] = {}
            record["w1_minus_matched_control_post_options"][condition] = {}
            for mi, metric in enumerate(natural_metrics[ci]):
                contrast = (
                    effects_by_condition[ci][post_index][metric]
                    - effects_by_condition[ci][control_index][metric]
                )
                post_control_contrasts[condition][metric] = contrast
                record["w1_minus_matched_control_post_options"][condition][metric] = _interval(
                    contrast[mask], labels,
                    args.seed + subset_index*100000 + 97000 + ci*100 + mi,
                    args.draws,
                )
        record["w1_minus_matched_control_post_options"]["Game_minus_Neutral"] = {}
        for mi, metric in enumerate(natural_metrics[0]):
            difference = (
                post_control_contrasts["Game"][metric]
                - post_control_contrasts["Neutral"][metric]
            )
            record["w1_minus_matched_control_post_options"]["Game_minus_Neutral"][metric] = _interval(
                difference[mask], labels,
                args.seed + subset_index*100000 + 98000 + mi,
                args.draws,
            )
        summary["subsets"][subset] = record

    main_cell = cell_ids.index("w1_all_later_pre_final__all_blocks")
    option_cell = cell_ids.index("w1_later_options__all_blocks")
    for split_name, split_mask in (
        ("discovery", discovery_mask),
        ("confirmation", ~discovery_mask),
    ):
        summary["displayed_w1_strata"][split_name] = {}
        for letter_index, letter in enumerate(LETTERS):
            mask = split_mask & conflict & (displayed == letter)
            labels = w1[mask]
            row: dict[str, Any] = {"n": int(mask.sum()), "conditions": {}}
            for ci, condition in enumerate(CONDITIONS):
                row["conditions"][condition] = {}
                for cell_index, cell_name in (
                    (main_cell, "all_later_pre_final"),
                    (option_cell, "later_options"),
                ):
                    row["conditions"][condition][cell_name] = {
                        metric: _interval(
                            effects_by_condition[ci][cell_index][metric][mask],
                            labels,
                            args.seed + 500000 + letter_index*10000 + ci*1000 + cell_index*10 + mi,
                            args.draws,
                        )
                        for mi, metric in enumerate(("w1_selection", "w1_minus_w2_margin"))
                    }
            summary["displayed_w1_strata"][split_name][letter] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    colors = {"Game": "#2f8df3", "Neutral": "#f07f32"}
    band_cells: list[str] = []
    band_labels: list[str] = []
    for label in BLOCK_BANDS:
        cell = f"w1_all_later_pre_final__blocks_{label}"
        if cell in cell_ids:
            band_cells.append(cell)
            band_labels.append(label.replace("_", "–"))

    ncols = 3 if band_cells else 2
    fig, axes = plt.subplots(1, ncols, figsize=((16 if band_cells else 11), 6.5), constrained_layout=True)
    main_cells = [
        "w1_all_later_pre_final__all_blocks",
        "w1_later_options__all_blocks",
        "w1_post_options_pre_final__all_blocks",
        "matched_control_post_options_pre_final__all_blocks",
    ]
    main_labels = ["All later\n(W1)", "Later options\n(W1)", "After options\n(W1)", "After options\n(control)"]
    heldout = summary["subsets"]["confirmation_conflict"]
    y = np.arange(len(main_cells))
    for axis, metric, title, xlabel, scale in (
        (axes[0], "w1_selection", "A  Held-out conflict: behavioral effect", "Change in W1 choice (percentage points)", 100.0),
        (axes[1], "w1_minus_w2_margin", "B  Held-out conflict: evidence effect", "Change in W1−W2 margin (logits)", 1.0),
    ):
        for offset, condition in zip((-0.10, 0.10), CONDITIONS):
            rows = [heldout["conditions"][condition][cell][metric] for cell in main_cells]
            means = np.asarray([row["mean"] for row in rows]) * scale
            cis = np.asarray([row["ci"] for row in rows]) * scale
            axis.errorbar(
                means, y + offset,
                xerr=np.vstack([means-cis[:, 0], cis[:, 1]-means]),
                fmt="o", color=colors[condition], capsize=4, label=condition,
            )
        axis.set_yticks(y, main_labels)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.axvline(0, color="#777777", linestyle="--", linewidth=1)
        axis.grid(axis="x", alpha=0.2)
    axes[0].legend(frameon=False)

    if band_cells:
        rows = [heldout["conditions"]["Game"][cell]["w1_selection"] for cell in band_cells]
        means = np.asarray([row["mean"] for row in rows]) * 100
        cis = np.asarray([row["ci"] for row in rows]) * 100
        axes[2].errorbar(
            means, np.arange(len(rows)),
            xerr=np.vstack([means-cis[:, 0], cis[:, 1]-means]),
            fmt="o", color=colors["Game"], capsize=4,
        )
        axes[2].set_yticks(np.arange(len(rows)), band_labels)
        axes[2].set_xlabel("Change in W1 choice (percentage points)")
        axes[2].axvline(0, color="#777777", linestyle="--", linewidth=1)
        axes[2].grid(axis="x", alpha=0.2)
        axes[2].set_title("C  Held-out Game: depth localization", loc="left", fontweight="bold")
    fig.suptitle("The repeated W1 line supports W1 reinstatement—especially in Neutral", fontsize=16, fontweight="bold")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    discovery_conflict = summary["subsets"]["discovery_conflict"]
    confirmation_conflict = summary["subsets"]["confirmation_conflict"]
    confirmation_no_conflict = summary["subsets"]["confirmation_no_conflict"]
    prerequisite = "w1_all_later_pre_final__all_blocks"
    later_options = "w1_later_options__all_blocks"
    post_options = "w1_post_options_pre_final__all_blocks"
    control = "matched_control_post_options_pre_final__all_blocks"
    lines = [
        "# Repeated-W1 downstream relay localization",
        "",
        "## Bottom line",
        "",
        "This experiment asks whether Game's suppression of W1 is carried by ordinary attention from the repeated W1 option line into intermediate, pre-final states. It is not. Blocking those reads makes W1 *less* likely, so this pathway normally supports W1 reinstatement rather than suppressing it.",
        "",
        f"On held-out conflict trials, blocking every later pre-final read from the repeated W1 line changed Game W1 choice by {_fmt(confirmation_conflict['conditions']['Game'][prerequisite]['w1_selection'], 100)} percentage points and W1−W2 margin by {_fmt(confirmation_conflict['conditions']['Game'][prerequisite]['w1_minus_w2_margin'])} logits. The corresponding discovery effects were {_fmt(discovery_conflict['conditions']['Game'][prerequisite]['w1_selection'], 100)} points and {_fmt(discovery_conflict['conditions']['Game'][prerequisite]['w1_minus_w2_margin'])} logits.",
        "",
        f"Neutral depended even more strongly on this pathway: the held-out changes were {_fmt(confirmation_conflict['conditions']['Neutral'][prerequisite]['w1_selection'], 100)} points and {_fmt(confirmation_conflict['conditions']['Neutral'][prerequisite]['w1_minus_w2_margin'])} logits. Thus the lesion increased Game-minus-Neutral W1 choice by {_fmt(confirmation_conflict['game_minus_neutral'][prerequisite]['w1_selection'], 100)} points and the W1−W2 margin by {_fmt(confirmation_conflict['game_minus_neutral'][prerequisite]['w1_minus_w2_margin'])} logits—not because it recovered W1 in Game, but because it removed substantially more W1 reinstatement from Neutral.",
        "",
        "This rules out the prespecified hypothesis that a pre-final repeated-W1 relay carries an active anti-W1 signal in Game. The gated depth-band run was therefore not performed. Together with the earlier original-line→repeated-line lesion, the cleaner interpretation is differential reinstatement: the repeated line provides pro-W1 evidence in both conditions, while Game weakens or negatively contextualizes that evidence relative to Neutral.",
        "",
        "## Held-out conflict decomposition",
        "",
        f"- Later repeated-option queries, Game W1 choice: {_fmt(confirmation_conflict['conditions']['Game'][later_options]['w1_selection'], 100)} points; margin: {_fmt(confirmation_conflict['conditions']['Game'][later_options]['w1_minus_w2_margin'])} logits.",
        f"- Post-options pre-final queries, Game W1 choice: {_fmt(confirmation_conflict['conditions']['Game'][post_options]['w1_selection'], 100)} points; margin: {_fmt(confirmation_conflict['conditions']['Game'][post_options]['w1_minus_w2_margin'])} logits.",
        f"- Post-options W1-line lesion minus matched-line lesion, Game W1 choice: {_fmt(confirmation_conflict['w1_minus_matched_control_post_options']['Game']['w1_selection'], 100)} points; margin: {_fmt(confirmation_conflict['w1_minus_matched_control_post_options']['Game']['w1_minus_w2_margin'])} logits.",
        "",
        "## Held-out no-conflict context",
        "",
        f"- All later pre-final reads, Game W1 choice: {_fmt(confirmation_no_conflict['conditions']['Game'][prerequisite]['w1_selection'], 100)} points.",
        f"- All later pre-final reads, Neutral W1 choice: {_fmt(confirmation_no_conflict['conditions']['Neutral'][prerequisite]['w1_selection'], 100)} points.",
        "",
        "## Validation",
        "",
        f"Same-batch natural choices matched the trusted run on `{int(choice_match.sum())}/{int(choice_match.size)}` condition-question outputs ({choice_match.mean()*100:.1f}%).",
        f"Maximum trusted-logit discrepancy: `{max_error}`. All causal effects use the same-batch natural companion.",
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
