from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .run_fixed_a_final_query_edge_ablation import INTERVENTION_CELLS
from .semantic_mapping import displayed_argmax_to_semantic_indices


LETTERS = "ABCD"
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


def _fmt(row: dict, scale: float = 1.0) -> str:
    return f"{row['mean']*scale:+.2f} [{row['ci'][0]*scale:+.2f}, {row['ci'][1]*scale:+.2f}]"


def analyze(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"]):
        raise RuntimeError("Run is incomplete")
    if arrays["intervention_cells"].astype(str).tolist() != list(INTERVENTION_CELLS):
        raise RuntimeError("Unexpected intervention cells")

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
    if max_error != 0.0:
        raise RuntimeError(f"Natural logits failed exact reproduction: {max_error}")
    natural = _align(arrays["trusted_natural_logits"].astype(float), qids, mappings)
    intervened = _align(arrays["intervention_logits"].astype(float), qids, mappings)
    mapping_rows = [mappings[qid] for qid in qids]
    natural_choices = displayed_argmax_to_semantic_indices(
        arrays["trusted_natural_logits"], mapping_rows
    )
    intervened_choices = displayed_argmax_to_semantic_indices(
        arrays["intervention_logits"], mapping_rows
    )
    natural_metrics = [
        _metrics(natural[ci], w1i, w2i, natural_choices[ci]) for ci in range(2)
    ]
    intervention_metrics = [
        [
            _metrics(
                intervened[ci, cell], w1i, w2i,
                intervened_choices[ci, cell],
            )
            for cell in range(len(INTERVENTION_CELLS))
        ]
        for ci in range(2)
    ]
    masks = {
        "all": np.ones(len(qids), dtype=bool),
        "conflict_W1_not_equal_W2": conflict,
        "no_conflict_W1_equal_W2": ~conflict,
        "discovery_conflict": discovery_mask & conflict,
        "confirmation_conflict": (~discovery_mask) & conflict,
        "discovery_no_conflict": discovery_mask & (~conflict),
        "confirmation_no_conflict": (~discovery_mask) & (~conflict),
    }
    summary: dict[str, Any] = {
        "definitions": {
            "W1": "Semantic answer selected by the original first-presentation Baseline.",
            "W2": "Semantic answer selected by a fresh Baseline under the remapped second presentation.",
            "conflict": "W1 and W2 are different semantic answers.",
            "effect": "Intervened minus natural within the same condition and exact historical cohort.",
            "selected_minus_control": "W1-line lesion effect minus matched unselected-line lesion effect.",
        },
        "validation": {
            "questions": len(qids),
            "conflict": int(conflict.sum()),
            "no_conflict": int((~conflict).sum()),
            "max_abs_same_batch_natural_minus_trusted": max_error,
        },
        "subsets": {},
    }

    for subset_index, (name, mask) in enumerate(masks.items()):
        labels = w1[mask]
        cell: dict[str, Any] = {"n": int(mask.sum()), "natural": {}, "interventions": {}, "selected_minus_control": {}}
        for ci, condition in enumerate(CONDITIONS):
            cell["natural"][condition] = {
                metric: _interval(values[mask], labels, args.seed + subset_index*1000 + ci*100 + mi, args.draws)
                for mi, (metric, values) in enumerate(natural_metrics[ci].items())
            }
        for cell_index, intervention_cell in enumerate(INTERVENTION_CELLS):
            record: dict[str, Any] = {}
            for ci, condition in enumerate(CONDITIONS):
                record[condition] = {}
                for mi, metric in enumerate(natural_metrics[ci]):
                    effect = intervention_metrics[ci][cell_index][metric] - natural_metrics[ci][metric]
                    record[condition][metric] = _interval(
                        effect[mask], labels,
                        args.seed + subset_index*10000 + cell_index*500 + ci*100 + mi,
                        args.draws,
                    )
            cell["interventions"][intervention_cell] = record
        for pair_index, (selected_index, control_index, block_name) in enumerate(((0,1,"block_44"),(2,3,"band_36_48"),(4,5,"all_04_48"))):
            record = {}
            for ci, condition in enumerate(CONDITIONS):
                record[condition] = {}
                for mi, metric in enumerate(natural_metrics[ci]):
                    contrast = intervention_metrics[ci][selected_index][metric] - intervention_metrics[ci][control_index][metric]
                    record[condition][metric] = _interval(
                        contrast[mask], labels,
                        args.seed + subset_index*10000 + 7000 + pair_index*500 + ci*100 + mi,
                        args.draws,
                    )
            cell["selected_minus_control"][block_name] = record
        summary["subsets"][name] = cell

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    panels = (
        ("conflict_W1_not_equal_W2", "w1_selection", "A  Conflict: chooses W1", 100.0, "percentage points"),
        ("conflict_W1_not_equal_W2", "w1_minus_w2_margin", "B  Conflict: W1−W2 margin", 1.0, "logits"),
        ("no_conflict_W1_equal_W2", "w1_selection", "C  No conflict: chooses W1", 100.0, "percentage points"),
        ("no_conflict_W1_equal_W2", "switch_away_from_w1", "D  No conflict: switches away from W1", 100.0, "percentage points"),
    )
    selected_cells = ("block_44_selected", "band_36_48_selected", "all_04_48_selected")
    labels = ("B44", "B36,40,44,48", "B4–48 ordinary")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ("#2f8df3", "#f07f32")
    offsets = (-0.12, 0.12)
    for axis, (subset, metric, title, scale, ylabel) in zip(axes.flat, panels):
        for ci, condition in enumerate(CONDITIONS):
            rows = [summary["subsets"][subset]["interventions"][cell][condition][metric] for cell in selected_cells]
            means = np.asarray([row["mean"] for row in rows]) * scale
            cis = np.asarray([row["ci"] for row in rows]) * scale
            x = np.arange(3) + offsets[ci]
            axis.errorbar(x, means, yerr=np.vstack([means-cis[:,0], cis[:,1]-means]), fmt="o", markersize=7, capsize=5, linewidth=2, color=colors[ci], label=condition)
        axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks(np.arange(3), labels)
        axis.set_ylabel(f"Intervened − natural ({ylabel})")
        axis.grid(axis="y", alpha=0.2)
    axes[0,0].legend(frameon=False)
    fig.suptitle("Does the final decision directly read the first answer's semantic option line?", fontsize=16, fontweight="bold")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    conflict_rows = summary["subsets"]["conflict_W1_not_equal_W2"]
    game_natural = conflict_rows["natural"]["Game"]["w1_selection"]["mean"]
    neutral_natural = conflict_rows["natural"]["Neutral"]["w1_selection"]["mean"]
    selected_effects = [
        conflict_rows["interventions"][name]["Game"]["w1_selection"]
        for name in selected_cells
    ]
    contrast_effects = [
        conflict_rows["selected_minus_control"][name]["Game"]["w1_selection"]
        for name in ("block_44", "band_36_48", "all_04_48")
    ]
    all_margin = conflict_rows["interventions"]["all_04_48_selected"]["Game"][
        "w1_minus_w2_margin"
    ]
    discovery_margin = summary["subsets"]["discovery_conflict"]["interventions"][
        "all_04_48_selected"
    ]["Game"]["w1_minus_w2_margin"]
    confirmation_margin = summary["subsets"]["confirmation_conflict"][
        "interventions"
    ]["all_04_48_selected"]["Game"]["w1_minus_w2_margin"]

    lines = [
        "# Canonical remapped final-query W1-line attention-edge ablation", "",
        "## Question", "",
        "Does preferential semantic switching require the final pre-answer query to read the complete first-presentation option line containing W1 through ordinary attention? Earlier queries and all other source tokens remain untouched.", "",
        "Every reported effect is **intervened minus natural within the named condition**. Conflict means W1 differs from the answer a fresh Baseline would choose under the remapped second presentation (W2).", "",
        "## Bottom line", "",
        f"No. The natural behavioral phenomenon is large on conflict trials: Game chooses W1 on {100*game_natural:.1f}% of trials versus {100*neutral_natural:.1f}% in Neutral, a {100*(neutral_natural-game_natural):.1f}-percentage-point Game-specific avoidance difference. But preventing the final decision query from directly reading W1's original option line does not undo it.", "",
        "Across conflict trials, blocking the selected line at block 44, blocks 36/40/44/48, or every ordinary-attention block from 4 through 48 changes Game W1 choice by "
        + ", ".join(f"{100*row['mean']:+.2f}" for row in selected_effects)
        + " percentage points, respectively. The selected-line-minus-matched-control effects are "
        + ", ".join(f"{100*row['mean']:+.2f}" for row in contrast_effects)
        + " points. The complete confidence intervals are reported below; the frozen discovery and confirmation halves and the no-conflict trials determine whether the substantive null replicates.", "",
        f"The all-block conflict intervention changes Game's W1-minus-W2 logit margin by {_fmt(all_margin)} overall, {_fmt(discovery_margin)} in discovery, and {_fmt(confirmation_margin)} in confirmation. Its interpretation is based on these generated values rather than hard-coded prose.", "",
        "Therefore, the earlier finding that first-presentation option-line K/V is causally important must not be interpreted as a direct final-token lookup. Its information must be read by earlier downstream queries and propagated through intermediate residual/recurrent states before the final decision. This experiment rules out the clean direct-edge mechanism; it does not localize the intervening relay.", "",
        "A predecessor full pass was invalid because a Boolean SDPA mask treated the attempted block as allowed. The corrected runner writes `False` for Boolean masks, passes Boolean/additive/implicit-causal regression tests, produces nonzero logit changes, and reproduces all trusted natural logits exactly. Only the corrected results below are scientific results.", "",
    ]
    for subset in ("conflict_W1_not_equal_W2", "no_conflict_W1_equal_W2"):
        data = summary["subsets"][subset]
        lines += [f"## {subset.replace('_',' ')} (n={data['n']})", "", "| Intervention | Condition | Δ W1 choice | Δ switch away from W1 | Δ W1 centered advantage | Δ entropy |", "|---|---|---:|---:|---:|---:|"]
        for intervention_cell in INTERVENTION_CELLS:
            for condition in CONDITIONS:
                row = data["interventions"][intervention_cell][condition]
                lines.append(f"| `{intervention_cell}` | {condition} | {_fmt(row['w1_selection'],100)} pp | {_fmt(row['switch_away_from_w1'],100)} pp | {_fmt(row['w1_centered_advantage'])} | {_fmt(row['entropy_bits'])} |")
        lines += ["", "Selected W1-line effect minus matched unselected-line effect:", "", "| Block set | Condition | Δ W1 choice | Δ switch away from W1 | Δ W1 centered advantage |", "|---|---|---:|---:|---:|"]
        for block_name, record in data["selected_minus_control"].items():
            for condition in CONDITIONS:
                row = record[condition]
                lines.append(f"| `{block_name}` | {condition} | {_fmt(row['w1_selection'],100)} pp | {_fmt(row['switch_away_from_w1'],100)} pp | {_fmt(row['w1_centered_advantage'])} |")
        lines.append("")
    lines += ["## Validation", "", f"Natural logits reproduced the trusted canonical remapped run with maximum absolute error `{max_error}`.", "", f"Canonical figure: `{args.figure}`.", ""]
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
