from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_final_decision_semantic_ablation import (
    LETTERS,
    _align,
    _entropy,
    _interval,
)


def _load_context(
    results: Path,
    baseline_path: Path,
    remapped_path: Path,
    manifest_path: Path,
    mapping_plan_path: Path,
    question_plan_path: Path | None = None,
) -> dict[str, Any]:
    arrays = dict(np.load(results, allow_pickle=False))
    if not np.all(arrays["completed"]):
        raise ValueError(f"Incomplete results: {results}")
    qids = arrays["question_ids"].astype(str).tolist()
    if question_plan_path is not None:
        wanted = set(json.loads(question_plan_path.read_text())["question_ids"])
        selected = np.asarray([qid in wanted for qid in qids])
        for key, values in list(arrays.items()):
            if key == "question_ids" or (values.ndim >= 1 and values.shape[0] == len(qids)):
                arrays[key] = values[selected]
            elif values.ndim >= 2 and values.shape[1] == len(qids):
                arrays[key] = values[:, selected]
        qids = arrays["question_ids"].astype(str).tolist()
        if len(qids) != len(wanted):
            raise ValueError(f"Plan/result mismatch for {question_plan_path}")
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped = json.loads(remapped_path.read_text())["results"]
    manifest = {
        row["id"]: row for row in json.loads(manifest_path.read_text())["questions"]
    }
    plan = {
        row["question_id"]: row
        for row in json.loads(mapping_plan_path.read_text())["rows"]
    }
    def semantic_answers(raw_logits: np.ndarray) -> np.ndarray:
        answers = np.empty(raw_logits.shape[:2], dtype=np.int64)
        for condition_index in range(raw_logits.shape[0]):
            for question_index, qid in enumerate(qids):
                new_letter = LETTERS[int(raw_logits[condition_index, question_index].argmax())]
                original_content = plan[qid]["new_to_original"][new_letter]
                answers[condition_index, question_index] = LETTERS.index(original_content)
        return answers

    w1 = np.asarray([LETTERS.index(baseline[qid]["answer"]) for qid in qids])
    w2_text = [remapped[qid].get("answer_original_content") for qid in qids]
    valid = np.asarray([answer in LETTERS for answer in w2_text])
    w2 = np.asarray(
        [LETTERS.index(answer) if answer in LETTERS else -1 for answer in w2_text]
    )
    correct = np.asarray(
        [LETTERS.index(manifest[qid]["correct_answer"]) for qid in qids]
    )
    return {
        "arrays": arrays,
        "qids": qids,
        "w1": w1,
        "w2": w2,
        "valid": valid,
        "correct": correct,
        "natural": _align(arrays["natural_logits"], qids, plan),
        "ablated": _align(arrays["ablated_logits"], qids, plan),
        # Determine emitted answers in displayed-letter coordinates before
        # mapping them back to original semantic content. Taking argmax after
        # semantic reordering changes the tie-break on exact A-D logit ties.
        "natural_answers": semantic_answers(arrays["natural_logits"]),
        "ablated_answers": semantic_answers(arrays["ablated_logits"]),
    }


def _condition_metrics(
    values: np.ndarray,
    answers: np.ndarray,
    mask: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    correct: np.ndarray,
) -> dict[str, float]:
    rows = np.arange(len(w1))
    w1_logits = values[rows, w1]
    centered_w1 = w1_logits - values.mean(axis=-1)
    best_other = np.max(
        np.where(np.eye(4, dtype=bool)[w1], -np.inf, values), axis=-1
    )
    output = {
        "n": int(mask.sum()),
        "w1_selection": float(np.mean(answers[mask] == w1[mask])),
        "accuracy": float(np.mean(answers[mask] == correct[mask])),
        "entropy_bits": float(_entropy(values[mask]).mean()),
        "centered_w1_logit": float(centered_w1[mask].mean()),
        "w1_vs_best_other_margin": float((w1_logits - best_other)[mask].mean()),
    }
    if np.all(w2[mask] >= 0):
        output["w2_selection"] = float(np.mean(answers[mask] == w2[mask]))
        output["other_selection"] = float(
            np.mean((answers[mask] != w1[mask]) & (answers[mask] != w2[mask]))
        )
        output["w1_vs_w2_margin"] = float(
            (w1_logits - values[rows, w2.clip(min=0)])[mask].mean()
        )
    return output


def analyze_split(
    results: Path,
    baseline: Path,
    remapped: Path,
    manifest: Path,
    mapping_plan: Path,
    question_plan: Path | None,
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    ctx = _load_context(
        results, baseline, remapped, manifest, mapping_plan, question_plan
    )
    arrays = ctx["arrays"]
    w1, w2, valid, correct = (
        ctx["w1"], ctx["w2"], ctx["valid"], ctx["correct"]
    )
    masks = {
        "all": np.ones(len(w1), dtype=bool),
        "conflict": valid & (w1 != w2),
        "agreement": valid & (w1 == w2),
    }
    summary: dict[str, Any] = {
        "root": str(results.parent),
        "n_questions": len(w1),
        "subsets": {},
        "projection_dose": {},
    }
    rng = np.random.default_rng(seed)
    for subset, mask in masks.items():
        strata = w1[mask]
        row: dict[str, Any] = {"n": int(mask.sum()), "conditions": {}}
        condition_trial_values: dict[str, dict[str, np.ndarray]] = {}
        for ci, condition in enumerate(("game", "neutral")):
            natural = ctx["natural"][ci]
            ablated = ctx["ablated"][ci]
            natural_answers = ctx["natural_answers"][ci]
            ablated_answers = ctx["ablated_answers"][ci]
            natural_metrics = _condition_metrics(
                natural, natural_answers, mask, w1, w2, correct
            )
            ablated_metrics = _condition_metrics(
                ablated, ablated_answers, mask, w1, w2, correct
            )
            row["conditions"][condition] = {
                "natural": natural_metrics,
                "positive_only": ablated_metrics,
                "changes": {
                    key: ablated_metrics[key] - natural_metrics[key]
                    for key in ablated_metrics
                    if key != "n" and key in natural_metrics
                },
            }
            selection_change = (
                (ablated_answers == w1).astype(float)
                - (natural_answers == w1).astype(float)
            )
            centered_natural = natural[np.arange(len(w1)), w1] - natural.mean(-1)
            centered_ablated = ablated[np.arange(len(w1)), w1] - ablated.mean(-1)
            condition_trial_values[condition] = {
                "w1_selection_change": selection_change,
                "centered_w1_change": centered_ablated - centered_natural,
            }
            if np.all(w2[mask] >= 0):
                rows = np.arange(len(w1))
                condition_trial_values[condition]["w1_vs_w2_margin_change"] = (
                    (ablated[rows, w1] - ablated[rows, w2.clip(min=0)])
                    - (natural[rows, w1] - natural[rows, w2.clip(min=0)])
                )
            row["conditions"][condition]["w1_selection_change_ci"] = _interval(
                selection_change[mask], strata, rng, draws
            )
            row["conditions"][condition]["centered_w1_change_ci"] = _interval(
                (centered_ablated - centered_natural)[mask], strata, rng, draws
            )
        row["game_minus_neutral_change"] = {
            key: _interval(
                (condition_trial_values["game"][key]
                 - condition_trial_values["neutral"][key])[mask],
                strata,
                rng,
                draws,
            )
            for key in condition_trial_values["game"]
            if key in condition_trial_values["neutral"]
        }
        summary["subsets"][subset] = row

    for ci, condition in enumerate(("game", "neutral")):
        natural = arrays["natural_projection"][ci]
        live = arrays["ablated_pre_projection"][ci]
        summary["projection_dose"][condition] = {}
        for subset, mask in masks.items():
            summary["projection_dose"][condition][subset] = {
                "natural_positive_fraction_by_layer": (natural[mask] > 0).mean(0).tolist(),
                "natural_positive_mean_by_layer": np.maximum(natural[mask], 0).mean(0).tolist(),
                "live_positive_fraction_by_layer": (live[mask] > 0).mean(0).tolist(),
                "live_positive_mean_removed_by_layer": np.maximum(live[mask], 0).mean(0).tolist(),
                "natural_l64_positive_fraction": float(np.mean(natural[mask, -1] > 0)),
                "natural_l64_positive_mean": float(np.maximum(natural[mask, -1], 0).mean()),
                "live_l64_positive_fraction": float(np.mean(live[mask, -1] > 0)),
                "live_l64_positive_mean_removed": float(np.maximum(live[mask, -1], 0).mean()),
            }
    return summary


def _fmt_change(row: dict[str, Any], scale: float = 1.0) -> str:
    return (
        f"{row['mean'] * scale:+.2f} "
        f"[{row['ci'][0] * scale:+.2f}, {row['ci'][1] * scale:+.2f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path)
    parser.add_argument("--confirmation-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "definitions": {
            "positive_only": "At every readout subtract max(h dot v_W1, 0) times v_W1; leave negative projections unchanged.",
            "W1": "semantic answer selected in the original Baseline presentation",
            "W2": "semantic answer selected by a fresh Baseline solution of the remapped presentation",
        },
        "discovery": analyze_split(
            args.discovery / "results.npz", args.baseline, args.remapped_baseline,
            args.manifest, args.mapping_plan, args.discovery_plan,
            seed=args.seed, draws=args.draws,
        ),
        "confirmation": analyze_split(
            args.confirmation / "results.npz", args.baseline,
            args.remapped_baseline, args.manifest, args.mapping_plan,
            args.confirmation_plan, seed=args.seed + 1000, draws=args.draws,
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Positive-only W1 semantic ablation at the final decision position",
        "",
        "At every post-block readout, the intervention subtracts `max(h · v_W1, 0) v_W1`. Negative projections are left untouched. This distinguishes removing positive W1-aligned activation from the earlier signed projection-zeroing intervention, which moved negative projections toward W1.",
        "",
        "## Frozen confirmation",
        "",
        "| Subset | Condition | Natural W1 | Positive-only W1 | W1 change (95% CI) | Natural W2 | Positive-only W2 | Centered W1-logit change (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in ("conflict", "agreement", "all"):
        row = summary["confirmation"]["subsets"][subset]
        for condition in ("game", "neutral"):
            values = row["conditions"][condition]
            nat, pos = values["natural"], values["positive_only"]
            lines.append(
                f"| {subset.title()} (n={row['n']}) | {condition.title()} | "
                f"{nat['w1_selection']:.1%} | {pos['w1_selection']:.1%} | "
                f"{_fmt_change(values['w1_selection_change_ci'], 100)} pp | "
                f"{nat.get('w2_selection', float('nan')):.1%} | "
                f"{pos.get('w2_selection', float('nan')):.1%} | "
                f"{_fmt_change(values['centered_w1_change_ci'])} |"
            )
    lines.extend([
        "",
        "## Game-specific effects",
        "",
        "Here a positive W1-selection value means that the intervention reduces the natural Neutral-minus-Game W1-selection gap: Game moves toward W1 more than Neutral does. This is the behavioral quantity relevant to the semantic-suppression hypothesis.",
        "",
        "| Subset | Reduction in Neutral-minus-Game W1-selection gap (95% CI) | Game-minus-Neutral centered-W1 logit change (95% CI) |",
        "|---|---:|---:|",
    ])
    for subset in ("conflict", "agreement", "all"):
        row = summary["confirmation"]["subsets"][subset]
        contrast = row["game_minus_neutral_change"]
        lines.append(
            f"| {subset.title()} (n={row['n']}) | "
            f"{_fmt_change(contrast['w1_selection_change'], 100)} pp | "
            f"{_fmt_change(contrast['centered_w1_change'])} |"
        )
    discovery_conflict = summary["discovery"]["subsets"]["conflict"][
        "game_minus_neutral_change"
    ]
    lines.extend([
        "",
        "On the frozen discovery conflict subset, the corresponding reduction in the W1-selection gap was "
        f"{_fmt_change(discovery_conflict['w1_selection_change'], 100)} percentage points, and the centered-W1 logit contrast was "
        f"{_fmt_change(discovery_conflict['centered_w1_change'])}. Thus the behavioral effect is small and uncertain in both splits, while the larger confirmation logit effect does not reproduce at comparable magnitude in discovery.",
    ])
    lines.extend([
        "",
        "## Positive-projection dose at readout 64",
        "",
        "| Subset | Condition | Natural positive trials | Natural mean positive projection | Intervention positive trials | Mean positive projection removed |",
        "|---|---|---:|---:|---:|---:|",
    ])
    dose = summary["confirmation"]["projection_dose"]
    for subset in ("conflict", "agreement", "all"):
        for condition in ("game", "neutral"):
            row = dose[condition][subset]
            lines.append(
                f"| {subset.title()} | {condition.title()} | "
                f"{row['natural_l64_positive_fraction']:.1%} | "
                f"{row['natural_l64_positive_mean']:.2f} | "
                f"{row['live_l64_positive_fraction']:.1%} | "
                f"{row['live_l64_positive_mean_removed']:.2f} |"
            )
    lines.extend([
        "",
        "## Validation",
        "",
        "The natural companion was compared question-by-question with the previous exact historical run across all 500 questions. Natural A-D logits, all 64 W1 projection readouts, and all 64 residual norms were bit-for-bit identical (maximum absolute difference 0.0 for every array).",
        "",
        "## Data files",
        "",
        "- `../all/results.npz` is the canonical raw result. Its condition axis is `[Game, Neutral]`. It contains 500 question IDs; natural and positive-only A-D logits with shape `[2, 500, 4]`; and natural projections, natural residual norms, intervention projections before removal, intervention residual norms, and projections after removal with shape `[2, 500, 64]`.",
        "- `../all/run_metadata.json` records the exact model revision, prompt and serialization configuration, answer-token IDs, vector definition, intervention, package versions, and host platform.",
        "- `../data/per_question_condition.csv` is the human-readable trial table: one row for each of 500 questions times two conditions. It gives the frozen split, W1, W2, natural and intervened answers, correctness, entropy, W1-centered logits, W1-vs-W2 margins, and all four logits both in displayed-letter and original-semantic-content coordinates.",
        "- `../data/per_question_condition_layer.csv` is the human-readable layer table: one row for each question times condition times 64 readouts. It gives the natural W1 projection and residual norm, the live intervention projection before and after removal, the intervention residual norm, and the positive projection removed at that readout.",
        "- `summary.json` contains the discovery and confirmation aggregates and letter-stratified bootstrap confidence intervals used in this report.",
        "",
        "Emitted answers are determined by taking the A-D argmax in displayed second-presentation letter coordinates and only then mapping that letter back to original semantic content. This preserves the model's actual A-before-B-before-C-before-D tie-break on exact aggregated-logit ties. Taking argmax after semantic reordering would silently change the answer on 15 of the 2,000 natural or intervened condition rows; the exported CSV and this report use the displayed-letter tie-break.",
    ])
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
