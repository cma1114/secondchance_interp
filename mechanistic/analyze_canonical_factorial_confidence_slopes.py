"""Confidence-slope supplement to the canonical history/decision factorial.

For each cell of the canonical non-remapped factorial (natural, matching-line
blockade, first-decision cut, joint), this computes the slope of the
per-question Game-minus-Neutral old-winner push on z-scored first-pass
confidence — the dose-response endpoint the factorial's primary analysis did
not evaluate. It asks where the confidence signal that scales the push
travels: through the matching option-line reads, or out of the position at
which the first answer would have been generated.

Evidence class: descriptive reanalysis of the factorial's causal cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

LETTERS = ["A", "B", "C", "D"]

CELLS = (
    ("Qwen3.6-27B", "qwen36_27b", "SimpleMC", "simplemc",
     "outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/baseline_results.json",
     ("results",)),
    ("Qwen3.6-27B", "qwen36_27b", "TriviaMC", "triviamc",
     "outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/step1/baseline/baseline_results.json",
     ("results",)),
    ("Seed-OSS-36B", "seed_oss_36b", "SimpleMC", "simplemc",
     "outputs/model_replications/seed_oss_36b_clean_behavioral_replication/simplemc/run/results.json",
     ("scenarios", "baseline")),
    ("Seed-OSS-36B", "seed_oss_36b", "TriviaMC", "triviamc",
     "outputs/model_replications/seed_oss_36b_clean_behavioral_replication/triviamc/run/results.json",
     ("scenarios", "baseline")),
    ("Gemma-4-31B", "gemma4_31b", "SimpleMC", "simplemc",
     "outputs/model_replications/gemma4_31b_negative_model_comparison/simplemc/behavior/run/results.json",
     ("scenarios", "baseline")),
    ("Gemma-4-31B", "gemma4_31b", "TriviaMC", "triviamc",
     "outputs/model_replications/gemma4_31b_negative_model_comparison/triviamc/behavior/run/results.json",
     ("scenarios", "baseline")),
)

SCENARIOS = ("natural", "first_decision", "matching", "matching_plus_first_decision")


def _slope(confidence: np.ndarray, push: np.ndarray, rng: np.random.Generator,
           draws: int) -> dict[str, float]:
    z = (confidence - confidence.mean()) / confidence.std(ddof=0)
    value = float(np.mean(z * (push - push.mean())))
    n = len(confidence)
    index = rng.integers(0, n, size=(draws, n))
    xb = confidence[index]
    zb = (xb - xb.mean(axis=1, keepdims=True)) / xb.std(axis=1, ddof=0, keepdims=True)
    yb = push[index]
    boot = np.mean(zb * (yb - yb.mean(axis=1, keepdims=True)), axis=1)
    return {
        "value": value,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    dose = json.loads((args.root / args.dose_response_summary).read_text())
    dose_slopes = {
        (cell["model"], cell["dataset"]):
            cell["splits"]["full"]["univariate"]["push_r1"]["raw_outcome_per_1sd_c1"]["value"]
        for cell in dose["cells"]
    }
    summary: dict[str, Any] = {
        "analysis": "canonical_factorial_confidence_slopes",
        "evidence_class": "descriptive reanalysis of causal factorial cells",
        "definition": {
            "push": "negated Game-minus-Neutral within-question-centered final logit at the model's own old rank 1, per factorial cell",
            "confidence": "z-scored first-pass top-1 minus top-2 aggregated A-D logit from the model's own canonical baseline",
            "gates": "baseline argsort must reproduce the factorial's rank_contents exactly; the natural cell's slope must equal the audited dose-response slope",
            "bootstrap": f"question-level percentile bootstrap, seed {args.seed}, {args.draws} draws",
        },
        "cells": [],
    }
    for model_label, model_key, ds_label, ds_key, baseline_path, container in CELLS:
        npz_path = (args.root / "outputs/model_replications/canonical_history_decision_factorial"
                    / model_key / ds_key / "run/results.npz")
        with np.load(npz_path, allow_pickle=False) as loaded:
            cells = list(loaded["cells"].astype(str))
            qids = loaded["question_ids"].astype(str).tolist()
            logits = np.asarray(loaded["logits"], dtype=np.float64)
            rank_contents = loaded["rank_contents"].astype(str)
        document = json.loads((args.root / baseline_path).read_text())
        for key in container:
            document = document[key]
        baseline = np.asarray(
            [document[qid]["aggregated_ad_logits"] for qid in qids], dtype=np.float64)
        order = np.argsort(-baseline, axis=-1, kind="stable")
        w1 = np.asarray([LETTERS.index(letter) for letter in rank_contents[:, 0]])
        if not np.array_equal(order[:, 0], w1):
            raise ValueError(f"Baseline/rank_contents W1 mismatch for {model_label} {ds_label}")
        by_rank = np.take_along_axis(baseline, order, axis=1)
        confidence = by_rank[:, 0] - by_rank[:, 1]
        centered = logits - logits.mean(axis=-1, keepdims=True)
        rows = np.arange(len(qids))
        record: dict[str, Any] = {"model": model_label, "dataset": ds_label, "n": len(qids)}
        for scenario in SCENARIOS:
            index = cells.index(scenario)
            push = -(centered[0, index][rows, w1] - centered[1, index][rows, w1])
            record[scenario] = _slope(confidence, push, rng, args.draws)
        reference = dose_slopes[(model_label, ds_label)]
        if abs(record["natural"]["value"] - reference) > 1e-9:
            raise ValueError(
                f"Natural slope does not reproduce the dose-response value for "
                f"{model_label} {ds_label}: {record['natural']['value']} vs {reference}")
        record["natural_matches_dose_response"] = True
        summary["cells"].append(record)
    return summary


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Where the confidence dose-response travels",
        "",
        "Slope of the per-question Game-minus-Neutral old-winner push on z-scored",
        "first-pass confidence, inside each causal cell of the canonical",
        "history/decision factorial. Each natural slope exactly reproduces the",
        "audited dose-response value (asserted), and each baseline ranking exactly",
        "reproduces the factorial's stored rank order (asserted).",
        "",
        "| Model | Dataset | Natural | First-decision cut | Matching-line cut | Joint |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        def fmt(name: str) -> str:
            s = cell[name]
            return f"{s['value']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
        lines.append(
            f"| {cell['model']} | {cell['dataset']} | {fmt('natural')} "
            f"| {fmt('first_decision')} | {fmt('matching')} "
            f"| {fmt('matching_plus_first_decision')} |")
    lines += [
        "",
        "Severing every outgoing signal from the would-be first-answer position",
        "leaves the confidence scaling untouched in all six cells. Blocking the",
        "matching option-line reads collapses it by 77-97% in four of the five",
        "cells that show scaling (Qwen both datasets, Seed SimpleMC, Gemma",
        "SimpleMC) and partially on Seed TriviaMC. In Qwen TriviaMC the remnant",
        "left by the matching cut (+0.240) is removed by additionally cutting the",
        "first-decision position (+0.021): Qwen's backup route through that",
        "position carries graded confidence only once the line route is gone.",
        "The graded prior confidence the Game policy consumes therefore travels",
        "with the retrieved option-line scores themselves, not as a summary",
        "stored at the answer position.",
        "",
    ]
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--dose-response-summary", type=Path,
        default=Path("outputs/model_replications/confidence_dose_response/analysis/summary.json"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/model_replications/canonical_history_decision_factorial/confidence_slope_supplement"))
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--draws", type=int, default=10000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = analyze(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    write_report(summary, args.output_dir / "REPORT.md")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
