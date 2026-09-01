"""Per-condition attribution of the confidence dose-response.

The primary dose-response analysis measures the Game-minus-Neutral old-winner
suppression, which either condition can move. This supplement decomposes it:
for Game and Neutral separately, it measures the old winner's final-position
centered logit relative to its own first-presentation centered logit, and the
slope of that change on first-pass confidence. A Game-side claim about
confidence-scaled suppression requires the Game column, not the difference.

Caveat stated up front: the two conditions share everything except the one
feedback word, including ordinary second-reading re-scoring of extreme
first-pass scores, so per-condition slopes conflate policy with shared
re-scoring. The Game-minus-Neutral difference cancels that shared part; these
per-condition numbers are for attribution, not precision. Evidence class:
descriptive/observational.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_confidence_dose_response import (
    CELLS,
    DATASET_LABELS,
    MODEL_SPECS,
    _load_cell,
)


def per_condition_change(
    baseline_logits: np.ndarray,
    direct_logits: np.ndarray,
    rank_order: np.ndarray,
) -> dict[str, np.ndarray]:
    baseline_logits = np.asarray(baseline_logits, dtype=np.float64)
    centered_baseline = baseline_logits - baseline_logits.mean(axis=1, keepdims=True)
    rows = np.arange(baseline_logits.shape[0])
    old_w1 = np.asarray(rank_order, dtype=np.int64)[:, 0]
    w1_first_pass = centered_baseline[rows, old_w1]
    centered_final = direct_logits - direct_logits.mean(axis=2, keepdims=True)
    return {
        "delta_w1_game": centered_final[0][rows, old_w1] - w1_first_pass,
        "delta_w1_neutral": centered_final[1][rows, old_w1] - w1_first_pass,
    }


def _slope_record(
    confidence: np.ndarray,
    outcome: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, float]:
    z = (confidence - confidence.mean()) / confidence.std(ddof=0)
    value = float(np.mean(z * (outcome - outcome.mean())))
    n = len(confidence)
    index = rng.integers(0, n, size=(draws, n))
    xb = confidence[index]
    zb = (xb - xb.mean(axis=1, keepdims=True)) / xb.std(axis=1, ddof=0, keepdims=True)
    yb = outcome[index]
    boot = np.mean(zb * (yb - yb.mean(axis=1, keepdims=True)), axis=1)
    return {
        "value": value,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    primary = json.loads((args.root / args.primary_summary).read_text())
    primary_slopes = {
        (cell["model"], cell["dataset"]): cell
        for cell in primary.get("cells", [])
    }
    summary: dict[str, Any] = {
        "analysis": "per_condition_confidence_scaling",
        "evidence_class": "descriptive/observational; per-condition slopes conflate policy with shared second-reading re-scoring",
        "definition": {
            "outcome": "final-position within-question-centered old-W1 logit minus the first-presentation within-question-centered old-W1 logit, per condition",
            "predictor": "z-scored first-pass top-1 minus top-2 logit confidence",
            "consistency": "Game slope minus Neutral slope must equal the negated primary W1-suppression slope",
            "bootstrap": f"question-level percentile bootstrap, seed {args.seed}, {args.draws} draws",
        },
        "cells": [],
    }
    for spec in CELLS:
        validation, quantities = _load_cell(args.root, spec)
        with np.load(args.root / spec["trajectory"], allow_pickle=False) as loaded:
            direct_logits = np.asarray(loaded["direct_logits"], dtype=np.float64)
            rank_order = np.asarray(loaded["rank_order"], dtype=np.int64)
        baseline = _reload_baseline(args.root, spec, quantities["question_ids"])
        deltas = per_condition_change(baseline, direct_logits, rank_order)
        confidence = quantities["confidence_c1"]
        game = _slope_record(confidence, deltas["delta_w1_game"], rng, args.draws)
        neutral = _slope_record(confidence, deltas["delta_w1_neutral"], rng, args.draws)
        record = {
            "model": validation["model"],
            "dataset": validation["dataset"],
            "n": validation["n"],
            "rank_order_gate_passed": validation["rank_order_gate_passed"],
            "mean_delta_w1_game": float(deltas["delta_w1_game"].mean()),
            "mean_delta_w1_neutral": float(deltas["delta_w1_neutral"].mean()),
            "game_slope_on_c1": game,
            "neutral_slope_on_c1": neutral,
        }
        key = (validation["model"], validation["dataset"])
        if key in primary_slopes:
            primary_value = primary_slopes[key]["splits"]["full"]["univariate"][
                "push_r1"]["raw_outcome_per_1sd_c1"]["value"]
            difference = game["value"] - neutral["value"]
            if abs(difference + primary_value) > 1e-9:
                raise ValueError(
                    f"Consistency failure for {key}: per-condition slopes do not "
                    f"reproduce the primary differential ({difference} vs {-primary_value})"
                )
            record["reproduces_primary_differential"] = True
        summary["cells"].append(record)
    return summary


def _reload_baseline(root: Path, spec: dict[str, Any], question_ids: np.ndarray) -> np.ndarray:
    document = json.loads((root / spec["baseline"]).read_text())
    for key in spec["baseline_container"]:
        document = document[key]
    return np.asarray(
        [document[str(qid)]["aggregated_ad_logits"] for qid in question_ids],
        dtype=np.float64,
    )


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Per-condition attribution of the confidence dose-response",
        "",
        "For each condition separately: the old winner's centered final logit minus",
        "its centered first-presentation logit, and that change's slope on z-scored",
        "first-pass confidence. The Game-minus-Neutral difference of the two slopes",
        "reproduces the primary dose-response slope exactly (asserted). Per-condition",
        "slopes include shared second-reading re-scoring, which the difference",
        "cancels; they attribute, they do not adjust. Descriptive only.",
        "",
        "| Model | Dataset | Game: mean change / slope on confidence | Neutral: mean change / slope on confidence |",
        "|---|---|---:|---:|",
    ]
    for cell in summary["cells"]:
        g, m = cell["game_slope_on_c1"], cell["neutral_slope_on_c1"]
        lines.append(
            f"| {cell['model']} | {cell['dataset']} "
            f"| {cell['mean_delta_w1_game']:+.3f} / {g['value']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}] "
            f"| {cell['mean_delta_w1_neutral']:+.3f} / {m['value']:+.3f} [{m['ci_low']:+.3f}, {m['ci_high']:+.3f}] |"
        )
    lines += [
        "",
        "Reading: only Qwen shows Game-side confidence scaling on both datasets",
        "(with Neutral flat on SimpleMC and far shallower on TriviaMC). Seed shows",
        "the same pattern on SimpleMC only. Gemma's Game slopes are flat; its",
        "SimpleMC Neutral slope is positive, so its differential dose-response may",
        "be Neutral-side reinstatement, supported in one cell only and therefore a",
        "hypothesis, not a finding.",
        "",
    ]
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--primary-summary", type=Path,
        default=Path("outputs/model_replications/confidence_dose_response/analysis/summary.json"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/model_replications/confidence_dose_response/per_condition_attribution"))
    parser.add_argument("--seed", type=int, default=20260902)
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
