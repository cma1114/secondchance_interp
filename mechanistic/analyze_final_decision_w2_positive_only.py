from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_final_decision_semantic_ablation import LETTERS, _align, _entropy, _interval


def _semantic_answers(raw: np.ndarray, qids: list[str], plan: dict[str, dict]) -> np.ndarray:
    output = np.empty(raw.shape[:2], dtype=np.int64)
    for ci in range(raw.shape[0]):
        for qi, qid in enumerate(qids):
            displayed = LETTERS[int(raw[ci, qi].argmax())]
            output[ci, qi] = LETTERS.index(plan[qid]["new_to_original"][displayed])
    return output


def _load(
    results_path: Path,
    baseline_path: Path,
    remapped_path: Path,
    manifest_path: Path,
    mapping_plan_path: Path,
    question_plan_path: Path,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    arrays = dict(np.load(results_path, allow_pickle=False))
    if not np.all(arrays["completed"]):
        raise ValueError(f"Incomplete result: {results_path}")
    all_qids = arrays["question_ids"].astype(str).tolist()
    wanted = set(json.loads(question_plan_path.read_text())["question_ids"])
    selected_rows = np.asarray([qid in wanted for qid in all_qids])
    arrays = {
        key: (
            value[:, selected_rows]
            if value.ndim >= 2 and value.shape[0] == 2 and value.shape[1] == len(all_qids)
            else value[selected_rows]
            if value.ndim >= 1 and value.shape[0] == len(all_qids)
            else value
        )
        for key, value in arrays.items()
    }
    qids = arrays["question_ids"].astype(str).tolist()
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped = json.loads(remapped_path.read_text())["results"]
    manifest = {row["id"]: row for row in json.loads(manifest_path.read_text())["questions"]}
    plan = {row["question_id"]: row for row in json.loads(mapping_plan_path.read_text())["rows"]}
    natural = _align(arrays["natural_logits"], qids, plan)
    ablated = _align(arrays["ablated_logits"], qids, plan)
    natural_answers = _semantic_answers(arrays["natural_logits"], qids, plan)
    ablated_answers = _semantic_answers(arrays["ablated_logits"], qids, plan)
    w1 = np.asarray([LETTERS.index(baseline[qid]["answer"]) for qid in qids])
    w2 = np.asarray([LETTERS.index(remapped[qid]["answer_original_content"]) for qid in qids])
    correct = np.asarray([LETTERS.index(manifest[qid]["correct_answer"]) for qid in qids])
    conflict = w1 != w2
    if not np.any(conflict):
        raise ValueError("No W1/W2 conflicts")
    strata = w2[conflict]
    rows = np.arange(len(qids))
    rng = np.random.default_rng(seed)
    summary: dict[str, Any] = {
        "root": str(results_path.parent),
        "n_questions": len(qids),
        "n_conflict": int(conflict.sum()),
        "conditions": {},
    }
    trial_changes = {}
    for ci, condition in enumerate(("game", "neutral")):
        nat, abl = natural[ci], ablated[ci]
        nat_ans, abl_ans = natural_answers[ci], ablated_answers[ci]
        nat_w2 = nat[rows, w2]
        abl_w2 = abl[rows, w2]
        nat_w1 = nat[rows, w1]
        abl_w1 = abl[rows, w1]
        nat_centered = nat_w2 - nat.mean(-1)
        abl_centered = abl_w2 - abl.mean(-1)
        changes = {
            "w2_selection": (abl_ans == w2).astype(float) - (nat_ans == w2).astype(float),
            "w1_selection": (abl_ans == w1).astype(float) - (nat_ans == w1).astype(float),
            "switch_rate": (abl_ans != w1).astype(float) - (nat_ans != w1).astype(float),
            "accuracy": (abl_ans == correct).astype(float) - (nat_ans == correct).astype(float),
            "centered_w2_logit": abl_centered - nat_centered,
            "w2_vs_w1_margin": (abl_w2 - abl_w1) - (nat_w2 - nat_w1),
            "entropy_bits": _entropy(abl) - _entropy(nat),
        }
        trial_changes[condition] = changes
        summary["conditions"][condition] = {
            "natural": {
                "w2_selection": float(np.mean(nat_ans[conflict] == w2[conflict])),
                "w1_selection": float(np.mean(nat_ans[conflict] == w1[conflict])),
                "accuracy": float(np.mean(nat_ans[conflict] == correct[conflict])),
                "entropy_bits": float(_entropy(nat[conflict]).mean()),
                "centered_w2_logit": float(nat_centered[conflict].mean()),
                "w2_vs_w1_margin": float((nat_w2 - nat_w1)[conflict].mean()),
            },
            "positive_only_w2": {
                "w2_selection": float(np.mean(abl_ans[conflict] == w2[conflict])),
                "w1_selection": float(np.mean(abl_ans[conflict] == w1[conflict])),
                "accuracy": float(np.mean(abl_ans[conflict] == correct[conflict])),
                "entropy_bits": float(_entropy(abl[conflict]).mean()),
                "centered_w2_logit": float(abl_centered[conflict].mean()),
                "w2_vs_w1_margin": float((abl_w2 - abl_w1)[conflict].mean()),
            },
            "changes": {
                key: _interval(value[conflict], strata, rng, draws)
                for key, value in changes.items()
            },
        }
    summary["game_minus_neutral_changes"] = {
        key: _interval(
            (trial_changes["game"][key] - trial_changes["neutral"][key])[conflict],
            strata, rng, draws,
        )
        for key in trial_changes["game"]
    }
    summary["dose"] = {}
    for ci, condition in enumerate(("game", "neutral")):
        natural_projection = arrays["natural_projection"][ci, conflict]
        live_projection = arrays["ablated_pre_projection"][ci, conflict]
        summary["dose"][condition] = {
            "natural_positive_fraction_by_layer": (natural_projection > 0).mean(0).tolist(),
            "natural_positive_mean_by_layer": np.maximum(natural_projection, 0).mean(0).tolist(),
            "live_positive_fraction_by_layer": (live_projection > 0).mean(0).tolist(),
            "live_positive_mean_removed_by_layer": np.maximum(live_projection, 0).mean(0).tolist(),
        }
    return summary


def _fmt(x: dict[str, Any], scale: float = 1.0) -> str:
    return f"{x['mean']*scale:+.2f} [{x['ci'][0]*scale:+.2f}, {x['ci'][1]*scale:+.2f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--confirmation-plan", type=Path, required=True)
    parser.add_argument("--cosines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    discovery = _load(
        args.results, args.baseline, args.remapped_baseline, args.manifest,
        args.mapping_plan, args.discovery_plan, draws=args.draws, seed=args.seed,
    )
    confirmation = _load(
        args.results, args.baseline, args.remapped_baseline, args.manifest,
        args.mapping_plan, args.confirmation_plan, draws=args.draws, seed=args.seed + 1000,
    )
    cosine_arrays = dict(np.load(args.cosines, allow_pickle=False))
    cosine_qids = cosine_arrays["question_ids"].astype(str).tolist()
    baseline = json.loads(args.baseline.read_text())["results"]
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    cosine_conflict = np.asarray([
        baseline[qid]["answer"] != remapped[qid]["answer_original_content"]
        for qid in cosine_qids
    ])
    cosine = cosine_arrays["cosine"][cosine_conflict]
    cosine_summary = {
        "subset": "W1 != W2 conflict trials only",
        "n": int(cosine_conflict.sum()),
        "mean_by_layer": cosine.mean(0).tolist(),
        "median_by_layer": np.median(cosine, axis=0).tolist(),
        "l64_mean": float(cosine[:, -1].mean()),
        "l64_median": float(np.median(cosine[:, -1])),
    }
    summary = {
        "definitions": {
            "W1": "semantic answer selected in the original Baseline presentation",
            "W2": "semantic answer selected by a fresh Baseline solution of the remapped second presentation",
            "intervention": "At every final-position readout subtract max(h dot v_W2, 0) times the exact layer-specific four-mapping W2 direction.",
            "primary_subset": "W1 != W2 conflict trials",
        },
        "discovery": discovery,
        "confirmation": confirmation,
        "w1_w2_direction_cosine": cosine_summary,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Positive-only W2 semantic ablation at the final decision position",
        "",
        "This intervention removes only positive projection onto the exact semantic direction for W2, the answer selected by a fresh Baseline solution of the remapped second presentation. The primary analysis is restricted to conflict trials where W1 differs from W2.",
        "",
    ]
    for split_name, split in (("Discovery", discovery), ("Confirmation", confirmation)):
        lines.extend([
            f"## {split_name}", "",
            "| Condition | Natural W2 | Ablated W2 | W2 change | Natural W1 | Ablated W1 | W1 change | W2–W1 margin change | Accuracy change | Entropy change |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for condition in ("game", "neutral"):
            row = split["conditions"][condition]
            n, a, c = row["natural"], row["positive_only_w2"], row["changes"]
            lines.append(
                f"| {condition.title()} (n={split['n_conflict']}) | {n['w2_selection']:.1%} | {a['w2_selection']:.1%} | {_fmt(c['w2_selection'],100)} pp | "
                f"{n['w1_selection']:.1%} | {a['w1_selection']:.1%} | {_fmt(c['w1_selection'],100)} pp | {_fmt(c['w2_vs_w1_margin'])} | "
                f"{_fmt(c['accuracy'],100)} pp | {_fmt(c['entropy_bits'])} bits |"
            )
        contrast = split["game_minus_neutral_changes"]
        lines.extend([
            "",
            f"Game-minus-Neutral intervention difference in W2 selection: **{_fmt(contrast['w2_selection'],100)} pp**. In W1 selection: **{_fmt(contrast['w1_selection'],100)} pp**.",
            f"Because a semantic switch is any second answer other than W1, the switch-rate changes were **{_fmt(split['conditions']['game']['changes']['switch_rate'],100)} pp** in Game and **{_fmt(split['conditions']['neutral']['changes']['switch_rate'],100)} pp** in Neutral.",
            "",
        ])
    lines.extend([
        "## Direction overlap", "",
        f"On the {cosine_summary['n']} W1 != W2 conflict trials, the mean W1–W2 semantic-direction cosine at L64 is {cosine_summary['l64_mean']:+.3f} (median {cosine_summary['l64_median']:+.3f}). Layerwise values are in `summary.json`. This quantifies how much W2 removal may mechanically favor W1 because the two within-question contrast directions overlap negatively.",
        "",
        "## Validation", "",
        "A fresh natural companion measured W2 projection in the same forward regime as the intervention, and its A–D logits were required question-by-question to match the specified trusted 500-question reference exactly. The runner preserved each question's historical physical batch-of-four cohort, SDPA implementation, prompt serialization, and model revision. Exact W2 directions were reconstructed from the already-saved four mapping residual arrays, avoiding new baseline collection.",
    ])
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
