from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import LETTERS


def analyze(
    results_path: Path,
    plan_path: Path,
    baseline_path: Path,
    discovery_plan_path: Path,
    output_dir: Path,
) -> None:
    run = np.load(results_path)
    if not np.asarray(run["completed"], dtype=bool).all():
        raise ValueError("Permutation screen is incomplete")
    qids = [str(value) for value in run["question_ids"]]
    logits = np.asarray(run["aggregated_ad_logits"], dtype=np.float64)
    plan = json.loads(plan_path.read_text())
    rows = {row["question_id"]: row for row in plan["rows"]}
    baseline = json.loads(baseline_path.read_text())["results"]
    discovery_payload = json.loads(discovery_plan_path.read_text())
    discovery_ids = set(
        discovery_payload.get("question_ids", discovery_payload.get("discovery_question_ids"))
    )

    displayed_choice = logits.argmax(axis=-1)
    semantic_choice = np.empty_like(displayed_choice)
    target_index = np.empty(len(qids), dtype=int)
    for qi, qid in enumerate(qids):
        target_index[qi] = LETTERS.index(rows[qid]["w1_displayed_letter"])
        for mi, mapping in enumerate(rows[qid]["mappings"]):
            chosen_display = LETTERS[int(displayed_choice[mi, qi])]
            semantic_choice[mi, qi] = LETTERS.index(
                mapping["new_to_original"][chosen_display]
            )
    target_chosen = semantic_choice == target_index[None, :]
    trusted = np.asarray(
        [baseline[qid]["aggregated_ad_logits"] for qid in qids], dtype=np.float64
    )
    identity_choice_match = np.mean(displayed_choice[0] == trusted.argmax(axis=-1))
    max_identity_error = float(np.max(np.abs(logits[0] - trusted)))

    eligible_rows = []
    letter_summary = {}
    for letter in LETTERS:
        mask = np.asarray([rows[qid]["w1_displayed_letter"] == letter for qid in qids])
        identity_valid = target_chosen[0] & mask
        flips = identity_valid & (~target_chosen[1:]).any(axis=0)
        letter_summary[letter] = {
            "n_questions": int(mask.sum()),
            "identity_companion_still_selects_w1": int(identity_valid.sum()),
            "has_same_position_unchosen_permutation": int(flips.sum()),
            "eligible_fraction": float(flips.sum() / max(identity_valid.sum(), 1)),
        }

    for qi, qid in enumerate(qids):
        if not target_chosen[0, qi]:
            continue
        unchosen = np.flatnonzero(~target_chosen[:, qi])
        if not len(unchosen):
            continue
        w1_index = target_index[qi]
        centered = logits[:, qi, w1_index] - logits[:, qi].mean(axis=-1)
        donor_unchosen = int(unchosen[np.argmin(centered[unchosen])])
        chosen_controls = np.flatnonzero(target_chosen[1:, qi]) + 1
        donor_control = (
            int(chosen_controls[np.argmin(np.abs(centered[chosen_controls] - centered[0]))])
            if len(chosen_controls) else None
        )
        eligible_rows.append({
            "question_id": qid,
            "split": "discovery" if qid in discovery_ids else "confirmation",
            "w1_original_content": rows[qid]["w1_original_content"],
            "w1_displayed_letter": rows[qid]["w1_displayed_letter"],
            "chosen_mapping_index": 0,
            "unchosen_mapping_index": donor_unchosen,
            "chosen_control_mapping_index": donor_control,
            "w1_centered_logit_chosen": float(centered[0]),
            "w1_centered_logit_unchosen": float(centered[donor_unchosen]),
        })

    by_split = Counter(row["split"] for row in eligible_rows)
    by_letter = Counter(row["w1_displayed_letter"] for row in eligible_rows)
    summary = {
        "definition": (
            "W1 remains at its original displayed letter while the other three options "
            "are permuted through all six arrangements. Eligible questions select W1 in "
            "the identity companion but not in at least one same-position permutation."
        ),
        "n_questions": len(qids),
        "identity_choice_agreement_with_trusted": float(identity_choice_match),
        "max_abs_identity_logit_error": max_identity_error,
        "eligible_questions": len(eligible_rows),
        "eligible_by_split": dict(by_split),
        "eligible_by_w1_letter": dict(by_letter),
        "letter_summary": letter_summary,
        "w1_choice_rate_across_all_permutations": float(target_chosen.mean()),
        "mean_ad_probability_mass": float(np.asarray(run["ad_probability_mass"]).mean()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "eligible_pairs.json").write_text(
        json.dumps({"status": "frozen_after_screen", "rows": eligible_rows}, indent=2) + "\n"
    )
    lines = [
        "# W1-fixed option-permutation feasibility screen",
        "",
        "## Bottom line",
        "",
        f"The identity companion reproduced trusted choices on {identity_choice_match*100:.1f}% of questions; maximum A-D logit discrepancy was {max_identity_error:.4f}.",
        "",
        f"Keeping W1 at the identical displayed letter while permuting only the other three options produced {len(eligible_rows)}/500 questions where W1 was selected in the identity ordering but lost in at least one alternative ordering.",
        f"The frozen discovery/confirmation counts are {by_split.get('discovery', 0)}/{by_split.get('confirmation', 0)}.",
        "",
        "## By original W1 letter",
        "",
    ]
    for letter in LETTERS:
        row = letter_summary[letter]
        lines.append(
            f"- {letter}: {row['has_same_position_unchosen_permutation']}/{row['identity_companion_still_selects_w1']} eligible ({row['eligible_fraction']*100:.1f}%)."
        )
    lines.extend([
        "",
        "For W1=A, the W1 option line and every preceding token are byte-identical across all six prompts. Any winner-status difference is therefore necessarily constructed only after the model encounters later competitors.",
        "",
        "The eligible-pair artifact freezes one strongest unchosen permutation and, when available, one nonidentity chosen-status control for each question. These pairs are the appropriate cohort for locating and causally transplanting candidate strength.",
        "",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.results, args.plan, args.baseline, args.discovery_plan, args.output_dir)


if __name__ == "__main__":
    main()
