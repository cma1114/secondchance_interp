from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .semantic_mapping import (
    align_displayed_logits_to_semantic,
    displayed_argmax_to_semantic_indices,
)


ROOT = Path("outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback")
MAPPING_PATH = Path("outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/plan.json")
LETTERS = "ABCD"


EXPERIMENTS = {
    "continuous final-decision signed ablation": (
        "final_decision_semantic_ablation",
        ("natural_logits", "ablated_logits"),
        False,
    ),
    "first-span GLA lesions": (
        "first_span_gla_ablation",
        ("natural_logits", "ablated_logits"),
        False,
    ),
    "first-decision cross-order patching": (
        "first_decision_cross_order_patching",
        ("natural_logits", "donor_patched_logits", "identity_patched_logits"),
        True,
    ),
    "first-boundary GLA memory rewrite": (
        "first_boundary_gla_memory_rewrite",
        ("natural_logits", "donor_patched_logits", "identity_patched_logits"),
        True,
    ),
    "first-boundary accumulated-state transplant": (
        "first_boundary_gla_state_transplant",
        (
            "natural_logits",
            "identity_state_logits",
            "different_winner_state_logits",
            "same_winner_state_logits",
        ),
        True,
    ),
}

CURRENT_ROOT = Path(
    "outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping"
)
CURRENT_RESULT_PATTERNS = (
    "feedback_factorial/evaluation_update_transplant/*/results.npz",
    "feedback_factorial/action_period_mediation/run/results.npz",
    "feedback_factorial/action_period_source_lesion/run/results.npz",
    "all_candidate_matched_relay/run/results.npz",
    "all_candidate_matched_relay_full_range/run/results.npz",
    "receiver_path_search/validation/results.npz",
    "repeated_w1_relay/run/results.npz",
    "final_query_repeated_option_ablation/run/results.npz",
    "final_query_edge_ablation/corrected_run/results.npz",
    "nonmatching_history_factorial/run/results.npz",
    "joint_option_score_decision_letter/run/results.npz",
    "option_newline_value_causal/*/results.npz",
    "option_newline_all_four_centered_projection/run_results.npz",
    "second_presentation_residual_workspace/policy_rank_factorial/run/results.npz",
)


def _mapping_rows(path: Path, qids: list[str], donor_plan: bool) -> list[dict]:
    if donor_plan:
        return json.loads((path.parent / "donor_plan.json").read_text())["rows"]
    by_qid = {
        row["question_id"]: row for row in json.loads(MAPPING_PATH.read_text())["rows"]
    }
    return [by_qid[qid] for qid in qids]


def _original_to_new(row: dict) -> dict[str, str]:
    mapping = row.get("second_mapping", row)
    return mapping["original_to_new"]


def _answers(values: np.ndarray, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    aligned = np.empty_like(values)
    correct = np.empty(values.shape[:-1], dtype=np.int64)
    displayed = values.argmax(axis=-1)
    for qi, row in enumerate(rows):
        original_to_new = _original_to_new(row)
        new_to_original = {new: original for original, new in original_to_new.items()}
        for content_index, content in enumerate(LETTERS):
            aligned[..., qi, content_index] = values[
                ..., qi, LETTERS.index(original_to_new[content])
            ]
        for new_index, new_letter in enumerate(LETTERS):
            correct[..., qi] = np.where(
                displayed[..., qi] == new_index,
                LETTERS.index(new_to_original[new_letter]),
                correct[..., qi],
            )
    return aligned.argmax(axis=-1), correct


def _audit_current_result(path: Path, mappings_by_qid: dict[str, dict]) -> list[dict]:
    output: list[dict] = []
    with np.load(path, allow_pickle=False) as arrays:
        if "question_ids" not in arrays:
            return output
        qids = arrays["question_ids"].astype(str).tolist()
        mapping_rows = [mappings_by_qid[qid] for qid in qids]
        for key in arrays.files:
            values = arrays[key]
            # Only output A-D logit tensors have the question axis immediately
            # before their final four displayed-letter coordinates.
            if (
                "logits" not in key
                or values.ndim < 2
                or values.shape[-2:] != (len(qids), 4)
            ):
                continue
            old = align_displayed_logits_to_semantic(values, mapping_rows).argmax(-1)
            corrected = displayed_argmax_to_semantic_indices(values, mapping_rows)
            tied = (values == values.max(axis=-1, keepdims=True)).sum(axis=-1) > 1
            output.append(
                {
                    "experiment": str(path.parent.relative_to(CURRENT_ROOT)),
                    "split": "stored",
                    "array": key,
                    "answer_cells": int(old.size),
                    "exact_max_ties": int(tied.sum()),
                    "changed_tie_breaks": int(np.sum(old != corrected)),
                    "percent": 100 * float(np.mean(old != corrected)),
                }
            )
    return output


def main() -> None:
    rows_out = []
    for label, (directory, keys, donor_plan) in EXPERIMENTS.items():
        for split in ("discovery", "confirmation"):
            split_name = split
            if directory == "first_decision_cross_order_patching":
                split_name += "_grouped_exact"
            path = ROOT / directory / split_name / "results.npz"
            arrays = np.load(path, allow_pickle=False)
            qids = arrays["question_ids"].astype(str).tolist()
            mappings = _mapping_rows(path, qids, donor_plan)
            for key in keys:
                old, corrected = _answers(arrays[key], mappings)
                changed = int(np.sum(old != corrected))
                values = arrays[key]
                tied = (values == values.max(axis=-1, keepdims=True)).sum(axis=-1) > 1
                rows_out.append(
                    {
                        "experiment": label,
                        "split": split,
                        "array": key,
                        "answer_cells": int(old.size),
                        "exact_max_ties": int(tied.sum()),
                        "changed_tie_breaks": changed,
                        "percent": 100 * changed / old.size,
                    }
                )

    mappings_by_qid = {
        str(row["question_id"]): row
        for row in json.loads(MAPPING_PATH.read_text())["rows"]
    }
    current_paths = sorted(
        {
            path
            for pattern in CURRENT_RESULT_PATTERNS
            for path in CURRENT_ROOT.glob(pattern)
        }
    )
    for path in current_paths:
        rows_out.extend(_audit_current_result(path, mappings_by_qid))

    output = ROOT / "REMAPPED_ARGMAX_TIE_AUDIT.json"
    output.write_text(json.dumps({"rows": rows_out}, indent=2) + "\n")
    report = [
        "# Remapped A-D argmax tie audit",
        "",
        "The affected analysis pattern reordered displayed A-D logits into original semantic-content order and then took `argmax`. On exact ties this changed the model's actual displayed-letter tie-break. The corrected analysis resolves the displayed A-D winner first and only then maps it to semantic content.",
        "",
        "| Experiment | Split | Result array | Exact max ties | Changed semantic choices | Total cells |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows_out:
        report.append(
            f"| {row['experiment']} | {row['split']} | `{row['array']}` | "
            f"{row['exact_max_ties']} | {row['changed_tie_breaks']} ({row['percent']:.2f}%) | "
            f"{row['answer_cells']} |"
        )
    report.extend(
        [
            "",
            "Only discrete answer identities and quantities derived from them (selection, change, accuracy, and transition counts) can move. Raw A-D logits, margins, entropy, projections, activation norms, and all other continuous causal effects are invariant to this correction.",
        ]
    )
    (ROOT / "REMAPPED_ARGMAX_TIE_AUDIT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
