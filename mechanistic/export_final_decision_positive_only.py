from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LETTERS = "ABCD"
CONDITIONS = ("game", "neutral")


def _entropy_bits(logits: np.ndarray) -> float:
    shifted = logits - logits.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--confirmation-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arrays = dict(np.load(args.results, allow_pickle=False))
    if not np.all(arrays["completed"]):
        raise ValueError("Results are incomplete")
    qids = arrays["question_ids"].astype(str).tolist()
    baseline = json.loads(args.baseline.read_text())["results"]
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    manifest = {
        row["id"]: row for row in json.loads(args.manifest.read_text())["questions"]
    }
    mapping = {
        row["question_id"]: row
        for row in json.loads(args.mapping_plan.read_text())["rows"]
    }
    discovery = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    confirmation = set(json.loads(args.confirmation_plan.read_text())["question_ids"])
    args.output.mkdir(parents=True, exist_ok=True)

    trial_path = args.output / "per_question_condition.csv"
    trial_fields = [
        "question_id", "split", "condition", "w1_original_content",
        "w2_original_content", "w1_w2_relation", "correct_original_content",
        "natural_answer_new_letter", "natural_answer_original_content",
        "positive_only_answer_new_letter", "positive_only_answer_original_content",
        "natural_selected_w1", "positive_only_selected_w1",
        "natural_selected_w2", "positive_only_selected_w2",
        "natural_accuracy", "positive_only_accuracy", "natural_entropy_bits",
        "positive_only_entropy_bits", "natural_centered_w1_logit",
        "positive_only_centered_w1_logit", "centered_w1_logit_change",
        "natural_w1_vs_w2_margin", "positive_only_w1_vs_w2_margin",
        "w1_vs_w2_margin_change",
    ]
    for prefix in ("natural", "positive_only"):
        trial_fields.extend(f"{prefix}_logit_original_{letter}" for letter in LETTERS)
        trial_fields.extend(f"{prefix}_logit_new_{letter}" for letter in LETTERS)

    layer_path = args.output / "per_question_condition_layer.csv"
    layer_fields = [
        "question_id", "split", "condition", "w1_original_content",
        "w2_original_content", "w1_w2_relation", "readout",
        "natural_w1_projection", "natural_residual_norm",
        "positive_only_projection_before_removal",
        "positive_only_projection_after_removal",
        "positive_only_residual_norm",
        "positive_projection_removed_this_readout",
    ]

    with trial_path.open("w", newline="") as trial_handle, layer_path.open(
        "w", newline=""
    ) as layer_handle:
        trial_writer = csv.DictWriter(trial_handle, fieldnames=trial_fields)
        layer_writer = csv.DictWriter(layer_handle, fieldnames=layer_fields)
        trial_writer.writeheader()
        layer_writer.writeheader()

        for qi, qid in enumerate(qids):
            split = "discovery" if qid in discovery else "confirmation"
            if qid not in discovery and qid not in confirmation:
                raise ValueError(f"Question missing from frozen splits: {qid}")
            w1 = baseline[qid]["answer"]
            w2 = remapped[qid]["answer_original_content"]
            correct = manifest[qid]["correct_answer"]
            original_to_new = mapping[qid]["original_to_new"]
            new_to_original = mapping[qid]["new_to_original"]
            content_indices = [LETTERS.index(original_to_new[letter]) for letter in LETTERS]
            w1_index = LETTERS.index(w1)
            w2_index = LETTERS.index(w2)
            relation = "agreement" if w1 == w2 else "conflict"

            for ci, condition in enumerate(CONDITIONS):
                natural_new = arrays["natural_logits"][ci, qi]
                positive_new = arrays["ablated_logits"][ci, qi]
                natural = natural_new[content_indices]
                positive = positive_new[content_indices]
                natural_answer_new = LETTERS[int(natural_new.argmax())]
                positive_answer_new = LETTERS[int(positive_new.argmax())]
                natural_answer = new_to_original[natural_answer_new]
                positive_answer = new_to_original[positive_answer_new]
                natural_centered_w1 = float(natural[w1_index] - natural.mean())
                positive_centered_w1 = float(positive[w1_index] - positive.mean())
                natural_margin = float(natural[w1_index] - natural[w2_index])
                positive_margin = float(positive[w1_index] - positive[w2_index])
                row = {
                    "question_id": qid,
                    "split": split,
                    "condition": condition,
                    "w1_original_content": w1,
                    "w2_original_content": w2,
                    "w1_w2_relation": relation,
                    "correct_original_content": correct,
                    "natural_answer_new_letter": natural_answer_new,
                    "natural_answer_original_content": natural_answer,
                    "positive_only_answer_new_letter": positive_answer_new,
                    "positive_only_answer_original_content": positive_answer,
                    "natural_selected_w1": int(natural_answer == w1),
                    "positive_only_selected_w1": int(positive_answer == w1),
                    "natural_selected_w2": int(natural_answer == w2),
                    "positive_only_selected_w2": int(positive_answer == w2),
                    "natural_accuracy": int(natural_answer == correct),
                    "positive_only_accuracy": int(positive_answer == correct),
                    "natural_entropy_bits": _entropy_bits(natural),
                    "positive_only_entropy_bits": _entropy_bits(positive),
                    "natural_centered_w1_logit": natural_centered_w1,
                    "positive_only_centered_w1_logit": positive_centered_w1,
                    "centered_w1_logit_change": positive_centered_w1 - natural_centered_w1,
                    "natural_w1_vs_w2_margin": natural_margin,
                    "positive_only_w1_vs_w2_margin": positive_margin,
                    "w1_vs_w2_margin_change": positive_margin - natural_margin,
                }
                for index, letter in enumerate(LETTERS):
                    row[f"natural_logit_original_{letter}"] = float(natural[index])
                    row[f"positive_only_logit_original_{letter}"] = float(positive[index])
                    row[f"natural_logit_new_{letter}"] = float(natural_new[index])
                    row[f"positive_only_logit_new_{letter}"] = float(positive_new[index])
                trial_writer.writerow(row)

                for layer in range(arrays["natural_projection"].shape[-1]):
                    pre = float(arrays["ablated_pre_projection"][ci, qi, layer])
                    layer_writer.writerow({
                        "question_id": qid,
                        "split": split,
                        "condition": condition,
                        "w1_original_content": w1,
                        "w2_original_content": w2,
                        "w1_w2_relation": relation,
                        "readout": layer + 1,
                        "natural_w1_projection": float(
                            arrays["natural_projection"][ci, qi, layer]
                        ),
                        "natural_residual_norm": float(
                            arrays["natural_residual_norm"][ci, qi, layer]
                        ),
                        "positive_only_projection_before_removal": pre,
                        "positive_only_projection_after_removal": float(
                            arrays["ablated_projection_after"][ci, qi, layer]
                        ),
                        "positive_only_residual_norm": float(
                            arrays["ablated_residual_norm"][ci, qi, layer]
                        ),
                        "positive_projection_removed_this_readout": max(pre, 0.0),
                    })

    print(trial_path)
    print(layer_path)


if __name__ == "__main__":
    main()
