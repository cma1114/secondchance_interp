from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


LETTERS = "ABCD"


def _semantic_centered(
    displayed_logits: np.ndarray, mapping: dict[str, Any]
) -> np.ndarray:
    semantic = np.empty(4, dtype=np.float64)
    for displayed_index, displayed in enumerate(LETTERS):
        original = mapping["new_to_original"][displayed]
        semantic[LETTERS.index(original)] = displayed_logits[displayed_index]
    return semantic - semantic.mean()


def prepare(args: argparse.Namespace) -> None:
    with np.load(args.screen, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not arrays["completed"].astype(bool).all():
        raise RuntimeError("The all-permutation fresh-score screen is incomplete")
    qids = arrays["question_ids"].astype(str).tolist()
    plan = json.loads(args.permutation_plan.read_text())
    mappings = plan["mappings"]
    if len(mappings) != 24 or qids != plan["question_ids"]:
        raise RuntimeError("Expected the canonical complete 24-permutation screen")
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])

    rows: list[dict[str, Any]] = []
    for qi, qid in enumerate(qids):
        semantic = np.stack([
            _semantic_centered(arrays["aggregated_ad_logits"][mi, qi], mapping)
            for mi, mapping in enumerate(mappings)
        ])
        candidates: list[tuple[float, str, str, int, int]] = []
        for original in LETTERS:
            oi = LETTERS.index(original)
            for displayed in LETTERS:
                mapping_indices = [
                    mi for mi, mapping in enumerate(mappings)
                    if mapping["new_to_original"][displayed] == original
                ]
                for left, right in itertools.combinations(mapping_indices, 2):
                    # The target semantic answer remains at the same displayed
                    # position.  The complete option block and final prompt
                    # boundary must also be token aligned; only the order of
                    # the other three candidates is allowed to change.
                    aligned = (
                        arrays["option_line_starts"][left, qi, 0]
                        == arrays["option_line_starts"][right, qi, 0]
                        and arrays["option_line_ends"][left, qi, 3]
                        == arrays["option_line_ends"][right, qi, 3]
                        and arrays["prompt_lengths"][left, qi]
                        == arrays["prompt_lengths"][right, qi]
                    )
                    if not aligned:
                        continue
                    difference = abs(semantic[left, oi] - semantic[right, oi])
                    candidates.append((difference, original, displayed, left, right))
        if not candidates:
            raise RuntimeError(f"{qid}: no aligned fresh-score crossover pair")
        difference, original, displayed, left, right = max(
            candidates, key=lambda row: (row[0], row[1], row[2], -row[3], -row[4])
        )
        oi = LETTERS.index(original)
        low, high = sorted((left, right), key=lambda mi: semantic[mi, oi])
        rows.append({
            "pair_id": f"{qid}:{original}:{displayed}",
            "question_id": qid,
            "split": "discovery" if qid in discovery_ids else "confirmation",
            "target_original_letter": original,
            "target_displayed_letter": displayed,
            "low_mapping_index": int(low),
            "high_mapping_index": int(high),
            "low_new_to_original": mappings[low]["new_to_original"],
            "high_new_to_original": mappings[high]["new_to_original"],
            "low_fresh_semantic_centered_logits": semantic[low].tolist(),
            "high_fresh_semantic_centered_logits": semantic[high].tolist(),
            "target_low_fresh_score": float(semantic[low, oi]),
            "target_high_fresh_score": float(semantic[high, oi]),
            "target_fresh_score_difference": float(difference),
            "screen_option_block_start": int(arrays["option_line_starts"][low, qi, 0]),
            "screen_option_block_end": int(arrays["option_line_ends"][low, qi, 3]),
            "screen_prompt_length": int(arrays["prompt_lengths"][low, qi]),
        })

    differences = np.asarray([row["target_fresh_score_difference"] for row in rows])
    payload = {
        "status": "frozen_before_causal_inference",
        "definition": (
            "Within question, keep the 1P history, task, target semantic candidate, "
            "and target 2P displayed position fixed; permute only the other three 2P "
            "candidates and choose the token-aligned pair with the largest screened "
            "target candidate-centered fresh-score difference."
        ),
        "selection_data": str(args.screen),
        "selection_uses_causal_outcome": False,
        "questions": len(rows),
        "discovery": sum(row["split"] == "discovery" for row in rows),
        "confirmation": sum(row["split"] == "confirmation" for row in rows),
        "fresh_difference_quantiles": {
            str(q): float(value)
            for q, value in zip(
                (0, .1, .25, .5, .75, .9, 1),
                np.quantile(differences, (0, .1, .25, .5, .75, .9, 1)),
            )
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "questions": len(rows),
        "discovery": payload["discovery"],
        "confirmation": payload["confirmation"],
        "fresh_difference_quantiles": payload["fresh_difference_quantiles"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--permutation-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
