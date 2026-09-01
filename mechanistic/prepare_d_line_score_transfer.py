from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LETTERS = "ABCD"


def _rank(logits: np.ndarray, mapping: dict[str, str], semantic: str) -> int:
    semantic_logits = np.asarray(
        [logits[LETTERS.index(mapping["original_to_new"][item])] for item in LETTERS]
    )
    value = semantic_logits[LETTERS.index(semantic)]
    return 1 + int(np.sum(semantic_logits > value))


def prepare(
    screen_path: Path,
    mapping_plan_path: Path,
    split_path: Path,
    output_path: Path,
    cap_per_split: int,
) -> None:
    mappings = json.loads(mapping_plan_path.read_text())["mappings"]
    split = json.loads(split_path.read_text())
    discovery = set(split["question_ids"])
    with np.load(screen_path, allow_pickle=False) as loaded:
        qids = loaded["question_ids"].astype(str)
        logits = loaded["aggregated_ad_logits"].astype(np.float64)
        starts = loaded["option_line_starts"].astype(np.int64)
        ends = loaded["option_line_ends"].astype(np.int64)
        completed = loaded["completed"].astype(bool)
    if not completed.all() or not np.isfinite(logits).all():
        raise ValueError("The complete 24-mapping screen is incomplete or non-finite")
    if logits.shape[:2] != (len(mappings), len(qids)):
        raise ValueError("Screen and mapping plan disagree")

    eligible: list[dict] = []
    for qi, qid in enumerate(qids):
        candidates: list[dict] = []
        for semantic in LETTERS:
            mapping_indices = [
                index
                for index, mapping in enumerate(mappings)
                if mapping["new_to_original"]["D"] == semantic
            ]
            if len(mapping_indices) != 6:
                raise RuntimeError("Expected six permutations with a fixed D candidate")
            records = []
            for index in mapping_indices:
                mapping = mappings[index]
                displayed = LETTERS.index(mapping["original_to_new"][semantic])
                centered = float(logits[index, qi, displayed] - logits[index, qi].mean())
                records.append(
                    {
                        "mapping_index": index,
                        "centered_score": centered,
                        "rank": _rank(logits[index, qi], mapping, semantic),
                    }
                )
            low = min(records, key=lambda row: (row["centered_score"], row["mapping_index"]))
            high = max(records, key=lambda row: (row["centered_score"], -row["mapping_index"]))
            if starts[low["mapping_index"], qi, 3] != starts[high["mapping_index"], qi, 3]:
                raise RuntimeError("Paired D lines start at different token positions")
            if ends[low["mapping_index"], qi, 3] != ends[high["mapping_index"], qi, 3]:
                raise RuntimeError("Paired D lines end at different token positions")
            candidates.append(
                {
                    "semantic_target": semantic,
                    "old_low": low,
                    "old_high": high,
                    "old_score_gap": high["centered_score"] - low["centered_score"],
                    "rank_changed": high["rank"] != low["rank"],
                    "winner_crossing": high["rank"] == 1 and low["rank"] > 1,
                }
            )
        chosen = max(
            candidates,
            key=lambda row: (
                row["old_score_gap"],
                row["winner_crossing"],
                row["rank_changed"],
                -LETTERS.index(row["semantic_target"]),
            ),
        )
        low_index = chosen["old_low"]["mapping_index"]
        high_index = chosen["old_high"]["mapping_index"]
        digest = hashlib.sha256(f"d-line-score-transfer:{qid}".encode()).hexdigest()
        eligible.append(
            {
                "question_id": qid,
                "split": "discovery" if qid in discovery else "confirmation",
                "semantic_target": chosen["semantic_target"],
                "old_low_mapping_index": low_index,
                "old_high_mapping_index": high_index,
                "old_low_new_to_original": mappings[low_index]["new_to_original"],
                "old_high_new_to_original": mappings[high_index]["new_to_original"],
                "current_low_mapping_index": low_index,
                "current_high_mapping_index": high_index,
                "current_low_new_to_original": mappings[low_index]["new_to_original"],
                "current_high_new_to_original": mappings[high_index]["new_to_original"],
                "old_low_score": chosen["old_low"]["centered_score"],
                "old_high_score": chosen["old_high"]["centered_score"],
                "old_score_gap": chosen["old_score_gap"],
                "old_low_rank": chosen["old_low"]["rank"],
                "old_high_rank": chosen["old_high"]["rank"],
                "rank_changed": chosen["rank_changed"],
                "winner_crossing": chosen["winner_crossing"],
                "d_line_start": int(starts[low_index, qi, 3]),
                "d_line_end": int(ends[low_index, qi, 3]),
                "sampling_hash": digest,
            }
        )

    selected: list[dict] = []
    available = {}
    for name in ("discovery", "confirmation"):
        rows = [row for row in eligible if row["split"] == name]
        available[name] = {
            "all": len(rows),
            "rank_changed": sum(row["rank_changed"] for row in rows),
            "winner_crossing": sum(row["winner_crossing"] for row in rows),
        }
        # Rank-changing rows answer the scientific question most directly.
        # Winner crossings are not privileged over other rank changes after
        # eligibility; hash order prevents selecting on causal outcomes.
        ranked = sorted(
            rows,
            key=lambda row: (
                not row["rank_changed"],
                row["sampling_hash"],
            ),
        )
        selected.extend(ranked[:cap_per_split])
    selected.sort(key=lambda row: (row["split"], row["sampling_hash"]))
    payload = {
        "status": "frozen before causal inference",
        "definition": (
            "For each question choose the semantic candidate with the largest "
            "first-pass centered-score range across the six permutations that keep "
            "that candidate on literal D. Pair its high- and low-score histories; "
            "the complete D line is therefore text- and position-identical, while "
            "the order of A-C differs. Use the same high/low mappings as two fixed "
            "second-presentation evidence strata."
        ),
        "screen_path": str(screen_path),
        "mapping_plan_path": str(mapping_plan_path),
        "split_path": str(split_path),
        "cap_per_split": cap_per_split,
        "available": available,
        "selected_counts": {
            name: sum(row["split"] == name for row in selected)
            for name in ("discovery", "confirmation")
        },
        "rows": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("available", "selected_counts")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--mapping-plan", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap-per-split", type=int, default=70)
    args = parser.parse_args()
    prepare(args.screen, args.mapping_plan, args.split, args.output, args.cap_per_split)


if __name__ == "__main__":
    main()
