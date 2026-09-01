from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RANK_LABELS = tuple(f"[Baseline rank {rank}]" for rank in range(1, 5))


def _baseline_order(question_ids: list[str], shard_root: Path) -> np.ndarray:
    order = np.empty((len(question_ids), 4), dtype=np.int64)
    for index, question_id in enumerate(question_ids):
        with np.load(shard_root / f"{question_id}.npz", allow_pickle=False) as shard:
            metadata = json.loads(str(shard["metadata"].item()))
            logits = shard["canonical_logits"][-1].astype(np.float64)
        winner = "ABCD".index(metadata["baseline_answer"])
        order[index, 0] = winner
        others = [option for option in range(4) if option != winner]
        order[index, 1:] = sorted(others, key=lambda option: logits[option], reverse=True)
    return order


def _bare_answer_indices(layout: list[dict]) -> list[int]:
    indices = []
    for letter in "ABCD":
        matches = [
            index
            for index, row in enumerate(layout)
            if row["family"] == f"answer_{letter}" and row["text"] == letter
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one bare-token entry for {letter}, found {matches}")
        indices.append(matches[0])
    return indices


def _align(scores: np.ndarray, order: np.ndarray) -> np.ndarray:
    shape = (1, order.shape[0]) + (1,) * (scores.ndim - 3) + (4,)
    aligned_order = np.broadcast_to(order.reshape(shape), scores.shape[:-1] + (4,))
    return np.take_along_axis(scores, aligned_order, axis=-1)


def _insert(row: dict, values: np.ndarray) -> None:
    for direction in ("top", "bottom"):
        row[direction] = [item for item in row[direction] if not item.get("tracked")]
    for rank, value in enumerate(values):
        item = {
            "token_id": -(rank + 1),
            "token": RANK_LABELS[rank],
            "score": float(value),
            "tracked": True,
        }
        row["top" if value >= 0 else "bottom"].append(item)
    row["top"].sort(key=lambda item: item["score"], reverse=True)
    row["bottom"].sort(key=lambda item: item["score"])


def augment(jlens_root: Path, baseline_shards: Path, output: Path) -> None:
    layout = json.loads((jlens_root / "selected_token_layout.json").read_text())
    bare_indices = _bare_answer_indices(layout)
    document = json.loads((jlens_root / "top_tokens.json").read_text())

    with np.load(jlens_root / "jlens_scores.npz", allow_pickle=False) as cached:
        final = cached["final_scores"].astype(np.float64)[..., bare_indices]
        positions = cached["position_scores"].astype(np.float64)[..., bare_indices]
        question_ids = cached["question_ids"].astype(str).tolist()
        position_question_ids = cached["position_question_ids"].astype(str).tolist()
        conditions = cached["conditions"].astype(str).tolist()
        position_conditions = cached["position_conditions"].astype(str).tolist()
        anchors = cached["anchors"].astype(str).tolist()
        availability = cached["position_availability"].astype(bool)
        display_question_ids = (
            cached["display_question_ids"].astype(str).tolist()
            if "display_question_ids" in cached.files
            else list(position_question_ids)
        )

    display_set = set(display_question_ids)
    final_display = np.asarray([qid in display_set for qid in question_ids], dtype=bool)
    position_display = np.asarray(
        [qid in display_set for qid in position_question_ids], dtype=bool
    )
    if not final_display.any() or not position_display.any():
        raise ValueError("No display/discovery questions found in the JLens score file")

    final_aligned = _align(final, _baseline_order(question_ids, baseline_shards))
    position_aligned = _align(positions, _baseline_order(position_question_ids, baseline_shards))

    final_means = final_aligned[:, final_display].mean(axis=1)
    for condition_index, condition in enumerate(conditions):
        for layer in range(64):
            _insert(document["final"][f"{condition}/L{layer}"], final_means[condition_index, layer])
    for first, second, name in (
        (1, 0, "game_minus_baseline"),
        (2, 0, "neutral_minus_baseline"),
        (1, 2, "game_minus_neutral"),
    ):
        contrast = (
            final_aligned[first, final_display]
            - final_aligned[second, final_display]
        ).mean(axis=0)
        for layer in range(64):
            _insert(document["final"][f"{name}/L{layer}"], contrast[layer])

    position_means = position_aligned[:, position_display].mean(axis=1)
    for condition_index, condition in enumerate(position_conditions):
        for anchor_index, anchor in enumerate(anchors):
            if not availability[condition_index, anchor_index]:
                continue
            for layer in range(64):
                _insert(
                    document["positions"][f"{condition}/{anchor}/L{layer}"],
                    position_means[condition_index, anchor_index, layer],
                )
    paired = (
        position_aligned[0, position_display]
        - position_aligned[1, position_display]
    ).mean(axis=0)
    for anchor_index, anchor in enumerate(anchors):
        if not availability[:, anchor_index].all():
            continue
        for layer in range(64):
            _insert(
                document["positions"][f"game_minus_neutral/{anchor}/L{layer}"],
                paired[anchor_index, layer],
            )

    document["rank_pseudotokens"] = {
        "labels": list(RANK_LABELS),
        "definition": (
            "Question-dependent bare A-D token selected by fixed Baseline answer rank, "
            "then averaged across questions on the same JLens-score scale as ordinary vocabulary tokens."
        ),
        "ranking": (
            "Rank 1 is the generated Baseline answer; ranks 2-4 are ordered by final Baseline canonical A-D logits."
        ),
        "aggregation": (
            f"Means use only the {int(final_display.sum())} frozen discovery questions listed in "
            "display_question_ids; confirmation questions are excluded."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Add fixed-Baseline-rank pseudo-tokens to JLens token lists")
    parser.add_argument("--jlens-root", type=Path, required=True)
    parser.add_argument("--baseline-shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    augment(args.jlens_root, args.baseline_shards, args.output)


if __name__ == "__main__":
    main()
