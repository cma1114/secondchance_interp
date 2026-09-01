from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _interval(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    boot = values[indices].mean(axis=1)
    low, high = np.quantile(boot, (0.025, 0.975))
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--prior-results", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=640052)
    args = parser.parse_args()

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    with np.load(args.prior_results, allow_pickle=False) as loaded:
        prior = {key: loaded[key] for key in loaded.files}

    blocks = arrays["ordinary_blocks_one_based"].astype(int)
    expected = np.arange(4, 65, 4)
    if not np.array_equal(blocks, expected):
        raise RuntimeError(f"Expected complete blocks 4--64, got {blocks.tolist()}")
    if len(arrays["question_ids"]) != 500 or not arrays["completed"].astype(bool).all():
        raise RuntimeError("Expected all 500 questions complete")
    for key in ("attention_mass", "context_norm", "projected_write_norm", "mean_gate"):
        if arrays[key].shape != (2, 16, 500, 4) or not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"Invalid complete metric array: {key} {arrays[key].shape}")

    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    confirmation = np.asarray(
        [qid not in discovery_ids for qid in arrays["question_ids"].astype(str)]
    )
    if int(confirmation.sum()) != 249:
        raise RuntimeError("Expected frozen 249-question confirmation split")
    attention = arrays["attention_mass"][:, :, confirmation, :].astype(float)
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0, int(confirmation.sum()), size=(args.draws, int(confirmation.sum()))
    )

    block_index = {int(block): index for index, block in enumerate(blocks)}
    late: dict[str, object] = {}
    for block in (52, 56, 60, 64):
        bi = block_index[block]
        entry: dict[str, object] = {}
        for ci, condition in enumerate(("Game", "Neutral")):
            r1_minus_rest = attention[ci, bi, :, 0] - attention[ci, bi, :, 1:].mean(1)
            entry[f"{condition.lower()}_r1_minus_r2_r4"] = _interval(
                r1_minus_rest, bootstrap_indices
            )
        for rank in range(4):
            game_minus_neutral = attention[0, bi, :, rank] - attention[1, bi, :, rank]
            entry[f"game_minus_neutral_r{rank + 1}"] = _interval(
                game_minus_neutral, bootstrap_indices
            )
        rank_interaction = (
            attention[0, bi, :, 0] - attention[0, bi, :, 1:].mean(1)
        ) - (
            attention[1, bi, :, 0] - attention[1, bi, :, 1:].mean(1)
        )
        entry["game_minus_neutral_r1_selectivity"] = _interval(
            rank_interaction, bootstrap_indices
        )
        late[str(block)] = entry

    drops: dict[str, object] = {}
    for ci, condition in enumerate(("Game", "Neutral")):
        for rank in range(4):
            drop = attention[ci, block_index[56], :, rank] - attention[
                ci, block_index[52], :, rank
            ]
            drops[f"{condition.lower()}_r{rank + 1}_l56_minus_l52"] = _interval(
                drop, bootstrap_indices
            )

    remeasured = arrays["attention_mass"][:, :12].astype(float)
    old = prior["attention_mass"].astype(float)
    differences = np.abs(remeasured - old)
    natural_new = arrays["natural_logits"].argmax(-1)
    natural_old = prior["natural_logits"].argmax(-1)
    summary = {
        "validation": {
            "questions": 500,
            "confirmation": 249,
            "blocks": blocks.tolist(),
            "question_ids_exact_to_prior": bool(
                np.array_equal(arrays["question_ids"], prior["question_ids"])
            ),
            "prompt_hashes_exact_to_prior": bool(
                np.array_equal(arrays["prompt_hashes"], prior["prompt_hashes"])
            ),
            "ranks_exact_to_prior": bool(
                np.array_equal(arrays["rank_contents"], prior["rank_contents"])
            ),
            "natural_answer_agreement_to_prior": float((natural_new == natural_old).mean()),
            "remeasured_l4_l48_attention_abs_mean_difference": float(differences.mean()),
            "remeasured_l4_l48_attention_abs_p99_difference": float(
                np.quantile(differences, 0.99)
            ),
        },
        "late_attention": late,
        "l52_to_l56_change": drops,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
