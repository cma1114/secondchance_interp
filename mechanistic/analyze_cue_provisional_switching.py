from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LETTERS = np.asarray(list("ABCD"))
TASKS = ("Game", "Neutral")


def _interval(
    values: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    samples = np.zeros(draws, dtype=float)
    for group in groups:
        selected = rng.choice(group, size=(draws, len(group)), replace=True)
        samples += values[selected].sum(axis=1)
    samples /= len(values)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(len(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cue-scores", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--neutral", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=83022)
    args = parser.parse_args()

    cue_payload = json.loads(args.cue_scores.read_text())
    qids = np.asarray(cue_payload["question_ids"])
    split = np.asarray(cue_payload["split"])
    scores = np.asarray(cue_payload["scores"], dtype=float)
    baseline = json.loads(args.baseline.read_text())["results"]
    remapped_baseline = json.loads(args.remapped_baseline.read_text())["results"]
    task_rows = [
        json.loads(args.game.read_text())["results"],
        json.loads(args.neutral.read_text())["results"],
    ]

    w1 = np.asarray([baseline[qid]["answer"] for qid in qids])
    w2 = np.asarray([remapped_baseline[qid]["answer_original_content"] for qid in qids])
    conflict = w1 != w2
    cue_switch = np.zeros((2, len(qids)), dtype=float)
    final_switch = np.zeros_like(cue_switch)
    for task_index, rows in enumerate(task_rows):
        cue_new_letter = LETTERS[scores[task_index, :, 0].argmax(axis=-1)]
        cue_content = np.asarray(
            [rows[qid]["new_to_original"][letter] for qid, letter in zip(qids, cue_new_letter)]
        )
        final_content = np.asarray(
            [rows[qid]["aggregated_ad_answer_original_content"] for qid in qids]
        )
        cue_switch[task_index] = cue_content != w1
        final_switch[task_index] = final_content != w1

    rng = np.random.default_rng(args.seed)
    summaries: dict[str, dict] = {}
    split_masks = {
        "discovery": split == "discovery",
        "confirmation": split == "confirmation",
        "all": np.ones(len(qids), dtype=bool),
    }
    subset_masks = {
        "overall": np.ones(len(qids), dtype=bool),
        "conflict": conflict,
        "no_conflict": ~conflict,
    }
    for split_name, split_mask in split_masks.items():
        summaries[split_name] = {}
        for subset_name, subset_mask in subset_masks.items():
            mask = split_mask & subset_mask
            local_strata = w1[mask]
            cell: dict[str, object] = {"n": int(mask.sum()), "tasks": {}}
            for task_index, task in enumerate(TASKS):
                cue = cue_switch[task_index, mask]
                final = final_switch[task_index, mask]
                cell["tasks"][task] = {
                    "cue_switch": _interval(cue, local_strata, rng, args.draws),
                    "final_switch": _interval(final, local_strata, rng, args.draws),
                    "cue_minus_final": _interval(cue - final, local_strata, rng, args.draws),
                }
            cue_gap = cue_switch[0, mask] - cue_switch[1, mask]
            final_gap = final_switch[0, mask] - final_switch[1, mask]
            cell["Game_minus_Neutral"] = {
                "cue_switch_gap": _interval(cue_gap, local_strata, rng, args.draws),
                "final_switch_gap": _interval(final_gap, local_strata, rng, args.draws),
                "cue_minus_final_gap": _interval(
                    cue_gap - final_gap, local_strata, rng, args.draws
                ),
            }
            summaries[split_name][subset_name] = cell

    result = {
        "definition": (
            "The cue's provisional answer is the highest log-sum-exp A-D score at the "
            "final-layer residual of the trailing space after the post-list choice cue. "
            "The selected remapped letter is converted back to original semantic content. "
            "Switching means that content differs from the first-presentation winner W1."
        ),
        "statistics": "10,000-draw paired bootstrap stratified by W1 letter.",
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
