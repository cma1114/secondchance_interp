from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LETTERS = "ABCD"
SELECTED_BLOCKS = (33, 42, 43, 47, 49, 53, 61, 63)


def _interval(values: np.ndarray, rng: np.random.Generator, resamples: int) -> dict:
    draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci95": [float(low), float(high)],
        "n": int(len(values)),
    }


def analyze(
    results_path: Path,
    plan_path: Path,
    remapped_baseline_path: Path,
    output_path: Path,
    seed: int,
    resamples: int,
) -> None:
    with np.load(results_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"]):
        raise RuntimeError("Boundary-lens collection is incomplete")

    qids = arrays["question_ids"].astype(str)
    layers = (arrays["gla_layers_zero_based"].astype(int) + 1).tolist()
    scores = arrays["lens_ad_scores"].astype(np.float64)
    scores -= scores.mean(axis=-1, keepdims=True)
    plan = {row["question_id"]: row for row in json.loads(plan_path.read_text())["rows"]}
    remapped = json.loads(remapped_baseline_path.read_text())["results"]
    w1 = np.asarray([
        LETTERS.index(plan[qid]["baseline_content_new_letter"]) for qid in qids
    ])
    w2 = np.asarray([
        LETTERS.index(remapped[qid]["answer_new_letter"]) for qid in qids
    ])
    conflict = np.flatnonzero(w1 != w2)
    if len(conflict) != 273:
        raise RuntimeError(f"Expected 273 canonical W1 != W2 questions, found {len(conflict)}")

    rng = np.random.default_rng(seed)
    summary = {
        "definition": (
            "On canonical W1 != W2 questions, the contextual GLA effect is "
            "[(Evaluation after GLA - before GLA) - "
            "(Matched Neutral after GLA - before GLA)] on the centered W1-minus-W2 "
            "workspace-lens margin. Complete states are separately lensed before subtraction."
        ),
        "results": str(results_path),
        "plan": str(plan_path),
        "remapped_baseline": str(remapped_baseline_path),
        "conflict_questions": int(len(conflict)),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "blocks": {},
    }

    rows = np.arange(len(conflict))
    conflict_w1 = w1[conflict]
    conflict_w2 = w2[conflict]
    for lens_index, lens_name in enumerate(("J-lens", "R-lens")):
        lens_rows = {}
        for block in SELECTED_BLOCKS:
            layer_slot = layers.index(block)
            evaluation = scores[lens_index, 0, conflict, layer_slot]
            neutral = scores[lens_index, 1, conflict, layer_slot]

            def margin(state: np.ndarray) -> np.ndarray:
                return state[rows, conflict_w1] - state[rows, conflict_w2]

            contextual = (
                margin(evaluation[:, 1])
                - margin(evaluation[:, 0])
                - margin(neutral[:, 1])
                + margin(neutral[:, 0])
            )
            lens_rows[str(block)] = _interval(contextual, rng, resamples)
        summary["blocks"][lens_name] = lens_rows

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args()
    analyze(
        args.results,
        args.plan,
        args.remapped_baseline,
        args.output,
        args.seed,
        args.resamples,
    )


if __name__ == "__main__":
    main()

