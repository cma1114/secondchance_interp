from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LETTERS = "ABCD"
SCENARIOS = ("identity_cached", "d_line_kv", "identity_trajectory", "d_closing_trajectory", "full_history")
BASELINES = {"d_line_kv": 0, "d_closing_trajectory": 2}


def _semantic_logits(displayed: np.ndarray, qids: np.ndarray, current: str, rows: dict) -> np.ndarray:
    semantic = np.empty_like(displayed)
    for qi, qid in enumerate(qids):
        mapping = rows[qid][f"current_{current}_new_to_original"]
        for shown, original in mapping.items():
            semantic[..., qi, LETTERS.index(original)] = displayed[..., qi, LETTERS.index(shown)]
    return semantic - semantic.mean(axis=-1, keepdims=True)


def _transfer(values: np.ndarray, scenario: int, baseline: int, high: int, low: int) -> np.ndarray:
    return 0.5 * ((values[scenario, low] - values[baseline, low]) - (values[scenario, high] - values[baseline, high]))


def _geometry(value: np.ndarray, reference: np.ndarray, targets: np.ndarray, rng: np.random.Generator, samples: int = 5000) -> dict:
    def one(v: np.ndarray, r: np.ndarray, t: np.ndarray) -> tuple[float, float, float, float]:
        denom = float(np.sum(r * r))
        slope = float(np.sum(v * r) / denom) if denom else np.nan
        norm = float(np.sqrt(np.sum(v * v) * denom))
        cosine = float(np.sum(v * r) / norm) if norm else np.nan
        mask = np.ones(v.shape, dtype=bool)
        for qi, target in enumerate(t):
            mask[qi, LETTERS.index(target)] = False
        vn, rn = v[mask], r[mask]
        ndenom = float(np.sum(rn * rn))
        nslope = float(np.sum(vn * rn) / ndenom) if ndenom else np.nan
        nnorm = float(np.sqrt(np.sum(vn * vn) * ndenom))
        ncosine = float(np.sum(vn * rn) / nnorm) if nnorm else np.nan
        return slope, cosine, nslope, ncosine

    point = one(value, reference, targets)
    indices = rng.integers(0, len(value), size=(samples, len(value)))
    boot = np.asarray([one(value[index], reference[index], targets[index]) for index in indices])
    names = ("all_candidate_slope", "all_candidate_cosine", "nontarget_slope", "nontarget_cosine")
    result = {"n": int(len(value))}
    for column, name in enumerate(names):
        low, high = np.nanquantile(boot[:, column], [0.025, 0.975])
        result[name] = {"value": point[column], "low": float(low), "high": float(high)}
    return result


def analyze(cohort_path: Path, discovery: Path, confirmation: Path, output: Path) -> None:
    cohort = json.loads(cohort_path.read_text())
    rows = {row["question_id"]: row for row in cohort["rows"]}
    rng = np.random.default_rng(20260821)
    result = {"definition": "Regression/cosine of the pooled-current four-semantic-candidate transfer vector against the exact complete-history transfer vector. Nontarget metrics exclude the candidate fixed on literal D."}
    for split, path in (("discovery", discovery), ("confirmation", confirmation)):
        with np.load(path, allow_pickle=False) as loaded:
            keep = loaded["exact_eligible"].astype(bool)
            qids = loaded["question_ids"].astype(str)[keep]
            targets = loaded["semantic_targets"].astype(str)[keep]
            displayed = loaded["final_logits"][:, :, :, keep, :].astype(np.float64)
        semantic = np.stack([
            _semantic_logits(displayed[index], qids, current, rows)
            for index, current in enumerate(("low", "high"))
        ])
        split_result = {}
        for condition, (high_cell, low_cell) in {"Game": (0, 2), "Neutral": (1, 3)}.items():
            full = np.mean([
                _transfer(semantic[current], 4, 2, high_cell, low_cell)
                for current in range(2)
            ], axis=0)
            split_result[condition] = {}
            for scenario in ("d_line_kv", "d_closing_trajectory"):
                scenario_index = SCENARIOS.index(scenario)
                baseline = BASELINES[scenario]
                value = np.mean([
                    _transfer(semantic[current], scenario_index, baseline, high_cell, low_cell)
                    for current in range(2)
                ], axis=0)
                split_result[condition][scenario] = _geometry(value, full, targets, rng)
        result[split] = split_result
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.cohort, args.discovery, args.confirmation, args.output)


if __name__ == "__main__":
    main()
