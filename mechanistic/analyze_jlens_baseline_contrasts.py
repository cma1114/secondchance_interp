from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_jlens_answer_content import RANKS, answer_letter_scores, baseline_rank_order
from .answer_emergence_figures import Z_975, macro_mean_and_se
from .data import load_activation_dataset


CONTRASTS = {
    "game_minus_baseline": (1, 0, "Game minus Baseline"),
    "neutral_minus_baseline": (2, 0, "Neutral minus Baseline"),
    "game_minus_neutral": (1, 2, "Game minus Neutral"),
}


def analyze(
    residual_root: Path,
    jlens_root: Path,
    output: Path,
    baseline_residual_root: Path | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    # Only Baseline activations are required: they define the generated winner,
    # runner-up, and letter-balanced strata.  This optional separate root lets
    # corrected Second Chance JLens runs reuse the token-identical Baseline run
    # without pretending their condition shards live in one directory.
    baseline_root = baseline_residual_root or residual_root
    data = load_activation_dataset(baseline_root, ["baseline"])
    order, prior = baseline_rank_order(data)
    layout = json.loads((jlens_root / "selected_token_layout.json").read_text())
    with np.load(jlens_root / "jlens_scores.npz", allow_pickle=False) as cached:
        if cached["question_ids"].astype(str).tolist() != data.question_ids:
            raise ValueError("JLens and activation question orders differ")
        scores = answer_letter_scores(cached["final_scores"].astype(np.float64), layout)
    scores -= scores.mean(axis=-1, keepdims=True)

    payload = {"layers": list(range(1, 65)), "ranks": list(RANKS), "contrasts": {}}
    csv_rows = []
    for name, (first, second, label) in CONTRASTS.items():
        aligned = np.take_along_axis(scores[first] - scores[second], order[:, None, :], axis=-1)
        series = []
        for rank, rank_label in enumerate(RANKS):
            mean, se = macro_mean_and_se(aligned[:, :, rank], prior)
            low, high = mean - Z_975 * se, mean + Z_975 * se
            series.append({
                "rank": rank_label,
                "mean": np.round(mean, 4).tolist(),
                "ci_low": np.round(low, 4).tolist(),
                "ci_high": np.round(high, 4).tolist(),
            })
            for layer, value, lower, upper in zip(range(1, 65), mean, low, high):
                csv_rows.append({
                    "contrast": name, "rank": rank_label, "layer": layer,
                    "mean": float(value), "ci_low": float(lower), "ci_high": float(upper),
                })
        payload["contrasts"][name] = {"label": label, "series": series}

    (output / "jlens_baseline_contrasts.json").write_text(json.dumps(payload, separators=(",", ":")))
    with (output / "jlens_baseline_contrasts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--baseline-residual-root", type=Path)
    parser.add_argument("--jlens-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(
        args.residual_root,
        args.jlens_root,
        args.output,
        args.baseline_residual_root,
    )


if __name__ == "__main__":
    main()
