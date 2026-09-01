from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_rank_opposition import _baseline_order, _load, _rank_slope
from .component_causal_metrics import bootstrap, center, entropy
from .io import shard_path


def _load_reference(
    root: Path,
    group: str,
    qids: list[str],
    field: str,
    fallback: np.ndarray,
) -> np.ndarray:
    values = []
    for index, qid in enumerate(qids):
        with np.load(shard_path(root, group, qid), allow_pickle=False) as data:
            values.append(data[field] if field in data else fallback[index])
    return np.asarray(values, dtype=np.float64)


def _metrics(values: np.ndarray, order: np.ndarray, winners: np.ndarray) -> dict[str, np.ndarray]:
    centered = center(values)
    row = np.arange(len(values))
    winner = centered[row, winners]
    aligned = np.take_along_axis(centered, order, axis=-1)
    return {
        "rank_slope": _rank_slope(aligned),
        "switch": (np.argmax(centered, axis=-1) != winners).astype(float),
        "entropy": entropy(centered),
        "spread": centered.std(axis=-1),
        "winner_advantage": winner - (centered.sum(axis=-1) - winner) / 3.0,
    }


def analyze(
    patch_root: Path,
    plan_path: Path,
    ranking_root: Path,
    output: Path,
    samples: int = 10_000,
    seed: int = 42,
) -> dict:
    plan = json.loads(plan_path.read_text())
    qids = plan["question_ids"]
    order, winners = _baseline_order(ranking_root, qids)
    natural = {
        "incorrect": _load(patch_root, "natural_game", qids),
        "neutral": _load(patch_root, "natural_neutral", qids),
    }
    natural_metrics = {
        condition: _metrics(values, order, winners) for condition, values in natural.items()
    }
    rng = np.random.default_rng(seed)
    rows = []
    raw = {}
    for scenario in plan["scenarios"]:
        values = _load(patch_root, scenario["id"], qids)
        condition = scenario["target_condition"]
        reference = _load_reference(
            patch_root,
            scenario["id"],
            qids,
            "matched_natural_logits",
            natural[condition],
        )
        reference_metrics = _metrics(reference, order, winners)
        effects = {
            metric: reference_metrics[metric] - values_metric
            for metric, values_metric in _metrics(values, order, winners).items()
        }
        key = ("+".join(row["component"] for row in scenario["targets"]), scenario["source"])
        raw.setdefault(key, {})[condition] = effects
        for metric, effect in effects.items():
            mean, low, high = bootstrap(effect, winners, "letter_macro", samples, rng)
            rows.append({
                "targets": key[0],
                "source": key[1],
                "contrast": "game_removal" if condition == "incorrect" else "neutral_removal",
                "metric": metric,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            })
    for (targets, source), conditions in raw.items():
        if {"incorrect", "neutral"}.issubset(conditions):
            for metric in conditions["incorrect"]:
                effect = conditions["incorrect"][metric] - conditions["neutral"][metric]
                mean, low, high = bootstrap(effect, winners, "letter_macro", samples, rng)
                rows.append({
                    "targets": targets,
                    "source": source,
                    "contrast": "game_minus_neutral_removal",
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "attention_source_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"n_questions": len(qids), "rows": rows}
    (output / "attention_source_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze semantic attention-source ablations")
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--ranking-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    args = parser.parse_args()
    analyze(
        Path(args.patch_root),
        Path(args.plan),
        Path(args.ranking_root),
        Path(args.output),
        args.samples,
    )


if __name__ == "__main__":
    main()
