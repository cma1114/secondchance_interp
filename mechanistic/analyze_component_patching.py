from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .io import read_metadata, shard_path


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _entropy(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    probability = np.exp(shifted); probability /= probability.sum(axis=-1, keepdims=True)
    return -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=-1)


def _load(root: Path, group: str, qids: list[str], key: str) -> np.ndarray:
    values = []
    for qid in qids:
        with np.load(shard_path(root, group, qid), allow_pickle=False) as data:
            value = data[key]
            if key == "boundary_canonical_logits": value = value[-1, -1]
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _metrics(values: np.ndarray, baseline: np.ndarray, winners: np.ndarray, correct: np.ndarray) -> dict[str, np.ndarray]:
    values, baseline = _center(values), _center(baseline)
    row = np.arange(len(values)); winner_score = values[row, winners]
    denominator = np.maximum(np.sum(baseline * baseline, axis=-1), 1e-12)
    choice = np.argmax(values, axis=-1)
    return {
        "compression": -np.sum((values - baseline) * baseline, axis=-1) / denominator,
        "winner_advantage": winner_score - (values.sum(axis=-1) - winner_score) / 3,
        "ad_entropy": _entropy(values),
        "ad_spread": values.std(axis=-1),
        "switch": (choice != winners).astype(float),
        "accuracy": (choice == correct).astype(float),
    }


def _macro_mean(values: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean([values[labels == label].mean() for label in range(4)]))


def _bootstrap(values: np.ndarray, labels: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Bootstrap within Baseline-winner strata, then give all four letters equal weight."""
    means = np.empty(samples)
    groups = [values[labels == label] for label in range(4)]
    for start in range(0, samples, 1000):
        stop = min(samples, start + 1000)
        group_means = []
        for group in groups:
            index = rng.integers(0, len(group), size=(stop - start, len(group)))
            group_means.append(group[index].mean(axis=1))
        means[start:stop] = np.mean(group_means, axis=0)
    return _macro_mean(values, labels), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _plot(rows: list[dict], output: Path) -> None:
    primary = [row for row in rows if row["target_condition"] == "incorrect" and row["metric"] in {"compression", "winner_advantage", "switch"}]
    scenarios = list(dict.fromkeys(row["scenario"] for row in primary))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis, metric, title in zip(axes, ("compression", "winner_advantage", "switch"), ("Compression", "Original-winner advantage", "Switch probability")):
        selected = [next(row for row in primary if row["scenario"] == scenario and row["metric"] == metric) for scenario in scenarios]
        mean = np.asarray([row["effect_mean"] for row in selected]); low = np.asarray([row["effect_ci_low"] for row in selected]); high = np.asarray([row["effect_ci_high"] for row in selected])
        if metric == "switch": mean, low, high = 100 * mean, 100 * low, 100 * high
        x = np.arange(len(scenarios)); axis.bar(x, mean, color="#3b6fb6"); axis.errorbar(x, mean, yerr=np.vstack([mean-low, high-mean]), fmt="none", color="black", capsize=3)
        axis.axhline(0, color="black", linewidth=.8); axis.set_title(title); axis.set_xticks(x, [value.replace("neutral_into_game__", "") for value in scenarios], rotation=55, ha="right")
        axis.set_ylabel("Neutral-output patch − natural Game" + (" (pp)" if metric == "switch" else ""))
    fig.savefig(output.with_suffix(".png"), dpi=220); fig.savefig(output.with_suffix(".svg")); plt.close(fig)


def analyze(natural_root: Path, patch_root: Path, plan_path: Path, output: Path, samples: int, seed: int) -> dict:
    plan = json.loads(plan_path.read_text()); qids = plan["confirmation_question_ids"]
    available = []
    for qid in qids:
        if all(shard_path(patch_root, group, qid).exists() for group in ["natural_game", "natural_neutral", *[s["id"] for s in plan["scenarios"]]]): available.append(qid)
    qids = available
    if not qids: raise FileNotFoundError("No complete component-patching confirmation questions")
    baseline = _load(natural_root, "baseline", qids, "boundary_canonical_logits")
    natural_game = _load(patch_root, "natural_game", qids, "final_canonical_logits")
    natural_neutral = _load(patch_root, "natural_neutral", qids, "final_canonical_logits")
    baseline_meta = []
    for qid in qids:
        with np.load(shard_path(natural_root, "baseline", qid), allow_pickle=False) as data: baseline_meta.append(read_metadata(data))
    winners = np.argmax(baseline, axis=-1); correct = np.asarray(["ABCD".index(value["correct_answer"]) for value in baseline_meta])
    natural = {"incorrect": _metrics(natural_game, baseline, winners, correct), "neutral": _metrics(natural_neutral, baseline, winners, correct)}
    gap = {metric: natural["incorrect"][metric] - natural["neutral"][metric] for metric in natural["incorrect"]}
    rows, letter_rows, summaries = [], [], {}; rng = np.random.default_rng(seed)
    for scenario in plan["scenarios"]:
        values = _load(patch_root, scenario["id"], qids, "final_canonical_logits")
        target = scenario["target_condition"]; metrics = _metrics(values, baseline, winners, correct); scenario_rows = {}
        for metric, result in metrics.items():
            effect = result - natural[target][metric]; mean, low, high = _bootstrap(effect, winners, samples, rng)
            gap_mean = _macro_mean(gap[metric], winners)
            desired_sign = -1 if target == "incorrect" else 1
            fraction = desired_sign * mean / gap_mean if abs(gap_mean) > 1e-12 else None
            row = {"scenario": scenario["id"], "source_condition": scenario["source_condition"], "target_condition": target, "metric": metric, "natural_mean": _macro_mean(natural[target][metric], winners), "intervened_mean": _macro_mean(result, winners), "effect_mean": mean, "effect_ci_low": low, "effect_ci_high": high, "natural_game_minus_neutral_gap": gap_mean, "fraction_gap_mediated": fraction}
            rows.append(row); scenario_rows[metric] = row
            for label in range(4):
                mask = winners == label
                letter_rows.append({
                    "scenario": scenario["id"], "source_condition": scenario["source_condition"],
                    "target_condition": target, "metric": metric, "baseline_winner": "ABCD"[label],
                    "n_questions": int(mask.sum()), "natural_mean": float(natural[target][metric][mask].mean()),
                    "intervened_mean": float(result[mask].mean()), "effect_mean": float(effect[mask].mean()),
                    "natural_game_minus_neutral_gap": float(gap[metric][mask].mean()),
                })
        summaries[scenario["id"]] = scenario_rows
    output.mkdir(parents=True, exist_ok=True)
    with (output / "component_patch_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output / "component_patch_effects_by_letter.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(letter_rows[0])); writer.writeheader(); writer.writerows(letter_rows)
    _plot(rows, output / "component_patch_effects")
    summary = {
        "n_confirmation_questions": len(qids),
        "aggregation": "equal-weight macro-average across Baseline-winner A/B/C/D strata",
        "baseline_winner_counts": {"ABCD"[label]: int(np.sum(winners == label)) for label in range(4)},
        "natural_game": {k: _macro_mean(v, winners) for k,v in natural["incorrect"].items()},
        "natural_neutral": {k: _macro_mean(v, winners) for k,v in natural["neutral"].items()},
        "natural_game_unweighted": {k: float(v.mean()) for k,v in natural["incorrect"].items()},
        "natural_neutral_unweighted": {k: float(v.mean()) for k,v in natural["neutral"].items()},
        "scenarios": summaries,
    }
    (output / "component_patch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    lines = ["# Qwen3.6-27B sublayer compression and causal replacement", "", f"Held-out confirmation questions: {len(qids)}.", "", "Effects below replace selected final-position Game component outputs with the paired same-question Neutral outputs. Estimates give Baseline-winner A/B/C/D strata equal weight; intervals are stratified paired-bootstrap 95% CIs.", "", "| Scenario | Compression effect | Winner-advantage effect | Switch effect |", "|---|---:|---:|---:|"]
    for scenario in plan["scenarios"]:
        if scenario["target_condition"] != "incorrect": continue
        values = summaries[scenario["id"]]
        def fmt(metric: str, scale: float = 1) -> str:
            row = values[metric]; return f"{scale*row['effect_mean']:+.4f} [{scale*row['effect_ci_low']:+.4f}, {scale*row['effect_ci_high']:+.4f}]"
        lines.append(f"| {scenario['id'].replace('neutral_into_game__','')} | {fmt('compression')} | {fmt('winner_advantage')} | {fmt('switch',100)} pp |")
    lines.extend(["", "A component supports causal mediation of Game compression if Neutral→Game replacement decreases compression, restores original-winner advantage, and reduces switching; reciprocal Game→Neutral replacement should move those outcomes oppositely.", ""])
    (output / "COMPONENT_PATCH_REPORT.md").write_text("\n".join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired component-output replacements")
    parser.add_argument("--natural-root", required=True); parser.add_argument("--patch-root", required=True); parser.add_argument("--plan", required=True); parser.add_argument("--output", required=True); parser.add_argument("--bootstrap-samples", type=int, default=10_000); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); analyze(Path(args.natural_root), Path(args.patch_root), Path(args.plan), Path(args.output), args.bootstrap_samples, args.seed)


if __name__ == "__main__": main()
