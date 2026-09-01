from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .attention_ablation_config import AttentionAblationConfig
from .io import shard_path


def _load_final(path: Path, group: str, qids: list[str], key: str) -> np.ndarray:
    rows = []
    for qid in qids:
        with np.load(shard_path(path, group, qid), allow_pickle=False) as data:
            value = data[key]
            rows.append(value[-1] if value.ndim == 2 else value)
    return np.asarray(rows, dtype=np.float64)


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _entropy(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=-1)


def _metrics(values: np.ndarray, baseline: np.ndarray, winners: np.ndarray, runners: np.ndarray) -> dict[str, np.ndarray]:
    values = _center(values)
    baseline = _center(baseline)
    row = np.arange(len(values))
    winner_score = values[row, winners]
    runner_score = values[row, runners]
    denominator = np.sum(baseline * baseline, axis=-1)
    return {
        "compression": -np.sum((values - baseline) * baseline, axis=-1) / np.maximum(denominator, 1e-12),
        "winner_advantage": winner_score - (values.sum(axis=-1) - winner_score) / 3.0,
        "winner_score": winner_score,
        "runner_score": runner_score,
        "runner_minus_winner": runner_score - winner_score,
        "ad_spread": values.std(axis=-1),
        "ad_entropy": _entropy(values),
        "switch": (np.argmax(values, axis=-1) != winners).astype(float),
    }


def _bootstrap_mean(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float, float]:
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        stop = min(start + 1000, samples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _plot(rows: list[dict], output: Path) -> None:
    metrics = [
        ("compression", "Effect on compression"),
        ("winner_advantage", "Effect on original-winner advantage"),
        ("ad_entropy", "Effect on A–D entropy"),
        ("switch", "Effect on switch probability"),
    ]
    scenarios = []
    for row in rows:
        if row["metric"] == "compression":
            scenarios.append(row["scenario"])
    labels = [scenario.replace("ablate_", "").replace("__", "\n") for scenario in scenarios]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        selected = [next(row for row in rows if row["scenario"] == scenario and row["metric"] == metric) for scenario in scenarios]
        means = np.asarray([row["effect_mean"] for row in selected])
        low = np.asarray([row["effect_ci_low"] for row in selected])
        high = np.asarray([row["effect_ci_high"] for row in selected])
        x = np.arange(len(selected))
        colors = ["#999999" if "system" in scenario else "#3b6fb6" for scenario in scenarios]
        axis.bar(x, means, color=colors)
        axis.errorbar(x, means, yerr=np.vstack([means - low, high - means]), fmt="none", color="black", capsize=3)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_title(title)
        axis.set_xticks(x, labels=labels, rotation=35, ha="right")
        axis.set_ylabel("Ablated − natural Game")
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def analyze(config: AttentionAblationConfig, output_dir: Path) -> dict:
    causal_dir = Path(config.output_dir)
    all_scenario_ids = [scenario["id"] for scenario in config.scenarios]
    reference_scenario = "natural_game_rerun"
    scenario_ids = [scenario for scenario in all_scenario_ids if scenario != reference_scenario]
    sets = [
        {path.stem for path in (causal_dir / "shards" / scenario).glob("*.npz")}
        for scenario in all_scenario_ids
    ]
    sets.extend([
        {path.stem for path in (Path(config.natural_attention_dir) / "shards" / "incorrect").glob("*.npz")},
        {path.stem for path in (Path(config.mechanistic_dir) / "shards" / "baseline").glob("*.npz")},
    ])
    qids = sorted(set.intersection(*sets))
    if not qids:
        raise FileNotFoundError("No questions complete across all intervention and reference conditions")
    baseline = _load_final(Path(config.mechanistic_dir), "baseline", qids, "canonical_logits")
    if reference_scenario in all_scenario_ids:
        natural = _load_final(causal_dir, reference_scenario, qids, "final_canonical_logits")
        natural_source = "matched unmodified rerun"
        prior_natural = _load_final(
            Path(config.natural_attention_dir), "incorrect", qids, "final_canonical_logits"
        )
        reference_agreement = {
            "exact_trials": int(np.all(natural == prior_natural, axis=-1).sum()),
            "total_trials": len(qids),
            "maximum_absolute_logit_difference": float(np.max(np.abs(natural - prior_natural))),
        }
    else:
        natural = _load_final(Path(config.natural_attention_dir), "incorrect", qids, "final_canonical_logits")
        natural_source = "prior eager attention collection"
        reference_agreement = None
    order = np.argsort(baseline, axis=-1)
    winners, runners = order[:, -1], order[:, -2]
    natural_metrics = _metrics(natural, baseline, winners, runners)
    rng = np.random.default_rng(config.seed)
    rows = []
    scenario_summaries = {}
    for scenario in scenario_ids:
        intervened = _load_final(causal_dir, scenario, qids, "final_canonical_logits")
        intervened_metrics = _metrics(intervened, baseline, winners, runners)
        per_metric = {}
        for metric in natural_metrics:
            effect = intervened_metrics[metric] - natural_metrics[metric]
            mean, low, high = _bootstrap_mean(effect, config.bootstrap_samples, rng)
            row = {
                "scenario": scenario,
                "metric": metric,
                "natural_game_mean": float(natural_metrics[metric].mean()),
                "intervened_mean": float(intervened_metrics[metric].mean()),
                "effect_mean": mean,
                "effect_ci_low": low,
                "effect_ci_high": high,
            }
            rows.append(row)
            per_metric[metric] = row
        changed = np.argmax(intervened, axis=-1) != np.argmax(natural, axis=-1)
        scenario_summaries[scenario] = {
            "fraction_final_choices_changed": float(changed.mean()),
            "n_final_choices_changed": int(changed.sum()),
            "metrics": per_metric,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "causal_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, output_dir / "causal_attention_ablation_effects")

    primary = "ablate_user_incorrect__all5"
    control = "ablate_system_incorrect__all5"
    control_contrasts = {}
    if primary in scenario_ids and control in scenario_ids:
        primary_logits = _load_final(causal_dir, primary, qids, "final_canonical_logits")
        control_logits = _load_final(causal_dir, control, qids, "final_canonical_logits")
        primary_metrics = _metrics(primary_logits, baseline, winners, runners)
        control_metrics = _metrics(control_logits, baseline, winners, runners)
        for metric in natural_metrics:
            contrast = primary_metrics[metric] - control_metrics[metric]
            mean, low, high = _bootstrap_mean(contrast, config.bootstrap_samples, rng)
            control_contrasts[metric] = {"mean": mean, "ci_low": low, "ci_high": high}

    summary = {
        "n_questions": len(qids),
        "natural_reference": natural_source,
        "natural_reference_agreement_with_prior_run": reference_agreement,
        "natural_game": {metric: float(values.mean()) for metric, values in natural_metrics.items()},
        "scenarios": scenario_summaries,
        "primary_user_minus_same_word_system_control": control_contrasts,
        "interpretation_rule": {
            "supports_feedback-edge-driven-compression": [
                "compression effect < 0",
                "winner_advantage effect > 0",
                "A-D entropy effect < 0",
                "switch-probability effect < 0",
                "and effects are larger than the same-word system-token control",
            ]
        },
    }
    (output_dir / "causal_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    primary_metrics = scenario_summaries.get(primary, {}).get("metrics", {})
    def effect(metric: str) -> str:
        row = primary_metrics[metric]
        scale = 100 if metric == "switch" else 1
        suffix = " pp" if metric == "switch" else ""
        return (
            f"{scale * row['effect_mean']:+.4f}{suffix} "
            f"[{scale * row['effect_ci_low']:+.4f}, {scale * row['effect_ci_high']:+.4f}]"
        )

    lines = [
        "# Causal attention-edge ablation",
        "",
        "## Experiment",
        "",
        f"Questions: {len(qids)}. The intervention removes only selected final-query → user-turn `incorrect` attention edges before softmax, with remaining attention renormalized.",
        "",
        f"Natural reference: {natural_source}. Effects are intervention minus natural Game; intervals are paired 95% bootstrap CIs.",
        "",
        "## Primary five-head intervention",
        "",
        "| Outcome | Effect [95% CI] |",
        "|---|---:|",
    ]
    for metric in ("compression", "winner_advantage", "ad_entropy", "switch"):
        lines.append(f"| {metric} | {effect(metric)} |")
    lines.extend([
        "",
        "Relative to the same-word system-token control, user-edge removal changed compression by "
        f"{control_contrasts['compression']['mean']:+.4f} "
        f"[{control_contrasts['compression']['ci_low']:+.4f}, {control_contrasts['compression']['ci_high']:+.4f}], "
        "winner advantage by "
        f"{control_contrasts['winner_advantage']['mean']:+.4f} "
        f"[{control_contrasts['winner_advantage']['ci_low']:+.4f}, {control_contrasts['winner_advantage']['ci_high']:+.4f}], "
        "and switching by "
        f"{100 * control_contrasts['switch']['mean']:+.1f} pp "
        f"[{100 * control_contrasts['switch']['ci_low']:+.1f}, {100 * control_contrasts['switch']['ci_high']:+.1f}].",
        "",
        "## Interpretation",
        "",
        "These edges make a small causal contribution to A–D compression and original-winner suppression, but their joint removal does not reduce switching. They therefore do not explain the behavioral Game effect. The full scenario table is in `causal_effects.csv`.",
    ])
    (output_dir / "CAUSAL_ATTENTION_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze causal feedback-token attention ablations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = AttentionAblationConfig.load(args.config)
    summary = analyze(config, Path(args.output_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
