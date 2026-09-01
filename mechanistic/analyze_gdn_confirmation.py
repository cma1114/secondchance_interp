from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analyze_gdn import _bootstrap, _load_final, _metrics
from .gdn_config import GDNExperimentConfig


def analyze(config: GDNExperimentConfig, plan_path: Path, output: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    qids = plan["question_ids"]
    root = Path(config.output_dir)
    baseline = _load_final(Path(config.mechanistic_dir), "baseline", qids, "canonical_logits")
    natural = _load_final(Path(config.natural_attention_dir), "natural_game_rerun", qids, "final_canonical_logits")
    winners = np.argmax(baseline, axis=-1)
    natural_metrics = _metrics(natural, baseline, winners)
    rng = np.random.default_rng(config.seed + 1)
    rows, values_by_scenario = [], {}
    for scenario in plan["scenarios"]:
        scenario_id = scenario["id"]
        logits = _load_final(root, scenario_id, qids, "final_canonical_logits")
        metrics = _metrics(logits, baseline, winners)
        values_by_scenario[scenario_id] = metrics
        for metric, values in metrics.items():
            effect = values - natural_metrics[metric]
            mean, low, high = _bootstrap(effect, config.bootstrap_samples, rng)
            rows.append({
                "scenario": scenario_id, "metric": metric,
                "natural_mean": float(natural_metrics[metric].mean()),
                "intervened_mean": float(values.mean()),
                "effect": mean, "ci_low": low, "ci_high": high,
            })
    contrasts = {}
    pairs = {
        "top8_heads_incorrect_minus_structural0": (
            "gdn_confirm__top8_heads__incorrect", "gdn_confirm__top8_heads__structural0"
        ),
        "top3_layers_incorrect_minus_structural0": (
            "gdn_confirm__top3_layers__incorrect", "gdn_confirm__top3_layers__structural0"
        ),
    }
    for name, (target, control) in pairs.items():
        contrasts[name] = {}
        for metric in natural_metrics:
            difference = values_by_scenario[target][metric] - values_by_scenario[control][metric]
            mean, low, high = _bootstrap(difference, config.bootstrap_samples, rng)
            contrasts[name][metric] = {"mean": mean, "ci_low": low, "ci_high": high}

    output.mkdir(parents=True, exist_ok=True)
    with (output / "gdn_confirmation_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"n_confirmation": len(qids), "contrasts": contrasts, "scenario_effects": rows}
    (output / "gdn_confirmation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    scenarios = [scenario["id"] for scenario in plan["scenarios"]]
    metrics_to_plot = [("compression", "Compression"), ("winner_advantage", "Original-winner advantage"), ("ad_entropy", "A–D entropy"), ("switch", "Switch probability")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics_to_plot):
        selected = [next(row for row in rows if row["scenario"] == scenario and row["metric"] == metric) for scenario in scenarios]
        means = np.asarray([row["effect"] for row in selected]); low = np.asarray([row["ci_low"] for row in selected]); high = np.asarray([row["ci_high"] for row in selected])
        x = np.arange(len(selected)); axis.bar(x, means, color="#3b6fb6")
        axis.errorbar(x, means, yerr=np.vstack([means-low, high-means]), fmt="none", color="black", capsize=3)
        axis.axhline(0, color="black", linewidth=.8); axis.set_title(title); axis.set_ylabel("Ablated − natural Game")
        axis.set_xticks(x, labels=[scenario.replace("gdn_confirm__", "").replace("__", "\n") for scenario in scenarios], rotation=35, ha="right")
    fig.savefig(output / "gdn_confirmation_effects.png", dpi=220); fig.savefig(output / "gdn_confirmation_effects.svg"); plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze held-out GDN localization")
    parser.add_argument("--config", required=True); parser.add_argument("--plan", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); config = GDNExperimentConfig.load(args.config)
    print(json.dumps(analyze(config, Path(args.plan), Path(args.output_dir)), indent=2, sort_keys=True))


if __name__ == "__main__": main()

