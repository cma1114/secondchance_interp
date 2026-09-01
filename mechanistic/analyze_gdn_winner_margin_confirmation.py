from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_baseline_mixer_function import _bootstrap_indices, _summary


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _entropy(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=-1, keepdims=True)
    return -np.sum(probability * np.log2(np.maximum(probability, 1e-300)), axis=-1)


def _metrics(values: np.ndarray, baseline: np.ndarray, winners: np.ndarray) -> dict[str, np.ndarray]:
    values = _center(values)
    baseline = _center(baseline)
    row = np.arange(len(values))
    winner = values[row, winners]
    denominator = np.maximum(np.sum(baseline * baseline, axis=-1), 1e-12)
    return {
        "winner_advantage": winner - (values.sum(axis=-1) - winner) / 3.0,
        "switch_rate_pp": 100.0 * (np.argmax(values, axis=-1) != winners),
        "ad_spread": values.std(axis=-1),
        "ad_entropy_bits": _entropy(values),
        "baseline_alignment": np.sum(values * baseline, axis=-1) / denominator,
    }


def _ci(values: np.ndarray, bootstrap: np.ndarray) -> dict:
    point, low, high = _summary(values, bootstrap)
    return {"estimate": float(point), "ci": [float(low), float(high)]}


def _holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    n = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (n - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _load_baseline(path: Path, qids: list[str]) -> np.ndarray:
    raw = json.loads(path.read_text())["results"]
    if isinstance(raw, list):
        raw = {row.get("question_id", row.get("id")): row for row in raw}
    return np.asarray([raw[qid]["aggregated_ad_logits"] for qid in qids], dtype=float)


def analyze(
    results_path: Path,
    plan_path: Path,
    baseline_path: Path,
    discovery_path: Path,
    output_dir: Path,
    draws: int,
    seed: int,
) -> dict:
    from scipy.stats import pearsonr, spearmanr, ttest_1samp

    with np.load(results_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if arrays["completed"].shape[0] != 1 or not arrays["completed"].all():
        raise ValueError("Expected a complete Game-only result")
    if not np.isfinite(arrays["intervened_logits"]).all():
        raise ValueError("Intervention logits contain non-finite values")
    plan = json.loads(plan_path.read_text())
    scenarios = plan["scenarios"]
    scenario_ids = arrays["scenario_ids"].astype(str).tolist()
    if scenario_ids != [row["id"] for row in scenarios]:
        raise ValueError("Scenario order differs between plan and result")
    qids = arrays["question_ids"].astype(str).tolist()
    baseline = _load_baseline(baseline_path, qids)
    winners = baseline.argmax(axis=-1)
    order = np.argsort(-baseline, axis=-1)
    bootstrap = _bootstrap_indices(winners, draws, seed)

    natural = arrays["natural_logits"][0].astype(float)
    intervened = arrays["intervened_logits"][0].astype(float)
    natural_metrics = _metrics(natural, baseline, winners)
    raw_effects: dict[str, dict[str, np.ndarray]] = {}
    effects = {}
    rank_contributions = {}
    centered_natural = _center(natural)
    for index, scenario in enumerate(scenarios):
        scenario_id = scenario["id"]
        metrics = _metrics(intervened[index], baseline, winners)
        raw_effects[scenario_id] = {
            metric: values - natural_metrics[metric] for metric, values in metrics.items()
        }
        effects[scenario_id] = {
            metric: _ci(values, bootstrap)
            for metric, values in raw_effects[scenario_id].items()
        }
        natural_contribution = centered_natural - _center(intervened[index])
        aligned = np.take_along_axis(natural_contribution, order, axis=-1)
        rank_contributions[scenario_id] = {
            f"rank_{rank + 1}": _ci(aligned[:, rank], bootstrap) for rank in range(4)
        }

    discovery = json.loads(discovery_path.read_text())["effects"]
    exact = {"evaluation_period": [], "action_clause": []}
    for source in exact:
        source_rows = [
            row for row in scenarios
            if row["kind"] == "exact_layer" and row["id"].startswith(source + "__")
        ]
        pvalues = []
        for row in source_rows:
            scenario_id = row["id"]
            values = raw_effects[scenario_id]["winner_advantage"]
            pvalues.append(float(ttest_1samp(values, 0.0).pvalue))
        adjusted = _holm(pvalues)
        for row, pvalue, holm in zip(source_rows, pvalues, adjusted, strict=True):
            scenario_id = row["id"]
            confirm = effects[scenario_id]["winner_advantage"]
            discover = discovery[scenario_id]["Game"]["winner_advantage"]
            exact[source].append({
                "block": row["human_block"],
                "scenario": scenario_id,
                "discovery": discover,
                "confirmation": confirm,
                "pvalue": pvalue,
                "holm_pvalue": holm,
                "switch_rate_pp": effects[scenario_id]["switch_rate_pp"],
                "spread": effects[scenario_id]["ad_spread"],
            })

    reliability = {}
    for source, rows in exact.items():
        discovery_values = np.asarray([row["discovery"]["estimate"] for row in rows])
        confirmation_values = np.asarray([row["confirmation"]["estimate"] for row in rows])
        reliability[source] = {
            "pearson_r": float(pearsonr(discovery_values, confirmation_values).statistic),
            "spearman_rho": float(spearmanr(discovery_values, confirmation_values).statistic),
            "same_sign_fraction": float(np.mean(np.sign(discovery_values) == np.sign(confirmation_values))),
        }

    joint = {}
    scenario_lookup = {row["id"]: row for row in scenarios}
    for scenario in scenarios:
        if scenario["kind"] != "joint_selected_layers":
            continue
        scenario_id = scenario["id"]
        source = scenario_id.split("__", 1)[0]
        member_ids = [f"{source}__block_{block:02d}" for block in scenario["selected_human_blocks"]]
        member_sum = np.sum(
            [raw_effects[member]["winner_advantage"] for member in member_ids], axis=0
        )
        synergy = raw_effects[scenario_id]["winner_advantage"] - member_sum
        full_id = f"{source}__all_gla"
        full_boot = np.mean(
            raw_effects[full_id]["winner_advantage"][bootstrap], axis=1
        )
        joint_boot = np.mean(
            raw_effects[scenario_id]["winner_advantage"][bootstrap], axis=1
        )
        ratio = joint_boot / np.where(np.abs(full_boot) > 1e-9, full_boot, np.nan)
        joint[scenario_id] = {
            "selected_blocks": scenario["selected_human_blocks"],
            "winner_advantage": effects[scenario_id]["winner_advantage"],
            "switch_rate_pp": effects[scenario_id]["switch_rate_pp"],
            "spread": effects[scenario_id]["ad_spread"],
            "synergy_beyond_sum_of_individuals": _ci(synergy, bootstrap),
            "fraction_of_full_all_gla_effect": {
                "estimate": float(
                    effects[scenario_id]["winner_advantage"]["estimate"]
                    / effects[full_id]["winner_advantage"]["estimate"]
                ),
                "ci": [float(np.nanpercentile(ratio, 2.5)), float(np.nanpercentile(ratio, 97.5))],
            },
        }

    rescues = {}
    for scenario in scenarios:
        if scenario["kind"] != "leave_selected_layers_natural":
            continue
        scenario_id = scenario["id"]
        source = scenario_id.split("__", 1)[0]
        full_id = f"{source}__all_gla"
        rescue_values = (
            raw_effects[full_id]["winner_advantage"]
            - raw_effects[scenario_id]["winner_advantage"]
        )
        rescues[scenario_id] = {
            "natural_blocks": scenario["natural_human_blocks"],
            "winner_advantage_rescue": _ci(rescue_values, bootstrap),
            "all_except_effect": effects[scenario_id]["winner_advantage"],
        }

    summary = {
        "status": plan["status"],
        "n_questions": len(qids),
        "primary_metric": plan["primary_metric"],
        "natural": {metric: float(values.mean()) for metric, values in natural_metrics.items()},
        "batch_control_max_abs_logit_drift": float(
            np.max(np.abs(arrays["batch_control_minus_natural"]))
        ),
        "full_effects": {
            source: effects[f"{source}__all_gla"] for source in exact
        },
        "exact_layers": exact,
        "cross_half_reliability": reliability,
        "joint_sets": joint,
        "rescues": rescues,
        "rank_contributions": {
            scenario_id: values for scenario_id, values in rank_contributions.items()
            if scenario_lookup[scenario_id]["kind"] in {"all_gla", "joint_selected_layers"}
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    with (output_dir / "exact_layer_effects.csv").open("w", newline="") as handle:
        fields = [
            "source", "block", "discovery_estimate", "confirmation_estimate",
            "ci_low", "ci_high", "pvalue", "holm_pvalue", "switch_rate_pp",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source, rows in exact.items():
            for row in rows:
                writer.writerow({
                    "source": source,
                    "block": row["block"],
                    "discovery_estimate": row["discovery"]["estimate"],
                    "confirmation_estimate": row["confirmation"]["estimate"],
                    "ci_low": row["confirmation"]["ci"][0],
                    "ci_high": row["confirmation"]["ci"][1],
                    "pvalue": row["pvalue"],
                    "holm_pvalue": row["holm_pvalue"],
                    "switch_rate_pp": row["switch_rate_pp"]["estimate"],
                })

    _plot(summary, output_dir / "winner_margin_by_gla_block.png")
    _write_report(summary, output_dir / "REPORT.md")
    return summary


def _plot(summary: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"evaluation_period": "#7b3294", "action_clause": "#008837"}
    labels = {"evaluation_period": "Evaluation period", "action_clause": "Action clause"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for axis, source in zip(axes, ("evaluation_period", "action_clause"), strict=True):
        rows = summary["exact_layers"][source]
        blocks = np.asarray([row["block"] for row in rows])
        disc = np.asarray([row["discovery"]["estimate"] for row in rows])
        point = np.asarray([row["confirmation"]["estimate"] for row in rows])
        low = np.asarray([row["confirmation"]["ci"][0] for row in rows])
        high = np.asarray([row["confirmation"]["ci"][1] for row in rows])
        axis.axhline(0, color="#777777", linewidth=1)
        axis.scatter(blocks, disc, s=28, color="#bdbdbd", label="Discovery", zorder=2)
        axis.errorbar(
            blocks, point, yerr=np.vstack([point - low, high - point]), fmt="o",
            markersize=4.5, linewidth=1, capsize=2, color=colors[source],
            label="Held-out confirmation", zorder=3,
        )
        axis.set_title(labels[source], loc="left", fontweight="bold")
        axis.set_ylabel("Winner-advantage change\n(ablation minus natural, logits)")
        axis.legend(frameon=False, ncol=2, loc="upper left")
    axes[-1].set_xlabel("Model block (GLA-containing blocks only)")
    fig.suptitle("Individual feedback-source GLA ablations", fontsize=16, fontweight="bold")
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fmt(record: dict, scale: float = 1.0) -> str:
    return (
        f"{scale * record['estimate']:+.3f} "
        f"[{scale * record['ci'][0]:+.3f}, {scale * record['ci'][1]:+.3f}]"
    )


def _write_report(summary: dict, output: Path) -> None:
    lines = [
        "# Held-out GLA winner-margin confirmation",
        "",
        f"Questions: **{summary['n_questions']}** held-out SimpleMC trials. The primary",
        "outcome is the within-Game change in frozen-Baseline winner advantage",
        "caused by source-write ablation. Positive values mean that removing the",
        "write restores the previous winner's advantage.",
        "",
        "## Full-path effects",
        "",
        "| Source | Winner advantage | Game switching | A–D spread |",
        "|---|---:|---:|---:|",
    ]
    for source, label in (("evaluation_period", "Evaluation period"), ("action_clause", "Action clause")):
        record = summary["full_effects"][source]
        lines.append(
            f"| {label} | {_fmt(record['winner_advantage'])} | "
            f"{_fmt(record['switch_rate_pp'])} pp | {_fmt(record['ad_spread'])} |"
        )
    lines += [
        "",
        "## Cross-half block-level reliability",
        "",
        "| Source | Pearson r | Spearman rho | Same-sign blocks |",
        "|---|---:|---:|---:|",
    ]
    for source, label in (("evaluation_period", "Evaluation period"), ("action_clause", "Action clause")):
        record = summary["cross_half_reliability"][source]
        lines.append(
            f"| {label} | {record['pearson_r']:.3f} | {record['spearman_rho']:.3f} | "
            f"{100 * record['same_sign_fraction']:.1f}% |"
        )
    lines += [
        "",
        "## Individually confirmed blocks",
        "",
        "The table includes blocks whose held-out 95% CI excludes zero. Holm-adjusted",
        "p-values are also shown for the 48-block family within each source.",
        "",
        "| Source | Block | Discovery | Confirmation | Holm p | Switching |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source, label in (("evaluation_period", "Evaluation"), ("action_clause", "Action")):
        rows = [
            row for row in summary["exact_layers"][source]
            if row["confirmation"]["ci"][0] > 0 or row["confirmation"]["ci"][1] < 0
        ]
        rows.sort(key=lambda row: abs(row["confirmation"]["estimate"]), reverse=True)
        for row in rows:
            lines.append(
                f"| {label} | {row['block']} | {row['discovery']['estimate']:+.3f} | "
                f"{_fmt(row['confirmation'])} | {row['holm_pvalue']:.3g} | "
                f"{_fmt(row['switch_rate_pp'])} pp |"
            )
    lines += [
        "",
        "## Frozen joint sets",
        "",
        "| Scenario | Blocks | Winner advantage | Fraction of full effect | Non-additive synergy | Switching |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scenario_id, record in summary["joint_sets"].items():
        fraction = record["fraction_of_full_all_gla_effect"]
        lines.append(
            f"| `{scenario_id}` | {', '.join(map(str, record['selected_blocks']))} | "
            f"{_fmt(record['winner_advantage'])} | {fraction['estimate']:.2f} "
            f"[{fraction['ci'][0]:.2f}, {fraction['ci'][1]:.2f}] | "
            f"{_fmt(record['synergy_beyond_sum_of_individuals'])} | "
            f"{_fmt(record['switch_rate_pp'])} pp |"
        )
    lines += [
        "",
        "## Rescue tests",
        "",
        "A positive rescue means that leaving the discovery-selected blocks natural",
        "reduces the winner restoration caused by ablating every GLA source write.",
        "",
        "| Scenario | Blocks left natural | Rescue | Effect from ablating all other blocks |",
        "|---|---|---:|---:|",
    ]
    for scenario_id, record in summary["rescues"].items():
        lines.append(
            f"| `{scenario_id}` | {', '.join(map(str, record['natural_blocks']))} | "
            f"{_fmt(record['winner_advantage_rescue'])} | {_fmt(record['all_except_effect'])} |"
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--discovery-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(
        args.results,
        args.plan,
        args.baseline_results,
        args.discovery_summary,
        args.output_dir,
        args.draws,
        args.seed,
    )


if __name__ == "__main__":
    main()
