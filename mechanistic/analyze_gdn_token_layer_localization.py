from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_baseline_mixer_function import _bootstrap_indices, _summary


CONDITIONS = ("Game", "Neutral")


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
        "switch_rate_pp": 100.0 * (np.argmax(values, axis=-1) != winners),
        "winner_advantage": winner - (values.sum(axis=-1) - winner) / 3.0,
        "ad_spread": values.std(axis=-1),
        "ad_entropy_bits": _entropy(values),
        "baseline_alignment": np.sum(values * baseline, axis=-1) / denominator,
    }


def _ci(values: np.ndarray, bootstrap: np.ndarray) -> dict:
    point, low, high = _summary(values, bootstrap)
    return {"estimate": float(point), "ci": [float(low), float(high)]}


def analyze(
    results: Path,
    plan_path: Path,
    baseline_results: Path,
    output_dir: Path,
    draws: int,
    seed: int,
) -> dict:
    with np.load(results, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if not arrays["completed"].all() or not np.isfinite(arrays["intervened_logits"]).all():
        raise ValueError("Localization run is incomplete")
    plan = json.loads(plan_path.read_text())
    scenarios = plan["scenarios"]
    if arrays["scenario_ids"].astype(str).tolist() != [row["id"] for row in scenarios]:
        raise ValueError("Scenario order differs between plan and result")
    qids = arrays["question_ids"].astype(str).tolist()
    baseline_rows = json.loads(baseline_results.read_text())["results"]
    baseline = np.asarray(
        [baseline_rows[qid]["aggregated_ad_logits"] for qid in qids], dtype=np.float64
    )
    winners = baseline.argmax(axis=-1)
    bootstrap = _bootstrap_indices(winners, draws, seed)
    natural = arrays["natural_logits"].astype(np.float64)
    intervened = arrays["intervened_logits"].astype(np.float64)
    natural_metrics = {
        condition: _metrics(natural[index], baseline, winners)
        for index, condition in enumerate(CONDITIONS)
    }
    effects = {}
    rows = []
    for scenario_index, scenario in enumerate(scenarios):
        scenario_id = scenario["id"]
        effects[scenario_id] = {}
        condition_raw = {}
        for condition_index, condition in enumerate(CONDITIONS):
            metrics = _metrics(intervened[condition_index, scenario_index], baseline, winners)
            condition_raw[condition] = {
                metric: values - natural_metrics[condition][metric]
                for metric, values in metrics.items()
            }
            natural_choice = np.argmax(natural[condition_index], axis=-1)
            intervened_choice = np.argmax(intervened[condition_index, scenario_index], axis=-1)
            condition_raw[condition]["choice_change_rate_pp"] = (
                100.0 * (natural_choice != intervened_choice)
            )
            effects[scenario_id][condition] = {
                metric: _ci(values, bootstrap)
                for metric, values in condition_raw[condition].items()
            }
            effects[scenario_id][condition]["choice_transitions"] = {
                "total_changed": int(np.sum(intervened_choice != natural_choice)),
                "new_switches": int(np.sum((natural_choice == winners) & (intervened_choice != winners))),
                "prevented_switches": int(np.sum((natural_choice != winners) & (intervened_choice == winners))),
                "alternative_to_alternative": int(np.sum(
                    (natural_choice != winners) & (intervened_choice != winners)
                    & (natural_choice != intervened_choice)
                )),
            }
        effects[scenario_id]["Game minus Neutral"] = {
            metric: _ci(
                condition_raw["Game"][metric] - condition_raw["Neutral"][metric],
                bootstrap,
            )
            for metric in condition_raw["Game"]
        }
        for condition in (*CONDITIONS, "Game minus Neutral"):
            for metric, record in effects[scenario_id][condition].items():
                if metric == "choice_transitions":
                    continue
                rows.append({
                    "scenario": scenario_id,
                    "kind": scenario["kind"],
                    "token_pair": scenario["token_pair"],
                    "window": scenario.get("window", "all_gla"),
                    "condition": condition,
                    "metric": metric,
                    "estimate": record["estimate"],
                    "ci_low": record["ci"][0],
                    "ci_high": record["ci"][1],
                })

    all_token = [
        row for row in scenarios if row["kind"] in {"all_gla_token", "confirm_all"}
    ]
    windows = [
        row for row in scenarios
        if row["kind"] in {"token_window", "confirm_window"}
    ]
    exact_layers = [row for row in scenarios if row["kind"] == "confirm_layer"]
    ranked_windows = sorted(
        windows,
        key=lambda row: abs(
            effects[row["id"]]["Game minus Neutral"]["switch_rate_pp"]["estimate"]
        ),
        reverse=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "causal_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": plan.get("status", "unknown"),
        "n_questions": len(qids),
        "batch_control_max_abs_logit_drift": float(
            np.nanmax(np.abs(arrays["batch_control_minus_natural"]))
        ),
        "natural": {
            condition: {metric: float(values.mean()) for metric, values in metrics.items()}
            for condition, metrics in natural_metrics.items()
        },
        "effects": effects,
        "ranked_windows_by_absolute_differential_switching": [
            row["id"] for row in ranked_windows
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    def cell(record: dict, metric: str) -> str:
        value = record[metric]
        return f"{value['estimate']:+.2f} [{value['ci'][0]:+.2f}, {value['ci'][1]:+.2f}]"

    is_confirmation = plan.get("status") == "frozen_confirmation"
    lines = [
        (
            "# Held-out confirmation: GLA feedback-token and layer localization"
            if is_confirmation
            else "# Discovery: GLA feedback-token and layer localization"
        ),
        "",
        f"Frozen {'confirmation' if is_confirmation else 'discovery'} questions: "
        f"**{len(qids)}**. Effects are write-ablation minus natural.",
        "Intervals are paired, Baseline-letter-stratified 95% bootstrap CIs.",
        "",
        "## Each feedback token across all 48 GLA layers",
        "",
        "| Token pair | Game switching | Differential switching | Game spread | Differential spread |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario in all_token:
        record = effects[scenario["id"]]
        lines.append(
            f"| `{scenario['token_pair']}` | {cell(record['Game'], 'switch_rate_pp')} | "
            f"{cell(record['Game minus Neutral'], 'switch_rate_pp')} | "
            f"{cell(record['Game'], 'ad_spread')} | "
            f"{cell(record['Game minus Neutral'], 'ad_spread')} |"
        )
    lines += [
        "",
        (
            "## Preselected token-window tests"
            if is_confirmation
            else "## Largest token-window differential switching effects"
        ),
        "",
        "| Token pair | Block window | Game switching | Differential switching | Game spread |",
        "|---|---|---:|---:|---:|",
    ]
    for scenario in ranked_windows[:24]:
        record = effects[scenario["id"]]
        if "human_blocks" in scenario:
            human = scenario["human_blocks"]
        else:
            human = [
                min(scenario["layers_zero_based"]) + 1,
                max(scenario["layers_zero_based"]) + 2,
            ]
        lines.append(
            f"| `{scenario['token_pair']}` | {human[0]}–{human[1]} | "
            f"{cell(record['Game'], 'switch_rate_pp')} | "
            f"{cell(record['Game minus Neutral'], 'switch_rate_pp')} | "
            f"{cell(record['Game'], 'ad_spread')} |"
        )
    if exact_layers:
        ranked_layers = sorted(
            exact_layers,
            key=lambda row: abs(
                effects[row["id"]]["Game minus Neutral"]["switch_rate_pp"]["estimate"]
            ),
            reverse=True,
        )
        lines += [
            "",
            "## Preselected exact-layer tests",
            "",
            "| Token pair | Block | Game switching | Differential switching | Game spread |",
            "|---|---:|---:|---:|---:|",
        ]
        for scenario in ranked_layers:
            record = effects[scenario["id"]]
            block = scenario["layers_zero_based"][0] + 1
            lines.append(
                f"| `{scenario['token_pair']}` | {block} | "
                f"{cell(record['Game'], 'switch_rate_pp')} | "
                f"{cell(record['Game minus Neutral'], 'switch_rate_pp')} | "
                f"{cell(record['Game'], 'ad_spread')} |"
            )
    if not is_confirmation:
        lines += [
            "",
            "This is the selection half only. Exact-layer hypotheses derived from these",
            "results must be tested on the frozen 249-question confirmation half.",
        ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GLA token-layer localization")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(args.results, args.plan, args.baseline_results, args.output_dir, args.draws, args.seed)


if __name__ == "__main__":
    main()
