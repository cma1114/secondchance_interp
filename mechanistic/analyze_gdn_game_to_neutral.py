from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analyze_baseline_mixer_function import _bootstrap_indices, _summary


ALL_LABELS = (
    "Natural Neutral",
    "Natural Game",
    "Game evaluation write\ninto Neutral",
    "Game action write\ninto Neutral",
    "Both Game writes\ninto Neutral",
)
SELECTED_LABELS = (
    "Natural Neutral",
    "Natural Game",
    "Selected 8 evaluation GLAs\ninto Neutral",
    "Selected 13 action GLAs\ninto Neutral",
    "Both selected sets\ninto Neutral",
)
COLORS = ("#e66b19", "#1689d8", "#8e6bbf", "#4cae68", "#cc4c8a")


def _entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=-1, keepdims=True)
    return -np.sum(probability * np.log2(np.maximum(probability, 1e-300)), axis=-1)


def _metrics(logits: np.ndarray, winners: np.ndarray) -> dict[str, np.ndarray]:
    row = np.arange(len(logits))
    winner = logits[row, winners]
    return {
        "switch_rate": (np.argmax(logits, axis=-1) != winners).astype(float),
        "winner_advantage": winner - (logits.sum(axis=-1) - winner) / 3.0,
        "ad_spread": logits.std(axis=-1),
        "ad_entropy_bits": _entropy(logits),
    }


def _record(values: np.ndarray, bootstrap: np.ndarray, scale: float = 1.0) -> dict:
    point, low, high = _summary(values, bootstrap)
    return {
        "estimate": float(point * scale),
        "ci": [float(low * scale), float(high * scale)],
    }


def _fraction_record(
    numerator: np.ndarray,
    denominator: np.ndarray,
    bootstrap: np.ndarray,
) -> dict:
    point = numerator.mean() / denominator.mean()
    numerator_draws = numerator[bootstrap].mean(axis=1)
    denominator_draws = denominator[bootstrap].mean(axis=1)
    valid = np.abs(denominator_draws) > 1e-9
    draws = numerator_draws[valid] / denominator_draws[valid]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(point),
        "ci": [float(low), float(high)],
        "valid_bootstrap_draws": int(valid.sum()),
    }


def _format(record: dict, digits: int = 2, suffix: str = "") -> str:
    return (
        f"{record['estimate']:.{digits}f}{suffix} "
        f"[{record['ci'][0]:.{digits}f}, {record['ci'][1]:.{digits}f}]"
    )


def _plot(
    output: Path,
    metric_values: list[dict[str, np.ndarray]],
    bootstrap: np.ndarray,
    labels: tuple[str, ...],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    specs = (
        ("switch_rate", "Switch rate", 100.0, "%"),
        ("winner_advantage", "Baseline-winner advantage", 1.0, " logits"),
        ("ad_spread", "A–D spread", 1.0, " logits"),
    )
    x = np.arange(len(labels))
    for axis, (metric, title, scale, unit) in zip(axes, specs):
        points, lows, highs = [], [], []
        for values in metric_values:
            record = _record(values[metric], bootstrap, scale)
            points.append(record["estimate"])
            lows.append(record["ci"][0])
            highs.append(record["ci"][1])
        points = np.asarray(points)
        lows = np.asarray(lows)
        highs = np.asarray(highs)
        for index in range(len(labels)):
            axis.errorbar(
                x[index], points[index],
                yerr=[[points[index] - lows[index]], [highs[index] - points[index]]],
                fmt="o", color=COLORS[index], capsize=4, markersize=7,
            )
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.set_ylabel(unit.strip())
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Are Game GLA feedback writes sufficient to make Neutral Game-like?", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def analyze(
    results: Path,
    natural_reference: Path,
    baseline_results: Path,
    output_dir: Path,
    draws: int,
    seed: int,
) -> dict:
    with np.load(results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not arrays["completed"].all() or not np.isfinite(arrays["patched_neutral_logits"]).all():
        raise ValueError("Game-to-Neutral run is incomplete")
    scenario_ids = arrays["scenario_ids"].astype(str).tolist()
    all_expected = [
        "evaluation_period_all_gla",
        "action_clause_all_gla",
        "evaluation_plus_action_all_gla",
    ]
    selected_expected = [
        "evaluation_period_selected_8_gla",
        "action_clause_selected_13_gla",
        "evaluation_8_plus_action_13_gla",
    ]
    if scenario_ids == all_expected:
        labels = ALL_LABELS
        intervention_description = (
            "Game's GLA key, value, decay gate, and write strength are copied at "
            "the named feedback positions into Neutral in all 48 GLA blocks."
        )
    elif scenario_ids == selected_expected:
        labels = SELECTED_LABELS
        intervention_description = (
            "Game's GLA key, value, decay gate, and write strength are copied only "
            "through the preselected 8 evaluation-period routes and/or 13 "
            "action-clause routes (17 distinct GLA blocks jointly)."
        )
    else:
        raise ValueError(f"Unexpected scenarios: {scenario_ids}")

    qids = arrays["question_ids"].astype(str).tolist()
    baseline_rows = json.loads(baseline_results.read_text())["results"]
    winners = np.asarray([
        "ABCD".index(baseline_rows[qid].get("answer", baseline_rows[qid].get("subject_answer")))
        for qid in qids
    ])
    bootstrap = _bootstrap_indices(winners, draws, seed)
    # The torch fallback GLA kernel has meaningful batch-composition drift.
    # Preserve the paired within-batch causal contrasts from this experiment,
    # but anchor them to the already measured single-row natural logits from
    # the exact same prompt, questions, and frozen split.
    with np.load(natural_reference, allow_pickle=False) as loaded:
        reference = {key: loaded[key] for key in loaded.files}
    reference_qids = reference["question_ids"].astype(str).tolist()
    if reference_qids != qids:
        raise ValueError("Natural-reference question IDs or order differ")
    if reference["natural_logits"].shape != (2, len(qids), 4):
        raise ValueError("Natural reference must contain Game and Neutral single-row logits")
    game = reference["natural_logits"][0].astype(np.float64)
    neutral = reference["natural_logits"][1].astype(np.float64)
    saved_neutral = arrays["natural_neutral_logits"].astype(np.float64)
    within_batch_effect = arrays["patched_neutral_logits"].astype(np.float64) - saved_neutral[None]
    patched = neutral[None] + within_batch_effect
    logits_sets = [neutral, game, *[patched[index] for index in range(3)]]
    metric_values = [_metrics(values, winners) for values in logits_sets]

    absolute = {
        label: {
            metric: _record(values[metric], bootstrap, 100.0 if metric == "switch_rate" else 1.0)
            for metric in values
        }
        for label, values in zip(labels, metric_values)
    }
    effects = {}
    transitions = {}
    neutral_choice = np.argmax(neutral, axis=-1)
    game_choice = np.argmax(game, axis=-1)
    for scenario_index, scenario_id in enumerate(scenario_ids):
        values = metric_values[scenario_index + 2]
        raw_effects = {
            metric: values[metric] - metric_values[0][metric]
            for metric in values
        }
        effects[scenario_id] = {
            metric: _record(delta, bootstrap, 100.0 if metric == "switch_rate" else 1.0)
            for metric, delta in raw_effects.items()
        }
        # Positive numerator means movement from Neutral toward Game for each
        # metric, including winner suppression and reduced spread.
        effects[scenario_id]["fraction_of_game_neutral_switch_gap_induced"] = _fraction_record(
            raw_effects["switch_rate"],
            metric_values[1]["switch_rate"] - metric_values[0]["switch_rate"],
            bootstrap,
        )
        effects[scenario_id]["fraction_of_game_neutral_winner_gap_induced"] = _fraction_record(
            -raw_effects["winner_advantage"],
            metric_values[0]["winner_advantage"] - metric_values[1]["winner_advantage"],
            bootstrap,
        )
        patched_choice = np.argmax(patched[scenario_index], axis=-1)
        disagreements = neutral_choice != game_choice
        changed = patched_choice != neutral_choice
        transitions[scenario_id] = {
            "neutral_choices_changed": int(changed.sum()),
            "changed_to_natural_game_choice": int(np.sum(changed & (patched_choice == game_choice))),
            "changed_to_other_choice": int(np.sum(changed & (patched_choice != game_choice))),
            "natural_game_neutral_disagreements": int(disagreements.sum()),
            "disagreements_resolved_to_game": int(np.sum(disagreements & (patched_choice == game_choice))),
        }

    baseline_logits = np.asarray(
        [baseline_rows[qid]["aggregated_ad_logits"] for qid in qids], dtype=np.float64
    )
    # Rank by baseline evidence, but force the generated Baseline answer to be
    # rank 1 if provider-logit aggregation and generated choice ever disagree.
    order = []
    for row, winner in zip(baseline_logits, winners):
        rest = [index for index in np.argsort(-row) if index != winner]
        order.append([winner, *rest])
    order = np.asarray(order)
    rank_effects = {}
    for scenario_index, scenario_id in enumerate(scenario_ids):
        delta = patched[scenario_index] - neutral
        delta -= delta.mean(axis=-1, keepdims=True)
        aligned = np.take_along_axis(delta, order, axis=-1)
        rank_effects[scenario_id] = {
            f"baseline_rank_{rank + 1}": _record(aligned[:, rank], bootstrap)
            for rank in range(4)
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    _plot(
        output_dir / "game_to_neutral_sufficiency.png",
        metric_values,
        bootstrap,
        labels,
    )
    rows = []
    for scenario_id, records in effects.items():
        for metric, record in records.items():
            rows.append({
                "scenario": scenario_id,
                "metric": metric,
                "estimate": record["estimate"],
                "ci_low": record["ci"][0],
                "ci_high": record["ci"][1],
            })
    with (output_dir / "causal_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_questions": len(qids),
        "definitions": {
            "switch_rate": "Fraction whose A-D argmax differs from that question's generated Baseline answer.",
            "effect": "Patched Neutral minus natural Neutral for the named metric.",
            "fraction_of_game_neutral_switch_gap_induced": (
                "Increase in Neutral switching caused by the transplant divided by natural Game switching minus natural Neutral switching."
            ),
            "fraction_of_game_neutral_winner_gap_induced": (
                "Decrease in Neutral Baseline-winner advantage caused by the transplant divided by natural Neutral advantage minus natural Game advantage."
            ),
        },
        "absolute": absolute,
        "effects": effects,
        "choice_transitions": transitions,
        "centered_rank_logit_effects": rank_effects,
        "batch_control_max_abs_logit_drift": {
            "neutral": float(np.max(np.abs(arrays["neutral_batch_control_minus_natural"]))),
            "game": float(np.max(np.abs(arrays["game_batch_control_minus_natural"]))),
        },
        "natural_reference": str(natural_reference),
        "intervention_description": intervention_description,
        "reanchoring": (
            "single-row natural Neutral plus the same-batch patched-minus-Neutral-control "
            "contrast from this experiment"
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Game GLA feedback writes into Neutral: sufficiency test",
        "",
        f"Frozen held-out SimpleMC questions: **{len(qids)}**.",
        "Intervals are paired, Baseline-letter-stratified bootstrap 95% CIs.",
        "",
        "**Switch rate** is the percentage of questions whose A–D argmax differs from",
        "that question's generated Baseline answer. An **effect** is patched Neutral",
        "minus natural Neutral; it does not subtract a separate Neutral intervention.",
        "",
        intervention_description,
        "Neutral's query/read and every nonselected token and block remain Neutral.",
        "",
        "## Absolute behavior",
        "",
        "| Condition | Switch rate | Baseline-winner advantage | A–D spread |",
        "|---|---:|---:|---:|",
    ]
    for label in labels:
        record = absolute[label]
        lines.append(
            f"| {label.replace(chr(10), ' ')} | {_format(record['switch_rate'], suffix='%')} | "
            f"{_format(record['winner_advantage'])} | {_format(record['ad_spread'])} |"
        )
    lines += [
        "",
        "## Causal movement of Neutral toward Game",
        "",
        "| Transplanted Game write | Neutral switching effect | Game–Neutral switch gap induced | Winner suppression | Winner gap induced |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario_id in scenario_ids:
        record = effects[scenario_id]
        lines.append(
            f"| `{scenario_id}` | {_format(record['switch_rate'], suffix=' pp')} | "
            f"{_format(record['fraction_of_game_neutral_switch_gap_induced'], suffix='×')} | "
            f"{_format({'estimate': -record['winner_advantage']['estimate'], 'ci': [-record['winner_advantage']['ci'][1], -record['winner_advantage']['ci'][0]]})} | "
            f"{_format(record['fraction_of_game_neutral_winner_gap_induced'], suffix='×')} |"
        )
    lines += [
        "",
        "## Choice-level audit",
        "",
        "| Scenario | Neutral choices changed | Changed to natural Game choice | Changed elsewhere | Natural Game/Neutral disagreements resolved to Game |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario_id in scenario_ids:
        record = transitions[scenario_id]
        lines.append(
            f"| `{scenario_id}` | {record['neutral_choices_changed']} | "
            f"{record['changed_to_natural_game_choice']} | {record['changed_to_other_choice']} | "
            f"{record['disagreements_resolved_to_game']}/{record['natural_game_neutral_disagreements']} |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Game-to-Neutral GLA sufficiency")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--natural-reference", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    analyze(
        args.results,
        args.natural_reference,
        args.baseline_results,
        args.output_dir,
        args.draws,
        args.seed,
    )


if __name__ == "__main__":
    main()
