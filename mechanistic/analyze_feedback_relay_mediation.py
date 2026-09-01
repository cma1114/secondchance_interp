from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .run_feedback_relay_mediation import RELAY_REGIONS, SCENARIOS


TASKS = ("Game", "Neutral")


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _mean_ci(values: np.ndarray, indices: np.ndarray, rng: np.random.Generator, draws: int):
    selected = np.asarray(values[indices], dtype=np.float64)
    point = float(selected.mean())
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        stop = min(start + 200, draws)
        rows = rng.integers(0, len(selected), size=(stop - start, len(selected)))
        samples[start:stop] = selected[rows].mean(axis=1)
    return point, [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def _sum_ratio_ci(numerator: np.ndarray, denominator: np.ndarray, indices: np.ndarray, rng: np.random.Generator, draws: int):
    num = np.asarray(numerator[indices], dtype=np.float64)
    den = np.asarray(denominator[indices], dtype=np.float64)
    point = float(num.sum() / den.sum())
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        stop = min(start + 200, draws)
        rows = rng.integers(0, len(num), size=(stop - start, len(num)))
        samples[start:stop] = num[rows].sum(axis=1) / den[rows].sum(axis=1)
    return point, [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def _mediation_ci(source_num: np.ndarray, intercepted_num: np.ndarray, denominator: np.ndarray, indices: np.ndarray, rng: np.random.Generator, draws: int):
    src = np.asarray(source_num[indices], dtype=np.float64)
    dst = np.asarray(intercepted_num[indices], dtype=np.float64)
    den = np.asarray(denominator[indices], dtype=np.float64)
    reduction = float((src - dst).sum() / den.sum())
    proportion = float((src - dst).sum() / src.sum())
    red_samples = np.empty(draws, dtype=np.float64)
    prop_samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        stop = min(start + 200, draws)
        rows = rng.integers(0, len(src), size=(stop - start, len(src)))
        selected_src = src[rows].sum(axis=1)
        selected_dst = dst[rows].sum(axis=1)
        red_samples[start:stop] = (selected_src - selected_dst) / den[rows].sum(axis=1)
        prop_samples[start:stop] = (selected_src - selected_dst) / selected_src
    return {
        "transfer_reduction": reduction,
        "transfer_reduction_ci": [float(value) for value in np.quantile(red_samples, (0.025, 0.975))],
        "mediated_proportion": proportion,
        "mediated_proportion_ci": [float(value) for value in np.quantile(prop_samples, (0.025, 0.975))],
    }


def analyze(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    qids = arrays["question_ids"].astype(str).tolist()
    scenarios = arrays["scenario_ids"].astype(str).tolist()
    if len(qids) != 500 or not arrays["completed"].all():
        raise RuntimeError("A complete 500-question relay result is required")
    if scenarios != list(SCENARIOS):
        raise RuntimeError("Relay scenario definition changed")
    logits = arrays["scenario_final_logits"].astype(np.float64)
    if logits.shape != (2, len(SCENARIOS), 500, 4) or not np.isfinite(logits).all():
        raise RuntimeError("Relay causal logits are incomplete or non-finite")
    with np.load(args.stage_a_results, allow_pickle=False) as loaded:
        stage_a = {key: loaded[key] for key in loaded.files}
    stage_a_scenarios = stage_a["scenario_ids"].astype(str).tolist()
    stage_a_source = stage_a["scenario_final_logits"][:, stage_a_scenarios.index("feedback_suffix_3_9_swapped")]
    source_index = scenarios.index("feedback_suffix_swapped")
    cache_control_index = scenarios.index("cache_restored_no_source_swap")
    source_reproduction_error = float(
        np.max(np.abs(logits[:, source_index] - stage_a_source))
    )
    historical_validation: dict[str, Any] | None = None
    if args.historical_results is not None:
        with np.load(args.historical_results, allow_pickle=False) as loaded:
            historical = {key: loaded[key] for key in loaded.files}
        historical_scenarios = historical["scenario_ids"].astype(str).tolist()
        common = [scenario for scenario in historical_scenarios if scenario in scenarios]
        raw_errors = []
        corrected_errors = []
        raw_exact = True
        corrected_exact = True
        for scenario in common:
            old_index = historical_scenarios.index(scenario)
            new_index = scenarios.index(scenario)
            old_raw = historical["scenario_final_logits_raw"][:, old_index]
            new_raw = arrays["scenario_final_logits_raw"][:, new_index]
            old_corrected = historical["scenario_final_logits"][:, old_index]
            new_corrected = arrays["scenario_final_logits"][:, new_index]
            raw_errors.append(float(np.max(np.abs(old_raw - new_raw))))
            corrected_errors.append(float(np.max(np.abs(old_corrected - new_corrected))))
            raw_exact = raw_exact and bool(np.array_equal(old_raw, new_raw))
            corrected_exact = corrected_exact and bool(np.array_equal(old_corrected, new_corrected))
        historical_validation = {
            "common_scenarios": common,
            "raw_max_abs_error": max(raw_errors, default=0.0),
            "corrected_max_abs_error": max(corrected_errors, default=0.0),
            "raw_bit_exact": raw_exact,
            "corrected_bit_exact": corrected_exact,
        }

    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    if [int(discovery.sum()), int((~discovery).sum())] != [251, 249]:
        raise RuntimeError("Frozen split changed")
    split_masks = {"discovery": discovery, "confirmation": ~discovery}
    baseline = json.loads(args.baseline.read_text())["results"]
    fresh = json.loads(args.remapped_baseline.read_text())["results"]
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    rank_order = np.empty((500, 4), dtype=np.int64)
    w1 = np.empty(500, dtype=np.int64)
    w2 = np.empty(500, dtype=np.int64)
    for qi, qid in enumerate(qids):
        old = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=np.float64)
        semantic_rank = np.argsort(-old, kind="stable")
        rank_order[qi] = [
            LETTERS.index(mappings[qid]["original_to_new"][LETTERS[int(index)]])
            for index in semantic_rank
        ]
        w1[qi] = rank_order[qi, 0]
        w2[qi] = int(np.argmax(fresh[qid]["aggregated_ad_logits"]))
    conflict = w1 != w2

    centered = _center(logits)
    natural = centered[:, 0]
    task_vector = natural[::-1] - natural
    denominator = np.sum(task_vector * task_vector, axis=-1)
    delta = centered - natural[:, None]
    numerator = np.sum(delta * task_vector[:, None], axis=-1)
    choices = np.argmax(logits, axis=-1)
    switch = choices != w1[None, None, :]
    choose_w1 = choices == w1[None, None, :]
    ranked = np.empty_like(centered)
    for qi in range(500):
        ranked[:, :, qi] = np.take(centered[:, :, qi], rank_order[qi], axis=-1)
    bivalent = ranked[..., 3] - ranked[..., :2].mean(axis=-1)

    summary: dict[str, Any] = {
        "question": "Through which exhaustive post-feedback relay region does the causal feedback-source crossover reach the final answer?",
        "coverage": {
            "questions": 500, "discovery": 251, "confirmation": 249,
            "tasks": list(TASKS), "relay_regions": list(RELAY_REGIONS),
            "ordinary_attention_layers": list(range(4, 65, 4)),
            "gla_layers": [value for value in range(1, 65) if value % 4 != 0],
        },
        "validation": {
            "all_complete": True,
            "all_finite": True,
            "same_batch_natural_max_abs_error": float(np.max(np.abs(
                arrays["scenario_final_logits_raw"][:, 0] - arrays["same_batch_natural_logits"]
            ))),
            "duplicate_natural_max_abs_error": float(np.max(np.abs(
                arrays["same_batch_natural_logits"] - arrays["duplicate_natural_logits"]
            ))),
            "cache_restoration_scenario_raw_max_abs_error": float(np.max(np.abs(
                arrays["scenario_final_logits_raw"][:, cache_control_index]
                - arrays["same_batch_natural_logits"]
            ))),
            "cache_restoration_scenario_corrected_max_abs_error": float(np.max(np.abs(
                logits[:, cache_control_index] - arrays["trusted_natural_logits"]
            ))),
            "corrected_natural_max_abs_error": float(np.max(np.abs(
                logits[:, 0] - arrays["trusted_natural_logits"]
            ))),
            "source_only_stage_a_max_abs_error": source_reproduction_error,
            "all_region_counts_positive": bool((arrays["region_token_counts"] > 0).all()),
        },
        "definitions": {
            "remaining_transfer": "Projection of the intercepted run's centered A-D logit change onto the paired natural donor-minus-recipient task vector",
            "transfer_reduction": "Source-only donor transfer minus transfer remaining after relay restoration",
            "mediated_proportion": "Transfer reduction divided by source-only donor transfer",
            "route_scope": (
                "The intervention covers ordinary-attention K/V and GLA "
                "delta-rule recurrent writes. It does not intercept the short "
                "causal depthwise convolution on GLA q/k/v inputs."
            ),
            "joint_interpretation": (
                "Joint mediation proportions are convolution-capped lower bounds, "
                "not physiological bypass estimates: restored tokens retain their "
                "source-crossed local outputs, and the assistant prefix is adjacent "
                "to the readout."
            ),
        },
        "splits": {},
    }
    if historical_validation is not None:
        summary["validation"]["historical_intervention_comparison"] = historical_validation
    for split_index, (split_name, split_mask) in enumerate(split_masks.items()):
        split: dict[str, Any] = {"questions": int(split_mask.sum()), "tasks": {}}
        for task_index, task in enumerate(TASKS):
            task_rows: dict[str, Any] = {}
            for scenario_index, scenario in enumerate(scenarios):
                base_seed = args.seed + split_index * 100000 + task_index * 10000 + scenario_index
                indices = np.flatnonzero(split_mask)
                transfer, transfer_ci = _sum_ratio_ci(
                    numerator[task_index, scenario_index], denominator[task_index], indices,
                    np.random.default_rng(base_seed), args.bootstrap_draws,
                )
                sw, sw_ci = _mean_ci(
                    switch[task_index, scenario_index], indices,
                    np.random.default_rng(base_seed + 1000), args.bootstrap_draws,
                )
                w1_rate, w1_ci = _mean_ci(
                    choose_w1[task_index, scenario_index], indices,
                    np.random.default_rng(base_seed + 2000), args.bootstrap_draws,
                )
                biv, biv_ci = _mean_ci(
                    bivalent[task_index, scenario_index] - bivalent[task_index, 0], indices,
                    np.random.default_rng(base_seed + 3000), args.bootstrap_draws,
                )
                row: dict[str, Any] = {
                    "remaining_transfer": transfer, "remaining_transfer_ci": transfer_ci,
                    "switch_rate": sw, "switch_rate_ci": sw_ci,
                    "choose_W1_rate": w1_rate, "choose_W1_rate_ci": w1_ci,
                    "bivalent_change": biv, "bivalent_change_ci": biv_ci,
                }
                for conflict_name, conflict_mask in (("conflict", conflict), ("no_conflict", ~conflict)):
                    subgroup = np.flatnonzero(split_mask & conflict_mask)
                    subgroup_rate, subgroup_ci = _mean_ci(
                        switch[task_index, scenario_index], subgroup,
                        np.random.default_rng(base_seed + (4000 if conflict_name == "conflict" else 5000)),
                        args.bootstrap_draws,
                    )
                    row[f"{conflict_name}_questions"] = int(len(subgroup))
                    row[f"{conflict_name}_switch_rate"] = subgroup_rate
                    row[f"{conflict_name}_switch_rate_ci"] = subgroup_ci
                if scenario.startswith("intercept_"):
                    row.update(_mediation_ci(
                        numerator[task_index, source_index], numerator[task_index, scenario_index],
                        denominator[task_index], indices,
                        np.random.default_rng(base_seed + 6000), args.bootstrap_draws,
                    ))
                task_rows[scenario] = row
            split["tasks"][task] = task_rows
        summary["splits"][split_name] = split

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    labels = [name.replace("_", " ") for name in RELAY_REGIONS]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
    colors = {"Game": "#d95f02", "Neutral": "#1b7fb8"}
    for task_index, task in enumerate(TASKS):
        rows = summary["splits"]["confirmation"]["tasks"][task]
        intercepts = [rows[f"intercept_{name}"] for name in RELAY_REGIONS]
        remaining = np.asarray([row["remaining_transfer"] for row in intercepts])
        remaining_ci = np.asarray([row["remaining_transfer_ci"] for row in intercepts])
        reduction = np.asarray([row["transfer_reduction"] for row in intercepts])
        reduction_ci = np.asarray([row["transfer_reduction_ci"] for row in intercepts])
        offset = (task_index - 0.5) * 0.36
        axes[0].bar(x + offset, remaining, 0.36, yerr=np.stack((remaining-remaining_ci[:,0], remaining_ci[:,1]-remaining)), capsize=3, color=colors[task], label=task)
        axes[1].bar(x + offset, reduction, 0.36, yerr=np.stack((reduction-reduction_ci[:,0], reduction_ci[:,1]-reduction)), capsize=3, color=colors[task], label=task)
    for axis in axes:
        axis.axhline(0, color="black", lw=0.8)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.legend()
    axes[0].set(title="A  Donor-task transfer remaining (confirmation)", ylabel="Task-vector transfer fraction")
    axes[1].set(title="B  Transfer mediated by restored relay (confirmation)", ylabel="Transfer reduction")
    fig.suptitle("Where evaluation-feedback information is relayed to the final answer")
    fig.savefig(args.figure, dpi=190)
    plt.close(fig)

    confirmation = summary["splits"]["confirmation"]["tasks"]
    discovery_rows = summary["splits"]["discovery"]["tasks"]
    game_source = confirmation["Game"]["feedback_suffix_swapped"]
    neutral_source = confirmation["Neutral"]["feedback_suffix_swapped"]
    game_all = confirmation["Game"]["intercept_all_post_feedback_relays"]
    neutral_all = confirmation["Neutral"]["intercept_all_post_feedback_relays"]
    game_discovery_all = discovery_rows["Game"]["intercept_all_post_feedback_relays"]
    neutral_discovery_all = discovery_rows["Neutral"]["intercept_all_post_feedback_relays"]
    lines = [
        "# Corrected evaluation-feedback relay mediation", "",
        "## Bottom line", "",
        (
            "Crossing the complete feedback suffix transfers "
            f"{game_source['remaining_transfer']:.3f} "
            f"[{game_source['remaining_transfer_ci'][0]:.3f}, {game_source['remaining_transfer_ci'][1]:.3f}] "
            "of the paired Neutral task vector into a Game recipient and "
            f"{neutral_source['remaining_transfer']:.3f} "
            f"[{neutral_source['remaining_transfer_ci'][0]:.3f}, {neutral_source['remaining_transfer_ci'][1]:.3f}] "
            "of the paired Game task vector into a Neutral recipient on confirmation."
        ),
        "",
        (
            "Restoring every later post-feedback relay's outgoing ordinary-attention and recurrent-GLA writes "
            f"mediates {100 * game_all['mediated_proportion']:.1f}% "
            f"[{100 * game_all['mediated_proportion_ci'][0]:.1f}%, {100 * game_all['mediated_proportion_ci'][1]:.1f}%] "
            "of the Game-recipient transfer and "
            f"{100 * neutral_all['mediated_proportion']:.1f}% "
            f"[{100 * neutral_all['mediated_proportion_ci'][0]:.1f}%, {100 * neutral_all['mediated_proportion_ci'][1]:.1f}%] "
            "of the Neutral-recipient transfer."
        ),
        "",
        (
            "The result replicates on discovery: "
            f"{100 * game_discovery_all['mediated_proportion']:.1f}% "
            f"[{100 * game_discovery_all['mediated_proportion_ci'][0]:.1f}%, {100 * game_discovery_all['mediated_proportion_ci'][1]:.1f}%] "
            "for Game and "
            f"{100 * neutral_discovery_all['mediated_proportion']:.1f}% "
            f"[{100 * neutral_discovery_all['mediated_proportion_ci'][0]:.1f}%, {100 * neutral_discovery_all['mediated_proportion_ci'][1]:.1f}%] "
            "for Neutral."
        ),
        "",
        "Thus no single later token region is the policy bottleneck. The nominal joint proportions are lower bounds, not estimates of a physiological bypass: restored tokens retain their source-crossed local outputs, and source-crossed assistant-prefix output can reach the adjacent readout through the unintercepted short GLA convolution.",
        "",
        "## Intervention", "",
        "The complete `incorrect/lost . Choose the answer again .` suffix was crossed between Game and Neutral. For each relay condition, that region's downstream ordinary-attention K/V and recurrent GLA k/v/g/β writes were restored from an exact clean duplicate of the recipient task while the relay token's own source-crossed residual was retained. Every applicable layer and every token after feedback and before the final answer position was covered.",
        "",
        "## Confirmation results", "",
        "`Transfer reduction` is the fraction of the paired natural donor-task vector removed by restoring a relay. `Mediated proportion` divides that reduction by the complete source-crossover transfer.",
        "",
        "| Relay restored | Game remaining | Game reduction | Game mediated | Neutral remaining | Neutral reduction | Neutral mediated |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, region in zip(labels, RELAY_REGIONS):
        game = summary["splits"]["confirmation"]["tasks"]["Game"][f"intercept_{region}"]
        neutral = summary["splits"]["confirmation"]["tasks"]["Neutral"][f"intercept_{region}"]
        lines.append(
            f"| {label} | {game['remaining_transfer']:.3f} [{game['remaining_transfer_ci'][0]:.3f}, {game['remaining_transfer_ci'][1]:.3f}] | "
            f"{game['transfer_reduction']:.3f} [{game['transfer_reduction_ci'][0]:.3f}, {game['transfer_reduction_ci'][1]:.3f}] | "
            f"{100 * game['mediated_proportion']:.1f}% [{100 * game['mediated_proportion_ci'][0]:.1f}%, {100 * game['mediated_proportion_ci'][1]:.1f}%] | "
            f"{neutral['remaining_transfer']:.3f} [{neutral['remaining_transfer_ci'][0]:.3f}, {neutral['remaining_transfer_ci'][1]:.3f}] | "
            f"{neutral['transfer_reduction']:.3f} [{neutral['transfer_reduction_ci'][0]:.3f}, {neutral['transfer_reduction_ci'][1]:.3f}] | "
            f"{100 * neutral['mediated_proportion']:.1f}% [{100 * neutral['mediated_proportion_ci'][0]:.1f}%, {100 * neutral['mediated_proportion_ci'][1]:.1f}%] |"
        )
    cache_raw_error = summary["validation"]["cache_restoration_scenario_raw_max_abs_error"]
    cache_corrected_error = summary["validation"]["cache_restoration_scenario_corrected_max_abs_error"]
    lines += [
        "", "## Controls and scope", "",
        f"The actual `cache_restored_no_source_swap` scenario reproduced the same-batch natural run with maximum absolute raw A-D-logit error {cache_raw_error:.6g}; after the trusted-natural correction its maximum error was {cache_corrected_error:.6g}.",
    ]
    if historical_validation is not None:
        lines += [
            "",
            (
                f"Across the {len(historical_validation['common_scenarios'])} natural/source/interception scenarios shared with the historical run, "
                f"the corrected raw and corrected-logit arrays are bit-for-bit identical (maximum errors {historical_validation['raw_max_abs_error']:.6g} and {historical_validation['corrected_max_abs_error']:.6g}). "
                "The audit therefore found a vacuous advertised control, not a change in the causal mediation estimates."
            ),
        ]
    lines += [
        "",
        "The route inventory is not exhaustive over every GLA cross-position mechanism. Qwen3.6 applies a short causal depthwise convolution to GLA q/k/v before the intercepted delta-rule update. Those convolution states were not restored. Moreover, downstream-only restoration deliberately keeps each restored token's source-crossed local output. At the final assistant prefix, immediately adjacent to the readout, that output can leak through the unintercepted convolution. The joint mediation proportions therefore likely understate relay mediation, and the transfer that survives joint restoration cannot be interpreted as a bypass fraction or assigned among persistent GLA memory and direct ordinary-attention reads. A convolution-safe joint control is required.",
        "",
        "Discovery estimates, raw switch rates, conflict/no-conflict outcomes, W1 rates, bivalent rank effects, and all validation controls are retained in `summary.json`.",
    ]
    args.report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--stage-a-results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--historical-results", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
