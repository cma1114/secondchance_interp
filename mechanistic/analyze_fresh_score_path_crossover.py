from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


LETTERS = "ABCD"
TASKS = ("Game", "Neutral")


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _mean_ci(
    values: np.ndarray, indices: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[float, list[float]]:
    selected = np.asarray(values[indices], dtype=np.float64)
    point = float(selected.mean())
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        stop = min(draws, start + 200)
        rows = rng.integers(0, len(selected), size=(stop-start, len(selected)))
        samples[start:stop] = selected[rows].mean(axis=1)
    return point, [float(x) for x in np.quantile(samples, (0.025, 0.975))]


def _ratio_ci(
    numerator: np.ndarray, denominator: np.ndarray, indices: np.ndarray,
    rng: np.random.Generator, draws: int,
) -> tuple[float, list[float]]:
    num = np.asarray(numerator[indices], dtype=np.float64)
    den = np.asarray(denominator[indices], dtype=np.float64)
    point = float(num.sum() / den.sum())
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        stop = min(draws, start + 200)
        rows = rng.integers(0, len(num), size=(stop-start, len(num)))
        samples[start:stop] = num[rows].sum(axis=1) / den[rows].sum(axis=1)
    return point, [float(x) for x in np.quantile(samples, (0.025, 0.975))]


def _align_semantics(
    logits: np.ndarray, plan_rows: list[dict[str, Any]]
) -> np.ndarray:
    # Input: task x variant x scenario x question x displayed A-D.
    result = np.empty_like(logits, dtype=np.float64)
    for qi, row in enumerate(plan_rows):
        for vi, mapping in enumerate((row["low_new_to_original"], row["high_new_to_original"])):
            for displayed_index, displayed in enumerate(LETTERS):
                original_index = LETTERS.index(mapping[displayed])
                result[:, vi, :, qi, original_index] = logits[:, vi, :, qi, displayed_index]
    return result


def analyze(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    plan_rows = json.loads(args.pair_plan.read_text())["rows"]
    if len(plan_rows) != 500 or not arrays["completed"].astype(bool).all():
        raise RuntimeError("A complete 500-question causal result is required")
    scenarios = arrays["scenario_ids"].astype(str).tolist()
    scenario_index = {name: index for index, name in enumerate(scenarios)}
    raw = arrays["scenario_final_logits"].astype(np.float64)
    if raw.shape != (2, 2, len(scenarios), 500, 4) or not np.isfinite(raw).all():
        raise RuntimeError("Causal logits are incomplete or non-finite")
    semantic = _center(_align_semantics(raw, plan_rows))
    natural_si = scenario_index["natural"]
    natural = semantic[:, :, natural_si]
    donor_vector = natural[:, ::-1] - natural
    denominator = np.sum(donor_vector * donor_vector, axis=-1)
    delta = semantic - natural[:, :, None]
    numerator = np.sum(delta * donor_vector[:, :, None], axis=-1)

    target_indices = np.asarray([
        LETTERS.index(row["target_original_letter"]) for row in plan_rows
    ])
    target_advantage = np.empty(semantic.shape[:-1], dtype=np.float64)
    for qi, target in enumerate(target_indices):
        target_advantage[..., qi] = semantic[..., qi, target]

    discovery = arrays["split"].astype(str) == "discovery"
    if [int(discovery.sum()), int((~discovery).sum())] != [251, 249]:
        raise RuntimeError("Frozen split changed")
    split_masks = {"discovery": discovery, "confirmation": ~discovery}

    summary: dict[str, Any] = {
        "question": "How does fresh 2P evidence causally reach the final answer?",
        "coverage": {
            "questions": 500, "discovery": 251, "confirmation": 249,
            "tasks": list(TASKS), "layers": "all applicable L1-L64",
            "source": "all four complete 2P option lines",
            "relays": ["2P choice cue and query", "final assistant prefix"],
        },
        "validation": {
            "all_complete": True,
            "all_finite": True,
            "duplicate_natural_max_abs_error": float(np.max(np.abs(
                raw[:, :, scenario_index["duplicate_natural"]]
                - raw[:, :, natural_si]
            ))),
            "minimum_screen_target_fresh_difference": float(np.min([
                row["target_fresh_score_difference"] for row in plan_rows
            ])),
        },
        "definitions": {
            "transfer_fraction": (
                "Projection of the intervention-induced centered semantic A-D change "
                "onto the same task/question's natural opposite-2P-order minus current-order vector"
            ),
            "mediation_loss": (
                "All-option-line crossover transfer minus transfer when the named relay's "
                "downstream writes are restored to their natural recipient values"
            ),
        },
        "splits": {},
    }

    for split_number, (split_name, mask) in enumerate(split_masks.items()):
        indices = np.flatnonzero(mask)
        split: dict[str, Any] = {"questions": int(mask.sum()), "tasks": {}}
        for ti, task in enumerate(TASKS):
            rows: dict[str, Any] = {}
            for si, scenario in enumerate(scenarios):
                rng = np.random.default_rng(args.seed + split_number*100000 + ti*10000 + si)
                transfer, transfer_ci = _ratio_ci(
                    numerator[ti, :, si].reshape(-1, 500).sum(axis=0),
                    denominator[ti].reshape(-1, 500).sum(axis=0),
                    indices, rng, args.bootstrap_draws,
                )
                # Average reciprocal target-semantic effects after orienting
                # each recipient toward its donor order.
                target_delta = target_advantage[ti, :, si] - target_advantage[ti, :, natural_si]
                natural_target_donor = (
                    target_advantage[ti, ::-1, natural_si] - target_advantage[ti, :, natural_si]
                )
                oriented = np.sign(natural_target_donor) * target_delta
                target_point, target_ci = _mean_ci(
                    oriented.mean(axis=0), indices, rng, args.bootstrap_draws
                )
                rows[scenario] = {
                    "transfer_fraction": transfer,
                    "transfer_fraction_ci": transfer_ci,
                    "oriented_target_logit_change": target_point,
                    "oriented_target_logit_change_ci": target_ci,
                }
            source = rows["option_lines_swapped"]
            mediation: dict[str, Any] = {}
            source_num = numerator[ti, :, scenario_index["option_lines_swapped"]].sum(axis=0)
            for offset, scenario in enumerate((
                "intercept_choice_cue_and_query",
                "intercept_final_assistant_prefix",
                "intercept_both_relays",
            )):
                intercept_num = numerator[ti, :, scenario_index[scenario]].sum(axis=0)
                rng = np.random.default_rng(
                    args.seed + 500000 + split_number*100000 + ti*10000 + offset
                )
                point, ci = _ratio_ci(
                    source_num - intercept_num, denominator[ti].sum(axis=0),
                    indices, rng, args.bootstrap_draws,
                )
                mediation[scenario] = {"mediation_loss": point, "mediation_loss_ci": ci}
            natural_target_difference = (
                target_advantage[ti, 1, natural_si] - target_advantage[ti, 0, natural_si]
            )
            screen_target_difference = np.asarray([
                row["target_high_fresh_score"] - row["target_low_fresh_score"]
                for row in plan_rows
            ])
            rng = np.random.default_rng(args.seed + 700000 + split_number*100000 + ti*10000)
            slope_values = natural_target_difference / screen_target_difference
            slope, slope_ci = _mean_ci(slope_values, indices, rng, args.bootstrap_draws)
            split["tasks"][task] = {
                "scenarios": rows,
                "mediation": mediation,
                "natural_target_final_change_per_screen_fresh_logit": slope,
                "natural_target_final_change_per_screen_fresh_logit_ci": slope_ci,
            }
        summary["splits"][split_name] = split

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    colors = {"Game": "#d95f02", "Neutral": "#1b7fb8"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    x = np.arange(2)
    for ti, task in enumerate(TASKS):
        row = summary["splits"]["confirmation"]["tasks"][task]
        value = row["natural_target_final_change_per_screen_fresh_logit"]
        ci = row["natural_target_final_change_per_screen_fresh_logit_ci"]
        axes[0].bar(ti, value, yerr=[[value-ci[0]], [ci[1]-value]], capsize=4,
                    color=colors[task])
        transfer = row["scenarios"]["option_lines_swapped"]["transfer_fraction"]
        transfer_ci = row["scenarios"]["option_lines_swapped"]["transfer_fraction_ci"]
        axes[1].bar(ti, transfer, yerr=[[transfer-transfer_ci[0]], [transfer_ci[1]-transfer]],
                    capsize=4, color=colors[task])
    for axis in axes[:2]:
        axis.axhline(0, color="black", lw=.8)
        axis.set_xticks(x, TASKS)
    axes[0].set(title="A  Natural fresh-evidence effect", ylabel="Final target change / screened fresh change")
    axes[1].set(title="B  Complete 2P-line crossover", ylabel="Donor-order transfer fraction")

    mediation_names = [
        "intercept_choice_cue_and_query", "intercept_final_assistant_prefix", "intercept_both_relays"
    ]
    labels = ["choice cue/query", "final prefix", "both"]
    xx = np.arange(3)
    for ti, task in enumerate(TASKS):
        mediation = summary["splits"]["confirmation"]["tasks"][task]["mediation"]
        values = np.asarray([mediation[name]["mediation_loss"] for name in mediation_names])
        cis = np.asarray([mediation[name]["mediation_loss_ci"] for name in mediation_names])
        axes[2].bar(xx + (ti-.5)*.36, values, .36,
                    yerr=np.stack((values-cis[:, 0], cis[:, 1]-values)), capsize=3,
                    color=colors[task], label=task)
    axes[2].axhline(0, color="black", lw=.8)
    axes[2].set_xticks(xx, labels, rotation=25, ha="right")
    axes[2].set(title="C  Downstream mediation", ylabel="Transfer removed")
    axes[2].legend()
    fig.suptitle("Causal path of fresh second-presentation evidence")
    fig.savefig(args.figure, dpi=190)
    plt.close(fig)

    lines = [
        "# Fresh second-presentation evidence path", "",
        "Within each question and task, the causal pair holds the entire 1P history, feedback policy, target semantic candidate, and target 2P display position fixed. Only the other three 2P candidates are reordered. All four complete 2P option-line writes are crossed at every applicable ordinary-attention and GLA layer. A duplicate-natural full forward is the exact execution control; a split cached-prefix path was rejected during preflight because it was not numerically identical and is not part of this result.", "",
        "Complete estimates for Game and Neutral, both frozen splits, and all relay interceptions are in `summary.json`.", "",
        f"Canonical figure: [{args.figure.name}]({args.figure.resolve()})",
    ]
    args.report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--pair-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=8232026)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
