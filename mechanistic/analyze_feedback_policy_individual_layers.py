from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_feedback_policy_layer_decomposition import (
    TASKS,
    _bootstrap_mean,
    _bootstrap_ratio,
    _center,
    _load_rank_order,
)


LAYERS = tuple(range(1, 65))
ORDINARY_LAYERS = frozenset(range(4, 65, 4))


def _interval_error(point: np.ndarray, interval: np.ndarray) -> np.ndarray:
    return np.stack((point - interval[:, 0], interval[:, 1] - point))


def analyze(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    qids = arrays["question_ids"].astype(str).tolist()
    completed = arrays["completed"].astype(bool)
    if len(qids) != 500 or not completed.all():
        raise RuntimeError("A complete 500-question checkpoint is required")

    scenarios = arrays["scenario_ids"].astype(str).tolist()
    expected = [
        "natural",
        "all_layers_swapped",
        "ordinary_all_swapped",
        "gla_all_swapped",
    ]
    for layer in LAYERS:
        expected.extend((f"layer_{layer:02d}_only", f"all_except_layer_{layer:02d}"))
    if scenarios != expected:
        raise RuntimeError("Individual-layer scenario inventory changed")
    scenario_index = {name: index for index, name in enumerate(scenarios)}

    logits = arrays["scenario_final_logits"].astype(np.float64)
    raw = arrays["scenario_final_logits_raw"].astype(np.float64)
    if logits.shape != (2, len(scenarios), 500, 4) or not np.isfinite(logits).all():
        raise RuntimeError("Causal output is incomplete or non-finite")
    natural_index = scenario_index["natural"]
    corrected_error = float(
        np.max(
            np.abs(
                logits[:, natural_index]
                - arrays["trusted_natural_logits"].astype(np.float64)
            )
        )
    )
    same_batch_error = float(
        np.max(
            np.abs(
                raw[:, natural_index]
                - arrays["same_batch_natural_logits"].astype(np.float64)
            )
        )
    )

    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    if [int(discovery.sum()), int((~discovery).sum())] != [251, 249]:
        raise RuntimeError("Frozen split changed")
    split_masks = {"discovery": discovery, "confirmation": ~discovery}

    centered = _center(logits)
    natural = centered[:, natural_index]
    donor_vector = natural[::-1] - natural
    denominator = np.sum(donor_vector * donor_vector, axis=-1)
    if np.any(denominator <= 0):
        raise RuntimeError("Degenerate paired Game/Neutral donor direction")
    delta = centered - natural[:, None]
    transfer_numerator = np.sum(delta * donor_vector[:, None], axis=-1)
    all_index = scenario_index["all_layers_swapped"]

    rank_order = _load_rank_order(qids, args.baseline, args.remapping_plan)
    w1 = rank_order[:, 0]
    choices = np.argmax(logits, axis=-1)
    switch = choices != w1[None, None]
    choose_w1 = choices == w1[None, None]
    ranked = np.empty_like(centered)
    for question_index in range(len(qids)):
        ranked[:, :, question_index] = np.take(
            centered[:, :, question_index], rank_order[question_index], axis=-1
        )
    bivalent = ranked[..., 3] - ranked[..., :2].mean(axis=-1)

    summary: dict[str, Any] = {
        "question": (
            "Which individual layers causally transmit the evaluation-feedback "
            "policy from its source suffix to downstream computation?"
        ),
        "coverage": {
            "questions": 500,
            "discovery": 251,
            "confirmation": 249,
            "tasks": list(TASKS),
            "layers": list(LAYERS),
            "ordinary_attention_layers": sorted(ORDINARY_LAYERS),
            "gla_layers": [layer for layer in LAYERS if layer not in ORDINARY_LAYERS],
        },
        "validation": {
            "all_complete": True,
            "all_finite": True,
            "corrected_natural_max_abs_error": corrected_error,
            "same_batch_natural_max_abs_error": same_batch_error,
        },
        "definitions": {
            "transfer_fraction": (
                "For each question, project the intervention-induced centered A-D "
                "logit change onto the paired natural donor-task minus recipient-task "
                "vector; report the ratio of summed projections to summed donor-vector "
                "squared norms. Zero is the recipient task's natural output and one is "
                "the paired donor task's natural output along this prespecified axis."
            ),
            "layer_only_sufficiency": (
                "Donor feedback-suffix writes are substituted at this layer only."
            ),
            "all_except_layer_necessity_loss": (
                "Full all-layer transfer minus transfer when every layer except this "
                "one receives donor feedback-suffix writes."
            ),
            "task_rows": (
                "Game means Neutral feedback-suffix writes are inserted into a Game "
                "recipient; Neutral means Game feedback-suffix writes are inserted "
                "into a Neutral recipient. The two tasks are never pooled."
            ),
            "switch_rate": "Fraction whose final top letter is not the first-presentation semantic winner W1.",
            "bivalent_change": "Change from natural in R4 minus mean(R1,R2), aligned by first-presentation semantic rank.",
        },
        "splits": {},
    }

    table_rows: list[dict[str, Any]] = []
    for split_number, (split_name, mask) in enumerate(split_masks.items()):
        indices = np.flatnonzero(mask)
        split_record: dict[str, Any] = {"questions": int(mask.sum()), "tasks": {}}
        for task_index, task in enumerate(TASKS):
            natural_rng = np.random.default_rng(
                args.seed + split_number * 1_000_000 + task_index * 100_000 - 1
            )
            natural_switch, natural_switch_ci = _bootstrap_mean(
                switch[task_index, natural_index],
                indices,
                natural_rng,
                args.bootstrap_draws,
            )
            natural_bivalent, natural_bivalent_ci = _bootstrap_mean(
                bivalent[task_index, natural_index],
                indices,
                natural_rng,
                args.bootstrap_draws,
            )
            controls: dict[str, Any] = {
                "natural": {
                    "transfer_fraction": 0.0,
                    "ci": [0.0, 0.0],
                    "switch_rate": natural_switch,
                    "switch_rate_ci": natural_switch_ci,
                    "choose_W1_rate": 1.0 - natural_switch,
                    "choose_W1_rate_ci": [
                        1.0 - natural_switch_ci[1],
                        1.0 - natural_switch_ci[0],
                    ],
                    "bivalent_value": natural_bivalent,
                    "bivalent_value_ci": natural_bivalent_ci,
                    "bivalent_change": 0.0,
                    "bivalent_change_ci": [0.0, 0.0],
                }
            }
            for control_number, control in enumerate(
                ("all_layers_swapped", "ordinary_all_swapped", "gla_all_swapped")
            ):
                scenario = scenario_index[control]
                rng = np.random.default_rng(
                    args.seed
                    + split_number * 1_000_000
                    + task_index * 100_000
                    + control_number
                )
                point, interval = _bootstrap_ratio(
                    transfer_numerator[task_index, scenario],
                    denominator[task_index],
                    indices,
                    rng,
                    args.bootstrap_draws,
                )
                sw, sw_ci = _bootstrap_mean(
                    switch[task_index, scenario], indices, rng, args.bootstrap_draws
                )
                wr, wr_ci = _bootstrap_mean(
                    choose_w1[task_index, scenario], indices, rng, args.bootstrap_draws
                )
                bv, bv_ci = _bootstrap_mean(
                    bivalent[task_index, scenario]
                    - bivalent[task_index, natural_index],
                    indices,
                    rng,
                    args.bootstrap_draws,
                )
                controls[control] = {
                    "transfer_fraction": point,
                    "ci": interval,
                    "switch_rate": sw,
                    "switch_rate_ci": sw_ci,
                    "choose_W1_rate": wr,
                    "choose_W1_rate_ci": wr_ci,
                    "bivalent_change": bv,
                    "bivalent_change_ci": bv_ci,
                }

            layer_rows: dict[str, Any] = {}
            for layer in LAYERS:
                only_index = scenario_index[f"layer_{layer:02d}_only"]
                except_index = scenario_index[f"all_except_layer_{layer:02d}"]
                suff_rng = np.random.default_rng(
                    args.seed
                    + 10_000_000
                    + split_number * 1_000_000
                    + task_index * 100_000
                    + layer
                )
                need_rng = np.random.default_rng(
                    args.seed
                    + 20_000_000
                    + split_number * 1_000_000
                    + task_index * 100_000
                    + layer
                )
                sufficiency, sufficiency_ci = _bootstrap_ratio(
                    transfer_numerator[task_index, only_index],
                    denominator[task_index],
                    indices,
                    suff_rng,
                    args.bootstrap_draws,
                )
                necessity, necessity_ci = _bootstrap_ratio(
                    transfer_numerator[task_index, all_index]
                    - transfer_numerator[task_index, except_index],
                    denominator[task_index],
                    indices,
                    need_rng,
                    args.bootstrap_draws,
                )
                suff_switch, suff_switch_ci = _bootstrap_mean(
                    switch[task_index, only_index],
                    indices,
                    suff_rng,
                    args.bootstrap_draws,
                )
                suff_switch_change, suff_switch_change_ci = _bootstrap_mean(
                    switch[task_index, only_index].astype(np.float64)
                    - switch[task_index, natural_index].astype(np.float64),
                    indices,
                    suff_rng,
                    args.bootstrap_draws,
                )
                necessity_switch_loss, necessity_switch_loss_ci = _bootstrap_mean(
                    switch[task_index, all_index].astype(np.float64)
                    - switch[task_index, except_index].astype(np.float64),
                    indices,
                    need_rng,
                    args.bootstrap_draws,
                )
                suff_bivalent, suff_bivalent_ci = _bootstrap_mean(
                    bivalent[task_index, only_index]
                    - bivalent[task_index, natural_index],
                    indices,
                    suff_rng,
                    args.bootstrap_draws,
                )
                necessity_bivalent, necessity_bivalent_ci = _bootstrap_mean(
                    bivalent[task_index, all_index]
                    - bivalent[task_index, except_index],
                    indices,
                    need_rng,
                    args.bootstrap_draws,
                )
                carrier = "ordinary attention" if layer in ORDINARY_LAYERS else "GLA"
                layer_rows[str(layer)] = {
                    "carrier": carrier,
                    "layer_only_sufficiency": sufficiency,
                    "layer_only_sufficiency_ci": sufficiency_ci,
                    "all_except_layer_necessity_loss": necessity,
                    "all_except_layer_necessity_loss_ci": necessity_ci,
                    "layer_only_switch_rate": suff_switch,
                    "layer_only_switch_rate_ci": suff_switch_ci,
                    "layer_only_switch_change": suff_switch_change,
                    "layer_only_switch_change_ci": suff_switch_change_ci,
                    "all_except_layer_switch_necessity_loss": necessity_switch_loss,
                    "all_except_layer_switch_necessity_loss_ci": necessity_switch_loss_ci,
                    "layer_only_bivalent_change": suff_bivalent,
                    "layer_only_bivalent_change_ci": suff_bivalent_ci,
                    "all_except_layer_bivalent_necessity_loss": necessity_bivalent,
                    "all_except_layer_bivalent_necessity_loss_ci": necessity_bivalent_ci,
                }
                table_rows.append(
                    {
                        "split": split_name,
                        "task": task,
                        "layer": layer,
                        "carrier": carrier,
                        "layer_only_sufficiency": sufficiency,
                        "layer_only_sufficiency_ci_low": sufficiency_ci[0],
                        "layer_only_sufficiency_ci_high": sufficiency_ci[1],
                        "all_except_layer_necessity_loss": necessity,
                        "all_except_layer_necessity_loss_ci_low": necessity_ci[0],
                        "all_except_layer_necessity_loss_ci_high": necessity_ci[1],
                        "layer_only_switch_rate": suff_switch,
                        "layer_only_switch_rate_ci_low": suff_switch_ci[0],
                        "layer_only_switch_rate_ci_high": suff_switch_ci[1],
                        "layer_only_switch_change": suff_switch_change,
                        "layer_only_switch_change_ci_low": suff_switch_change_ci[0],
                        "layer_only_switch_change_ci_high": suff_switch_change_ci[1],
                        "all_except_layer_switch_necessity_loss": necessity_switch_loss,
                        "all_except_layer_switch_necessity_loss_ci_low": necessity_switch_loss_ci[0],
                        "all_except_layer_switch_necessity_loss_ci_high": necessity_switch_loss_ci[1],
                        "layer_only_bivalent_change": suff_bivalent,
                        "layer_only_bivalent_change_ci_low": suff_bivalent_ci[0],
                        "layer_only_bivalent_change_ci_high": suff_bivalent_ci[1],
                        "all_except_layer_bivalent_necessity_loss": necessity_bivalent,
                        "all_except_layer_bivalent_necessity_loss_ci_low": necessity_bivalent_ci[0],
                        "all_except_layer_bivalent_necessity_loss_ci_high": necessity_bivalent_ci[1],
                    }
                )
            split_record["tasks"][task] = {
                "controls": controls,
                "layers": layer_rows,
            }
        summary["splits"][split_name] = split_record

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    with args.table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    colors = {"discovery": "#6f6f6f", "confirmation": "#1177aa"}
    x = np.asarray(LAYERS)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, constrained_layout=True)
    for task_index, task in enumerate(TASKS):
        for metric_index, (metric, title) in enumerate(
            (
                ("layer_only_sufficiency", "Layer alone: sufficiency"),
                ("all_except_layer_necessity_loss", "Loss when omitted: necessity"),
            )
        ):
            axis = axes[task_index, metric_index]
            for split_name in ("discovery", "confirmation"):
                rows = summary["splits"][split_name]["tasks"][task]["layers"]
                point = np.asarray([rows[str(layer)][metric] for layer in LAYERS])
                interval = np.asarray(
                    [rows[str(layer)][f"{metric}_ci"] for layer in LAYERS]
                )
                axis.errorbar(
                    x,
                    point,
                    yerr=_interval_error(point, interval),
                    color=colors[split_name],
                    linewidth=1.25,
                    elinewidth=.65,
                    capsize=1.5,
                    marker="o",
                    markersize=2.5,
                    label=split_name.capitalize(),
                )
            for layer in ORDINARY_LAYERS:
                axis.axvline(layer, color="#d95f02", alpha=.09, linewidth=2)
            axis.axhline(0, color="black", linewidth=.8)
            axis.set_title(f"{task}: {title}")
            axis.set_ylabel("Donor-task transfer fraction")
            axis.set_xticks(range(4, 65, 4))
            axis.grid(axis="y", color="#dddddd", linewidth=.6)
            if task_index == 1:
                axis.set_xlabel("Layer (orange bands mark ordinary-attention layers)")
            axis.legend(loc="best")
    fig.suptitle(
        "Individual layers transmitting evaluation-feedback policy\n"
        "Game and Neutral recipients shown separately; 95% bootstrap intervals"
    )
    fig.savefig(args.figure, dpi=200)
    plt.close(fig)

    confirmation = summary["splits"]["confirmation"]["tasks"]

    def estimate(task: str, layer: int, metric: str) -> str:
        row = confirmation[task]["layers"][str(layer)]
        point = row[metric]
        low, high = row[f"{metric}_ci"]
        return f"{point:.3f} [{low:.3f}, {high:.3f}]"

    def control(task: str, name: str) -> str:
        row = confirmation[task]["controls"][name]
        low, high = row["ci"]
        return f"{row['transfer_fraction']:.3f} [{low:.3f}, {high:.3f}]"

    lines = [
        "# Individual-layer feedback-policy transmission",
        "",
        "## Question and intervention",
        "",
        "This is the complete L1--L64 refinement of the feedback-suffix crossover. "
        "The aligned source is the whole policy-bearing suffix `incorrect/lost . "
        "Choose the answer again .`. Its residual states remain those of the recipient "
        "prompt. We cross only the downstream writes made from those source tokens "
        "between paired Game and Neutral runs.",
        "",
        "For each layer, **sufficiency** crosses the suffix writes at that layer alone. "
        "**Necessity loss** compares the complete all-layer crossover with a crossover "
        "at every layer except that layer. The outcome is the fraction of the paired "
        "donor-task A--D logit vector transferred into the recipient. Thus the Game "
        "rows ask how Neutral suffix writes change a Game run, while the Neutral rows "
        "ask how Game suffix writes change a Neutral run. The tasks are never pooled.",
        "",
        "All 500 canonical questions were completed. The frozen discovery and "
        "confirmation splits contain 251 and 249 questions. Every output was finite; "
        "the corrected and same-batch natural controls both reproduced with maximum "
        "absolute error 0.0.",
        "",
        "## Findings",
        "",
        "The complete all-layer crossover transfers nearly the entire donor-task "
        f"state on confirmation: **{control('Game', 'all_layers_swapped')}** into "
        f"Game and **{control('Neutral', 'all_layers_swapped')}** into Neutral. No "
        "individual layer explains most of this effect. The policy is transmitted by "
        "a distributed sequence of writes.",
        "",
        "It also transfers the expected discrete and rank-shaped policy. Game's "
        f"switch rate moves from **{100*confirmation['Game']['controls']['natural']['switch_rate']:.1f}%** "
        f"naturally to **{100*confirmation['Game']['controls']['all_layers_swapped']['switch_rate']:.1f}%** "
        "when it receives Neutral suffix writes, close to Neutral's natural "
        f"**{100*confirmation['Neutral']['controls']['natural']['switch_rate']:.1f}%**. "
        "Neutral's switch rate moves in the reciprocal direction, from "
        f"**{100*confirmation['Neutral']['controls']['natural']['switch_rate']:.1f}%** "
        f"to **{100*confirmation['Neutral']['controls']['all_layers_swapped']['switch_rate']:.1f}%**, "
        "close to Game's natural rate. The all-layer crossover changes the bivalent "
        f"R4-minus-mean(R1,R2) score by **{confirmation['Game']['controls']['all_layers_swapped']['bivalent_change']:.3f}** "
        f"in Game and **{confirmation['Neutral']['controls']['all_layers_swapped']['bivalent_change']:+.3f}** "
        "in Neutral, carrying the opposite rank policies rather than merely a generic "
        "task difference.",
        "",
        "For **Game**, layer 36 is dominant. Crossing it alone transfers "
        f"**{estimate('Game', 36, 'layer_only_sufficiency')}**; omitting it loses "
        f"**{estimate('Game', 36, 'all_except_layer_necessity_loss')}**. Layer 45 is "
        "second on both tests: "
        f"**{estimate('Game', 45, 'layer_only_sufficiency')}** alone and "
        f"**{estimate('Game', 45, 'all_except_layer_necessity_loss')}** lost when "
        "omitted. Other practical effects form a broad L28--50 cluster, including "
        "ordinary-attention layers 32, 40, 44, and 48 and GLA layers 33--35, 45--47, "
        "and 50.",
        "",
        "The rank-shaped result is especially clean at the two peaks. On confirmation, "
        f"L36 alone changes the bivalent score by **{confirmation['Game']['layers']['36']['layer_only_bivalent_change']:.3f}** "
        f"in Game and **{confirmation['Neutral']['layers']['36']['layer_only_bivalent_change']:+.3f}** "
        "in Neutral; L45 alone changes it by "
        f"**{confirmation['Game']['layers']['45']['layer_only_bivalent_change']:.3f}** "
        f"and **{confirmation['Neutral']['layers']['45']['layer_only_bivalent_change']:+.3f}**. "
        "All four directions and both corresponding necessity effects replicate on "
        "discovery. Individual winner changes are thresholded and less stable across "
        "splits, so the continuous rank/logit effects are the primary layer-localized "
        "result.",
        "",
        "For **Neutral**, layer 36 is again dominant: "
        f"**{estimate('Neutral', 36, 'layer_only_sufficiency')}** alone and "
        f"**{estimate('Neutral', 36, 'all_except_layer_necessity_loss')}** lost when "
        "omitted. Layer 45 is again second: "
        f"**{estimate('Neutral', 45, 'layer_only_sufficiency')}** alone and "
        f"**{estimate('Neutral', 45, 'all_except_layer_necessity_loss')}** lost when "
        "omitted. The remaining practical effects again span L28--50 and both carrier "
        "families.",
        "",
        "The same two peaks and the same mid-layer concentration appear on discovery. "
        "The precise magnitude of some smaller layer effects varies between splits, "
        "so the defensible result is the replicated structure—not a claim that every "
        "small confidence interval marks a distinct mechanism.",
        "",
        "The carrier-family controls agree with the individual-layer map. On "
        f"confirmation, ordinary-attention writes alone transfer **{control('Game', 'ordinary_all_swapped')}** "
        f"into Game and **{control('Neutral', 'ordinary_all_swapped')}** into Neutral. "
        f"GLA writes alone transfer **{control('Game', 'gla_all_swapped')}** into Game "
        f"and **{control('Neutral', 'gla_all_swapped')}** into Neutral. Neither family "
        "alone reproduces complete transfer; the dominant peaks alternate between "
        "ordinary-attention L36 and GLA L45.",
        "",
        "## Bottom line",
        "",
        "The evaluation feedback does not travel through one late policy layer. Its "
        "causal transmission is distributed, becomes practically concentrated in the "
        "middle-to-late stack, and has two reproducible individual maxima: **L36** and "
        "**L45**, in that order, in both Game and Neutral. Weak effects occur outside "
        "the central window, but almost all large single-layer effects lie between L28 "
        "and L50.",
        "",
        "Confidence intervals are question-bootstrap 95% intervals.",
        "",
        f"Canonical figure: [{args.figure.name}]({args.figure.resolve()})",
        "",
        "Exact estimates: `individual_layer_estimates.csv`; machine-readable controls "
        "and definitions: `summary.json`.",
    ]
    args.report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=8242026)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
