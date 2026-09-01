from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .run_feedback_source_group_crossover import SCENARIOS, SOURCE_GROUP_OFFSETS


TASKS = ("Game", "Neutral")
COMPONENTS = (
    "feedback_sentence",
    "following_instruction",
    "additive_sum",
    "complete_suffix",
    "nonlinear_interaction",
)
LABELS = {
    "feedback_sentence": "Feedback sentence",
    "following_instruction": "Following instruction",
    "additive_sum": "Separate effects summed",
    "complete_suffix": "Complete suffix",
    "nonlinear_interaction": "Complete − sum",
}


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _mean_ci(
    values: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, Any]:
    selected = np.asarray(values[indices], dtype=np.float64)
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        stop = min(start + 200, draws)
        rows = rng.integers(0, len(selected), size=(stop - start, len(selected)))
        samples[start:stop] = selected[rows].mean(axis=1)
    return {
        "mean": float(selected.mean()),
        "ci": [float(value) for value in np.quantile(samples, (0.025, 0.975))],
        "n": int(len(selected)),
    }


def _decomposition(
    component_numerators: dict[str, np.ndarray],
    denominator: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, Any]:
    selected = {
        name: np.asarray(values[indices], dtype=np.float64)
        for name, values in component_numerators.items()
    }
    den = np.asarray(denominator[indices], dtype=np.float64)
    full = selected["complete_suffix"]
    output: dict[str, Any] = {}
    boot_transfer = {name: np.empty(draws, dtype=np.float64) for name in COMPONENTS}
    boot_relative = {name: np.empty(draws, dtype=np.float64) for name in COMPONENTS}
    for start in range(0, draws, 200):
        stop = min(start + 200, draws)
        rows = rng.integers(0, len(den), size=(stop - start, len(den)))
        selected_den = den[rows].sum(axis=1)
        selected_full = full[rows].sum(axis=1)
        for name in COMPONENTS:
            selected_num = selected[name][rows].sum(axis=1)
            boot_transfer[name][start:stop] = selected_num / selected_den
            boot_relative[name][start:stop] = selected_num / selected_full
    for name in COMPONENTS:
        numerator = selected[name].sum()
        output[name] = {
            "transfer_fraction": float(numerator / den.sum()),
            "transfer_fraction_ci": [
                float(value) for value in np.quantile(boot_transfer[name], (0.025, 0.975))
            ],
            "fraction_of_complete": float(numerator / full.sum()),
            "fraction_of_complete_ci": [
                float(value) for value in np.quantile(boot_relative[name], (0.025, 0.975))
            ],
        }
    return output


def analyze(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    qids = arrays["question_ids"].astype(str).tolist()
    scenarios = arrays["scenario_ids"].astype(str).tolist()
    if scenarios != list(SCENARIOS):
        raise RuntimeError("Grouped source scenario definition changed")
    if len(qids) != 500 or not arrays["completed"].all():
        raise RuntimeError("A complete 500-question grouped-source result is required")
    logits = arrays["scenario_final_logits"].astype(np.float64)
    raw = arrays["scenario_final_logits_raw"].astype(np.float64)
    if logits.shape != (2, len(SCENARIOS), 500, 4) or not np.isfinite(logits).all():
        raise RuntimeError("Grouped-source logits are incomplete or non-finite")

    historical_validation: dict[str, Any] | None = None
    if args.historical_source_results is not None:
        with np.load(args.historical_source_results, allow_pickle=False) as loaded:
            historical = {key: loaded[key] for key in loaded.files}
        if not np.array_equal(historical["question_ids"], arrays["question_ids"]):
            raise RuntimeError("Historical complete-suffix question order changed")
        old_scenarios = historical["scenario_ids"].astype(str).tolist()
        old_full = old_scenarios.index("feedback_suffix_3_9_swapped")
        new_full = scenarios.index("complete_suffix_swapped")
        historical_validation = {
            "natural_raw_max_abs_error": float(np.max(np.abs(
                historical["scenario_final_logits_raw"][:, old_scenarios.index("natural")]
                - raw[:, scenarios.index("natural")]
            ))),
            "complete_suffix_raw_max_abs_error": float(np.max(np.abs(
                historical["scenario_final_logits_raw"][:, old_full] - raw[:, new_full]
            ))),
            "complete_suffix_corrected_max_abs_error": float(np.max(np.abs(
                historical["scenario_final_logits"][:, old_full] - logits[:, new_full]
            ))),
            "complete_suffix_corrected_bit_exact": bool(np.array_equal(
                historical["scenario_final_logits"][:, old_full], logits[:, new_full]
            )),
        }

    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    if [int(discovery.sum()), int((~discovery).sum())] != [251, 249]:
        raise RuntimeError("Frozen discovery/confirmation split changed")
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
    natural = centered[:, scenarios.index("natural")]
    task_vector = natural[::-1] - natural
    denominator = np.sum(task_vector * task_vector, axis=-1)
    delta = centered - natural[:, None]
    numerator = np.sum(delta * task_vector[:, None], axis=-1)
    scenario_index = {name: scenarios.index(f"{name}_swapped") for name in SOURCE_GROUP_OFFSETS}
    choices = np.argmax(logits, axis=-1)
    switch = choices != w1[None, None, :]
    choose_w1 = choices == w1[None, None, :]
    ranked = np.empty_like(centered)
    for qi in range(500):
        ranked[:, :, qi] = np.take(centered[:, :, qi], rank_order[qi], axis=-1)
    bivalent = ranked[..., 3] - ranked[..., :2].mean(axis=-1)

    summary: dict[str, Any] = {
        "question": "How do the feedback sentence and following identical instruction separately and jointly transmit policy?",
        "coverage": {
            "questions": 500,
            "discovery": 251,
            "confirmation": 249,
            "tasks": list(TASKS),
            "source_groups": {key: list(value) for key, value in SOURCE_GROUP_OFFSETS.items()},
            "ordinary_attention_layers": list(range(4, 65, 4)),
            "gla_layers": [value for value in range(1, 65) if value % 4 != 0],
        },
        "definitions": {
            "transfer_fraction": "Projection of the intervention-induced centered A-D logit change onto the paired natural donor-minus-recipient task vector.",
            "fraction_of_complete": "Component task-vector numerator divided by the complete-suffix numerator in the same task and split.",
            "additive_sum": "Feedback-sentence numerator plus following-instruction numerator, paired question by question.",
            "nonlinear_interaction": "Complete-suffix numerator minus the feedback-sentence and following-instruction numerators; positive is synergy and negative is subadditivity/redundancy.",
            "source_scope": "All 16 ordinary-attention K/V writes and all 48 recurrent GLA k/v/g/beta writes, with source-token output preserved.",
        },
        "validation": {
            "all_complete": True,
            "all_finite": True,
            "corrected_natural_max_abs_error": float(np.max(np.abs(
                logits[:, scenarios.index("natural")] - arrays["trusted_natural_logits"]
            ))),
            "same_batch_natural_max_abs_error": float(np.max(np.abs(
                raw[:, scenarios.index("natural")] - arrays["same_batch_natural_logits"]
            ))),
            "historical_complete_suffix": historical_validation,
        },
        "splits": {},
    }

    for split_index, (split_name, mask) in enumerate(split_masks.items()):
        indices = np.flatnonzero(mask)
        split: dict[str, Any] = {
            "questions": int(mask.sum()),
            "conflict_questions": int((mask & conflict).sum()),
            "tasks": {},
        }
        for task_index, task in enumerate(TASKS):
            fb = numerator[task_index, scenario_index["feedback_sentence"]]
            instruction = numerator[task_index, scenario_index["following_instruction"]]
            full = numerator[task_index, scenario_index["complete_suffix"]]
            component_numerators = {
                "feedback_sentence": fb,
                "following_instruction": instruction,
                "additive_sum": fb + instruction,
                "complete_suffix": full,
                "nonlinear_interaction": full - fb - instruction,
            }
            rng = np.random.default_rng(
                args.seed + split_index * 100000 + task_index * 10000
            )
            decomposition = _decomposition(
                component_numerators, denominator[task_index], indices, rng,
                args.bootstrap_draws,
            )
            behavior: dict[str, Any] = {}
            for source_name, index in scenario_index.items():
                behavior[source_name] = {
                    "switch_rate": _mean_ci(
                        switch[task_index, index], indices, rng, args.bootstrap_draws
                    ),
                    "choose_W1_rate": _mean_ci(
                        choose_w1[task_index, index], indices, rng, args.bootstrap_draws
                    ),
                    "bivalent_change": _mean_ci(
                        bivalent[task_index, index] - bivalent[task_index, 0],
                        indices, rng, args.bootstrap_draws,
                    ),
                }
            behavior["natural"] = {
                "switch_rate": _mean_ci(
                    switch[task_index, 0], indices, rng, args.bootstrap_draws
                ),
                "choose_W1_rate": _mean_ci(
                    choose_w1[task_index, 0], indices, rng, args.bootstrap_draws
                ),
            }
            split["tasks"][task] = {
                "decomposition": decomposition,
                "behavior": behavior,
            }
        summary["splits"][split_name] = split

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#B9A43A", "#54A24B", "#E45756"]
    x = np.arange(len(COMPONENTS))
    for row_index, split_name in enumerate(("discovery", "confirmation")):
        for col_index, task in enumerate(TASKS):
            axis = axes[row_index, col_index]
            rows = summary["splits"][split_name]["tasks"][task]["decomposition"]
            means = np.asarray([rows[name]["transfer_fraction"] for name in COMPONENTS])
            cis = np.asarray([rows[name]["transfer_fraction_ci"] for name in COMPONENTS])
            axis.bar(
                x, means, color=colors,
                yerr=np.stack((means - cis[:, 0], cis[:, 1] - means)), capsize=4,
            )
            axis.axhline(0, color="black", lw=0.8)
            axis.set_xticks(x, [LABELS[name] for name in COMPONENTS], rotation=28, ha="right")
            axis.set_ylabel("Donor task-vector transfer fraction")
            axis.set_title(f"{'A' if row_index == 0 and col_index == 0 else 'B' if row_index == 0 else 'C' if col_index == 0 else 'D'}  {split_name.title()} — {task}")
            axis.grid(axis="y", color="0.9", lw=0.8)
    fig.suptitle("Feedback sentence and following instruction: separate, joint, and nonlinear policy transfer", fontsize=14)
    fig.savefig(args.figure, dpi=190)
    plt.close(fig)

    lines = [
        "# Grouped feedback-source crossover",
        "",
        "## Question and intervention",
        "",
        "The seven-token policy suffix was divided into two prespecified groups: the feedback sentence "
        "`incorrect/lost | .` and the following identical instruction `Choose | the | answer | again | .`. "
        "Each group and their complete union were reciprocally crossed between Game and Neutral across all "
        "16 ordinary-attention layers and all 48 GLA layers. Source-token outputs were preserved, so the "
        "intervention changes downstream ordinary-attention K/V and recurrent GLA k/v/g/β writes rather than "
        "replacing the source token's own residual.",
        "",
        "The nonlinear interaction is `complete suffix − feedback sentence − following instruction`. "
        "Positive values indicate synergy; negative values indicate subadditivity or redundant information.",
        "",
    ]
    if historical_validation is not None:
        lines += [
            "## Replication control",
            "",
            f"The new complete-suffix corrected logits differ from the established complete-suffix run by at most "
            f"{historical_validation['complete_suffix_corrected_max_abs_error']:.6g} logits "
            f"(bit-exact: {historical_validation['complete_suffix_corrected_bit_exact']}).",
            "",
        ]
    for split_name in ("confirmation", "discovery"):
        lines += [
            f"## {split_name.title()} results",
            "",
            "| Task | Component | Transfer (95% CI) | Fraction of complete (95% CI) |",
            "|---|---|---:|---:|",
        ]
        for task in TASKS:
            rows = summary["splits"][split_name]["tasks"][task]["decomposition"]
            for component in COMPONENTS:
                row = rows[component]
                lines.append(
                    f"| {task} | {LABELS[component]} | {row['transfer_fraction']:.3f} "
                    f"[{row['transfer_fraction_ci'][0]:.3f}, {row['transfer_fraction_ci'][1]:.3f}] | "
                    f"{100*row['fraction_of_complete']:.1f}% "
                    f"[{100*row['fraction_of_complete_ci'][0]:.1f}%, {100*row['fraction_of_complete_ci'][1]:.1f}%] |"
                )
        lines.append("")
    lines += [
        "## Scope",
        "",
        "The intervention covers every applicable ordinary-attention and recurrent GLA write layer. As in the "
        "established complete-suffix source crossover, it does not patch Qwen3.6's short causal GLA q/k/v "
        "convolution state; estimates therefore concern the measured outgoing K/V and delta-rule write channels.",
        "",
        "Raw switch rates, W1 rates, bivalent changes, discovery results, and validation controls are retained in `summary.json`.",
    ]
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["validation"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--historical-source-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


if __name__ == "__main__":
    analyze(build_parser().parse_args())
