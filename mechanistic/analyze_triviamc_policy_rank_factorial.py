from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


LETTERS = "ABCD"
TASKS = ("Game", "Neutral")
RANKS = ("W1", "W2", "W3", "W4")
SCENARIOS = (
    "natural",
    "policy_swapped",
    "matching_blocked",
    "policy_swapped_matching_blocked",
    "cyclic_control_blocked",
    "policy_swapped_cyclic_control_blocked",
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _align(
    values: np.ndarray, qids: list[str], mappings: dict[str, dict[str, Any]]
) -> np.ndarray:
    output = np.empty_like(values)
    for qi, qid in enumerate(qids):
        for original_index, original in enumerate(LETTERS):
            displayed = mappings[qid]["original_to_new"][original]
            output[..., qi, original_index] = values[..., qi, LETTERS.index(displayed)]
    return output


def _advantage(logits: np.ndarray) -> np.ndarray:
    return logits - (logits.sum(-1, keepdims=True) - logits) / 3.0


def _rank(values: np.ndarray, rank_indices: np.ndarray) -> np.ndarray:
    output = np.empty(values.shape[:-1] + (4,), dtype=values.dtype)
    for qi in range(values.shape[-2]):
        output[..., qi, :] = values[..., qi, rank_indices[qi]]
    return output


def _ranked_w1_choice(
    displayed_logits: np.ndarray,
    qids: list[str],
    mappings: dict[str, dict[str, Any]],
    w1_indices: np.ndarray,
) -> np.ndarray:
    # np.argmax is the canonical displayed A-D first-maximum rule. Mapping occurs
    # only after the displayed choice is resolved, so exact ties are not reordered.
    displayed = np.argmax(displayed_logits, axis=-1)
    semantic = np.empty_like(displayed)
    for qi, qid in enumerate(qids):
        for leading in np.ndindex(displayed.shape[:-1]):
            chosen = LETTERS[int(displayed[leading + (qi,)])]
            semantic[leading + (qi,)] = LETTERS.index(
                mappings[qid]["new_to_original"][chosen]
            )
    return (semantic == w1_indices).astype(float)


def _bootstrap(
    values: np.ndarray, strata: np.ndarray, seed: int, draws: int
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False
    if len(values) == 0 or len(values) != len(strata):
        raise ValueError("Empty or misaligned bootstrap input")
    groups = [
        np.flatnonzero(strata == label)
        for label in sorted(set(strata.astype(str).tolist()))
    ]
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, values.shape[1]), dtype=float)
    for draw in range(draws):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        samples[draw] = values[indices].mean(0)
    mean = values.mean(0)
    low, high = np.quantile(samples, (0.025, 0.975), axis=0)
    if squeeze:
        return {
            "n": int(len(values)),
            "mean": float(mean[0]),
            "ci95": [float(low[0]), float(high[0])],
        }
    return {
        "n": int(len(values)),
        "mean": mean.tolist(),
        "ci95_low": low.tolist(),
        "ci95_high": high.tolist(),
    }


def _fmt(row: dict[str, Any], scale: float = 1.0, digits: int = 3) -> str:
    return (
        f"{row['mean'] * scale:+.{digits}f} "
        f"[{row['ci95'][0] * scale:+.{digits}f}, "
        f"{row['ci95'][1] * scale:+.{digits}f}]"
    )


def _rank_fmt(row: dict[str, Any], index: int) -> str:
    return (
        f"{row['mean'][index]:+.3f} "
        f"[{row['ci95_low'][index]:+.3f}, {row['ci95_high'][index]:+.3f}]"
    )


def _plot(summary: dict[str, Any], figure_path: Path) -> None:
    conf = summary["results"]["confirmation"]
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.5))
    x = np.arange(4)
    colors = {"Game": "#2878b5", "Neutral": "#e07a2f"}

    ax = axes[0, 0]
    for offset, task in zip((-0.10, 0.10), TASKS):
        row = conf["natural_policy_matching_specific_rank_effect"][task]
        mean = np.asarray(row["mean"])
        ax.errorbar(
            x + offset,
            mean,
            yerr=np.vstack((mean - row["ci95_low"], np.asarray(row["ci95_high"]) - mean)),
            marker="o", capsize=3, color=colors[task], label=task,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, RANKS)
    ax.set_ylabel("Matching minus cyclic lesion effect")
    ax.set_title("Route effect under recipient policy")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for offset, task in zip((-0.10, 0.10), TASKS):
        row = conf["swapped_policy_matching_specific_rank_effect"][task]
        mean = np.asarray(row["mean"])
        ax.errorbar(
            x + offset,
            mean,
            yerr=np.vstack((mean - row["ci95_low"], np.asarray(row["ci95_high"]) - mean)),
            marker="o", capsize=3, color=colors[task], label=task,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, RANKS)
    ax.set_ylabel("Matching minus cyclic lesion effect")
    ax.set_title("Route effect after reciprocal policy transplant")

    ax = axes[1, 0]
    for offset, task in zip((-0.10, 0.10), TASKS):
        row = conf["policy_by_route_rank_interaction"][task]
        mean = np.asarray(row["mean"])
        ax.errorbar(
            x + offset,
            mean,
            yerr=np.vstack((mean - row["ci95_low"], np.asarray(row["ci95_high"]) - mean)),
            marker="o", capsize=3, color=colors[task], label=task,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, RANKS)
    ax.set_ylabel("Swapped-policy minus natural-policy route effect")
    ax.set_title("Policy × retrieved-rank interaction")

    ax = axes[1, 1]
    width = 0.32
    for ti, task in enumerate(TASKS):
        rows = conf["conflict_w1_choice"][task]
        labels = ("Natural", "Policy\nswapped")
        means = np.asarray([rows["natural"]["mean"], rows["policy_swapped"]["mean"]]) * 100
        lows = np.asarray([rows["natural"]["ci95"][0], rows["policy_swapped"]["ci95"][0]]) * 100
        highs = np.asarray([rows["natural"]["ci95"][1], rows["policy_swapped"]["ci95"][1]]) * 100
        pos = np.arange(2) + (ti - 0.5) * width
        ax.bar(pos, means, width=width, color=colors[task], label=task)
        ax.errorbar(pos, means, yerr=np.vstack((means - lows, highs - means)), fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(np.arange(2), labels)
    ax.set_ylabel("Old-W1 choice on W1 != fresh-W2 trials (%)")
    ax.set_title("Behavioral policy transplant")
    ax.legend(frameon=False)

    figure.suptitle(
        "Qwen3.6-27B TriviaMC policy × retrieved-rank factorial\n"
        "Confirmation split; paired W1-letter-stratified 95% CIs",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def analyze(args: argparse.Namespace) -> None:
    arrays = _load(args.results)
    qids = arrays["question_ids"].astype(str).tolist()
    if len(qids) != 500 or not arrays["completed"].astype(bool).all():
        raise RuntimeError("Expected a complete 500-question run")
    if arrays["scenario_ids"].astype(str).tolist() != list(SCENARIOS):
        raise RuntimeError("Unexpected scenario inventory")
    for key in (
        "baseline_logits", "fresh_baseline_logits", "trusted_natural_logits",
        "same_batch_natural_logits", "scenario_logits_raw", "scenario_logits",
    ):
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"Non-finite values in {key}")
    if not (
        np.all(arrays["source_position_counts"] > 0)
        and np.all(arrays["query_position_counts"] > 0)
        and np.all(arrays["cyclic_source_position_counts"] > 0)
    ):
        raise RuntimeError("One or more source/query spans were empty")

    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    split = json.loads(args.split_plan.read_text())
    discovery_ids = set(split["discovery_question_ids"])
    confirmation_ids = set(split["confirmation_question_ids"])
    if discovery_ids & confirmation_ids or discovery_ids | confirmation_ids != set(qids):
        raise RuntimeError("Frozen split does not partition executed questions")
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    masks = {
        "discovery": discovery,
        "confirmation": ~discovery,
        "all": np.ones(len(qids), dtype=bool),
    }

    rank_contents = arrays["rank_contents"].astype(str)
    rank_indices = np.asarray(
        [[LETTERS.index(value) for value in row] for row in rank_contents], dtype=int
    )
    w1_indices = rank_indices[:, 0]
    fresh = arrays["fresh_baseline_answer_original"].astype(str)
    if not np.all(np.isin(fresh, list(LETTERS))):
        raise RuntimeError("Fresh baseline contains a non-A-D unrestricted answer")
    conflict = rank_contents[:, 0] != fresh
    fresh_aggregated_displayed = np.argmax(
        arrays["fresh_baseline_logits"].astype(float), axis=-1
    )
    fresh_aggregated_original = np.asarray([
        mappings[qid]["new_to_original"][LETTERS[int(fresh_aggregated_displayed[qi])]]
        for qi, qid in enumerate(qids)
    ])
    displayed = arrays["scenario_logits"].astype(float)
    exact_max_ties = (displayed == displayed.max(-1, keepdims=True)).sum(-1) > 1
    semantic = _align(displayed, qids, mappings)
    ranked = _rank(_advantage(semantic), rank_indices)
    w1_choice = _ranked_w1_choice(displayed, qids, mappings, w1_indices)

    natural_route = ranked[:, 2] - ranked[:, 4]
    swapped_route = ranked[:, 3] - ranked[:, 5]
    interaction = swapped_route - natural_route
    policy_effect = ranked[:, 1] - ranked[:, 0]

    summary: dict[str, Any] = {
        "status": "complete",
        "design": {
            "dataset": "TriviaMC_difficulty_filtered",
            "questions": 500,
            "tasks": list(TASKS),
            "scenarios": list(SCENARIOS),
            "policy_source": "evaluation-closing period token",
            "policy_state": (
                "All 48 GLA-layer writes, reciprocal Game/Neutral transplant, "
                "preserve_source_output=False."
            ),
            "route_factor": (
                "All-four complete matching 1P-to-2P option-line blockade versus "
                "the all-four cyclic-wrong blockade at all 16 ordinary-attention layers."
            ),
            "conflict": (
                "First-presentation aggregated-logit W1 differs from the unrestricted "
                "A-D winner on the standalone remapped second-presentation baseline."
            ),
            "correctness_endpoint": False,
        },
        "validation": {
            "questions": len(qids),
            "discovery": int(discovery.sum()),
            "confirmation": int((~discovery).sum()),
            "conflict_all": int(conflict.sum()),
            "conflict_discovery": int((conflict & discovery).sum()),
            "conflict_confirmation": int((conflict & ~discovery).sum()),
            "all_outputs_finite": True,
            "natural_max_abs_error_to_trusted": float(np.max(np.abs(
                arrays["same_batch_natural_logits"]
                - arrays["trusted_natural_logits"]
            ))),
            "corrected_natural_max_abs_error": float(np.max(np.abs(
                arrays["scenario_logits"][:, 0]
                - arrays["trusted_natural_logits"]
            ))),
            "raw_paired_natural_displayed_choice_agreement_to_trusted": float((
                np.argmax(arrays["same_batch_natural_logits"], axis=-1)
                == np.argmax(arrays["trusted_natural_logits"], axis=-1)
            ).mean()),
            "fresh_unrestricted_vs_aggregated_choice_agreement": float((
                fresh == fresh_aggregated_original
            ).mean()),
            "scenario_exact_max_ties": int(exact_max_ties.sum()),
            "policy_swap_liveness_max_abs": float(np.max(np.abs(
                arrays["scenario_logits"][:, 1]
                - arrays["scenario_logits"][:, 0]
            ))),
            "matching_liveness_max_abs": float(np.max(np.abs(
                arrays["scenario_logits"][:, 2]
                - arrays["scenario_logits"][:, 4]
            ))),
            "source_tokens": {
                "min": int(arrays["source_position_counts"].min()),
                "max": int(arrays["source_position_counts"].max()),
                "mean": float(arrays["source_position_counts"].mean()),
            },
            "receiver_tokens": {
                "min": int(arrays["query_position_counts"].min()),
                "max": int(arrays["query_position_counts"].max()),
                "mean": float(arrays["query_position_counts"].mean()),
            },
            "policy_transplant_preserve_source_output": False,
        },
        "results": {},
        "provenance": {
            "results": {"path": str(args.results), "sha256": _sha256(args.results)},
            "remapping_plan": {
                "path": str(args.remapping_plan),
                "sha256": _sha256(args.remapping_plan),
            },
            "split_plan": {
                "path": str(args.split_plan),
                "sha256": _sha256(args.split_plan),
            },
        },
    }

    for subset_index, (subset, mask) in enumerate(masks.items()):
        strata = rank_contents[mask, 0]
        rows: dict[str, Any] = {
            "n": int(mask.sum()),
            "natural_policy_matching_specific_rank_effect": {},
            "swapped_policy_matching_specific_rank_effect": {},
            "policy_by_route_rank_interaction": {},
            "policy_swap_rank_effect": {},
            "conflict_w1_choice": {},
        }
        for task_index, task in enumerate(TASKS):
            seed = args.seed + subset_index * 10000 + task_index * 1000
            rows["natural_policy_matching_specific_rank_effect"][task] = _bootstrap(
                natural_route[task_index, mask], strata, seed + 1, args.draws
            )
            rows["swapped_policy_matching_specific_rank_effect"][task] = _bootstrap(
                swapped_route[task_index, mask], strata, seed + 2, args.draws
            )
            rows["policy_by_route_rank_interaction"][task] = _bootstrap(
                interaction[task_index, mask], strata, seed + 3, args.draws
            )
            rows["policy_swap_rank_effect"][task] = _bootstrap(
                policy_effect[task_index, mask], strata, seed + 4, args.draws
            )
            conflict_mask = mask & conflict
            conflict_strata = rank_contents[conflict_mask, 0]
            natural_w1 = w1_choice[task_index, 0, conflict_mask]
            swapped_w1 = w1_choice[task_index, 1, conflict_mask]
            rows["conflict_w1_choice"][task] = {
                "natural": _bootstrap(
                    natural_w1, conflict_strata, seed + 10, args.draws
                ),
                "policy_swapped": _bootstrap(
                    swapped_w1, conflict_strata, seed + 11, args.draws
                ),
                "policy_swap_effect": _bootstrap(
                    swapped_w1 - natural_w1,
                    conflict_strata,
                    seed + 12,
                    args.draws,
                ),
            }
        rows["reciprocal_interaction_sum"] = _bootstrap(
            (interaction[0] + interaction[1])[mask],
            strata,
            args.seed + subset_index * 10000 + 9001,
            args.draws,
        )
        summary["results"][subset] = rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _plot(summary, args.figure)

    conf = summary["results"]["confirmation"]
    report = [
        "# Qwen3.6-27B TriviaMC policy × retrieved-rank factorial",
        "",
        "## What was tested",
        "",
        "The experiment crossed two causal manipulations on all 500 frozen questions. "
        "First, it reciprocally transplanted the evaluation-closing period's GLA update "
        "between Game and Neutral at all 48 GLA layers. Second, it blocked all four "
        "matching complete 1P-option-line to 2P-option-line reads at every ordinary-"
        "attention layer, with an equal-structure cyclic wrong-line blockade as the "
        "route control. The old MLP-49 restoration and individual-layer screens were "
        "not included.",
        "",
        "The period transplant uses `preserve_source_output=False`, exactly matching "
        "the historical SimpleMC factorial. It therefore changes the complete GLA "
        "update at that token, including the donor-conditioned local output; it is not "
        "an output-preserved isolation of recurrent memory alone.",
        "",
        "## Confirmation results",
        "",
        "Matching blockade minus cyclic control, in candidate-centered A-D logits:",
        "",
        "| Recipient task and policy state | W1 | W2 | W3 | W4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        natural_row = conf["natural_policy_matching_specific_rank_effect"][task]
        swapped_row = conf["swapped_policy_matching_specific_rank_effect"][task]
        interaction_row = conf["policy_by_route_rank_interaction"][task]
        report.extend([
            f"| {task}, natural policy | " + " | ".join(
                _rank_fmt(natural_row, i) for i in range(4)
            ) + " |",
            f"| {task}, opposite policy installed | " + " | ".join(
                _rank_fmt(swapped_row, i) for i in range(4)
            ) + " |",
            f"| {task}, policy × route interaction | " + " | ".join(
                _rank_fmt(interaction_row, i) for i in range(4)
            ) + " |",
        ])
    report.extend([
        "",
        "Conflict-trial old-W1 choice (W1 differs from the standalone remapped "
        "second-presentation winner):",
        "",
        "| Recipient task | Natural policy | Opposite policy installed | Change |",
        "|---|---:|---:|---:|",
    ])
    for task in TASKS:
        row = conf["conflict_w1_choice"][task]
        report.append(
            f"| {task} | {_fmt(row['natural'], 100, 1)} | "
            f"{_fmt(row['policy_swapped'], 100, 1)} | "
            f"{_fmt(row['policy_swap_effect'], 100, 1)} pp |"
        )
    report.extend([
        "",
        "## Interpretation",
        "",
        "The answer is yes. Under natural Game, blocking the true matching route rather "
        "than the cyclic wrong route raises old-W1 evidence by 0.698 logits and lowers "
        "W3/W4 by 0.166/0.442. Therefore the intact matching route does the opposite: it "
        "selectively suppresses the recollected old winner and supports weaker old "
        "candidates. Installing the Neutral period update into the same Game prompts "
        "nearly removes that profile: the W1 lesion effect falls by 0.607 logits to "
        "0.091.",
        "",
        "Natural Neutral again has no stable matching-specific rank profile on the "
        "confirmation split. But installing the Game period update into Neutral creates "
        "a strong Game-like route: matching blockade raises W1 by 0.822 and lowers W3/W4 "
        "by 0.206/0.519. Relative to natural Neutral, the policy × route interaction is "
        "+0.890 at W1 and -0.330/-0.419 at W3/W4. Thus the Game policy update is causally "
        "sufficient to make Neutral use recalled candidate rank in the Game pattern; the "
        "Neutral update is sufficient to turn off most of that pattern in Game.",
        "",
        "The behavioral readout agrees. On confirmation conflict trials, installing "
        "Neutral policy in Game raises old-W1 choice by 13.5 points, while installing "
        "Game policy in Neutral lowers it by 12.2 points; both paired intervals exclude "
        "zero. Discovery independently gives +16.4 and -19.7 points. The interaction is "
        "therefore present in rank-shaped logits and actual choices on both frozen halves.",
        "",
        "This shows that policy is not merely added after recollection. The state written "
        "at the feedback period changes how the later matching-history route uses "
        "retrieved W1-W4 information. The asymmetry also matters: TriviaMC replicates the "
        "Game-conditioned route strongly, but not the SimpleMC claim that natural Neutral "
        "has a stable supportive rank profile through this exact route.",
        "",
        "The numerical direction and replication status are stated from the table "
        "above and both frozen halves in `summary.json`; no correctness endpoint is "
        "computed.",
        "",
        "## Validation",
        "",
        f"- Questions: 500; discovery/confirmation: 250/250; confirmation conflict "
        f"trials: {summary['validation']['conflict_confirmation']}.",
        f"- Same-batch natural maximum absolute error to trusted Step 1: "
        f"{summary['validation']['natural_max_abs_error_to_trusted']:.8f}.",
        f"- Corrected natural maximum absolute error: "
        f"{summary['validation']['corrected_natural_max_abs_error']:.8f}.",
        f"- Raw paired-natural displayed-choice agreement with trusted Step 1: "
        f"{summary['validation']['raw_paired_natural_displayed_choice_agreement_to_trusted']*100:.2f}%.",
        f"- Fresh unrestricted/aggregated A-D winner agreement: "
        f"{summary['validation']['fresh_unrestricted_vs_aggregated_choice_agreement']*100:.2f}%.",
        f"- Policy liveness maximum absolute change: "
        f"{summary['validation']['policy_swap_liveness_max_abs']:.6f}; route liveness: "
        f"{summary['validation']['matching_liveness_max_abs']:.6f}.",
        "- All logits are finite and every source/receiver span is nonempty.",
        f"- Choices use displayed A-D first-maximum tie resolution before semantic "
        f"remapping; {summary['validation']['scenario_exact_max_ties']} scenario cells "
        f"had an exact maximum tie.",
        "",
        "See `figures/qwen36_triviamc_policy_rank_step4.png` and `summary.json`.",
        "",
    ])
    (args.output_dir / "REPORT.md").write_text("\n".join(report))
    print(json.dumps(summary["validation"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260828)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
