from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .analyze_triviamc_matching_history_blockade import (
    LETTERS,
    RANKS,
    SCENARIOS,
    TASKS,
    _advantage,
    _align_to_original,
    _bootstrap,
    _fmt,
    _load,
    _semantic_choices,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plot(summary: dict[str, Any], path: Path) -> None:
    result = summary["results"]["confirmation"]
    colors = {"Game": "#2878b5", "Neutral": "#e07a2f"}
    x = np.arange(4)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.5))

    ax = axes[0, 0]
    for offset, task in zip((-0.10, 0.10), TASKS):
        row = result["rankwise_matching_minus_cyclic"][task]
        mean = np.asarray(row["mean"])
        low = np.asarray(row["ci95_low"])
        high = np.asarray(row["ci95_high"])
        ax.errorbar(
            x + offset,
            mean,
            yerr=np.vstack((mean - low, high - mean)),
            marker="o",
            linewidth=1.8,
            capsize=4,
            color=colors[task],
            label=task,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, RANKS)
    ax.set_ylabel("Candidate-centered logit change")
    ax.set_title("Matching blockade minus cyclic control")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    row = result["rankwise_task_interaction"]
    mean = np.asarray(row["mean"])
    low = np.asarray(row["ci95_low"])
    high = np.asarray(row["ci95_high"])
    ax.errorbar(
        x,
        mean,
        yerr=np.vstack((mean - low, high - mean)),
        marker="o",
        linewidth=1.8,
        capsize=4,
        color="#6a3d9a",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, RANKS)
    ax.set_ylabel("Game minus Neutral effect")
    ax.set_title("Task difference in matching-history use")

    ax = axes[1, 0]
    labels = ("Natural", "Matching\nblockade", "Cyclic wrong\nblockade")
    width = 0.22
    for task_index, task in enumerate(TASKS):
        rows = [result["w1_choice_rates"][scenario][task] for scenario in SCENARIOS]
        means = np.asarray([row["mean"] * 100 for row in rows])
        lows = np.asarray([row["ci95"][0] * 100 for row in rows])
        highs = np.asarray([row["ci95"][1] * 100 for row in rows])
        positions = np.arange(3) + (task_index - 0.5) * width
        ax.bar(positions, means, width, color=colors[task], label=task)
        ax.errorbar(
            positions,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Choice is old semantic W1 (%)")
    ax.set_title("Old-winner choice under each condition")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    names = ("Matching − cyclic", "Matching − natural")
    rows = [
        result["w1_task_gap_changes"]["matching_minus_cyclic"],
        result["w1_task_gap_changes"]["matching_minus_natural"],
    ]
    means = np.asarray([row["mean"] * 100 for row in rows])
    lows = np.asarray([row["ci95"][0] * 100 for row in rows])
    highs = np.asarray([row["ci95"][1] * 100 for row in rows])
    ax.bar(np.arange(2), means, width=0.55, color=("#6a3d9a", "#4daf4a"))
    ax.errorbar(
        np.arange(2),
        means,
        yerr=np.vstack((means - lows, highs - means)),
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(2), names)
    ax.set_ylabel("Change in Game − Neutral W1 choice (pp)")
    ax.set_title("Does the route support preferential Game switching?")

    fig.suptitle(
        f"{summary['design']['model_label']} {summary['design']['dataset']}: causal matching-history blockade\n"
        f"All {summary['design']['attention_layer_count']} attention layers; frozen confirmation split "
        f"(n={summary['validation']['confirmation']})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> None:
    arrays = _load(args.results)
    qids = arrays["question_ids"].astype(str).tolist()
    if len(qids) != 500 or not np.all(arrays["completed"].astype(bool)):
        raise RuntimeError("Expected a complete 500-question run")
    if arrays["scenarios"].astype(str).tolist() != list(SCENARIOS):
        raise RuntimeError("Unexpected scenario inventory")
    for key in (
        "baseline_logits",
        "natural_logits",
        "trusted_natural_logits",
        "joint_matching_logits",
        "joint_cyclic_wrong_logits",
    ):
        if not np.all(np.isfinite(arrays[key])):
            raise RuntimeError(f"Non-finite values in {key}")
    if not all(
        np.all(arrays[key] > 0)
        for key in (
            "source_position_counts",
            "query_position_counts",
            "cyclic_source_position_counts",
        )
    ):
        raise RuntimeError("One or more executed source/query spans were empty")

    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    if args.split_plan is not None:
        split = json.loads(args.split_plan.read_text())
        discovery_ids = set(split["discovery_question_ids"])
        confirmation_ids = set(split["confirmation_question_ids"])
    else:
        if args.discovery_plan is None or args.confirmation_plan is None:
            raise RuntimeError("Provide --split-plan or both separate split plans")
        discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
        confirmation_ids = set(json.loads(args.confirmation_plan.read_text())["question_ids"])
    if discovery_ids & confirmation_ids or discovery_ids | confirmation_ids != set(qids):
        raise RuntimeError("Frozen split does not partition the executed questions")
    masks = {
        "discovery": np.asarray([qid in discovery_ids for qid in qids]),
        "confirmation": np.asarray([qid in confirmation_ids for qid in qids]),
        "all": np.ones(len(qids), dtype=bool),
    }

    rank_contents = arrays["rank_contents"].astype(str)
    rank_indices = np.asarray(
        [[LETTERS.index(value) for value in row] for row in rank_contents], dtype=int
    ).T
    strata_all = rank_contents[:, 0]
    natural = _align_to_original(arrays["natural_logits"].astype(float), qids, mappings)
    matching = _align_to_original(
        arrays["joint_matching_logits"].astype(float), qids, mappings
    )
    cyclic = _align_to_original(
        arrays["joint_cyclic_wrong_logits"].astype(float), qids, mappings
    )
    choices = {
        "natural": _semantic_choices(arrays["natural_logits"], qids, mappings),
        "joint_matching": _semantic_choices(
            arrays["joint_matching_logits"], qids, mappings
        ),
        "joint_cyclic_wrong": _semantic_choices(
            arrays["joint_cyclic_wrong_logits"], qids, mappings
        ),
    }
    w1 = rank_indices[0]

    summary: dict[str, Any] = {
        "status": "complete",
        "design": {
            "model": args.model_id,
            "model_label": args.model_label,
            "dataset": args.dataset,
            "questions": 500,
            "tasks": list(TASKS),
            "scenarios": list(SCENARIOS),
            "attention_layers_one_based": list(range(1, args.layer_count + 1)),
            "attention_layer_count": args.layer_count,
            "attention_architecture": args.attention_architecture,
            "matching_blockade": (
                "Every token of each complete 2P option line is denied attention reads "
                "from every token of its semantically matching complete 1P option line."
            ),
            "cyclic_wrong_control": "W1<-W2, W2<-W3, W3<-W4, W4<-W1",
            "correctness_endpoint": False,
        },
        "validation": {
            "questions": len(qids),
            "discovery": int(masks["discovery"].sum()),
            "confirmation": int(masks["confirmation"].sum()),
            "all_outputs_finite": True,
            "all_source_query_counts_positive": True,
            "natural_max_abs_error_to_trusted": float(
                np.max(
                    np.abs(
                        arrays["natural_logits"].astype(float)
                        - arrays["trusted_natural_logits"].astype(float)
                    )
                )
            ),
            "natural_displayed_choice_agreement_to_trusted": float(
                (
                    np.argmax(arrays["natural_logits"], axis=-1)
                    == np.argmax(arrays["trusted_natural_logits"], axis=-1)
                ).mean()
            ),
            "matching_source_tokens": {
                "min": int(arrays["source_position_counts"].min()),
                "max": int(arrays["source_position_counts"].max()),
                "mean": float(arrays["source_position_counts"].mean()),
            },
            "receiver_tokens": {
                "min": int(arrays["query_position_counts"].min()),
                "max": int(arrays["query_position_counts"].max()),
                "mean": float(arrays["query_position_counts"].mean()),
            },
        },
        "results": {},
        "provenance": {
            "results": {"path": str(args.results), "sha256": _sha256(args.results)},
            "remapping_plan": {
                "path": str(args.remapping_plan),
                "sha256": _sha256(args.remapping_plan),
            },
            "split_plan": (
                {"path": str(args.split_plan), "sha256": _sha256(args.split_plan)}
                if args.split_plan is not None
                else None
            ),
            "discovery_plan": (
                {"path": str(args.discovery_plan), "sha256": _sha256(args.discovery_plan)}
                if args.discovery_plan is not None
                else None
            ),
            "confirmation_plan": (
                {"path": str(args.confirmation_plan), "sha256": _sha256(args.confirmation_plan)}
                if args.confirmation_plan is not None
                else None
            ),
        },
    }

    for subset_index, (subset, mask) in enumerate(masks.items()):
        strata = strata_all[mask]
        rank_specific = np.empty((2, len(qids), 4), dtype=float)
        matching_natural = np.empty_like(rank_specific)
        cyclic_natural = np.empty_like(rank_specific)
        for task in range(2):
            for rank in range(4):
                target = rank_indices[rank]
                natural_advantage = _advantage(natural[task], target)
                matching_advantage = _advantage(matching[task], target)
                cyclic_advantage = _advantage(cyclic[task], target)
                rank_specific[task, :, rank] = matching_advantage - cyclic_advantage
                matching_natural[task, :, rank] = matching_advantage - natural_advantage
                cyclic_natural[task, :, rank] = cyclic_advantage - natural_advantage

        rows: dict[str, Any] = {
            "n": int(mask.sum()),
            "rankwise_matching_minus_cyclic": {},
            "rankwise_matching_minus_natural": {},
            "rankwise_cyclic_minus_natural": {},
            "w1_choice_rates": {},
            "w1_task_gaps": {},
            "w1_task_gap_changes": {},
        }
        for task_index, task in enumerate(TASKS):
            seed = args.seed + subset_index * 10000 + task_index * 1000
            rows["rankwise_matching_minus_cyclic"][task] = _bootstrap(
                rank_specific[task_index, mask], strata, seed + 1, args.draws
            )
            rows["rankwise_matching_minus_natural"][task] = _bootstrap(
                matching_natural[task_index, mask], strata, seed + 2, args.draws
            )
            rows["rankwise_cyclic_minus_natural"][task] = _bootstrap(
                cyclic_natural[task_index, mask], strata, seed + 3, args.draws
            )
        rows["rankwise_task_interaction"] = _bootstrap(
            (rank_specific[0] - rank_specific[1])[mask],
            strata,
            args.seed + subset_index * 10000 + 9001,
            args.draws,
        )

        indicators: dict[str, np.ndarray] = {}
        for scenario_index, scenario in enumerate(SCENARIOS):
            indicator = (choices[scenario] == w1[None, :]).astype(float)
            indicators[scenario] = indicator
            rows["w1_choice_rates"][scenario] = {}
            for task_index, task in enumerate(TASKS):
                rows["w1_choice_rates"][scenario][task] = _bootstrap(
                    indicator[task_index, mask],
                    strata,
                    args.seed + subset_index * 10000 + scenario_index * 100 + task_index,
                    args.draws,
                )
            rows["w1_task_gaps"][scenario] = _bootstrap(
                (indicator[0] - indicator[1])[mask],
                strata,
                args.seed + subset_index * 10000 + scenario_index * 100 + 50,
                args.draws,
            )
        gap_matching = indicators["joint_matching"][0] - indicators["joint_matching"][1]
        gap_cyclic = indicators["joint_cyclic_wrong"][0] - indicators["joint_cyclic_wrong"][1]
        gap_natural = indicators["natural"][0] - indicators["natural"][1]
        rows["w1_task_gap_changes"]["matching_minus_cyclic"] = _bootstrap(
            (gap_matching - gap_cyclic)[mask],
            strata,
            args.seed + subset_index * 10000 + 9801,
            args.draws,
        )
        rows["w1_task_gap_changes"]["matching_minus_natural"] = _bootstrap(
            (gap_matching - gap_natural)[mask],
            strata,
            args.seed + subset_index * 10000 + 9802,
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
        f"# {args.model_label} {args.dataset} matching-history blockade",
        "",
        "## What was tested",
        "",
        f"For every frozen {args.dataset} question and in both Game and Neutral, every token "
        "of each complete 2P option line was prevented from attending to every token of "
        f"its semantically matching complete 1P option line at all {args.layer_count} attention "
        "layers. The cyclic control edited the same receivers and layers but used the "
        "next old-rank source line. This is a whole-line attention-edge intervention, "
        "not a residual replacement or token-localization result.",
        "",
        "## Frozen confirmation results",
        "",
        "Matching blockade minus cyclic control, candidate-centered A-D logits:",
        "",
        "| Task | W1 | W2 | W3 | W4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        row = conf["rankwise_matching_minus_cyclic"][task]
        report.append(
            "| " + task + " | " + " | ".join(
                f"{row['mean'][i]:+.3f} [{row['ci95_low'][i]:+.3f}, {row['ci95_high'][i]:+.3f}]"
                for i in range(4)
            ) + " |"
        )
    row = conf["rankwise_task_interaction"]
    report.append(
        "| Game minus Neutral | " + " | ".join(
            f"{row['mean'][i]:+.3f} [{row['ci95_low'][i]:+.3f}, {row['ci95_high'][i]:+.3f}]"
            for i in range(4)
        ) + " |"
    )
    report.extend(
        [
            "",
            "Old-semantic-W1 choice rates:",
            "",
            "| Scenario | Game | Neutral | Game minus Neutral |",
            "|---|---:|---:|---:|",
        ]
    )
    labels = {
        "natural": "Natural",
        "joint_matching": "Matching blockade",
        "joint_cyclic_wrong": "Cyclic wrong-line blockade",
    }
    for scenario in SCENARIOS:
        report.append(
            f"| {labels[scenario]} | "
            f"{_fmt(conf['w1_choice_rates'][scenario]['Game'], 100, 1)} | "
            f"{_fmt(conf['w1_choice_rates'][scenario]['Neutral'], 100, 1)} | "
            f"{_fmt(conf['w1_task_gaps'][scenario], 100, 1)} |"
        )
    report.extend(
        [
            "",
            "Primary matching-minus-cyclic change in the Game-minus-Neutral W1-choice "
            f"gap: **{_fmt(conf['w1_task_gap_changes']['matching_minus_cyclic'], 100, 1)} "
            "percentage points**.",
            "",
            "## Validation and scope",
            "",
            f"- Natural reproduction maximum absolute error: "
            f"{summary['validation']['natural_max_abs_error_to_trusted']:.8f}.",
            f"- Natural displayed-choice agreement: "
            f"{summary['validation']['natural_displayed_choice_agreement_to_trusted'] * 100:.2f}%.",
            "- All outputs were finite and every executed source and receiver span was nonempty.",
            f"- The intervention covered all {args.layer_count} attention layers. {args.architecture_scope}",
            "- Correctness is not an endpoint.",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(report))
    print(json.dumps(summary["validation"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--dataset", choices=("SimpleMC", "TriviaMC"), required=True)
    parser.add_argument("--split-plan", type=Path)
    parser.add_argument("--discovery-plan", type=Path)
    parser.add_argument("--confirmation-plan", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--model-id", default="ByteDance-Seed/Seed-OSS-36B-Instruct")
    parser.add_argument("--model-label", default="Seed-OSS 36B")
    parser.add_argument("--layer-count", type=int, default=64)
    parser.add_argument(
        "--attention-architecture",
        default="64 grouped-query causal self-attention layers",
    )
    parser.add_argument(
        "--architecture-scope",
        default="No GLA or recurrent state exists in Seed.",
    )
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
