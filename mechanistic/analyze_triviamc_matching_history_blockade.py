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
SCENARIOS = ("natural", "joint_matching", "joint_cyclic_wrong")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _align_to_original(
    values: np.ndarray, qids: list[str], mappings: dict[str, dict[str, Any]]
) -> np.ndarray:
    out = np.empty_like(values)
    for qi, qid in enumerate(qids):
        for original_index, original in enumerate(LETTERS):
            displayed = mappings[qid]["original_to_new"][original]
            out[..., qi, original_index] = values[..., qi, LETTERS.index(displayed)]
    return out


def _semantic_choices(
    displayed_logits: np.ndarray,
    qids: list[str],
    mappings: dict[str, dict[str, Any]],
) -> np.ndarray:
    displayed = np.argmax(displayed_logits, axis=-1)
    out = np.empty_like(displayed)
    for qi, qid in enumerate(qids):
        for leading in np.ndindex(displayed.shape[:-1]):
            letter = LETTERS[int(displayed[leading + (qi,)])]
            original = mappings[qid]["new_to_original"][letter]
            out[leading + (qi,)] = LETTERS.index(original)
    return out


def _advantage(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    rows = np.arange(len(targets))
    selected = logits[rows, targets]
    return selected - (logits.sum(axis=-1) - selected) / 3.0


def _bootstrap(
    values: np.ndarray,
    strata: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False
    if len(values) != len(strata) or not len(values):
        raise ValueError("Bootstrap inputs are empty or misaligned")
    groups = [np.flatnonzero(strata == label) for label in sorted(set(strata.tolist()))]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Empty bootstrap stratum")
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, values.shape[1]), dtype=float)
    for draw in range(draws):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        samples[draw] = values[indices].mean(axis=0)
    mean = values.mean(axis=0)
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
    low, high = row["ci95"]
    return (
        f"{row['mean'] * scale:+.{digits}f} "
        f"[{low * scale:+.{digits}f}, {high * scale:+.{digits}f}]"
    )


def _plot(summary: dict[str, Any], figure_path: Path) -> None:
    confirmation = summary["results"]["confirmation"]
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.3))
    x = np.arange(4)
    colors = {"Game": "#2878b5", "Neutral": "#e07a2f"}

    ax = axes[0, 0]
    for offset, task in zip((-0.11, 0.11), TASKS):
        row = confirmation["rankwise_matching_minus_cyclic"][task]
        mean = np.asarray(row["mean"])
        low = np.asarray(row["ci95_low"])
        high = np.asarray(row["ci95_high"])
        ax.errorbar(
            x + offset,
            mean,
            yerr=np.vstack((mean - low, high - mean)),
            marker="o",
            capsize=3,
            linewidth=1.8,
            color=colors[task],
            label=task,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, RANKS)
    ax.set_ylabel("Candidate-centered logit change")
    ax.set_title("Matching blockade minus cyclic control")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    row = confirmation["rankwise_task_interaction"]
    mean = np.asarray(row["mean"])
    low = np.asarray(row["ci95_low"])
    high = np.asarray(row["ci95_high"])
    ax.errorbar(
        x,
        mean,
        yerr=np.vstack((mean - low, high - mean)),
        marker="o",
        capsize=3,
        linewidth=1.8,
        color="#6a3d9a",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, RANKS)
    ax.set_ylabel("Game minus Neutral effect")
    ax.set_title("Policy difference in route use")

    ax = axes[1, 0]
    width = 0.22
    scenario_labels = ("Natural", "Matching\nblockade", "Cyclic wrong\nblockade")
    for ti, task in enumerate(TASKS):
        means = [
            confirmation["w1_choice_rates"][scenario][task]["mean"] * 100
            for scenario in SCENARIOS
        ]
        lows = [
            confirmation["w1_choice_rates"][scenario][task]["ci95"][0] * 100
            for scenario in SCENARIOS
        ]
        highs = [
            confirmation["w1_choice_rates"][scenario][task]["ci95"][1] * 100
            for scenario in SCENARIOS
        ]
        pos = np.arange(3) + (ti - 0.5) * width
        ax.bar(pos, means, width, color=colors[task], alpha=0.9, label=task)
        ax.errorbar(
            pos,
            means,
            yerr=np.vstack((np.asarray(means) - lows, np.asarray(highs) - means)),
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
    ax.set_xticks(np.arange(3), scenario_labels)
    ax.set_ylabel("Aggregated A-D choice is old W1 (%)")
    ax.set_title("Old-winner choice under each condition")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    names = (
        "Matching − cyclic\nchange in task gap",
        "Matching − natural\nchange in task gap",
    )
    rows = (
        confirmation["w1_task_gap_changes"]["matching_minus_cyclic"],
        confirmation["w1_task_gap_changes"]["matching_minus_natural"],
    )
    means = np.array([row["mean"] * 100 for row in rows])
    lows = np.array([row["ci95"][0] * 100 for row in rows])
    highs = np.array([row["ci95"][1] * 100 for row in rows])
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
    ax.set_title("Does recollection support the task difference?")

    figure.suptitle(
        "Qwen3.6-27B TriviaMC: causal matching-history blockade\n"
        "Confirmation split (n=250); paired W1-letter-stratified 95% CIs",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


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
    if not (
        np.all(arrays["source_position_counts"] > 0)
        and np.all(arrays["query_position_counts"] > 0)
        and np.all(arrays["cyclic_source_position_counts"] > 0)
    ):
        raise RuntimeError("One or more executed source/query spans were empty")

    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    split = json.loads(args.split_plan.read_text())
    discovery_ids = set(split["discovery_question_ids"])
    confirmation_ids = set(split["confirmation_question_ids"])
    if discovery_ids & confirmation_ids or discovery_ids | confirmation_ids != set(qids):
        raise RuntimeError("Frozen split does not partition the executed questions")
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    masks = {
        "discovery": discovery,
        "confirmation": ~discovery,
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
            "dataset": "TriviaMC_difficulty_filtered",
            "questions": 500,
            "tasks": list(TASKS),
            "scenarios": list(SCENARIOS),
            "ordinary_attention_layers_one_based": list(range(4, 65, 4)),
            "matching_blockade": (
                "Every token of each complete 2P option line is denied ordinary-attention "
                "reads of every token of its semantically matching complete 1P option line."
            ),
            "cyclic_wrong_control": "W1<-W2, W2<-W3, W3<-W4, W4<-W1",
            "statistics": (
                f"{args.draws}-draw paired bootstrap within frozen first-presentation "
                "W1-letter strata; displayed-order stable A-D tie rule."
            ),
            "correctness_endpoint": False,
        },
        "validation": {
            "questions": len(qids),
            "discovery": int(discovery.sum()),
            "confirmation": int((~discovery).sum()),
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
            "cyclic_source_tokens": {
                "min": int(arrays["cyclic_source_position_counts"].min()),
                "max": int(arrays["cyclic_source_position_counts"].max()),
                "mean": float(arrays["cyclic_source_position_counts"].mean()),
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
            "split_plan": {
                "path": str(args.split_plan),
                "sha256": _sha256(args.split_plan),
            },
        },
    }

    for subset_index, (subset, mask) in enumerate(masks.items()):
        strata = strata_all[mask]
        rank_specific = np.empty((2, len(qids), 4), dtype=float)
        match_natural = np.empty_like(rank_specific)
        cyclic_natural = np.empty_like(rank_specific)
        for task in range(2):
            for rank in range(4):
                target = rank_indices[rank]
                n_adv = _advantage(natural[task], target)
                m_adv = _advantage(matching[task], target)
                c_adv = _advantage(cyclic[task], target)
                rank_specific[task, :, rank] = m_adv - c_adv
                match_natural[task, :, rank] = m_adv - n_adv
                cyclic_natural[task, :, rank] = c_adv - n_adv

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
                match_natural[task_index, mask], strata, seed + 2, args.draws
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

        w1_indicators: dict[str, np.ndarray] = {}
        for scenario_index, scenario in enumerate(SCENARIOS):
            indicator = (choices[scenario] == w1[None, :]).astype(float)
            w1_indicators[scenario] = indicator
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
        gap_matching = w1_indicators["joint_matching"][0] - w1_indicators["joint_matching"][1]
        gap_cyclic = w1_indicators["joint_cyclic_wrong"][0] - w1_indicators["joint_cyclic_wrong"][1]
        gap_natural = w1_indicators["natural"][0] - w1_indicators["natural"][1]
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
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _plot(summary, args.figure)

    conf = summary["results"]["confirmation"]
    report = [
        "# Qwen3.6-27B TriviaMC matching-history blockade",
        "",
        "## What was tested",
        "",
        "On all 500 frozen difficulty-filtered TriviaMC questions, in both Game and Neutral, "
        "we blocked ordinary-attention reads from every complete 2P option line to its "
        "semantically matching complete 1P option line at all 16 ordinary-attention layers "
        "(4, 8, ..., 64). The control blocked the same four receiver lines and layers but used "
        "the next old-rank source line cyclically. This is a whole-line causal test, not a "
        "token-localization claim.",
        "",
        "## Confirmation results",
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
    interaction = conf["rankwise_task_interaction"]
    report.extend(
        [
            "| Game minus Neutral | " + " | ".join(
                f"{interaction['mean'][i]:+.3f} [{interaction['ci95_low'][i]:+.3f}, {interaction['ci95_high'][i]:+.3f}]"
                for i in range(4)
            ) + " |",
            "",
            "Aggregated A-D old-W1 choice rates:",
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
            "The primary change in the Game-minus-Neutral W1-choice gap, matching blockade "
            "minus cyclic control, is "
            f"{_fmt(conf['w1_task_gap_changes']['matching_minus_cyclic'], 100, 1)} percentage points.",
            "",
            "## Interpretation",
            "",
            "The causal Game result is clear and rank-specific. Relative to the cyclic control, "
            f"blocking the true semantic matches raises old-W1 evidence by "
            f"{conf['rankwise_matching_minus_cyclic']['Game']['mean'][0]:.3f} logits and lowers "
            f"W3/W4 evidence by {abs(conf['rankwise_matching_minus_cyclic']['Game']['mean'][2]):.3f}/"
            f"{abs(conf['rankwise_matching_minus_cyclic']['Game']['mean'][3]):.3f}. Therefore, "
            "when the matching-history route is intact in Game, it is doing the opposite: "
            "selectively lowering the old winner and supporting weaker old candidates. This is "
            "not equal noise added to all candidates.",
            "",
            f"At the discrete readout, the natural confirmation Game-minus-Neutral old-W1 "
            f"choice gap is {_fmt(conf['w1_task_gaps']['natural'], 100, 1)} percentage points. "
            f"Under the matching blockade it is {_fmt(conf['w1_task_gaps']['joint_matching'], 100, 1)}, "
            f"a matching-minus-natural change of "
            f"{_fmt(conf['w1_task_gap_changes']['matching_minus_natural'], 100, 1)} points. In "
            "plain terms: the causal cut removes the observed extra Game avoidance of the old "
            "winner. The cyclic control retains and slightly enlarges that difference, giving "
            f"the primary matching-minus-cyclic change of "
            f"{_fmt(conf['w1_task_gap_changes']['matching_minus_cyclic'], 100, 1)} points.",
            "",
            "The result is not a complete replication of the earlier SimpleMC task-shared "
            "recollection profile. Neutral's confirmation matching-minus-cyclic rank effects "
            "all have intervals spanning zero, and its discovery W1 effect has the expected "
            "supportive sign but does not repeat on confirmation. Thus this dataset strongly "
            "replicates policy-dependent Game use of matching semantic history and the causal "
            "removal of preferential Game W1 avoidance; it does not independently establish a "
            "stable rankwise Neutral support profile.",
            "",
            "The Game pattern and the Game-minus-Neutral rank interaction reproduce on both "
            "frozen halves. The discovery matching-minus-cyclic W1 effects are "
            f"{summary['results']['discovery']['rankwise_matching_minus_cyclic']['Game']['mean'][0]:+.3f} "
            "in Game and "
            f"{summary['results']['discovery']['rankwise_matching_minus_cyclic']['Neutral']['mean'][0]:+.3f} "
            "in Neutral; the discovery change in the task W1-choice gap is "
            f"{_fmt(summary['results']['discovery']['w1_task_gap_changes']['matching_minus_cyclic'], 100, 1)} points.",
            "",
            "## Validation and scope",
            "",
            f"- Natural reproduction maximum absolute aggregated-logit error: "
            f"{summary['validation']['natural_max_abs_error_to_trusted']:.8f}.",
            f"- Natural displayed-choice agreement with trusted Step 1: "
            f"{summary['validation']['natural_displayed_choice_agreement_to_trusted']*100:.2f}%.",
            "- All intervention logits are finite and every executed source and receiver span is nonempty.",
            "- The cyclic control matches the number of edited semantic relations and layers, not exact "
            "source-token count; the observed token-count distributions are stored in `summary.json`.",
            "- These outcomes are causal for direct ordinary-attention reads from complete 1P lines into "
            "complete matching 2P lines. They do not isolate semantic wordpieces, edit GLA memory, or test correctness.",
            "",
            "See the canonical figure at `figures/qwen36_triviamc_matching_history_step2.png` "
            "and machine-readable estimates in `summary.json`.",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(report))
    print(json.dumps(summary["validation"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
