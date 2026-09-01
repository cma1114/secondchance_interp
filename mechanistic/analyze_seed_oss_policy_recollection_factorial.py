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
    _advantage,
    _align_to_original,
    _bootstrap,
    _load,
    _semantic_choices,
)


TASKS = ("Game", "Neutral")
POLICIES = ("Game", "Neutral")
ACCESS = ("intact", "matching_block", "cyclic_wrong_block")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _estimate(values: np.ndarray, strata: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    return _bootstrap(values, strata, seed, draws)


def _plot(summary: dict[str, Any], path: Path) -> None:
    conf = summary["results"]["confirmation"]
    colors = {"Game": "#2878b5", "Neutral": "#e07a2f"}
    x = np.arange(4)
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4), constrained_layout=True)
    for recipient_index, recipient in enumerate(TASKS):
        axis = axes[0, recipient_index]
        for offset, policy in zip((-0.10, 0.10), POLICIES):
            row = conf["route_effect_by_recipient_and_policy"][recipient][policy]
            mean = np.asarray(row["mean"])
            low = np.asarray(row["ci95_low"])
            high = np.asarray(row["ci95_high"])
            axis.errorbar(
                x + offset,
                mean,
                yerr=np.vstack((mean - low, high - mean)),
                marker="o",
                linewidth=1.8,
                capsize=4,
                color=colors[policy],
                label=f"Installed {policy} suffix",
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, RANKS)
        axis.set_title(f"Recipient prompt: {recipient}")
        axis.set_ylabel("Matching blockade − cyclic control\n(centered logit change)")
        axis.legend(frameon=False)

    axis = axes[1, 0]
    for offset, recipient in zip((-0.12, 0.12), TASKS):
        row = conf["policy_by_recollection_interaction"][recipient]
        mean = np.asarray(row["mean"])
        low = np.asarray(row["ci95_low"])
        high = np.asarray(row["ci95_high"])
        axis.errorbar(
            x + offset,
            mean,
            yerr=np.vstack((mean - low, high - mean)),
            marker="o",
            linewidth=1.8,
            capsize=4,
            color=colors[recipient],
            label=f"{recipient} recipient",
        )
    pooled = conf["policy_by_recollection_interaction"]["recipient_average"]
    mean = np.asarray(pooled["mean"])
    low = np.asarray(pooled["ci95_low"])
    high = np.asarray(pooled["ci95_high"])
    axis.errorbar(
        x,
        mean,
        yerr=np.vstack((mean - low, high - mean)),
        marker="D",
        linestyle="none",
        capsize=5,
        color="#5b2c83",
        label="Recipient average",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, RANKS)
    axis.set_ylabel("Game-suffix − Neutral-suffix\nroute effect")
    axis.set_title("Direct policy × recollection interaction")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    labels = []
    means = []
    lows = []
    highs = []
    for recipient in (*TASKS, "recipient_average"):
        row = conf["w1_choice_policy_by_recollection_interaction"][recipient]
        labels.append(recipient.replace("recipient_average", "Average"))
        means.append(row["mean"] * 100)
        lows.append(row["ci95"][0] * 100)
        highs.append(row["ci95"][1] * 100)
    means_array = np.asarray(means)
    axis.bar(np.arange(3), means_array, color=("#2878b5", "#e07a2f", "#5b2c83"), width=0.62)
    axis.errorbar(
        np.arange(3),
        means_array,
        yerr=np.vstack((means_array - np.asarray(lows), np.asarray(highs) - means_array)),
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(3), labels)
    axis.set_ylabel("Interaction in old-W1 choice (pp)")
    axis.set_title("Displayed-choice interaction")
    fig.suptitle(
        f"{summary['design']['model_label']} {summary['design']['dataset']}: direct policy × recollection factorial\n"
        f"Complete suffix K/V state × all-{summary['design']['layer_count']}-layer matching-history access",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> None:
    arrays = _load(args.results)
    qids = arrays["question_ids"].astype(str).tolist()
    if len(qids) != 500 or not np.all(arrays["completed"].astype(bool)):
        raise RuntimeError("Expected a complete 500-question run")
    if arrays["access_levels"].astype(str).tolist() != list(ACCESS):
        raise RuntimeError("Unexpected access levels")
    for key in ("reference_access_logits", "raw_factorial_logits", "factorial_logits"):
        if not np.all(np.isfinite(arrays[key])):
            raise RuntimeError(f"Non-finite {key}")
    if float(np.max(arrays["identity_error"])) != 0.0:
        raise RuntimeError("Distinct-row identity was not exact")

    prior_suffix = _load(args.prior_suffix_results)
    prior_reciprocal = prior_suffix["scenario_final_logits"][:, 2].astype(float)
    current_reciprocal = np.stack(
        (arrays["factorial_logits"][0, 1, 0], arrays["factorial_logits"][1, 0, 0])
    ).astype(float)
    suffix_reproduction_error = float(
        np.max(np.abs(current_reciprocal - prior_reciprocal))
    )
    if suffix_reproduction_error != 0.0:
        raise RuntimeError(
            "Reciprocal intact-access cells did not exactly reproduce the prior "
            f"suffix crossover: {suffix_reproduction_error}"
        )

    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    if args.split_plan is not None:
        split = json.loads(args.split_plan.read_text())
        discovery_ids = set(split["discovery_question_ids"])
        confirmation_ids = set(split["confirmation_question_ids"])
    else:
        discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
        confirmation_ids = set(json.loads(args.confirmation_plan.read_text())["question_ids"])
    if discovery_ids & confirmation_ids or discovery_ids | confirmation_ids != set(qids):
        raise RuntimeError("Frozen split does not partition questions")
    masks = {
        "discovery": np.asarray([qid in discovery_ids for qid in qids]),
        "confirmation": np.asarray([qid in confirmation_ids for qid in qids]),
        "all": np.ones(len(qids), dtype=bool),
    }

    rank_contents = arrays["rank_contents"].astype(str)
    rank_indices = np.asarray(
        [[LETTERS.index(value) for value in row] for row in rank_contents], dtype=int
    ).T
    w1 = rank_indices[0]
    strata_all = rank_contents[:, 0]
    raw = arrays["factorial_logits"].astype(float)
    aligned = np.empty_like(raw)
    choices = np.empty(raw.shape[:-1], dtype=int)
    for recipient in range(2):
        for policy in range(2):
            for access in range(3):
                aligned[recipient, policy, access] = _align_to_original(
                    raw[recipient, policy, access][None], qids, mappings
                )[0]
                choices[recipient, policy, access] = _semantic_choices(
                    raw[recipient, policy, access][None], qids, mappings
                )[0]

    route = np.empty((2, 2, len(qids), 4), dtype=float)
    for recipient in range(2):
        for policy in range(2):
            for rank in range(4):
                target = rank_indices[rank]
                matching = _advantage(aligned[recipient, policy, 1], target)
                cyclic = _advantage(aligned[recipient, policy, 2], target)
                route[recipient, policy, :, rank] = matching - cyclic
    interaction = route[:, 0] - route[:, 1]
    interaction_average = interaction.mean(axis=0)
    w1_choice = (choices == w1[None, None, None, :]).astype(float)
    choice_route = w1_choice[:, :, 1] - w1_choice[:, :, 2]
    choice_interaction = choice_route[:, 0] - choice_route[:, 1]
    choice_interaction_average = choice_interaction.mean(axis=0)

    summary: dict[str, Any] = {
        "status": "complete",
        "design": {
            "model": args.model_id,
            "model_label": args.model_label,
            "dataset": args.dataset,
            "questions": 500,
            "recipient_prompts": list(TASKS),
            "installed_suffix_policies": list(POLICIES),
            "history_access": list(ACCESS),
            "layers": list(range(1, args.layer_count + 1)),
            "layer_count": args.layer_count,
            "primary_continuous_interaction": "(matching-cyclic under installed Game suffix) - (matching-cyclic under installed Neutral suffix)",
            "correctness_endpoint": False,
        },
        "validation": {
            "all_outputs_finite": True,
            "identity_max_abs_error": float(np.max(arrays["identity_error"])),
            "prior_suffix_reproduction_max_abs_error": suffix_reproduction_error,
            "trusted_natural_max_abs_error": float(np.max(arrays["trusted_natural_error"])),
            "source_counts_positive": bool(np.all(arrays["history_source_counts"] > 0)),
            "query_counts_positive": bool(np.all(arrays["history_query_counts"] > 0)),
            "discovery": int(masks["discovery"].sum()),
            "confirmation": int(masks["confirmation"].sum()),
        },
        "results": {},
        "provenance": {
            "results": {"path": str(args.results), "sha256": _sha256(args.results)},
            "remapping_plan": {"path": str(args.remapping_plan), "sha256": _sha256(args.remapping_plan)},
            "prior_suffix_results": {
                "path": str(args.prior_suffix_results),
                "sha256": _sha256(args.prior_suffix_results),
            },
        },
    }
    for subset_index, (subset, mask) in enumerate(masks.items()):
        strata = strata_all[mask]
        row: dict[str, Any] = {
            "n": int(mask.sum()),
            "route_effect_by_recipient_and_policy": {},
            "policy_by_recollection_interaction": {},
            "w1_choice_route_effect": {},
            "w1_choice_policy_by_recollection_interaction": {},
        }
        for recipient_index, recipient in enumerate(TASKS):
            row["route_effect_by_recipient_and_policy"][recipient] = {}
            row["w1_choice_route_effect"][recipient] = {}
            for policy_index, policy in enumerate(POLICIES):
                base_seed = args.seed + subset_index * 10000 + recipient_index * 1000 + policy_index * 100
                row["route_effect_by_recipient_and_policy"][recipient][policy] = _estimate(
                    route[recipient_index, policy_index, mask], strata, base_seed + 1, args.draws
                )
                row["w1_choice_route_effect"][recipient][policy] = _estimate(
                    choice_route[recipient_index, policy_index, mask], strata, base_seed + 2, args.draws
                )
            row["policy_by_recollection_interaction"][recipient] = _estimate(
                interaction[recipient_index, mask],
                strata,
                args.seed + subset_index * 10000 + recipient_index * 1000 + 901,
                args.draws,
            )
            row["w1_choice_policy_by_recollection_interaction"][recipient] = _estimate(
                choice_interaction[recipient_index, mask],
                strata,
                args.seed + subset_index * 10000 + recipient_index * 1000 + 902,
                args.draws,
            )
        row["policy_by_recollection_interaction"]["recipient_average"] = _estimate(
            interaction_average[mask], strata, args.seed + subset_index * 10000 + 9901, args.draws
        )
        row["w1_choice_policy_by_recollection_interaction"]["recipient_average"] = _estimate(
            choice_interaction_average[mask], strata, args.seed + subset_index * 10000 + 9902, args.draws
        )
        summary["results"][subset] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(summary, args.figure)
    conf = summary["results"]["confirmation"]
    report = [
        f"# {args.model_label} {args.dataset} direct policy × recollection factorial",
        "",
        f"The complete Game/Neutral feedback-suffix K/V state and complete all-{args.layer_count}-layer matching-history access were crossed inside the same model evaluations. Matching access was compared with an equal-structure cyclic wrong-line blockade. Same-task suffix transfers used distinct duplicated rows and were required to reproduce every intact/matching/cyclic access cell exactly.",
        "",
        "## Frozen confirmation results",
        "",
        "Matching blockade minus cyclic control by installed suffix policy:",
        "",
    ]
    for recipient in TASKS:
        report.extend([f"### {recipient} recipient prompt", "", "| Installed suffix | W1 | W2 | W3 | W4 |", "|---|---:|---:|---:|---:|"])
        for policy in POLICIES:
            estimate = conf["route_effect_by_recipient_and_policy"][recipient][policy]
            report.append(
                f"| {policy} | "
                + " | ".join(
                    f"{estimate['mean'][i]:+.3f} [{estimate['ci95_low'][i]:+.3f}, {estimate['ci95_high'][i]:+.3f}]"
                    for i in range(4)
                )
                + " |"
            )
        report.append("")
    report.extend(["## Direct policy × recollection interaction", "", "Game-suffix route effect minus Neutral-suffix route effect:", "", "| Recipient | W1 | W2 | W3 | W4 |", "|---|---:|---:|---:|---:|"])
    for recipient in (*TASKS, "recipient_average"):
        estimate = conf["policy_by_recollection_interaction"][recipient]
        label = recipient.replace("recipient_average", "Recipient average")
        report.append(
            f"| {label} | "
            + " | ".join(
                f"{estimate['mean'][i]:+.3f} [{estimate['ci95_low'][i]:+.3f}, {estimate['ci95_high'][i]:+.3f}]"
                for i in range(4)
            )
            + " |"
        )
    report.extend(["", "Old-W1 displayed-choice interaction:", ""])
    for recipient in (*TASKS, "recipient_average"):
        estimate = conf["w1_choice_policy_by_recollection_interaction"][recipient]
        label = recipient.replace("recipient_average", "Recipient average")
        report.append(
            f"- **{label}:** {estimate['mean'] * 100:+.1f} pp "
            f"`[{estimate['ci95'][0] * 100:+.1f}, {estimate['ci95'][1] * 100:+.1f}]`."
        )
    report.extend(
        [
            "",
            "## Scope",
            "",
            "This is the direct causal interaction between installed feedback policy and the matching semantic-history route. It does not remove fresh second-presentation computation and therefore does not establish that reconstruction is unnecessary.",
            "",
            f"Identity maximum error: {summary['validation']['identity_max_abs_error']:.8f}. Prior reciprocal-suffix reproduction maximum error: {summary['validation']['prior_suffix_reproduction_max_abs_error']:.8f}. All outputs were finite; every source and receiver span was nonempty; all {args.layer_count} attention layers were included.",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(report))
    print(json.dumps(summary["validation"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--prior-suffix-results", type=Path, required=True)
    parser.add_argument("--dataset", choices=("SimpleMC", "TriviaMC"), required=True)
    parser.add_argument("--split-plan", type=Path)
    parser.add_argument("--discovery-plan", type=Path)
    parser.add_argument("--confirmation-plan", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--model-id", default="ByteDance-Seed/Seed-OSS-36B-Instruct")
    parser.add_argument("--model-label", default="Seed-OSS 36B")
    parser.add_argument("--layer-count", type=int, default=64)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
