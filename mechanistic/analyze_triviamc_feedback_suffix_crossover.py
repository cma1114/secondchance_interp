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
SCENARIOS = ("natural", "identity_complete_suffix", "reciprocal_complete_suffix")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _stratified_draws(
    strata: np.ndarray, seed: int, draws: int
) -> list[np.ndarray]:
    groups = [
        np.flatnonzero(strata == label)
        for label in sorted(set(strata.astype(str).tolist()))
    ]
    if not groups or any(len(group) == 0 for group in groups):
        raise ValueError("Bootstrap strata are empty")
    rng = np.random.default_rng(seed)
    return [
        np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        for _ in range(draws)
    ]


def _bootstrap_mean(
    values: np.ndarray, strata: np.ndarray, seed: int, draws: int
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(strata) or not len(values):
        raise ValueError("Bootstrap inputs are empty or misaligned")
    scalar = values.ndim == 1
    if scalar:
        values = values[:, None]
    samples = np.asarray(
        [values[indices].mean(axis=0) for indices in _stratified_draws(strata, seed, draws)]
    )
    point = values.mean(axis=0)
    low, high = np.quantile(samples, (0.025, 0.975), axis=0)
    if scalar:
        return {
            "n": int(len(values)),
            "mean": float(point[0]),
            "ci95": [float(low[0]), float(high[0])],
        }
    return {
        "n": int(len(values)),
        "mean": point.tolist(),
        "ci95_low": low.tolist(),
        "ci95_high": high.tolist(),
    }


def _bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    strata: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if len(numerator) != len(strata) or len(denominator) != len(strata):
        raise ValueError("Ratio inputs are misaligned")
    if denominator.sum() <= 0:
        raise ValueError("Task-vector denominator is zero")
    samples = np.asarray(
        [
            numerator[indices].sum() / denominator[indices].sum()
            for indices in _stratified_draws(strata, seed, draws)
        ],
        dtype=np.float64,
    )
    return {
        "n": int(len(strata)),
        "ratio_of_sums": float(numerator.sum() / denominator.sum()),
        "ci95": [float(value) for value in np.quantile(samples, (0.025, 0.975))],
    }


def _fmt(row: dict[str, Any], scale: float = 1.0, digits: int = 3) -> str:
    low, high = row["ci95"]
    key = "ratio_of_sums" if "ratio_of_sums" in row else "mean"
    return (
        f"{row[key] * scale:+.{digits}f} "
        f"[{low * scale:+.{digits}f}, {high * scale:+.{digits}f}]"
    )


def _plot(summary: dict[str, Any], figure_path: Path) -> None:
    colors = {"Game": "#2878b5", "Neutral": "#e07a2f"}
    figure, axes = plt.subplots(2, 2, figsize=(12.3, 8.4))

    ax = axes[0, 0]
    width = 0.34
    split_labels = ("Discovery", "Confirmation")
    for task_index, task in enumerate(TASKS):
        rows = [
            summary["results"][split.lower()]["tasks"][task]["transfer_fraction"]
            for split in split_labels
        ]
        means = np.asarray([row["ratio_of_sums"] for row in rows])
        lows = np.asarray([row["ci95"][0] for row in rows])
        highs = np.asarray([row["ci95"][1] for row in rows])
        positions = np.arange(2) + (task_index - 0.5) * width
        ax.bar(positions, means, width=width, color=colors[task], label=task)
        ax.errorbar(
            positions,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            fmt="none",
            ecolor="black",
            capsize=4,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(np.arange(2), split_labels)
    ax.set_ylabel("Donor-policy transfer fraction")
    ax.set_title("Complete feedback-suffix crossover")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    confirmation = summary["results"]["confirmation"]
    scenario_names = ("Natural", "Identity", "Opposite-task\nsuffix")
    scenario_keys = SCENARIOS
    width = 0.34
    for task_index, task in enumerate(TASKS):
        rows = [
            confirmation["tasks"][task]["w1_choice_rates"][scenario]
            for scenario in scenario_keys
        ]
        means = np.asarray([row["mean"] * 100 for row in rows])
        lows = np.asarray([row["ci95"][0] * 100 for row in rows])
        highs = np.asarray([row["ci95"][1] * 100 for row in rows])
        positions = np.arange(3) + (task_index - 0.5) * width
        ax.bar(positions, means, width=width, color=colors[task], label=task)
        ax.errorbar(
            positions,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            fmt="none",
            ecolor="black",
            capsize=4,
        )
    ax.set_xticks(np.arange(3), scenario_names)
    ax.set_ylabel("Final choice is old W1 (%)")
    ax.set_title("Confirmation old-winner choice")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    x = np.arange(4)
    for offset, task in zip((-0.09, 0.09), TASKS):
        row = confirmation["tasks"][task]["rankwise_centered_logit_change"]
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
    ax.set_ylabel("Swapped minus natural centered logit")
    ax.set_title("Confirmation effect by old rank")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    changes = [
        confirmation["tasks"][task]["w1_choice_change"] for task in TASKS
    ]
    means = np.asarray([row["mean"] * 100 for row in changes])
    lows = np.asarray([row["ci95"][0] * 100 for row in changes])
    highs = np.asarray([row["ci95"][1] * 100 for row in changes])
    ax.bar(np.arange(2), means, color=[colors[task] for task in TASKS], width=0.55)
    ax.errorbar(
        np.arange(2),
        means,
        yerr=np.vstack((means - lows, highs - means)),
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(2), TASKS)
    ax.set_ylabel("Change in old-W1 choice (percentage points)")
    ax.set_title("Opposite-task suffix changes choice")

    figure.suptitle(
        f"{summary['design']['model_label']} {summary['design']['dataset_label']}: "
        "complete feedback-suffix policy crossover\n"
        "Paired W1-letter-stratified 95% bootstrap intervals",
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
    if arrays["scenario_ids"].astype(str).tolist() != list(SCENARIOS):
        raise RuntimeError("Unexpected scenario inventory")
    corrected = arrays["scenario_final_logits"].astype(np.float64)
    raw = arrays["scenario_final_logits_raw"].astype(np.float64)
    trusted = arrays["trusted_natural_logits"].astype(np.float64)
    same_batch = arrays["same_batch_natural_logits"].astype(np.float64)
    if corrected.shape != (2, 3, 500, 4) or not np.all(np.isfinite(corrected)):
        raise RuntimeError("Corrected causal logits are incomplete or non-finite")
    if raw.shape != (2, 3, 500, 4) or not np.all(np.isfinite(raw)):
        raise RuntimeError("Raw causal logits are incomplete or non-finite")

    raw_identity_error = float(np.max(np.abs(raw[:, 1] - raw[:, 0])))
    recorded_identity_error = float(
        np.max(arrays["identity_error_by_question"].astype(np.float64))
    )
    if raw_identity_error != 0.0 or recorded_identity_error != 0.0:
        raise RuntimeError("The real restoration identity control is not bit-exact")
    corrected_natural_error = float(np.max(np.abs(corrected[:, 0] - trusted)))
    corrected_identity_error = float(np.max(np.abs(corrected[:, 1] - trusted)))
    same_batch_natural_error = float(np.max(np.abs(raw[:, 0] - same_batch)))
    if corrected_natural_error != 0.0 or corrected_identity_error != 0.0:
        raise RuntimeError("Same-batch correction failed exact natural reproduction")

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
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    if not discovery.any() or discovery.all():
        raise RuntimeError("Frozen discovery and confirmation splits must both be nonempty")
    masks = {
        "discovery": discovery,
        "confirmation": ~discovery,
        "all": np.ones(500, dtype=bool),
    }

    baseline_payload = json.loads(args.baseline.read_text())
    if "results" in baseline_payload:
        baseline = baseline_payload["results"]
    else:
        baseline = baseline_payload["scenarios"]["baseline"]
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    old_rank_original = np.empty((500, 4), dtype=np.int64)
    old_rank_displayed = np.empty((500, 4), dtype=np.int64)
    strata = np.empty(500, dtype="<U1")
    for qi, qid in enumerate(qids):
        order = np.argsort(
            -np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=np.float64),
            kind="stable",
        )
        old_rank_original[qi] = order
        strata[qi] = LETTERS[int(order[0])]
        old_rank_displayed[qi] = [
            LETTERS.index(mappings[qid]["original_to_new"][LETTERS[int(index)]])
            for index in order
        ]

    centered = _center(corrected)
    natural = centered[:, 0]
    donor_minus_recipient = natural[::-1] - natural
    denominator = np.sum(donor_minus_recipient**2, axis=-1)
    swapped_delta = centered[:, 2] - natural
    numerator = np.sum(swapped_delta * donor_minus_recipient, axis=-1)
    choices = np.argmax(corrected, axis=-1)
    w1_displayed = old_rank_displayed[:, 0]

    rankwise_delta = np.empty((2, 500, 4), dtype=np.float64)
    for task_index in range(2):
        for qi in range(500):
            rankwise_delta[task_index, qi] = swapped_delta[
                task_index, qi, old_rank_displayed[qi]
            ]

    summary: dict[str, Any] = {
        "status": "complete",
        "question": (
            "Does the complete contextualized feedback suffix causally impose the "
            "opposite task's Game/Neutral final scoring policy on TriviaMC?"
        ),
        "design": {
            "model_label": args.model_label,
            "dataset_label": args.dataset_label,
            "dataset": args.dataset_label,
            "questions": 500,
            "tasks": list(TASKS),
            "scenarios": list(SCENARIOS),
            "source_tokens": (
                "Every tokenizer token overlapping the complete contiguous text from "
                "incorrect/lost through the final period in 'Choose the answer again.'"
            ),
            "source_token_count": int(arrays["source_positions"].shape[-1]),
            "source_state": (
                f"Downstream ordinary-attention K/V at all {args.layer_count} layers; each source "
                "token's local residual output is recipient-natural."
                if args.architecture in {"seed", "gemma"}
                else "Downstream ordinary-attention K/V and recurrent GLA k/v/g/beta "
                "writes; each source token's local output is recipient-natural."
            ),
            "ordinary_attention_layers_one_based": (
                list(range(1, args.layer_count + 1))
                if args.architecture in {"seed", "gemma"}
                else list(range(4, 65, 4))
            ),
            "gla_layers_one_based": (
                [] if args.architecture in {"seed", "gemma"}
                else [value for value in range(1, 65) if value % 4 != 0]
            ),
            "statistics": (
                f"{args.draws}-draw paired bootstrap within frozen first-presentation "
                "W1-letter strata; displayed-order stable A-D tie rule."
            ),
            "correctness_endpoint": False,
        },
        "validation": {
            "all_complete": True,
            "all_finite": True,
            "discovery_questions": int(discovery.sum()),
            "confirmation_questions": int((~discovery).sum()),
            "raw_identity_max_abs_error": raw_identity_error,
            "recorded_identity_max_abs_error": recorded_identity_error,
            "corrected_natural_max_abs_error_to_trusted": corrected_natural_error,
            "corrected_identity_max_abs_error_to_trusted": corrected_identity_error,
            "same_batch_natural_storage_max_abs_error": same_batch_natural_error,
            "source_positions_all_valid": bool(np.all(arrays["source_positions"] >= 0)),
            "source_positions_all_contiguous": bool(
                np.all(np.diff(arrays["source_positions"], axis=-1) == 1)
            ),
        },
        "definitions": {
            "transfer_fraction": (
                "Ratio of sums: the dot product of the swap-induced centered A-D "
                "logit change with the same-question natural donor-minus-recipient "
                "task vector, divided by that task vector's squared length."
            ),
            "w1": "Highest aggregated A-D first-presentation candidate, mapped to its 2P letter.",
            "rankwise_change": "Reciprocal complete-suffix crossover minus natural, candidate-centered.",
        },
        "results": {},
        "provenance": {
            "results": {"path": str(args.results), "sha256": _sha256(args.results)},
            "baseline": {"path": str(args.baseline), "sha256": _sha256(args.baseline)},
            "remapping_plan": {
                "path": str(args.remapping_plan),
                "sha256": _sha256(args.remapping_plan),
            },
            "split_plan": (
                {"path": str(args.split_plan), "sha256": _sha256(args.split_plan)}
                if args.split_plan is not None else None
            ),
            "discovery_plan": (
                {"path": str(args.discovery_plan), "sha256": _sha256(args.discovery_plan)}
                if args.discovery_plan is not None else None
            ),
            "confirmation_plan": (
                {"path": str(args.confirmation_plan), "sha256": _sha256(args.confirmation_plan)}
                if args.confirmation_plan is not None else None
            ),
        },
    }

    gate_checks: dict[str, Any] = {}
    gate_passed = True
    for subset_index, (subset, mask) in enumerate(masks.items()):
        rows: dict[str, Any] = {"n": int(mask.sum()), "tasks": {}}
        subset_strata = strata[mask]
        for task_index, task in enumerate(TASKS):
            base_seed = args.seed + subset_index * 10000 + task_index * 1000
            transfer = _bootstrap_ratio(
                numerator[task_index, mask],
                denominator[task_index, mask],
                subset_strata,
                base_seed + 1,
                args.draws,
            )
            w1_rates = {
                scenario: _bootstrap_mean(
                    (choices[task_index, scenario_index, mask] == w1_displayed[mask]).astype(float),
                    subset_strata,
                    base_seed + 10 + scenario_index,
                    args.draws,
                )
                for scenario_index, scenario in enumerate(SCENARIOS)
            }
            w1_change = _bootstrap_mean(
                (
                    (choices[task_index, 2, mask] == w1_displayed[mask]).astype(float)
                    - (choices[task_index, 0, mask] == w1_displayed[mask]).astype(float)
                ),
                subset_strata,
                base_seed + 20,
                args.draws,
            )
            rank_change = _bootstrap_mean(
                rankwise_delta[task_index, mask],
                subset_strata,
                base_seed + 30,
                args.draws,
            )
            task_rows = {
                "transfer_fraction": transfer,
                "w1_choice_rates": w1_rates,
                "w1_choice_change": w1_change,
                "rankwise_centered_logit_change": rank_change,
            }
            rows["tasks"][task] = task_rows
            if subset in ("discovery", "confirmation"):
                passed = transfer["ratio_of_sums"] > 0 and transfer["ci95"][0] > 0
                gate_checks[f"{subset}_{task}"] = {
                    **transfer,
                    "passed": bool(passed),
                }
                gate_passed = gate_passed and passed
        summary["results"][subset] = rows

    summary["gate"] = {
        "passed": bool(gate_passed),
        "criterion": (
            "Reciprocal complete-suffix transfer is positive with a 95% CI lower "
            "bound above zero in Game and Neutral on both frozen splits."
        ),
        "checks": gate_checks,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _plot(summary, args.figure)

    confirmation = summary["results"]["confirmation"]
    discovery_rows = summary["results"]["discovery"]
    report = [
        f"# {args.model_label} {args.dataset_label} complete-feedback-suffix policy crossover",
        "",
        "## What was tested",
        "",
        f"For every frozen {args.dataset_label} question, the Game and Neutral prompts differ only at "
        "`incorrect` versus `lost`. We crossed the downstream state emitted by every tokenizer "
        "token overlapping the complete feedback suffix (`incorrect/lost. Choose the answer "
        "again.`) from Game into Neutral and from Neutral into Game. The edit covers ordinary-"
        "attention K/V at every applicable layer"
        + ("." if args.architecture in {"seed", "gemma"} else " and every recurrent GLA write.")
        + " It preserves each source token's own local output, "
        "so this is a causal test of information sent from the feedback suffix to later tokens, "
        "not a residual replacement at the source tokens.",
        "",
        "The natural, real same-task identity, and reciprocal crossover scenarios were the only "
        "conditions. No individual-token, layer-localization, matching-history, or Step-4 "
        "factorial condition was included.",
        "",
        "## Primary result",
        "",
        f"The prespecified replication gate **{'passed' if gate_passed else 'did not pass'}**.",
        "",
        "| Frozen split | Recipient | Donor state | Task-vector transfer (95% CI) |",
        "|---|---|---|---:|",
    ]
    for subset, display in (("discovery", "Discovery"), ("confirmation", "Confirmation")):
        for task in TASKS:
            donor = "Neutral" if task == "Game" else "Game"
            row = summary["results"][subset]["tasks"][task]["transfer_fraction"]
            report.append(f"| {display} | {task} | {donor} | {_fmt(row)} |")
    report.extend(
        [
            "",
            "A value of 1 would mean that, along the measured natural Game-versus-Neutral "
            "A-D scoring difference, the recipient moved all the way to its paired donor. A "
            "value of 0 would mean no movement in that donor-policy direction.",
            "",
            "## Confirmation secondary readouts",
            "",
            "| Recipient | Natural old-W1 choice | Opposite-task suffix old-W1 choice | Change |",
            "|---|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        task_rows = confirmation["tasks"][task]
        report.append(
            f"| {task} | {_fmt(task_rows['w1_choice_rates']['natural'], 100, 1)}% | "
            f"{_fmt(task_rows['w1_choice_rates']['reciprocal_complete_suffix'], 100, 1)}% | "
            f"{_fmt(task_rows['w1_choice_change'], 100, 1)} pp |"
        )
    report.extend(
        [
            "",
            "Opposite-task suffix minus natural candidate-centered logit changes by old rank:",
            "",
            "| Recipient | W1 | W2 | W3 | W4 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        row = confirmation["tasks"][task]["rankwise_centered_logit_change"]
        report.append(
            "| " + task + " | " + " | ".join(
                f"{row['mean'][index]:+.3f} "
                f"[{row['ci95_low'][index]:+.3f}, {row['ci95_high'][index]:+.3f}]"
                for index in range(4)
            ) + " |"
        )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "This experiment asks whether the contextualized feedback suffix is sufficient to "
            f"send the task policy forward through {args.model_label}'s ordinary-attention architecture. "
            "Near-unit reciprocal transfer on both frozen halves means that replacing only what "
            "those seven source tokens make available through downstream K/V moves final answer "
            "scoring essentially all the way to the opposite task's natural pattern. It does not "
            "identify which individual suffix token or layer carries the policy, or which later "
            "token first uses it.",
            "",
            "Discovery transfer estimates were "
            f"{discovery_rows['tasks']['Game']['transfer_fraction']['ratio_of_sums']:.3f} into Game "
            "and "
            f"{discovery_rows['tasks']['Neutral']['transfer_fraction']['ratio_of_sums']:.3f} into Neutral. "
            "Confirmation estimates were "
            f"{confirmation['tasks']['Game']['transfer_fraction']['ratio_of_sums']:.3f} and "
            f"{confirmation['tasks']['Neutral']['transfer_fraction']['ratio_of_sums']:.3f}, respectively.",
            "",
            "## Validation and scope",
            "",
            f"- Raw same-task identity maximum absolute error: {raw_identity_error:.8f}.",
            f"- Corrected natural maximum absolute error to trusted Step 1: {corrected_natural_error:.8f}.",
            f"- Corrected identity maximum absolute error to trusted Step 1: {corrected_identity_error:.8f}.",
            "- All 500 questions completed; every saved logit was finite; all audited source spans "
            f"contained {int(arrays['source_positions'].shape[-1])} contiguous token positions.",
            (
                f"- {args.model_label} has no GLA or recurrent-attention state; the edit covers "
                f"ordinary-attention K/V at every one of its {args.layer_count} decoder layers."
                if args.architecture in {"seed", "gemma"}
                else "- GLA recurrent writes are crossed, but the short local GLA convolution is not. "
                "The causal claim is therefore about the complete intercepted downstream output "
                "state, not about every possible architectural channel."
            ),
            "- Correctness is not an endpoint.",
            "",
            f"See `{args.figure}` and `summary.json`.",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(report))
    print(json.dumps({"validation": summary["validation"], "gate": summary["gate"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path)
    parser.add_argument("--discovery-plan", type=Path)
    parser.add_argument("--confirmation-plan", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--model-label", default="Qwen3.6-27B")
    parser.add_argument("--dataset-label", default="TriviaMC")
    parser.add_argument("--architecture", choices=("qwen", "seed", "gemma"), default="qwen")
    parser.add_argument("--layer-count", type=int, default=64)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
