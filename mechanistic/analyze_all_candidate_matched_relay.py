from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .semantic_mapping import displayed_argmax_to_semantic_indices


LETTERS = "ABCD"
RANKS = ("W1", "W2", "W3", "W4")
DISPLAY_RANKS = ("R1 (winner)", "R2", "R3", "R4")
CONDITIONS = ("Game", "Neutral")
CAUSAL_ORDINARY_BLOCKS = tuple(range(4, 49, 4))


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _align(values: np.ndarray, qids: list[str], mappings: dict[str, dict]) -> np.ndarray:
    out = np.empty_like(values)
    for qi, qid in enumerate(qids):
        for original_index, original in enumerate(LETTERS):
            new_letter = mappings[qid]["original_to_new"][original]
            out[..., qi, original_index] = values[..., qi, LETTERS.index(new_letter)]
    return out


def _candidate_advantage(logits: np.ndarray, indices: np.ndarray) -> np.ndarray:
    rows = np.arange(len(indices))
    target = logits[rows, indices]
    return target - (logits.sum(-1) - target) / 3.0


def _candidate_choice(choices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return (choices == indices).astype(float)


def _interval(values: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"n": 0, "mean": float("nan"), "ci": [float("nan"), float("nan")]}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(samples, (0.025, 0.975)).tolist(),
    }


def _fmt(row: dict[str, Any], scale: float = 1.0) -> str:
    return f"{row['mean']*scale:+.3f} [{row['ci'][0]*scale:+.3f}, {row['ci'][1]*scale:+.3f}]"


def _fit_question_centered(
    y: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    y = y[:, mask].T
    score = score[:, mask].T
    n = y.shape[0]
    w1 = np.zeros((n, 4), dtype=float)
    w1[:, 0] = 1.0

    def fit(indices: np.ndarray) -> np.ndarray:
        yy = y[indices]
        xx = score[indices]
        ww = w1[indices]
        yy = yy - yy.mean(1, keepdims=True)
        xx = xx - xx.mean(1, keepdims=True)
        ww = ww - ww.mean(1, keepdims=True)
        design = np.column_stack([xx.ravel(), ww.ravel()])
        return np.linalg.lstsq(design, yy.ravel(), rcond=None)[0]

    point = fit(np.arange(n))
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, 2), dtype=float)
    for draw in range(draws):
        boot[draw] = fit(rng.integers(0, n, size=n))
    return {
        "n_questions": int(n),
        "score_coefficient": {
            "mean": float(point[0]),
            "ci": np.quantile(boot[:, 0], (0.025, 0.975)).tolist(),
        },
        "W1_discontinuity": {
            "mean": float(point[1]),
            "ci": np.quantile(boot[:, 1], (0.025, 0.975)).tolist(),
        },
    }


def analyze(args: argparse.Namespace) -> None:
    arrays = _load(args.results)
    if not np.all(arrays["completed"].astype(bool)):
        raise RuntimeError("All-candidate relay checkpoint is incomplete")
    qids = arrays["question_ids"].astype(str).tolist()
    if len(qids) != 500:
        raise RuntimeError(f"Expected 500 questions, got {len(qids)}")
    for key in ("natural_logits", "matched_logits", "control_logits", "joint_logits"):
        if not np.all(np.isfinite(arrays[key])):
            raise RuntimeError(f"Non-finite values in {key}")

    observational = (
        _load(args.observational_results)
        if args.observational_results is not None
        else arrays
    )
    if not np.array_equal(observational["question_ids"], arrays["question_ids"]):
        raise RuntimeError("Observational companion question order does not match causal run")
    if not np.array_equal(observational["rank_contents"], arrays["rank_contents"]):
        raise RuntimeError("Observational companion rank definitions do not match causal run")
    if "ordinary_blocks_one_based" in observational:
        observational_blocks = observational["ordinary_blocks_one_based"].astype(int)
    else:
        observational_blocks = np.asarray(CAUSAL_ORDINARY_BLOCKS)
    if args.observational_results is not None and not np.array_equal(
        observational_blocks, np.arange(4, 65, 4)
    ):
        raise RuntimeError("Complete observational companion must contain blocks 4--64")

    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    baseline = json.loads(args.baseline.read_text())["results"]
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    if int(discovery.sum()) != 251:
        raise RuntimeError("Frozen discovery split is not 251 questions")

    rank_contents = arrays["rank_contents"].astype(str)
    rank_indices = np.asarray(
        [[LETTERS.index(value) for value in row] for row in rank_contents], dtype=int
    ).T
    w1 = rank_contents[:, 0]
    w2 = np.asarray([remapped[qid]["answer_original_content"] for qid in qids])
    conflict = w1 != w2
    w1i = rank_indices[0]
    w2i = np.asarray([LETTERS.index(value) for value in w2])
    baseline_logits = arrays["baseline_logits"].astype(float)
    score = np.stack(
        [baseline_logits[np.arange(len(qids)), rank_indices[rank]] for rank in range(4)]
    )
    score -= score.mean(0, keepdims=True)

    natural = _align(arrays["natural_logits"].astype(float), qids, mappings)
    trusted = _align(arrays["trusted_natural_logits"].astype(float), qids, mappings)
    matched = _align(arrays["matched_logits"].astype(float), qids, mappings)
    control = _align(arrays["control_logits"].astype(float), qids, mappings)
    joint = _align(arrays["joint_logits"].astype(float), qids, mappings)
    mapping_rows = [mappings[qid] for qid in qids]
    natural_choices = displayed_argmax_to_semantic_indices(
        arrays["natural_logits"], mapping_rows
    )
    matched_choices = displayed_argmax_to_semantic_indices(
        arrays["matched_logits"], mapping_rows
    )
    control_choices = displayed_argmax_to_semantic_indices(
        arrays["control_logits"], mapping_rows
    )
    joint_choices = displayed_argmax_to_semantic_indices(
        arrays["joint_logits"], mapping_rows
    )
    natural_error = float(np.max(np.abs(natural - trusted)))
    behavioral_agreement = float(
        (
            arrays["natural_logits"].argmax(-1)
            == arrays["trusted_natural_logits"].argmax(-1)
        ).mean()
    )

    effects: dict[str, np.ndarray] = {}
    for metric in ("advantage", "choice"):
        values = np.empty((2, 4, len(qids)), dtype=float)
        for ci in range(2):
            for rank in range(4):
                indices = rank_indices[rank]
                if metric == "advantage":
                    values[ci, rank] = (
                        _candidate_advantage(matched[ci, rank], indices)
                        - _candidate_advantage(control[ci, rank], indices)
                    )
                else:
                    values[ci, rank] = (
                        _candidate_choice(matched_choices[ci, rank], indices)
                        - _candidate_choice(control_choices[ci, rank], indices)
                    )
        effects[metric] = values

    masks = {
        "discovery_all": discovery,
        "confirmation_all": ~discovery,
        "discovery_conflict": discovery & conflict,
        "confirmation_conflict": (~discovery) & conflict,
        "discovery_no_conflict": discovery & (~conflict),
        "confirmation_no_conflict": (~discovery) & (~conflict),
    }
    summary: dict[str, Any] = {
        "definitions": {
            "matching_specific_effect": "matching-edge lesion minus cyclic nonmatching-edge lesion",
            "condition_interaction": "Game matching-specific effect minus Neutral matching-specific effect",
            "positive_interaction": "blocking the matching relay recovers more candidate evidence in Game than Neutral",
            "W1_discontinuity": "question-centered W1 indicator coefficient after simultaneous control for continuous first-pass candidate evidence",
        },
        "validation": {
            "questions": len(qids),
            "discovery": int(discovery.sum()),
            "confirmation": int((~discovery).sum()),
            "conflict": int(conflict.sum()),
            "natural_logits_max_abs_error": natural_error,
            "natural_behavioral_agreement": behavioral_agreement,
            "max_abs_causal_logit_change": float(
                max(
                    np.max(np.abs(matched - natural[:, None])),
                    np.max(np.abs(control - natural[:, None])),
                    np.max(np.abs(joint - natural)),
                )
            ),
        },
        "subsets": {},
        "direct_W1_matching_vs_natural": {},
        "regressions": {},
        "joint_mediation": {},
        "observational_layerwise": {},
    }

    csv_rows: list[list[Any]] = []
    for subset_index, (subset, mask) in enumerate(masks.items()):
        record: dict[str, Any] = {"n": int(mask.sum()), "ranks": {}}
        for rank, label in enumerate(RANKS):
            rank_record: dict[str, Any] = {}
            for metric_index, metric in enumerate(("advantage", "choice")):
                game = _interval(
                    effects[metric][0, rank, mask],
                    args.seed + subset_index * 1000 + rank * 100 + metric_index * 10,
                    args.draws,
                )
                neutral = _interval(
                    effects[metric][1, rank, mask],
                    args.seed + subset_index * 1000 + rank * 100 + metric_index * 10 + 1,
                    args.draws,
                )
                interaction = _interval(
                    (effects[metric][0, rank] - effects[metric][1, rank])[mask],
                    args.seed + subset_index * 1000 + rank * 100 + metric_index * 10 + 2,
                    args.draws,
                )
                rank_record[metric] = {
                    "Game": game,
                    "Neutral": neutral,
                    "Game_minus_Neutral": interaction,
                }
                for condition, row in (("Game", game), ("Neutral", neutral), ("Game_minus_Neutral", interaction)):
                    csv_rows.append([subset, label, metric, condition, row["n"], row["mean"], *row["ci"]])
            record["ranks"][label] = rank_record
        summary["subsets"][subset] = record

        direct_record: dict[str, Any] = {}
        for ci, condition in enumerate(CONDITIONS):
            direct_values = (
                _candidate_advantage(matched[ci, 0], w1i)
                - _candidate_advantage(natural[ci], w1i)
            )[mask]
            direct_record[condition] = _interval(
                direct_values,
                args.seed + 500000 + subset_index * 1000 + ci * 100,
                args.draws,
            )
        direct_record["Game_minus_Neutral"] = _interval(
            (
                _candidate_advantage(matched[0, 0], w1i)
                - _candidate_advantage(natural[0], w1i)
                - _candidate_advantage(matched[1, 0], w1i)
                + _candidate_advantage(natural[1], w1i)
            )[mask],
            args.seed + 500000 + subset_index * 1000 + 250,
            args.draws,
        )
        summary["direct_W1_matching_vs_natural"][subset] = direct_record

    interaction_advantage = effects["advantage"][0] - effects["advantage"][1]
    for subset_index, (subset, mask) in enumerate(masks.items()):
        summary["regressions"][subset] = _fit_question_centered(
            interaction_advantage,
            score,
            mask,
            args.seed + 100000 + subset_index * 10000,
            args.regression_draws,
        )

    rows = np.arange(len(qids))
    natural_choice = natural_choices
    joint_choice = joint_choices
    natural_margin = natural[:, rows, w1i] - natural[:, rows, w2i]
    joint_margin = joint[:, rows, w1i] - joint[:, rows, w2i]
    natural_w1_choice = (natural_choice == w1i[None]).astype(float)
    joint_w1_choice = (joint_choice == w1i[None]).astype(float)
    individual_w1_effect = np.empty((2, 4, len(qids)), dtype=float)
    for ci in range(2):
        for rank in range(4):
            individual_w1_effect[ci, rank] = (
                _candidate_advantage(matched[ci, rank], w1i)
                - _candidate_advantage(natural[ci], w1i)
            )
    for subset_index, subset in enumerate(("discovery_conflict", "confirmation_conflict")):
        mask = masks[subset]
        joint_choice_effect = joint_w1_choice[:, mask] - natural_w1_choice[:, mask]
        joint_margin_effect = joint_margin[:, mask] - natural_margin[:, mask]
        joint_advantage_effect = np.stack(
            [
                _candidate_advantage(joint[ci], w1i)
                - _candidate_advantage(natural[ci], w1i)
                for ci in range(2)
            ]
        )[:, mask]
        record: dict[str, Any] = {"n": int(mask.sum()), "conditions": {}}
        for ci, condition in enumerate(CONDITIONS):
            record["conditions"][condition] = {
                "W1_choice_effect": _interval(joint_choice_effect[ci], args.seed + 200000 + subset_index*1000 + ci*100, args.draws),
                "W1_minus_W2_margin_effect": _interval(joint_margin_effect[ci], args.seed + 200010 + subset_index*1000 + ci*100, args.draws),
                "W1_advantage_effect": _interval(joint_advantage_effect[ci], args.seed + 200020 + subset_index*1000 + ci*100, args.draws),
                "joint_minus_sum_individual_W1_advantage": _interval(
                    joint_advantage_effect[ci] - individual_w1_effect[ci][:, mask].sum(0),
                    args.seed + 200030 + subset_index*1000 + ci*100,
                    args.draws,
                ),
            }
        record["gap_reduction"] = {
            "W1_choice": _interval(joint_choice_effect[0] - joint_choice_effect[1], args.seed + 200040 + subset_index*1000, args.draws),
            "W1_minus_W2_margin": _interval(joint_margin_effect[0] - joint_margin_effect[1], args.seed + 200050 + subset_index*1000, args.draws),
        }
        summary["joint_mediation"][subset] = record

    blocks = observational_blocks
    for split, mask in (("discovery", discovery), ("confirmation", ~discovery)):
        split_record: dict[str, Any] = {"ordinary_blocks": blocks.tolist(), "metrics": {}}
        for metric in ("attention_mass", "context_norm", "projected_write_norm", "mean_gate"):
            values = observational[metric].astype(float)[:, :, mask]
            metric_record: dict[str, Any] = {}
            for rank, label in enumerate(RANKS):
                metric_record[label] = {
                    "Game": values[0, :, :, rank].mean(1).tolist(),
                    "Neutral": values[1, :, :, rank].mean(1).tolist(),
                    "Game_minus_Neutral": (
                        values[0, :, :, rank] - values[1, :, :, rank]
                    ).mean(1).tolist(),
                }
            split_record["metrics"][metric] = metric_record
        summary["observational_layerwise"][split] = split_record

    confirmation_w1 = summary["regressions"]["confirmation_all"]["W1_discontinuity"]
    discovery_w1 = summary["regressions"]["discovery_all"]["W1_discontinuity"]
    linear_w1_gate = bool(
        discovery_w1["mean"] > 0 and confirmation_w1["ci"][0] > 0
    )
    # The original linear-score gate is not an identification test for a
    # categorical winner representation: W1 is itself a nonlinear thresholded
    # function of relative scores. The frozen flexible-score audit supersedes
    # this gate and found no held-out categorical increment. Preserve the
    # historical linear result, but never revive the withdrawn binding claim.
    summary["interpretation_gate"] = {
        "explicit_winner_binding_supported": False,
        "status": "superseded_by_flexible_nonlinear_score_audit",
        "historical_linear_gate_passed": linear_w1_gate,
        "historical_rule": "Discovery W1 discontinuity positive and held-out 95% interval entirely above zero after controlling only linear first-pass evidence.",
        "current_interpretation": "The linear remainder does not identify a categorical winner code; flexible score-and-gap controls and near-tie tests do not establish one.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "effects.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subset", "rank", "metric", "condition", "n", "mean", "ci_low", "ci_high"])
        writer.writerows(csv_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    colors = {"Game": "#b13c2e", "Neutral": "#28658c"}
    offsets = {"Game": -0.08, "Neutral": 0.08}
    rows_summary = summary["subsets"]["confirmation_all"]["ranks"]
    for condition in CONDITIONS:
        means, lows, highs = [], [], []
        for rank in RANKS:
            row = rows_summary[rank]["advantage"][condition]
            means.append(row["mean"]); lows.append(row["ci"][0]); highs.append(row["ci"][1])
        x = np.arange(4) + offsets[condition]
        axes[0, 0].errorbar(x, means, yerr=[np.asarray(means)-lows, np.asarray(highs)-means], fmt="o-", color=colors[condition], capsize=3, label=condition)
    axes[0, 0].axhline(0, color="#777", lw=1)
    axes[0, 0].set_xticks(range(4), DISPLAY_RANKS)
    axes[0, 0].set_title("A  Held-out matching-specific causal effects", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Candidate centered-advantage effect (logits)")
    axes[0, 0].legend(frameon=False)

    for split, color, marker in (("discovery_all", "#8ab6d6", "o"), ("confirmation_all", "#125a8a", "s")):
        rows_split = summary["subsets"][split]["ranks"]
        means, lows, highs = [], [], []
        for rank in RANKS:
            row = rows_split[rank]["advantage"]["Game_minus_Neutral"]
            means.append(row["mean"]); lows.append(row["ci"][0]); highs.append(row["ci"][1])
        axes[0, 1].errorbar(range(4), means, yerr=[np.asarray(means)-lows, np.asarray(highs)-means], fmt=marker+"-", color=color, capsize=3, label=split.split("_")[0].capitalize())
    axes[0, 1].axhline(0, color="#777", lw=1)
    axes[0, 1].set_xticks(range(4), DISPLAY_RANKS)
    axes[0, 1].set_title("B  Policy interaction by first-pass rank", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("Game minus Neutral effect (logits)")
    axes[0, 1].legend(frameon=False)

    obs = summary["observational_layerwise"]["confirmation"]
    for rank, label in enumerate(RANKS):
        axes[1, 0].plot(blocks, obs["metrics"]["attention_mass"][label]["Game_minus_Neutral"], marker="o", label=DISPLAY_RANKS[rank])
        axes[1, 1].plot(blocks, obs["metrics"]["projected_write_norm"][label]["Game_minus_Neutral"], marker="o", label=DISPLAY_RANKS[rank])
    axes[1, 0].axhline(0, color="#777", lw=1)
    axes[1, 0].set_title("C  Held-out attention-mass difference", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Ordinary-attention block")
    axes[1, 0].set_ylabel("Game minus Neutral")
    axes[1, 1].axhline(0, color="#777", lw=1)
    axes[1, 1].set_title("D  Held-out projected-write norm difference", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Ordinary-attention block")
    axes[1, 1].set_ylabel("Game minus Neutral")
    axes[1, 1].legend(frameon=False, ncol=2)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("All-candidate semantic relay: generic, graded, or winner-specific?", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)

    conf = summary["subsets"]["confirmation_all"]["ranks"]
    joint_conf = summary["joint_mediation"]["confirmation_conflict"]
    direct_conf = summary["direct_W1_matching_vs_natural"]["confirmation_all"]
    lines = [
        "# All-candidate matched semantic relay factorial",
        "",
        "## Bottom line",
        "",
        (
            "The original linear-score W1 gate passed, but it is superseded by the flexible nonlinear score-and-gap audit and does not establish an explicit categorical winner tag."
        ),
        "",
        "## Direct W1 matching-edge effect versus natural",
        "",
        f"- Game: {_fmt(direct_conf['Game'])} logits.",
        f"- Neutral: {_fmt(direct_conf['Neutral'])} logits.",
        f"- Game minus Neutral: {_fmt(direct_conf['Game_minus_Neutral'])} logits.",
        "",
        "## Held-out matching-specific centered-advantage effects",
        "",
        "| Rank | Game | Neutral | Game minus Neutral |",
        "|---|---:|---:|---:|",
    ]
    for rank in RANKS:
        row = conf[rank]["advantage"]
        lines.append(f"| {rank} | {_fmt(row['Game'])} | {_fmt(row['Neutral'])} | {_fmt(row['Game_minus_Neutral'])} |")
    lines += [
        "",
        "## Graded versus categorical test",
        "",
        f"- Discovery W1 discontinuity: {_fmt(discovery_w1)}.",
        f"- Confirmation W1 discontinuity: {_fmt(confirmation_w1)}.",
        f"- Confirmation continuous-score coefficient: {_fmt(summary['regressions']['confirmation_all']['score_coefficient'])}.",
        "",
        "## Joint mediation on held-out conflict trials",
        "",
        f"- Game W1-choice effect: {_fmt(joint_conf['conditions']['Game']['W1_choice_effect'], 100)} percentage points.",
        f"- Neutral W1-choice effect: {_fmt(joint_conf['conditions']['Neutral']['W1_choice_effect'], 100)} percentage points.",
        f"- Reduction in the Game--Neutral W1-choice gap: {_fmt(joint_conf['gap_reduction']['W1_choice'], 100)} percentage points.",
        f"- Reduction in the W1--W2 margin gap: {_fmt(joint_conf['gap_reduction']['W1_minus_W2_margin'])} logits.",
        "",
        "## Validation",
        "",
        f"- Questions: {len(qids)} (251 discovery, 249 confirmation).",
        f"- Same-host versus trusted natural behavioral agreement: {behavioral_agreement*100:.1f}%.",
        f"- Maximum trusted natural-logit drift: {natural_error:.3f} logits.",
        f"- Maximum causal A--D logit change: {summary['validation']['max_abs_causal_logit_change']:.3f} logits.",
        "",
        f"Canonical figure: `{args.figure}`.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["interpretation_gate"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--observational-results", type=Path)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--regression-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=9192026)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
