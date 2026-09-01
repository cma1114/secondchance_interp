from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .semantic_mapping import displayed_argmax_to_semantic_indices


CONDITIONS = ("Game", "Neutral")
INTERVENTIONS = (
    "natural",
    "matching_only_blocked",
    "nonmatching_three_blocked",
    "all_four_blocked",
)
INTERVENTION_LABELS = {
    "natural": "Natural",
    "matching_only_blocked": "Matching line blocked",
    "nonmatching_three_blocked": "Three nonmatches blocked",
    "all_four_blocked": "All four 1P lines blocked",
}
RANKS = ("R1", "R2", "R3", "R4")
LETTERS = ("A", "B", "C", "D")
RANK_AXIS = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=float)


def _normal_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Interval requires at least two finite observations")
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def _rank_slope(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] != 4:
        raise ValueError("Rank slope requires four candidates")
    return np.sum(values * RANK_AXIS, axis=-1) / np.sum(RANK_AXIS**2)


def _load_results(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _aligned_logits(
    logits: np.ndarray,
    qids: list[str],
    rank_contents: np.ndarray,
    mappings: dict[str, dict[str, object]],
) -> np.ndarray:
    # Input: condition, intervention, question, current 2P A-D letter.
    # Output: condition, intervention, question, first-pass rank R1-R4.
    aligned = np.full(logits.shape, np.nan, dtype=float)
    for qi, qid in enumerate(qids):
        original_to_new = mappings[qid]["original_to_new"]
        if not isinstance(original_to_new, dict):
            raise RuntimeError("Malformed original_to_new mapping")
        for rank, original_letter in enumerate(rank_contents[qi].astype(str)):
            current_letter = str(original_to_new[original_letter])
            aligned[:, :, qi, rank] = logits[:, :, qi, LETTERS.index(current_letter)]
    if not np.isfinite(aligned).all():
        raise RuntimeError("Rank-aligned logits are non-finite")
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--original-baseline", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--prior-full-range", type=Path, required=True)
    parser.add_argument("--prior-attention", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    arrays = _load_results(args.results)
    prior = _load_results(args.prior_full_range)
    attention = _load_results(args.prior_attention)
    qids = arrays["question_ids"].astype(str).tolist()
    completed = arrays["completed"].astype(bool)
    logits = arrays["logits"].astype(float)
    rank_contents = arrays["rank_contents"].astype(str)
    if len(qids) != 500 or not completed.all():
        raise RuntimeError("Expected all 500 questions complete")
    if logits.shape != (2, 4, 500, 4) or not np.isfinite(logits).all():
        raise RuntimeError(f"Invalid logits array: {logits.shape}")
    if arrays["interventions"].astype(str).tolist() != list(INTERVENTIONS):
        raise RuntimeError("Intervention labels changed")
    if arrays["ordinary_layers_one_based"].astype(int).tolist() != list(
        range(4, 65, 4)
    ):
        raise RuntimeError("The causal layer range is incomplete")
    for key in (
        "source_position_counts",
        "query_position_counts",
        "matching_blocked_counts",
        "nonmatching_blocked_counts",
        "all_four_blocked_counts",
    ):
        if np.any(arrays[key] <= 0):
            raise RuntimeError(f"Nonpositive token count in {key}")
    if not np.array_equal(
        arrays["matching_blocked_counts"] + arrays["nonmatching_blocked_counts"],
        arrays["all_four_blocked_counts"],
    ):
        raise RuntimeError("Matching and nonmatching token counts do not partition all four")

    mapping_rows = json.loads(args.remapping_plan.read_text())["rows"]
    mappings = {row["question_id"]: row for row in mapping_rows}
    remapped_baseline = json.loads(args.remapped_baseline.read_text())["results"]
    original_baseline = json.loads(args.original_baseline.read_text())["results"]
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    confirmation = ~discovery
    conflict = np.asarray(
        [
            original_baseline[qid]["answer"]
            != remapped_baseline[qid]["answer_original_content"]
            for qid in qids
        ]
    )
    if int(discovery.sum()) != 251 or int(confirmation.sum()) != 249:
        raise RuntimeError("Frozen discovery/confirmation split changed")
    if int(conflict.sum()) != 273:
        raise RuntimeError("Canonical conflict definition changed")

    if prior["question_ids"].astype(str).tolist() != qids:
        raise RuntimeError("Prior full-range question order changed")
    full_band = int(np.flatnonzero(prior["bands"].astype(str) == "full_04_64")[0])
    prior_matching = prior["joint_matched_logits"][full_band].astype(float)
    matching_reproduction_error = np.abs(logits[:, 1] - prior_matching)
    if attention["question_ids"].astype(str).tolist() != qids:
        raise RuntimeError("Prior attention question order changed")
    prompt_hash_agreement = np.array_equal(
        arrays["prompt_hashes"], attention["prompt_hashes"]
    )
    rank_agreement = np.array_equal(
        rank_contents, attention["rank_letters"].astype(str)
    )
    natural_error = np.abs(logits[:, 0] - arrays["trusted_natural_logits"])
    natural_answers = logits[:, 0].argmax(-1)
    trusted_answers = arrays["trusted_natural_logits"].argmax(-1)

    aligned = _aligned_logits(logits, qids, rank_contents, mappings)
    centered = aligned - aligned.mean(axis=-1, keepdims=True)
    policy = centered[0] - centered[1]  # intervention, question, rank
    slopes = _rank_slope(policy)
    condition_slopes = _rank_slope(centered)  # condition, intervention, question
    lower_minus_upper = policy[..., 2:].mean(-1) - policy[..., :2].mean(-1)
    semantic_choice = displayed_argmax_to_semantic_indices(
        logits, [mappings[qid] for qid in qids]
    )
    chosen_rank = np.empty_like(semantic_choice)
    for qi in range(len(qids)):
        semantic_to_rank = np.empty(4, dtype=np.int64)
        for rank, semantic_letter in enumerate(rank_contents[qi]):
            semantic_to_rank[LETTERS.index(semantic_letter)] = rank
        chosen_rank[..., qi] = semantic_to_rank[semantic_choice[..., qi]]
    w1_choice = chosen_rank == 0
    switch = ~w1_choice

    split_masks = {"discovery": discovery, "confirmation": confirmation}
    subset_masks = {
        "all": np.ones(len(qids), dtype=bool),
        "conflict": conflict,
        "no_conflict": ~conflict,
    }
    rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    for split_name, split_mask in split_masks.items():
        for subset_name, subset_mask in subset_masks.items():
            mask = split_mask & subset_mask
            for intervention_index, intervention in enumerate(INTERVENTIONS):
                for rank, rank_name in enumerate(RANKS):
                    for ci, condition in enumerate(CONDITIONS):
                        values = centered[ci, intervention_index, mask, rank]
                        mean, low, high = _normal_interval(values)
                        lesion = (
                            centered[ci, intervention_index, mask, rank]
                            - centered[ci, 0, mask, rank]
                        )
                        lesion_mean, lesion_low, lesion_high = _normal_interval(lesion)
                        rows.append(
                            {
                                "split": split_name,
                                "subset": subset_name,
                                "intervention": intervention,
                                "rank": rank_name,
                                "condition": condition,
                                "n": int(mask.sum()),
                                "centered_logit": mean,
                                "centered_logit_ci_low": low,
                                "centered_logit_ci_high": high,
                                "lesion_effect": lesion_mean,
                                "lesion_effect_ci_low": lesion_low,
                                "lesion_effect_ci_high": lesion_high,
                            }
                        )
                    policy_values = policy[intervention_index, mask, rank]
                    pmean, plow, phigh = _normal_interval(policy_values)
                    policy_change = (
                        policy[intervention_index, mask, rank]
                        - policy[0, mask, rank]
                    )
                    pcmean, pclow, pchigh = _normal_interval(policy_change)
                    rows.append(
                        {
                            "split": split_name,
                            "subset": subset_name,
                            "intervention": intervention,
                            "rank": rank_name,
                            "condition": "Game_minus_Neutral",
                            "n": int(mask.sum()),
                            "centered_logit": pmean,
                            "centered_logit_ci_low": plow,
                            "centered_logit_ci_high": phigh,
                            "lesion_effect": pcmean,
                            "lesion_effect_ci_low": pclow,
                            "lesion_effect_ci_high": pchigh,
                        }
                    )

                slope_values = slopes[intervention_index, mask]
                spread_values = lower_minus_upper[intervention_index, mask]
                slope_change = slope_values - slopes[0, mask]
                spread_change = spread_values - lower_minus_upper[0, mask]
                slope_stats = _normal_interval(slope_values)
                slope_change_stats = _normal_interval(slope_change)
                spread_stats = _normal_interval(spread_values)
                spread_change_stats = _normal_interval(spread_change)
                condition_slope_stats = []
                condition_slope_change_stats = []
                w1_choice_change_stats = []
                for ci in range(2):
                    condition_slope_stats.append(
                        _normal_interval(condition_slopes[ci, intervention_index, mask])
                    )
                    condition_slope_change_stats.append(
                        _normal_interval(
                            condition_slopes[ci, intervention_index, mask]
                            - condition_slopes[ci, 0, mask]
                        )
                    )
                    w1_choice_change_stats.append(
                        _normal_interval(
                            w1_choice[ci, intervention_index, mask].astype(float)
                            - w1_choice[ci, 0, mask].astype(float)
                        )
                    )
                w1_choice_interaction_stats = _normal_interval(
                    (
                        w1_choice[0, intervention_index, mask].astype(float)
                        - w1_choice[0, 0, mask].astype(float)
                    )
                    - (
                        w1_choice[1, intervention_index, mask].astype(float)
                        - w1_choice[1, 0, mask].astype(float)
                    )
                )
                aggregate_rows.append(
                    {
                        "split": split_name,
                        "subset": subset_name,
                        "intervention": intervention,
                        "n": int(mask.sum()),
                        "policy_rank_slope": slope_stats[0],
                        "policy_rank_slope_ci_low": slope_stats[1],
                        "policy_rank_slope_ci_high": slope_stats[2],
                        "policy_rank_slope_change": slope_change_stats[0],
                        "policy_rank_slope_change_ci_low": slope_change_stats[1],
                        "policy_rank_slope_change_ci_high": slope_change_stats[2],
                        "lower_minus_upper_policy": spread_stats[0],
                        "lower_minus_upper_policy_ci_low": spread_stats[1],
                        "lower_minus_upper_policy_ci_high": spread_stats[2],
                        "lower_minus_upper_policy_change": spread_change_stats[0],
                        "lower_minus_upper_policy_change_ci_low": spread_change_stats[1],
                        "lower_minus_upper_policy_change_ci_high": spread_change_stats[2],
                        "game_w1_choice": float(w1_choice[0, intervention_index, mask].mean()),
                        "neutral_w1_choice": float(w1_choice[1, intervention_index, mask].mean()),
                        "game_switch": float(switch[0, intervention_index, mask].mean()),
                        "neutral_switch": float(switch[1, intervention_index, mask].mean()),
                        "game_rank_slope": condition_slope_stats[0][0],
                        "game_rank_slope_ci_low": condition_slope_stats[0][1],
                        "game_rank_slope_ci_high": condition_slope_stats[0][2],
                        "neutral_rank_slope": condition_slope_stats[1][0],
                        "neutral_rank_slope_ci_low": condition_slope_stats[1][1],
                        "neutral_rank_slope_ci_high": condition_slope_stats[1][2],
                        "game_rank_slope_change": condition_slope_change_stats[0][0],
                        "game_rank_slope_change_ci_low": condition_slope_change_stats[0][1],
                        "game_rank_slope_change_ci_high": condition_slope_change_stats[0][2],
                        "neutral_rank_slope_change": condition_slope_change_stats[1][0],
                        "neutral_rank_slope_change_ci_low": condition_slope_change_stats[1][1],
                        "neutral_rank_slope_change_ci_high": condition_slope_change_stats[1][2],
                        "game_w1_choice_change": w1_choice_change_stats[0][0],
                        "game_w1_choice_change_ci_low": w1_choice_change_stats[0][1],
                        "game_w1_choice_change_ci_high": w1_choice_change_stats[0][2],
                        "neutral_w1_choice_change": w1_choice_change_stats[1][0],
                        "neutral_w1_choice_change_ci_low": w1_choice_change_stats[1][1],
                        "neutral_w1_choice_change_ci_high": w1_choice_change_stats[1][2],
                        "w1_choice_change_interaction": w1_choice_interaction_stats[0],
                        "w1_choice_change_interaction_ci_low": w1_choice_interaction_stats[1],
                        "w1_choice_change_interaction_ci_high": w1_choice_interaction_stats[2],
                    }
                )

    # Nonadditivity on the centered candidate effect: all - match - nonmatch + natural.
    factorial_interaction = centered[:, 3] - centered[:, 1] - centered[:, 2] + centered[:, 0]
    factorial_summary: dict[str, object] = {}
    for split_name, split_mask in split_masks.items():
        split_result: dict[str, object] = {}
        for subset_name, subset_mask in subset_masks.items():
            mask = split_mask & subset_mask
            split_result[subset_name] = {
                condition: {
                    rank: dict(
                        zip(
                            ("mean", "ci_low", "ci_high"),
                            _normal_interval(factorial_interaction[ci, mask, ri]),
                        )
                    )
                    for ri, rank in enumerate(RANKS)
                }
                for ci, condition in enumerate(CONDITIONS)
            }
        factorial_summary[split_name] = split_result

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rank_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "aggregate_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    def select_aggregate(split: str, subset: str, intervention: str) -> dict[str, object]:
        return next(
            row
            for row in aggregate_rows
            if row["split"] == split
            and row["subset"] == subset
            and row["intervention"] == intervention
        )

    confirmation_all = {
        intervention: select_aggregate("confirmation", "all", intervention)
        for intervention in INTERVENTIONS
    }
    summary = {
        "validation": {
            "questions": len(qids),
            "discovery": int(discovery.sum()),
            "confirmation": int(confirmation.sum()),
            "conflict": int(conflict.sum()),
            "no_conflict": int((~conflict).sum()),
            "all_outputs_finite": bool(np.isfinite(logits).all()),
            "natural_max_abs_logit_error_to_trusted": float(natural_error.max()),
            "natural_answer_agreement_to_trusted": float(
                (natural_answers == trusted_answers).mean()
            ),
            "prompt_hashes_match_prior": bool(prompt_hash_agreement),
            "rank_contents_match_prior": bool(rank_agreement),
            "matching_only_max_abs_error_to_prior_full_range": float(
                matching_reproduction_error.max()
            ),
            "factorial_token_partition_exact": True,
        },
        "measurement": {
            "rank_alignment": "R1-R4 are candidates ranked by first-presentation Baseline A-D logits.",
            "centered_logit": "Candidate logit minus the mean A-D logit within the same condition/intervention/question.",
            "policy_vector": "Game centered logit minus Neutral centered logit for each first-pass rank.",
            "positive_rank_slope": "Game shifts evidence toward lower-ranked first-pass candidates relative to Neutral.",
            "primary_intervention": "Block all three nonmatching 1P option lines from every 2P option line while preserving its semantic match.",
        },
        "confirmation_all": confirmation_all,
        "factorial_interaction": factorial_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#222222", "#d95f02", "#1b9e77", "#7570b3")
    condition_colors = ("#4c78a8", "#f58518")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    mask = confirmation
    x = np.arange(4)

    primary_index = INTERVENTIONS.index("nonmatching_three_blocked")
    for ci, (condition, color) in enumerate(zip(CONDITIONS, condition_colors)):
        effects = centered[ci, primary_index, mask] - centered[ci, 0, mask]
        means = effects.mean(axis=0)
        sems = effects.std(axis=0, ddof=1) / np.sqrt(mask.sum())
        axes[0, 0].errorbar(
            x + (ci - 0.5) * 0.05,
            means,
            yerr=1.96 * sems,
            marker="o",
            capsize=4,
            linewidth=2,
            color=color,
            label=condition,
        )
    axes[0, 0].axhline(0, color="0.5", linewidth=1)
    axes[0, 0].set_xticks(x, RANKS)
    axes[0, 0].set_ylabel("Change in candidate-centered logit")
    axes[0, 0].set_title("A  Blocking the three nonmatching 1P lines")
    axes[0, 0].legend()

    group_x = np.arange(3)
    shown_interventions = INTERVENTIONS[1:]
    width = 0.34
    for ci, (condition, color) in enumerate(zip(CONDITIONS, condition_colors)):
        values = []
        lows = []
        highs = []
        for intervention in shown_interventions:
            row = confirmation_all[intervention]
            prefix = condition.lower()
            values.append(float(row[f"{prefix}_rank_slope_change"]))
            lows.append(float(row[f"{prefix}_rank_slope_change_ci_low"]))
            highs.append(float(row[f"{prefix}_rank_slope_change_ci_high"]))
        values_array = np.asarray(values)
        axes[0, 1].bar(
            group_x + (ci - 0.5) * width,
            values_array,
            width=width,
            color=color,
            label=condition,
            yerr=np.asarray(
                [values_array - np.asarray(lows), np.asarray(highs) - values_array]
            ),
            capsize=4,
        )
    axes[0, 1].axhline(0, color="0.5", linewidth=1)
    axes[0, 1].set_xticks(
        group_x, ("No match", "No 3 others", "No 1P lines")
    )
    axes[0, 1].set_ylabel("Change in within-task rank slope")
    axes[0, 1].set_title("B  Rank effect within each task")
    axes[0, 1].legend()

    slope_means = []
    slope_lows = []
    slope_highs = []
    for intervention in INTERVENTIONS:
        row = confirmation_all[intervention]
        slope_means.append(float(row["policy_rank_slope"]))
        slope_lows.append(float(row["policy_rank_slope_ci_low"]))
        slope_highs.append(float(row["policy_rank_slope_ci_high"]))
    slope_means_array = np.asarray(slope_means)
    axes[1, 0].bar(
        np.arange(4),
        slope_means_array,
        color=colors,
        yerr=np.asarray(
            [slope_means_array - slope_lows, np.asarray(slope_highs) - slope_means_array]
        ),
        capsize=4,
    )
    axes[1, 0].axhline(0, color="0.5", linewidth=1)
    axes[1, 0].set_xticks(
        np.arange(4), ("Natural", "No match", "No 3 others", "No 1P lines"), rotation=15
    )
    axes[1, 0].set_ylabel("Game − Neutral rank slope")
    axes[1, 0].set_title("C  Task-specific rank transformation")

    bar_positions = []
    bar_labels = []
    cursor = 0
    for intervention in shown_interventions:
        for subset_name in ("conflict", "no_conflict"):
            row = select_aggregate("confirmation", subset_name, intervention)
            for ci, (condition, color) in enumerate(zip(CONDITIONS, condition_colors)):
                prefix = condition.lower()
                value = 100 * float(row[f"{prefix}_w1_choice_change"])
                low = 100 * float(row[f"{prefix}_w1_choice_change_ci_low"])
                high = 100 * float(row[f"{prefix}_w1_choice_change_ci_high"])
                position = cursor + ci * 0.32
                axes[1, 1].bar(
                    position,
                    value,
                    width=0.30,
                    color=color,
                    yerr=np.asarray([[value - low], [high - value]]),
                    capsize=3,
                    label=condition if cursor == 0 else None,
                )
            bar_positions.append(cursor + 0.16)
            bar_labels.append(
                f"{INTERVENTION_LABELS[intervention].replace(' blocked', '')}\n{subset_name.replace('_', ' ')}"
            )
            cursor += 1.0
        cursor += 0.35
    axes[1, 1].axhline(0, color="0.5", linewidth=1)
    axes[1, 1].set_xticks(bar_positions, bar_labels, rotation=18, ha="right")
    axes[1, 1].set_ylabel("Change in W1 choice (percentage points)")
    axes[1, 1].set_title("D  Behavioral effects by trial type")
    axes[1, 1].legend()
    fig.suptitle(
        "Other 1P lines provide shared rank evidence; task-specific policy survives without them",
        fontsize=15,
    )
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)

    primary = confirmation_all["nonmatching_three_blocked"]
    natural = confirmation_all["natural"]
    matching = confirmation_all["matching_only_blocked"]
    all_four = confirmation_all["all_four_blocked"]
    primary_conflict = select_aggregate(
        "confirmation", "conflict", "nonmatching_three_blocked"
    )
    primary_no_conflict = select_aggregate(
        "confirmation", "no_conflict", "nonmatching_three_blocked"
    )
    report = f"""# Nonmatching first-presentation history factorial

## Bottom line

The other three first-presentation lines are causally used, but **not to create
the Game-versus-Neutral rank policy**. Blocking all three while preserving each
candidate's semantic match shifts evidence away from R1 and toward R3/R4 almost
identically in Game and Neutral:

- Game within-task rank-slope change: **{float(primary['game_rank_slope_change']):+.3f}**
  [{float(primary['game_rank_slope_change_ci_low']):+.3f}, {float(primary['game_rank_slope_change_ci_high']):+.3f}].
- Neutral within-task rank-slope change: **{float(primary['neutral_rank_slope_change']):+.3f}**
  [{float(primary['neutral_rank_slope_change_ci_low']):+.3f}, {float(primary['neutral_rank_slope_change_ci_high']):+.3f}].

Thus intact nonmatching-line reads support the old high-ranked candidates in
both tasks. They are a shared ranking/evidence route, not the source of the
distinctive revision policy.

## The task-specific result

- Natural Game-minus-Neutral rank slope: **{float(natural['policy_rank_slope']):+.3f}**
  [{float(natural['policy_rank_slope_ci_low']):+.3f}, {float(natural['policy_rank_slope_ci_high']):+.3f}].
- After blocking all three nonmatching lines while preserving the semantic
  match: **{float(primary['policy_rank_slope']):+.3f}**
  [{float(primary['policy_rank_slope_ci_low']):+.3f}, {float(primary['policy_rank_slope_ci_high']):+.3f}].
- Change caused by that intervention: **{float(primary['policy_rank_slope_change']):+.3f}**
  [{float(primary['policy_rank_slope_change_ci_low']):+.3f}, {float(primary['policy_rank_slope_change_ci_high']):+.3f}].

The null task interaction replicates across trial types:

- conflict: **{float(primary_conflict['policy_rank_slope_change']):+.3f}**
  [{float(primary_conflict['policy_rank_slope_change_ci_low']):+.3f}, {float(primary_conflict['policy_rank_slope_change_ci_high']):+.3f}];
- no conflict: **{float(primary_no_conflict['policy_rank_slope_change']):+.3f}**
  [{float(primary_no_conflict['policy_rank_slope_change_ci_low']):+.3f}, {float(primary_no_conflict['policy_rank_slope_change_ci_high']):+.3f}].

The behavioral endpoint says the same thing. Blocking the three nonmatches
changes held-out W1 choice by **{100 * float(primary['game_w1_choice_change']):+.1f}**
points in Game and **{100 * float(primary['neutral_w1_choice_change']):+.1f}**
points in Neutral; their interaction is **{100 * float(primary['w1_choice_change_interaction']):+.1f}**
[{100 * float(primary['w1_choice_change_interaction_ci_low']):+.1f},
{100 * float(primary['w1_choice_change_interaction_ci_high']):+.1f}] points.

In contrast, blocking only the semantic matches collapses the held-out
Game-minus-Neutral rank slope to **{float(matching['policy_rank_slope']):+.3f}**
[{float(matching['policy_rank_slope_ci_low']):+.3f}, {float(matching['policy_rank_slope_ci_high']):+.3f}],
and blocking all four lines gives **{float(all_four['policy_rank_slope']):+.3f}**
[{float(all_four['policy_rank_slope_ci_low']):+.3f}, {float(all_four['policy_rank_slope_ci_high']):+.3f}].
Removing the nonmatching lines after the match is already absent adds no
reliable overall policy-slope effect.

The narrow conclusion is therefore: direct access to the other three original
lines contributes shared rank evidence, but the condition-specific mapping of
first-pass rank into retention versus revision is carried through the matching
line or another preserved input already bound to it. This experiment does not
by itself distinguish a contextualized rank code inside that matching line
from a rank/policy signal supplied through the preserved answer boundary, GLA
state, or feedback pathway.

See `rank_results.csv` for R1--R4 effects and `aggregate_results.csv` for all,
conflict, and no-conflict endpoints in discovery and confirmation.

## Exact intervention

At every ordinary-attention layer 4--64 and for all four repeated option lines
simultaneously, the primary lesion blocks attention to the other three complete
first-presentation option lines while preserving the complete semantically
matching line. Matching-only and all-four blockades complete the causal
factorial. No GLA state, feedback token, question token, answer boundary, or
later decision query is directly edited.

## Validation

- Natural A-D-logit maximum error versus trusted results:
  **{summary['validation']['natural_max_abs_logit_error_to_trusted']:.6f}**.
- Natural answer agreement: **{100 * summary['validation']['natural_answer_agreement_to_trusted']:.1f}%**.
- Prompt hashes match the prior exhaustive source run:
  **{summary['validation']['prompt_hashes_match_prior']}**.
- Rank/content alignment matches the prior run:
  **{summary['validation']['rank_contents_match_prior']}**.
- Matching-only maximum A-D-logit error versus the prior independently run
  layers-4--64 joint matching lesion:
  **{summary['validation']['matching_only_max_abs_error_to_prior_full_range']:.6f}**.
- Matching and nonmatching token sets are disjoint and exactly partition all
  four first-presentation option lines.

## Artifacts

- Canonical figure: `{args.figure}`
- Machine-readable summary: `summary.json`
- Rank-level table: `rank_results.csv`
- Aggregate behavioral/rank table: `aggregate_results.csv`
"""
    (args.output_dir / "REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
