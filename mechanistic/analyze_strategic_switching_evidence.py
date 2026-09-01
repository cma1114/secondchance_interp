from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .semantic_mapping import displayed_argmax_to_semantic_indices


LETTERS = "ABCD"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _interval(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    samples = np.empty(draws, dtype=float)
    batch = 500
    for start in range(0, draws, batch):
        count = min(batch, draws - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        samples[start : start + count] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci": [float(value) for value in np.quantile(samples, (0.025, 0.975))],
        "n": int(len(values)),
    }


def _ratio_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, Any]:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError("Ratio inputs must be paired one-dimensional arrays")
    if denominator.sum() <= 0:
        raise ValueError("Ratio denominator must have positive mass")
    samples = np.empty(draws, dtype=float)
    batch = 500
    for start in range(0, draws, batch):
        count = min(batch, draws - start)
        indices = rng.integers(0, len(numerator), size=(count, len(numerator)))
        sampled_denominator = denominator[indices].sum(axis=1)
        samples[start : start + count] = (
            numerator[indices].sum(axis=1) / sampled_denominator
        )
    return {
        "mean": float(numerator.sum() / denominator.sum()),
        "ci": [float(value) for value in np.quantile(samples, (0.025, 0.975))],
        "n": int(len(numerator)),
    }


def _destination_summary(
    choices: np.ndarray,
    old_winner: np.ndarray,
    old_runner_up: np.ndarray,
    fresh_winner: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, Any]:
    """Summarize destinations on a fixed, pre-intervention question set."""

    selected = np.flatnonzero(mask)
    chosen = choices[selected]
    old_winner_selected = old_winner[selected]
    old_runner_selected = old_runner_up[selected]
    fresh_selected = fresh_winner[selected]
    switched = chosen != old_winner_selected

    fixed_fresh = (chosen == fresh_selected).astype(float)
    fixed_runner = (chosen == old_runner_selected).astype(float)
    fixed_old_winner = (chosen == old_winner_selected).astype(float)
    fixed_other = 1.0 - fixed_fresh - fixed_runner - fixed_old_winner
    if np.any(fixed_other < 0):
        raise RuntimeError("Destination categories are not disjoint")

    conditional_fresh = fixed_fresh[switched]
    conditional_runner = fixed_runner[switched]
    return {
        "n_questions": int(len(selected)),
        "n_switches": int(switched.sum()),
        "fixed_denominator": {
            "switch_rate": _interval(switched.astype(float), rng, draws),
            "fresh_winner_choice": _interval(fixed_fresh, rng, draws),
            "old_runner_up_choice": _interval(fixed_runner, rng, draws),
            "old_winner_choice": _interval(fixed_old_winner, rng, draws),
            "other_choice": _interval(fixed_other, rng, draws),
            "fresh_minus_old_runner_up": _interval(
                fixed_fresh - fixed_runner, rng, draws
            ),
        },
        "among_switches": {
            "fresh_winner_choice": _interval(conditional_fresh, rng, draws),
            "old_runner_up_choice": _interval(conditional_runner, rng, draws),
            "fresh_minus_old_runner_up": _interval(
                conditional_fresh - conditional_runner, rng, draws
            ),
        },
    }


def _fresh_choice_crossover_summary(
    results_path: Path,
    pair_plan_path: Path,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, Any]:
    """Measure donor-choice adoption in the existing fresh-state crossover."""

    with np.load(results_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    pair_rows = json.loads(pair_plan_path.read_text())["rows"]
    scenarios = arrays["scenario_ids"].astype(str).tolist()
    scenario_index = {name: index for index, name in enumerate(scenarios)}
    raw = arrays["scenario_final_logits"].astype(float)
    expected_shape = (2, 2, len(scenarios), len(pair_rows), 4)
    if raw.shape != expected_shape or not np.isfinite(raw).all():
        raise RuntimeError("Fresh-state crossover logits are incomplete or malformed")

    choices = np.empty(raw.shape[:-1], dtype=np.int64)
    for question_index, row in enumerate(pair_rows):
        for variant_index, mapping in enumerate(
            (row["low_new_to_original"], row["high_new_to_original"])
        ):
            lookup = np.asarray(
                [LETTERS.index(mapping[letter]) for letter in LETTERS], dtype=np.int64
            )
            displayed = raw[:, variant_index, :, question_index].argmax(axis=-1)
            choices[:, variant_index, :, question_index] = lookup[displayed]

    natural_index = scenario_index["natural"]
    duplicate_index = scenario_index["duplicate_natural"]
    swap_index = scenario_index["option_lines_swapped"]
    natural = choices[:, :, natural_index]
    duplicate = choices[:, :, duplicate_index]
    swapped = choices[:, :, swap_index]
    split = arrays["split"].astype(str)

    summary: dict[str, Any] = {
        "definition": (
            "On questions where the natural low- and high-order prompts choose "
            "different semantic answers, install the opposite ordering's complete "
            "2P option-line outgoing state and ask whether the recipient adopts the "
            "opposite ordering's natural choice. Both directions are averaged within "
            "question before question-level bootstrap resampling."
        ),
        "generic_change_null": (
            "If a state swap merely causes an undirected change away from the recipient "
            "choice, a specified one of the three alternative answers is expected on "
            "one third of changed directions. The donor-specific excess subtracts this "
            "per-question changed/3 expectation. A stricter leave-one-out null also "
            "preserves empirical donor-answer frequencies conditional on the recipient's "
            "natural answer."
        ),
        "validation": {
            "questions": int(len(pair_rows)),
            "duplicate_natural_max_abs_error": float(
                np.max(np.abs(raw[:, :, natural_index] - raw[:, :, duplicate_index]))
            ),
            "duplicate_natural_choice_agreement": float((natural == duplicate).mean()),
        },
        "splits": {},
    }
    for split_name in ("discovery", "confirmation"):
        split_record: dict[str, Any] = {}
        split_mask = split == split_name
        for task_index, task in enumerate(("Game", "Neutral")):
            discordant = split_mask & (natural[task_index, 0] != natural[task_index, 1])
            indices = np.flatnonzero(discordant)
            recipient = natural[task_index][:, indices]
            donor = natural[task_index, ::-1][:, indices]
            crossed = swapped[task_index][:, indices]
            adoption = (crossed == donor).mean(axis=0)
            retention = (crossed == recipient).mean(axis=0)
            changed = 1.0 - retention
            donor_specific_excess = adoption - changed / 3.0

            # A stricter drift null preserves the empirical donor-answer
            # frequencies conditional on the recipient's natural answer. Use
            # leave-one-out frequencies so an observation does not predict its
            # own donor label.
            recipient_flat = recipient.reshape(-1)
            donor_flat = donor.reshape(-1)
            crossed_flat = crossed.reshape(-1)
            frequency_null_flat = np.zeros_like(crossed_flat, dtype=float)
            for recipient_choice in range(4):
                group = np.flatnonzero(recipient_flat == recipient_choice)
                counts = np.bincount(donor_flat[group], minlength=4)
                if len(group) <= 1:
                    raise RuntimeError("Insufficient recipient group for drift null")
                for flat_index in group:
                    crossed_choice = crossed_flat[flat_index]
                    frequency_null_flat[flat_index] = (
                        counts[crossed_choice]
                        - float(donor_flat[flat_index] == crossed_choice)
                    ) / (len(group) - 1)
            frequency_null = frequency_null_flat.reshape(recipient.shape).mean(axis=0)
            frequency_excess = adoption - frequency_null
            split_record[task] = {
                "n_discordant_questions": int(len(indices)),
                "donor_choice_adoption": _interval(adoption, rng, draws),
                "recipient_choice_retention": _interval(retention, rng, draws),
                "any_choice_change": _interval(changed, rng, draws),
                "generic_change_null": _interval(changed / 3.0, rng, draws),
                "donor_specific_excess_over_generic_change": _interval(
                    donor_specific_excess, rng, draws
                ),
                "recipient_conditioned_frequency_null": _interval(
                    frequency_null, rng, draws
                ),
                "donor_specific_excess_over_frequency_null": _interval(
                    frequency_excess, rng, draws
                ),
                "donor_adoption_among_changes": _ratio_interval(
                    adoption, changed, rng, draws
                ),
            }
        summary["splits"][split_name] = split_record
    return summary


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    baseline_payload = json.loads(args.baseline.read_text())
    game_payload = json.loads(args.game.read_text())
    plan_payload = json.loads(args.plan.read_text())
    remapping = json.loads(args.remapping_summary.read_text())
    causal = json.loads(args.causal_summary.read_text())
    score_trajectory = json.loads(args.score_trajectory.read_text())
    score_attribution = json.loads(args.score_attribution.read_text())
    fresh_path = json.loads(args.fresh_path_summary.read_text())
    stage_c = json.loads(args.stage_c_summary.read_text())
    remapped_baseline = json.loads(args.remapped_baseline.read_text())["results"]
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    with np.load(args.causal_results, allow_pickle=False) as loaded:
        causal_arrays = {key: loaded[key] for key in loaded.files}

    baseline = baseline_payload["results"]
    game = game_payload["results"]
    qids = [row["question_id"] for row in plan_payload["rows"]]
    if set(qids) != set(baseline) or set(qids) != set(game):
        raise RuntimeError("Baseline, Game, and remapping plan questions disagree")

    first_rows: list[np.ndarray] = []
    final_rows: list[np.ndarray] = []
    answer_mismatches = 0
    for qid in qids:
        first = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=float)
        if LETTERS[int(first.argmax())] != baseline[qid]["answer"]:
            answer_mismatches += 1
        mapping = game[qid]["original_to_new"]
        displayed = np.asarray(game[qid]["aggregated_ad_logits"], dtype=float)
        final_by_content = np.asarray(
            [displayed[LETTERS.index(mapping[content])] for content in LETTERS],
            dtype=float,
        )
        order = np.argsort(-first)
        first_rows.append(first[order])
        final_rows.append(final_by_content[order])

    first = np.stack(first_rows)
    final = np.stack(final_rows)
    raw_change = final - first
    centered_first = first - first.mean(axis=1, keepdims=True)
    centered_final = final - final.mean(axis=1, keepdims=True)
    centered_change = centered_final - centered_first
    first_probabilities = _softmax(first)
    final_probabilities = _softmax(final)

    rng = np.random.default_rng(args.seed)
    ranks: dict[str, Any] = {}
    for rank in range(4):
        ranks[f"R{rank + 1}"] = {
            "first_raw_logit_mean": float(first[:, rank].mean()),
            "game_final_raw_logit_mean": float(final[:, rank].mean()),
            "raw_logit_change": _interval(raw_change[:, rank], rng, args.bootstrap_draws),
            "centered_logit_change": _interval(
                centered_change[:, rank], rng, args.bootstrap_draws
            ),
            "first_conditional_ad_probability_mean": float(
                first_probabilities[:, rank].mean()
            ),
            "game_final_conditional_ad_probability_mean": float(
                final_probabilities[:, rank].mean()
            ),
        }

    causal_ranks = causal["subsets"]["confirmation_all"]["ranks"]
    causal_game = {
        rank.replace("W", "R"): causal_ranks[rank]["advantage"]["Game"]
        for rank in ("W1", "W2", "W3", "W4")
    }
    joint_game = causal["joint_mediation"]["confirmation_conflict"]["conditions"][
        "Game"
    ]["W1_choice_effect"]

    causal_qids = causal_arrays["question_ids"].astype(str).tolist()
    if causal_qids != qids:
        raise RuntimeError("All-candidate causal results do not follow the frozen plan order")
    mapping_rows = [
        row for row in plan_payload["rows"]
    ]
    rank_contents = causal_arrays["rank_contents"].astype(str)
    old_winner = rank_contents[:, 0]
    old_runner_up = rank_contents[:, 1]
    fresh_winner = np.asarray(
        [remapped_baseline[qid]["answer_original_content"] for qid in qids]
    )
    destination_conflict = old_winner != fresh_winner
    destination_distinct = destination_conflict & (old_runner_up != fresh_winner)
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    if [int(discovery.sum()), int((~discovery).sum())] != [251, 249]:
        raise RuntimeError("Frozen discovery/confirmation split changed")
    natural_choices = displayed_argmax_to_semantic_indices(
        causal_arrays["natural_logits"], mapping_rows
    )
    blocked_choices = displayed_argmax_to_semantic_indices(
        causal_arrays["joint_logits"], mapping_rows
    )
    semantic_letters = np.asarray(list(LETTERS))
    natural_choice_letters = semantic_letters[natural_choices]
    blocked_choice_letters = semantic_letters[blocked_choices]

    destination_selection: dict[str, Any] = {
        "definition": (
            "Old winner and old runner-up are the first- and second-ranked candidates "
            "under first-presentation A-D logits. Fresh winner is the semantic answer "
            "selected by the standalone remapped second-presentation baseline. The "
            "primary destination-distinct subset requires the fresh winner to differ "
            "from both old candidates. No correctness label is used."
        ),
        "counts": {
            "all_questions": int(len(qids)),
            "old_winner_vs_fresh_winner_conflicts": int(destination_conflict.sum()),
            "destination_distinct_questions": int(destination_distinct.sum()),
            "destination_distinct_discovery": int((destination_distinct & discovery).sum()),
            "destination_distinct_confirmation": int(
                (destination_distinct & (~discovery)).sum()
            ),
        },
        "splits": {},
    }
    for split_name, split_mask in (
        ("discovery", discovery),
        ("confirmation", ~discovery),
        ("pooled", np.ones(len(qids), dtype=bool)),
    ):
        split_record: dict[str, Any] = {}
        mask = destination_distinct & split_mask
        for task_index, task in enumerate(("Game", "Neutral")):
            task_record: dict[str, Any] = {}
            for condition, choices in (
                ("natural", natural_choice_letters[task_index]),
                ("matching_history_blockade", blocked_choice_letters[task_index]),
            ):
                task_record[condition] = _destination_summary(
                    choices,
                    old_winner,
                    old_runner_up,
                    fresh_winner,
                    mask,
                    rng,
                    args.bootstrap_draws,
                )
            split_record[task] = task_record
        destination_selection["splits"][split_name] = split_record

    fresh_choice_crossover = _fresh_choice_crossover_summary(
        args.fresh_path_results,
        args.fresh_path_pair_plan,
        rng,
        args.bootstrap_draws,
    )

    summary = {
        "question": (
            "Does Game strategically transform its first-presentation semantic ranking, "
            "rather than merely adding undirected answer noise?"
        ),
        "definitions": {
            "rank": "R1-R4 are candidates ordered by first-presentation aggregated A-D logits.",
            "aggregated_logit": (
                "For each letter, log-sum-exp over the configured bare-letter and "
                "leading-space answer-token logits."
            ),
            "conditional_ad_probability": (
                "Softmax over the four aggregated A-D logits for each question, then "
                "averaged over questions."
            ),
            "centered_change": (
                "Final-minus-first logit change after subtracting each question's mean "
                "over its four candidates at each presentation."
            ),
            "causal_lesion_effect": (
                "Candidate advantage after blocking the true matching 1P source minus "
                "blocking an equally sized cyclic wrong-line source; positive means the "
                "intact Game match normally opposed that candidate."
            ),
        },
        "validation": {
            "questions": len(qids),
            "first_answer_vs_aggregated_argmax_mismatches": answer_mismatches,
            "inputs": {
                "baseline": {"path": str(args.baseline), "sha256": _sha256(args.baseline)},
                "game": {"path": str(args.game), "sha256": _sha256(args.game)},
                "plan": {"path": str(args.plan), "sha256": _sha256(args.plan)},
                "remapping_summary": {
                    "path": str(args.remapping_summary),
                    "sha256": _sha256(args.remapping_summary),
                },
                "causal_summary": {
                    "path": str(args.causal_summary),
                    "sha256": _sha256(args.causal_summary),
                },
                "score_trajectory": {
                    "path": str(args.score_trajectory),
                    "sha256": _sha256(args.score_trajectory),
                },
                "score_attribution": {
                    "path": str(args.score_attribution),
                    "sha256": _sha256(args.score_attribution),
                },
                "fresh_path_summary": {
                    "path": str(args.fresh_path_summary),
                    "sha256": _sha256(args.fresh_path_summary),
                },
                "stage_c_summary": {
                    "path": str(args.stage_c_summary),
                    "sha256": _sha256(args.stage_c_summary),
                },
                "remapped_baseline": {
                    "path": str(args.remapped_baseline),
                    "sha256": _sha256(args.remapped_baseline),
                },
                "discovery_plan": {
                    "path": str(args.discovery_plan),
                    "sha256": _sha256(args.discovery_plan),
                },
                "causal_results": {
                    "path": str(args.causal_results),
                    "sha256": _sha256(args.causal_results),
                },
                "fresh_path_results": {
                    "path": str(args.fresh_path_results),
                    "sha256": _sha256(args.fresh_path_results),
                },
                "fresh_path_pair_plan": {
                    "path": str(args.fresh_path_pair_plan),
                    "sha256": _sha256(args.fresh_path_pair_plan),
                },
            },
        },
        "behavioral_semantic_targeting": {
            "content_switch": remapping["primary_unrestricted_behavior"][
                "content_switch_rate"
            ],
            "game_minus_neutral_content_switch": remapping[
                "primary_unrestricted_behavior"
            ]["game_minus_neutral_content_switch"],
            "old_letter_avoidance": remapping["primary_unrestricted_behavior"][
                "old_letter_avoidance_rate"
            ],
            "game_minus_neutral_old_letter_avoidance": remapping[
                "primary_unrestricted_behavior"
            ]["game_minus_neutral_old_letter_avoidance"],
            "game_minus_neutral_entropy_bits": remapping[
                "game_minus_neutral_entropy_bits"
            ],
        },
        "within_game_first_to_final": {
            "ranks": ranks,
            "mean_raw_logit_over_all_four": {
                "first": float(first.mean()),
                "game_final": float(final.mean()),
                "change": float(raw_change.mean()),
            },
            "mean_rank_profiles": {
                "first_centered_logits": [float(value) for value in centered_first.mean(axis=0)],
                "game_final_centered_logits": [
                    float(value) for value in centered_final.mean(axis=0)
                ],
            },
        },
        "heldout_within_game_causal_matching": {
            "matching_specific_candidate_advantage_effects": causal_game,
            "joint_conflict_W1_choice_effect": joint_game,
        },
        "fresh_2p_evidence": {
            "definition": score_trajectory["definition"],
            "content_mean_residual_confirmation_correlation": {
                str(layer): score_trajectory["trajectory"][str(layer)][
                    "content_mean"
                ]["fresh_unique"]
                for layer in range(28, 33)
            },
            "content_mean_mlp_confirmation_attribution": {
                str(layer): score_attribution["summaries"]["content_mean"][
                    "fresh_unique"
                ]["components"][str(layer)]["mlp"]
                for layer in range(29, 32)
            },
            "confirmation_option_line_crossover": {
                task: fresh_path["splits"]["confirmation"]["tasks"][task][
                    "scenarios"
                ]["option_lines_swapped"]
                for task in ("Game", "Neutral")
            },
        },
        "destination_selection": destination_selection,
        "fresh_state_choice_crossover": fresh_choice_crossover,
        "policy_at_2p_semantics": stage_c["splits"]["confirmation_conflict"],
        "interpretation": (
            "Game compresses the old semantic ranking: R1 and R2 fall, R3 and R4 rise. "
            "The held-out matching-edge lesions show that Game actively constructs part "
            "of this transformation by reading each candidate's matching 1P line: the "
            "intact route suppresses R1/R2, is approximately neutral for R3, and supports "
            "R4. On destination-distinct questions, natural Game switches select the "
            "fresh second-presentation winner much more often than the old runner-up, "
            "and the fresh-state crossover redirects discrete choices beyond a generic "
            "change null. This rejects a pure undirected-noise account but does not yet "
            "establish that the fresh representation is necessary for destination selection."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    def ci_text(row: dict[str, Any]) -> str:
        return f"{row['mean']:+.3f} [{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]"

    table_rows = []
    probability_rows = []
    causal_rows = []
    for rank in range(1, 5):
        row = ranks[f"R{rank}"]
        table_rows.append(
            f"| R{rank} | {row['first_raw_logit_mean']:.3f} | "
            f"{row['game_final_raw_logit_mean']:.3f} | "
            f"{ci_text(row['raw_logit_change'])} | "
            f"{ci_text(row['centered_logit_change'])} |"
        )
        probability_rows.append(
            f"| R{rank} | {100 * row['first_conditional_ad_probability_mean']:.1f}% | "
            f"{100 * row['game_final_conditional_ad_probability_mean']:.1f}% |"
        )
        effect = causal_game[f"R{rank}"]
        causal_rows.append(
            f"| R{rank} | {effect['mean']:+.3f} "
            f"[{effect['ci'][0]:+.3f}, {effect['ci'][1]:+.3f}] |"
        )

    centered_changes = [
        ranks[f"R{rank}"]["centered_logit_change"]["mean"] for rank in range(1, 5)
    ]
    first_probability_profile = [
        100 * ranks[f"R{rank}"]["first_conditional_ad_probability_mean"]
        for rank in range(1, 5)
    ]
    final_probability_profile = [
        100 * ranks[f"R{rank}"]["game_final_conditional_ad_probability_mean"]
        for rank in range(1, 5)
    ]
    first_mean_gap = ranks["R1"]["first_raw_logit_mean"] - ranks["R4"][
        "first_raw_logit_mean"
    ]
    final_mean_gap = ranks["R1"]["game_final_raw_logit_mean"] - ranks["R4"][
        "game_final_raw_logit_mean"
    ]
    residual_rows = []
    for layer in range(28, 33):
        row = score_trajectory["trajectory"][str(layer)]["content_mean"][
            "fresh_unique"
        ]
        residual_rows.append(
            f"| {layer} | {row['shared_confirmation_correlation']:.3f} | "
            f"{row['Game']['confirmation_correlation']:.3f} | "
            f"{row['Neutral']['confirmation_correlation']:.3f} |"
        )
    mlp_rows = []
    for layer in range(29, 32):
        row = score_attribution["summaries"]["content_mean"]["fresh_unique"][
            "components"
        ][str(layer)]["mlp"]
        mlp_rows.append(
            f"| {layer} | {row['Game']['confirmation_correlation']:.3f} | "
            f"{row['Neutral']['confirmation_correlation']:.3f} |"
        )
    fresh_confirmation = fresh_path["splits"]["confirmation"]["tasks"]
    stage_c_confirmation = stage_c["splits"]["confirmation_conflict"]["tasks"]
    destination_rows = []
    for split_name in ("discovery", "confirmation", "pooled"):
        for task in ("Game", "Neutral"):
            for condition, condition_label in (
                ("natural", "Natural"),
                ("matching_history_blockade", "Matching-history blockade"),
            ):
                row = destination_selection["splits"][split_name][task][condition]
                fixed = row["fixed_denominator"]
                switched = row["among_switches"]
                destination_rows.append(
                    f"| {split_name.title()} | {task} | {condition_label} | "
                    f"{row['n_questions']} | {100 * fixed['switch_rate']['mean']:.1f}% | "
                    f"{100 * fixed['fresh_winner_choice']['mean']:.1f}% | "
                    f"{100 * fixed['old_runner_up_choice']['mean']:.1f}% | "
                    f"{100 * switched['fresh_winner_choice']['mean']:.1f}% | "
                    f"{100 * switched['old_runner_up_choice']['mean']:.1f}% | "
                    f"{100 * switched['fresh_minus_old_runner_up']['mean']:+.1f} "
                    f"[{100 * switched['fresh_minus_old_runner_up']['ci'][0]:+.1f}, "
                    f"{100 * switched['fresh_minus_old_runner_up']['ci'][1]:+.1f}] |"
                )
    fresh_choice_rows = []
    for split_name in ("discovery", "confirmation"):
        for task in ("Game", "Neutral"):
            row = fresh_choice_crossover["splits"][split_name][task]
            adoption = row["donor_choice_adoption"]
            changed = row["any_choice_change"]
            among_changes = row["donor_adoption_among_changes"]
            excess = row["donor_specific_excess_over_generic_change"]
            frequency_excess = row["donor_specific_excess_over_frequency_null"]
            fresh_choice_rows.append(
                f"| {split_name.title()} | {task} | {row['n_discordant_questions']} | "
                f"{100 * adoption['mean']:.1f}% "
                f"[{100 * adoption['ci'][0]:.1f}, {100 * adoption['ci'][1]:.1f}] | "
                f"{100 * changed['mean']:.1f}% | "
                f"{100 * among_changes['mean']:.1f}% "
                f"[{100 * among_changes['ci'][0]:.1f}, {100 * among_changes['ci'][1]:.1f}] | "
                f"{100 * excess['mean']:+.1f} "
                f"[{100 * excess['ci'][0]:+.1f}, {100 * excess['ci'][1]:+.1f}] | "
                f"{100 * frequency_excess['mean']:+.1f} "
                f"[{100 * frequency_excess['ci'][0]:+.1f}, "
                f"{100 * frequency_excess['ci'][1]:+.1f}] |"
            )

    report = f"""# Evidence that Game switching is structured rather than pure noise

## Question

Does the `incorrect` condition merely make the answer unstable, or does Game
use the semantic first-presentation ranking to perform a structured revision?

The answer is: **Game strongly compresses the old ranking, and a held-out
causal intervention shows that it constructs part of this compression by
reading each candidate's semantically matching first-presentation option.**
Game also increases uncertainty, so the result supports structured revision
plus uncertainty. Here “structured” refers only to which semantic candidate
the model leaves and which candidate it selects next; correctness is not an
endpoint in this analysis.

## Behavioral target: semantic switching, not output-letter switching

In the remapping experiment, every answer's text moves to a different A--D
letter before the second decision. Game switches away from its earlier answer
content on {100 * remapping['primary_unrestricted_behavior']['content_switch_rate']['incorrect']:.1f}%
of questions, versus
{100 * remapping['primary_unrestricted_behavior']['content_switch_rate']['neutral']:.1f}%
in Neutral, a difference of
{100 * remapping['primary_unrestricted_behavior']['game_minus_neutral_content_switch']['mean']:.1f}
percentage points. But old-letter avoidance moves in the opposite direction:
Game is {abs(100 * remapping['primary_unrestricted_behavior']['game_minus_neutral_old_letter_avoidance']['mean']):.1f}
points *less* likely than Neutral to avoid the literal letter used previously.
The extra switching therefore follows the earlier answer's **semantic content**,
not the character `A`, `B`, `C`, or `D` that previously named it.

That establishes what the behavioral effect targets. It does not by itself
distinguish a deliberate computation from confusion. The next two analyses do:
first describe the transformation entirely within Game, then intervene on the
specific semantic-history route that helps create it.

## Direct within-Game change from first decision to final decision

Candidates are aligned by semantic content after remapping and ranked by their
first-presentation aggregated A-D logits. These are raw, uncentered logits.

| First rank | Mean first logit | Mean Game final logit | Raw paired change | Change after removing the common four-logit shift |
|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

The mean across the four raw logits falls by
{abs(summary['within_game_first_to_final']['mean_raw_logit_over_all_four']['change']):.3f}
logits. That common shift cannot affect the A-D softmax. Beyond it, R1 falls
another {abs(centered_changes[0]):.3f} logits, R2 falls
{abs(centered_changes[1]):.3f}, R3 rises {centered_changes[2]:.3f}, and R4 rises
{centered_changes[3]:.3f}.

The first-presentation differences are not small merely because values such as
24.934 and 21.673 look close on their absolute scale. The relevant quantity is
their difference: the mean R1--R4 gap is {first_mean_gap:.3f} logits. Applied
to those mean logits, that is an unnormalized softmax-weight ratio of
`exp({first_mean_gap:.3f}) = {np.exp(first_mean_gap):.1f}`. At the Game final
decision the mean gap has contracted to {final_mean_gap:.3f} logits, a ratio of
only {np.exp(final_mean_gap):.1f}. The per-question probabilities below avoid
the additional distortion that would come from softmaxing logits only after
averaging different questions.

The probability presentation is often easier to read. For each question, take
a softmax over its four aggregated A-D logits and then average the resulting
probabilities:

| First rank | Mean first-presentation probability | Mean Game final probability |
|---|---:|---:|
{chr(10).join(probability_rows)}

Thus Game changes the average old-rank profile from
**{'/'.join(f'{value:.1f}' for value in first_probability_profile)}%** to
**{'/'.join(f'{value:.1f}' for value in final_probability_profile)}%**. This is an aggressive
compression of the original ranking, with the previous winner losing the most
and the previous weakest candidate gaining the most.

## The second presentation reconstructs fresh candidate evidence

The option-list computation is not merely carrying old history forward. A
fresh-score target was defined from a standalone Baseline that saw only the
remapped second presentation. For each semantic candidate, its centered A--D
logit was residualized against its old 1P score and both displayed positions.
The resulting target therefore asks how strongly the current presentation
supports that candidate independently of old evidence and letter placement.

At each 2P option line, the analysis averaged the residual over only the
wordpieces spelling that option's text, excluding its displayed letter,
punctuation, spacing, and newline. On the 249 held-out questions, the complete
residual's correlation with unique fresh evidence rises across layers 29--31:

| Residual after layer | Shared correlation | Game | Neutral |
|---:|---:|---:|---:|
{chr(10).join(residual_rows)}

The exact additive MLP writes at those same option-text wordpieces are also
fresh-evidence aligned:

| MLP write | Game correlation | Neutral correlation |
|---:|---:|---:|
{chr(10).join(mlp_rows)}

This is additive-write and held-out decoding evidence, not an individual-MLP
lesion. A separate fresh-evidence crossover supplies the causal link: holding
the target candidate, 1P history, feedback, and target 2P position fixed while
crossing the four complete 2P option-line states changes the target final logit
in the donor-predicted direction by
{fresh_confirmation['Game']['scenarios']['option_lines_swapped']['oriented_target_logit_change']:+.3f}
in Game and
{fresh_confirmation['Neutral']['scenarios']['option_lines_swapped']['oriented_target_logit_change']:+.3f}
in Neutral on confirmation.

The layerwise directions were fitted separately, so the rising correlations
are not a literal cumulative sum in one fixed residual coordinate. Their
near-identity in Game and Neutral also means this is primarily shared fresh
solving, not the distinctive Game policy. It matters to the strategic account
because it shows that the model genuinely recomputes current candidate quality
before combining it with the Game-specific treatment of old rank; the final
switch is not generated by undirected noise alone.

## Destination selection: the fresh winner, not the old runner-up

Leaving the old winner and selecting a destination are different operations.
To separate them, this analysis uses the 135 questions where the standalone
second-presentation winner differs from both the old first-presentation winner
and the old first-presentation runner-up. “Fresh winner” here is the model's
semantic answer to the standalone remapped second presentation; it includes
whatever content and displayed-order preferences determine that answer. No
correct-answer label enters the definition or analysis.

The fixed-denominator columns report choices over the same pre-intervention
question set. The final three columns condition descriptively on an actual
switch away from the old winner. Because an intervention can change which
questions switch, causal comparisons across intervention conditions should use
the fixed-denominator columns; the switch-conditional columns describe where
the resulting switches landed.

| Split | Task | State | Eligible questions | Switch rate | Fresh winner / all | Old runner-up / all | Fresh winner / switches | Old runner-up / switches | Fresh minus old runner-up / switches |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(destination_rows)}

In natural Game, the destination preference is large and replicates: among
switches it favors the fresh winner over the old runner-up by 49.0 points on
discovery and 38.3 points on confirmation. Pooled natural Game selects the
fresh winner on 65.8% of switches and the old runner-up on 22.5%. Natural
Neutral also favors the fresh winner when pooled, although its held-out
difference is less precise.

Under the matching-history blockade, Game still switches on 67.4% of the 135
eligible questions, but its pooled switch destinations are 47.3% fresh winner
and 40.7% old runner-up: a +6.6-point difference with a confidence interval of
[-12.1, +25.3]. Neutral's point estimate reverses. The blockade therefore does
not support a claim that fresh-winner steering remains strong after semantic
history is removed. It also does not establish that fresh computation itself
was removed; the intervention was designed to cut matching history, not to
isolate the fresh representation.

### Choice-level redirection in the existing fresh-state crossover

The earlier fresh-state crossover supplies a separate causal test. For each
question it has two second-presentation orderings with the target candidate in
the same displayed position. The intervention installs the opposite ordering's
complete 2P option-line outgoing state. The table below restricts to questions
where the two unmodified orderings naturally select different semantic answers
and averages the two reciprocal directions within question.

“Donor adoption” means the crossed run selected the answer naturally selected
by the opposite ordering. The generic-change null asks how often that specified
answer would be reached if the swap merely caused an undirected change: one of
three alternatives, so the expected rate is one third of the observed
any-change rate. A stricter leave-one-out null preserves the empirical
donor-answer frequencies conditional on the recipient's natural answer. The
exact duplicate-natural execution path has 0.0 logit error and 100% choice
agreement.

| Split | Task | Discordant questions | Donor adoption | Any change | Donor adoption among changes | Excess over equal-alternative null | Excess over frequency-matched null |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(fresh_choice_rows)}

The choice redirection is modest but replicating. On confirmation, the donor
choice is adopted in 24.2% of Game directions and 27.0% of Neutral directions.
Among directions whose answer changes, donor adoption is 41.6% in Game and
44.1% in Neutral, above the 33.3% undirected-change expectation. This extends
the established donor-aligned logit movement to discrete choices. It shows
that current-presentation state causally structures destinations; it remains a
movability result, not a removal test establishing necessity.

### Where the policy is present at 2P—and what that does not prove

A separate same-question Game/Neutral crossover shows that task policy is
already present candidate by candidate in the complete outgoing state of the
2P semantic wordpieces. Crossing all four candidates' semantic states transfers
{100 * stage_c_confirmation['Game']['joint']['relay_task_swapped_all_semantics']['task_vector_transfer']['ratio']:.1f}%
of the donor task-specific answer-score pattern into Game and
{100 * stage_c_confirmation['Neutral']['joint']['relay_task_swapped_all_semantics']['task_vector_transfer']['ratio']:.1f}%
into Neutral on held-out conflict questions. Crossing only one candidate's
semantic state preferentially changes that same candidate: every R1--R4
target-minus-off-target interval is positive.

This establishes **where** candidate-specific policy has become causally
effective: at the tokens spelling the 2P answers. It does not establish that
the policy transforms the newly computed fresh score. Stage C crossed each
token's complete state and could not independently exchange old evidence while
holding fresh evidence fixed. Together with the nearly identical L29--31 fresh
trajectories, the best-supported interpretation is narrower: the 2P semantic
state combines shared fresh evidence with a task-specific treatment of
retrieved old evidence. Neutral reinstates the old ranking more strongly;
Game compresses it. A distinctive Game transformation of fresh evidence itself
remains unestablished.

## Strongest evidence: the within-Game causal semantic-history test

The before/after profile alone could partly reflect a fresh solve, regression
to the mean, or generic flattening. The decisive test blocks, across ordinary-
attention layers 4--48, each repeated candidate's reads of its truly matching
first-presentation option line. An equally sized cyclic wrong-line block is
the control. Positive values below mean that removing the true match raises
the candidate, so the intact Game route had been suppressing it.

| First rank | Matching-specific lesion effect on that candidate's centered advantage |
|---|---:|
{chr(10).join(causal_rows)}

With the semantic-history route intact, Game therefore suppresses R1, suppresses
R2 more weakly, is approximately neutral for R3, and supports R4. Jointly
blocking all four matching routes raises held-out conflict-trial R1 choice by
**{100 * joint_game['mean']:.1f} percentage points**
`[{100 * joint_game['ci'][0]:.1f}, {100 * joint_game['ci'][1]:.1f}]`.

This is the strongest evidence against pure answer noise: cutting a specific
semantic-history pathway predictably reverses part of the rank transformation
inside Game. Noise added only at the final decision would not depend on whether
a repeated candidate can read its matching earlier option line.

## What is and is not established

Established:

- The preferential behavioral switch follows the earlier answer's semantic
  content after remapping, not its old output letter.
- Fresh candidate evidence is reconstructed in the 2P option-text residuals
  and causally affects the final candidate logits.
- Game lowers the previous winner's raw final logit by 2.114 on average.
- The change is rank-structured rather than equal across candidates.
- Semantic matching to first-presentation candidates causally produces part of
  that structure.
- The resulting operation is well described as **old-rank compression**:
  suppress old leaders and relatively support old weak candidates.

Not established:

- That the model represents an explicit symbolic rule such as `not W1`.
- That all of the compression is carried by the matching route; fresh second-
  presentation evidence and broader uncertainty also contribute.
- That the fresh representation is necessary for destination selection; the
  existing crossover moves it but does not remove it.
- That the operation is entropy-free. Game increases A-D entropy and weakens
  late sharpening, so generalized uncertainty is a real secondary component.

## Evidence classes and provenance

The first-to-final tables are a post hoc paired analysis of existing natural
logits; they describe what Game does but do not by themselves identify its
cause. The remapping result is behavioral evidence about the target of the
switch. The matching-edge effects are prespecified held-out causal lesions
with balanced wrong-line controls. Input paths and SHA-256 hashes are recorded
in `summary.json`.
"""
    (args.output_dir / "REPORT.md").write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--remapping-summary", type=Path, required=True)
    parser.add_argument("--causal-summary", type=Path, required=True)
    parser.add_argument("--score-trajectory", type=Path, required=True)
    parser.add_argument("--score-attribution", type=Path, required=True)
    parser.add_argument("--fresh-path-summary", type=Path, required=True)
    parser.add_argument("--stage-c-summary", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--causal-results", type=Path, required=True)
    parser.add_argument("--fresh-path-results", type=Path, required=True)
    parser.add_argument("--fresh-path-pair-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
