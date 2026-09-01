from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mechanistic.semantic_mapping import displayed_argmax_to_semantic_indices


OPTION_ROOT = ROOT / "outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping"


def _compact(path: Path) -> str:
    return re.sub(r"\s+", " ", path.read_text()).strip()


def _pct(value: float) -> float:
    return 100.0 * float(value)


def _all_candidate_rates() -> dict[str, float]:
    result_path = OPTION_ROOT / "all_candidate_matched_relay/run/results.npz"
    with np.load(result_path, allow_pickle=False) as arrays:
        qids = arrays["question_ids"].astype(str).tolist()
        rank_contents = arrays["rank_contents"].astype(str)
        natural_logits = arrays["natural_logits"]
        joint_logits = arrays["joint_logits"]
    mapping_by_qid = {
        row["question_id"]: row
        for row in json.loads((OPTION_ROOT / "plan.json").read_text())["rows"]
    }
    mapping_rows = [mapping_by_qid[qid] for qid in qids]
    discovery_ids = set(json.loads(
        (ROOT / "outputs/causal/qwen36_27b_simplemc_causal_sweep/plans/discovery_plan.json").read_text()
    )["question_ids"])
    remapped = json.loads((OPTION_ROOT / "remapped_baseline_results.json").read_text())["results"]
    confirmation = np.asarray([qid not in discovery_ids for qid in qids])
    w1 = rank_contents[:, 0]
    w2 = np.asarray([remapped[qid]["answer_original_content"] for qid in qids])
    mask = confirmation & (w1 != w2)
    w1_indices = np.asarray(["ABCD".index(value) for value in w1])
    natural = displayed_argmax_to_semantic_indices(natural_logits, mapping_rows)
    joint = displayed_argmax_to_semantic_indices(joint_logits, mapping_rows)
    rows = np.arange(len(qids))
    return {
        "game_natural": float((natural[0, mask] == w1_indices[mask]).mean()),
        "neutral_natural": float((natural[1, mask] == w1_indices[mask]).mean()),
        "game_joint": float((joint[0, mask] == w1_indices[mask]).mean()),
        "neutral_joint": float((joint[1, mask] == w1_indices[mask]).mean()),
    }


def collect_errors(root: Path = ROOT) -> list[str]:
    option_root = root / "outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping"
    policy = json.loads((
        option_root
        / "second_presentation_residual_workspace/policy_rank_factorial/analysis/summary.json"
    ).read_text())
    action = json.loads((
        option_root / "feedback_factorial/action_period_mediation/analysis/summary.json"
    ).read_text())
    all_summary = json.loads((
        option_root / "all_candidate_matched_relay/analysis/summary.json"
    ).read_text())
    rates = _all_candidate_rates()

    policy_conf = policy["splits"]["confirmation_conflict"]["conditions"]
    game_policy = _pct(policy_conf["Game"]["conflict_W1_choice"]["policy_swap_effect"]["mean"])
    neutral_policy = _pct(policy_conf["Neutral"]["conflict_W1_choice"]["policy_swap_effect"]["mean"])

    action_conflict = action["subsets"]["conflict_W1_not_equal_W2"]["scenarios"]
    action_values: dict[str, tuple[float, float, float]] = {}
    for scenario in ("gla_state", "joint", "residual_trajectory"):
        for direction in ("neutral_into_evaluation", "evaluation_into_neutral"):
            row = action_conflict[scenario][direction]["w1_minus_w2_margin"]["fraction_of_natural_gap"]
            action_values[f"{scenario}:{direction}"] = (
                _pct(row["mean"]), _pct(row["ci_low"]), _pct(row["ci_high"])
            )

    gap = all_summary["joint_mediation"]["confirmation_conflict"]["gap_reduction"]["W1_choice"]
    game_nat = _pct(rates["game_natural"])
    neutral_nat = _pct(rates["neutral_natural"])
    game_joint = _pct(rates["game_joint"])
    neutral_joint = _pct(rates["neutral_joint"])
    gap_mean, gap_low, gap_high = _pct(gap["mean"]), _pct(gap["ci"][0]), _pct(gap["ci"][1])

    documents = {
        "README": _compact(root / "README.md"),
        "synthesis": _compact(root / "QWEN36_GAME_NEUTRAL_MECHANISTIC_SYNTHESIS.md"),
        "root PLAN": _compact(option_root / "PLAN.md"),
        "all-candidate PLAN": _compact(option_root / "all_candidate_matched_relay/PLAN.md"),
    }
    expected = {
        "README": [
            f"+{game_policy:.1f} points in Game and {neutral_policy:.1f} in Neutral",
            (
                f"{action_values['gla_state:neutral_into_evaluation'][0]:.1f}% "
                f"[{action_values['gla_state:neutral_into_evaluation'][1]:.1f}, "
                f"{action_values['gla_state:neutral_into_evaluation'][2]:.1f}]"
            ),
            (
                f"{action_values['joint:evaluation_into_neutral'][0]:.1f}% "
                f"[{action_values['joint:evaluation_into_neutral'][1]:.1f}, "
                f"{action_values['joint:evaluation_into_neutral'][2]:.1f}]"
            ),
            f"Game {game_nat:.1f}%→{game_joint:.1f}%; Neutral {neutral_nat:.1f}%→{neutral_joint:.1f}%",
            f"[+{gap_low:.1f}, +{gap_high:.1f}] points",
        ],
        "synthesis": [
            f"+{game_policy:.1f} points; swapping Game's update into Neutral",
            f"from {game_nat:.1f}%/{neutral_nat:.1f}% in Game/Neutral to {game_joint:.1f}%/{neutral_joint:.1f}%",
            f"{gap_mean:.1f} [{gap_low:.1f}, {gap_high:.1f}] points",
        ],
        "root PLAN": [
            f"Game W1 choice from {game_nat:.1f}% to {game_joint:.1f}% and Neutral from {neutral_nat:.1f}% to {neutral_joint:.1f}%",
            f"interval is +{gap_low:.1f} to +{gap_high:.1f} points",
        ],
        "all-candidate PLAN": [
            f"-{gap_mean:.1f} to a 0.0-point estimate in confirmation",
            f"interval is +{gap_low:.1f} to +{gap_high:.1f} points",
        ],
    }
    errors: list[str] = []
    for label, snippets in expected.items():
        for snippet in snippets:
            if snippet not in documents[label]:
                errors.append(f"{label} is missing machine-derived headline: {snippet}")

    seed_root = root / "outputs/model_replications/seed_oss_36b_mechanistic_replication"
    seed_simple = json.loads(
        (seed_root / "simplemc/fresh_removal/analysis/summary.json").read_text()
    )
    seed_trivia = json.loads(
        (seed_root / "triviamc/fresh_removal/analysis/summary.json").read_text()
    )

    def seed_interaction(
        summary: dict[str, object], subset: str, contrast: str, endpoint: str
    ) -> dict[str, float]:
        return summary["subset_contrasts"][subset][contrast][endpoint][
            "Game_minus_Neutral_interaction"
        ]

    simple_fresh = seed_interaction(
        seed_simple,
        "confirmation_all",
        "fresh_minus_random",
        "old_winner_centered_advantage",
    )
    trivia_fresh = seed_interaction(
        seed_trivia,
        "confirmation_conflict",
        "fresh_minus_random",
        "old_winner_centered_advantage",
    )
    simple_joint = seed_simple["subsets"]["confirmation_conflict"][
        "matching_plus_fresh"
    ]["Game_minus_Neutral"]["old_winner_centered_advantage"]
    trivia_joint = seed_trivia["subsets"]["confirmation_conflict"][
        "matching_plus_fresh"
    ]["Game_minus_Neutral"]["old_winner_centered_advantage"]

    simple_fresh_text = (
        f"{simple_fresh['mean']:+.3f} "
        f"`[{simple_fresh['ci'][0]:+.3f},{simple_fresh['ci'][1]:+.3f}]`"
    )
    trivia_fresh_text = (
        f"{trivia_fresh['mean']:+.3f} "
        f"`[{trivia_fresh['ci'][0]:+.3f},{trivia_fresh['ci'][1]:+.3f}]`"
    )
    joint_text = f"{simple_joint['mean']:.3f}/{trivia_joint['mean']:.3f}"
    seed_documents = {
        "Seed README": _compact(root / "README.md"),
        "Seed integrated report": _compact(seed_root / "REPORT.md"),
        "Seed PLAN": _compact(seed_root / "PLAN.md"),
    }
    seed_expected = {
        "Seed README": [simple_fresh_text, trivia_fresh_text, joint_text],
        "Seed integrated report": [simple_fresh_text, trivia_fresh_text],
        "Seed PLAN": [simple_fresh_text, trivia_fresh_text],
    }
    for label, snippets in seed_expected.items():
        for snippet in snippets:
            if snippet not in seed_documents[label]:
                errors.append(
                    f"{label} is missing Seed fresh-removal headline: {snippet}"
                )

    qwen_fresh = json.loads(
        (
            option_root
            / "fresh_history_double_dissociation/analysis/summary.json"
        ).read_text()
    )
    qwen_full_fresh = seed_interaction(
        qwen_fresh,
        "confirmation_all",
        "fresh_minus_random",
        "old_winner_centered_advantage",
    )
    qwen_conflict_fresh = seed_interaction(
        qwen_fresh,
        "confirmation_conflict",
        "fresh_minus_random",
        "old_winner_centered_advantage",
    )
    qwen_snippets = [
        f"{qwen_full_fresh['mean']:+.3f}",
        f"{qwen_conflict_fresh['mean']:+.3f}",
    ]
    for label, document in {
        "Qwen README": documents["README"],
        "Qwen synthesis": documents["synthesis"],
    }.items():
        for snippet in qwen_snippets:
            if snippet not in document:
                errors.append(
                    f"{label} is missing Qwen fresh-removal headline: {snippet}"
                )
    return errors


def main() -> None:
    errors = collect_errors()
    if errors:
        raise SystemExit("\n".join(errors))
    print("Canonical headline quotations match the machine-readable results.")


if __name__ == "__main__":
    main()
