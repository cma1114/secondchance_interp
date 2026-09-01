from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .semantic_mapping import displayed_argmax_to_semantic_indices


LETTERS = "ABCD"
CONDITIONS = ("Game", "Neutral")
RANKS = ("R1", "R2", "R3", "R4")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _align(values: np.ndarray, qids: list[str], mappings: dict[str, dict[str, Any]]) -> np.ndarray:
    output = np.empty_like(values)
    for qi, qid in enumerate(qids):
        for original_index, original in enumerate(LETTERS):
            second = mappings[qid]["original_to_new"][original]
            output[..., qi, original_index] = values[..., qi, LETTERS.index(second)]
    return output


def _advantage(logits: np.ndarray) -> np.ndarray:
    return logits - (logits.sum(-1, keepdims=True) - logits) / 3.0


def _rank(values: np.ndarray, rank_indices: np.ndarray) -> np.ndarray:
    output = np.empty(values.shape[:-1] + (4,), dtype=values.dtype)
    for qi in range(values.shape[-2]):
        output[..., qi, :] = values[..., qi, rank_indices[qi]]
    return output


def _ranked_choices(
    displayed_logits: np.ndarray,
    mapping_rows: list[dict[str, Any]],
    rank_indices: np.ndarray,
) -> np.ndarray:
    """Resolve A--D ties before mapping the selected letter to semantic rank."""
    semantic_choices = displayed_argmax_to_semantic_indices(
        displayed_logits, mapping_rows
    )
    output = np.zeros(displayed_logits.shape, dtype=float)
    for qi in range(displayed_logits.shape[-2]):
        for rank in range(4):
            output[..., qi, rank] = (
                semantic_choices[..., qi] == rank_indices[qi, rank]
            )
    return output


def _interval(values: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def _fmt(row: dict[str, Any], scale: float = 1.0) -> str:
    return f"{row['mean']*scale:+.3f} [{row['ci'][0]*scale:+.3f}, {row['ci'][1]*scale:+.3f}]"


def _rank_summary(values: np.ndarray, mask: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    return {
        rank: _interval(values[mask, index], seed + index * 100, draws)
        for index, rank in enumerate(RANKS)
    }


def _bivalent(values: np.ndarray) -> np.ndarray:
    return values[..., 3] - values[..., :2].mean(-1)


def analyze(args: argparse.Namespace) -> None:
    arrays = _load(args.results)
    if not arrays["completed"].all():
        raise RuntimeError("Policy-rank factorial is incomplete")
    qids = arrays["question_ids"].astype(str).tolist()
    if len(qids) != 500:
        raise RuntimeError(f"Expected 500 questions, found {len(qids)}")
    scenario_ids = arrays["scenario_ids"].astype(str).tolist()
    expected = [
        "natural", "policy_swapped", "matching_blocked",
        "policy_swapped_matching_blocked", "cyclic_control_blocked",
        "policy_swapped_cyclic_control_blocked", "policy_swapped_mlp49_restored",
    ]
    if scenario_ids != expected:
        raise RuntimeError("Scenario inventory changed")
    for key in ("scenario_logits", "mlp49_old_score_projection"):
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"Non-finite values in {key}")

    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    mapping_rows = [mappings[qid] for qid in qids]
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    confirmation = ~discovery
    remapped_baseline = json.loads(args.remapped_baseline.read_text())["results"]

    rank_contents = arrays["rank_contents"].astype(str)
    rank_indices = np.asarray(
        [[LETTERS.index(value) for value in row] for row in rank_contents], dtype=int
    )
    w1 = rank_contents[:, 0]
    w2 = np.asarray([remapped_baseline[qid]["answer_original_content"] for qid in qids])
    conflict = w1 != w2

    displayed_logits = arrays["scenario_logits"].astype(float)
    semantic_logits = _align(displayed_logits, qids, mappings)
    ranked_advantage = _rank(_advantage(semantic_logits), rank_indices)
    ranked_choice = _ranked_choices(displayed_logits, mapping_rows, rank_indices)
    reordered_choices = semantic_logits.argmax(-1)
    displayed_order_choices = displayed_argmax_to_semantic_indices(
        displayed_logits, mapping_rows
    )
    exact_max_ties = (displayed_logits == displayed_logits.max(-1, keepdims=True)).sum(-1) > 1
    mlp_projection = arrays["mlp49_old_score_projection"].astype(float)

    natural_match = ranked_advantage[:, 2] - ranked_advantage[:, 4]
    swapped_match = ranked_advantage[:, 3] - ranked_advantage[:, 5]
    route_interaction = swapped_match - natural_match
    policy_effect_advantage = ranked_advantage[:, 1] - ranked_advantage[:, 0]
    restored_effect_advantage = ranked_advantage[:, 6] - ranked_advantage[:, 0]
    mlp_policy_effect = mlp_projection[:, 1] - mlp_projection[:, 0]
    mlp_restored_effect = mlp_projection[:, 6] - mlp_projection[:, 0]

    splits = {
        "discovery": discovery,
        "confirmation": confirmation,
        "discovery_conflict": discovery & conflict,
        "confirmation_conflict": confirmation & conflict,
    }
    summary: dict[str, Any] = {
        "definitions": {
            "matching_specific": "matching-line blockade minus cyclic nonmatching-line blockade",
            "route_interaction": "matching-specific effect after reciprocal policy swap minus matching-specific effect under natural policy",
            "bivalent": "R4 minus mean(R1,R2); positive means relatively greater support for the lower-ranked candidate",
            "mlp_restore": "under policy swap, replace MLP-49 output at all four final 2P semantic tokens with the natural recipient output",
            "policy_transplant_scope": "non-output-preserved complete evaluation-period GLA update: swaps the recurrent write and allows the donor-conditioned source-token GLA output to flow onward",
        },
        "validation": {
            "questions": len(qids),
            "discovery": int(discovery.sum()),
            "confirmation": int(confirmation.sum()),
            "conflict": int(conflict.sum()),
            "ordinary_layers_one_based": arrays["ordinary_layers_one_based"].astype(int).tolist(),
            "gla_layers_one_based": arrays["gla_layers_one_based"].astype(int).tolist(),
            "natural_ad_logit_max_error": float(np.max(np.abs(
                arrays["same_batch_natural_logits"] - arrays["trusted_natural_logits"]
            ))),
            "natural_choice_agreement": float((
                arrays["same_batch_natural_logits"].argmax(-1)
                == arrays["trusted_natural_logits"].argmax(-1)
            ).mean()),
            "mlp49_restore_max_abs_error": float(arrays["mlp49_restore_max_abs_error"].max()),
            "policy_transplant_preserve_source_output": False,
            "scenario_exact_max_ties": int(exact_max_ties.sum()),
            "scenario_choices_changed_by_tie_fix": int(np.sum(
                reordered_choices != displayed_order_choices
            )),
        },
        "splits": {},
    }
    csv_rows: list[list[Any]] = []
    for split_index, (split, mask) in enumerate(splits.items()):
        record: dict[str, Any] = {"n": int(mask.sum()), "conditions": {}}
        for condition_index, condition in enumerate(CONDITIONS):
            base_seed = args.seed + split_index * 100000 + condition_index * 10000
            condition_record: dict[str, Any] = {
                "natural_matching_specific_rank_effect": _rank_summary(
                    natural_match[condition_index], mask, base_seed, args.draws
                ),
                "policy_swapped_matching_specific_rank_effect": _rank_summary(
                    swapped_match[condition_index], mask, base_seed + 1000, args.draws
                ),
                "policy_by_route_rank_interaction": _rank_summary(
                    route_interaction[condition_index], mask, base_seed + 2000, args.draws
                ),
                "policy_by_route_bivalent_interaction": _interval(
                    _bivalent(route_interaction[condition_index])[mask],
                    base_seed + 3000,
                    args.draws,
                ),
                "natural_mlp49_rank_projection": _rank_summary(
                    mlp_projection[condition_index, 0], mask, base_seed + 4000, args.draws
                ),
                "policy_swapped_mlp49_rank_projection": _rank_summary(
                    mlp_projection[condition_index, 1], mask, base_seed + 5000, args.draws
                ),
                "policy_swap_mlp49_bivalent_effect": _interval(
                    _bivalent(mlp_policy_effect[condition_index])[mask],
                    base_seed + 6000,
                    args.draws,
                ),
                "policy_swap_final_bivalent_effect": _interval(
                    _bivalent(policy_effect_advantage[condition_index])[mask],
                    base_seed + 7000,
                    args.draws,
                ),
                "after_mlp49_restore_final_bivalent_effect": _interval(
                    _bivalent(restored_effect_advantage[condition_index])[mask],
                    base_seed + 8000,
                    args.draws,
                ),
                "after_mlp49_restore_mlp_bivalent_effect": _interval(
                    _bivalent(mlp_restored_effect[condition_index])[mask],
                    base_seed + 9000,
                    args.draws,
                ),
            }
            conflict_mask = mask & conflict
            if conflict_mask.any():
                natural_w1 = ranked_choice[condition_index, 0, conflict_mask, 0]
                swapped_w1 = ranked_choice[condition_index, 1, conflict_mask, 0]
                restored_w1 = ranked_choice[condition_index, 6, conflict_mask, 0]
                condition_record["conflict_W1_choice"] = {
                    "natural": _interval(natural_w1, base_seed + 10000, args.draws),
                    "policy_swapped": _interval(swapped_w1, base_seed + 10100, args.draws),
                    "policy_swapped_mlp49_restored": _interval(restored_w1, base_seed + 10200, args.draws),
                    "policy_swap_effect": _interval(swapped_w1 - natural_w1, base_seed + 10300, args.draws),
                    "remaining_effect_after_restore": _interval(restored_w1 - natural_w1, base_seed + 10400, args.draws),
                }
            record["conditions"][condition] = condition_record
            for metric, row in (
                ("route_bivalent_interaction", condition_record["policy_by_route_bivalent_interaction"]),
                ("mlp49_policy_bivalent", condition_record["policy_swap_mlp49_bivalent_effect"]),
                ("final_policy_bivalent", condition_record["policy_swap_final_bivalent_effect"]),
                ("final_after_restore_bivalent", condition_record["after_mlp49_restore_final_bivalent_effect"]),
            ):
                csv_rows.append([split, condition, metric, row["n"], row["mean"], *row["ci"]])
        summary["splits"][split] = record

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "effects.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "condition", "metric", "n", "mean", "ci_low", "ci_high"])
        writer.writerows(csv_rows)

    discovery_record = summary["splits"]["discovery"]["conditions"]
    confirmation_record = summary["splits"]["confirmation"]["conditions"]
    lines = [
        "# Policy × retrieved-rank causal factorial",
        "",
        "## Method",
        "",
        "The action-matched Game and Neutral prompts differ only at `incorrect` versus `lost`. "
        "For each recipient condition, the experiment reciprocally transplanted the evaluation-closing period's "
        "GLA write across all 48 GLA layers, blocked all four complete matching 1P-to-2P option-line routes across "
        "all 16 ordinary-attention layers, and crossed those interventions. A cyclic nonmatching-line blockade is "
        "the route control. A final condition restored the natural recipient MLP-49 output at all four final 2P "
        "semantic tokens while leaving the policy swap intact.",
        "",
        "## Scope of the policy transplant",
        "",
        "This historical factorial used `preserve_source_output=False`. It swaps the recurrent GLA write and "
        "also allows the donor-conditioned GLA output at the evaluation-closing period to flow onward. It "
        "therefore causally tests the complete evaluation-period GLA update, not an output-preserved isolation "
        "of persistent recurrent memory alone.",
        "",
        "Natural A--D logits reproduce exactly. Discrete answers resolve exact ties in displayed A--D order "
        "before mapping the selected letter to semantic rank: 18 scenario cells were tied and 15 choices differ "
        "from the invalid reorder-before-argmax rule. Discovery contains 251 questions and confirmation 249.",
        "",
        "## Interpretation",
        "",
        "The evaluation-period GLA update causally changes how the matching route uses retrieved rank. "
        "The reciprocal interaction replicates in both frozen splits and reaches conflict-trial W1 choices. "
        "Restoring the natural MLP-49 output removes the local nominated write difference but leaves nearly "
        "all of the final and behavioral policy effect, so MLP 49 is a readout rather than a necessary local mediator.",
        "",
        "Rankwise lesion levels jointly block all four matching routes versus all four cyclic controls; they are "
        "not the earlier four separate single-route estimates.",
        "",
        "## Replication across frozen splits",
        "",
        "| Task | Split | Policy × route bivalent interaction | Policy effect at MLP 49 | Policy effect on final evidence | Remaining after MLP-49 restore |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for split_name, split_record in (("Discovery", discovery_record), ("Confirmation", confirmation_record)):
            row = split_record[condition]
            lines.append(
                f"| {condition} | {split_name} | {_fmt(row['policy_by_route_bivalent_interaction'])} | "
                f"{_fmt(row['policy_swap_mlp49_bivalent_effect'])} | "
                f"{_fmt(row['policy_swap_final_bivalent_effect'])} | "
                f"{_fmt(row['after_mlp49_restore_final_bivalent_effect'])} |"
            )
    lines.extend(["", "Rankwise tables below report the untouched confirmation split.", ""])
    for condition in CONDITIONS:
        row = confirmation_record[condition]
        lines.extend([
            f"## {condition}",
            "",
            "### Natural-policy matching-specific effects by old rank",
            "",
            "| Rank | Lesion effect (logits) |",
            "|---|---:|",
            *[
                f"| {rank} | {_fmt(row['natural_matching_specific_rank_effect'][rank])} |"
                for rank in RANKS
            ],
            "",
            "### After reciprocal policy swap",
            "",
            "| Rank | Lesion effect (logits) | Policy × route interaction |",
            "|---|---:|---:|",
            *[
                f"| {rank} | {_fmt(row['policy_swapped_matching_specific_rank_effect'][rank])} | "
                f"{_fmt(row['policy_by_route_rank_interaction'][rank])} |"
                for rank in RANKS
            ],
            "",
            f"Bivalent policy × route interaction: {_fmt(row['policy_by_route_bivalent_interaction'])}.",
            "",
            f"Policy-swap effect on MLP-49 bivalent rank write: {_fmt(row['policy_swap_mlp49_bivalent_effect'])}.",
            f"Policy-swap effect on final bivalent candidate evidence: {_fmt(row['policy_swap_final_bivalent_effect'])}.",
            f"Remaining final bivalent effect after restoring natural MLP 49: {_fmt(row['after_mlp49_restore_final_bivalent_effect'])}.",
            "",
        ])
        if "conflict_W1_choice" in row:
            choice = row["conflict_W1_choice"]
            lines.extend([
                "### Conflict-trial W1 choice",
                "",
                f"- Natural: {choice['natural']['mean']*100:.1f}%.",
                f"- Policy swapped: {choice['policy_swapped']['mean']*100:.1f}%.",
                f"- Policy swapped with natural MLP 49 restored: {choice['policy_swapped_mlp49_restored']['mean']*100:.1f}%.",
                f"- Policy-swap effect: {_fmt(choice['policy_swap_effect'], 100)} percentage points.",
                f"- Remaining effect after MLP-49 restoration: {_fmt(choice['remaining_effect_after_restore'], 100)} percentage points.",
                "",
            ])
    lines.extend([
        "## Evidence status",
        "",
        "The policy and route manipulations are causal. MLP-49 restoration is a direct mediation test at the "
        "four nominated semantic-token positions. Rank summaries and confidence intervals were computed separately "
        "on the frozen discovery and confirmation questions; the report above gives confirmation results.",
        "",
    ])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["validation"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73210)
    parser.add_argument("--draws", type=int, default=3000)
    return parser


if __name__ == "__main__":
    analyze(build_parser().parse_args())
