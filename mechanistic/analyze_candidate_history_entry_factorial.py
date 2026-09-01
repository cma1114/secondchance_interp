from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .run_candidate_history_entry_factorial import (
    ALL_OPEN_MASK,
    RANKS,
    TOKEN_CLASSES,
)
from .semantic_mapping import (
    align_displayed_logits_to_semantic,
    displayed_argmax_to_semantic_indices,
)

CONDITION_LABELS = ("Game", "Neutral")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _rank_values(values: np.ndarray, rank_indices: np.ndarray) -> np.ndarray:
    output = np.empty(values.shape, dtype=values.dtype)
    for question_index in range(values.shape[-2]):
        output[..., question_index, :] = values[
            ..., question_index, rank_indices[question_index]
        ]
    return output


def _ranked_advantage(
    displayed_logits: np.ndarray,
    mapping_rows: list[dict[str, Any]],
    rank_indices: np.ndarray,
) -> np.ndarray:
    semantic = align_displayed_logits_to_semantic(displayed_logits, mapping_rows)
    centered = semantic - (semantic.sum(-1, keepdims=True) - semantic) / 3.0
    return _rank_values(centered, rank_indices)


def _ranked_choices(
    displayed_logits: np.ndarray,
    mapping_rows: list[dict[str, Any]],
    rank_indices: np.ndarray,
) -> np.ndarray:
    semantic_choice = displayed_argmax_to_semantic_indices(
        displayed_logits, mapping_rows
    )
    output = np.zeros(displayed_logits.shape, dtype=float)
    for question_index in range(displayed_logits.shape[-2]):
        for rank in range(4):
            output[..., question_index, rank] = (
                semantic_choice[..., question_index]
                == rank_indices[question_index, rank]
            )
    return output


def _shapley(metric: np.ndarray) -> np.ndarray:
    """Five-factor per-question Shapley values; mask is the first axis."""

    if metric.shape[0] != 32:
        raise ValueError(f"Expected 32 masks, got {metric.shape}")
    n_factors = len(TOKEN_CLASSES)
    output = np.zeros((n_factors,) + metric.shape[1:], dtype=float)
    for factor in range(n_factors):
        bit = 1 << factor
        for mask in range(32):
            if mask & bit:
                continue
            size = int(mask.bit_count())
            weight = (
                math.factorial(size)
                * math.factorial(n_factors - size - 1)
                / math.factorial(n_factors)
            )
            output[factor] += weight * (metric[mask | bit] - metric[mask])
    return output


def _interval(values: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"Interval expects one value per question, got {values.shape}")
    if len(values) == 0:
        return {"n": 0, "mean": float("nan"), "ci": [float("nan"), float("nan")]}
    rng = np.random.default_rng(seed)
    bootstrap = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "ci": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
    }


def _summarize_ranks(
    values: np.ndarray, mask: np.ndarray, seed: int, draws: int
) -> dict[str, Any]:
    return {
        rank: _interval(values[mask, rank_index], seed + rank_index, draws)
        for rank_index, rank in enumerate(RANKS)
    }


def _fmt(row: dict[str, Any], scale: float = 1.0) -> str:
    return (
        f"{row['mean'] * scale:+.3f} "
        f"[{row['ci'][0] * scale:+.3f}, {row['ci'][1] * scale:+.3f}]"
    )


def analyze(args: argparse.Namespace) -> None:
    arrays = _load(args.results)
    if len(arrays["question_ids"]) != 500 or not arrays["completed"].all():
        raise RuntimeError("Stage A requires a complete 500-question checkpoint")
    if arrays["availability_masks"].tolist() != list(range(32)):
        raise RuntimeError("Checkpoint is not the complete 32-cell factorial")
    if arrays["token_classes"].astype(str).tolist() != list(TOKEN_CLASSES):
        raise RuntimeError(
            "Checkpoint token partition differs from the frozen analysis"
        )
    for key in (
        "baseline_logits",
        "trusted_natural_logits",
        "natural_logits",
        "factorial_logits",
    ):
        if not np.all(np.isfinite(arrays[key])):
            raise RuntimeError(f"Non-finite values in {key}")

    qids = arrays["question_ids"].astype(str).tolist()
    mapping_lookup = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    mapping_rows = [mapping_lookup[qid] for qid in qids]
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    if int(discovery.sum()) != 251:
        raise RuntimeError("Frozen discovery split is not 251 questions")
    confirmation = ~discovery

    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    rank_contents = arrays["rank_contents"].astype(str)
    rank_indices = np.asarray(
        [[LETTERS.index(letter) for letter in row] for row in rank_contents],
        dtype=np.int64,
    )
    w1 = rank_contents[:, 0]
    w2 = np.asarray([remapped[qid]["answer_original_content"] for qid in qids])
    conflict = w1 != w2

    natural_logits = arrays["natural_logits"].astype(float)
    trusted_logits = arrays["trusted_natural_logits"].astype(float)
    factorial_logits = arrays["factorial_logits"].astype(float)
    natural_advantage = _ranked_advantage(natural_logits, mapping_rows, rank_indices)
    factorial_advantage = _ranked_advantage(
        factorial_logits, mapping_rows, rank_indices
    )
    natural_choice = _ranked_choices(natural_logits, mapping_rows, rank_indices)
    factorial_choice = _ranked_choices(factorial_logits, mapping_rows, rank_indices)

    # Availability values. Source mode 0 blocks matching sources; source mode 1
    # applies the same receiver lesion to a balanced wrong source line.
    matching = factorial_advantage[:, 0]
    wrong = factorial_advantage[:, 1]
    specificity = matching - wrong
    shapley_raw = np.stack([_shapley(matching[ci]) for ci in range(2)])
    shapley_specific = np.stack([_shapley(specificity[ci]) for ci in range(2)])

    splits = {
        "discovery": discovery,
        "confirmation": confirmation,
        "discovery_conflict": discovery & conflict,
        "confirmation_conflict": confirmation & conflict,
        "discovery_no_conflict": discovery & ~conflict,
        "confirmation_no_conflict": confirmation & ~conflict,
    }
    summary: dict[str, Any] = {
        "definitions": {
            "availability_mask": "bit=1 permits that 2P token class to read its matching 1P option line; bit=0 blocks that edge",
            "necessity": "effect of blocking only one matching receiver class from the natural all-open state",
            "sufficiency": "effect of opening only one matching receiver class from the all-closed state",
            "matching_specific": "matching-source effect minus the same receiver lesion aimed at a balanced wrong 1P line",
            "candidate_advantage": "candidate logit minus the mean of the other three candidate logits, in semantic rank coordinates",
        },
        "validation": {
            "questions": len(qids),
            "discovery": int(discovery.sum()),
            "confirmation": int(confirmation.sum()),
            "conflict": int(conflict.sum()),
            "natural_max_abs_error": float(
                np.max(np.abs(natural_logits - trusted_logits))
            ),
            "all_open_identity_max_abs_error": float(
                np.max(
                    np.abs(
                        factorial_logits[:, :, ALL_OPEN_MASK] - natural_logits[:, None]
                    )
                )
            ),
            "max_abs_intervention_change": float(
                np.max(np.abs(factorial_logits - natural_logits[:, None, None]))
            ),
            "wrong_source_offset_counts": {
                str(offset): int(
                    np.sum(
                        np.mod(
                            arrays["wrong_source_ranks"]
                            - np.arange(4, dtype=np.int8)[None, :],
                            4,
                        )
                        == offset
                    )
                )
                for offset in (1, 2, 3)
            },
        },
        "splits": {},
    }

    csv_rows: list[dict[str, Any]] = []
    seed_counter = args.seed
    for split_name, split_mask in splits.items():
        split_record: dict[str, Any] = {"n": int(split_mask.sum()), "conditions": {}}
        for condition_index, condition in enumerate(CONDITION_LABELS):
            condition_record: dict[str, Any] = {
                "complete_block": {},
                "token_classes": {},
            }
            complete_raw = (
                matching[condition_index, 0] - natural_advantage[condition_index]
            )
            complete_specific = specificity[condition_index, 0]
            complete_choice_raw = (
                factorial_choice[condition_index, 0, 0]
                - natural_choice[condition_index]
            )
            complete_choice_specific = (
                factorial_choice[condition_index, 0, 0]
                - factorial_choice[condition_index, 1, 0]
            )
            condition_record["complete_block"] = {
                "raw_candidate_advantage_change": _summarize_ranks(
                    complete_raw, split_mask, seed_counter, args.draws
                ),
                "matching_specific_advantage_change": _summarize_ranks(
                    complete_specific, split_mask, seed_counter + 10, args.draws
                ),
                "raw_choice_probability_change": _summarize_ranks(
                    complete_choice_raw, split_mask, seed_counter + 20, args.draws
                ),
                "matching_specific_choice_change": _summarize_ranks(
                    complete_choice_specific,
                    split_mask,
                    seed_counter + 30,
                    args.draws,
                ),
            }
            seed_counter += 100

            for class_index, token_class in enumerate(TOKEN_CLASSES):
                block_one_mask = ALL_OPEN_MASK ^ (1 << class_index)
                allow_only_mask = 1 << class_index
                necessity_raw = (
                    matching[condition_index, block_one_mask]
                    - natural_advantage[condition_index]
                )
                necessity_specific = specificity[condition_index, block_one_mask]
                sufficiency_raw = (
                    matching[condition_index, allow_only_mask]
                    - matching[condition_index, 0]
                )
                sufficiency_specific = (
                    specificity[condition_index, allow_only_mask]
                    - specificity[condition_index, 0]
                )
                token_record = {
                    "block_one_mask": block_one_mask,
                    "allow_only_mask": allow_only_mask,
                    "necessity_raw": _summarize_ranks(
                        necessity_raw, split_mask, seed_counter, args.draws
                    ),
                    "necessity_matching_specific": _summarize_ranks(
                        necessity_specific,
                        split_mask,
                        seed_counter + 10,
                        args.draws,
                    ),
                    "sufficiency_raw": _summarize_ranks(
                        sufficiency_raw, split_mask, seed_counter + 20, args.draws
                    ),
                    "sufficiency_matching_specific": _summarize_ranks(
                        sufficiency_specific,
                        split_mask,
                        seed_counter + 30,
                        args.draws,
                    ),
                    "shapley_raw": _summarize_ranks(
                        shapley_raw[condition_index, class_index],
                        split_mask,
                        seed_counter + 40,
                        args.draws,
                    ),
                    "shapley_matching_specific": _summarize_ranks(
                        shapley_specific[condition_index, class_index],
                        split_mask,
                        seed_counter + 50,
                        args.draws,
                    ),
                }
                seed_counter += 100
                condition_record["token_classes"][token_class] = token_record
                for metric in (
                    "necessity_raw",
                    "necessity_matching_specific",
                    "sufficiency_raw",
                    "sufficiency_matching_specific",
                    "shapley_raw",
                    "shapley_matching_specific",
                ):
                    for rank in RANKS:
                        row = token_record[metric][rank]
                        csv_rows.append(
                            {
                                "split": split_name,
                                "condition": condition,
                                "token_class": token_class,
                                "metric": metric,
                                "rank": rank,
                                "n": row["n"],
                                "mean": row["mean"],
                                "ci_low": row["ci"][0],
                                "ci_high": row["ci"][1],
                            }
                        )
            split_record["conditions"][condition] = condition_record
        summary["splits"][split_name] = split_record

    # Exact Shapley additivity is a deterministic analysis invariant.
    shapley_raw_error = float(
        np.max(
            np.abs(shapley_raw.sum(1) - (matching[:, ALL_OPEN_MASK] - matching[:, 0]))
        )
    )
    shapley_specific_error = float(
        np.max(
            np.abs(
                shapley_specific.sum(1)
                - (specificity[:, ALL_OPEN_MASK] - specificity[:, 0])
            )
        )
    )
    summary["validation"]["shapley_raw_max_abs_additivity_error"] = shapley_raw_error
    summary["validation"]["shapley_specific_max_abs_additivity_error"] = (
        shapley_specific_error
    )
    if shapley_raw_error > 1e-10 or shapley_specific_error > 1e-10:
        raise RuntimeError("Shapley allocation failed exact additivity")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "token_class_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(24, 10), sharey=True)
    x = np.arange(len(TOKEN_CLASSES))
    rank_colors = ("#a83232", "#d17b20", "#398a46", "#245f9e")
    panel_specs = (
        ("necessity_matching_specific", "Block-one necessity (matched − wrong)"),
        ("sufficiency_matching_specific", "Allow-only sufficiency (matched − wrong)"),
        ("shapley_matching_specific", "Shapley allocation (matched − wrong)"),
    )
    confirmation_record = summary["splits"]["confirmation"]["conditions"]
    for condition_index, condition in enumerate(CONDITION_LABELS):
        for panel_index, (metric, title) in enumerate(panel_specs):
            axis = axes[condition_index, panel_index]
            for rank_index, rank in enumerate(RANKS):
                rank_rows = [
                    confirmation_record[condition]["token_classes"][token_class][
                        metric
                    ][rank]
                    for token_class in TOKEN_CLASSES
                ]
                means = np.asarray([row["mean"] for row in rank_rows])
                lows = np.asarray([row["ci"][0] for row in rank_rows])
                highs = np.asarray([row["ci"][1] for row in rank_rows])
                axis.errorbar(
                    x + (rank_index - 1.5) * 0.055,
                    means,
                    yerr=np.vstack([means - lows, highs - means]),
                    color=rank_colors[rank_index],
                    marker="o",
                    linewidth=1.2,
                    capsize=2,
                    label=rank,
                )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(title)
            axis.set_xticks(x, TOKEN_CLASSES, rotation=25, ha="right")
            axis.grid(axis="y", alpha=0.2)

        axis = axes[condition_index, 3]
        rows = confirmation_record[condition]["complete_block"][
            "matching_specific_advantage_change"
        ]
        means = np.asarray([rows[rank]["mean"] for rank in RANKS])
        lows = np.asarray([rows[rank]["ci"][0] for rank in RANKS])
        highs = np.asarray([rows[rank]["ci"][1] for rank in RANKS])
        rank_x = np.arange(4)
        axis.errorbar(
            rank_x,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            color="#333333",
            marker="o",
            capsize=3,
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(rank_x, RANKS)
        axis.set_title("Complete matching-edge blockade")
        axis.grid(axis="y", alpha=0.2)
        axes[condition_index, 0].set_ylabel(
            f"{condition}\ncandidate-centered logits"
        )

    # Candidate-centered effects sum to zero across R1-R4 by construction, so
    # rank averaging would erase the signal. Keep every rank explicit.
    axes[0, 0].legend(frameon=False, ncol=4, loc="upper left")
    fig.suptitle(
        "Where matching 1P history enters 2P option lines — confirmation split",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)

    confirmation = summary["splits"]["confirmation"]["conditions"]
    lines = [
        "# Candidate-history entry factorial",
        "",
        "## Design",
        "",
        "All 32 availability subsets of the five disjoint 2P option-line token classes were tested across ordinary-attention layers L4–L64. Every non-natural matching-source lesion has an equal receiver-cell lesion aimed at a balanced wrong 1P line.",
        "",
        "## Validation",
        "",
        f"- 500 questions: 251 discovery, 249 confirmation; {int(conflict.sum())} 1P/2P conflicts.",
        f"- Natural maximum A–D error: {summary['validation']['natural_max_abs_error']:.8f}.",
        f"- All-open identity maximum A–D error: {summary['validation']['all_open_identity_max_abs_error']:.8f}.",
        f"- Maximum intervention change: {summary['validation']['max_abs_intervention_change']:.3f} logits.",
        "",
        "## Confirmation complete-block effect",
        "",
        "Values are matching-source minus balanced-wrong-source changes in candidate-centered logits.",
        "",
        "| Rank | Game | Neutral |",
        "|---|---:|---:|",
    ]
    for rank in RANKS:
        game = confirmation["Game"]["complete_block"][
            "matching_specific_advantage_change"
        ][rank]
        neutral = confirmation["Neutral"]["complete_block"][
            "matching_specific_advantage_change"
        ][rank]
        lines.append(f"| {rank} | {_fmt(game)} | {_fmt(neutral)} |")
    lines.extend(
        [
            "",
            "## Primary token-class finding",
            "",
            "The semantic wordpieces are the dominant entry route. The table compares the full matching-edge blockade with blocking only semantic-wordpiece receivers and with opening only semantic-wordpiece receivers from the all-closed state. The allow-only sign is reversed because reopening a successful route restores the natural state.",
            "",
            "| Task | Rank | Complete block | Block semantic only | Open semantic only |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for condition in CONDITION_LABELS:
        complete = confirmation[condition]["complete_block"][
            "matching_specific_advantage_change"
        ]
        semantic = confirmation[condition]["token_classes"]["semantic"]
        for rank in RANKS:
            lines.append(
                f"| {condition} | {rank} | {_fmt(complete[rank])} | "
                f"{_fmt(semantic['necessity_matching_specific'][rank])} | "
                f"{_fmt(semantic['sufficiency_matching_specific'][rank])} |"
            )
    lines.extend(
        [
            "",
            "On the confirmation split, blocking semantic-wordpiece reads while leaving all other routes open nearly reproduces the complete-block rank vector in both tasks. Conversely, semantic reads alone recover most of the route from the all-closed state. Leading spaces, option letters, and colons are individually small. Newlines carry a smaller secondary Game signal, especially R1/R4, but are neither the main necessary route nor sufficient for the full effect.",
            "",
            "The discovery split independently shows the same route-level conclusion: semantic wordpieces dominate necessity, sufficiency, and Shapley allocation. The exact Neutral redistribution across R1-R4 varies between splits, so the robust claim is about the semantic receiver route, not one frozen Neutral rank profile.",
            "",
            "## Confirmation choice effects of complete blockade",
            "",
            "These are matching-source minus balanced-wrong-source changes in the probability of each semantic rank being the top A-D choice.",
            "",
            "| Rank | Game | Neutral |",
            "|---|---:|---:|",
        ]
    )
    for rank in RANKS:
        game = confirmation["Game"]["complete_block"][
            "matching_specific_choice_change"
        ][rank]
        neutral = confirmation["Neutral"]["complete_block"][
            "matching_specific_choice_change"
        ][rank]
        lines.append(f"| {rank} | {_fmt(game)} | {_fmt(neutral)} |")
    lines.extend(
        [
            "",
            "## Reading the token-class results",
            "",
            "- Block-one necessity asks whether removing a class hurts when every other class remains open.",
            "- Allow-only sufficiency asks whether that class alone recovers the matching-history effect from the all-closed state.",
            "- Shapley values average each class's marginal contribution across every possible background of the other four classes, retaining redundancy and synergy in the measured 32-cell response surface.",
            "- The figure shows confirmation results. Machine-readable discovery and confirmation results, separately for Game and Neutral and R1–R4, are in `summary.json` and `token_class_effects.csv`.",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=25082026)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
