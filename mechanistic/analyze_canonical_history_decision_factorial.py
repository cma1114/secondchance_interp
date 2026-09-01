from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODELS = ("qwen36_27b", "seed_oss_36b", "gemma4_31b")
MODEL_LABELS = {
    "qwen36_27b": "Qwen3.6-27B",
    "seed_oss_36b": "Seed-OSS 36B",
    "gemma4_31b": "Gemma 4 31B",
}
DATASETS = ("simplemc", "triviamc")
DATASET_LABELS = {"simplemc": "SimpleMC", "triviamc": "TriviaMC"}
CELLS = (
    "natural",
    "matching",
    "cyclic_wrong",
    "first_decision",
    "matching_plus_first_decision",
)
CELL_LABELS = {
    "natural": "Natural",
    "matching": "Matching\n1P→2P",
    "cyclic_wrong": "Wrong-line\ncontrol",
    "first_decision": "First-decision\nsource",
    "matching_plus_first_decision": "Matching +\nfirst decision",
}


def _split_ids(root: Path, dataset: str) -> set[str]:
    if dataset == "simplemc":
        path = root / "outputs/causal/qwen36_27b_simplemc_causal_sweep/plans/discovery_plan.json"
        payload = json.loads(path.read_text())
        values = payload.get("question_ids", payload.get("discovery_question_ids"))
    else:
        path = root / "outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/split_plan.json"
        payload = json.loads(path.read_text())
        values = payload.get("discovery_question_ids", payload.get("question_ids"))
    if not values:
        raise ValueError(f"Frozen discovery split has no question IDs: {path}")
    return set(str(value) for value in values)


def _rank_aligned(logits: np.ndarray, rank_contents: np.ndarray) -> np.ndarray:
    indices = np.asarray([[ord(letter) - ord("A") for letter in row] for row in rank_contents])
    return np.take_along_axis(logits, indices[None, None, :, :], axis=-1)


def _stratified_bootstrap_indices(
    rank_contents: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    selected: list[np.ndarray] = []
    labels = rank_contents[:, 0]
    for letter in "ABCD":
        group = np.flatnonzero(mask & (labels == letter))
        if group.size:
            selected.append(rng.choice(group, size=group.size, replace=True))
    if not selected:
        raise ValueError("Bootstrap mask is empty")
    return np.concatenate(selected)


def _metrics(aligned: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    # aligned: task x cell x question x old-rank
    centered = aligned - aligned.mean(axis=-1, keepdims=True)
    choices = np.argmax(aligned, axis=-1) == 0
    result: dict[str, Any] = {}
    for cell_index, cell in enumerate(CELLS):
        game_w1 = float(choices[0, cell_index, mask].mean())
        neutral_w1 = float(choices[1, cell_index, mask].mean())
        result[cell] = {
            "game_w1_choice": game_w1,
            "neutral_w1_choice": neutral_w1,
            "old_winner_avoidance_gap_pp": 100.0 * (neutral_w1 - game_w1),
            "game_w1_minus_w2": float(
                (aligned[0, cell_index, mask, 0] - aligned[0, cell_index, mask, 1]).mean()
            ),
            "neutral_w1_minus_w2": float(
                (aligned[1, cell_index, mask, 0] - aligned[1, cell_index, mask, 1]).mean()
            ),
            "game_minus_neutral_w1_centered_logit": float(
                (centered[0, cell_index, mask, 0] - centered[1, cell_index, mask, 0]).mean()
            ),
            "game_centered_rank_effect": [
                float(value)
                for value in (
                    centered[0, cell_index, mask].mean(axis=0)
                    - centered[0, 0, mask].mean(axis=0)
                )
            ],
            "neutral_centered_rank_effect": [
                float(value)
                for value in (
                    centered[1, cell_index, mask].mean(axis=0)
                    - centered[1, 0, mask].mean(axis=0)
                )
            ],
        }
    natural_choice_gap = result["natural"]["old_winner_avoidance_gap_pp"]
    natural_logit_gap = result["natural"]["game_minus_neutral_w1_centered_logit"]
    for cell in CELLS[1:]:
        result[cell]["avoidance_gap_reduction_pp"] = (
            natural_choice_gap - result[cell]["old_winner_avoidance_gap_pp"]
        )
        result[cell]["w1_logit_gap_reduction"] = (
            result[cell]["game_minus_neutral_w1_centered_logit"] - natural_logit_gap
        )
    return result


def _bootstrap(
    aligned: np.ndarray,
    rank_contents: np.ndarray,
    mask: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = {
        cell: {"avoidance_gap_reduction_pp": [], "w1_logit_gap_reduction": []}
        for cell in CELLS[1:]
    }
    contrasts = {
        "matching_minus_wrong": {
            "avoidance_gap_reduction_pp": [],
            "w1_logit_gap_reduction": [],
        },
        "joint_minus_matching": {
            "avoidance_gap_reduction_pp": [],
            "w1_logit_gap_reduction": [],
        },
    }
    for _ in range(draws):
        indices = _stratified_bootstrap_indices(rank_contents, mask, rng)
        sample_mask = np.zeros(mask.shape, dtype=bool)
        # Preserve repeats by indexing directly rather than converting indices to a mask.
        sample = aligned[:, :, indices]
        centered = sample - sample.mean(axis=-1, keepdims=True)
        choices = np.argmax(sample, axis=-1) == 0
        natural_choice_gap = 100.0 * (choices[1, 0].mean() - choices[0, 0].mean())
        natural_logit_gap = float((centered[0, 0, :, 0] - centered[1, 0, :, 0]).mean())
        for cell_index, cell in enumerate(CELLS[1:], 1):
            choice_gap = 100.0 * (
                choices[1, cell_index].mean() - choices[0, cell_index].mean()
            )
            logit_gap = float(
                (centered[0, cell_index, :, 0] - centered[1, cell_index, :, 0]).mean()
            )
            values[cell]["avoidance_gap_reduction_pp"].append(
                natural_choice_gap - choice_gap
            )
            values[cell]["w1_logit_gap_reduction"].append(logit_gap - natural_logit_gap)
        for endpoint in ("avoidance_gap_reduction_pp", "w1_logit_gap_reduction"):
            contrasts["matching_minus_wrong"][endpoint].append(
                values["matching"][endpoint][-1] - values["cyclic_wrong"][endpoint][-1]
            )
            contrasts["joint_minus_matching"][endpoint].append(
                values["matching_plus_first_decision"][endpoint][-1]
                - values["matching"][endpoint][-1]
            )
    return {
        "cells": {
            cell: {
                endpoint: [float(x) for x in np.percentile(samples, [2.5, 97.5])]
                for endpoint, samples in endpoints.items()
            }
            for cell, endpoints in values.items()
        },
        "contrasts": {
            name: {
                endpoint: [float(x) for x in np.percentile(samples, [2.5, 97.5])]
                for endpoint, samples in endpoints.items()
            }
            for name, endpoints in contrasts.items()
        },
    }


def _load_cell(root: Path, model: str, dataset: str) -> dict[str, Any]:
    run_dir = (
        root
        / "outputs/model_replications/canonical_history_decision_factorial"
        / model
        / dataset
        / "run"
    )
    with np.load(run_dir / "results.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    if arrays["cells"].astype(str).tolist() != list(CELLS):
        raise ValueError(f"{model}/{dataset}: cell order changed")
    if not np.all(arrays["completed"]):
        raise ValueError(f"{model}/{dataset}: run is incomplete")
    if not np.all(np.isfinite(arrays["logits"])):
        raise ValueError(f"{model}/{dataset}: logits are non-finite")
    if float(metadata["natural_reproduction_max_absolute_error"]) != 0.0:
        raise ValueError(f"{model}/{dataset}: natural reproduction is not exact")
    qids = arrays["question_ids"].astype(str)
    rank_contents = arrays["rank_contents"].astype(str)
    aligned = _rank_aligned(np.asarray(arrays["logits"], dtype=float), rank_contents)
    discovery_ids = _split_ids(root, dataset)
    discovery = np.asarray([qid in discovery_ids for qid in qids], dtype=bool)
    if not discovery.any() or discovery.all():
        raise ValueError(f"{model}/{dataset}: frozen split did not partition questions")
    subsets = {
        "full": np.ones(len(qids), dtype=bool),
        "discovery": discovery,
        "confirmation": ~discovery,
    }
    result: dict[str, Any] = {
        "validity": {
            "n": len(qids),
            "n_discovery": int(discovery.sum()),
            "n_confirmation": int((~discovery).sum()),
            "all_complete": True,
            "all_logits_finite": True,
            "natural_reproduction_max_absolute_error": 0.0,
            "ordinary_attention_layers_one_based": metadata[
                "ordinary_attention_layers_one_based"
            ],
            "gla_layers": metadata["gla_layers"],
        },
        "subsets": {},
    }
    for offset, (subset, mask) in enumerate(subsets.items()):
        metrics = _metrics(aligned, mask)
        bootstrap = _bootstrap(aligned, rank_contents, mask, 10000, 20260901 + offset)
        for cell in CELLS[1:]:
            metrics[cell]["ci95"] = bootstrap["cells"][cell]
        contrasts = {
            "matching_minus_wrong": {
                endpoint: (
                    metrics["matching"][endpoint]
                    - metrics["cyclic_wrong"][endpoint]
                )
                for endpoint in ("avoidance_gap_reduction_pp", "w1_logit_gap_reduction")
            },
            "joint_minus_matching": {
                endpoint: (
                    metrics["matching_plus_first_decision"][endpoint]
                    - metrics["matching"][endpoint]
                )
                for endpoint in ("avoidance_gap_reduction_pp", "w1_logit_gap_reduction")
            },
        }
        for name in contrasts:
            contrasts[name]["ci95"] = bootstrap["contrasts"][name]
        result["subsets"][subset] = {
            "n": int(mask.sum()),
            "cells": metrics,
            "contrasts": contrasts,
        }
    return result


def _figure(summary: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), sharex=True)
    intervention_cells = CELLS[1:]
    x = np.arange(len(intervention_cells))
    for row, dataset in enumerate(DATASETS):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            cells = summary["results"][model][dataset]["subsets"]["confirmation"]["cells"]
            choice = [cells[cell]["avoidance_gap_reduction_pp"] for cell in intervention_cells]
            choice_ci = np.asarray(
                [cells[cell]["ci95"]["avoidance_gap_reduction_pp"] for cell in intervention_cells]
            )
            logit = [cells[cell]["w1_logit_gap_reduction"] for cell in intervention_cells]
            logit_ci = np.asarray(
                [cells[cell]["ci95"]["w1_logit_gap_reduction"] for cell in intervention_cells]
            )
            ax.axhline(0, color="#bbbbbb", linewidth=1)
            ax.errorbar(
                x - 0.06,
                choice,
                yerr=np.vstack((np.asarray(choice) - choice_ci[:, 0], choice_ci[:, 1] - np.asarray(choice))),
                marker="o",
                linewidth=2,
                capsize=4,
                color="#7b3294",
                label="Choice-gap reduction",
            )
            twin = ax.twinx()
            twin.errorbar(
                x + 0.06,
                logit,
                yerr=np.vstack((np.asarray(logit) - logit_ci[:, 0], logit_ci[:, 1] - np.asarray(logit))),
                marker="s",
                linewidth=2,
                capsize=4,
                color="#008837",
                label="W1-logit-gap reduction",
            )
            ax.set_title(f"{MODEL_LABELS[model]} — {DATASET_LABELS[dataset]}")
            ax.set_xticks(x)
            ax.set_xticklabels([CELL_LABELS[cell] for cell in intervention_cells], fontsize=8)
            if col == 0:
                ax.set_ylabel("Reduction in W1 avoidance gap (pp)", color="#7b3294")
            if col == 2:
                twin.set_ylabel("Reduction in W1 logit suppression", color="#008837")
            ax.tick_params(axis="y", colors="#7b3294")
            twin.tick_params(axis="y", colors="#008837")
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Canonical non-remapped route effects (frozen confirmation; 95% bootstrap CIs)",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _format_ci(value: float, ci: list[float]) -> str:
    return f"{value:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"


def analyze(root: Path, output_dir: Path, figure_path: Path) -> None:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "results": {},
        "interpretation_scope": {
            "matching": "Complete matching 1P option-line reads by complete 2P option lines",
            "cyclic_wrong": "Same all-four intervention form with cyclic nonmatching sources",
            "first_decision": "Complete first-decision-token outgoing attention route; plus its GLA write on Qwen",
            "not_established": "The first-decision intervention does not isolate a literal-letter coordinate inside that token state",
            "semantic_scope": "On non-remapped prompts, semantic candidate identity, displayed letter, and line identity coincide. Semantic mapping invariance comes from the prior remapped experiments; this experiment establishes that the candidate-matched line route is also used in the canonical prompt.",
        },
    }
    for model in MODELS:
        summary["results"][model] = {}
        for dataset in DATASETS:
            summary["results"][model][dataset] = _load_cell(root, model, dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _figure(summary, figure_path)

    lines = [
        "# Canonical non-remapped matching-history and first-decision-source factorial",
        "",
        "## Question",
        "",
        "On prompts with no option remapping, does the later answer depend on candidate-specific reads from the first option lines, on the complete state at the first assistant answer boundary, or on redundant use of both routes?",
        "",
        "## Exact interventions",
        "",
        "- **Matching:** every token of each second-presentation option line was prevented from attending to every token of the identical first-presentation option line at every ordinary-attention layer.",
        "- **Cyclic wrong-line control:** the same four receiver lines were instead denied a cyclically wrong first option line.",
        "- **First-decision source:** every later ordinary-attention query was prevented from reading the exact token at which the first assistant answer would have begun. In Qwen, that token's write into every recurrent GLA layer was also removed.",
        "- **Joint:** matching-line and first-decision routes were removed together.",
        "",
        "The first-decision intervention removes the token's complete outgoing state. It tests whether that position is a causal source, but does **not** by itself prove that any effect is carried specifically by a literal answer-letter coordinate.",
        "The answer letter itself is omitted from the history in this paradigm. The tested source is the final token of the first assistant-generation prefix—the residual state whose logits would have generated that letter (Qwen's final blank-line token, Seed's final `assistant` wordpiece, or Gemma's following newline).",
        "",
        "## Main finding",
        "",
        "The candidate-matched first-option-line route, not the first answer-boundary state, carries the dominant old-winner policy effect on canonical non-remapped prompts. On frozen confirmation, matching-line blockade weakened the Game-versus-Neutral W1-logit difference in all six model/dataset cells. The cyclic wrong-line control was approximately null. In contrast, cutting every outgoing route from the first answer-boundary state was small: it did not reproducibly reduce the choice gap in any cell, and its W1-logit effect was only a small fraction of the natural gap.",
        "",
        "This run alone cannot separate semantic content from displayed letter because those are aligned in a non-remapped prompt. Combined with the prior remapped result—where matching follows candidate meaning rather than the old literal letter—it shows that the semantic matching route is not an artifact created by remapping. The missing first-answer boundary is not the main recollection source.",
        "",
        "## Frozen-confirmation results",
        "",
        "The natural columns give the unedited Neutral-minus-Game old-W1 choice gap and Game-minus-Neutral centered W1 logit. Negative natural logits mean that Game suppresses W1 relative to Neutral. Intervention entries are reductions in those natural effects; positive values mean that the intervention weakened the policy difference.",
        "",
        "| Model | Dataset | Natural choice pp | Natural W1 logit | Matching choice pp | Decision choice pp | Joint choice pp | Matching W1 logit | Decision W1 logit | Joint W1 logit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for dataset in DATASETS:
            cells = summary["results"][model][dataset]["subsets"]["confirmation"]["cells"]
            row = [
                f"{cells['natural']['old_winner_avoidance_gap_pp']:+.2f}",
                f"{cells['natural']['game_minus_neutral_w1_centered_logit']:+.2f}",
            ]
            for cell in ("matching", "first_decision", "matching_plus_first_decision"):
                row.append(
                    _format_ci(
                        cells[cell]["avoidance_gap_reduction_pp"],
                        cells[cell]["ci95"]["avoidance_gap_reduction_pp"],
                    )
                )
            for cell in ("matching", "first_decision", "matching_plus_first_decision"):
                row.append(
                    _format_ci(
                        cells[cell]["w1_logit_gap_reduction"],
                        cells[cell]["ci95"]["w1_logit_gap_reduction"],
                    )
                )
            lines.append(
                f"| {MODEL_LABELS[model]} | {DATASET_LABELS[dataset]} | "
                + " | ".join(row)
                + " |"
            )
    lines += [
        "",
        "## Redundancy check",
        "",
        "The joint cell matters because a null first-decision blockade alone could be hidden by a backup route. Direct joint-minus-matching contrasts show little additional effect in Seed or Gemma. Qwen is the exception at the continuous-logit endpoint: once the matching line route is already cut, also cutting the first boundary pushes W1 farther in the same direction. This is a nonlinear backup/interaction, not evidence that the boundary is the primary route: the boundary cut alone remains approximately null, while matching blockade alone removes the replicated choice effect and most of the logit effect.",
        "",
        "| Model | Dataset | Joint − matching choice pp | Joint − matching W1 logit |",
        "|---|---|---:|---:|",
    ]
    for model in MODELS:
        for dataset in DATASETS:
            contrast = summary["results"][model][dataset]["subsets"]["confirmation"]["contrasts"]["joint_minus_matching"]
            lines.append(
                f"| {MODEL_LABELS[model]} | {DATASET_LABELS[dataset]} | "
                + _format_ci(
                    contrast["avoidance_gap_reduction_pp"],
                    contrast["ci95"]["avoidance_gap_reduction_pp"],
                )
                + " | "
                + _format_ci(
                    contrast["w1_logit_gap_reduction"],
                    contrast["ci95"]["w1_logit_gap_reduction"],
                )
                + " |"
            )
    lines += [
        "",
        "## Validity",
        "",
        "All six runs used 500 questions, reproduced the frozen natural A–D logits exactly, produced finite outputs, and edited the complete architecture-specific ordinary-attention inventory. Qwen's first-decision cells additionally edited all 48 recurrent layers. Discovery/confirmation membership came from the previously frozen dataset split files.",
        "",
        f"Canonical figure: `{figure_path}`",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/model_replications/canonical_history_decision_factorial/analysis",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=root / "figures/model_replications/canonical_history_decision_factorial.png",
    )
    args = parser.parse_args()
    analyze(args.root, args.output_dir, args.figure)


if __name__ == "__main__":
    main()
