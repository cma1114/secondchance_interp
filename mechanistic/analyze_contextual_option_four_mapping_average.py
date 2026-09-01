from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mechanistic.analyze_contextual_option_cross_mapping import (
    _balanced_bootstrap_ci,
    _candidate_layer,
    _fit_scores_fast,
    _cv_scores_fast,
    _ordinary_accuracy,
)
from mechanistic.analyze_contextual_option_representations import (
    LAMBDAS,
    _balanced_accuracy,
    _center_normalize,
    _folds,
    _labels,
    _load,
)


LETTERS = "ABCD"
ANCHORS = ("content_end", "line_end")


def _raw_candidates(
    residuals: np.ndarray, metadata: dict, anchor: str, layer: int
) -> np.ndarray:
    indices = [metadata["anchors"].index(f"{anchor}_{letter}") for letter in LETTERS]
    return np.asarray(residuals[:, layer, indices], dtype=np.float32).copy()


def _align_to_original_content(
    candidates: np.ndarray, metadata: dict, qids: list[str]
) -> np.ndarray:
    if not metadata.get("mappings"):
        return candidates
    indices = np.empty((len(qids), 4), dtype=np.int64)
    for qi, qid in enumerate(qids):
        original_to_new = metadata["mappings"][qid]["original_to_new"]
        indices[qi] = [LETTERS.index(original_to_new[letter]) for letter in LETTERS]
    return np.take_along_axis(candidates, indices[:, :, None], axis=1)


def _averaged_candidates(
    sets: list[tuple[np.ndarray, dict]],
    qids: list[str],
    anchor: str,
    layer: int,
) -> np.ndarray:
    aligned = [
        _align_to_original_content(_raw_candidates(values, meta, anchor, layer), meta, qids)
        for values, meta in sets
    ]
    # Average raw residuals first, then remove the question-common component and
    # normalize. Equal weighting makes every semantic option occupy each letter
    # exactly once, cancelling any additive absolute-position code.
    return _center_normalize(np.mean(aligned, axis=0))


def _evaluate_matcher(
    discovery_query: np.ndarray,
    discovery_candidates: np.ndarray,
    discovery_labels: np.ndarray,
    confirmation_query: np.ndarray,
    confirmation_candidates: np.ndarray,
    confirmation_labels: np.ndarray,
    folds: list[np.ndarray],
    seed: int,
) -> dict:
    _, cv = _cv_scores_fast(
        discovery_query, discovery_candidates, discovery_labels, folds
    )
    selected = int(np.argmax(cv))
    scores = _fit_scores_fast(
        discovery_query,
        discovery_candidates,
        discovery_labels,
        confirmation_query,
        confirmation_candidates,
        float(LAMBDAS[selected]),
    )
    prediction = scores.argmax(axis=1)
    return {
        "selected_lambda": float(LAMBDAS[selected]),
        "discovery_cv_balanced_accuracy": float(cv[selected]),
        "confirmation_balanced_accuracy": _balanced_accuracy(
            prediction, confirmation_labels
        ),
        "confirmation_ordinary_accuracy": _ordinary_accuracy(
            prediction, confirmation_labels
        ),
        "confirmation_ci": _balanced_bootstrap_ci(
            prediction, confirmation_labels, seed
        ),
        "prediction": prediction.tolist(),
    }


def _plot(summary: dict, output: Path) -> None:
    layers = np.arange(1, 65)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), sharey=True)
    styles = {
        "same_prompt_control": ("#777777", "Same-prompt candidates (control)"),
        "four_mapping_average": ("#2f8df3", "Candidates averaged across A–D positions"),
        "shuffled_average_null": ("#b65fcf", "Averaged candidates with content labels shuffled"),
    }
    for axis, anchor in zip(axes, ANCHORS):
        for key, (color, label) in styles.items():
            values = np.asarray(
                [row["confirmation_balanced_accuracy"] for row in summary["results"][anchor][key]]
            )
            ci = np.asarray(
                [row["confirmation_ci"] for row in summary["results"][anchor][key]]
            )
            axis.plot(layers, values, lw=2.2, color=color, label=label)
            axis.fill_between(layers, ci[:, 0], ci[:, 1], color=color, alpha=0.14)
        axis.axhline(0.25, color="#444444", ls="--", lw=1)
        axis.set_xlim(1, 64)
        axis.set_ylim(0, 1)
        axis.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
        axis.set_xlabel("Residual readout")
        axis.grid(axis="y", color="#dddddd", lw=0.7, alpha=0.7)
    axes[0].set_title("A  Final option-content token", loc="left", fontsize=14)
    axes[1].set_title("B  Option-closing newline", loc="left", fontsize=14)
    axes[0].set_ylabel("Held-out balanced accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.suptitle("Does averaging every option across A–D reveal answer content?", y=1.095, fontsize=17)
    fig.text(
        0.5,
        -0.025,
        "Shading: paired question-bootstrap 95% confidence intervals. Dashed line: 25% chance.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(output / "layerwise_four_mapping_average_match.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(
    discovery_roots: list[Path],
    confirmation_roots: list[Path],
    baseline_results: Path,
    output: Path,
    seed: int,
) -> dict:
    if len(discovery_roots) != 4 or len(confirmation_roots) != 4:
        raise ValueError("Exactly four discovery and four confirmation mappings are required")
    discovery_sets = [_load(path) for path in discovery_roots]
    confirmation_sets = [_load(path) for path in confirmation_roots]
    discovery_qids = list(discovery_sets[0][1]["question_ids"])
    confirmation_qids = list(confirmation_sets[0][1]["question_ids"])
    for _, meta in discovery_sets[1:]:
        if list(meta["question_ids"]) != discovery_qids:
            raise ValueError("Discovery question order differs across mappings")
    for _, meta in confirmation_sets[1:]:
        if list(meta["question_ids"]) != confirmation_qids:
            raise ValueError("Confirmation question order differs across mappings")

    # Verify the crucial design property rather than trusting the plan files.
    for sets, qids in ((discovery_sets, discovery_qids), (confirmation_sets, confirmation_qids)):
        for qid in qids:
            for original in LETTERS:
                positions = []
                for _, meta in sets:
                    positions.append(
                        original if not meta.get("mappings")
                        else meta["mappings"][qid]["original_to_new"][original]
                    )
                if set(positions) != set(LETTERS):
                    raise ValueError(f"{qid}: {original} does not occupy A-D exactly once")

    discovery_labels = _labels(baseline_results, discovery_qids, "answer")
    confirmation_labels = _labels(baseline_results, confirmation_qids, "answer")
    folds = _folds(discovery_labels, 5, seed)
    rng = np.random.default_rng(seed + 900001)
    # A nonzero cyclic shift independently assigned to every question. This
    # retains the exact same averaged vectors and their A-D balance but breaks
    # which vector is called the answer's semantic content.
    discovery_shift = rng.integers(1, 4, size=len(discovery_qids))
    confirmation_shift = rng.integers(1, 4, size=len(confirmation_qids))
    discovery_permutation = (
        np.arange(4)[None, :] + discovery_shift[:, None]
    ) % 4
    confirmation_permutation = (
        np.arange(4)[None, :] + confirmation_shift[:, None]
    ) % 4
    discovery_decision = discovery_sets[0][1]["anchors"].index("first_answer_decision")
    confirmation_decision = confirmation_sets[0][1]["anchors"].index("first_answer_decision")
    results: dict[str, dict] = {}
    rows: list[dict] = []

    for anchor_index, anchor in enumerate(ANCHORS):
        anchor_results = {
            "same_prompt_control": [],
            "four_mapping_average": [],
            "shuffled_average_null": [],
        }
        for layer in range(64):
            discovery_query = np.asarray(
                discovery_sets[0][0][:, layer, discovery_decision], dtype=np.float32
            ).copy()
            confirmation_query = np.asarray(
                confirmation_sets[0][0][:, layer, confirmation_decision], dtype=np.float32
            ).copy()
            discovery_control = _candidate_layer(
                discovery_sets[0][0], discovery_sets[0][1], anchor, layer
            )
            confirmation_control = _candidate_layer(
                confirmation_sets[0][0], confirmation_sets[0][1], anchor, layer
            )
            discovery_average = _averaged_candidates(
                discovery_sets, discovery_qids, anchor, layer
            )
            confirmation_average = _averaged_candidates(
                confirmation_sets, confirmation_qids, anchor, layer
            )
            discovery_shuffled = np.take_along_axis(
                discovery_average,
                discovery_permutation[:, :, None],
                axis=1,
            )
            confirmation_shuffled = np.take_along_axis(
                confirmation_average,
                confirmation_permutation[:, :, None],
                axis=1,
            )
            for condition_index, (condition, d_candidates, c_candidates) in enumerate(
                (
                    ("same_prompt_control", discovery_control, confirmation_control),
                    ("four_mapping_average", discovery_average, confirmation_average),
                    ("shuffled_average_null", discovery_shuffled, confirmation_shuffled),
                )
            ):
                result = _evaluate_matcher(
                    discovery_query,
                    d_candidates,
                    discovery_labels,
                    confirmation_query,
                    c_candidates,
                    confirmation_labels,
                    folds,
                    seed + 10000 * anchor_index + 1000 * condition_index + layer,
                )
                anchor_results[condition].append(result)
                rows.append(
                    {
                        "anchor": anchor,
                        "layer": layer + 1,
                        "condition": condition,
                        **{key: value for key, value in result.items() if key != "prediction"},
                    }
                )
            print(
                f"four-map average {anchor} {layer + 1}/64: "
                f"control={anchor_results['same_prompt_control'][-1]['confirmation_balanced_accuracy']:.3f} "
                f"average={anchor_results['four_mapping_average'][-1]['confirmation_balanced_accuracy']:.3f} "
                f"shuffled={anchor_results['shuffled_average_null'][-1]['confirmation_balanced_accuracy']:.3f}",
                flush=True,
            )
        results[anchor] = anchor_results

    summary = {
        "definition": (
            "For every question and semantic option, raw contextual residuals are "
            "aligned by content and averaged across four prompt mappings in which that "
            "content occupies A, B, C, and D exactly once. After within-question "
            "centering and normalization, a ridge matcher maps the original Baseline "
            "decision residual to the Baseline-selected averaged option representation."
        ),
        "split": {"discovery": len(discovery_qids), "confirmation": len(confirmation_qids)},
        "chance": 0.25,
        "shuffled_null": (
            "For every question, the four correctly averaged candidates receive an "
            "independent nonzero cyclic permutation before fitting and evaluation. "
            "This preserves the vectors and exact A-D averaging while destroying "
            "their semantic-content labels."
        ),
        "results": results,
        "confirmation_labels": confirmation_labels.tolist(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "layerwise_four_mapping_average_match.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (output / "layerwise_four_mapping_average_match.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-roots", nargs=4, type=Path, required=True)
    parser.add_argument("--confirmation-roots", nargs=4, type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    analyze(
        args.discovery_roots,
        args.confirmation_roots,
        args.baseline_results,
        args.output,
        args.seed,
    )


if __name__ == "__main__":
    main()
