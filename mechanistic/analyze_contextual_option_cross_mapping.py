from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mechanistic.analyze_contextual_option_representations import (
    LAMBDAS,
    _balanced_accuracy,
    _center_normalize,
    _folds,
    _labels,
    _load,
    _normalize_queries,
)


LETTERS = "ABCD"
ANCHORS = ("content_end", "line_end")


def _candidate_layer(
    residuals: np.ndarray, metadata: dict, anchor: str, layer: int
) -> np.ndarray:
    indices = [metadata["anchors"].index(f"{anchor}_{letter}") for letter in LETTERS]
    return _center_normalize(np.asarray(residuals[:, layer, indices]).copy())


def _align_remapped_candidates_to_original_content(
    candidates: np.ndarray, metadata: dict, qids: list[str]
) -> np.ndarray:
    """Return candidate axis A-D indexed by original content identity.

    For example, output candidate A is the remapped-prompt residual at whichever
    new letter contains the text that was A in the original prompt.
    """
    indices = np.empty((len(qids), 4), dtype=np.int64)
    for qi, qid in enumerate(qids):
        original_to_new = metadata["mappings"][qid]["original_to_new"]
        indices[qi] = [LETTERS.index(original_to_new[letter]) for letter in LETTERS]
    return np.take_along_axis(candidates, indices[:, :, None], axis=1)


def _content_labels(path: Path, qids: list[str]) -> np.ndarray:
    rows = json.loads(path.read_text())["results"]
    values = []
    for qid in qids:
        value = rows[qid]["answer_original_content"]
        if value not in LETTERS:
            raise ValueError(f"Non-A-D semantic content label for {qid}: {value!r}")
        values.append(LETTERS.index(value))
    return np.asarray(values, dtype=np.int64)


def _balanced_bootstrap_ci(
    prediction: np.ndarray,
    labels: np.ndarray,
    seed: int,
    draws: int = 5000,
) -> list[float]:
    correct = prediction == labels
    rng = np.random.default_rng(seed)
    values = np.zeros(draws, dtype=np.float32)
    groups = [np.flatnonzero(labels == label) for label in range(4)]
    for group in groups:
        sampled = rng.choice(group, size=(draws, len(group)), replace=True)
        values += correct[sampled].mean(axis=1) / 4
    return np.quantile(values, (0.025, 0.975)).tolist()


def _ordinary_accuracy(prediction: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(prediction == labels))


def _cv_scores_fast(
    queries: np.ndarray,
    candidates: np.ndarray,
    labels: np.ndarray,
    folds: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Algebraically equivalent ridge CV without cross-question candidates.

    The original helper forms scores for every validation query against every
    other validation question's candidates and then keeps only the diagonal.
    Here each predicted target vector is dotted directly with its own four
    candidates. On a CUDA host these small ridge systems and large hidden-width
    matrix products run on the already-rented GPU.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = candidates[np.arange(len(labels)), labels].astype(np.float32)
    predictions = np.empty((len(LAMBDAS), len(labels), 4), dtype=np.float32)
    all_indices = np.arange(len(labels))
    for validation in folds:
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        mean = queries[training].astype(np.float32).mean(axis=0)
        x_train = torch.from_numpy(
            _normalize_queries(queries[training], mean)
        ).to(device)
        x_validation = torch.from_numpy(
            _normalize_queries(queries[validation], mean)
        ).to(device)
        target_tensor = torch.from_numpy(targets[training]).to(device)
        candidate_tensor = torch.from_numpy(
            candidates[validation].astype(np.float32)
        ).to(device)
        gram = x_train @ x_train.T
        cross = x_validation @ x_train.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        basis = cross @ eigenvectors
        for li, penalty in enumerate(LAMBDAS):
            coefficients = (
                basis / (eigenvalues[None, :] + float(penalty))
            ) @ eigenvectors.T
            decoded = coefficients @ target_tensor
            scores = torch.einsum("vd,vkd->vk", decoded, candidate_tensor)
            predictions[li, validation] = scores.float().cpu().numpy()
    balanced = np.asarray(
        [_balanced_accuracy(row.argmax(axis=1), labels) for row in predictions]
    )
    return predictions, balanced


def _fit_scores_fast(
    train_queries: np.ndarray,
    train_candidates: np.ndarray,
    train_labels: np.ndarray,
    test_queries: np.ndarray,
    test_candidates: np.ndarray,
    penalty: float,
) -> np.ndarray:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = train_candidates[
        np.arange(len(train_labels)), train_labels
    ].astype(np.float32)
    mean = train_queries.astype(np.float32).mean(axis=0)
    x_train = torch.from_numpy(_normalize_queries(train_queries, mean)).to(device)
    x_test = torch.from_numpy(_normalize_queries(test_queries, mean)).to(device)
    target_tensor = torch.from_numpy(targets).to(device)
    candidate_tensor = torch.from_numpy(
        test_candidates.astype(np.float32)
    ).to(device)
    gram = x_train @ x_train.T
    cross = x_test @ x_train.T
    regularized = gram + float(penalty) * torch.eye(
        len(gram), dtype=gram.dtype, device=device
    )
    coefficients = torch.linalg.solve(regularized, cross.T).T
    decoded = coefficients @ target_tensor
    return torch.einsum("vd,vkd->vk", decoded, candidate_tensor).float().cpu().numpy()


def _plot(summary: dict, output: Path) -> None:
    layers = np.arange(1, 65)
    colors = {
        "original_to_original": "#777777",
        "remapped_decision_to_original_options": "#2f8df3",
        "original_decision_to_remapped_options": "#f07f2f",
    }
    labels = {
        "original_to_original": "Original decision → original options (control)",
        "remapped_decision_to_original_options": "Remapped decision → original options",
        "original_decision_to_remapped_options": "Original decision → remapped options",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    for axis, anchor in zip(axes, ANCHORS):
        result = summary["results"][anchor]
        for key in colors:
            accuracy = np.asarray(result[key]["balanced_accuracy"])
            ci = np.asarray(result[key]["ci"])
            axis.plot(layers, accuracy, color=colors[key], lw=2, label=labels[key])
            axis.fill_between(
                layers, ci[:, 0], ci[:, 1], color=colors[key], alpha=0.12, linewidth=0
            )
        axis.axhline(0.25, color="#444444", ls="--", lw=1, alpha=0.7)
        axis.set_xlim(1, 64)
        axis.set_ylim(0, 1)
        axis.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
        axis.set_xlabel("Residual readout")
        axis.grid(axis="y", color="#dddddd", lw=0.7, alpha=0.7)
    axes[0].set_title("A  Final option-content token", loc="left", fontsize=14)
    axes[1].set_title("B  Option-closing newline", loc="left", fontsize=14)
    axes[0].set_ylabel("Held-out balanced accuracy")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Does the decision identify option content across A–D remapping?",
        y=1.10,
        fontsize=17,
    )
    fig.text(
        0.5,
        -0.02,
        "Shading: paired question-bootstrap 95% confidence intervals. Dashed line: 25% chance.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(output / "layerwise_cross_mapping_match.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(
    discovery_original_root: Path,
    confirmation_original_root: Path,
    confirmation_remapped_root: Path,
    baseline_results: Path,
    remapped_baseline_results: Path,
    output: Path,
    seed: int,
) -> dict:
    discovery, discovery_meta = _load(discovery_original_root)
    original, original_meta = _load(confirmation_original_root)
    remapped, remapped_meta = _load(confirmation_remapped_root)
    discovery_qids = list(discovery_meta["question_ids"])
    confirmation_qids = list(original_meta["question_ids"])
    if confirmation_qids != list(remapped_meta["question_ids"]):
        raise ValueError("Original/remapped confirmation orders differ")

    discovery_labels = _labels(baseline_results, discovery_qids, "answer")
    original_labels = _labels(baseline_results, confirmation_qids, "answer")
    remapped_content_labels = _content_labels(
        remapped_baseline_results, confirmation_qids
    )
    folds = _folds(discovery_labels, 5, seed)
    discovery_decision = discovery_meta["anchors"].index("first_answer_decision")
    original_decision = original_meta["anchors"].index("first_answer_decision")
    remapped_decision = remapped_meta["anchors"].index("first_answer_decision")

    results: dict[str, dict] = {}
    rows: list[dict] = []
    for anchor_index, anchor in enumerate(ANCHORS):
        conditions = {
            key: {"balanced_accuracy": [], "ordinary_accuracy": [], "ci": [], "prediction": []}
            for key in (
                "original_to_original",
                "remapped_decision_to_original_options",
                "original_decision_to_remapped_options",
            )
        }
        result: dict = {
            "discovery_cv_balanced_accuracy": [],
            "selected_lambda": [],
            **conditions,
        }
        for layer in range(64):
            discovery_candidates = _candidate_layer(
                discovery, discovery_meta, anchor, layer
            )
            original_candidates = _candidate_layer(
                original, original_meta, anchor, layer
            )
            remapped_candidates = _candidate_layer(
                remapped, remapped_meta, anchor, layer
            )
            remapped_candidates_by_content = _align_remapped_candidates_to_original_content(
                remapped_candidates, remapped_meta, confirmation_qids
            )
            discovery_query = np.asarray(
                discovery[:, layer, discovery_decision]
            ).copy()
            original_query = np.asarray(original[:, layer, original_decision]).copy()
            remapped_query = np.asarray(remapped[:, layer, remapped_decision]).copy()

            _, cv_by_lambda = _cv_scores_fast(
                discovery_query,
                discovery_candidates,
                discovery_labels,
                folds,
            )
            selected = int(np.argmax(cv_by_lambda))
            penalty = float(LAMBDAS[selected])
            score_sets = {
                "original_to_original": _fit_scores_fast(
                    discovery_query,
                    discovery_candidates,
                    discovery_labels,
                    original_query,
                    original_candidates,
                    penalty,
                ),
                "remapped_decision_to_original_options": _fit_scores_fast(
                    discovery_query,
                    discovery_candidates,
                    discovery_labels,
                    remapped_query,
                    original_candidates,
                    penalty,
                ),
                "original_decision_to_remapped_options": _fit_scores_fast(
                    discovery_query,
                    discovery_candidates,
                    discovery_labels,
                    original_query,
                    remapped_candidates_by_content,
                    penalty,
                ),
            }
            targets = {
                "original_to_original": original_labels,
                "remapped_decision_to_original_options": remapped_content_labels,
                "original_decision_to_remapped_options": original_labels,
            }
            result["discovery_cv_balanced_accuracy"].append(float(cv_by_lambda[selected]))
            result["selected_lambda"].append(penalty)
            for condition_index, (condition, scores) in enumerate(score_sets.items()):
                prediction = scores.argmax(axis=1)
                target = targets[condition]
                balanced = _balanced_accuracy(prediction, target)
                ordinary = _ordinary_accuracy(prediction, target)
                ci = _balanced_bootstrap_ci(
                    prediction,
                    target,
                    seed + 10000 * anchor_index + 1000 * condition_index + layer,
                )
                result[condition]["balanced_accuracy"].append(balanced)
                result[condition]["ordinary_accuracy"].append(ordinary)
                result[condition]["ci"].append(ci)
                result[condition]["prediction"].append(prediction.tolist())
                rows.append(
                    {
                        "anchor": anchor,
                        "layer": layer + 1,
                        "condition": condition,
                        "selected_lambda": penalty,
                        "discovery_cv_balanced_accuracy": float(cv_by_lambda[selected]),
                        "confirmation_balanced_accuracy": balanced,
                        "confirmation_ordinary_accuracy": ordinary,
                        "confirmation_ci_low": ci[0],
                        "confirmation_ci_high": ci[1],
                    }
                )
            print(
                f"cross-mapping {anchor} {layer + 1}/64: "
                f"control={result['original_to_original']['balanced_accuracy'][-1]:.3f} "
                f"remapped-query/original-options="
                f"{result['remapped_decision_to_original_options']['balanced_accuracy'][-1]:.3f} "
                f"original-query/remapped-options="
                f"{result['original_decision_to_remapped_options']['balanced_accuracy'][-1]:.3f}",
                flush=True,
            )
        results[anchor] = result

    summary = {
        "definition": (
            "A ridge matcher is trained only on original-mapping discovery questions. "
            "At each layer it maps the first-answer decision residual to the selected "
            "option residual. The primary held-out test scores remapped-prompt decision "
            "residuals against original-prompt candidates and labels the target by "
            "original semantic content identity. The reverse test scores original-prompt "
            "decisions against remapped candidates aligned by content. Neither cross-mapping "
            "test can be solved by matching the current decision letter to the same absolute "
            "option position."
        ),
        "split": {"discovery": len(discovery_qids), "confirmation": len(confirmation_qids)},
        "chance": 0.25,
        "analysis_backend": "torch CUDA when available; otherwise torch CPU",
        "anchors": {
            "content_end": "Last token overlapping the option text.",
            "line_end": "Identical newline token closing the option line.",
        },
        "results": results,
        "confirmation_original_labels": original_labels.tolist(),
        "confirmation_remapped_content_labels": remapped_content_labels.tolist(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "layerwise_cross_mapping_match.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (output / "layerwise_cross_mapping_match.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-original", type=Path, required=True)
    parser.add_argument("--confirmation-original", type=Path, required=True)
    parser.add_argument("--confirmation-remapped", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--remapped-baseline-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    analyze(
        args.discovery_original,
        args.confirmation_original,
        args.confirmation_remapped,
        args.baseline_results,
        args.remapped_baseline_results,
        args.output,
        args.seed,
    )


if __name__ == "__main__":
    main()
