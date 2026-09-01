from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


LETTERS = "ABCD"


def _bootstrap_interval(
    values: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Question-bootstrap mean interval for a [layer, question] matrix."""
    sampled = values[:, indices].mean(axis=2)
    return np.quantile(sampled, 0.025, axis=1), np.quantile(sampled, 0.975, axis=1)


def _score_layer(
    train: np.ndarray,
    train_choice: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit an isotropic linear ranker after removing static letter means."""
    n_train, _, width = train.shape
    letter_mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = train - letter_mean[None, :, :]
    scale = centered.reshape(-1, width).std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-3)
    standardized = centered / scale[None, None, :]
    rows = np.arange(n_train)
    chosen = standardized[rows, train_choice]
    unchosen = (standardized.sum(axis=1) - chosen) / 3.0
    weight = (chosen - unchosen).mean(axis=0, dtype=np.float64).astype(np.float32)
    weight /= max(float(np.linalg.norm(weight)), 1e-12)
    test_standardized = (test - letter_mean[None, :, :]) / scale[None, None, :]
    scores = np.einsum("pod,d->po", test_standardized, weight, optimize=True)
    return scores.astype(np.float32), weight, letter_mean, scale


def analyze(
    cache_dir: Path,
    screen_results_path: Path,
    discovery_plan_path: Path,
    eligible_pairs_path: Path,
    output_dir: Path,
    figure_path: Path,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    residuals = np.load(cache_dir / "option_newline_residuals.npy", mmap_mode="r")
    with np.load(cache_dir / "results.npz", allow_pickle=False) as loaded:
        qids = loaded["question_ids"].astype(str).tolist()
        logits = loaded["aggregated_ad_logits"].astype(np.float32)
        completed = loaded["completed"].astype(bool)
    if not completed.all():
        raise ValueError("Residual collection is incomplete")
    with np.load(screen_results_path, allow_pickle=False) as loaded:
        screen_qids = loaded["question_ids"].astype(str).tolist()
        screen_logits = loaded["aggregated_ad_logits"].astype(np.float32)
    if screen_qids != qids:
        raise ValueError("Screen and residual-cache question order differ")
    current_choice = logits.argmax(axis=-1)
    screen_choice = screen_logits.argmax(axis=-1)
    choice_agreement = float(np.mean(current_choice == screen_choice))
    max_logit_error = float(np.max(np.abs(logits - screen_logits)))

    discovery_payload = json.loads(discovery_plan_path.read_text())
    discovery_ids = set(
        discovery_payload.get(
            "question_ids", discovery_payload.get("discovery_question_ids", [])
        )
    )
    if not discovery_ids:
        raise ValueError("Discovery plan has no question IDs")
    train_indices = np.asarray(
        [index for index, qid in enumerate(qids) if qid in discovery_ids], dtype=int
    )
    test_indices = np.asarray(
        [index for index, qid in enumerate(qids) if qid not in discovery_ids], dtype=int
    )
    if len(train_indices) != 251 or len(test_indices) != 249:
        raise ValueError(
            f"Expected 251/249 question split, got {len(train_indices)}/{len(test_indices)}"
        )

    n_layers = residuals.shape[2]
    width = residuals.shape[-1]
    n_test_presentations = 6 * len(test_indices)
    train_choice = current_choice[:, train_indices].reshape(-1)
    test_choice = current_choice[:, test_indices].reshape(-1)
    test_qids = [qids[index] for index in test_indices]
    test_qid_to_index = {qid: index for index, qid in enumerate(test_qids)}

    weights = np.empty((n_layers, width), dtype=np.float32)
    letter_means = np.empty((n_layers, 4, width), dtype=np.float32)
    scales = np.empty((n_layers, width), dtype=np.float32)
    heldout_scores = np.empty(
        (n_layers, 6, len(test_indices), 4), dtype=np.float32
    )
    accuracy_by_question = np.empty((n_layers, len(test_indices)), dtype=np.float32)
    margin_by_question = np.empty((n_layers, len(test_indices)), dtype=np.float32)

    for layer in range(n_layers):
        train = np.asarray(
            residuals[:, train_indices, layer, :, :], dtype=np.float32
        ).reshape(-1, 4, width)
        test = np.asarray(
            residuals[:, test_indices, layer, :, :], dtype=np.float32
        ).reshape(-1, 4, width)
        scores, weight, letter_mean, scale = _score_layer(
            train, train_choice, test
        )
        weights[layer] = weight
        letter_means[layer] = letter_mean
        scales[layer] = scale
        heldout_scores[layer] = scores.reshape(6, len(test_indices), 4)
        prediction = scores.argmax(axis=-1)
        accuracy_by_question[layer] = (
            (prediction == test_choice).reshape(6, len(test_indices)).mean(axis=0)
        )
        rows = np.arange(n_test_presentations)
        chosen_score = scores[rows, test_choice]
        masked = scores.copy()
        masked[rows, test_choice] = -np.inf
        best_other = masked.max(axis=-1)
        margin_by_question[layer] = (
            (chosen_score - best_other).reshape(6, len(test_indices)).mean(axis=0)
        )
        print(f"option-newline ranker: layer {layer + 1}/{n_layers}", flush=True)

    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0, len(test_indices), size=(draws, len(test_indices))
    )
    accuracy_mean = accuracy_by_question.mean(axis=1)
    accuracy_low, accuracy_high = _bootstrap_interval(
        accuracy_by_question, bootstrap_indices
    )
    margin_mean = margin_by_question.mean(axis=1)
    margin_low, margin_high = _bootstrap_interval(
        margin_by_question, bootstrap_indices
    )

    train_counts = np.bincount(train_choice, minlength=4)
    majority_letter = int(train_counts.argmax())
    letter_only_by_question = (
        (current_choice[:, test_indices] == majority_letter).mean(axis=0)
    )
    letter_only_accuracy = float(letter_only_by_question.mean())
    letter_boot = letter_only_by_question[bootstrap_indices].mean(axis=1)
    letter_only_ci = np.quantile(letter_boot, [0.025, 0.975]).tolist()

    eligible_payload = json.loads(eligible_pairs_path.read_text())
    pair_rows = []
    for row in eligible_payload["rows"]:
        qid = row["question_id"]
        if qid not in test_qid_to_index:
            continue
        qi = test_qid_to_index[qid]
        letter = LETTERS.index(row["w1_displayed_letter"])
        chosen_mapping = int(row["chosen_mapping_index"])
        unchosen_mapping = int(row["unchosen_mapping_index"])
        if current_choice[chosen_mapping, test_indices[qi]] != letter:
            continue
        if current_choice[unchosen_mapping, test_indices[qi]] == letter:
            continue
        pair_rows.append((qid, qi, letter, chosen_mapping, unchosen_mapping))

    pair_target_difference = np.empty((n_layers, len(pair_rows)), dtype=np.float32)
    pair_competitor_difference = np.empty_like(pair_target_difference)
    pair_letters = np.asarray([row[2] for row in pair_rows], dtype=int)
    for layer in range(n_layers):
        scores = heldout_scores[layer]
        for pi, (_qid, qi, letter, chosen_mapping, unchosen_mapping) in enumerate(
            pair_rows
        ):
            others = [index for index in range(4) if index != letter]
            pair_target_difference[layer, pi] = (
                scores[chosen_mapping, qi, letter]
                - scores[unchosen_mapping, qi, letter]
            )
            pair_competitor_difference[layer, pi] = (
                scores[unchosen_mapping, qi, others].max()
                - scores[chosen_mapping, qi, others].max()
            )

    pair_bootstrap = rng.integers(
        0, len(pair_rows), size=(draws, len(pair_rows))
    )
    pair_target_mean = pair_target_difference.mean(axis=1)
    pair_target_low, pair_target_high = _bootstrap_interval(
        pair_target_difference, pair_bootstrap
    )
    pair_competitor_mean = pair_competitor_difference.mean(axis=1)
    pair_competitor_low, pair_competitor_high = _bootstrap_interval(
        pair_competitor_difference, pair_bootstrap
    )

    by_letter: dict[str, Any] = {}
    for letter_index, letter in enumerate(LETTERS):
        mask = pair_letters == letter_index
        if not np.any(mask):
            continue
        values = pair_target_difference[:, mask]
        letter_indices = rng.integers(0, int(mask.sum()), size=(draws, int(mask.sum())))
        low, high = _bootstrap_interval(values, letter_indices)
        by_letter[letter] = {
            "n": int(mask.sum()),
            "mean": values.mean(axis=1).tolist(),
            "ci_low": low.tolist(),
            "ci_high": high.tolist(),
        }

    best_layer = int(np.argmax(accuracy_mean))
    summary = {
        "definition": (
            "At each post-block readout, a shared linear ranker scores the four "
            "option-closing-newline residuals. Static displayed-letter means are "
            "estimated on discovery questions and removed before fitting."
        ),
        "train_questions": len(train_indices),
        "heldout_questions": len(test_indices),
        "presentations_per_question": 6,
        "heldout_presentations": n_test_presentations,
        "screen_choice_agreement": choice_agreement,
        "screen_max_abs_logit_error": max_logit_error,
        "letter_only_baseline": {
            "majority_letter": LETTERS[majority_letter],
            "accuracy": letter_only_accuracy,
            "ci": letter_only_ci,
        },
        "best_descriptive_heldout_layer": best_layer + 1,
        "best_descriptive_heldout_accuracy": float(accuracy_mean[best_layer]),
        "best_descriptive_heldout_accuracy_ci": [
            float(accuracy_low[best_layer]),
            float(accuracy_high[best_layer]),
        ],
        "final_layer_accuracy": float(accuracy_mean[-1]),
        "final_layer_accuracy_ci": [float(accuracy_low[-1]), float(accuracy_high[-1])],
        "matched_sensitive_pairs": len(pair_rows),
        "matched_sensitive_pairs_by_letter": {
            letter: int((pair_letters == index).sum())
            for index, letter in enumerate(LETTERS)
        },
        "interpretive_scope": (
            "Held-out rank decoding establishes that candidate-value information is "
            "linearly available at the option-closing newline. Matched same-content "
            "differences test whether that local value itself changes with ordering; "
            "they do not by themselves establish causal use."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "probe_results.npz",
        layers=np.arange(1, n_layers + 1),
        weights=weights,
        letter_means=letter_means,
        scales=scales,
        heldout_scores=heldout_scores.astype(np.float16),
        accuracy_by_question=accuracy_by_question,
        accuracy_mean=accuracy_mean,
        accuracy_ci_low=accuracy_low,
        accuracy_ci_high=accuracy_high,
        margin_mean=margin_mean,
        margin_ci_low=margin_low,
        margin_ci_high=margin_high,
        pair_target_difference=pair_target_difference,
        pair_target_mean=pair_target_mean,
        pair_target_ci_low=pair_target_low,
        pair_target_ci_high=pair_target_high,
        pair_competitor_mean=pair_competitor_mean,
        pair_competitor_ci_low=pair_competitor_low,
        pair_competitor_ci_high=pair_competitor_high,
        pair_letters=pair_letters,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    with (output_dir / "layerwise_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "readout",
                "heldout_accuracy",
                "accuracy_ci_low",
                "accuracy_ci_high",
                "chosen_minus_best_other_score",
                "margin_ci_low",
                "margin_ci_high",
                "matched_w1_score_chosen_minus_unchosen",
                "matched_w1_ci_low",
                "matched_w1_ci_high",
                "matched_competitor_score_unchosen_minus_chosen",
                "matched_competitor_ci_low",
                "matched_competitor_ci_high",
            ]
        )
        for layer in range(n_layers):
            writer.writerow(
                [
                    layer + 1,
                    accuracy_mean[layer],
                    accuracy_low[layer],
                    accuracy_high[layer],
                    margin_mean[layer],
                    margin_low[layer],
                    margin_high[layer],
                    pair_target_mean[layer],
                    pair_target_low[layer],
                    pair_target_high[layer],
                    pair_competitor_mean[layer],
                    pair_competitor_low[layer],
                    pair_competitor_high[layer],
                ]
            )

    _plot(
        figure_path,
        accuracy_mean,
        accuracy_low,
        accuracy_high,
        letter_only_accuracy,
        margin_mean,
        margin_low,
        margin_high,
        pair_target_mean,
        pair_target_low,
        pair_target_high,
        by_letter,
    )
    _write_report(output_dir / "REPORT.md", summary, figure_path, by_letter)
    return summary


def _plot(
    output: Path,
    accuracy: np.ndarray,
    accuracy_low: np.ndarray,
    accuracy_high: np.ndarray,
    letter_baseline: float,
    margin: np.ndarray,
    margin_low: np.ndarray,
    margin_high: np.ndarray,
    pair_target: np.ndarray,
    pair_target_low: np.ndarray,
    pair_target_high: np.ndarray,
    by_letter: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    layers = np.arange(1, len(accuracy) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    axes[0].plot(layers, accuracy * 100, color="#2F8EF4", linewidth=2)
    axes[0].fill_between(
        layers, accuracy_low * 100, accuracy_high * 100, color="#2F8EF4", alpha=0.18
    )
    axes[0].axhline(25, color="#999999", linestyle=":", label="Chance")
    axes[0].axhline(
        letter_baseline * 100,
        color="#F28A35",
        linestyle="--",
        label="Letter-only",
    )
    axes[0].set_title("A  Which option will the model choose?", loc="left", fontweight="bold")
    axes[0].set_ylabel("Held-out top-1 accuracy (%)")
    axes[0].legend(frameon=False)

    axes[1].plot(layers, margin, color="#22A06B", linewidth=2)
    axes[1].fill_between(layers, margin_low, margin_high, color="#22A06B", alpha=0.18)
    axes[1].axhline(0, color="#999999", linestyle="--")
    axes[1].set_title("B  Chosen option versus best alternative", loc="left", fontweight="bold")
    axes[1].set_ylabel("Probe-score margin")

    axes[2].plot(
        layers,
        pair_target,
        color="#7A4FB7",
        linewidth=2.4,
        label="All matched pairs",
    )
    axes[2].fill_between(
        layers, pair_target_low, pair_target_high, color="#7A4FB7", alpha=0.18
    )
    colors = {"A": "#2F8EF4", "B": "#F28A35", "C": "#22A06B", "D": "#D95F9F"}
    for letter in LETTERS:
        if letter not in by_letter:
            continue
        axes[2].plot(
            layers,
            np.asarray(by_letter[letter]["mean"]),
            color=colors[letter],
            linewidth=1.25,
            alpha=0.85,
            label=f"W1={letter} (n={by_letter[letter]['n']})",
        )
    axes[2].axhline(0, color="#999999", linestyle="--")
    axes[2].set_title("C  Same W1: chosen versus unchosen", loc="left", fontweight="bold")
    axes[2].set_ylabel("W1 probe-score difference")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xlabel("Post-block residual readout")
        axis.set_xlim(1, len(layers))
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle(
        "Does the option-newline residual carry a candidate-value signal?",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    path: Path, summary: dict[str, Any], figure: Path, by_letter: dict[str, Any]
) -> None:
    lines = [
        "# Option-newline selected-answer probe",
        "",
        "## Design",
        "",
        "At every post-block readout, a single linear ranker scores the four residuals at the exact option-closing newline positions previously used to construct content-aligned option directions. It is trained on 251 discovery questions across all six option permutations and tested on 249 entirely held-out questions. Static A-D position means are removed using discovery data. This establishes candidate-value decodability at the newline, not a clean mapping-invariant semantic code.",
        "",
        "## Results",
        "",
        f"The collector reproduced the prior six-permutation choices on {summary['screen_choice_agreement']*100:.2f}% of presentations (maximum A-D logit difference {summary['screen_max_abs_logit_error']:.4g}).",
        "",
        f"The letter-only Baseline chose {summary['letter_only_baseline']['majority_letter']} and achieved {summary['letter_only_baseline']['accuracy']*100:.1f}% [{summary['letter_only_baseline']['ci'][0]*100:.1f}, {summary['letter_only_baseline']['ci'][1]*100:.1f}] on held-out questions.",
        f"The strongest descriptive held-out probe readout was {summary['best_descriptive_heldout_layer']} with {summary['best_descriptive_heldout_accuracy']*100:.1f}% [{summary['best_descriptive_heldout_accuracy_ci'][0]*100:.1f}, {summary['best_descriptive_heldout_accuracy_ci'][1]*100:.1f}] top-1 accuracy. Final-readout accuracy was {summary['final_layer_accuracy']*100:.1f}% [{summary['final_layer_accuracy_ci'][0]*100:.1f}, {summary['final_layer_accuracy_ci'][1]*100:.1f}].",
        "The best readout is a descriptive maximum selected from the held-out layer trajectory, not a separately preregistered confirmatory layer. The layer curve is broad around the maximum, and the final-readout estimate is reported alongside it to expose the limited optimization gain.",
        "",
        f"The exact matched selectedness analysis retained {summary['matched_sensitive_pairs']} held-out same-content/same-letter pairs: "
        + ", ".join(
            f"{letter}={count}"
            for letter, count in summary["matched_sensitive_pairs_by_letter"].items()
        )
        + ".",
        "",
        "Panel C asks the particularly strict question: does the score attached to the same W1 option at the same displayed letter change when distractor ordering makes it win versus lose? W1=A is a built-in causal-prefix sanity check: its option-line residual is token-for-token identical before any later distractor is seen, so any A difference should be exactly zero.",
        "",
        "This is a correlational decoding test. Strong held-out accuracy establishes a linearly readable candidate-value signal at the option-closing newline; it does not establish mapping-invariant semantic content or that the model causally uses the fitted direction.",
        "",
        f"Canonical figure: `{figure}`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--screen-results", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--eligible-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(
        args.cache_dir,
        args.screen_results,
        args.discovery_plan,
        args.eligible_pairs,
        args.output_dir,
        args.figure,
        args.draws,
        args.seed,
    )


if __name__ == "__main__":
    main()
