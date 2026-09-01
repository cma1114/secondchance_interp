from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .answer_emergence_figures import (
    RANK_COLORS,
    RANK_LABELS,
    Z_975,
    macro_mean_and_se,
)
from .data import load_activation_dataset
from .io import shard_path
from .probes import stratified_folds


CONDITIONS = ("baseline", "incorrect", "neutral")
DISPLAY_NAMES = {"incorrect": "Second Chance", "neutral": "Neutral"}


def load_selected_residuals(
    input_dir: str | Path,
    question_ids: list[str],
    layers: np.ndarray,
) -> np.ndarray:
    with np.load(shard_path(input_dir, "baseline", question_ids[0]), allow_pickle=False) as shard:
        width = shard["residuals"].shape[-1]
    values = np.empty((len(CONDITIONS), len(question_ids), len(layers), width), dtype=np.float16)
    for ci, condition in enumerate(CONDITIONS):
        for qi, question_id in enumerate(question_ids):
            with np.load(shard_path(input_dir, condition, question_id), allow_pickle=False) as shard:
                values[ci, qi] = shard["residuals"][layers]
    return values


def centroid_candidate_scores(
    residuals: np.ndarray,
    winner_labels: np.ndarray,
    folds: int,
    seed: int,
) -> np.ndarray:
    """Cross-fitted candidate evidence from a baseline-winner centroid probe.

    Each question in every condition is scored by a probe trained without that
    question. The probe is trained only on baseline residuals and baseline final
    winner identity. Scores are centered within question and standardized by the
    held-out baseline score dispersion at each layer.
    """
    n_conditions, n_questions, n_layers, _ = residuals.shape
    scores = np.empty((n_conditions, n_questions, n_layers, 4), dtype=np.float32)
    split = stratified_folds(winner_labels, folds, seed)
    all_indices = np.arange(n_questions)

    for li in range(n_layers):
        for test in split:
            train = np.setdiff1d(all_indices, test, assume_unique=True)
            x_train = residuals[0, train, li].astype(np.float32)
            mean = x_train.mean(axis=0)
            scale = x_train.std(axis=0)
            scale[scale < 1e-6] = 1.0
            standardized_train = (x_train - mean) / scale
            centers = np.stack([
                standardized_train[winner_labels[train] == letter].mean(axis=0)
                for letter in range(4)
            ])
            for ci in range(n_conditions):
                x_test = (residuals[ci, test, li].astype(np.float32) - mean) / scale
                scores[ci, test, li] = x_test @ centers.T

        scores[:, :, li] -= scores[:, :, li].mean(axis=-1, keepdims=True)
        baseline_sd = scores[0, :, li].reshape(-1).std(ddof=1)
        if baseline_sd > 1e-8:
            scores[:, :, li] /= baseline_sd
        else:
            scores[:, :, li] = 0.0
    return scores


def align_options(values: np.ndarray, baseline_order: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, baseline_order[:, None, :], axis=-1)


def summarize_subset(
    aligned: np.ndarray,
    baseline_order: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_layers = aligned.shape[1]
    means = np.empty((n_layers, 4))
    halfwidths = np.empty((n_layers, 4))
    for rank in range(4):
        values = aligned[mask, :, rank]
        strata = baseline_order[mask, rank]
        # Letter-balanced summaries are preferable, but a rare switch subset can
        # contain only one example of a particular answer letter.  In that case
        # the within-letter sampling variance is undefined.  Keep the ancillary
        # switch-status figure available by falling back to the ordinary
        # question-weighted mean and SE for that rank; the full-sample figures
        # continue to use the prespecified letter-balanced estimand.
        if np.all(np.bincount(strata, minlength=4) >= 2):
            mean, se = macro_mean_and_se(values, strata)
        else:
            mean = values.mean(axis=0)
            se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
        means[:, rank] = mean
        halfwidths[:, rank] = Z_975 * se
    return means, halfwidths


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _panel(ax, layers: np.ndarray, means: np.ndarray, halfwidths: np.ndarray, zero: bool) -> None:
    for rank, (label, color) in enumerate(zip(RANK_LABELS, RANK_COLORS)):
        mean, half = means[:, rank], halfwidths[:, rank]
        ax.fill_between(layers, mean - half, mean + half, color=color, alpha=0.20, linewidth=0)
        ax.plot(layers, mean, color=color, lw=1.45, label=f"Original {label.lower()}")
    if zero:
        ax.axhline(0, color="#444444", lw=0.65)
    max_layer = int(layers[-1])
    ax.set_xlim(0, max_layer)
    step = max(1, round(max_layer / 8))
    ticks = list(np.arange(0, max_layer + 1, step))
    if ticks[-1] != max_layer:
        ticks.append(max_layer)
    ax.set_xticks(ticks)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def save_pair(
    output_dir: Path,
    pdf_dir: Path,
    condition: str,
    view: str,
    lens_layers: np.ndarray,
    probe_layers: np.ndarray,
    summaries: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    counts: dict[str, int],
) -> None:
    import matplotlib.pyplot as plt

    _style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.55), sharex="col")
    metric_names = ("native_lens", "probe_evidence")
    for row, split_name in enumerate(("switch", "non_switch")):
        split_label = "Switch" if split_name == "switch" else "Non-switch"
        for col, metric in enumerate(metric_names):
            layers = lens_layers if metric == "native_lens" else probe_layers
            mean, half = summaries[(split_name, metric)]
            _panel(axes[row, col], layers, mean, half, zero=True)
        axes[row, 0].set_title(f"Native logit lens - {split_label} (n = {counts[split_name]})")
        axes[row, 1].set_title(
            f"Cross-fitted centroid probe - {split_label} (n = {counts[split_name]})"
        )
        axes[row, 0].set_ylabel(
            ("Centered pseudo-logit\n(natural-logit units)" if view == "raw" else
             "Change from baseline\n(centered pseudo-logit)")
        )
        axes[row, 1].set_ylabel(
            ("Standardized probe evidence\n(baseline SD units)" if view == "raw" else
             "Change from baseline\n(probe-evidence SD units)")
        )

    # Use the same vertical scale across switch and non-switch rows. Without
    # this, the selected subsets can look artificially similar.
    for col, metric in enumerate(metric_names):
        lows, highs = [], []
        for split_name in ("switch", "non_switch"):
            mean, half = summaries[(split_name, metric)]
            lows.append(float(np.min(mean - half)))
            highs.append(float(np.max(mean + half)))
        low, high = min(lows), max(highs)
        if view == "delta":
            bound = max(abs(low), abs(high)) * 1.06
            low, high = -bound, bound
        else:
            span = high - low
            low, high = low - 0.05 * span, high + 0.05 * span
        axes[0, col].set_ylim(low, high)
        axes[1, col].set_ylim(low, high)

    qualifier = "condition trajectories" if view == "raw" else "within-question change from baseline"
    for ax in axes[-1]:
        ax.set_xlabel(
            f"Residual readout (0 = embedding; {int(lens_layers[-1])} = final block)"
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.54, 0.965))
    figure.suptitle(
        f"{DISPLAY_NAMES[condition]}: {qualifier}, aligned by original baseline rank",
        y=0.995, fontweight="bold", fontsize=10.5,
    )
    figure.tight_layout(rect=(0.03, 0.02, 1, 0.91), h_pad=1.4, w_pad=2.0)
    stem = f"{condition}_switch_stratified_{view}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    figure.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def analyze_and_plot(
    input_dir: str | Path,
    output_dir: str | Path,
    pdf_dir: str | Path,
    layer_step: int,
    folds: int,
    seed: int,
) -> None:
    data = load_activation_dataset(input_dir, list(CONDITIONS))
    centered_logits = data.logits - data.logits.mean(axis=-1, keepdims=True)
    baseline_order = np.argsort(-centered_logits[:, 0, -1], axis=-1)
    lens_aligned = np.stack([
        align_options(centered_logits[:, ci], baseline_order) for ci in range(len(CONDITIONS))
    ])
    final_choices = np.argmax(centered_logits[:, :, -1], axis=-1)
    lens_layers = np.arange(centered_logits.shape[2])
    probe_layers = np.asarray(sorted(set(range(0, len(lens_layers), layer_step)) | {len(lens_layers) - 1}))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    score_path = output / "cross_fitted_candidate_probe_scores.npz"
    probe_scores = None
    if score_path.exists():
        with np.load(score_path, allow_pickle=False) as cached:
            cached_layers = cached["layers"]
            cached_qids = cached["question_ids"].astype(str).tolist()
            cached_conditions = cached["conditions"].astype(str).tolist()
            if (
                np.array_equal(cached_layers, probe_layers)
                and cached_qids == data.question_ids
                and cached_conditions == list(CONDITIONS)
            ):
                probe_scores = cached["scores"]
    if probe_scores is None:
        residuals = load_selected_residuals(input_dir, data.question_ids, probe_layers)
        probe_scores = centroid_candidate_scores(residuals, baseline_order[:, 0], folds, seed)
        del residuals
        np.savez_compressed(
            score_path,
            scores=probe_scores,
            layers=probe_layers,
            question_ids=np.asarray(data.question_ids),
            conditions=np.asarray(CONDITIONS),
            baseline_order=baseline_order,
        )
    probe_aligned = np.stack([align_options(probe_scores[ci], baseline_order) for ci in range(len(CONDITIONS))])

    pdf_output = Path(pdf_dir)

    rows = []
    for ci, condition in enumerate(CONDITIONS[1:], start=1):
        switched = final_choices[:, ci] != baseline_order[:, 0]
        masks = {"switch": switched, "non_switch": ~switched}
        counts = {name: int(mask.sum()) for name, mask in masks.items()}
        for view in ("raw", "delta"):
            summaries = {}
            for split_name, mask in masks.items():
                lens_values = lens_aligned[ci] if view == "raw" else lens_aligned[ci] - lens_aligned[0]
                probe_values = probe_aligned[ci] if view == "raw" else probe_aligned[ci] - probe_aligned[0]
                summaries[(split_name, "native_lens")] = summarize_subset(lens_values, baseline_order, mask)
                summaries[(split_name, "probe_evidence")] = summarize_subset(probe_values, baseline_order, mask)

                for metric, layers in (("native_lens", lens_layers), ("probe_evidence", probe_layers)):
                    means, halfwidths = summaries[(split_name, metric)]
                    for li, layer in enumerate(layers):
                        for rank in range(4):
                            rows.append({
                                "condition": condition,
                                "split": split_name,
                                "n": counts[split_name],
                                "view": view,
                                "metric": metric,
                                "layer": int(layer),
                                "original_rank": rank + 1,
                                "mean": means[li, rank],
                                "ci_low": means[li, rank] - halfwidths[li, rank],
                                "ci_high": means[li, rank] + halfwidths[li, rank],
                            })
            save_pair(output, pdf_output, condition, view, lens_layers, probe_layers, summaries, counts)

    with (output / "condition_switch_trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Game/neutral trajectories by switch status")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="PNG, SVG, CSV, and probe-score directory")
    parser.add_argument("--pdf-output", required=True, help="Directory for render-checked paper PDFs")
    parser.add_argument("--layer-step", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze_and_plot(args.input, args.output, args.pdf_output, args.layer_step, args.folds, args.seed)


if __name__ == "__main__":
    main()
