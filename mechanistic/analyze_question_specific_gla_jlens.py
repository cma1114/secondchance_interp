from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LETTERS = "ABCD"


def _bootstrap(values: np.ndarray, strata: np.ndarray, rng, draws: int = 4000):
    values = np.asarray(values, dtype=float)
    samples = np.zeros((draws, values.shape[1]), dtype=float)
    for label in np.unique(strata):
        group = np.flatnonzero(strata == label)
        # Chunk the resamples so a large W1 stratum does not materialize a
        # draws x questions x layers tensor all at once.
        for start in range(0, draws, 250):
            stop = min(start + 250, draws)
            selected = rng.choice(group, size=(stop - start, len(group)), replace=True)
            samples[start:stop] += values[selected].sum(axis=1)
    samples /= len(values)
    return values.mean(0), np.quantile(samples, .025, axis=0), np.quantile(samples, .975, axis=0)


def _aligned(scores: np.ndarray, w1: np.ndarray, w2: np.ndarray):
    centered = scores - scores.mean(axis=-1, keepdims=True)
    q = np.arange(len(scores))
    i1 = np.asarray([LETTERS.index(value) for value in w1])
    i2 = np.asarray([LETTERS.index(value) for value in w2])
    first = centered[q, :, i1]
    second = centered[q, :, i2]
    others = np.empty_like(first)
    for qi in q:
        keep = [index for index in range(4) if index not in {i1[qi], i2[qi]}]
        if keep:
            others[qi] = centered[qi, :, keep].mean(axis=0)
        else:
            others[qi] = (centered[qi].sum(axis=-1) - centered[qi, :, i1[qi]]) / 3
    return centered, first, second, others, i1, i2


def _bootstrap_difference(
    first: np.ndarray,
    first_strata: np.ndarray,
    second: np.ndarray,
    second_strata: np.ndarray,
    rng,
    draws: int = 4000,
):
    first_mean, _, _ = _bootstrap(first, first_strata, rng, draws)
    second_mean, _, _ = _bootstrap(second, second_strata, rng, draws)
    samples = np.zeros((draws, first.shape[1]), dtype=float)
    for sign, values, strata in ((1.0, first, first_strata), (-1.0, second, second_strata)):
        for label in np.unique(strata):
            group = np.flatnonzero(strata == label)
            for start in range(0, draws, 250):
                stop = min(start + 250, draws)
                selected = rng.choice(group, size=(stop - start, len(group)), replace=True)
                samples[start:stop] += sign * values[selected].sum(axis=1) / len(values)
    return first_mean - second_mean, np.quantile(samples, .025, axis=0), np.quantile(samples, .975, axis=0)


def analyze(
    results_path: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
    trusted_evaluation_path: Path,
    trusted_neutral_path: Path,
    output_dir: Path,
    figure_dir: Path,
    seed: int,
) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(results_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"]):
        raise RuntimeError("Collection incomplete")
    qids = arrays["question_ids"].astype(str).tolist()
    layers = arrays["gla_layers_zero_based"].astype(int) + 1
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped = json.loads(remapped_baseline_path.read_text())["results"]
    w1 = np.asarray([baseline[qid]["answer"] for qid in qids])
    w2 = np.asarray([remapped[qid]["answer_original_content"] for qid in qids])
    conflict = w1 != w2
    strata = w1.copy()

    trusted_evaluation = json.loads(trusted_evaluation_path.read_text())["results"]
    trusted_neutral = json.loads(trusted_neutral_path.read_text())["results"]
    trusted = np.stack([
        np.asarray([trusted_evaluation[qid]["aggregated_ad_logits"] for qid in qids]),
        np.asarray([trusted_neutral[qid]["aggregated_ad_logits"] for qid in qids]),
    ])
    max_error = float(np.max(np.abs(arrays["natural_logits"] - trusted)))

    score_keys = (
        "jlens_option_max", "jlens_option_mean", "linear_option_max",
        "jlens_cumulative_option_max", "jlens_cumulative_option_mean",
        "linear_cumulative_option_max",
    )
    aligned = {key: _aligned(arrays[key].astype(float), w1, w2) for key in score_keys}
    rng = np.random.default_rng(seed)
    rows = []
    summary = {
        "definitions": {
            "difference_vector": "The question-specific final-position GLA output in Evaluation minus the corresponding output in Matched Neutral.",
            "standard_jlens": "The difference vector is transported with that block's learned JLens map, final-RMS-normalized, and unembedded. This is a direction readout.",
            "option_score": "Maximum JLens score among substantive tokenizer tokens in the option text; mean-token scoring is a robustness analysis.",
            "cumulative": "Sum of the layer-specific JLens-transported GLA differences through the indicated GLA block, then final-RMS-normalized and unembedded.",
        },
        "validation": {"questions": len(qids), "conflict": int(conflict.sum()), "no_conflict": int((~conflict).sum()), "max_abs_natural_logit_error": max_error},
        "metrics": {},
    }

    for key in score_keys:
        centered, first, second, others, i1, _i2 = aligned[key]
        summary["metrics"][key] = {}
        for subset_name, mask in (("conflict", conflict), ("no_conflict", ~conflict)):
            local = {}
            for name, values in (("W1", first), ("W2", second), ("other", others), ("W1_minus_alternatives", first - (centered.sum(-1) - first) / 3)):
                mean, low, high = _bootstrap(values[mask], strata[mask], rng)
                local[name] = {"mean": mean.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist()}
                for li, layer in enumerate(layers):
                    rows.append({"score": key, "subset": subset_name, "metric": name, "block": int(layer), "mean": mean[li], "ci_low": low[li], "ci_high": high[li]})
            bottom = np.empty((mask.sum(), len(layers)), dtype=float)
            subset_centered = centered[mask]
            subset_i1 = i1[mask]
            for row, index in enumerate(subset_i1):
                bottom[row] = subset_centered[row, :, index] == subset_centered[row].min(axis=-1)
            mean, low, high = _bootstrap(bottom, strata[mask], rng)
            local["W1_bottom_rate"] = {"mean": mean.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist()}
            summary["metrics"][key][subset_name] = local
        centered, first, _second, _others, i1, _i2 = aligned[key]
        bottom = np.empty((len(qids), len(layers)), dtype=float)
        for row, index in enumerate(i1):
            bottom[row] = centered[row, :, index] == centered[row].min(axis=-1)
        difference_metrics = {
            "W1_centered_score": first,
            "W1_minus_alternatives": first - (centered.sum(-1) - first) / 3,
            "W1_bottom_rate": bottom,
        }
        summary["metrics"][key]["conflict_minus_no_conflict"] = {}
        for name, values in difference_metrics.items():
            mean, low, high = _bootstrap_difference(
                values[conflict], strata[conflict], values[~conflict], strata[~conflict], rng
            )
            summary["metrics"][key]["conflict_minus_no_conflict"][name] = {
                "mean": mean.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist()
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "layerwise_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    colors = {"W1": "#318cf5", "W2": "#f17c35", "other": "#55b96b"}
    primary = summary["metrics"]["jlens_option_max"]["conflict"]
    cumulative = summary["metrics"]["jlens_cumulative_option_max"]["conflict"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.subplots_adjust(left=.065, right=.985, bottom=.15, top=.90, wspace=.32)
    for name in ("W1", "W2", "other"):
        cell = primary[name]
        axes[0].plot(layers, cell["mean"], color=colors[name], linewidth=2, label=name)
        axes[0].fill_between(layers, cell["ci_low"], cell["ci_high"], color=colors[name], alpha=.23)
        cell = cumulative[name]
        axes[1].plot(layers, cell["mean"], color=colors[name], linewidth=2, label=name)
        axes[1].fill_between(layers, cell["ci_low"], cell["ci_high"], color=colors[name], alpha=.23)
    for score_key, color, label in (
        ("jlens_option_max", "#8a5cf6", "Maximum option-token score"),
        ("jlens_option_mean", "#15a38d", "Mean option-token score"),
    ):
        cell = summary["metrics"][score_key]["conflict_minus_no_conflict"]["W1_bottom_rate"]
        axes[2].plot(layers, cell["mean"], color=color, linewidth=2, label=label)
        axes[2].fill_between(layers, cell["ci_low"], cell["ci_high"], color=color, alpha=.23)
    axes[2].axhline(0, color="#777", linestyle="--", linewidth=1)
    for axis in axes[:2]:
        axis.axhline(0, color="#777", linestyle="--", linewidth=1)
    titles = (
        "A  Individual GLA difference",
        "B  Cumulative transported differences",
        "C  Conflict-specific W1-last rate",
    )
    ylabels = (
        "Centered option-content JLens score",
        "Centered option-content JLens score",
        "Conflict minus no-conflict fraction",
    )
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("GLA block")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.18)
        axis.legend(frameon=False)
    figure = figure_dir / "qwen36_action_matched_question_specific_gla_jlens.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)
    summary["figure"] = str(figure)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--trusted-evaluation", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    print(json.dumps(analyze(args.results, args.baseline, args.remapped_baseline, args.trusted_evaluation, args.trusted_neutral, args.output_dir, args.figure_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
