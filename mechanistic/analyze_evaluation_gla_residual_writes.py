from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LETTERS = "ABCD"
CONDITIONS = ("Evaluation", "Matched Neutral")
POSITIONS = ("evaluation period", "final decision")


def _bootstrap_curve(values: np.ndarray, strata: np.ndarray, rng, draws: int = 4000):
    values = np.asarray(values, dtype=float)
    samples = np.zeros((draws, values.shape[1]), dtype=float)
    for label in np.unique(strata):
        group = np.flatnonzero(strata == label)
        selected = rng.choice(group, size=(draws, len(group)), replace=True)
        samples += values[selected].sum(axis=1)
    samples /= len(values)
    return (
        values.mean(axis=0),
        np.quantile(samples, 0.025, axis=0),
        np.quantile(samples, 0.975, axis=0),
    )


def _content_align(values: np.ndarray, qids: list[str], plan: dict) -> np.ndarray:
    aligned = np.empty_like(values)
    for qi, qid in enumerate(qids):
        original_to_new = plan[qid]["original_to_new"]
        aligned[:, qi] = np.stack([
            values[:, qi, ..., LETTERS.index(original_to_new[original])]
            for original in LETTERS
        ], axis=-1)
    return aligned


def analyze(
    results_path: Path,
    metadata_path: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
    remapping_plan_path: Path,
    evaluation_results_path: Path,
    neutral_results_path: Path,
    vocab_path: Path | None,
    output_dir: Path,
    figure_dir: Path,
    seed: int,
) -> dict:
    import matplotlib.pyplot as plt

    with np.load(results_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"]):
        raise RuntimeError("GLA residual-write collection is incomplete")
    metadata = json.loads(metadata_path.read_text())
    qids = arrays["question_ids"].astype(str).tolist()
    layers = arrays["gla_layers_zero_based"].astype(int) + 1
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped_baseline = json.loads(remapped_baseline_path.read_text())["results"]
    plan = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    trusted_e = json.loads(evaluation_results_path.read_text())["results"]
    trusted_n = json.loads(neutral_results_path.read_text())["results"]
    trusted = np.stack([
        np.asarray([trusted_e[qid]["aggregated_ad_logits"] for qid in qids]),
        np.asarray([trusted_n[qid]["aggregated_ad_logits"] for qid in qids]),
    ])
    max_logit_error = float(np.max(np.abs(arrays["natural_logits"] - trusted)))
    w1 = np.asarray([baseline[qid]["answer"] for qid in qids])
    w2 = np.asarray([remapped_baseline[qid]["answer_original_content"] for qid in qids])
    conflict = w1 != w2
    strata = w1.copy()
    writes = arrays["canonical_ad_write"].astype(float)
    centered = writes - writes.mean(axis=-1, keepdims=True)
    final_aligned = _content_align(centered[:, :, :, 1, :], qids, plan)
    eval_aligned = centered[:, :, :, 0, :]
    qi = np.arange(len(qids))
    w1_index = np.asarray([LETTERS.index(value) for value in w1])
    w2_index = np.asarray([LETTERS.index(value) for value in w2])
    w1_take = np.broadcast_to(w1_index[None, :, None, None], (*final_aligned.shape[:-1], 1))
    w2_take = np.broadcast_to(w2_index[None, :, None, None], (*final_aligned.shape[:-1], 1))
    final_w1 = np.take_along_axis(final_aligned, w1_take, axis=-1)[..., 0]
    final_w2 = np.take_along_axis(final_aligned, w2_take, axis=-1)[..., 0]
    final_other = np.empty_like(final_w1)
    for index in range(len(qids)):
        keep = [value for value in range(4) if value not in {w1_index[index], w2_index[index]}]
        final_other[:, index] = np.take(
            final_aligned[:, index], keep, axis=-1
        ).mean(axis=-1)
    eval_take = np.broadcast_to(w1_index[None, :, None, None], (*eval_aligned.shape[:-1], 1))
    eval_w1 = np.take_along_axis(eval_aligned, eval_take, axis=-1)[..., 0]
    rng = np.random.default_rng(seed)

    summary = {
        "design": metadata,
        "validation": {
            "max_abs_aggregated_ad_logit_error_vs_trusted": max_logit_error,
            "complete_questions": int(arrays["completed"].sum()),
        },
        "subsets": {
            "all": int(len(qids)),
            "w1_not_equal_w2": int(conflict.sum()),
            "w1_equal_w2": int((~conflict).sum()),
        },
        "definitions": {
            "GLA residual write": "The post-output-projection GLA vector actually added to the residual stream.",
            "direct A-D write": "Dot product of that vector with canonical A, B, C, and D unembedding rows, centered across the four letters.",
            "cumulative direct write": "Sum of direct writes over ordered GLA blocks; exact in raw unembedding space but not an estimate of the final causal effect after intervening nonlinearities.",
        },
    }

    masks = {"all": np.ones(len(qids), dtype=bool), "conflict": conflict, "no_conflict": ~conflict}
    summary["metrics"] = {}
    rows = []
    for subset, mask in masks.items():
        cell = {}
        local_strata = strata[mask]
        for position, position_name in enumerate(POSITIONS):
            norm_difference = arrays["output_norm"][0, mask, :, position] - arrays["output_norm"][1, mask, :, position]
            delta_norm = arrays["paired_delta_norm"][mask, :, position]
            cosine = arrays["paired_cosine"][mask, :, position]
            cell[position_name] = {}
            for metric_name, values in (
                ("evaluation_minus_neutral_output_norm", norm_difference),
                ("paired_output_delta_norm", delta_norm),
                ("paired_output_cosine", cosine),
            ):
                mean, low, high = _bootstrap_curve(values, local_strata, rng)
                cell[position_name][metric_name] = {
                    "mean": mean.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist()
                }
                for li, layer in enumerate(layers):
                    rows.append({
                        "subset": subset, "position": position_name,
                        "metric": metric_name, "block": int(layer),
                        "mean": mean[li], "ci_low": low[li], "ci_high": high[li],
                    })
        summary["metrics"][subset] = cell

    # Primary answer-aligned result uses conflict trials.
    mask = conflict
    local_strata = strata[mask]
    answer_curves = {
        "W1": final_w1[0, mask] - final_w1[1, mask],
        "W2": final_w2[0, mask] - final_w2[1, mask],
        "other options": final_other[0, mask] - final_other[1, mask],
    }
    summary["answer_aligned_conflict"] = {}
    summary["final_w1_by_condition_conflict"] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        mean, low, high = _bootstrap_curve(final_w1[condition_index, mask], local_strata, rng)
        summary["final_w1_by_condition_conflict"][condition] = {
            "mean": mean.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist()
        }
    for name, values in answer_curves.items():
        mean, low, high = _bootstrap_curve(values, local_strata, rng)
        cumulative = np.cumsum(values, axis=1)
        cumulative_mean, cumulative_low, cumulative_high = _bootstrap_curve(
            cumulative, local_strata, rng
        )
        summary["answer_aligned_conflict"][name] = {
            "per_block": {"mean": mean.tolist(), "ci_low": low.tolist(), "ci_high": high.tolist()},
            "cumulative": {
                "mean": cumulative_mean.tolist(),
                "ci_low": cumulative_low.tolist(),
                "ci_high": cumulative_high.tolist(),
            },
        }
        for li, layer in enumerate(layers):
            rows.extend([
                {"subset": "conflict", "position": "final decision", "metric": f"E-N direct {name} write", "block": int(layer), "mean": mean[li], "ci_low": low[li], "ci_high": high[li]},
                {"subset": "conflict", "position": "final decision", "metric": f"E-N cumulative {name} write", "block": int(layer), "mean": cumulative_mean[li], "ci_low": cumulative_low[li], "ci_high": cumulative_high[li]},
            ])
    old_w1_difference = eval_w1[0] - eval_w1[1]
    old_mean, old_low, old_high = _bootstrap_curve(old_w1_difference, strata, rng)
    summary["evaluation_period_old_w1_direct_write"] = {
        "mean": old_mean.tolist(), "ci_low": old_low.tolist(), "ci_high": old_high.tolist()
    }
    no_conflict_w1 = final_w1[0, ~conflict] - final_w1[1, ~conflict]
    no_conflict_cumulative = np.cumsum(no_conflict_w1, axis=1)
    nc_mean, nc_low, nc_high = _bootstrap_curve(
        no_conflict_cumulative, strata[~conflict], rng
    )
    summary["answer_aligned_no_conflict_w1_cumulative"] = {
        "mean": nc_mean.tolist(), "ci_low": nc_low.tolist(), "ci_high": nc_high.tolist()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "layerwise_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    colors = {"evaluation period": "#8a5cf6", "final decision": "#15a38d"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for position, position_name in enumerate(POSITIONS):
        for axis, metric, ylabel in (
            (axes[0], "paired_output_delta_norm", "||Evaluation − Neutral write||"),
            (axes[1], "paired_output_cosine", "Cosine(Evaluation, Neutral write)"),
        ):
            values = arrays["paired_delta_norm"][:, :, position] if metric.endswith("delta_norm") else arrays["paired_cosine"][:, :, position]
            mean, low, high = _bootstrap_curve(values, strata, rng)
            axis.plot(layers, mean, color=colors[position_name], label=position_name.title(), linewidth=2)
            axis.fill_between(layers, low, high, color=colors[position_name], alpha=.23)
            axis.set_ylabel(ylabel)
    for position, position_name in enumerate(POSITIONS):
        values = arrays["output_norm"][0, :, :, position] - arrays["output_norm"][1, :, :, position]
        mean, low, high = _bootstrap_curve(values, strata, rng)
        axes[2].plot(layers, mean, color=colors[position_name], label=position_name.title(), linewidth=2)
        axes[2].fill_between(layers, low, high, color=colors[position_name], alpha=.23)
    axes[2].axhline(0, color="#777", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Evaluation − Neutral output norm")
    for label, axis in zip("ABC", axes):
        axis.set_title(f"{label}  {axis.get_ylabel()}", loc="left", fontweight="bold")
        axis.set_xlabel("GLA block")
        axis.grid(alpha=.18)
    axes[0].legend(frameon=False)
    figure1 = figure_dir / "qwen36_action_matched_gla_output_geometry.png"
    fig.savefig(figure1, dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    answer_colors = {"W1": "#318cf5", "W2": "#f17c35", "other options": "#55b96b"}
    condition_colors = {"Evaluation": "#8a5cf6", "Matched Neutral": "#15a38d"}
    for condition_index, condition in enumerate(CONDITIONS):
        mean, low, high = _bootstrap_curve(final_w1[condition_index, mask], local_strata, rng)
        axes[0].plot(layers, mean, color=condition_colors[condition], label=condition, linewidth=2)
        axes[0].fill_between(layers, low, high, color=condition_colors[condition], alpha=.22)
    for name, values in answer_curves.items():
        mean, low, high = _bootstrap_curve(values, local_strata, rng)
        axes[1].plot(layers, mean, color=answer_colors[name], label=name, linewidth=2)
        axes[1].fill_between(layers, low, high, color=answer_colors[name], alpha=.22)
        cumulative = np.cumsum(values, axis=1)
        mean, low, high = _bootstrap_curve(cumulative, local_strata, rng)
        axes[2].plot(layers, mean, color=answer_colors[name], label=name, linewidth=2)
        axes[2].fill_between(layers, low, high, color=answer_colors[name], alpha=.22)
    nc_plot_mean, nc_plot_low, nc_plot_high = _bootstrap_curve(
        no_conflict_cumulative, strata[~conflict], rng
    )
    axes[2].plot(
        layers, nc_plot_mean, color="#777777", linestyle=":", linewidth=2,
        label="W1 (no conflict)",
    )
    axes[2].fill_between(
        layers, nc_plot_low, nc_plot_high, color="#777777", alpha=.14
    )
    for label, axis, title in zip(
        "ABC", axes,
        ("Direct W1 write by condition", "Evaluation − Neutral per-block write", "Cumulative Evaluation − Neutral write"),
    ):
        axis.axhline(0, color="#777", linestyle="--", linewidth=1)
        axis.set_title(f"{label}  {title}", loc="left", fontweight="bold")
        axis.set_xlabel("GLA block")
        axis.set_ylabel("Centered direct A–D units")
        axis.grid(alpha=.18)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    axes[2].legend(frameon=False)
    figure2 = figure_dir / "qwen36_action_matched_gla_answer_writes.png"
    fig.savefig(figure2, dpi=220, bbox_inches="tight"); plt.close(fig)

    vocab = json.loads(vocab_path.read_text()) if vocab_path and vocab_path.exists() else None
    summary["vocabulary_readout_available"] = vocab is not None
    if vocab is not None:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Action-matched GLA residual writes", "",
        f"All {len(qids)} questions completed; {int(conflict.sum())} have W1 != W2. "
        f"Natural A-D logits reproduce the trusted factorial with maximum absolute error {max_logit_error:.6g}.",
        "",
        "This analysis records the actual projected vector that each GLA module adds to the residual stream. It is distinct from the earlier recurrent-memory-update transplant.",
        "",
        "## Figures", "",
        f"![GLA output geometry]({figure1})", "",
        f"![GLA answer-aligned writes]({figure2})", "",
        "## Interpretation", "",
        "The per-block and cumulative curves are immediate raw-unembedding descriptions of the natural GLA writes. They show whether the Evaluation condition distributes a common answer-directed write across blocks or instead changes heterogeneous operations. They are not, by themselves, causal effects after later nonlinear computation.",
        "",
    ]
    if vocab is not None:
        lines.extend([
            "Mean raw- and JLens-decoded vocabulary directions are saved in `vocabulary_readouts.json`. Only readable ASCII tokens are surfaced there; this avoids presenting untranslated Chinese or other script fragments as if they were English concepts.",
            "",
        ])
    (output_dir / "REPORT.md").write_text("\n".join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze action-matched GLA residual writes")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--evaluation-results", type=Path, required=True)
    parser.add_argument("--neutral-results", type=Path, required=True)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = analyze(
        args.results, args.metadata, args.baseline, args.remapped_baseline,
        args.remapping_plan, args.evaluation_results, args.neutral_results,
        args.vocab, args.output_dir, args.figure_dir, args.seed,
    )
    print(json.dumps({"validation": summary["validation"], "subsets": summary["subsets"]}, indent=2))


if __name__ == "__main__":
    main()
