from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .collect_evaluation_period_source_trace import ANCHOR_NAMES, ROLE_NAMES
from .semantic_mapping import displayed_argmax_to_semantic_indices


LETTERS = "ABCD"
CONDITION_NAMES = ("Evaluation", "Matched Neutral")


def _bootstrap(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator, draws: int = 4000):
    values = np.asarray(values, dtype=float)
    samples = np.zeros((draws, *values.shape[1:]), dtype=float)
    for label in np.unique(strata):
        group = np.flatnonzero(strata == label)
        selected = rng.choice(group, size=(draws, len(group)), replace=True)
        samples += values[selected].sum(axis=1)
    samples /= len(values)
    return values.mean(axis=0), np.quantile(samples, 0.025, axis=0), np.quantile(samples, 0.975, axis=0)


def _interval(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    mean, low, high = _bootstrap(np.asarray(values)[:, None], strata, rng)
    return {"mean": float(mean[0]), "ci_low": float(low[0]), "ci_high": float(high[0])}


def _ratio_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    draws: int = 4000,
) -> dict[str, float]:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    num_samples = np.zeros(draws, dtype=float)
    den_samples = np.zeros(draws, dtype=float)
    for label in np.unique(strata):
        group = np.flatnonzero(strata == label)
        selected = rng.choice(group, size=(draws, len(group)), replace=True)
        num_samples += numerator[selected].sum(axis=1)
        den_samples += denominator[selected].sum(axis=1)
    samples = np.divide(
        num_samples,
        den_samples,
        out=np.full(draws, np.nan),
        where=den_samples != 0,
    )
    low, high = np.nanquantile(samples, (0.025, 0.975))
    return {
        "mean": float(numerator.sum() / denominator.sum()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -(probabilities * np.log2(probabilities)).sum(axis=-1)


def _content_align(values: np.ndarray, qids: list[str], mappings: dict[str, dict]) -> np.ndarray:
    result = np.empty_like(values)
    for qi, qid in enumerate(qids):
        original_to_new = mappings[qid]["original_to_new"]
        result[:, qi] = np.stack([
            values[:, qi, ..., LETTERS.index(original_to_new[original])]
            for original in LETTERS
        ], axis=-1)
    return result


def _take_semantic(values: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    # values begins condition, question and ends answer content.
    result = np.empty(values.shape[:-1], dtype=values.dtype)
    for qi, letter in enumerate(semantic):
        result[:, qi] = values[:, qi, ..., LETTERS.index(letter)]
    return result


def _load_shards(run_dir: Path):
    shards = sorted((run_dir / "cohorts").glob("cohort_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No cohort shards under {run_dir}")
    qids: list[str] = []
    natural, ablated, anchor_norm, anchor_ad, role_mean = [], [], [], [], []
    for path in shards:
        with np.load(path, allow_pickle=False) as loaded:
            qids.extend(loaded["question_ids"].astype(str).tolist())
            natural.append(loaded["natural_logits"].astype(np.float32))
            ablated.append(loaded["ablated_logits"].astype(np.float32))
            trace_norm = loaded["trace_norm"].astype(np.float32)
            trace_ad = loaded["trace_ad"].astype(np.float32)
            anchors = loaded["anchor_relative_positions"].astype(int)
            roles = loaded["role_ids"].astype(int)
        conditions, batch, layers, _ = trace_norm.shape
        n_anchors = anchors.shape[-1]
        a_norm = np.empty((conditions, batch, layers, n_anchors), dtype=np.float32)
        a_ad = np.empty((conditions, batch, layers, n_anchors, 4), dtype=np.float32)
        r_mean = np.full((conditions, batch, layers, len(ROLE_NAMES)), np.nan, dtype=np.float32)
        for ci in range(conditions):
            for bi in range(batch):
                for ai, position in enumerate(anchors[ci, bi]):
                    a_norm[ci, bi, :, ai] = trace_norm[ci, bi, :, position]
                    a_ad[ci, bi, :, ai] = trace_ad[ci, bi, :, position]
                for ri in range(len(ROLE_NAMES)):
                    mask = roles[ci, bi] == ri
                    if mask.any():
                        r_mean[ci, bi, :, ri] = np.nanmean(
                            trace_norm[ci, bi][:, mask], axis=-1
                        )
        anchor_norm.append(a_norm)
        anchor_ad.append(a_ad)
        role_mean.append(r_mean)
    return {
        "qids": qids,
        "natural": np.concatenate(natural, axis=1),
        "ablated": np.concatenate(ablated, axis=1),
        "anchor_norm": np.concatenate(anchor_norm, axis=1),
        "anchor_ad": np.concatenate(anchor_ad, axis=1),
        "role_mean_norm": np.concatenate(role_mean, axis=1),
        "n_shards": len(shards),
    }


def analyze(
    run_dir: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
    remapping_plan_path: Path,
    evaluation_results_path: Path,
    neutral_results_path: Path,
    discovery_plan_path: Path,
    output_dir: Path,
    figure_path: Path,
    seed: int,
) -> dict:
    import matplotlib.pyplot as plt

    data = _load_shards(run_dir)
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    qids = data["qids"]
    if qids != metadata["question_ids"]:
        raise RuntimeError("Shard order does not match run metadata")
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped_baseline = json.loads(remapped_baseline_path.read_text())["results"]
    mappings = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    trusted_rows = [
        json.loads(evaluation_results_path.read_text())["results"],
        json.loads(neutral_results_path.read_text())["results"],
    ]
    trusted = np.stack([
        np.asarray([rows[qid]["aggregated_ad_logits"] for qid in qids], dtype=np.float32)
        for rows in trusted_rows
    ])
    max_natural_error = float(np.max(np.abs(data["natural"] - trusted)))
    if max_natural_error != 0:
        raise RuntimeError(f"Natural logits do not reproduce trusted run: {max_natural_error}")

    w1 = np.asarray([baseline[qid]["answer"] for qid in qids])
    w2 = np.asarray([remapped_baseline[qid]["answer_original_content"] for qid in qids])
    conflict = w1 != w2
    discovery = set(json.loads(discovery_plan_path.read_text())["question_ids"])
    discovery_mask = np.asarray([qid in discovery for qid in qids])
    split_masks = {
        "all": np.ones(len(qids), dtype=bool),
        "discovery": discovery_mask,
        "confirmation": ~discovery_mask,
    }
    layers = np.asarray(metadata["gla_layers_zero_based"], dtype=int) + 1
    strata = w1
    rng = np.random.default_rng(seed)

    natural = _content_align(data["natural"], qids, mappings)
    ablated = _content_align(data["ablated"], qids, mappings)
    final_index = ANCHOR_NAMES.index("final decision")
    final_ad = _content_align(data["anchor_ad"][..., final_index, :], qids, mappings)
    final_ad -= final_ad.mean(axis=-1, keepdims=True)
    final_w1 = _take_semantic(final_ad, w1)
    final_w2 = _take_semantic(final_ad, w2)
    final_other = np.empty_like(final_w1)
    for qi in range(len(qids)):
        keep = [i for i, letter in enumerate(LETTERS) if letter not in {w1[qi], w2[qi]}]
        if conflict[qi]:
            final_other[:, qi] = final_ad[:, qi][:, :, keep].mean(axis=-1)
        else:
            keep = [i for i, letter in enumerate(LETTERS) if letter != w1[qi]]
            final_other[:, qi] = final_ad[:, qi][:, :, keep].mean(axis=-1)

    summary: dict[str, object] = {
        "design": metadata,
        "definitions": {
            "source trace": (
                "For one GLA block, its natural post-output-projection output minus a within-block replay "
                "with beta=0 only at the evaluation-closing period. This is the exact deletion effect of "
                "that source write inside the block, including later recurrent interactions; it is not an additive decomposition."
            ),
            "direct answer contribution": (
                "Dot product of the source-trace residual vector with canonical A-D unembedding rows, "
                "then aligned to semantic option identity and centered across the four options."
            ),
            "global period-write ablation": (
                "A complete model forward with the evaluation-closing period's beta set to zero in all 48 GLAs simultaneously."
            ),
            "conflict": "W1 differs from W2, the answer chosen by a fresh remapped Baseline.",
        },
        "validation": {
            "questions": len(qids),
            "cohort_shards": data["n_shards"],
            "max_abs_natural_logit_error_vs_trusted": max_natural_error,
        },
        "subsets": {
            "conflict": int(conflict.sum()),
            "no_conflict": int((~conflict).sum()),
            "discovery": int(discovery_mask.sum()),
            "confirmation": int((~discovery_mask).sum()),
        },
        "global_ablation": {},
        "final_source_trace": {},
    }

    mapping_rows = [mappings[qid] for qid in qids]
    natural_choice = displayed_argmax_to_semantic_indices(
        data["natural"], mapping_rows
    )
    ablated_choice = displayed_argmax_to_semantic_indices(
        data["ablated"], mapping_rows
    )
    w1_index = np.asarray([LETTERS.index(letter) for letter in w1])
    w2_index = np.asarray([LETTERS.index(letter) for letter in w2])
    for split_name, split in split_masks.items():
        summary["global_ablation"][split_name] = {}
        for subset_name, subset in (("conflict", conflict), ("no_conflict", ~conflict)):
            mask = split & subset
            cell = {}
            for ci, condition in enumerate(CONDITION_NAMES):
                w1_natural = natural_choice[ci, mask] == w1_index[mask]
                w1_ablated = ablated_choice[ci, mask] == w1_index[mask]
                w2_natural = natural_choice[ci, mask] == w2_index[mask]
                w2_ablated = ablated_choice[ci, mask] == w2_index[mask]
                rows = np.flatnonzero(mask)
                natural_w1_logit = np.asarray([natural[ci, q, w1_index[q]] for q in rows])
                ablated_w1_logit = np.asarray([ablated[ci, q, w1_index[q]] for q in rows])
                natural_w2_logit = np.asarray([natural[ci, q, w2_index[q]] for q in rows])
                ablated_w2_logit = np.asarray([ablated[ci, q, w2_index[q]] for q in rows])
                cell[condition] = {
                    "n": int(mask.sum()),
                    "w1_selection_change_pp": _interval(
                        100 * (w1_ablated.astype(float) - w1_natural.astype(float)), strata[mask], rng
                    ),
                    "w2_selection_change_pp": _interval(
                        100 * (w2_ablated.astype(float) - w2_natural.astype(float)), strata[mask], rng
                    ),
                    "w1_logit_change": _interval(ablated_w1_logit - natural_w1_logit, strata[mask], rng),
                    "w1_minus_w2_margin_change": _interval(
                        (ablated_w1_logit - ablated_w2_logit)
                        - (natural_w1_logit - natural_w2_logit),
                        strata[mask], rng,
                    ),
                    "entropy_change_bits": _interval(
                        _entropy(ablated[ci, mask]) - _entropy(natural[ci, mask]), strata[mask], rng
                    ),
                }
            summary["global_ablation"][split_name][subset_name] = cell

        # On conflict trials, the natural behavioral gap is Neutral minus
        # Evaluation W1 selection.  If period-write deletion removes the
        # Game-specific mechanism, this gap should shrink.
        mask = split & conflict
        rows = np.flatnonzero(mask)
        natural_w1 = np.stack([
            natural_choice[ci, mask] == w1_index[mask] for ci in range(2)
        ])
        ablated_w1 = np.stack([
            ablated_choice[ci, mask] == w1_index[mask] for ci in range(2)
        ])
        natural_gap_by_question = 100 * (
            natural_w1[1].astype(float) - natural_w1[0].astype(float)
        )
        ablated_gap_by_question = 100 * (
            ablated_w1[1].astype(float) - ablated_w1[0].astype(float)
        )
        natural_margin = np.stack([
            np.asarray([
                natural[ci, q, w1_index[q]] - natural[ci, q, w2_index[q]]
                for q in rows
            ]) for ci in range(2)
        ])
        ablated_margin = np.stack([
            np.asarray([
                ablated[ci, q, w1_index[q]] - ablated[ci, q, w2_index[q]]
                for q in rows
            ]) for ci in range(2)
        ])
        gap_reduction = natural_gap_by_question - ablated_gap_by_question
        margin_gap_reduction = (
            (ablated_margin[0] - natural_margin[0])
            - (ablated_margin[1] - natural_margin[1])
        )
        natural_gap = float(natural_gap_by_question.mean())
        summary["global_ablation"][split_name]["conflict_gap"] = {
            "natural_neutral_minus_evaluation_w1_selection_pp": natural_gap,
            "ablated_neutral_minus_evaluation_w1_selection_pp": float(
                ablated_gap_by_question.mean()
            ),
            "gap_reduction_pp": _interval(gap_reduction, strata[mask], rng),
            "fraction_of_natural_gap_removed": _ratio_interval(
                gap_reduction, natural_gap_by_question, strata[mask], rng
            ),
            "w1_minus_w2_margin_gap_reduction": _interval(
                margin_gap_reduction, strata[mask], rng
            ),
        }

    mask = conflict
    local_strata = strata[mask]
    for name, values in (("W1", final_w1), ("W2", final_w2), ("other options", final_other)):
        summary["final_source_trace"][name] = {}
        for ci, condition in enumerate(CONDITION_NAMES):
            cumulative = np.cumsum(values[ci, mask], axis=-1)
            mean, low, high = _bootstrap(cumulative, local_strata, rng)
            summary["final_source_trace"][name][condition] = {
                "cumulative_mean": mean.tolist(),
                "cumulative_ci_low": low.tolist(),
                "cumulative_ci_high": high.tolist(),
                "final": {"mean": float(mean[-1]), "ci_low": float(low[-1]), "ci_high": float(high[-1])},
            }
        difference = np.cumsum(values[0, mask] - values[1, mask], axis=-1)
        mean, low, high = _bootstrap(difference, local_strata, rng)
        summary["final_source_trace"][name]["Evaluation minus Matched Neutral"] = {
            "cumulative_mean": mean.tolist(),
            "cumulative_ci_low": low.tolist(),
            "cumulative_ci_high": high.tolist(),
            "final": {"mean": float(mean[-1]), "ci_low": float(low[-1]), "ci_high": float(high[-1])},
        }
    margin_difference = np.cumsum(
        (final_w1[0, mask] - final_w2[0, mask])
        - (final_w1[1, mask] - final_w2[1, mask]),
        axis=-1,
    )
    mean, low, high = _bootstrap(margin_difference, local_strata, rng)
    summary["final_source_trace"]["W1 minus W2 margin, Evaluation minus Matched Neutral"] = {
        "cumulative_mean": mean.tolist(),
        "cumulative_ci_low": low.tolist(),
        "cumulative_ci_high": high.tolist(),
        "final": {"mean": float(mean[-1]), "ci_low": float(low[-1]), "ci_high": float(high[-1])},
    }

    role_heat = np.nanmean(
        data["role_mean_norm"][0, conflict] - data["role_mean_norm"][1, conflict], axis=0
    ).T
    summary["downstream_retrieval_peaks_conflict"] = {}
    for role_index, role in enumerate(ROLE_NAMES[1:], start=1):
        peak = int(np.nanargmax(np.abs(role_heat[role_index])))
        summary["downstream_retrieval_peaks_conflict"][role] = {
            "block": int(layers[peak]),
            "evaluation_minus_neutral_mean_norm": float(role_heat[role_index, peak]),
        }

    # Plot one canonical four-panel figure.
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11})
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5), constrained_layout=True)
    colors = {"Evaluation": "#2f8df5", "Matched Neutral": "#f07f2f"}
    styles = {"conflict": "-", "no conflict": "--"}
    for subset_name, subset in (("conflict", conflict), ("no conflict", ~conflict)):
        for ci, condition in enumerate(CONDITION_NAMES):
            values = data["anchor_norm"][ci, subset, :, final_index]
            mean, low, high = _bootstrap(values, strata[subset], rng)
            label = f"{condition}, {subset_name}"
            axes[0, 0].plot(layers, mean, color=colors[condition], linestyle=styles[subset_name], label=label)
            axes[0, 0].fill_between(layers, low, high, color=colors[condition], alpha=0.13)
    axes[0, 0].set_title("A  Period-write retrieval at final decision")
    axes[0, 0].set_xlabel("GLA block")
    axes[0, 0].set_ylabel("Residual contribution norm")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)

    candidates = {
        "W1: original winner": final_w1,
        "W2: fresh remapped winner": final_w2,
        "Other: mean remaining two": final_other,
    }
    candidate_colors = {
        "W1: original winner": "#2f8df5",
        "W2: fresh remapped winner": "#f07f2f",
        "Other: mean remaining two": "#55b96b",
    }
    for name, values in candidates.items():
        difference = np.cumsum(values[0, conflict] - values[1, conflict], axis=-1)
        mean, low, high = _bootstrap(difference, strata[conflict], rng)
        axes[0, 1].plot(layers, mean, color=candidate_colors[name], label=name)
        axes[0, 1].fill_between(layers, low, high, color=candidate_colors[name], alpha=0.15)
    axes[0, 1].axhline(0, color="#777777", linewidth=0.8)
    axes[0, 1].set_title("B  Final-decision effect sourced by period write (Eval−Neutral)")
    axes[0, 1].set_xlabel("GLA block (cumulative)")
    axes[0, 1].set_ylabel("Centered direct A–D contribution")
    axes[0, 1].legend(frameon=False)

    # Exclude the source token itself.  Its immediate write is much larger and
    # otherwise makes every genuinely downstream retrieval invisible.
    heat = role_heat[1:]
    bound = float(np.nanmax(np.abs(heat)))
    image = axes[1, 0].imshow(
        heat, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-bound, vmax=bound,
        extent=(layers.min() - 0.5, layers.max() + 0.5, -0.5, len(ROLE_NAMES[1:]) - 0.5),
    )
    axes[1, 0].set_yticks(range(len(ROLE_NAMES[1:])), ROLE_NAMES[1:])
    axes[1, 0].set_xlabel("GLA block")
    axes[1, 0].set_title("C  Where downstream tokens retrieve it")
    fig.colorbar(image, ax=axes[1, 0], label="Evaluation−Neutral mean contribution norm")

    x = np.asarray([0, 1, 3, 4], dtype=float)
    labels = ["Eval\nconflict", "Neutral\nconflict", "Eval\nno conflict", "Neutral\nno conflict"]
    cells = [
        summary["global_ablation"]["all"]["conflict"]["Evaluation"]["w1_selection_change_pp"],
        summary["global_ablation"]["all"]["conflict"]["Matched Neutral"]["w1_selection_change_pp"],
        summary["global_ablation"]["all"]["no_conflict"]["Evaluation"]["w1_selection_change_pp"],
        summary["global_ablation"]["all"]["no_conflict"]["Matched Neutral"]["w1_selection_change_pp"],
    ]
    means = np.asarray([cell["mean"] for cell in cells])
    error = np.stack([means - [cell["ci_low"] for cell in cells], [cell["ci_high"] for cell in cells] - means])
    axes[1, 1].errorbar(x, means, yerr=error, fmt="o", color="#222222", capsize=4, markersize=7)
    axes[1, 1].axhline(0, color="#777777", linewidth=0.8)
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("Change in W1 selection (percentage points)")
    axes[1, 1].set_title("D  Causal removal from all 48 GLAs")
    fig.suptitle("What the evaluation-closing period writes into GLA memory", fontsize=17)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (output_dir / "global_ablation_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "split", "subset", "condition", "metric", "mean", "ci_low", "ci_high"
        ])
        writer.writeheader()
        for split, split_cell in summary["global_ablation"].items():
            for subset, subset_cell in split_cell.items():
                if subset == "conflict_gap":
                    continue
                for condition, condition_cell in subset_cell.items():
                    for metric, interval in condition_cell.items():
                        if metric == "n":
                            continue
                        writer.writerow({"split": split, "subset": subset, "condition": condition,
                                         "metric": metric, **interval})

    conflict_all = summary["global_ablation"]["all"]["conflict"]
    conflict_gap = summary["global_ablation"]["all"]["conflict_gap"]
    discovery_gap = summary["global_ablation"]["discovery"]["conflict_gap"]
    confirmation_gap = summary["global_ablation"]["confirmation"]["conflict_gap"]
    direct_margin = summary["final_source_trace"][
        "W1 minus W2 margin, Evaluation minus Matched Neutral"
    ]["final"]
    final_norm_difference = np.nanmean(
        data["anchor_norm"][0, conflict, :, final_index]
        - data["anchor_norm"][1, conflict, :, final_index],
        axis=0,
    )
    top_final_norm = np.argsort(-final_norm_difference)[:3]
    top_final_norm_text = ", ".join(
        f"{int(layers[index])} ({final_norm_difference[index]:+.3f})"
        for index in top_final_norm
    )
    report = f"""# Exact source trace of the evaluation-closing GLA write

## Bottom line

This experiment asks what the recurrent memory update made by the period after
`Your answer was incorrect.` (or `lost.`) contributes when later tokens query
each GLA. It does **not** apply another vocabulary lens. Each block is replayed
with only that period write removed, and the resulting residual-stream
difference is measured at every later token.

Natural A-D logits reproduce the trusted 500-question run exactly (maximum
absolute error {max_natural_error:.1f}). The primary subset contains
{int(conflict.sum())} W1 != W2 questions.

The full causal deletion sets the period write to zero in all 48 GLAs at once.
On conflict trials it changes W1 selection by:

- Evaluation: {conflict_all['Evaluation']['w1_selection_change_pp']['mean']:+.1f} pp
  [{conflict_all['Evaluation']['w1_selection_change_pp']['ci_low']:+.1f},
  {conflict_all['Evaluation']['w1_selection_change_pp']['ci_high']:+.1f}].
- Matched Neutral: {conflict_all['Matched Neutral']['w1_selection_change_pp']['mean']:+.1f} pp
  [{conflict_all['Matched Neutral']['w1_selection_change_pp']['ci_low']:+.1f},
  {conflict_all['Matched Neutral']['w1_selection_change_pp']['ci_high']:+.1f}].

Consequently, the natural **{conflict_gap['natural_neutral_minus_evaluation_w1_selection_pp']:.1f}-point** Neutral-minus-Evaluation W1-selection
gap falls to **{conflict_gap['ablated_neutral_minus_evaluation_w1_selection_pp']:.1f}
points**. The deletion removes
**{conflict_gap['gap_reduction_pp']['mean']:.1f} points**
[{conflict_gap['gap_reduction_pp']['ci_low']:.1f},
{conflict_gap['gap_reduction_pp']['ci_high']:.1f}], or
{100 * conflict_gap['fraction_of_natural_gap_removed']['mean']:.1f}%
[{100 * conflict_gap['fraction_of_natural_gap_removed']['ci_low']:.1f},
{100 * conflict_gap['fraction_of_natural_gap_removed']['ci_high']:.1f}] of the
natural gap.
The W1-minus-W2 margin gap shrinks by
{conflict_gap['w1_minus_w2_margin_gap_reduction']['mean']:.3f} logits
[{conflict_gap['w1_minus_w2_margin_gap_reduction']['ci_low']:.3f},
{conflict_gap['w1_minus_w2_margin_gap_reduction']['ci_high']:.3f}].
This replicates under the frozen split: the W1-selection gap reduction is
{discovery_gap['gap_reduction_pp']['mean']:.1f} points
[{discovery_gap['gap_reduction_pp']['ci_low']:.1f},
{discovery_gap['gap_reduction_pp']['ci_high']:.1f}] in discovery and
{confirmation_gap['gap_reduction_pp']['mean']:.1f} points
[{confirmation_gap['gap_reduction_pp']['ci_low']:.1f},
{confirmation_gap['gap_reduction_pp']['ci_high']:.1f}] in confirmation.

The source trace shows where that causal effect is expressed. At the final
decision, the Evaluation-period write is read much more strongly than the
Neutral-period write in several GLAs. The three largest Evaluation-minus-Neutral
final-decision retrieval-norm differences are at blocks {top_final_norm_text}.
The complete cumulative answer-aligned trajectory is shown in Panel B rather
than assigning an onset by visual inspection. By the end, its
Evaluation-minus-Neutral direct W1-versus-W2 contribution
is {direct_margin['mean']:.4f}
[{direct_margin['ci_low']:.4f}, {direct_margin['ci_high']:.4f}]: it favors W2
over W1. The raw direct contribution is small because it is measured before
downstream amplification; the separate global deletion above establishes the
final causal effect.

The corrected route has no reliable entropy effect. Removing the period write
changes Evaluation entropy on conflict trials by
{conflict_all['Evaluation']['entropy_change_bits']['mean']:+.3f} bits
[{conflict_all['Evaluation']['entropy_change_bits']['ci_low']:+.3f},
{conflict_all['Evaluation']['entropy_change_bits']['ci_high']:+.3f}]. The
output-preserved persistent-memory route is therefore more specifically tied to
answer redistribution than the historical broad intervention suggested.

The four panels separate retrieval strength, the final-decision answer effect
sourced by the period write, downstream token location, and the complete causal
deletion. See the definitions below before interpreting cumulative direct
writes as final logits.

[Canonical PNG](../../../../../../figures/{figure_path.name})

## Definitions and limits

- **W1** is the semantic answer chosen on the original first presentation.
- **W2** is the semantic answer chosen by a fresh Baseline under the remapped
  second presentation. It is **not** the runner-up from the first presentation.
- **Other** in Panel B is the mean of the two remaining semantic candidates on
  W1 != W2 trials. Because all four direct A-D contributions are centered,
  `W1 + W2 + 2 * Other = 0`.
- **Panel B** is measured at the final decision, after the remapped second
  presentation has been processed. It therefore shows the later answer-specific
  effect causally sourced by the earlier period write; it does not show which
  answer identities were locally encoded at the moment the write was made.
- **Source trace** is natural GLA output minus a within-block replay with beta
  zeroed only at the evaluation-closing period. It is an exact deletion effect
  inside that recurrent block, including later interactions, but source traces
  from different blocks are not guaranteed to add causally through the whole
  nonlinear model.
- **Direct A-D contribution** unembeds the source-trace residual vector with the
  four canonical answer rows, aligns by semantic identity, and centers across
  options. It is not a final logit.
- **Global ablation** is a separate complete forward in which that period write
  is removed from all 48 GLAs simultaneously.

Earlier complete-residual crossover work localizes decisive task-dependent
final-answer impact much later than many component-level source traces. These
are different measurements: a small isolated causal trace can be present before
it dominates the net residual state.

Machine-readable intervals, including the frozen 251-question discovery and
249-question confirmation splits, are in [`summary.json`](summary.json).
"""
    (output_dir / "REPORT.md").write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze evaluation-period GLA source traces")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--evaluation-results", type=Path, required=True)
    parser.add_argument("--neutral-results", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(
        args.run_dir, args.baseline, args.remapped_baseline, args.remapping_plan,
        args.evaluation_results, args.neutral_results, args.discovery_plan,
        args.output_dir, args.figure, args.seed,
    )


if __name__ == "__main__":
    main()
