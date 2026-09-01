from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .collect_evaluation_period_source_trace import ANCHOR_NAMES
from .semantic_mapping import displayed_argmax_to_semantic_indices


LETTERS = "ABCD"
CONDITIONS = ("Evaluation", "Matched Neutral")


def _bootstrap(values: np.ndarray, strata: np.ndarray, seed: int, draws: int = 2000):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, *values.shape[1:]), dtype=float)
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    for start in range(0, draws, 100):
        stop = min(start + 100, draws)
        total = np.zeros((stop - start, *values.shape[1:]), dtype=float)
        for group in groups:
            selected = rng.choice(group, size=(stop - start, len(group)), replace=True)
            total += values[selected].sum(axis=1)
        samples[start:stop] = total / len(values)
    return values.mean(axis=0), np.quantile(samples, 0.025, axis=0), np.quantile(samples, 0.975, axis=0)


def _interval(values: np.ndarray, strata: np.ndarray, seed: int) -> dict[str, float]:
    mean, low, high = _bootstrap(values, strata, seed)
    return {"mean": float(mean.squeeze()), "ci_low": float(low.squeeze()), "ci_high": float(high.squeeze())}


def _align_content(values: np.ndarray, qids: list[str], mappings: dict[str, dict]) -> np.ndarray:
    """Convert last-axis displayed letters into original semantic-option letters."""
    result = np.empty_like(values)
    for qi, qid in enumerate(qids):
        original_to_new = mappings[qid]["original_to_new"]
        for oi, original in enumerate(LETTERS):
            ni = LETTERS.index(original_to_new[original])
            result[:, qi, ..., oi] = values[:, qi, ..., ni]
    return result


def _candidate_advantage(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Candidate minus mean of the other three; condition and question lead."""
    result = np.empty(values.shape[:-1], dtype=float)
    for qi, letter in enumerate(candidates):
        ci = LETTERS.index(letter)
        other = [i for i in range(4) if i != ci]
        result[:, qi] = values[:, qi, ..., ci] - np.take(
            values[:, qi], other, axis=-1
        ).mean(axis=-1)
    return result


def _semantic_value(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    result = np.empty(values.shape[:-1], dtype=float)
    for qi, letter in enumerate(candidates):
        result[:, qi] = values[:, qi, ..., LETTERS.index(letter)]
    return result


def _rank_order(baseline: dict, qids: list[str]) -> np.ndarray:
    order = np.empty((len(qids), 4), dtype=int)
    for qi, qid in enumerate(qids):
        row = baseline[qid]
        if "aggregated_ad_logits" in row:
            evidence = np.asarray(row["aggregated_ad_logits"], dtype=float)
        else:
            probs = row["probs"]
            evidence = np.asarray([float(probs.get(letter, 0.0)) for letter in LETTERS])
        winner = LETTERS.index(row["answer"])
        rest = [i for i in range(4) if i != winner]
        order[qi] = [winner] + sorted(rest, key=lambda i: evidence[i], reverse=True)
    return order


def _align_ranks(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    for qi in range(values.shape[1]):
        result[:, qi] = np.take(values[:, qi], order[qi], axis=-1)
    return result


def _load_source(run_dir: Path):
    shards = sorted((run_dir / "cohorts").glob("cohort_*.npz"))
    qids: list[str] = []
    natural, ablated, anchor_norm, anchor_ad = [], [], [], []
    for path in shards:
        with np.load(path, allow_pickle=False) as z:
            qids.extend(z["question_ids"].astype(str).tolist())
            natural.append(z["natural_logits"].astype(np.float32))
            ablated.append(z["ablated_logits"].astype(np.float32))
            norm = z["trace_norm"].astype(np.float32)
            ad = z["trace_ad"].astype(np.float32)
            anchors = z["anchor_relative_positions"].astype(int)
        c, b, layers, _ = norm.shape
        a_norm = np.empty((c, b, layers, len(ANCHOR_NAMES)), dtype=np.float32)
        a_ad = np.empty((c, b, layers, len(ANCHOR_NAMES), 4), dtype=np.float32)
        for ci in range(c):
            for bi in range(b):
                for ai, position in enumerate(anchors[ci, bi]):
                    a_norm[ci, bi, :, ai] = norm[ci, bi, :, position]
                    a_ad[ci, bi, :, ai] = ad[ci, bi, :, position]
        anchor_norm.append(a_norm)
        anchor_ad.append(a_ad)
    return {
        "qids": qids,
        "natural": np.concatenate(natural, axis=1),
        "ablated": np.concatenate(ablated, axis=1),
        "anchor_norm": np.concatenate(anchor_norm, axis=1),
        "anchor_ad": np.concatenate(anchor_ad, axis=1),
        "n_shards": len(shards),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        out = np.empty(len(v), dtype=float)
        out[order] = np.arange(len(v), dtype=float)
        return out
    rx, ry = ranks(np.asarray(x)), ranks(np.asarray(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze(args):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    source = _load_source(args.source_run)
    qids = source["qids"]
    with np.load(args.period_jlens, allow_pickle=False) as z:
        jqids = z["question_ids"].astype(str).tolist()
        jlens = z["bare_ad_scores"].astype(np.float32)
    if qids != jqids:
        raise RuntimeError("Source-trace and period-JLens question order differs")

    mappings = {row["question_id"]: row for row in json.loads(args.remapping_plan.read_text())["rows"]}
    baseline = json.loads(args.baseline.read_text())["results"]
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    eval_rows = json.loads(args.evaluation_results.read_text())["results"]
    neutral_rows = json.loads(args.neutral_results.read_text())["results"]
    discovery = set(json.loads(args.discovery_plan.read_text())["question_ids"])

    w1 = np.asarray([baseline[qid]["answer"] for qid in qids])
    w2 = np.asarray([remapped[qid]["answer_original_content"] for qid in qids])
    conflict = w1 != w2
    discovery_mask = np.asarray([qid in discovery for qid in qids])
    rank_order = _rank_order(baseline, qids)
    strata = w1

    natural = _align_content(source["natural"], qids, mappings)
    ablated = _align_content(source["ablated"], qids, mappings)
    common_ad_shift = (natural - ablated).mean(axis=-1)
    natural -= natural.mean(axis=-1, keepdims=True)
    ablated -= ablated.mean(axis=-1, keepdims=True)
    intact_effect = natural - ablated
    rank_effect = _align_ranks(intact_effect, rank_order)
    natural_choice = displayed_argmax_to_semantic_indices(
        source["natural"], [mappings[qid] for qid in qids]
    )
    w1_index = np.asarray([LETTERS.index(x) for x in w1])
    switched = natural_choice != w1_index[None, :]

    jlens = _align_content(jlens, qids, mappings)
    jlens -= jlens.mean(axis=-1, keepdims=True)
    jlens_w1_adv = _candidate_advantage(jlens, w1)
    jlens_contrast = jlens_w1_adv[0] - jlens_w1_adv[1]

    anchor_ad = _align_content(source["anchor_ad"], qids, mappings)
    anchor_ad -= anchor_ad.mean(axis=-1, keepdims=True)
    source_w1_adv = _candidate_advantage(anchor_ad, w1)
    source_contrast = source_w1_adv[0] - source_w1_adv[1]

    # The exact source trace is per GLA; summing is a direct-readout summary,
    # not a whole-model causal decomposition.
    action_ai = ANCHOR_NAMES.index("action-clause end")
    final_ai = ANCHOR_NAMES.index("final decision")
    action_direct = source_contrast[:, :, action_ai].sum(axis=1)
    final_direct = source_contrast[:, :, final_ai].sum(axis=1)
    eval_w1_causal = _candidate_advantage(intact_effect, w1)[0]

    masks = {"conflict": conflict, "no_conflict": ~conflict}
    summary = {
        "definitions": {
            "W1": "Semantic answer selected on the original first-presentation Baseline.",
            "W2": "Semantic answer selected by a fresh Baseline under the remapped presentation.",
            "intact period-write effect": "Natural logits minus logits after deleting the evaluation-period write from all 48 GLAs.",
            "candidate advantage": "Candidate's centered A-D score minus the mean score of the other three candidates.",
            "source-trace direct readout": "Sum across per-GLA direct A-D source traces; descriptive and not an additive whole-model causal estimate.",
        },
        "validation": {
            "questions": len(qids), "source_shards": source["n_shards"],
            "question_order_match": True, "conflict": int(conflict.sum()),
            "no_conflict": int((~conflict).sum()),
        },
        "period_jlens": {}, "source_transmission": {}, "global_deletion": {},
        "old_feedback_end_replacement": json.loads(args.old_replacement_summary.read_text())["effects"]["Neutral into Game"]["all_layers"],
    }

    # Summary statistics for key layers and outcome splits.
    for subset, mask in masks.items():
        summary["period_jlens"][subset] = {}
        for anchor_name, ai in (("evaluation_period", 0), ("action_period", 1)):
            summary["period_jlens"][subset][anchor_name] = {}
            for layer in (42, 43, 49, 56, 64):
                summary["period_jlens"][subset][anchor_name][f"L{layer}"] = _interval(
                    jlens_contrast[mask, ai, layer - 1], strata[mask], 1000 + layer + ai
                )
        summary["global_deletion"][subset] = {}
        for ci, condition in enumerate(CONDITIONS):
            summary["global_deletion"][subset][condition] = {
                f"rank_{rank+1}": _interval(rank_effect[ci, mask, rank], strata[mask], 2000 + ci * 20 + rank)
                for rank in range(4)
            }
            summary["global_deletion"][subset][condition]["common_ad_offset"] = _interval(
                common_ad_shift[ci, mask], strata[mask], 2090 + ci
            )
        summary["source_transmission"][subset] = {
            "action_direct_vs_final_causal_spearman": _spearman(action_direct[mask], eval_w1_causal[mask]),
            "final_direct_vs_final_causal_spearman": _spearman(final_direct[mask], eval_w1_causal[mask]),
        }

    for outcome, mask in (("game_switched", switched[0]), ("game_repeated", ~switched[0])):
        summary["source_transmission"][outcome] = {
            "n": int(mask.sum()),
            "action_period_jlens_contrast_L56": _interval(jlens_contrast[mask, 1, 55], strata[mask], 3100),
            "action_period_source_direct_sum": _interval(action_direct[mask], strata[mask], 3101),
            "final_source_direct_sum": _interval(final_direct[mask], strata[mask], 3102),
        }

    # Discovery/confirmation stability for the central rank-1 causal effect.
    summary["global_deletion"]["split_stability"] = {}
    for split_name, split in (("discovery", discovery_mask), ("confirmation", ~discovery_mask)):
        summary["global_deletion"]["split_stability"][split_name] = {}
        for subset, subset_mask in masks.items():
            mask = split & subset_mask
            summary["global_deletion"]["split_stability"][split_name][subset] = {
                condition: _interval(rank_effect[ci, mask, 0], strata[mask], 4000 + ci)
                for ci, condition in enumerate(CONDITIONS)
            }

    # Figure.
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.05, 0.9])
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0]); ax_f = fig.add_subplot(gs[2, 1])
    colors = {"evaluation": "#377eb8", "action": "#e41a1c"}

    layers = np.arange(1, 65)
    for ai, (label, color) in enumerate(colors.items()):
        for subset, mask, ls in (("conflict", conflict, "-"), ("no conflict", ~conflict, "--")):
            mean, low, high = _bootstrap(jlens_contrast[mask, ai], strata[mask], 5000 + ai)
            ax_a.plot(layers, mean, color=color, ls=ls, label=f"{label}; {subset}")
            ax_a.fill_between(layers, low, high, color=color, alpha=0.12)
    ax_a.axhline(0, color="0.45", lw=0.8); ax_a.axvline(49, color="0.55", lw=0.8, ls=":")
    ax_a.set(title="A  JLens: Game-minus-Neutral W1 advantage", xlabel="Residual readout", ylabel="W1 vs other three (score units)")
    ax_a.legend(frameon=False, fontsize=8, ncol=2)

    x = np.arange(len(ANCHOR_NAMES))
    for ci, (condition, color) in enumerate(zip(CONDITIONS, ("#377eb8", "#e41a1c"))):
        values = source["anchor_norm"][ci].mean(axis=1)
        mean, low, high = _bootstrap(values, strata, 6000 + ci)
        ax_b.plot(x, mean, marker="o", color=color, label=condition)
        ax_b.fill_between(x, low, high, color=color, alpha=0.13)
    ax_b.set_xticks(x, [s.replace("-", " ").replace(" end", "") for s in ANCHOR_NAMES], rotation=38, ha="right")
    ax_b.set(title="B  Evaluation-period write persists downstream", ylabel="Mean source-trace norm across GLAs")
    ax_b.legend(frameon=False)

    heatmaps = []
    for ax, subset, mask, title in ((ax_c, "conflict", conflict, "C  W1-specific trace: conflict"), (ax_d, "no conflict", ~conflict, "D  W1-specific trace: no conflict")):
        matrix = source_contrast[mask].mean(axis=0).T  # anchors x GLA blocks
        heatmaps.append(matrix)
        limit = max(abs(matrix).max(), 1e-6)
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit), origin="lower")
        ax.set_yticks(range(len(ANCHOR_NAMES)), [s.replace("-", " ").replace(" end", "") for s in ANCHOR_NAMES])
        ax.set_xticks(np.arange(0, 48, 4), np.arange(1, 49, 4))
        ax.set(title=title, xlabel="GLA index", ylabel="Downstream position")
        fig.colorbar(im, ax=ax, shrink=0.75, label="Game − Neutral W1 advantage")

    offsets = np.asarray([-0.18, 0.18])
    rank_x = np.arange(1, 5)
    for ax, subset, mask, title in ((ax_e, "conflict", conflict, "E  Causal effect of intact evaluation-period write: conflict"), (ax_f, "no_conflict", ~conflict, "F  Causal effect of intact evaluation-period write: no conflict")):
        for ci, (condition, color) in enumerate(zip(CONDITIONS, ("#377eb8", "#e41a1c"))):
            mean, low, high = _bootstrap(rank_effect[ci, mask], strata[mask], 7000 + ci)
            ax.errorbar(rank_x + offsets[ci], mean, yerr=[mean-low, high-mean], fmt="o", capsize=3, color=color, label=condition)
        ax.axhline(0, color="0.45", lw=0.8)
        ax.set_xticks(rank_x, ["W1\n(rank 1)", "Rank 2", "Rank 3", "Rank 4"])
        ax.set(title=title, ylabel="Natural − ablated centered logit", xlabel="Original Baseline rank")
        ax.legend(frameon=False)

    fig.suptitle("How the evaluation-period state reaches and reshapes the final decision", fontsize=16)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def fmt(v): return f"{v['mean']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]"
    nc_e = summary["global_deletion"]["no_conflict"]["Evaluation"]
    nc_n = summary["global_deletion"]["no_conflict"]["Matched Neutral"]
    cf_e = summary["global_deletion"]["conflict"]["Evaluation"]
    action56_c = summary["period_jlens"]["conflict"]["action_period"]["L56"]
    action56_n = summary["period_jlens"]["no_conflict"]["action_period"]["L56"]
    report = f"""# Existing-data audit: evaluation state to final decision

## Bottom line

No new model run was used. The saved tensors show that the evaluation-period
state is not merely generic noise. In the global deletion, the intact Game
write has its most negative centered effect on W1 in both conflict and
non-conflict trials. On non-conflict trials its rank-resolved effects are
{fmt(nc_e['rank_1'])} for W1, {fmt(nc_e['rank_2'])} for rank 2,
{fmt(nc_e['rank_3'])} for rank 3, and {fmt(nc_e['rank_4'])} for rank 4.
The entropy increase is therefore principally a consequence of depressing the
dominant W1 candidate, not evidence of isotropic noise.

The action-closing period is representationally downstream of that state. Its
JLens Game-minus-Neutral W1-advantage contrast is {fmt(action56_c)} at L56 on
conflict trials but {fmt(action56_n)} on non-conflict trials. It is also
{fmt(summary['source_transmission']['game_switched']['action_period_jlens_contrast_L56'])}
on trials where Game eventually switches, versus
{fmt(summary['source_transmission']['game_repeated']['action_period_jlens_contrast_L56'])}
where it repeats W1. Thus the final-period A-D representation is behaviorally
aligned, not merely a condition-average vocabulary curiosity.

The evaluation-period source trace remains large in norm at the action period,
but Panels C-D show that its **direct W1-aligned component there is weak**. Its
per-question action-period direct readout has Spearman correlation only
{summary['source_transmission']['conflict']['action_direct_vs_final_causal_spearman']:+.3f}
with the final causal W1 effect on conflict trials. The available state is
therefore high-dimensional: the explicit answer-targeted representation in the
complete action-period residual cannot be equated with a directly propagated
W1 vector from the earlier write.

![Canonical transmission figure](/Users/christopherackerman/repos/secondchance_interp/{args.figure})

## Panel guide

- **A:** question-level JLens A-D evidence at both feedback periods. Negative
  values mean Game represents W1 less strongly than Matched Neutral.
- **B:** the norm of the exact evaluation-period GLA source trace at successive
  downstream prompt positions. This shows persistence, not answer identity.
- **C-D:** for each GLA and later position, the Game-minus-Neutral direct
  W1-versus-other contribution of the evaluation-period source trace. These are
  direct readouts, not additive whole-model causal effects.
- **E-F:** the actual causal effect of the intact evaluation-period write on
  final centered logits, computed as natural minus globally ablated. These are
  the decisive targeting panels.

## Targeting versus flattening

On conflict trials, Game's intact period-write effects by original Baseline
rank are W1 {fmt(cf_e['rank_1'])}, rank 2 {fmt(cf_e['rank_2'])}, rank 3
{fmt(cf_e['rank_3'])}, and rank 4 {fmt(cf_e['rank_4'])}. On non-conflict trials
the corresponding Game effects are W1 {fmt(nc_e['rank_1'])}, rank 2
{fmt(nc_e['rank_2'])}, rank 3 {fmt(nc_e['rank_3'])}, and rank 4
{fmt(nc_e['rank_4'])}. The W1 effect is directionally the strongest negative
effect in both subsets. This supports a targeted-W1 operation whose
distributional consequence is flattening when W1 is the current winner.

Matched Neutral does change the raw A-D logits. Its non-conflict centered rank effects, however, are W1
{fmt(nc_n['rank_1'])}, rank 2 {fmt(nc_n['rank_2'])}, rank 3
{fmt(nc_n['rank_3'])}, and rank 4 {fmt(nc_n['rank_4'])}. This is why causal
comparisons must distinguish common offsets from candidate redistribution.
Crucially, these centered effects also resolve the apparently
large raw Neutral W1-logit response reported earlier: most of it is a common
A-D offset ({fmt(nc_n['common_ad_offset'])} for natural minus ablated on
non-conflict trials), which cannot change the relative A-D distribution. Once
that offset is removed, Neutral has essentially no rank-specific effect.

## What the final action period contributes

The current files establish three things:

1. The action period contains a strong readable exclusion policy in the mean
   full-vocabulary JLens.
2. Question-specific A-D evidence there is negative for W1 on conflict and
   eventual-switch trials but reverses or disappears on non-conflict and
   repeat trials (Panel A and `summary.json`).
3. The evaluation-period causal write remains present there in high-dimensional
   norm, but its directly readable W1 component is weak (Panels B-D).

They do **not** establish that the action-period update is necessary or
sufficient. The earlier standard-prompt all-layer residual replacement at the
final feedback period restored 13.6% of the winner-advantage gap, 23.4% of the
spread gap, and 26.0% of the entropy gap, but changed net switching by only
-1.2 percentage points with a confidence interval spanning zero. The current
action-matched final-period token-state swaps were likewise small. Thus the
best existing conclusion is that the action period is a behaviorally aligned
downstream decision state, not yet a demonstrated causal bottleneck.

## Limits

- Full-vocabulary JLens states were retained only as across-question means, so
  `exclude`-token strength cannot be correlated with individual switching
  without another run.
- Summed per-GLA source traces are descriptive direct readouts. Only the global
  write deletion and bidirectional transplant are whole-model causal tests.
- Intermediate-state collection slightly changes low-order SDPA numerics; the
  separate period-JLens report documents the 94.4%/96.6% A-D argmax agreement.

## Files

- Machine-readable summary: [summary.json](summary.json)
- Figure: [qwen36_action_matched_period_transmission.png](/Users/christopherackerman/repos/secondchance_interp/{args.figure})
"""
    (args.output_dir / "REPORT.md").write_text(report)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--period-jlens", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--evaluation-results", type=Path, required=True)
    parser.add_argument("--neutral-results", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--old-replacement-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
