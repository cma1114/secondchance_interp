from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONDITION_LABELS = ("Game", "Neutral")
DISPLAY_LABELS = {
    "system_and_header": "System prompt + header",
    "first_task_instruction": "1P task instruction",
    "first_question_stem": "1P question stem/separators",
    "first_R1_line": "1P R1 option line",
    "first_R2_line": "1P R2 option line",
    "first_R3_line": "1P R3 option line",
    "first_R4_line": "1P R4 option line",
    "first_answer_boundary": "1P answer cue + boundary",
    "feedback_sentence": "Feedback sentence",
    "second_answer_instruction": "2P answer-only instruction",
    "second_question_stem": "2P question stem/separators",
    "second_R1_line": "2P R1 option line",
    "second_R2_line": "2P R2 option line",
    "second_R3_line": "2P R3 option line",
    "second_R4_line": "2P R4 option line",
    "chat_separators_other": "Chat separators / other",
}


def _normal_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--prior-matched-results", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    with np.load(args.prior_matched_results, allow_pickle=False) as loaded:
        prior = {key: loaded[key] for key in loaded.files}

    qids = arrays["question_ids"].astype(str)
    layers = arrays["ordinary_layers_one_based"].astype(int)
    ranks = arrays["ranks"].astype(str)
    source_bins = arrays["source_bins"].astype(str)
    attention = arrays["attention_mass"].astype(float)
    if len(qids) != 500 or not arrays["completed"].astype(bool).all():
        raise RuntimeError("Expected all 500 questions complete")
    if not np.array_equal(layers, np.arange(4, 65, 4)):
        raise RuntimeError(f"Expected all ordinary-attention layers 4--64: {layers}")
    expected_shape = (2, 16, 500, 4, len(source_bins))
    if attention.shape != expected_shape or not np.isfinite(attention).all():
        raise RuntimeError(f"Invalid attention array: {attention.shape}")
    if float(np.max(arrays["max_partition_error"])) > 0.02:
        raise RuntimeError("Attention source bins do not exhaust the distribution")

    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    confirmation = ~discovery
    if int(discovery.sum()) != 251 or int(confirmation.sum()) != 249:
        raise RuntimeError("Frozen discovery/confirmation split changed")

    bin_index = {name: index for index, name in enumerate(source_bins)}
    matching = np.empty((2, 16, 500, 4), dtype=float)
    for rank in range(4):
        matching[..., rank] = attention[..., rank, bin_index[f"first_R{rank + 1}_line"]]
    if not np.array_equal(qids, prior["question_ids"].astype(str)):
        raise RuntimeError("Question order differs from prior matched-line trajectory")
    matching_error = np.abs(matching - prior["attention_mass"].astype(float))

    same_answer = arrays["natural_logits"].argmax(-1) == arrays[
        "trusted_natural_logits"
    ].argmax(-1)
    means = attention[:, :, confirmation].mean(axis=2)
    discovery_means = attention[:, :, discovery].mean(axis=2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "attention_distribution.csv"
    rows: list[list[object]] = []
    heldout = attention[:, :, confirmation]
    for ci, condition in enumerate(CONDITION_LABELS):
        for li, layer in enumerate(layers):
            for ri, rank in enumerate(ranks):
                for si, source in enumerate(source_bins):
                    mean, low, high = _normal_interval(heldout[ci, li, :, ri, si])
                    rows.append(
                        [condition, int(layer), rank, source, int(confirmation.sum()), mean, low, high]
                    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["condition", "layer", "target_2p_rank", "source_region", "n", "mean", "ci_low", "ci_high"]
        )
        writer.writerows(rows)

    selected_layers = (4, 12, 28, 36, 44, 48, 52, 56, 60, 64)
    top_sources: dict[str, object] = {}
    for ci, condition in enumerate(CONDITION_LABELS):
        condition_rows: dict[str, object] = {}
        for layer in selected_layers:
            li = int(np.flatnonzero(layers == layer)[0])
            rank_rows: dict[str, object] = {}
            for ri, rank in enumerate(ranks):
                order = np.argsort(-means[ci, li, ri])[:6]
                rank_rows[rank] = [
                    {
                        "source": str(source_bins[si]),
                        "mean_attention": float(means[ci, li, ri, si]),
                    }
                    for si in order
                ]
            condition_rows[str(layer)] = rank_rows
        top_sources[condition] = condition_rows

    summary = {
        "validation": {
            "questions": 500,
            "discovery": int(discovery.sum()),
            "confirmation": int(confirmation.sum()),
            "layers": layers.tolist(),
            "source_bins": source_bins.tolist(),
            "max_attention_partition_error": float(
                np.max(arrays["max_partition_error"])
            ),
            "natural_answer_agreement_with_trusted": float(same_answer.mean()),
            "prior_matched_line_mean_absolute_error": float(matching_error.mean()),
            "prior_matched_line_p99_absolute_error": float(
                np.quantile(matching_error, 0.99)
            ),
            "maximum_discovery_confirmation_mean_difference": float(
                np.max(np.abs(means - discovery_means))
            ),
        },
        "measurement": {
            "query": "All tokens in each complete second-presentation option line, averaged over query tokens and attention heads.",
            "source": "Every non-padding prompt token assigned to exactly one named source region.",
            "conditions": ["Game (incorrect)", "Neutral (lost)"],
            "rank_alignment": "R1--R4 are the first-presentation evidence ranks of the semantic candidates.",
        },
        "top_sources_at_selected_layers": top_sources,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    layer_index = {int(layer): index for index, layer in enumerate(layers)}
    first_line_indices = [
        bin_index[f"first_R{rank + 1}_line"] for rank in range(4)
    ]
    all_first_options = means[..., first_line_indices].sum(axis=-1).mean(axis=(0, 2))
    matching_heldout = matching[:, :, confirmation].mean(axis=(0, 2, 3))
    other_first_options = all_first_options - matching_heldout
    answer_boundary = means[..., bin_index["first_answer_boundary"]].mean(axis=(0, 2))
    feedback_by_task = means[..., bin_index["feedback_sentence"]].mean(axis=2)
    repeated_stem = means[..., bin_index["second_question_stem"]].mean(axis=(0, 2))
    own_second_prefix = np.asarray([
        np.mean([
            means[condition, li, rank, bin_index[f"second_R{rank + 1}_line"]]
            for condition in range(2)
            for rank in range(4)
        ])
        for li in range(len(layers))
    ])

    def pct(series: np.ndarray, layer: int) -> str:
        return f"{100 * float(series[layer_index[layer]]):.1f}%"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    vmax = float(means.max())
    fig, axes = plt.subplots(
        2, 4, figsize=(20, 10.8), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    for ci, condition in enumerate(CONDITION_LABELS):
        for ri, rank in enumerate(ranks):
            axis = axes[ci, ri]
            image = axis.imshow(
                means[ci, :, ri].T,
                origin="upper",
                aspect="auto",
                interpolation="nearest",
                cmap="magma",
                vmin=0,
                vmax=vmax,
            )
            matching_row = bin_index[f"first_R{ri + 1}_line"]
            axis.add_patch(
                Rectangle(
                    (-0.5, matching_row - 0.5),
                    len(layers),
                    1,
                    fill=False,
                    edgecolor="#28d7b5",
                    linewidth=2.2,
                )
            )
            axis.set_title(f"{condition}: queries in 2P {rank} line", fontsize=11.5)
            axis.set_xticks(np.arange(len(layers)))
            axis.set_xticklabels(layers, rotation=45, ha="right")
            if ci == 1:
                axis.set_xlabel("Ordinary-attention layer")
    axes[0, 0].set_yticks(np.arange(len(source_bins)))
    axes[0, 0].set_yticklabels([DISPLAY_LABELS[name] for name in source_bins])
    axes[0, 0].set_ylabel("Source region in causal prefix")
    axes[1, 0].set_ylabel("Source region in causal prefix")
    if image is None:
        raise RuntimeError("No heatmap was drawn")
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.82, pad=0.015)
    colorbar.set_label("Mean attention mass (held-out 249 questions)")
    fig.suptitle(
        "Where each second-presentation option line attends\n"
        "Absolute Game and Neutral distributions; green outline marks the matching 1P line",
        fontsize=16,
        fontweight="bold",
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = f"""# Exhaustive attention distribution from second-presentation option lines

## What was measured

For every SimpleMC question, every ordinary-attention layer **4--64**, and both
Game (`incorrect`) and Neutral (`lost`), this analysis asks where each complete
second-presentation option line attends.  The four target lines are aligned by
the candidate's first-presentation rank, R1 through R4.  Attention is averaged
over all tokens in the target line and all attention heads.

Every non-padding source token belongs to exactly one row in the figure.  Thus
the rows are an exhaustive distribution, not a selected set of destinations.
The green outline marks the first-presentation line containing the same answer
content as the second-presentation target line.

## What the distribution shows

The matching 1P line is important, but it is not the only substantial historical
source. Averaging across the four 2P target lines on held-out questions:

- At layer 4, all four 1P option lines receive **{pct(all_first_options, 4)}** of attention. Their
  combined share is **{pct(all_first_options, 12)} at layer 12**,
  **{pct(all_first_options, 36)} at layer 36**, **{pct(all_first_options, 52)}
  at layer 52**, and **{pct(all_first_options, 64)} at layer 64**.
- The single semantically matching 1P line rises from **{pct(matching_heldout, 4)} at layer 4** to
  **{pct(matching_heldout, 12)} at layer 12**, and is **{pct(matching_heldout, 48)}
  at layer 48** and **{pct(matching_heldout, 52)} at layer 52**. The other
  three 1P option lines jointly receive **{pct(other_first_options, 12)} at
  layer 12** and **{pct(other_first_options, 52)} at layer 52**. Thus each 2P option line reads the whole first-pass
  candidate set while preferentially reading its semantic match.
- The 1P answer cue and decision boundary receive **{pct(answer_boundary, 36)}
  at layer 36**, **{pct(answer_boundary, 60)} at layer 60**, and
  **{pct(answer_boundary, 64)} at layer 64**. This is a plausible place
  for relational first-pass information, but the attention measurement does
  not establish what information is read or whether it is causally necessary.
- Game and Neutral allocate attention differently to the feedback sentence.
  Game attention is **{pct(feedback_by_task[0], 28)}, {pct(feedback_by_task[0], 36)},
  {pct(feedback_by_task[0], 44)}, and {pct(feedback_by_task[0], 48)}** at layers
  28, 36, 44, and 48; Neutral is **{pct(feedback_by_task[1], 28)},
  {pct(feedback_by_task[1], 36)}, {pct(feedback_by_task[1], 44)}, and
  {pct(feedback_by_task[1], 48)}**. These are direct reads
  of the policy-bearing sentence, not evidence that the sentence itself
  contains candidate rankings.
- The repeated question stem is also a major source: **{pct(repeated_stem, 4)}
  at layer 4**, **{pct(repeated_stem, 12)} at layer 12**, and
  **{pct(repeated_stem, 36)} at layer 36**. The current 2P option line's own
  causal prefix receives **{pct(own_second_prefix, 36)} at layer 36**,
  **{pct(own_second_prefix, 52)} at layer 52**, and
  **{pct(own_second_prefix, 64)} at layer 64**.

The central new fact is therefore that the 2P line has simultaneous access to
three ingredients: its matching 1P semantic line, the other three 1P candidate
lines, and the first-answer boundary, while Game additionally reads the
`incorrect` feedback much more strongly at layers 28--48. This makes a
distributed rank/policy computation plausible. It does **not** yet show which
of those nonmatching reads carries winner rank or whether any one is causally
required.

## Validation

- All 500 questions completed; the canonical figure uses the frozen 249-question confirmation split.
- Maximum error when summing the exhaustive source rows to one: **{summary['validation']['max_attention_partition_error']:.6f}**.
- Natural answer agreement with the trusted canonical outputs: **{summary['validation']['natural_answer_agreement_with_trusted']:.1%}**.
- Mean absolute error against the previously measured matching-line trajectory: **{summary['validation']['prior_matched_line_mean_absolute_error']:.6f}** (99th percentile **{summary['validation']['prior_matched_line_p99_absolute_error']:.6f}**).

## Artifacts

- Canonical figure: `{args.figure}`
- Cell-level means and confidence intervals: `{csv_path}`
- Machine-readable summary and top source regions: `{summary_path}`
"""
    hand_curated = args.output_dir / "HAND_CURATED_FINDINGS.md"
    if hand_curated.exists():
        report += (
            "\n---\n\n"
            + hand_curated.read_text().strip()
            + "\n"
        )
    (args.output_dir / "REPORT.md").write_text(report)
    print(json.dumps(summary["validation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
