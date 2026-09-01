from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("Game", "Neutral")
METRICS = (
    "all_1p_options",
    "matching_1p_option",
    "matching_selectivity",
)
METRIC_LABELS = {
    "all_1p_options": "Attention to all four 1P option lines",
    "matching_1p_option": "Attention to matching 1P option line",
    "matching_selectivity": "Match minus mean nonmatch",
}
PRESENTATIONS = ("Remapped", "Non-remapped")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _validate(
    remapped: dict[str, np.ndarray],
    identity: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for name, arrays in (("remapped", remapped), ("non-remapped", identity)):
        if len(arrays["question_ids"]) != 500 or not arrays["completed"].astype(bool).all():
            raise RuntimeError(f"{name}: expected all 500 questions complete")
        if not np.isfinite(arrays["attention_mass"]).all():
            raise RuntimeError(f"{name}: non-finite attention values")
        if not np.isfinite(arrays["natural_logits"]).all():
            raise RuntimeError(f"{name}: non-finite natural logits")
        if float(np.max(arrays["max_partition_error"])) > 0.02:
            raise RuntimeError(f"{name}: attention partition does not sum to one")
    qids = remapped["question_ids"].astype(str)
    layers = remapped["ordinary_layers_one_based"].astype(int)
    bins = remapped["source_bins"].astype(str)
    for key in ("question_ids", "ordinary_layers_one_based", "ranks", "source_bins"):
        if not np.array_equal(remapped[key].astype(str), identity[key].astype(str)):
            raise RuntimeError(f"Runs disagree on {key}")
    if not np.array_equal(layers, np.arange(4, 65, 4)):
        raise RuntimeError(f"Expected all 16 ordinary-attention layers, got {layers}")
    return qids, layers, bins


def _metrics(arrays: dict[str, np.ndarray], bins: np.ndarray) -> dict[str, np.ndarray]:
    attention = arrays["attention_mass"].astype(float)
    index = {name: i for i, name in enumerate(bins)}
    first = [index[f"first_R{rank}_line"] for rank in range(1, 5)]
    all_options = attention[..., first].sum(axis=-1)
    matching = np.empty(attention.shape[:-1], dtype=float)
    for rank in range(4):
        matching[..., rank] = attention[..., rank, first[rank]]
    nonmatching_mean = (all_options - matching) / 3.0
    return {
        "all_1p_options": all_options,
        "matching_1p_option": matching,
        "matching_selectivity": matching - nonmatching_mean,
        "matching_share_of_1p": matching / np.maximum(all_options, 1e-12),
    }


def _bootstrap_curve(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Input is layer x question after averaging the four target ranks.
    means = values.mean(axis=1)
    sampled = np.empty((draws, values.shape[0]), dtype=np.float32)
    for draw in range(draws):
        indices = rng.integers(0, values.shape[1], size=values.shape[1])
        sampled[draw] = values[:, indices].mean(axis=1)
    return means, np.quantile(sampled, 0.025, axis=0), np.quantile(sampled, 0.975, axis=0)


def _interval(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, float | list[float]]:
    mean = float(values.mean())
    sampled = np.empty(draws, dtype=np.float32)
    for draw in range(draws):
        indices = rng.integers(0, len(values), size=len(values))
        sampled[draw] = values[indices].mean()
    low, high = np.quantile(sampled, (0.025, 0.975))
    return {"mean": mean, "ci95": [float(low), float(high)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remapped-results", type=Path, required=True)
    parser.add_argument("--nonremapped-results", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    remapped = _load(args.remapped_results)
    identity = _load(args.nonremapped_results)
    qids, layers, bins = _validate(remapped, identity)
    values = {
        "Remapped": _metrics(remapped, bins),
        "Non-remapped": _metrics(identity, bins),
    }
    remapping = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    bin_index = {name: i for i, name in enumerate(bins)}
    remapped_attention = remapped["attention_mass"].astype(float)
    rank_letters = remapped["rank_letters"].astype(str)
    same_letter_source = np.empty(remapped_attention.shape[:-1], dtype=float)
    for qi, qid in enumerate(qids):
        for rank in range(4):
            first_letter = rank_letters[qi, rank]
            second_letter = remapping[qid]["original_to_new"][first_letter]
            source_rank = int(np.flatnonzero(rank_letters[qi] == second_letter)[0])
            same_letter_source[:, :, qi, rank] = remapped_attention[
                :, :, qi, rank, bin_index[f"first_R{source_rank + 1}_line"]
            ]
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    splits = {"discovery": discovery, "confirmation": ~discovery}
    if int(discovery.sum()) != 251 or int((~discovery).sum()) != 249:
        raise RuntimeError("Frozen split changed")

    rng = np.random.default_rng(args.seed)
    summary: dict[str, object] = {
        "definitions": {
            "query": "Every token in each complete 2P option line, averaged over tokens, heads, and the four candidate ranks.",
            "all_1p_options": "Total ordinary-attention mass to all four first-presentation option lines.",
            "matching_1p_option": "Ordinary-attention mass to the first-presentation line containing the same semantic candidate.",
            "matching_selectivity": "Matching-line mass minus the mean mass to one nonmatching 1P option line.",
        },
        "validation": {
            "questions": len(qids),
            "discovery": int(discovery.sum()),
            "confirmation": int((~discovery).sum()),
            "layers": layers.tolist(),
            "remapped_max_partition_error": float(np.max(remapped["max_partition_error"])),
            "nonremapped_max_partition_error": float(np.max(identity["max_partition_error"])),
            "remapped_natural_exact_logit_error": float(
                np.max(np.abs(remapped["natural_logits"] - remapped["trusted_natural_logits"]))
            ),
            "nonremapped_natural_exact_logit_error": float(
                np.max(np.abs(identity["natural_logits"] - identity["trusted_natural_logits"]))
            ),
            "remapped_natural_choice_agreement": float(
                np.mean(remapped["natural_logits"].argmax(-1) == remapped["trusted_natural_logits"].argmax(-1))
            ),
            "nonremapped_natural_choice_agreement": float(
                np.mean(identity["natural_logits"].argmax(-1) == identity["trusted_natural_logits"].argmax(-1))
            ),
        },
        "results": {},
        "remapped_semantic_vs_same_displayed_letter": {},
    }

    csv_rows: list[list[object]] = []
    selected_layers = (4, 12, 28, 36, 44, 48, 52, 60, 64)
    for split_name, keep in splits.items():
        split_result: dict[str, object] = {}
        for ci, condition in enumerate(CONDITIONS):
            condition_result: dict[str, object] = {}
            for metric in (*METRICS, "matching_share_of_1p"):
                rem = values["Remapped"][metric][ci][:, keep].mean(axis=-1)
                non = values["Non-remapped"][metric][ci][:, keep].mean(axis=-1)
                difference = non - rem
                layer_rows: dict[str, object] = {}
                for layer in selected_layers:
                    li = int(np.flatnonzero(layers == layer)[0])
                    layer_rows[str(layer)] = {
                        "remapped": _interval(rem[li], rng, args.bootstrap_draws),
                        "nonremapped": _interval(non[li], rng, args.bootstrap_draws),
                        "nonremapped_minus_remapped": _interval(
                            difference[li], rng, args.bootstrap_draws
                        ),
                    }
                per_question_rem = rem.mean(axis=0)
                per_question_non = non.mean(axis=0)
                condition_result[metric] = {
                    "mean_across_all_16_layers": {
                        "remapped": _interval(per_question_rem, rng, args.bootstrap_draws),
                        "nonremapped": _interval(per_question_non, rng, args.bootstrap_draws),
                        "nonremapped_minus_remapped": _interval(
                            per_question_non - per_question_rem,
                            rng,
                            args.bootstrap_draws,
                        ),
                    },
                    "selected_layers": layer_rows,
                }
                for li, layer in enumerate(layers):
                    rem_stats = _interval(rem[li], rng, args.bootstrap_draws)
                    non_stats = _interval(non[li], rng, args.bootstrap_draws)
                    diff_stats = _interval(difference[li], rng, args.bootstrap_draws)
                    csv_rows.append(
                        [
                            split_name,
                            condition,
                            metric,
                            int(layer),
                            int(keep.sum()),
                            rem_stats["mean"],
                            *rem_stats["ci95"],
                            non_stats["mean"],
                            *non_stats["ci95"],
                            diff_stats["mean"],
                            *diff_stats["ci95"],
                        ]
                    )
            split_result[condition] = condition_result
        summary["results"][split_name] = split_result

        disambiguation: dict[str, object] = {}
        for ci, condition in enumerate(CONDITIONS):
            semantic = values["Remapped"]["matching_1p_option"][ci][:, keep].mean(axis=-1)
            same_letter = same_letter_source[ci][:, keep].mean(axis=-1)
            paired = semantic - same_letter
            disambiguation[condition] = {
                "mean_across_all_16_layers": {
                    "semantic_match": _interval(semantic.mean(axis=0), rng, args.bootstrap_draws),
                    "same_displayed_letter": _interval(same_letter.mean(axis=0), rng, args.bootstrap_draws),
                    "semantic_minus_same_letter": _interval(paired.mean(axis=0), rng, args.bootstrap_draws),
                },
                "selected_layers": {
                    str(layer): {
                        "semantic_match": _interval(
                            semantic[int(np.flatnonzero(layers == layer)[0])], rng, args.bootstrap_draws
                        ),
                        "same_displayed_letter": _interval(
                            same_letter[int(np.flatnonzero(layers == layer)[0])], rng, args.bootstrap_draws
                        ),
                        "semantic_minus_same_letter": _interval(
                            paired[int(np.flatnonzero(layers == layer)[0])], rng, args.bootstrap_draws
                        ),
                    }
                    for layer in selected_layers
                },
            }
        summary["remapped_semantic_vs_same_displayed_letter"][split_name] = disambiguation

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "attention_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split", "condition", "metric", "layer", "n",
                "remapped_mean", "remapped_ci_low", "remapped_ci_high",
                "nonremapped_mean", "nonremapped_ci_low", "nonremapped_ci_high",
                "difference_mean", "difference_ci_low", "difference_ci_high",
            ]
        )
        writer.writerows(csv_rows)

    figure, axes = plt.subplots(2, 3, figsize=(14.5, 7.8), sharex=True)
    colors = {"Remapped": "#d95f02", "Non-remapped": "#1b9e77"}
    keep = splits["confirmation"]
    plot_rng = np.random.default_rng(args.seed + 1)
    for ci, condition in enumerate(CONDITIONS):
        for mi, metric in enumerate(METRICS):
            axis = axes[ci, mi]
            for presentation in PRESENTATIONS:
                data = values[presentation][metric][ci][:, keep].mean(axis=-1)
                mean, low, high = _bootstrap_curve(data, plot_rng, 3000)
                axis.plot(layers, mean * 100, color=colors[presentation], lw=2, label=presentation)
                axis.fill_between(layers, low * 100, high * 100, color=colors[presentation], alpha=0.17)
            axis.axhline(0, color="#777777", lw=0.7)
            axis.grid(alpha=0.2)
            axis.set_title(f"{condition}: {METRIC_LABELS[metric]}")
            axis.set_xlabel("Ordinary-attention layer")
            axis.set_ylabel("Attention mass (%)" if metric != "matching_selectivity" else "Difference (percentage points)")
            axis.set_xticks(layers[::2])
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Qwen3.6-27B: 2P option-line reads of 1P options\nHeld-out confirmation questions; paired prompt variants", fontsize=14)
    figure.tight_layout()
    figure.savefig(args.figure, dpi=180)
    plt.close(figure)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def pct(entry: dict[str, object]) -> str:
        mean = 100 * float(entry["mean"])
        low, high = [100 * float(value) for value in entry["ci95"]]
        return f"{mean:.1f}% [{low:.1f}%, {high:.1f}%]"

    lines = [
        "# Remapped versus non-remapped 2P→1P attention",
        "",
        "## Design",
        "",
        "The comparison uses the same 500 questions, prompts differing only in whether the second-presentation option order is permuted or left identical, both Game (`incorrect`) and Neutral (`lost`), and every ordinary-attention layer L4--64. Each query is a complete 2P option line. Sources exhaustively partition every prompt token. Candidates are aligned by semantic identity and first-pass rank, not displayed letter.",
        "",
        "## Held-out confirmation summary",
        "",
    ]
    confirmation_result = summary["results"]["confirmation"]
    for condition in CONDITIONS:
        lines.extend([f"### {condition}", ""])
        for metric in METRICS:
            aggregate = confirmation_result[condition][metric]["mean_across_all_16_layers"]
            lines.append(
                f"- {METRIC_LABELS[metric]} across all 16 layers: remapped {pct(aggregate['remapped'])}; non-remapped {pct(aggregate['nonremapped'])}; paired difference {pct(aggregate['nonremapped_minus_remapped'])}."
            )
        lines.append("")
    lines.extend(
        [
            "## Semantic identity versus displayed-letter position",
            "",
            "In the non-remapped prompt, a candidate's semantic match is also the line with the same displayed letter and list position. The remapped prompt separates those two possible targets. Within the remapped confirmation run:",
            "",
        ]
    )
    heldout_disambiguation = summary["remapped_semantic_vs_same_displayed_letter"]["confirmation"]
    for condition in CONDITIONS:
        aggregate = heldout_disambiguation[condition]["mean_across_all_16_layers"]
        lines.append(
            f"- {condition}: semantic match {pct(aggregate['semantic_match'])}; same displayed letter {pct(aggregate['same_displayed_letter'])}; semantic advantage {pct(aggregate['semantic_minus_same_letter'])}."
        )
    lines.append("")
    lines.extend(
        [
            "## Interpretation rule",
            "",
            "Total 1P-option attention answers whether 2P reads the old candidate set as much. Matching-line attention answers whether it reads the same semantic candidate as much. Matching selectivity subtracts the average wrong 1P line, separating semantic matching from a generic increase in attention to the entire first option list. In the non-remapped prompt, semantic identity and displayed letter/position coincide; therefore matching selectivity must be interpreted alongside the remapped condition, where those factors are separated.",
            "",
            "## Mechanistic interpretation",
            "",
            "The complete question and all four answer texts are already present again in 2P. Therefore, the 2P-to-1P read is not needed merely to recover problem text missing from the second presentation. The information distinctive to a 1P option-line state is the model's earlier, context-dependent processing of that candidate: its first-pass evidence and its relation to the other candidates. The attention comparison is observational by itself, so it cannot prove which feature is read. Combined with the separate balanced matching-line lesions—which causally change final candidate scores according to first-pass rank and remove the discrete Game-minus-Neutral switching difference—the best-supported interpretation is that both remapped and non-remapped prompts reuse prior candidate evaluation while 2P also constructs fresh candidate evidence. Identity order makes that retrieval sharper because semantic identity, displayed letter, and list position all point to the same old line; remapping separates those cues but leaves a clear semantic preference.",
            "",
            "## Validation",
            "",
            f"- Questions: 500; discovery: 251; confirmation: 249.",
            f"- Remapped/non-remapped maximum partition errors: {summary['validation']['remapped_max_partition_error']:.6f} / {summary['validation']['nonremapped_max_partition_error']:.6f}.",
            f"- Remapped/non-remapped natural choice agreement with trusted outputs: {summary['validation']['remapped_natural_choice_agreement']:.1%} / {summary['validation']['nonremapped_natural_choice_agreement']:.1%}.",
            "",
            f"Canonical figure: `{args.figure}`.",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["validation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
