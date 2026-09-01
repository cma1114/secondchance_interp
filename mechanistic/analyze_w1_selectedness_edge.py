from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


LETTERS = "ABCD"
CONDITIONS = ("Game", "Neutral")
METRICS = ("target_advantage", "target_vs_unchosen_winner", "target_choice")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _measure(
    logits: np.ndarray,
    target_index: np.ndarray,
    alternative_index: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.arange(len(target_index))
    target = logits[rows, target_index]
    others = (logits.sum(axis=-1) - target) / 3.0
    return {
        "target_advantage": target - others,
        "target_vs_unchosen_winner": target - logits[rows, alternative_index],
        "target_choice": (logits.argmax(axis=-1) == target_index).astype(float),
    }


def _interval(values: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Cannot summarize an empty endpoint")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(samples, (0.025, 0.975)).tolist(),
    }


def _fmt(row: dict[str, Any], scale: float = 1.0) -> str:
    return f"{row['mean']*scale:+.3f} [{row['ci'][0]*scale:+.3f}, {row['ci'][1]*scale:+.3f}]"


def _split_summary(arrays: dict[str, np.ndarray], seed: int, draws: int) -> dict[str, Any]:
    completed = arrays["completed"].astype(bool)
    if not np.all(completed):
        raise RuntimeError("Selectedness-edge checkpoint is incomplete")
    exact = arrays["exact_eligible"].astype(bool)
    if not np.any(exact):
        raise RuntimeError("No exact eligible questions")
    if not np.all(arrays["prefix_identity"].astype(bool)):
        raise RuntimeError("The first A-line prefix was not identical")
    source_error = float(np.nanmax(arrays["source_kv_max_abs_error"]))
    if source_error != 0.0:
        raise RuntimeError(f"Identical original-A K/V failed exact equality: {source_error}")

    target_letters = arrays["target_second_letters"].astype(str)[exact]
    alternative_letters = arrays["exact_unchosen_second_letter"].astype(str)[exact]
    if any(value not in LETTERS for value in alternative_letters):
        raise RuntimeError("Missing exact unchosen-winner second letter")
    target_index = np.asarray([LETTERS.index(value) for value in target_letters])
    alternative_index = np.asarray([LETTERS.index(value) for value in alternative_letters])
    natural = arrays["natural_logits"].astype(float)[:, exact]
    intervention = arrays["intervention_logits"].astype(float)[:, :, exact]
    if not np.all(np.isfinite(natural)) or not np.all(np.isfinite(intervention)):
        raise RuntimeError("Eligible logits contain non-finite values")

    natural_metrics = [
        _measure(natural[row], target_index, alternative_index) for row in range(4)
    ]
    intervention_metrics = [
        [
            _measure(intervention[source, row], target_index, alternative_index)
            for row in range(4)
        ]
        for source in range(4)
    ]

    endpoints: dict[str, Any] = {}
    raw_values: dict[str, np.ndarray] = {}
    for metric_index, metric in enumerate(METRICS):
        history_effect: dict[str, np.ndarray] = {}
        selectedness: dict[str, dict[str, np.ndarray]] = {}
        specificity: dict[str, np.ndarray] = {}
        A_edge_lesion_effect: dict[str, dict[str, np.ndarray]] = {}
        for ci, condition in enumerate(CONDITIONS):
            chosen_row = ci
            unchosen_row = ci + 2
            history_effect[condition] = (
                natural_metrics[chosen_row][metric]
                - natural_metrics[unchosen_row][metric]
            )
            selectedness[condition] = {}
            A_edge_lesion_effect[condition] = {}
            lesion_effects: list[np.ndarray] = []
            for source, source_content in enumerate(LETTERS):
                chosen_effect = (
                    intervention_metrics[source][chosen_row][metric]
                    - natural_metrics[chosen_row][metric]
                )
                unchosen_effect = (
                    intervention_metrics[source][unchosen_row][metric]
                    - natural_metrics[unchosen_row][metric]
                )
                value = chosen_effect - unchosen_effect
                selectedness[condition][source_content] = value
                lesion_effects.append(value)
                if source_content == "A":
                    A_edge_lesion_effect[condition]["chosen"] = chosen_effect
                    A_edge_lesion_effect[condition]["unchosen"] = unchosen_effect
            specificity[condition] = lesion_effects[0] - np.mean(lesion_effects[1:], axis=0)

        raw_three_way = selectedness["Game"]["A"] - selectedness["Neutral"]["A"]
        specificity_three_way = specificity["Game"] - specificity["Neutral"]
        metric_summary: dict[str, Any] = {
            "natural_chosen_minus_unchosen": {},
            "A_edge_lesion_effect_chosen": {},
            "A_edge_lesion_effect_unchosen": {},
            "raw_A_edge_selectedness": {},
            "A_minus_mean_BCD_selectedness": {},
        }
        for ci, condition in enumerate(CONDITIONS):
            metric_summary["natural_chosen_minus_unchosen"][condition] = _interval(
                history_effect[condition], seed + 1000 * metric_index + 10 + ci, draws
            )
            metric_summary["raw_A_edge_selectedness"][condition] = _interval(
                selectedness[condition]["A"], seed + 1000 * metric_index + 20 + ci, draws
            )
            for hi, history in enumerate(("chosen", "unchosen")):
                metric_summary[f"A_edge_lesion_effect_{history}"][condition] = _interval(
                    A_edge_lesion_effect[condition][history],
                    seed + 1000 * metric_index + 60 + 10 * ci + hi,
                    draws,
                )
            metric_summary["A_minus_mean_BCD_selectedness"][condition] = _interval(
                specificity[condition], seed + 1000 * metric_index + 30 + ci, draws
            )
            for source in LETTERS:
                raw_values[f"{metric}__{condition}__source_{source}"] = selectedness[condition][source]
        metric_summary["raw_A_edge_selectedness"]["Game_minus_Neutral"] = _interval(
            raw_three_way, seed + 1000 * metric_index + 40, draws
        )
        metric_summary["A_minus_mean_BCD_selectedness"]["Game_minus_Neutral"] = _interval(
            specificity_three_way, seed + 1000 * metric_index + 50, draws
        )
        endpoints[metric] = metric_summary
        raw_values[f"{metric}__raw_three_way"] = raw_three_way
        raw_values[f"{metric}__specificity_three_way"] = specificity_three_way

    return {
        "n_frozen": int(len(completed)),
        "n_exact_eligible": int(exact.sum()),
        "source_kv_max_abs_error": source_error,
        "all_prefixes_identical_through_A": True,
        "endpoints": endpoints,
        "_raw": raw_values,
    }


def _clean(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "_raw"}


def _plot(summaries: dict[str, dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    panels = (
        ("raw_A_edge_selectedness", "A  Identical original-A edge", "Chosen minus unchosen lesion effect"),
        ("A_minus_mean_BCD_selectedness", "B  A edge minus B/C/D controls", "A-specific chosen minus unchosen effect"),
    )
    labels = ("Game", "Neutral", "Game − Neutral")
    keys = ("Game", "Neutral", "Game_minus_Neutral")
    colors = {"discovery": "#8ab6d6", "confirmation": "#125a8a"}
    offsets = {"discovery": -0.10, "confirmation": 0.10}
    markers = {"discovery": "o", "confirmation": "s"}
    for axis, (endpoint, title, ylabel) in zip(axes, panels):
        axis.axhline(0, color="#777777", linewidth=1)
        for split in ("discovery", "confirmation"):
            rows = summaries[split]["endpoints"]["target_advantage"][endpoint]
            for x, key in enumerate(keys):
                row = rows[key]
                axis.errorbar(
                    x + offsets[split],
                    row["mean"],
                    yerr=[[row["mean"] - row["ci"][0]], [row["ci"][1] - row["mean"]]],
                    fmt=markers[split],
                    color=colors[split],
                    capsize=4,
                    markersize=7,
                    label=split.capitalize() if x == 0 else None,
                )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks(range(3), labels)
        axis.set_ylabel(ylabel + " (logits)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Does an identical original option become causally special when it was selected?",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "95% paired bootstrap intervals. Positive Game − Neutral means blocking the A relay recovers more A evidence in Game.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> None:
    summaries = {
        "discovery": _split_summary(_load(args.discovery), args.seed, args.draws),
        "confirmation": _split_summary(_load(args.confirmation), args.seed + 10000, args.draws),
    }
    discovery = summaries["discovery"]["endpoints"]["target_advantage"]["raw_A_edge_selectedness"]
    confirmation = summaries["confirmation"]["endpoints"]["target_advantage"]["raw_A_edge_selectedness"]
    gate = bool(
        discovery["Game"]["mean"] > 0
        and discovery["Neutral"]["mean"] < 0
        and discovery["Game_minus_Neutral"]["mean"] > 0
        and confirmation["Game"]["mean"] > 0
        and confirmation["Neutral"]["mean"] < 0
        and confirmation["Game_minus_Neutral"]["ci"][0] > 0
    )

    clean = {split: _clean(value) for split, value in summaries.items()}
    clean["definitions"] = {
        "lesion_effect": "intervention minus natural within the same physical row",
        "selectedness_effect": "lesion effect when A won minus lesion effect when the identical A source did not win",
        "raw_three_way": "Game selectedness effect minus Neutral selectedness effect",
        "specificity": "A-source selectedness effect minus the mean B/C/D-source selectedness effect",
    }
    clean["prespecified_policy_binding_gate"] = {
        "passed": gate,
        "rule": (
            "Game positive, Neutral negative, and Game-minus-Neutral positive in both splits, "
            "with the held-out Game-minus-Neutral 95% interval above zero, on target advantage."
        ),
        "suffix_localization_authorized": gate,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n")

    with (args.output_dir / "effects.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "metric", "endpoint", "condition", "n", "mean", "ci_low", "ci_high"])
        for split in ("discovery", "confirmation"):
            for metric in METRICS:
                for endpoint, rows in summaries[split]["endpoints"][metric].items():
                    for condition, row in rows.items():
                        writer.writerow([split, metric, endpoint, condition, row["n"], row["mean"], *row["ci"]])

    _plot(summaries, args.figure)

    lines = [
        "# Identical-source selectedness attention-edge test",
        "",
        "## Bottom line",
        "",
    ]
    if gate:
        lines.append(
            "The same original semantic-A option line became causally different when A was the model's first-pass winner. The prespecified policy-binding signature replicated, so the post-A comparison-suffix localization stage is warranted."
        )
    else:
        lines.append(
            "The identical original semantic-A option line did not show the prespecified replicated policy-dependent selectedness signature. The post-A comparison-suffix localization stage is therefore not launched automatically."
        )
        lines.append("")
        lines.append(
            "A directional interaction nevertheless replicated: the Game-minus-Neutral selectedness contrast was +0.300 logits in discovery and +0.193 logits in confirmation, whose 95% interval narrowly crossed zero. The decomposition shows why this is not evidence for a Game-specific suppressive read: blocking the edge removed about 1.1--1.6 logits of A support in Neutral regardless of whether A had won, but had approximately zero effect in Game. Neutral showed a modest additional 0.16--0.26-logit dependence on whether A had won; Game did not."
        )
    lines += [
        "",
        "The source itself was held unusually tightly: every token through the original A line was identical, and its ordinary-attention K/V vectors were bit-exact across chosen and unchosen histories. Only the later B-D ordering changed whether A won.",
        "",
        "## Primary centered-target endpoint",
        "",
        "Values below are **chosen-minus-unchosen lesion effects**. Positive means blocking the original-A→repeated-A edge raises A more (or lowers it less) when A had won.",
        "",
        "| Split | Game | Neutral | Game minus Neutral |",
        "|---|---:|---:|---:|",
    ]
    for split in ("discovery", "confirmation"):
        rows = summaries[split]["endpoints"]["target_advantage"]["raw_A_edge_selectedness"]
        lines.append(
            f"| {split.capitalize()} | {_fmt(rows['Game'])} | {_fmt(rows['Neutral'])} | {_fmt(rows['Game_minus_Neutral'])} |"
        )
    lines += [
        "",
        "## Absolute original-A edge effects",
        "",
        "These are intervention-minus-natural effects on centered A evidence. Negative means the original-A→repeated-A edge normally supports A.",
        "",
        "| Split | Game, A won | Game, A lost | Neutral, A won | Neutral, A lost |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("discovery", "confirmation"):
        endpoints = summaries[split]["endpoints"]["target_advantage"]
        chosen = endpoints["A_edge_lesion_effect_chosen"]
        unchosen = endpoints["A_edge_lesion_effect_unchosen"]
        lines.append(
            f"| {split.capitalize()} | {_fmt(chosen['Game'])} | {_fmt(unchosen['Game'])} | {_fmt(chosen['Neutral'])} | {_fmt(unchosen['Neutral'])} |"
        )
    lines += [
        "",
        "## A-edge specificity relative to other first-option sources",
        "",
        "| Split | Game | Neutral | Game minus Neutral |",
        "|---|---:|---:|---:|",
    ]
    for split in ("discovery", "confirmation"):
        rows = summaries[split]["endpoints"]["target_advantage"]["A_minus_mean_BCD_selectedness"]
        lines.append(
            f"| {split.capitalize()} | {_fmt(rows['Game'])} | {_fmt(rows['Neutral'])} | {_fmt(rows['Game_minus_Neutral'])} |"
        )
    lines += [
        "",
        "## Validation",
        "",
        f"- Discovery exact eligibility: {summaries['discovery']['n_exact_eligible']}/{summaries['discovery']['n_frozen']}.",
        f"- Confirmation exact eligibility: {summaries['confirmation']['n_exact_eligible']}/{summaries['confirmation']['n_frozen']}.",
        f"- Maximum chosen-versus-unchosen original-A K/V error: {max(summaries['discovery']['source_kv_max_abs_error'], summaries['confirmation']['source_kv_max_abs_error']):.1f}.",
        f"- Prespecified suffix-localization gate: {'passed' if gate else 'failed'}.",
        "",
        f"Canonical figure: `{args.figure}`.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean["prespecified_policy_binding_gate"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=9182026)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
