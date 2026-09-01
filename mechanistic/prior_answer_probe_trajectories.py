from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .all_trial_figures import CONDITION_LABELS, _style
from .answer_emergence_figures import Z_975, macro_mean_and_se
from .data import decision_letter, load_activation_dataset


CONDITIONS = ("incorrect", "neutral")
COLORS = {"prior_answer": "#0072B2", "prior_runner": "#D55E00", "other": "#777777"}
LABELS = {
    "prior_answer": "Prior answer",
    "prior_runner": "Prior runner-up",
    "other": "Other letter",
}


def _macro_binary_summary(values: np.ndarray, strata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, se = macro_mean_and_se(values, strata)
    return mean, Z_975 * se


def _target_margin(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    row = np.arange(len(target))
    target_score = scores[row, :, target]
    competitors = scores.copy()
    competitors[row, :, target] = -np.inf
    return target_score - competitors.max(axis=-1)


def analyze(score_file: str | Path, input_dir: str | Path, output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    with np.load(score_file, allow_pickle=False) as cached:
        scores = cached["scores"].astype(np.float64)
        layers = cached["layers"].astype(int)
        score_conditions = cached["conditions"].astype(str).tolist()
        score_question_ids = cached["question_ids"].astype(str).tolist()

    data = load_activation_dataset(input_dir, ["baseline", *CONDITIONS])
    if data.question_ids != score_question_ids:
        raise ValueError("Probe-score and activation-dataset question order differs")
    if score_conditions != ["baseline", *CONDITIONS]:
        raise ValueError(f"Unexpected score conditions: {score_conditions}")

    prior_answer = np.asarray([
        "ABCD".index(decision_letter(data.metadata[(qid, "baseline")]))
        for qid in data.question_ids
    ])
    baseline_final = data.logits[:, 0, -1].copy()
    baseline_final[np.arange(len(prior_answer)), prior_answer] = -np.inf
    prior_runner = baseline_final.argmax(axis=-1)

    generated = {}
    for condition in CONDITIONS:
        generated[condition] = np.asarray([
            "ABCD".index(decision_letter(data.metadata[(qid, condition)]))
            for qid in data.question_ids
        ])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    summaries = {}

    for condition in CONDITIONS:
        ci = score_conditions.index(condition)
        condition_scores = scores[ci]
        leader = condition_scores.argmax(axis=-1)

        margin_summaries = {}
        leader_summaries = {}
        for key, target in (("prior_answer", prior_answer), ("prior_runner", prior_runner)):
            margins = _target_margin(condition_scores, target)
            mean, half = _macro_binary_summary(margins, target)
            margin_summaries[key] = mean, half
            is_leader = leader == target[:, None]
            leader_mean = is_leader.mean(axis=0)
            leader_half = Z_975 * np.sqrt(
                leader_mean * (1.0 - leader_mean) / len(is_leader)
            )
            leader_summaries[key] = leader_mean, leader_half
            for layer, value, width in zip(layers, mean, half):
                rows.append({
                    "condition": condition, "view": "lead_margin", "candidate": key,
                    "layer": int(layer), "mean": float(value),
                    "ci_low": float(value - width), "ci_high": float(value + width),
                })
            for layer, value, width in zip(layers, leader_mean, leader_half):
                rows.append({
                    "condition": condition, "view": "leader_fraction", "candidate": key,
                    "layer": int(layer), "mean": float(value),
                    "ci_low": float(value - width), "ci_high": float(value + width),
                })

        other = (leader != prior_answer[:, None]) & (leader != prior_runner[:, None])
        other_mean = other.mean(axis=0)
        other_half = Z_975 * np.sqrt(other_mean * (1.0 - other_mean) / len(other))
        leader_summaries["other"] = other_mean, other_half
        for layer, value, width in zip(layers, other_mean, other_half):
            rows.append({
                "condition": condition, "view": "leader_fraction", "candidate": "other",
                "layer": int(layer), "mean": float(value),
                "ci_low": float(value - width), "ci_high": float(value + width),
            })

        final_fractions = {
            "prior_answer": float(np.mean(generated[condition] == prior_answer)),
            "prior_runner": float(np.mean(generated[condition] == prior_runner)),
            "other": float(np.mean(
                (generated[condition] != prior_answer) & (generated[condition] != prior_runner)
            )),
        }
        summaries[condition] = margin_summaries, leader_summaries, final_fractions

    _style()
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharex=True)
    for row, condition in enumerate(CONDITIONS):
        margin_summaries, leader_summaries, final_fractions = summaries[condition]
        margin_axis, fraction_axis = axes[row]
        for axis in (margin_axis, fraction_axis):
            axis.axvspan(0, 46, color="#EEEEEE", alpha=0.85, zorder=0)
            axis.axvline(48, color="#777777", linewidth=0.8, linestyle=":")
            axis.set_xlim(0, 66 if axis is fraction_axis else int(layers[-1]))
            axis.set_xticks(np.arange(0, int(layers[-1]) + 1, 8))
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)

        for key in ("prior_answer", "prior_runner"):
            mean, half = margin_summaries[key]
            margin_axis.fill_between(layers, mean - half, mean + half,
                                     color=COLORS[key], alpha=0.18, linewidth=0)
            margin_axis.plot(layers, mean, color=COLORS[key], linewidth=1.65,
                             label=LABELS[key])
        margin_axis.axhline(0, color="#333333", linewidth=0.75)
        margin_axis.set_ylabel(
            f"{CONDITION_LABELS[condition]}\nCandidate score minus strongest competitor\n"
            "(baseline probe-SD units)"
        )

        for key in ("prior_answer", "prior_runner", "other"):
            mean, half = leader_summaries[key]
            fraction_axis.fill_between(layers, mean - half, mean + half,
                                       color=COLORS[key], alpha=0.14, linewidth=0)
            fraction_axis.plot(layers, mean, color=COLORS[key], linewidth=1.55,
                               label=LABELS[key])
            fraction_axis.scatter(
                [65], [final_fractions[key]], marker="D", s=28,
                facecolors="white", edgecolors=COLORS[key], linewidths=1.2, zorder=5,
            )
        fraction_axis.set_ylim(0, 1)
        fraction_axis.set_ylabel("Fraction of trials\nprobe predicts candidate")

    axes[0, 0].set_title("A  Candidate lead margin", loc="left", fontweight="bold")
    axes[0, 1].set_title("B  Probe-predicted candidate", loc="left", fontweight="bold")
    axes[0, 0].text(23, axes[0, 0].get_ylim()[1], "Probe not reliably decoding final answer",
                    ha="center", va="top", fontsize=7.2, color="#666666")
    axes[0, 1].text(0.99, 0.98, "Open diamonds: actual final-output fractions",
                    transform=axes[0, 1].transAxes, ha="right", va="top",
                    fontsize=7.2, color="#555555")
    for axis in axes[-1]:
        axis.set_xlabel("Residual readout (0 = embedding; 64 = final block)")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.53, 0.97))
    figure.suptitle(
        "SimpleMC: does the redo computation recover the prior answer or runner-up?",
        y=1.01, fontsize=10.5, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.0, h_pad=1.5)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output / f"prior_answer_runner_probe_trajectories.{suffix}",
                       dpi=300, bbox_inches="tight")
    plt.close(figure)

    with (output / "prior_answer_runner_probe_trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for condition in CONDITIONS:
        final_fractions = summaries[condition][2]
        summary_rows.append({"condition": condition, **final_fractions})
    with (output / "prior_answer_runner_final_fractions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track prior-answer and prior-runner candidate evidence in redo conditions"
    )
    parser.add_argument("--scores", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(args.scores, args.input, args.output)


if __name__ == "__main__":
    main()
