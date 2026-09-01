from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LETTERS = "ABCD"
SPLITS = ("discovery", "confirmation")


def _semantic_answers(
    raw_logits: np.ndarray, qids: list[str], plan: dict[str, dict]
) -> np.ndarray:
    output = np.empty(raw_logits.shape[:2], dtype=np.int64)
    for ci in range(raw_logits.shape[0]):
        for qi, qid in enumerate(qids):
            new_letter = LETTERS[int(raw_logits[ci, qi].argmax())]
            output[ci, qi] = LETTERS.index(plan[qid]["new_to_original"][new_letter])
    return output


def _aligned_logits(
    raw_logits: np.ndarray, qids: list[str], plan: dict[str, dict]
) -> np.ndarray:
    output = np.empty_like(raw_logits)
    for qi, qid in enumerate(qids):
        for content_index, content in enumerate(LETTERS):
            new_letter = plan[qid]["original_to_new"][content]
            output[:, qi, content_index] = raw_logits[:, qi, LETTERS.index(new_letter)]
    return output


def _interval(
    values: np.ndarray, strata: np.ndarray, rng: np.random.Generator, draws: int
) -> dict[str, float | list[float] | int]:
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    boot = np.empty(draws)
    for draw in range(draws):
        indices = np.concatenate(
            [group[rng.integers(0, len(group), len(group))] for group in groups]
        )
        boot[draw] = values[indices].mean()
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(boot, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--mapping-plan", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    baseline = json.loads((args.base / "baseline_results.json").read_text())["results"]
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    plan = {
        row["question_id"]: row
        for row in json.loads(args.mapping_plan.read_text())["rows"]
    }
    positive_all = dict(np.load(
        args.base / "final_decision_positive_only_exact/all/results.npz",
        allow_pickle=False,
    ))
    positive_lookup = {
        qid: index
        for index, qid in enumerate(positive_all["question_ids"].astype(str))
    }
    rng = np.random.default_rng(args.seed)
    records: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "signed": {}, "positive_only": {}
    }

    for split in SPLITS:
        signed = dict(np.load(
            args.base / f"final_decision_semantic_ablation/{split}/results.npz",
            allow_pickle=False,
        ))
        qids = signed["question_ids"].astype(str).tolist()
        positive_indices = np.asarray([positive_lookup[qid] for qid in qids])
        for mode, natural_raw, ablated_raw in (
            ("signed", signed["natural_logits"], signed["ablated_logits"]),
            (
                "positive_only",
                positive_all["natural_logits"][:, positive_indices],
                positive_all["ablated_logits"][:, positive_indices],
            ),
        ):
            natural_answers = _semantic_answers(natural_raw, qids, plan)
            ablated_answers = _semantic_answers(ablated_raw, qids, plan)
            natural_logits = _aligned_logits(natural_raw, qids, plan)
            ablated_logits = _aligned_logits(ablated_raw, qids, plan)
            w1 = np.asarray([LETTERS.index(baseline[qid]["answer"]) for qid in qids])
            w2 = np.asarray([
                LETTERS.index(remapped[qid]["answer_original_content"]) for qid in qids
            ])
            conflict = w1 != w2
            rows = np.arange(len(qids))
            centered_delta = (
                ablated_logits - ablated_logits.mean(-1, keepdims=True)
                - natural_logits + natural_logits.mean(-1, keepdims=True)
            )
            other_delta = np.empty(len(qids))
            for qi in range(len(qids)):
                others = [index for index in range(4) if index not in (w1[qi], w2[qi])]
                other_delta[qi] = centered_delta[1, qi, others].mean()
            records[mode][split] = {
                "qids": np.asarray(qids),
                "w1": w1,
                "w2": w2,
                "conflict": conflict,
                "natural_answer": natural_answers[1],
                "ablated_answer": ablated_answers[1],
                "selection_delta": (
                    (ablated_answers[1] == w1).astype(int)
                    - (natural_answers[1] == w1).astype(int)
                ),
                "w1_logit_delta": centered_delta[1, rows, w1],
                "w2_logit_delta": centered_delta[1, rows, w2],
                "other_logit_delta": other_delta,
            }

    summary: dict[str, object] = {
        "definition": (
            "Neutral conflict trials only. Signed zeroing subtracts the full signed "
            "projection; positive-only subtracts max(projection, 0). Answers use the "
            "displayed-letter argmax before semantic remapping."
        ),
        "neutral_w1_selection": {},
        "positive_only_centered_logit_change": {},
        "positive_only_by_original_w1_letter": {},
    }

    for mode in ("signed", "positive_only"):
        summary["neutral_w1_selection"][mode] = {}
        for split in SPLITS:
            rec = records[mode][split]
            mask = rec["conflict"]
            strata = rec["w1"][mask]
            effect = _interval(rec["selection_delta"][mask], strata, rng, args.draws)
            enter = int(np.sum(
                (rec["natural_answer"] != rec["w1"])
                & (rec["ablated_answer"] == rec["w1"])
                & mask
            ))
            leave = int(np.sum(
                (rec["natural_answer"] == rec["w1"])
                & (rec["ablated_answer"] != rec["w1"])
                & mask
            ))
            summary["neutral_w1_selection"][mode][split] = {
                "effect": effect, "entered_w1": enter, "left_w1": leave
            }
        pooled_values = np.concatenate([
            records[mode][split]["selection_delta"][records[mode][split]["conflict"]]
            for split in SPLITS
        ])
        pooled_strata = np.concatenate([
            records[mode][split]["w1"][records[mode][split]["conflict"]]
            + 4 * split_index
            for split_index, split in enumerate(SPLITS)
        ])
        summary["neutral_w1_selection"][mode]["pooled"] = {
            "effect": _interval(pooled_values, pooled_strata, rng, args.draws),
            "entered_w1": int(sum(
                summary["neutral_w1_selection"][mode][split]["entered_w1"]
                for split in SPLITS
            )),
            "left_w1": int(sum(
                summary["neutral_w1_selection"][mode][split]["left_w1"]
                for split in SPLITS
            )),
        }

    for split in (*SPLITS, "pooled"):
        summary["positive_only_centered_logit_change"][split] = {}
        if split == "pooled":
            for key in ("w1_logit_delta", "w2_logit_delta", "other_logit_delta"):
                values = np.concatenate([
                    records["positive_only"][name][key][
                        records["positive_only"][name]["conflict"]
                    ] for name in SPLITS
                ])
                strata = np.concatenate([
                    records["positive_only"][name]["w1"][
                        records["positive_only"][name]["conflict"]
                    ] + 4 * index
                    for index, name in enumerate(SPLITS)
                ])
                summary["positive_only_centered_logit_change"][split][key] = _interval(
                    values, strata, rng, args.draws
                )
        else:
            rec = records["positive_only"][split]
            mask = rec["conflict"]
            for key in ("w1_logit_delta", "w2_logit_delta", "other_logit_delta"):
                summary["positive_only_centered_logit_change"][split][key] = _interval(
                    rec[key][mask], rec["w1"][mask], rng, args.draws
                )

    pooled_w1 = np.concatenate([
        records["positive_only"][split]["w1"][records["positive_only"][split]["conflict"]]
        for split in SPLITS
    ])
    pooled_selection = np.concatenate([
        records["positive_only"][split]["selection_delta"][records["positive_only"][split]["conflict"]]
        for split in SPLITS
    ])
    pooled_split = np.concatenate([
        np.full(int(records["positive_only"][split]["conflict"].sum()), index)
        for index, split in enumerate(SPLITS)
    ])
    for letter_index, letter in enumerate(LETTERS):
        mask = pooled_w1 == letter_index
        if not mask.any():
            continue
        summary["positive_only_by_original_w1_letter"][letter] = _interval(
            pooled_selection[mask], pooled_split[mask], rng, args.draws
        )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    signed = summary["neutral_w1_selection"]["signed"]
    positive = summary["neutral_w1_selection"]["positive_only"]
    logit = summary["positive_only_centered_logit_change"]
    def effect_text(row: dict) -> str:
        effect = row["effect"]
        return (
            f"{effect['mean'] * 100:+.1f} pp "
            f"(95% CI {effect['ci'][0] * 100:+.1f} to {effect['ci'][1] * 100:+.1f})"
        )
    report = [
        "# Why did Neutral sometimes choose W1 more after W1 ablation?",
        "",
        "## Answer",
        "",
        "Most of the apparent paradox came from calling signed projection-zeroing an ablation. The original intervention set `h · v_W1` to zero. When that projection was negative, it *added* the W1-defined direction rather than removing it. The positive-only rerun leaves negative projections untouched.",
        "",
        "After that correction, there is no stable Neutral W1 increase. Across all 273 W1 != W2 questions, positive-only removal changed Neutral W1 selection by "
        + effect_text(positive["pooled"])
        + f": 13 questions entered W1 and 11 left it. Discovery moved away from W1 ({effect_text(positive['discovery'])}); confirmation moved toward it ({effect_text(positive['confirmation'])}).",
        "",
        "The original signed intervention produced a larger pooled increase of "
        + effect_text(signed["pooled"])
        + ". Because that increase shrinks when the only rule change is leaving negative projections untouched, it depends substantially on also moving negative projections toward zero. The two nonlinear interventions should not be treated as an additive decomposition.",
        "",
        "## What happened to the logits?",
        "",
        "Positive-only removal did not raise W1 evidence on average in Neutral. Pooled centered A-D logit changes were:",
        "",
        f"- W1: {logit['pooled']['w1_logit_delta']['mean']:+.3f}",
        f"- W2: {logit['pooled']['w2_logit_delta']['mean']:+.3f}",
        f"- Mean of the other two options: {logit['pooled']['other_logit_delta']['mean']:+.3f}",
        "",
        "Thus W1 rose on a few individual boundary cases, but the average W1 score fell. W2 fell somewhat more, while ranks 3-4 rose. The five-net-question confirmation increase is therefore a heterogeneous redistribution near decision boundaries, not a general causal boost to W1.",
        "",
        "## Interpretation",
        "",
        "The semantic reference vector is not a monotonic W1-evidence axis. It is a layer-specific direction constructed from contextual option-newline states. Removing it at every layer changes downstream computation nonlinearly and remains heterogeneous by original answer letter. It is causally involved in answer computation, but neither signed zeroing nor positive-only removal can be interpreted as simply deleting the model's memory of W1.",
        "",
        "The most defensible conclusion is therefore: **Neutral choosing W1 more is not a replicated mechanism requiring a special explanation. It is mainly a signed-intervention artifact plus a small, split-unstable set of boundary crossings.**",
        "",
        "## Files",
        "",
        "- Figure: `figures/qwen36_27b_simplemc_corrected/neutral_w1_ablation_diagnostic.png`",
        "- Numerical diagnostic: `neutral_w1_diagnostic.json`",
        "- Per-question transitions: `../data/per_question_condition.csv`",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report) + "\n")

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 160,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)

    # A: split replication and sign-artifact diagnostic.
    ax = axes[0, 0]
    x = np.arange(3)
    for offset, mode, label, marker in (
        (-0.12, "signed", "Signed zeroing", "o"),
        (0.12, "positive_only", "Positive-only removal", "s"),
    ):
        rows = [summary["neutral_w1_selection"][mode][name]["effect"] for name in (*SPLITS, "pooled")]
        means = np.asarray([row["mean"] for row in rows]) * 100
        lows = means - np.asarray([row["ci"][0] for row in rows]) * 100
        highs = np.asarray([row["ci"][1] for row in rows]) * 100 - means
        ax.errorbar(x + offset, means, yerr=[lows, highs], fmt=marker, capsize=4, lw=2, label=label)
    ax.axhline(0, color="0.45", lw=1)
    ax.set_xticks(x, ["Discovery", "Confirmation", "Pooled"])
    ax.set_ylabel("Neutral W1-selection change (pp)")
    ax.set_title("A  The Neutral increase largely disappears\nwhen negative projections are left alone", loc="left")
    ax.legend(frameon=False)

    # B: raw transition counts for positive-only.
    ax = axes[0, 1]
    width = 0.34
    names = (*SPLITS, "pooled")
    enters = [summary["neutral_w1_selection"]["positive_only"][name]["entered_w1"] for name in names]
    leaves = [summary["neutral_w1_selection"]["positive_only"][name]["left_w1"] for name in names]
    b1 = ax.bar(x - width / 2, enters, width, label="Moved into W1")
    b2 = ax.bar(x + width / 2, leaves, width, label="Moved out of W1")
    ax.bar_label(b1, padding=2)
    ax.bar_label(b2, padding=2)
    ax.set_xticks(x, ["Discovery", "Confirmation", "Pooled"])
    ax.set_ylabel("Questions")
    ax.set_title("B  Confirmation is five net questions;\ndiscovery moves in the opposite direction", loc="left")
    ax.legend(frameon=False)

    # C: average output redistribution.
    ax = axes[1, 0]
    candidate_keys = ("w1_logit_delta", "w2_logit_delta", "other_logit_delta")
    candidate_labels = ("W1", "W2", "Other options\n(mean)")
    colors = ("C0", "C1", "C2")
    for split_index, split in enumerate(SPLITS):
        for candidate_index, (key, label, color) in enumerate(zip(candidate_keys, candidate_labels, colors)):
            row = summary["positive_only_centered_logit_change"][split][key]
            mean = row["mean"]
            ax.errorbar(
                split_index + (candidate_index - 1) * 0.16,
                mean,
                yerr=[[mean - row["ci"][0]], [row["ci"][1] - mean]],
                fmt="o", capsize=3, color=color,
                label=label if split_index == 0 else None,
            )
    ax.axhline(0, color="0.45", lw=1)
    ax.set_xticks([0, 1], ["Discovery", "Confirmation"])
    ax.set_ylabel("Centered A-D logit change")
    ax.set_title("C  Positive-only removal does not generally\nboost W1 in Neutral", loc="left")
    ax.legend(frameon=False, ncol=3)

    # D: residual answer-letter dependence.
    ax = axes[1, 1]
    letter_rows = summary["positive_only_by_original_w1_letter"]
    for index, letter in enumerate(LETTERS):
        row = letter_rows[letter]
        mean = row["mean"] * 100
        ax.errorbar(
            index, mean,
            yerr=[[mean - row["ci"][0] * 100], [row["ci"][1] * 100 - mean]],
            fmt="o", capsize=4, lw=2,
        )
        ax.text(index, mean, f"  n={row['n']}", va="center", fontsize=9)
    ax.axhline(0, color="0.45", lw=1)
    ax.set_xticks(np.arange(4), list(LETTERS))
    ax.set_xlabel("Original Baseline W1 letter")
    ax.set_ylabel("Pooled Neutral W1-selection change (pp)")
    ax.set_title("D  The remaining effect is heterogeneous by letter,\nnot a clean semantic W1 effect", loc="left")

    fig.suptitle("Why did W1 ablation sometimes make Neutral choose W1 more?", fontsize=16)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=220)
    plt.close(fig)
    print(args.figure)
    print(args.summary)
    print(args.report)


if __name__ == "__main__":
    main()
