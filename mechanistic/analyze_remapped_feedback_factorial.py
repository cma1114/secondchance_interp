from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LETTERS = "ABCD"
CONDITIONS = (
    "incorrect_different",
    "incorrect_again",
    "lost_different",
    "lost_again",
)


def _entropy_bits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-300)), axis=-1)


def _interval(values, strata, rng, draws=10_000):
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    samples = np.zeros(draws, dtype=float)
    for group in groups:
        chosen = rng.choice(group, size=(draws, len(group)), replace=True)
        samples += values[chosen].sum(axis=1)
    samples /= len(values)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def _fmt(value, scale=1.0, digits=1, signed=True):
    sign = "+" if signed else ""
    return (
        f"{value['mean'] * scale:{sign}.{digits}f} "
        f"[{value['ci_low'] * scale:{sign}.{digits}f}, "
        f"{value['ci_high'] * scale:{sign}.{digits}f}]"
    )


def _load_payloads(canonical_root: Path, hybrid_root: Path):
    sources = {
        "incorrect_different": canonical_root / "incorrect_results.json",
        "lost_again": canonical_root / "neutral_results.json",
        "incorrect_again": hybrid_root / "incorrect_again_results.json",
        "lost_different": hybrid_root / "lost_different_results.json",
    }
    return {condition: json.loads(path.read_text()) for condition, path in sources.items()}


def analyze(
    canonical_root: Path,
    hybrid_root: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
    plan_path: Path,
    output_dir: Path,
    seed: int,
):
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped_baseline = json.loads(remapped_baseline_path.read_text())["results"]
    plan_payload = json.loads(plan_path.read_text())
    qids = [row["question_id"] for row in plan_payload["rows"]]
    payloads = _load_payloads(canonical_root, hybrid_root)
    for condition, payload in payloads.items():
        if set(payload["results"]) != set(qids):
            raise RuntimeError(
                f"{condition}: found {len(payload['results'])} rows, expected {len(qids)}"
            )

    w1 = np.asarray([baseline[qid]["answer"] for qid in qids])
    w2 = np.asarray([remapped_baseline[qid]["answer_original_content"] for qid in qids])
    correct = np.asarray([baseline[qid]["correct_answer"] for qid in qids])
    strata = w1.copy()
    conflict = w1 != w2
    rows = {
        condition: [payloads[condition]["results"][qid] for qid in qids]
        for condition in CONDITIONS
    }
    answer_content = {}
    answer_letter = {}
    logits = {}
    metrics = {}
    for condition in CONDITIONS:
        answer_letter[condition] = np.asarray([
            row["answer_new_letter"] for row in rows[condition]
        ])
        if not np.all(np.isin(answer_letter[condition], list(LETTERS))):
            raise RuntimeError(f"{condition} contains non-A-D unrestricted answers")
        answer_content[condition] = np.asarray([
            row["new_to_original"][letter]
            for row, letter in zip(rows[condition], answer_letter[condition])
        ])
        logits[condition] = np.asarray([
            row["aggregated_ad_logits"] for row in rows[condition]
        ], dtype=float)
        centered = logits[condition] - logits[condition].mean(axis=1, keepdims=True)
        w1_new_index = np.asarray([
            LETTERS.index(row["original_to_new"][old])
            for row, old in zip(rows[condition], w1)
        ])
        metrics[condition] = {
            "content_switch": (answer_content[condition] != w1).astype(float),
            "w1_selection": (answer_content[condition] == w1).astype(float),
            "w2_selection": (answer_content[condition] == w2).astype(float),
            "accuracy": (answer_content[condition] == correct).astype(float),
            "entropy_bits": _entropy_bits(logits[condition]),
            "w1_centered_logit": centered[np.arange(len(qids)), w1_new_index],
        }

    rng = np.random.default_rng(seed)
    contrasts = {
        "evaluation_effect_when_action_again": ("incorrect_again", "lost_again"),
        "evaluation_effect_when_action_different": ("incorrect_different", "lost_different"),
        "action_effect_when_evaluation_lost": ("lost_different", "lost_again"),
        "action_effect_when_evaluation_incorrect": ("incorrect_different", "incorrect_again"),
    }

    def summarize(mask):
        local_strata = strata[mask]
        result = {
            "n": int(mask.sum()),
            "condition_means": {},
            "contrasts": {},
            "factorial_main_effects": {},
        }
        for condition in CONDITIONS:
            result["condition_means"][condition] = {
                metric: _interval(values[mask], local_strata, rng)
                for metric, values in metrics[condition].items()
            }
        for label, (first, second) in contrasts.items():
            result["contrasts"][label] = {
                metric: _interval(
                    metrics[first][metric][mask] - metrics[second][metric][mask],
                    local_strata,
                    rng,
                )
                for metric in metrics[first]
            }
        for metric in metrics[CONDITIONS[0]]:
            evaluation = 0.5 * (
                metrics["incorrect_different"][metric]
                - metrics["lost_different"][metric]
                + metrics["incorrect_again"][metric]
                - metrics["lost_again"][metric]
            )
            action = 0.5 * (
                metrics["incorrect_different"][metric]
                - metrics["incorrect_again"][metric]
                + metrics["lost_different"][metric]
                - metrics["lost_again"][metric]
            )
            interaction = (
                metrics["incorrect_different"][metric]
                - metrics["incorrect_again"][metric]
                - metrics["lost_different"][metric]
                + metrics["lost_again"][metric]
            )
            result["factorial_main_effects"][metric] = {
                "incorrect_minus_lost": _interval(evaluation[mask], local_strata, rng),
                "different_minus_again": _interval(action[mask], local_strata, rng),
                "interaction": _interval(interaction[mask], local_strata, rng),
            }
        return result

    summary = {
        "design": {
            "n": len(qids),
            "conditions": {
                "incorrect_different": "Your answer was incorrect. Choose a different answer.",
                "incorrect_again": "Your answer was incorrect. Choose the answer again.",
                "lost_different": "Your answer was lost. Choose a different answer.",
                "lost_again": "Your answer was lost. Choose the answer again.",
            },
            "primary_question": (
                "Does incorrect versus lost induce semantic W1 avoidance when the action "
                "clause is held fixed at Choose the answer again?"
            ),
            "positive_contrast_sign": "first named condition minus second named condition",
        },
        "all_questions": summarize(np.ones(len(qids), dtype=bool)),
        "w1_not_equal_w2": summarize(conflict),
        "w1_equal_w2": summarize(~conflict),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "trial_table.csv").open("w", newline="") as stream:
        fields = ["question_id", "w1", "w2", "w1_not_equal_w2", "correct"]
        for condition in CONDITIONS:
            fields += [
                f"{condition}_answer_content",
                f"{condition}_content_switch",
                f"{condition}_entropy_bits",
                f"{condition}_w1_centered_logit",
            ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, qid in enumerate(qids):
            row = {
                "question_id": qid,
                "w1": w1[index],
                "w2": w2[index],
                "w1_not_equal_w2": bool(conflict[index]),
                "correct": correct[index],
            }
            for condition in CONDITIONS:
                row.update({
                    f"{condition}_answer_content": answer_content[condition][index],
                    f"{condition}_content_switch": bool(metrics[condition]["content_switch"][index]),
                    f"{condition}_entropy_bits": metrics[condition]["entropy_bits"][index],
                    f"{condition}_w1_centered_logit": metrics[condition]["w1_centered_logit"][index],
                })
            writer.writerow(row)

    all_results = summary["all_questions"]
    conflict_results = summary["w1_not_equal_w2"]
    evaluation_only = all_results["contrasts"]["evaluation_effect_when_action_again"]
    action_only = all_results["contrasts"]["action_effect_when_evaluation_lost"]
    full_gap = {
        metric: _interval(
            metrics["incorrect_different"][metric]
            - metrics["lost_again"][metric],
            strata,
            rng,
        )
        for metric in metrics["incorrect_different"]
    }
    lines = [
        "# Remapped feedback-clause 2x2 factorial",
        "",
        "## Design",
        "",
        "The second presentation uses the same frozen option remapping in every condition. "
        "Only the evaluation word (`incorrect`/`lost`) and action instruction "
        "(`different`/`again`) vary. Unrestricted A-D output is primary.",
        "",
        "| Condition | Exact feedback |",
        "|---|---|",
    ]
    for condition, prompt in summary["design"]["conditions"].items():
        lines.append(f"| `{condition}` | `{prompt}` |")
    lines += [
        "",
        "## Bottom line",
        "",
        "The incorrectness evaluation is behaviorally sufficient for most of the "
        "semantic W1-avoidance effect even when the action clause merely says to answer "
        "again. Relative to `lost_again`, `incorrect_again` increases W1 avoidance by "
        f"{_fmt(evaluation_only['content_switch'], 100, 1)} percentage points and lowers "
        f"the centered W1 logit by {_fmt(evaluation_only['w1_centered_logit'], 1, 3)}. "
        f"The full standard-Game contrast is {_fmt(full_gap['content_switch'], 100, 1)} "
        "percentage points, so the evaluation-only manipulation reproduces about "
        f"{100 * evaluation_only['content_switch']['mean'] / full_gap['content_switch']['mean']:.1f}% "
        "of that behavioral difference.",
        "",
        "However, evaluation-only also raises A-D entropy by "
        f"{_fmt(evaluation_only['entropy_bits'], 1, 3)} bits. The result therefore shows "
        "active semantic revision, but not entropy-free revision. The action-only "
        f"comparison raises W1 avoidance by only {_fmt(action_only['content_switch'], 100, 1)} "
        f"points while raising entropy by {_fmt(action_only['entropy_bits'], 1, 3)} bits. "
        "Thus the evaluation clause is the main driver of which semantic answer loses, "
        "whereas both clauses contribute to uncertainty.",
        "",
        f"On the {conflict_results['n']} W1 != W2 questions, evaluation-only is almost "
        "behaviorally indistinguishable from standard Game on the key semantic endpoint: "
        f"W1 is selected on {100 * conflict_results['condition_means']['incorrect_again']['w1_selection']['mean']:.1f}% "
        "of evaluation-only trials versus "
        f"{100 * conflict_results['condition_means']['incorrect_different']['w1_selection']['mean']:.1f}% "
        "of standard-Game trials, compared with "
        f"{100 * conflict_results['condition_means']['lost_again']['w1_selection']['mean']:.1f}% "
        "under Neutral.",
        "",
        "## All 500 questions",
        "",
        "| Condition | W1 avoidance / content switch | W1 selection | Entropy (bits) | W1 centered logit |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        cell = all_results["condition_means"][condition]
        lines.append(
            f"| `{condition}` | {_fmt(cell['content_switch'], 100, 1, False)}% | "
            f"{_fmt(cell['w1_selection'], 100, 1, False)}% | "
            f"{_fmt(cell['entropy_bits'], 1, 3, False)} | "
            f"{_fmt(cell['w1_centered_logit'], 1, 3, True)} |"
        )
    lines += [
        "",
        "## Decisive held-action comparisons",
        "",
        "Effects are the first condition minus the second on the same questions.",
        "",
        "| Contrast | W1 avoidance | Entropy | W1 centered logit |",
        "|---|---:|---:|---:|",
        f"| `incorrect_again - lost_again` | {_fmt(evaluation_only['content_switch'], 100, 1)} pp | "
        f"{_fmt(evaluation_only['entropy_bits'], 1, 3)} | {_fmt(evaluation_only['w1_centered_logit'], 1, 3)} |",
        f"| `lost_different - lost_again` | {_fmt(action_only['content_switch'], 100, 1)} pp | "
        f"{_fmt(action_only['entropy_bits'], 1, 3)} | {_fmt(action_only['w1_centered_logit'], 1, 3)} |",
        "",
        f"## W1 != W2 conflict questions (n={conflict_results['n']})",
        "",
        "| Contrast | W1 avoidance | W1 selection | W2 selection | Entropy |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in (
        "evaluation_effect_when_action_again",
        "action_effect_when_evaluation_lost",
        "evaluation_effect_when_action_different",
        "action_effect_when_evaluation_incorrect",
    ):
        cell = conflict_results["contrasts"][label]
        lines.append(
            f"| `{label}` | {_fmt(cell['content_switch'], 100, 1)} pp | "
            f"{_fmt(cell['w1_selection'], 100, 1)} pp | "
            f"{_fmt(cell['w2_selection'], 100, 1)} pp | "
            f"{_fmt(cell['entropy_bits'], 1, 3)} |"
        )
    lines += [
        "",
        "## Interpretation rule",
        "",
        "`incorrect_again - lost_again` directly tests whether the incorrectness evaluation "
        "is behaviorally sufficient without the different-answer instruction. It is large "
        "for W1 avoidance, but it also increases entropy; the experiment therefore supports "
        "independent semantic revision without establishing a purely entropy-neutral route.",
        "",
        "Machine-readable results: `summary.json`; question-level data: `trial_table.csv`.",
        "",
        "## Preferred follow-up",
        "",
        "Use `incorrect_again` versus `lost_again` for the next mechanistic experiment. "
        "At the period closing the evaluation clause, bidirectionally transplant the GLA "
        "recurrent state updates—not merely the token residual—while keeping the shared "
        "`Choose the answer again.` clause and repeated question fixed. First establish an "
        "all-GLA positive control, then localize with eight-block bands and targeted "
        "individual/leave-one-out tests. The decisive endpoints are transfer and rescue of "
        "semantic W1 avoidance, with entropy reported as a separate outcome.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze the remapped 2x2 feedback factorial")
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--hybrid-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    analyze(
        args.canonical_root,
        args.hybrid_root,
        args.baseline,
        args.remapped_baseline,
        args.plan,
        args.output_dir,
        args.seed,
    )


if __name__ == "__main__":
    main()
