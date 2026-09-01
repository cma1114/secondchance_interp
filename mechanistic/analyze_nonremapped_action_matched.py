from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


LETTERS = "ABCD"


def _load(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text())["results"]


def _ci(values: np.ndarray, rng: np.random.Generator, draws: int = 10000) -> dict:
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    bootstrap = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "mean": float(values.mean()),
        "low": float(low),
        "high": float(high),
        "n": int(len(values)),
    }


def _entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    return -(probs * np.log2(np.clip(probs, 1e-30, None))).sum(axis=-1)


def _center(logits: np.ndarray) -> np.ndarray:
    return logits - logits.mean(axis=-1, keepdims=True)


def _fmt(cell: dict, scale: float = 1.0, digits: int = 1) -> str:
    return (
        f"{scale * cell['mean']:+.{digits}f} "
        f"[{scale * cell['low']:+.{digits}f}, {scale * cell['high']:+.{digits}f}]"
    )


def analyze(
    baseline_path: Path,
    evaluation_path: Path,
    neutral_path: Path,
    standard_game_path: Path | None,
    output_dir: Path,
    seed: int,
) -> dict:
    baseline = _load(baseline_path)
    evaluation = _load(evaluation_path)
    neutral = _load(neutral_path)
    qids = list(baseline)
    if set(qids) != set(evaluation) or set(qids) != set(neutral):
        raise ValueError("Baseline, Evaluation, and Neutral question sets differ")
    standard = _load(standard_game_path) if standard_game_path else None
    if standard is not None and set(qids) != set(standard):
        raise ValueError("Standard-Game question set differs")

    def answer(row: dict, kind: str) -> str:
        if kind == "evaluation":
            value = row["aggregated_ad_answer_new_letter"]
        else:
            value = row["answer"]
        if value not in LETTERS:
            raise ValueError(f"Non-A-D answer: {value!r}")
        return value

    labels = {
        "baseline": np.asarray([LETTERS.index(answer(baseline[q], "baseline")) for q in qids]),
        "evaluation": np.asarray([LETTERS.index(answer(evaluation[q], "evaluation")) for q in qids]),
        "neutral": np.asarray([LETTERS.index(answer(neutral[q], "neutral")) for q in qids]),
    }
    if standard is not None:
        labels["standard_game"] = np.asarray(
            [LETTERS.index(answer(standard[q], "standard_game")) for q in qids]
        )
    logits = {
        "baseline": np.asarray([baseline[q]["aggregated_ad_logits"] for q in qids], dtype=float),
        "evaluation": np.asarray([evaluation[q]["aggregated_ad_logits"] for q in qids], dtype=float),
        "neutral": np.asarray([neutral[q]["aggregated_ad_logits"] for q in qids], dtype=float),
    }
    if standard is not None:
        logits["standard_game"] = np.asarray(
            [standard[q]["aggregated_ad_logits"] for q in qids], dtype=float
        )
    correct = np.asarray([LETTERS.index(baseline[q]["correct_answer"]) for q in qids])
    prior = labels["baseline"]
    rng = np.random.default_rng(seed)
    entropy = {key: _entropy(value) for key, value in logits.items()}
    switch = {
        key: labels[key] != prior
        for key in labels
        if key != "baseline"
    }

    baseline_other = logits["baseline"].copy()
    baseline_other[np.arange(len(qids)), prior] = -np.inf
    runner = baseline_other.argmax(axis=-1)
    baseline_wrong = prior != correct

    conditions = list(labels)
    summary: dict = {
        "n": len(qids),
        "switch_readout": (
            "Aggregated A-D answer argmax: evaluation uses "
            "aggregated_ad_answer_new_letter; reused baseline/neutral artifacts "
            "use their answer field from the same aggregated A-D pipeline. This "
            "is distinct from the unrestricted top-token endpoint used by the "
            "feedback factorial."
        ),
        "prompts": {
            "evaluation": "Your answer was incorrect. Choose the answer again.",
            "neutral": "Your answer was lost. Choose the answer again.",
        },
        "accuracy": {
            key: float(np.mean(labels[key] == correct)) for key in conditions
        },
        "switch_rate": {
            key: float(switch[key].mean()) for key in switch
        },
        "mean_ad_entropy_bits": {
            key: float(entropy[key].mean()) for key in conditions
        },
        "evaluation_minus_neutral": {
            "switch_rate": _ci(switch["evaluation"].astype(float) - switch["neutral"].astype(float), rng),
            "entropy_bits": _ci(entropy["evaluation"] - entropy["neutral"], rng),
        },
        "evaluation_minus_baseline_entropy_bits": _ci(
            entropy["evaluation"] - entropy["baseline"], rng
        ),
        "neutral_minus_baseline_entropy_bits": _ci(
            entropy["neutral"] - entropy["baseline"], rng
        ),
    }
    for key in ("evaluation", "neutral"):
        changed = switch[key]
        changed_wrong = changed & baseline_wrong
        summary[f"{key}_switches_to_runner"] = {
            "hits": int(np.sum(changed & (labels[key] == runner))),
            "n": int(changed.sum()),
            "rate": float(np.mean(labels[key][changed] == runner[changed])),
            "one_sided_binomial_p_vs_one_third": float(
                binomtest(
                    int(np.sum(changed & (labels[key] == runner))),
                    int(changed.sum()),
                    1 / 3,
                    alternative="greater",
                ).pvalue
            ),
        }
        summary[f"{key}_accincor_changed_baseline_wrong"] = {
            "hits": int(np.sum(changed_wrong & (labels[key] == correct))),
            "n": int(changed_wrong.sum()),
            "rate": float(np.mean(labels[key][changed_wrong] == correct[changed_wrong])),
            "one_sided_binomial_p_vs_one_third": float(
                binomtest(
                    int(np.sum(changed_wrong & (labels[key] == correct))),
                    int(changed_wrong.sum()),
                    1 / 3,
                    alternative="greater",
                ).pvalue
            ),
        }

    summary["by_baseline_letter"] = {}
    for li, letter in enumerate(LETTERS):
        mask = prior == li
        summary["by_baseline_letter"][letter] = {
            "n": int(mask.sum()),
            "evaluation_switch_rate": float(switch["evaluation"][mask].mean()),
            "neutral_switch_rate": float(switch["neutral"][mask].mean()),
            "evaluation_minus_neutral_switch_rate": _ci(
                switch["evaluation"][mask].astype(float)
                - switch["neutral"][mask].astype(float),
                rng,
            ),
        }

    # Question-aligned Baseline-rank redistribution.
    order = np.argsort(-logits["baseline"], axis=-1, kind="stable")
    rank_summary = {}
    for left, right, name in (
        ("evaluation", "baseline", "evaluation_minus_baseline"),
        ("neutral", "baseline", "neutral_minus_baseline"),
        ("evaluation", "neutral", "evaluation_minus_neutral"),
    ):
        delta = _center(logits[left]) - _center(logits[right])
        aligned = np.take_along_axis(delta, order, axis=-1)
        rank_summary[name] = {
            f"baseline_rank_{rank + 1}": _ci(aligned[:, rank], rng)
            for rank in range(4)
        }
    summary["centered_logit_redistribution"] = rank_summary
    if standard is not None:
        summary["standard_different_action_context"] = {
            "prompt": "Your answer was incorrect. Choose a different answer.",
            "accuracy": summary["accuracy"]["standard_game"],
            "switch_rate": summary["switch_rate"]["standard_game"],
            "mean_ad_entropy_bits": summary["mean_ad_entropy_bits"]["standard_game"],
            "standard_minus_neutral_switch_rate": float(
                summary["switch_rate"]["standard_game"]
                - summary["switch_rate"]["neutral"]
            ),
            "evaluation_again_minus_neutral_fraction_of_standard_lift": float(
                summary["evaluation_minus_neutral"]["switch_rate"]["mean"]
                / (
                    summary["switch_rate"]["standard_game"]
                    - summary["switch_rate"]["neutral"]
                )
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lift = summary["evaluation_minus_neutral"]["switch_rate"]
    ent = summary["evaluation_minus_neutral"]["entropy_bits"]
    e_runner = summary["evaluation_switches_to_runner"]
    e_acc = summary["evaluation_accincor_changed_baseline_wrong"]
    lines = [
        "# Non-remapped action-matched SimpleMC behavior",
        "",
        "## Design",
        "",
        "The option mapping is identical on both presentations. Evaluation and Neutral differ by exactly one model-visible word:",
        "",
        "- Evaluation: `Your answer was incorrect. Choose the answer again.`",
        "- Neutral: `Your answer was lost. Choose the answer again.`",
        "",
        "Baseline and Neutral are reused from the exact existing run; only Evaluation was newly collected.",
        "",
        "### Baseline comparator",
        "",
        "The Baseline artifact contains two answer fields from two different runs:",
        "",
        "- `answer`: the model's answer in the current, exactly matched prompt format;",
        "- `baseline_answer`: an answer imported from an earlier compiled run.",
        "",
        "This analysis uses `answer`, because it is the Baseline generated under the same current formatting as Evaluation and Neutral. The two fields disagree on 59/500 questions and must not be pooled or interchanged. An older report in `qwen36_27b_simplemc_token_matched_feedback/analysis/` used the imported field; that report is now explicitly marked as historical.",
        "",
        "Here **switch** is defined by the aggregated A-D answer argmax, not the unrestricted top token. The feedback factorial's primary unrestricted-token endpoint is a different readout and should not be mixed with these rates.",
        "",
        "## Results",
        "",
        "| Condition | Accuracy | Switch rate | Mean A-D entropy |",
        "|---|---:|---:|---:|",
        f"| Baseline | {summary['accuracy']['baseline']:.1%} | -- | {summary['mean_ad_entropy_bits']['baseline']:.3f} |",
        f"| Evaluation | {summary['accuracy']['evaluation']:.1%} | {summary['switch_rate']['evaluation']:.1%} | {summary['mean_ad_entropy_bits']['evaluation']:.3f} |",
        f"| Neutral | {summary['accuracy']['neutral']:.1%} | {summary['switch_rate']['neutral']:.1%} | {summary['mean_ad_entropy_bits']['neutral']:.3f} |",
        "",
        f"Evaluation-minus-Neutral switching: **{_fmt(lift, 100)} percentage points**.",
        f"Evaluation-minus-Neutral entropy: **{_fmt(ent, 1, 3)} bits**.",
        f"Among Evaluation switches, {e_runner['hits']}/{e_runner['n']} ({e_runner['rate']:.1%}) select the Baseline runner-up.",
        f"The one-sided binomial p-value against 1/3 is {e_runner['one_sided_binomial_p_vs_one_third']:.3g}.",
        f"Among changed Baseline-wrong Evaluation trials, {e_acc['hits']}/{e_acc['n']} ({e_acc['rate']:.1%}) move to the correct answer.",
        f"The one-sided binomial p-value against 1/3 is {e_acc['one_sided_binomial_p_vs_one_third']:.3g}.",
        "",
        "Thus Evaluation passes lift, runner-up, and changed-wrong accuracy checks, but fails entropy preservation: its A-D entropy rises substantially relative to both Baseline and Neutral.",
        "",
        "## Switching by Baseline answer letter",
        "",
        "| Baseline letter | n | Evaluation | Neutral | Difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for letter in LETTERS:
        row = summary["by_baseline_letter"][letter]
        lines.append(
            f"| {letter} | {row['n']} | {row['evaluation_switch_rate']:.1%} | "
            f"{row['neutral_switch_rate']:.1%} | "
            f"{_fmt(row['evaluation_minus_neutral_switch_rate'], 100)} pp |"
        )
    lines += [
        "",
        "Complete continuous rank-aligned logit redistributions and confidence intervals are in `summary.json`.",
    ]
    if standard is not None:
        context = summary["standard_different_action_context"]
        lines += [
            "",
            "## Context: explicit different-answer instruction",
            "",
            f"Under the same current Baseline comparator, the existing `incorrect + different` condition switches on {context['switch_rate']:.1%} of questions, versus {summary['switch_rate']['neutral']:.1%} for Neutral: a {context['standard_minus_neutral_switch_rate']:.1%} lift. The single-word action-matched Evaluation condition therefore reproduces {context['evaluation_again_minus_neutral_fraction_of_standard_lift']:.1%} of that raw switching lift.",
        ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--neutral", type=Path, required=True)
    parser.add_argument("--standard-game", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    analyze(
        args.baseline,
        args.evaluation,
        args.neutral,
        args.standard_game,
        args.output_dir,
        args.seed,
    )


if __name__ == "__main__":
    main()
