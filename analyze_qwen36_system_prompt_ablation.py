from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportions_ztest


LETTERS = "ABCD"


def load_shards(root: Path) -> tuple[dict[str, str], dict[str, np.ndarray], dict[str, dict]]:
    choices: dict[str, str] = {}
    logits: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}
    for path in sorted(root.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["metadata"]))
        qid = meta["question_id"]
        values = np.asarray(data["canonical_logits"][-1], dtype=float)
        choices[qid] = LETTERS[int(values.argmax())]
        logits[qid] = values
        metadata[qid] = meta
    return choices, logits, metadata


def paired_change_test(first: np.ndarray, second: np.ndarray) -> dict:
    first_only = int(np.sum(first & ~second))
    second_only = int(np.sum(second & ~first))
    discordant = first_only + second_only
    p = float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "first_rate": float(first.mean()),
        "second_rate": float(second.mean()),
        "difference": float(first.mean() - second.mean()),
        "first_only": first_only,
        "second_only": second_only,
        "both": int(np.sum(first & second)),
        "neither": int(np.sum(~first & ~second)),
        "mcnemar_exact_p": p,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        type=Path,
        default=Path("outputs/reproduction/qwen36_27b_simplemc_no_system_incorrect/incorrect_no_system_setup_results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reproduction/qwen36_27b_simplemc_no_system_incorrect"),
    )
    args = parser.parse_args()

    baseline, baseline_logits, metadata = load_shards(
        Path("outputs/mechanistic/qwen36_27b_simplemc/shards/baseline")
    )
    game, _, _ = load_shards(
        Path("outputs/mechanistic/qwen36_27b_simplemc/shards/incorrect")
    )
    neutral, _, _ = load_shards(
        Path("outputs/mechanistic/qwen36_27b_simplemc_clean/shards/neutral")
    )
    raw_ablation = json.loads(args.ablation.read_text())
    ablation = {qid: row["answer"] for qid, row in raw_ablation["results"].items()}
    qids = sorted(set(baseline) & set(game) & set(neutral) & set(ablation))
    if len(qids) != 500:
        raise ValueError(f"Expected 500 common questions, found {len(qids)}")
    if any(raw_ablation["results"][qid]["full_vocab_top_token"] not in LETTERS for qid in qids):
        raise ValueError("At least one ablation response was not answer-only A-D")

    base = np.array([baseline[qid] for qid in qids])
    standard = np.array([game[qid] for qid in qids])
    lost = np.array([neutral[qid] for qid in qids])
    ablated = np.array([ablation[qid] for qid in qids])
    standard_changed = standard != base
    neutral_changed = lost != base
    ablated_changed = ablated != base

    correct = np.array([metadata[qid]["correct_answer"] for qid in qids])
    baseline_correct = base == correct
    changed_wrong = ablated_changed & ~baseline_correct
    acc_hits = int(np.sum(ablated[changed_wrong] == correct[changed_wrong]))
    acc_n = int(np.sum(changed_wrong))
    acc_z, acc_z_p = proportions_ztest(acc_hits, acc_n, value=1 / 3)
    acc_exact = binomtest(acc_hits, acc_n, 1 / 3, alternative="greater")

    runner = []
    for qid in qids:
        values = baseline_logits[qid]
        order = np.argsort(values)[::-1]
        runner.append(LETTERS[int(order[1])])
    runner = np.array(runner)
    runner_hits = int(np.sum(ablated[ablated_changed] == runner[ablated_changed]))
    runner_n = int(np.sum(ablated_changed))
    runner_z, runner_z_p = proportions_ztest(runner_hits, runner_n, value=1 / 3)
    runner_exact = binomtest(runner_hits, runner_n, 1 / 3, alternative="greater")

    versus_neutral = paired_change_test(ablated_changed, neutral_changed)
    versus_game = paired_change_test(standard_changed, ablated_changed)
    natural_gap = float(standard_changed.mean() - neutral_changed.mean())
    removed_gap = float(standard_changed.mean() - ablated_changed.mean())
    summary = {
        "n": len(qids),
        "prompt_ablation": (
            "Remove 'Your previous answer was incorrect. Respond ONLY with your new answer' "
            "from the initial system message; retain the ordinary Game user feedback."
        ),
        "accuracy": {
            "baseline": float(np.mean(base == correct)),
            "standard_game": float(np.mean(standard == correct)),
            "ablated_game": float(np.mean(ablated == correct)),
            "neutral": float(np.mean(lost == correct)),
        },
        "switch_rates": {
            "standard_game": float(standard_changed.mean()),
            "ablated_game": float(ablated_changed.mean()),
            "neutral": float(neutral_changed.mean()),
        },
        "ablated_game_vs_neutral": versus_neutral,
        "standard_game_vs_ablated_game": versus_game,
        "fraction_standard_game_neutral_gap_removed": removed_gap / natural_gap,
        "answer_agreement_standard_vs_ablated": float(np.mean(standard == ablated)),
        "accincor_changed_baseline_incorrect": {
            "hits": acc_hits,
            "n": acc_n,
            "rate": acc_hits / acc_n,
            "two_sided_z_p_vs_one_third": float(acc_z_p),
            "exact_one_sided_p_vs_one_third": float(acc_exact.pvalue),
        },
        "second_choice": {
            "hits": runner_hits,
            "n": runner_n,
            "rate": runner_hits / runner_n,
            "two_sided_z_p_vs_one_third": float(runner_z_p),
            "exact_one_sided_p_vs_one_third": float(runner_exact.pvalue),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "behavioral_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    report = f"""# Qwen3.6-27B SimpleMC system-prompt ablation

## Manipulation

All 500 frozen SimpleMC questions were run through the exact self-hosted BF16
checkpoint. The final user feedback remained `Your answer was incorrect. Choose
a different answer.` The condition-specific system line `Your previous answer
was incorrect. Respond ONLY with your new answer` was removed, leaving only the
common A--D answer-format instruction in the system message.

## Behavioral result

| Condition | Accuracy | Switch rate |
|---|---:|---:|
| Standard Game | {summary['accuracy']['standard_game']:.1%} | {summary['switch_rates']['standard_game']:.1%} |
| Game without advance system cue | {summary['accuracy']['ablated_game']:.1%} | {summary['switch_rates']['ablated_game']:.1%} |
| Neutral | {summary['accuracy']['neutral']:.1%} | {summary['switch_rates']['neutral']:.1%} |

Removing the system cue reduced switching by {100 * removed_gap:.1f} percentage points ({versus_game['first_only']}
standard-only versus {versus_game['second_only']} ablation-only switch trials; exact paired
p={versus_game['mcnemar_exact_p']:.3g}). This removes
{summary['fraction_standard_game_neutral_gap_removed']:.1%} of the original
Game--Neutral switch-rate gap. The ablated Game nevertheless remains well above
Neutral: {100 * versus_neutral['difference']:.1f} percentage points ({versus_neutral['first_only']} ablation-only
versus {versus_neutral['second_only']} Neutral-only switch trials; exact paired
p={versus_neutral['mcnemar_exact_p']:.3g}).

| Behavioral diagnostic for ablated Game | Result |
|---|---:|
| AccIncor among changed baseline-wrong trials | {acc_hits}/{acc_n} = {acc_hits / acc_n:.1%}; z p={acc_z_p:.3g} |
| Switches to Baseline runner-up | {runner_hits}/{runner_n} = {runner_hits / runner_n:.1%}; z p={runner_z_p:.3g} |
| Exact answer agreement with standard Game | {summary['answer_agreement_standard_vs_ablated']:.1%} |

## Conclusion

The advance system cue matters, but it is not the primary cause of Second Chance
behavior. Removing it eliminates about {summary['fraction_standard_game_neutral_gap_removed']:.0%}
of the differential switching, while approximately
{1 - summary['fraction_standard_game_neutral_gap_removed']:.0%} remains driven by the
ordinary feedback turn or its interaction with the rest of the prompt.
"""
    (args.output_dir / "BEHAVIORAL_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
