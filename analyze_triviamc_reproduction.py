#!/usr/bin/env python3
"""Compute the paper-exact Qwen TriviaMC Second Chance replication tests."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, ttest_rel, wilcoxon
from statsmodels.stats.proportion import proportion_confint, proportions_ztest


STATE_PATH = Path("outputs/reproduction/triviamc/run_state.json")
OUTPUT_JSON = Path("outputs/reproduction/triviamc/replication_summary.json")
OUTPUT_REPORT = Path("outputs/reproduction/triviamc/REPLICATION_REPORT.md")
LETTERS = ("A", "B", "C", "D")
LETTER_RE = re.compile(r"^\s*([A-D])\s*[\.)\]:,-]?\s*$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_JSON.parent)
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_results(paths: list[str | Path]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for path in paths:
        results = load_json(path)["results"]
        overlap = set(merged).intersection(results)
        if overlap:
            raise ValueError(f"Duplicate IDs across condition files: {len(overlap)}")
        merged.update(results)
    return merged


def top_raw_probs(raw: object, k: int = 4) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    valid = [
        (str(token), float(value))
        for token, value in raw.items()
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    ]
    return dict(sorted(valid, key=lambda item: item[1], reverse=True)[:k])


def canonical_probs(raw: object) -> dict[str, float]:
    result: dict[str, float] = {}
    if not isinstance(raw, dict):
        return result
    for token, value in raw.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            continue
        match = LETTER_RE.match(str(token))
        if match:
            letter = match.group(1).upper()
            result[letter] = result.get(letter, 0.0) + float(value)
    return result


def entropy(values: list[float], normalize: bool = False) -> float | None:
    probabilities = np.asarray([value for value in values if value > 1e-12], dtype=float)
    if not len(probabilities):
        return None
    if normalize:
        probabilities = probabilities / probabilities.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def observed_letter_entropy(probabilities: dict[str, float]) -> float | None:
    """Entropy after zero-filling censored A-D letters and renormalizing."""
    if not probabilities:
        return None
    values = np.asarray([probabilities.get(letter, 0.0) for letter in LETTERS])
    values = values / values.sum()
    positive = values[values > 0]
    return float(-np.sum(positive * np.log2(positive)))


def paired(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    n = len(array)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(n))
    t = ttest_rel(array, np.zeros(n))
    w = wilcoxon(array)
    return {
        "n": n,
        "mean": mean,
        "ci95_normal": [mean - 1.96 * se, mean + 1.96 * se],
        "paired_t_p": float(t.pvalue),
        "wilcoxon_p": float(w.pvalue),
    }


def entropy_contrast(reference: dict[str, dict], condition: dict[str, dict]) -> dict:
    legacy_deltas = []
    answer_deltas = []
    observed_answer_deltas = []
    reference_complete = 0
    condition_complete = 0
    pair_complete = 0
    for qid in sorted(set(reference).intersection(condition)):
        reference_top4 = top_raw_probs(reference[qid].get("probs"), 4)
        condition_top4 = top_raw_probs(condition[qid].get("probs"), 4)
        reference_h = entropy(list(reference_top4.values()))
        condition_h = entropy(list(condition_top4.values()))
        if reference_h is not None and condition_h is not None:
            legacy_deltas.append(condition_h - reference_h)

        reference_letters = canonical_probs(reference[qid].get("probs"))
        condition_letters = canonical_probs(condition[qid].get("probs"))
        reference_observed_h = observed_letter_entropy(reference_letters)
        condition_observed_h = observed_letter_entropy(condition_letters)
        if reference_observed_h is not None and condition_observed_h is not None:
            observed_answer_deltas.append(condition_observed_h - reference_observed_h)
        reference_is_complete = set(reference_letters) == set(LETTERS)
        condition_is_complete = set(condition_letters) == set(LETTERS)
        reference_complete += int(reference_is_complete)
        condition_complete += int(condition_is_complete)
        if reference_is_complete and condition_is_complete:
            pair_complete += 1
            answer_deltas.append(
                entropy([condition_letters[x] for x in LETTERS], normalize=True)
                - entropy([reference_letters[x] for x in LETTERS], normalize=True)
            )
    return {
        "paper_top4_raw_token_entropy": paired(legacy_deltas),
        "complete_A_D_normalized_entropy": paired(answer_deltas) if len(answer_deltas) > 1 else {"n": len(answer_deltas)},
        "observed_A_D_normalized_zero_fill": paired(observed_answer_deltas),
        "reference_complete_A_D": reference_complete,
        "condition_complete_A_D": condition_complete,
        "pair_complete_A_D": pair_complete,
    }


def paper_runner_up(baseline_trial: dict) -> str | None:
    clean = {
        token.strip(): probability
        for token, probability in top_raw_probs(baseline_trial.get("probs"), 4).items()
        if token.strip() != "T"
    }
    if len(clean) < 2:
        return None
    return sorted(clean, key=clean.get, reverse=True)[1]


def runner_up_test(baseline: dict[str, dict], condition: dict[str, dict]) -> dict:
    changed = [qid for qid, trial in condition.items() if trial.get("answer_changed")]
    covered = 0
    hits = 0
    for qid in changed:
        runner = paper_runner_up(baseline[qid])
        if runner is None:
            continue
        covered += 1
        hits += int(str(condition[qid].get("new_answer", "")).strip() == runner)
    z_stat, z_p = proportions_ztest(hits, covered, value=1 / 3)
    ci = proportion_confint(hits, covered, alpha=0.05, method="normal")
    return {
        "changed": len(changed),
        "covered": covered,
        "hits": hits,
        "rate": hits / covered,
        "ci95_normal": [float(ci[0]), float(ci[1])],
        "paper_two_sided_z_p": float(z_p),
        "exact_one_sided_binomial_p": float(
            binomtest(hits, covered, 1 / 3, alternative="greater").pvalue
        ),
    }


def canonical_runner_up_test(
    baseline: dict[str, dict], condition: dict[str, dict]
) -> dict:
    """Sensitivity check using aggregated A-D mass from the richer capture."""
    changed = [qid for qid, trial in condition.items() if trial.get("answer_changed")]
    covered = 0
    hits = 0
    for qid in changed:
        probabilities = canonical_probs(baseline[qid].get("probs"))
        first = str(baseline[qid].get("subject_answer", "")).strip()
        alternatives = [letter for letter in LETTERS if letter != first]
        if first not in LETTERS or not all(letter in probabilities for letter in alternatives):
            continue
        runner = max(alternatives, key=probabilities.get)
        covered += 1
        hits += int(str(condition[qid].get("new_answer", "")).strip() == runner)
    return {
        "changed": len(changed),
        "covered": covered,
        "hits": hits,
        "rate": hits / covered,
        "exact_one_sided_binomial_p": float(
            binomtest(hits, covered, 1 / 3, alternative="greater").pvalue
        ),
    }


def observed_runner_up_test(
    baseline: dict[str, dict], condition: dict[str, dict]
) -> dict:
    """Runner-up after aggregating captured A-D variants.

    A changed trial is covered whenever at least one alternative letter appears
    in the capture; censored alternatives are treated as zero probability.
    """
    changed = [qid for qid, trial in condition.items() if trial.get("answer_changed")]
    covered = 0
    hits = 0
    for qid in changed:
        probabilities = canonical_probs(baseline[qid].get("probs"))
        first = str(baseline[qid].get("subject_answer", "")).strip()
        alternatives = [
            letter for letter in LETTERS if letter != first and letter in probabilities
        ]
        if first not in LETTERS or not alternatives:
            continue
        runner = max(alternatives, key=probabilities.get)
        covered += 1
        hits += int(str(condition[qid].get("new_answer", "")).strip() == runner)
    z_stat, z_p = proportions_ztest(hits, covered, value=1 / 3)
    return {
        "changed": len(changed),
        "covered": covered,
        "hits": hits,
        "rate": hits / covered,
        "paper_two_sided_z_p": float(z_p),
        "exact_one_sided_binomial_p": float(
            binomtest(hits, covered, 1 / 3, alternative="greater").pvalue
        ),
    }


def provider_metadata(*condition_results: dict[str, dict]) -> dict:
    providers = Counter()
    returned = Counter()
    requested = Counter()
    missing = 0
    for results in condition_results:
        for trial in results.values():
            metadata = trial.get("call_metadata")
            if not isinstance(metadata, dict):
                missing += 1
                continue
            providers[str(metadata.get("serving_provider"))] += 1
            returned[str(metadata.get("top_logprobs_returned"))] += 1
            requested[str(metadata.get("top_logprobs_requested"))] += 1
    return {
        "serving_providers": dict(providers),
        "top_logprobs_requested": dict(requested),
        "top_logprobs_returned": dict(returned),
        "missing_metadata": missing,
    }


def main() -> None:
    args = parse_args()
    state = load_json(args.state)
    output_json = args.output_dir / "replication_summary.json"
    output_report = args.output_dir / "REPLICATION_REPORT.md"
    n_expected = int(state.get("n_questions", 500))
    if not state.get("complete"):
        raise RuntimeError("TriviaMC run is not complete")
    baseline = load_results([state["baseline_compiled_path"]])
    stages = state["second_chance"]
    game = load_results([
        stages["incorrect_baseline_incorrect"]["path"],
        stages["incorrect_baseline_correct"]["path"],
    ])
    neutral = load_results([
        stages["neutral_baseline_incorrect"]["path"],
        stages["neutral_baseline_correct"]["path"],
    ])
    if not (len(baseline) == len(game) == len(neutral) == n_expected):
        raise ValueError(
            f"Expected {n_expected} trials per condition: {len(baseline)}, {len(game)}, {len(neutral)}"
        )

    game_changed = sum(bool(trial.get("answer_changed")) for trial in game.values())
    neutral_changed = sum(bool(trial.get("answer_changed")) for trial in neutral.values())
    both = sum(game[qid].get("answer_changed") and neutral[qid].get("answer_changed") for qid in baseline)
    game_only = sum(game[qid].get("answer_changed") and not neutral[qid].get("answer_changed") for qid in baseline)
    neutral_only = sum(neutral[qid].get("answer_changed") and not game[qid].get("answer_changed") for qid in baseline)
    discordant = game_only + neutral_only
    mcnemar_p = binomtest(game_only, discordant, 0.5).pvalue if discordant else 1.0
    game_rate = game_changed / n_expected
    neutral_rate = neutral_changed / n_expected

    baseline_incorrect = [qid for qid, trial in baseline.items() if not trial.get("is_correct")]
    changed_baseline_incorrect = [
        qid for qid in baseline_incorrect if game[qid].get("answer_changed")
    ]
    game_correct_on_incorrect = sum(
        bool(game[qid].get("is_correct")) for qid in baseline_incorrect
    )
    game_correct_on_changed_incorrect = sum(
        bool(game[qid].get("is_correct")) for qid in changed_baseline_incorrect
    )
    if game_correct_on_incorrect != game_correct_on_changed_incorrect:
        raise ValueError("An unchanged baseline-incorrect answer became correct")
    unconditional_accuracy_incorrect = game_correct_on_incorrect / len(baseline_incorrect)
    accuracy_incorrect = (
        game_correct_on_changed_incorrect / len(changed_baseline_incorrect)
    )
    acc_z_stat, acc_z_p = proportions_ztest(
        game_correct_on_changed_incorrect,
        len(changed_baseline_incorrect),
        value=1 / 3,
    )
    acc_exact_one_third = binomtest(
        game_correct_on_changed_incorrect,
        len(changed_baseline_incorrect),
        1 / 3,
        alternative="greater",
    )

    game_entropy = entropy_contrast(baseline, game)
    neutral_entropy = entropy_contrast(baseline, neutral)
    game_runner = runner_up_test(baseline, game)
    neutral_runner = runner_up_test(baseline, neutral)
    game_canonical_runner = canonical_runner_up_test(baseline, game)
    neutral_canonical_runner = canonical_runner_up_test(baseline, neutral)
    game_observed_runner = observed_runner_up_test(baseline, game)
    neutral_observed_runner = observed_runner_up_test(baseline, neutral)
    game_entropy_p = game_entropy["paper_top4_raw_token_entropy"]["wilcoxon_p"]
    game_entropy_mean = game_entropy["paper_top4_raw_token_entropy"]["mean"]
    game_observed_entropy = game_entropy["observed_A_D_normalized_zero_fill"]

    summary = {
        "model": state["model"],
        "dataset": state["dataset"],
        "n": n_expected,
        "baseline": {
            "correct": sum(bool(trial.get("is_correct")) for trial in baseline.values()),
            "accuracy": sum(bool(trial.get("is_correct")) for trial in baseline.values()) / n_expected,
        },
        "lift": {
            "game_changed": game_changed,
            "game_rate": game_rate,
            "neutral_changed": neutral_changed,
            "neutral_rate": neutral_rate,
            "absolute_lift": game_rate - neutral_rate,
            "normalized_lift": (game_rate - neutral_rate) / (1 - neutral_rate),
            "both_changed": both,
            "game_only": game_only,
            "neutral_only": neutral_only,
            "mcnemar_exact_p": float(mcnemar_p),
        },
        "accuracy_on_baseline_incorrect": {
            "n": len(baseline_incorrect),
            "correct": game_correct_on_incorrect,
            "accuracy": unconditional_accuracy_incorrect,
            "definition": "diagnostic accuracy over every baseline-incorrect trial",
        },
        "paper_accincor_changed_baseline_incorrect": {
            "n": len(changed_baseline_incorrect),
            "correct": game_correct_on_changed_incorrect,
            "accuracy": accuracy_incorrect,
            "two_sided_z_stat_vs_one_third": float(acc_z_stat),
            "two_sided_z_p_vs_one_third": float(acc_z_p),
            "exact_one_sided_p_vs_one_third": float(acc_exact_one_third.pvalue),
            "definition": "accuracy conditional on baseline incorrect and answer changed",
        },
        "runner_up": {"game": game_runner, "neutral": neutral_runner},
        "runner_up_canonical_A_D": {
            "game": game_canonical_runner,
            "neutral": neutral_canonical_runner,
        },
        "runner_up_observed_A_D": {
            "game": game_observed_runner,
            "neutral": neutral_observed_runner,
        },
        "entropy_minus_baseline": {"game": game_entropy, "neutral": neutral_entropy},
        "paper_tests": {
            "Lift": bool(game_rate > neutral_rate and mcnemar_p < 0.05),
            "AccIncor": bool(accuracy_incorrect > 1 / 3 and acc_z_p < 0.05),
            "SecChoice": bool(game_runner["rate"] > 1 / 3 and game_runner["paper_two_sided_z_p"] < 0.05),
            "NoEntInc": not bool(game_entropy_mean > 0 and game_entropy_p < 0.05),
        },
        "coverage_aware_tests": {
            "Lift": bool(game_rate > neutral_rate and mcnemar_p < 0.05),
            "AccIncor": bool(accuracy_incorrect > 1 / 3 and acc_z_p < 0.05),
            "SecChoice": bool(
                game_observed_runner["rate"] > 1 / 3
                and game_observed_runner["paper_two_sided_z_p"] < 0.05
            ),
            "NoEntInc": not bool(
                game_observed_entropy["mean"] > 0
                and game_observed_entropy["wilcoxon_p"] < 0.05
            ),
        },
        "capture_metadata": provider_metadata(baseline, game, neutral),
    }
    write_parent = output_json.parent
    write_parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    game_h = game_entropy["paper_top4_raw_token_entropy"]
    neutral_h = neutral_entropy["paper_top4_raw_token_entropy"]
    game_answer_h = game_entropy["complete_A_D_normalized_entropy"]
    neutral_answer_h = neutral_entropy["complete_A_D_normalized_entropy"]
    game_observed_h = game_entropy["observed_A_D_normalized_zero_fill"]
    neutral_observed_h = neutral_entropy["observed_A_D_normalized_zero_fill"]
    captured_top_logprobs = int(state.get("top_logprobs", 4))
    report = f"""# {state['model']} {state['dataset']} Second Chance replication

## Paper tests

| Test | Result | Pass |
|---|---:|:---:|
| Lift | Game {game_rate:.1%}; neutral {neutral_rate:.1%}; normalized lift {summary['lift']['normalized_lift']:.3f}; paired p={mcnemar_p:.3g} | {'✓' if summary['paper_tests']['Lift'] else 'X'} |
| AccIncor | {game_correct_on_changed_incorrect}/{len(changed_baseline_incorrect)} changed baseline-incorrect trials = {accuracy_incorrect:.1%}; two-sided z p vs 1/3={acc_z_p:.3g}; exact one-sided p vs 1/3={acc_exact_one_third.pvalue:.3g} | {'✓' if summary['paper_tests']['AccIncor'] else 'X'} |
| SecChoice | {game_runner['hits']}/{game_runner['covered']} = {game_runner['rate']:.1%}; paper z p={game_runner['paper_two_sided_z_p']:.3g} | {'✓' if summary['paper_tests']['SecChoice'] else 'X'} |
| NoEntInc | Game minus baseline {game_h['mean']:+.3f} bits [{game_h['ci95_normal'][0]:+.3f}, {game_h['ci95_normal'][1]:+.3f}]; Wilcoxon p={game_h['wilcoxon_p']:.3g} | {'✓' if summary['paper_tests']['NoEntInc'] else 'X'} |

Baseline accuracy was {summary['baseline']['correct']}/{n_expected} = {summary['baseline']['accuracy']:.1%}. Neutral minus baseline top-four-token entropy was {neutral_h['mean']:+.3f} bits [{neutral_h['ci95_normal'][0]:+.3f}, {neutral_h['ci95_normal'][1]:+.3f}].

As a separately named diagnostic, unconditional Game accuracy across all {len(baseline_incorrect)} baseline-incorrect trials was {game_correct_on_incorrect}/{len(baseline_incorrect)} = {unconditional_accuracy_incorrect:.1%}.

## Probability coverage

{captured_top_logprobs} top tokens were requested. Complete canonical A-D coverage was {game_entropy['reference_complete_A_D']}/{n_expected} at baseline, {game_entropy['condition_complete_A_D']}/{n_expected} in the Game, and {neutral_entropy['condition_complete_A_D']}/{n_expected} in neutral. The paper-exact entropy test above still uses only the four highest-probability raw tokens.

After aggregating A-D token variants, assigning zero to censored letters, and renormalizing observed answer mass, Game minus baseline entropy was {game_observed_h['mean']:+.3f} bits [{game_observed_h['ci95_normal'][0]:+.3f}, {game_observed_h['ci95_normal'][1]:+.3f}] (Wilcoxon p={game_observed_h['wilcoxon_p']:.3g}); neutral minus baseline was {neutral_observed_h['mean']:+.3f} bits [{neutral_observed_h['ci95_normal'][0]:+.3f}, {neutral_observed_h['ci95_normal'][1]:+.3f}]. The stricter complete-A-D subset gives Game minus baseline {game_answer_h['mean']:+.3f} bits on {game_answer_h['n']} trials (Wilcoxon p={game_answer_h['wilcoxon_p']:.3g}).

Aggregating letter-token variants identifies a captured baseline alternative on {game_observed_runner['covered']}/{game_observed_runner['changed']} changed Game trials. The model selected that runner-up {game_observed_runner['hits']}/{game_observed_runner['covered']} times ({game_observed_runner['rate']:.1%}); neutral was {neutral_observed_runner['hits']}/{neutral_observed_runner['covered']} ({neutral_observed_runner['rate']:.1%}). Requiring complete A-D coverage instead gives Game {game_canonical_runner['hits']}/{game_canonical_runner['covered']} ({game_canonical_runner['rate']:.1%}).
"""
    output_report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
