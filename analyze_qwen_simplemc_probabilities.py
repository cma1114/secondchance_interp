#!/usr/bin/env python3
"""Reproduce Qwen SimpleMC probability and entropy diagnostics.

The paper's entropy statistic is retained exactly as implemented in
``logres_sc_sqa_multi.py``: entropy is computed over the four raw tokens returned
by the API, without renormalization.  Additional answer-letter-only analyses are
reported because a returned non-answer token can displace one of A--D.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, ttest_rel, wilcoxon
from statsmodels.stats.proportion import proportion_confint, proportions_ztest


LETTERS = ("A", "B", "C", "D")
LETTER_RE = re.compile(r"^\s*([A-D])\s*[\.)\]:,-]?\s*$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("compiled_results_smc/qwen3-235b-a22b-2507_phase1_compiled.json"),
    )
    parser.add_argument(
        "--game",
        type=Path,
        nargs="+",
        default=[
            Path("secondchance_game_logs/qwen3-235b-a22b-2507_SimpleMC_redacted_temp0.0_1785521299_game_data.json"),
            Path("secondchance_game_logs/qwen3-235b-a22b-2507_SimpleMC_redacted_cor_temp0.0_1785522010_game_data.json"),
        ],
    )
    parser.add_argument(
        "--neutral",
        type=Path,
        nargs="+",
        default=[
            Path("secondchance_game_logs/qwen3-235b-a22b-2507_SimpleMC_neut_redacted_temp0.0_1785523104_game_data.json"),
            Path("secondchance_game_logs/qwen3-235b-a22b-2507_SimpleMC_neut_redacted_cor_temp0.0_1785523823_game_data.json"),
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reproduction/qwen_simplemc_probability_entropy.json"),
    )
    parser.add_argument(
        "--exclude-ids",
        type=Path,
        help="Optional format_exclusions.json; remove its excluded_union from all conditions.",
    )
    return parser.parse_args()


def load_results(paths: list[Path]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            results = json.load(handle)["results"]
        overlap = set(merged).intersection(results)
        if overlap:
            raise ValueError(f"Duplicate question IDs across {path}: {len(overlap)}")
        merged.update(results)
    return merged


def canonical_probs(raw: object) -> dict[str, float]:
    totals: dict[str, float] = {}
    if not isinstance(raw, dict):
        return totals
    for token, value in raw.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            continue
        match = LETTER_RE.match(str(token))
        if match:
            letter = match.group(1).upper()
            totals[letter] = totals.get(letter, 0.0) + float(value)
    return totals


def raw_entropy(raw: object) -> float | None:
    if not isinstance(raw, dict) or not raw:
        return None
    values = sorted([
        float(value)
        for value in raw.values()
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 1e-9
    ], reverse=True)[:4]
    if not values:
        return None
    return -sum(value * math.log2(value) for value in values)


def letter_entropy(probs: dict[str, float], normalize: bool) -> float | None:
    if set(probs) != set(LETTERS):
        return None
    values = np.array([probs[letter] for letter in LETTERS], dtype=float)
    if normalize:
        values = values / values.sum()
    return float(-np.sum(values * np.log2(values)))


def observed_letter_entropy(probs: dict[str, float]) -> float | None:
    """Entropy of observed A--D mass, treating censored letters as zero."""
    if not probs:
        return None
    values = np.array([probs.get(letter, 0.0) for letter in LETTERS], dtype=float)
    values = values / values.sum()
    positive = values[values > 0]
    return float(-np.sum(positive * np.log2(positive)))


def paired_summary(deltas: list[float]) -> dict:
    values = np.asarray(deltas, dtype=float)
    n = len(values)
    if n < 2:
        return {"n": n}
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(n))
    t_result = ttest_rel(values, np.zeros(n))
    try:
        w_result = wilcoxon(values)
        w_stat = float(w_result.statistic)
        w_p = float(w_result.pvalue)
    except ValueError:
        w_stat = 0.0
        w_p = 1.0
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(values)),
        "positive_fraction": float(np.mean(values > 0)),
        "ci95_normal": [mean - 1.96 * se, mean + 1.96 * se],
        "paired_t_stat": float(t_result.statistic),
        "paired_t_p": float(t_result.pvalue),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
    }


def entropy_contrast(
    baseline: dict[str, dict], condition: dict[str, dict]
) -> dict[str, dict]:
    legacy: list[float] = []
    complete_raw: list[float] = []
    complete_normalized: list[float] = []
    observed_normalized: list[float] = []
    condition_nonletter = 0
    baseline_nonletter = 0
    for qid in sorted(set(baseline).intersection(condition)):
        base_raw = baseline[qid].get("probs")
        cond_raw = condition[qid].get("probs")
        base_h = raw_entropy(base_raw)
        cond_h = raw_entropy(cond_raw)
        if base_h is not None and cond_h is not None:
            legacy.append(cond_h - base_h)

        base_letters = canonical_probs(base_raw)
        cond_letters = canonical_probs(cond_raw)
        base_observed_h = observed_letter_entropy(base_letters)
        cond_observed_h = observed_letter_entropy(cond_letters)
        if base_observed_h is not None and cond_observed_h is not None:
            observed_normalized.append(cond_observed_h - base_observed_h)
        baseline_nonletter += int(set(base_letters) != set(LETTERS))
        condition_nonletter += int(set(cond_letters) != set(LETTERS))
        base_letter_h = letter_entropy(base_letters, normalize=False)
        cond_letter_h = letter_entropy(cond_letters, normalize=False)
        if base_letter_h is not None and cond_letter_h is not None:
            complete_raw.append(cond_letter_h - base_letter_h)
            complete_normalized.append(
                letter_entropy(cond_letters, normalize=True)
                - letter_entropy(base_letters, normalize=True)
            )
    return {
        "legacy_raw_top4": paired_summary(legacy),
        "complete_A_D_raw": paired_summary(complete_raw),
        "complete_A_D_renormalized": paired_summary(complete_normalized),
        "observed_A_D_renormalized_zero_fill": paired_summary(observed_normalized),
        "reference_incomplete_A_D": baseline_nonletter,
        "condition_incomplete_A_D": condition_nonletter,
    }


def rank_contrast(
    baseline: dict[str, dict],
    target: dict[str, dict],
    reference: dict[str, dict],
) -> dict[str, dict]:
    raw_first: list[float] = []
    raw_runner: list[float] = []
    margin: list[float] = []
    centered_first: list[float] = []
    centered_runner: list[float] = []

    common_ids = sorted(set(baseline).intersection(target).intersection(reference))
    for qid in common_ids:
        base = canonical_probs(baseline[qid].get("probs"))
        target_probs = canonical_probs(target[qid].get("probs"))
        ref_probs = canonical_probs(reference[qid].get("probs"))
        first = str(baseline[qid].get("subject_answer", "")).strip()
        if first not in LETTERS or first not in base:
            continue
        alternatives = [letter for letter in LETTERS if letter != first and letter in base]
        if not alternatives:
            continue
        runner = max(alternatives, key=lambda letter: base[letter])
        if first in target_probs and runner in target_probs and first in ref_probs and runner in ref_probs:
            d_first = math.log(target_probs[first]) - math.log(ref_probs[first])
            d_runner = math.log(target_probs[runner]) - math.log(ref_probs[runner])
            raw_first.append(d_first)
            raw_runner.append(d_runner)
            margin.append(d_first - d_runner)

        if set(base) == set(LETTERS) and set(target_probs) == set(LETTERS) and set(ref_probs) == set(LETTERS):
            others = [letter for letter in LETTERS if letter not in (first, runner)]
            target_center = 0.5 * sum(math.log(target_probs[letter]) for letter in others)
            ref_center = 0.5 * sum(math.log(ref_probs[letter]) for letter in others)
            centered_first.append(
                (math.log(target_probs[first]) - target_center)
                - (math.log(ref_probs[first]) - ref_center)
            )
            centered_runner.append(
                (math.log(target_probs[runner]) - target_center)
                - (math.log(ref_probs[runner]) - ref_center)
            )
    return {
        "first_choice_raw_logprob": paired_summary(raw_first),
        "runner_up_raw_logprob": paired_summary(raw_runner),
        "first_minus_runner_margin": paired_summary(margin),
        "first_choice_centered_vs_ranks_3_4": paired_summary(centered_first),
        "runner_up_centered_vs_ranks_3_4": paired_summary(centered_runner),
    }


def normalized_letter_probs(raw: object) -> dict[str, float] | None:
    probabilities = canonical_probs(raw)
    if set(probabilities) != set(LETTERS):
        return None
    total = sum(probabilities.values())
    if total <= 0:
        return None
    return {letter: probabilities[letter] / total for letter in LETTERS}


def probability_contrast(
    baseline: dict[str, dict],
    target: dict[str, dict],
    reference: dict[str, dict],
) -> dict[str, dict]:
    """A-D-normalized probability changes for baseline-defined answer ranks."""
    first_changes: list[float] = []
    runner_changes: list[float] = []
    lower_changes: list[float] = []
    common_ids = sorted(set(baseline).intersection(target).intersection(reference))
    for qid in common_ids:
        base = normalized_letter_probs(baseline[qid].get("probs"))
        target_probs = normalized_letter_probs(target[qid].get("probs"))
        reference_probs = normalized_letter_probs(reference[qid].get("probs"))
        first = str(baseline[qid].get("subject_answer", "")).strip()
        if base is None or target_probs is None or reference_probs is None or first not in LETTERS:
            continue
        alternatives = [letter for letter in LETTERS if letter != first]
        runner = max(alternatives, key=base.get)
        lower = [letter for letter in alternatives if letter != runner]
        first_changes.append(target_probs[first] - reference_probs[first])
        runner_changes.append(target_probs[runner] - reference_probs[runner])
        lower_changes.append(
            0.5 * sum(target_probs[letter] - reference_probs[letter] for letter in lower)
        )
    return {
        "first_choice_probability": paired_summary(first_changes),
        "runner_up_probability": paired_summary(runner_changes),
        "mean_rank_3_4_probability": paired_summary(lower_changes),
    }


def rank_probability_levels(
    baseline: dict[str, dict], condition: dict[str, dict]
) -> dict[str, float]:
    first_values: list[float] = []
    runner_values: list[float] = []
    lower_values: list[float] = []
    for qid in sorted(set(baseline).intersection(condition)):
        base = normalized_letter_probs(baseline[qid].get("probs"))
        probabilities = normalized_letter_probs(condition[qid].get("probs"))
        first = str(baseline[qid].get("subject_answer", "")).strip()
        if base is None or probabilities is None or first not in LETTERS:
            continue
        alternatives = [letter for letter in LETTERS if letter != first]
        runner = max(alternatives, key=base.get)
        lower = [letter for letter in alternatives if letter != runner]
        first_values.append(probabilities[first])
        runner_values.append(probabilities[runner])
        lower_values.append(0.5 * sum(probabilities[letter] for letter in lower))
    return {
        "n": len(first_values),
        "first_choice_mean": float(np.mean(first_values)),
        "runner_up_mean": float(np.mean(runner_values)),
        "mean_rank_3_4": float(np.mean(lower_values)),
    }


def runner_up_test(
    baseline: dict[str, dict], condition: dict[str, dict], canonicalize: bool
) -> dict:
    hits = 0
    covered = 0
    changed_total = 0
    for qid in sorted(set(baseline).intersection(condition)):
        trial = condition[qid]
        if not trial.get("answer_changed"):
            continue
        changed_total += 1
        first = str(baseline[qid].get("subject_answer", "")).strip()
        if canonicalize:
            probs = canonical_probs(baseline[qid].get("probs"))
            alternatives = [letter for letter in LETTERS if letter != first and letter in probs]
            if not alternatives:
                continue
            runner = max(alternatives, key=lambda letter: probs[letter])
        else:
            raw = baseline[qid].get("probs")
            if not isinstance(raw, dict):
                continue
            # This deliberately matches the paper code, including overwrite on
            # whitespace-equivalent tokens and no A--D validation. Richer API
            # captures are first truncated to the four most probable raw tokens.
            probs = {
                str(token).strip(): float(value)
                for token, value in sorted(
                    raw.items(), key=lambda item: item[1], reverse=True
                )[:4]
                if str(token).strip() != "T" and isinstance(value, (int, float))
            }
            if len(probs) < 2:
                continue
            runner = sorted(probs, key=probs.get, reverse=True)[1]
        covered += 1
        hits += int(str(trial.get("new_answer", "")).strip() == runner)
    test = binomtest(hits, covered, 1 / 3, alternative="greater") if covered else None
    z_stat, z_p = proportions_ztest(hits, covered, value=1 / 3) if covered else (None, None)
    ci_low, ci_high = proportion_confint(hits, covered, alpha=0.05, method="normal") if covered else (None, None)
    return {
        "changed_total": changed_total,
        "covered": covered,
        "hits": hits,
        "rate": hits / covered if covered else None,
        "ci95_normal": [float(ci_low), float(ci_high)] if covered else None,
        "paper_two_sided_z_stat": float(z_stat) if covered else None,
        "paper_two_sided_z_p": float(z_p) if covered else None,
        "exact_binomial_one_sided_p": float(test.pvalue) if test else None,
    }


def main() -> None:
    args = parse_args()
    baseline = load_results([args.baseline])
    game = load_results(args.game)
    neutral = load_results(args.neutral)
    excluded_ids: set[str] = set()
    if args.exclude_ids:
        with args.exclude_ids.open("r", encoding="utf-8") as handle:
            excluded_ids = set(json.load(handle).get("excluded_union", []))
        baseline = {qid: value for qid, value in baseline.items() if qid not in excluded_ids}
        game = {qid: value for qid, value in game.items() if qid not in excluded_ids}
        neutral = {qid: value for qid, value in neutral.items() if qid not in excluded_ids}
    expected = 500 - len(excluded_ids)
    if not (len(baseline) == len(game) == len(neutral) == expected):
        raise ValueError(
            f"Expected {expected} trials per condition; got "
            f"{len(baseline)}, {len(game)}, {len(neutral)}"
        )

    output = {
        "coverage": {
            "n_attempted": 500,
            "n_analyzed": expected,
            "format_exclusions": len(excluded_ids),
        },
        "definitions": {
            "legacy_entropy": "-sum(p*log2(p)) over every raw top-four token, without renormalization",
            "complete_A_D_raw_entropy": "same calculation restricted to trials with all four canonical A-D tokens",
            "complete_A_D_renormalized_entropy": "Shannon entropy after renormalizing the four A-D probabilities to sum to one",
            "observed_A_D_renormalized_zero_fill": "entropy after aggregating observed A-D variants, assigning zero to any censored letter, and renormalizing observed answer mass",
            "rank_units": "option movements are natural-log changes in aggregated A-D probability; first-minus-runner and centered contrasts are changes in relative effective answer logits (logsumexp over each letter's token variants) because the softmax normalizer cancels",
        },
        "runner_up": {
            "paper_legacy": {
                "incorrect_feedback": runner_up_test(baseline, game, canonicalize=False),
                "neutral": runner_up_test(baseline, neutral, canonicalize=False),
            },
            "canonical_A_D_sensitivity": {
                "incorrect_feedback": runner_up_test(baseline, game, canonicalize=True),
                "neutral": runner_up_test(baseline, neutral, canonicalize=True),
            },
        },
        "entropy_contrasts": {
            "incorrect_feedback": entropy_contrast(baseline, game),
            "neutral": entropy_contrast(baseline, neutral),
            "incorrect_minus_neutral": entropy_contrast(neutral, game),
        },
        "rank_movements": {
            "incorrect_minus_baseline": rank_contrast(baseline, game, baseline),
            "neutral_minus_baseline": rank_contrast(baseline, neutral, baseline),
            "incorrect_minus_neutral": rank_contrast(baseline, game, neutral),
        },
        "normalized_A_D_probability_levels": {
            "baseline": rank_probability_levels(baseline, baseline),
            "incorrect_feedback": rank_probability_levels(baseline, game),
            "neutral": rank_probability_levels(baseline, neutral),
        },
        "normalized_A_D_probability_movements": {
            "incorrect_minus_baseline": probability_contrast(baseline, game, baseline),
            "neutral_minus_baseline": probability_contrast(baseline, neutral, baseline),
            "incorrect_minus_neutral": probability_contrast(baseline, game, neutral),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
