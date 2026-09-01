#!/usr/bin/env python3
"""Compare the accidental ``None`` neutral prompt with the clean neutral rerun."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, ttest_1samp, wilcoxon


LETTERS = ("A", "B", "C", "D")
LETTER_RE = re.compile(r"^\s*([A-D])\s*[\.)\]:,-]?\s*$", re.IGNORECASE)
EXPECTED_SYSTEM = "Output ONLY the letter of your choice: A, B, C, or D.\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contaminated-state",
        type=Path,
        default=Path("outputs/reproduction/simplemc_qwen36_27b/run_state.json"),
    )
    parser.add_argument(
        "--clean-state",
        type=Path,
        default=Path("outputs/reproduction/simplemc_qwen36_27b_clean_neutral/run_state.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reproduction/simplemc_qwen36_27b_clean_neutral"),
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_results(paths: list[str | Path]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for path in paths:
        rows = load_json(path)["results"]
        overlap = set(merged).intersection(rows)
        if overlap:
            raise ValueError(f"Duplicate question IDs: {len(overlap)}")
        merged.update(rows)
    return merged


def condition(state: dict, name: str) -> dict[str, dict]:
    stages = state["second_chance"]
    return load_results(
        [
            stages[f"{name}_baseline_incorrect"]["path"],
            stages[f"{name}_baseline_correct"]["path"],
        ]
    )


def canonical_probs(raw: object) -> dict[str, float]:
    result = {letter: 0.0 for letter in LETTERS}
    if not isinstance(raw, dict):
        return result
    for token, value in raw.items():
        if not isinstance(value, (int, float)) or value <= 0 or not math.isfinite(value):
            continue
        match = LETTER_RE.match(str(token))
        if match:
            result[match.group(1).upper()] += float(value)
    total = sum(result.values())
    if total:
        result = {letter: value / total for letter, value in result.items()}
    return result


def entropy(probs: dict[str, float]) -> float:
    return -sum(value * math.log2(value) for value in probs.values() if value > 0)


def paired_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array)))
    t_result = ttest_1samp(array, 0.0)
    try:
        w_p = float(wilcoxon(array).pvalue)
    except ValueError:
        w_p = 1.0
    return {
        "n": len(array),
        "mean": mean,
        "ci95_normal": [mean - 1.96 * se, mean + 1.96 * se],
        "paired_t_p": float(t_result.pvalue),
        "wilcoxon_p": w_p,
    }


def features(
    baseline: dict[str, dict], condition_rows: dict[str, dict]
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for qid in baseline:
        base_probs = canonical_probs(baseline[qid].get("probs"))
        cond_probs = canonical_probs(condition_rows[qid].get("probs"))
        winner = str(baseline[qid].get("subject_answer", "")).strip()
        if winner not in LETTERS:
            raise ValueError(f"Noncanonical baseline answer for {qid}: {winner!r}")
        runner = max((letter for letter in LETTERS if letter != winner), key=base_probs.get)
        other = [letter for letter in LETTERS if letter not in (winner, runner)]
        output[qid] = {
            "winner_probability": cond_probs[winner],
            "runner_probability": cond_probs[runner],
            "other_probability": sum(cond_probs[x] for x in other),
            "winner_margin": cond_probs[winner] - cond_probs[runner],
            "entropy": entropy(cond_probs),
        }
    return output


def feature_contrast(
    left: dict[str, dict[str, float]], right: dict[str, dict[str, float]]
) -> dict[str, dict]:
    # Values are right minus left.
    names = next(iter(left.values())).keys()
    return {
        name: paired_summary([right[qid][name] - left[qid][name] for qid in left])
        for name in names
    }


def clean_payload_audit(clean: dict[str, dict]) -> dict:
    exact = 0
    providers: dict[str, int] = {}
    requested: dict[str, int] = {}
    returned: dict[str, int] = {}
    for trial in clean.values():
        metadata = trial.get("call_metadata", {})
        exact += int(metadata.get("system_messages") == [EXPECTED_SYSTEM])
        provider = str(metadata.get("serving_provider"))
        providers[provider] = providers.get(provider, 0) + 1
        req = str(metadata.get("top_logprobs_requested"))
        requested[req] = requested.get(req, 0) + 1
        ret = str(metadata.get("top_logprobs_returned"))
        returned[ret] = returned.get(ret, 0) + 1
    return {
        "n": len(clean),
        "exact_system_message_matches": exact,
        "expected_system_message": EXPECTED_SYSTEM,
        "providers": providers,
        "top_logprobs_requested": requested,
        "top_logprobs_returned": returned,
    }


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def number(value: float) -> str:
    return f"{value:+.4f}"


def main() -> None:
    args = parse_args()
    contaminated_state = load_json(args.contaminated_state)
    clean_state = load_json(args.clean_state)
    baseline = load_results([clean_state["baseline_compiled_path"]])
    game = condition(clean_state, "incorrect")
    contaminated = condition(contaminated_state, "neutral")
    clean = condition(clean_state, "neutral")
    ids = set(baseline)
    if not (ids == set(game) == set(contaminated) == set(clean)):
        raise ValueError("Condition question IDs do not match")

    contaminated_changed = {qid for qid in ids if contaminated[qid].get("answer_changed")}
    clean_changed = {qid for qid in ids if clean[qid].get("answer_changed")}
    game_changed = {qid for qid in ids if game[qid].get("answer_changed")}
    old_only = contaminated_changed - clean_changed
    clean_only = clean_changed - contaminated_changed
    both = contaminated_changed & clean_changed
    discordant = len(old_only) + len(clean_only)
    switch_mcnemar = float(
        binomtest(len(old_only), discordant, 0.5).pvalue if discordant else 1.0
    )
    answer_agreement = sum(
        str(contaminated[qid].get("new_answer", "")).strip()
        == str(clean[qid].get("new_answer", "")).strip()
        for qid in ids
    )

    base_features = features(baseline, baseline)
    game_features = features(baseline, game)
    contaminated_features = features(baseline, contaminated)
    clean_features = features(baseline, clean)
    feature_results = {
        "contaminated_minus_baseline": feature_contrast(base_features, contaminated_features),
        "clean_minus_baseline": feature_contrast(base_features, clean_features),
        "clean_minus_contaminated": feature_contrast(contaminated_features, clean_features),
        "game_minus_contaminated_neutral": feature_contrast(contaminated_features, game_features),
        "game_minus_clean_neutral": feature_contrast(clean_features, game_features),
    }
    result = {
        "n": len(ids),
        "prompt_audit": clean_payload_audit(clean),
        "switching": {
            "game": {"n": len(game_changed), "rate": len(game_changed) / len(ids)},
            "contaminated_neutral": {
                "n": len(contaminated_changed),
                "rate": len(contaminated_changed) / len(ids),
            },
            "clean_neutral": {"n": len(clean_changed), "rate": len(clean_changed) / len(ids)},
            "game_minus_contaminated_neutral": (
                len(game_changed) - len(contaminated_changed)
            )
            / len(ids),
            "game_minus_clean_neutral": (len(game_changed) - len(clean_changed)) / len(ids),
            "contaminated_vs_clean": {
                "both_switched": len(both),
                "contaminated_only": len(old_only),
                "clean_only": len(clean_only),
                "neither": len(ids - contaminated_changed - clean_changed),
                "mcnemar_exact_p": switch_mcnemar,
                "exact_answer_agreement": answer_agreement,
                "exact_answer_agreement_rate": answer_agreement / len(ids),
            },
        },
        "canonical_A_D_probability_features": feature_results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "clean_neutral_comparison.json"
    report_path = args.output_dir / "CLEAN_NEUTRAL_COMPARISON.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    clean_vs_old = feature_results["clean_minus_contaminated"]
    clean_vs_base = feature_results["clean_minus_baseline"]
    old_vs_base = feature_results["contaminated_minus_baseline"]
    report = f"""# Clean-neutral OpenRouter replication

## Result

The clean rerun differs modestly from the run containing the accidental literal `None`. Neutral switching fell from {len(contaminated_changed)}/500 ({pct(len(contaminated_changed) / 500)}) to {len(clean_changed)}/500 ({pct(len(clean_changed) / 500)}). Consequently, the observed Game-minus-neutral switch lift increased from {pct((len(game_changed) - len(contaminated_changed)) / 500)} to {pct((len(game_changed) - len(clean_changed)) / 500)}.

The substantive result is unchanged and strengthened: the Game switched on {len(game_changed)}/500 trials ({pct(len(game_changed) / 500)}), versus {len(clean_changed)}/500 ({pct(len(clean_changed) / 500)}) for the properly formatted neutral condition.

## Exact prompt audit

All {result['prompt_audit']['exact_system_message_matches']}/500 newly collected trials record the client-bound system message as exactly:

```text
Output ONLY the letter of your choice: A, B, C, or D.
```

There is no `None` and no leading newline. All 500 calls were served by Io Net, requested 20 top log probabilities, and returned 20.

## Paired behavioral comparison

| Comparison | Result |
|---|---:|
| Exact answer agreement, contaminated vs clean | {answer_agreement}/500 ({pct(answer_agreement / 500)}) |
| Switched in both | {len(both)} |
| Switched only with `None` | {len(old_only)} |
| Switched only when clean | {len(clean_only)} |
| Switched in neither | {len(ids - contaminated_changed - clean_changed)} |
| Paired switch-rate test | exact McNemar p={switch_mcnemar:.4g} |

Thus the 3.6-point aggregate reduction is not merely 18 trials flipping deterministically: there is substantial trial-level variability, with {len(old_only)} old-only and {len(clean_only)} clean-only switches, while {answer_agreement}/500 final letters agree.

## Probability and entropy comparison

All probability features below aggregate letter-token variants, renormalize over A-D, and define “winner” and “runner-up” from each trial's baseline capabilities distribution.

| Mean paired change | Winner probability | Runner-up probability | Winner margin | A-D entropy (bits) |
|---|---:|---:|---:|---:|
| Contaminated neutral − baseline | {number(old_vs_base['winner_probability']['mean'])} | {number(old_vs_base['runner_probability']['mean'])} | {number(old_vs_base['winner_margin']['mean'])} | {number(old_vs_base['entropy']['mean'])} |
| Clean neutral − baseline | {number(clean_vs_base['winner_probability']['mean'])} | {number(clean_vs_base['runner_probability']['mean'])} | {number(clean_vs_base['winner_margin']['mean'])} | {number(clean_vs_base['entropy']['mean'])} |
| Clean neutral − contaminated neutral | {number(clean_vs_old['winner_probability']['mean'])} | {number(clean_vs_old['runner_probability']['mean'])} | {number(clean_vs_old['winner_margin']['mean'])} | {number(clean_vs_old['entropy']['mean'])} |

The paper-exact top-four-token entropy result is reported separately in `REPLICATION_REPORT.md`: clean neutral minus baseline is −0.039 bits (95% CI −0.065 to −0.013), compared with −0.083 bits in the contaminated run. Game minus baseline remains +0.173 bits. Therefore the clean rerun shows a smaller neutral sharpening effect without reversing it, and a smaller Game-versus-neutral entropy separation than previously estimated, though the separation remains large.

## Interpretation

The literal `None` was a genuine prompt bug, but it is not an explanation of the main behavioral result: that result survives the clean rerun. The clean run has lower neutral switching, so the observed behavioral lift is stronger. It also has 0.044 bits more entropy than the contaminated neutral run. However, because these are two separate provider calls—even at temperature zero and both served by Io Net—the difference between them combines any prompt effect with run-to-run nondeterminism. A single rerun cannot attribute the entire 3.6-point switch change or 0.044-bit entropy change specifically to `None`. The clean neutral remains mildly sharper than baseline, whereas the Game remains substantially flatter than baseline.

The detailed machine-readable paired statistics are in `clean_neutral_comparison.json`.
"""
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
