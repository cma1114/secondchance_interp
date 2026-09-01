#!/usr/bin/env python3
"""Cross-model trial-level decomposition of Second Chance excess switches.

The analysis asks whether Game-only switches are predominantly reversals from
the capabilities-test winner to its runner-up, and whether those reversals
require appreciable growth in A-D entropy.  Ranks are always defined from the
capabilities-test distribution for that same question.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint


LETTERS = ("A", "B", "C", "D")
LETTER_RE = re.compile(r"^\s*([A-D])\s*[\.)\]:,-]?\s*$", re.IGNORECASE)
DEFAULT_OUTPUT = Path("outputs/reproduction/selective_reranking_cross_model")


RUNS = (
    {
        "id": "gpt41_triviamc",
        "label": "GPT-4.1\nTriviaMC",
        "model": "GPT-4.1",
        "dataset": "TriviaMC",
        "baseline": ("outputs/reproduction/triviamc_gpt_4_1/baseline_results.json",),
        "game": ("outputs/reproduction/triviamc_gpt_4_1/game_results.json",),
        "neutral": ("outputs/reproduction/triviamc_gpt_4_1/neutral_results.json",),
        "expected": (499, 113, 33),
    },
    {
        "id": "qwen36_27b_simplemc",
        "label": "Qwen3.6-27B\nSimpleMC",
        "model": "Qwen3.6-27B",
        "dataset": "SimpleMC",
        "baseline": ("compiled_results_simplemc_qwen36_27b/qwen3.6-27b_phase1_compiled.json",),
        "game": (
            "secondchance_game_logs/qwen3.6-27b_SimpleMC_redacted_temp0.0_1785547932_game_data.json",
            "secondchance_game_logs/qwen3.6-27b_SimpleMC_redacted_cor_temp0.0_1785548142_game_data.json",
        ),
        "neutral": (
            "secondchance_game_logs/qwen3.6-27b_SimpleMC_neut_redacted_temp0.0_1785634992_game_data.json",
            "secondchance_game_logs/qwen3.6-27b_SimpleMC_neut_redacted_cor_temp0.0_1785635106_game_data.json",
        ),
        "expected": (500, 284, 157),
    },
    {
        "id": "qwen3_235b_simplemc",
        "label": "Qwen3-235B\nSimpleMC",
        "model": "Qwen3-235B",
        "dataset": "SimpleMC",
        "baseline": ("compiled_results_smc/qwen3-235b-a22b-2507_phase1_compiled.json",),
        "game": (
            "secondchance_game_logs/qwen3-235b-a22b-2507_SimpleMC_redacted_temp0.0_1785521299_game_data.json",
            "secondchance_game_logs/qwen3-235b-a22b-2507_SimpleMC_redacted_cor_temp0.0_1785522010_game_data.json",
        ),
        "neutral": (
            "secondchance_game_logs/qwen3-235b-a22b-2507_SimpleMC_neut_redacted_temp0.0_1785523104_game_data.json",
            "secondchance_game_logs/qwen3-235b-a22b-2507_SimpleMC_neut_redacted_cor_temp0.0_1785523823_game_data.json",
        ),
        "expected": (500, 154, 96),
    },
    {
        "id": "qwen3_235b_popmc",
        "label": "Qwen3-235B\nPopMC",
        "model": "Qwen3-235B",
        "dataset": "PopMC",
        "baseline": ("compiled_results_popmc/qwen3-235b-a22b-2507_phase1_compiled.json",),
        "game": (
            "secondchance_game_logs/qwen3-235b-a22b-2507_PopMC_redacted_temp0.0_1785536940_game_data.json",
            "secondchance_game_logs/qwen3-235b-a22b-2507_PopMC_redacted_cor_temp0.0_1785537492_game_data.json",
        ),
        "neutral": (
            "secondchance_game_logs/qwen3-235b-a22b-2507_PopMC_neut_redacted_temp0.0_1785538650_game_data.json",
            "secondchance_game_logs/qwen3-235b-a22b-2507_PopMC_neut_redacted_cor_temp0.0_1785539204_game_data.json",
        ),
        "expected": (500, 120, 74),
    },
    {
        "id": "qwen35_397b_simplemc",
        "label": "Qwen3.5-397B\nSimpleMC",
        "model": "Qwen3.5-397B",
        "dataset": "SimpleMC",
        "baseline": ("outputs/reproduction/simplemc_qwen35_397b_a17b/baseline_results.json",),
        "game": ("outputs/reproduction/simplemc_qwen35_397b_a17b/game_results.json",),
        "neutral": ("outputs/reproduction/simplemc_qwen35_397b_a17b/neutral_results.json",),
        "expected": (500, 104, 74),
    },
    {
        "id": "qwen35_397b_triviamc",
        "label": "Qwen3.5-397B\nTriviaMC",
        "model": "Qwen3.5-397B",
        "dataset": "TriviaMC",
        "baseline": ("outputs/reproduction/triviamc_qwen35_397b_a17b/baseline_results.json",),
        "game": ("outputs/reproduction/triviamc_qwen35_397b_a17b/game_results.json",),
        "neutral": ("outputs/reproduction/triviamc_qwen35_397b_a17b/neutral_results.json",),
        "expected": (500, 61, 45),
    },
    {
        "id": "qwen35_397b_popmc",
        "label": "Qwen3.5-397B\nPopMC",
        "model": "Qwen3.5-397B",
        "dataset": "PopMC",
        "baseline": ("outputs/reproduction/popmc_qwen35_397b_a17b/baseline_results.json",),
        "game": ("outputs/reproduction/popmc_qwen35_397b_a17b/game_results.json",),
        "neutral": ("outputs/reproduction/popmc_qwen35_397b_a17b/neutral_results.json",),
        "expected": (500, 72, 42),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_results(paths: tuple[str, ...]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        results = payload.get("results", payload)
        overlap = set(merged).intersection(results)
        if overlap:
            raise ValueError(f"Duplicate question IDs in {paths}: {len(overlap)}")
        merged.update(results)
    return merged


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


def answer(result: dict, baseline: bool = False) -> str:
    key = "subject_answer" if baseline else "new_answer"
    value = str(result.get(key, "")).strip().upper()
    return value if value in LETTERS else ""


def observed_entropy(probs: dict[str, float]) -> float:
    values = np.array([probs.get(letter, 0.0) for letter in LETTERS], dtype=float)
    total = float(values.sum())
    if total <= 0:
        return math.nan
    values /= total
    positive = values[values > 0]
    return float(-np.sum(positive * np.log2(positive)))


def complete_entropy(probs: dict[str, float]) -> float:
    if set(probs) != set(LETTERS):
        return math.nan
    return observed_entropy(probs)


def log_margin(probs: dict[str, float], winner: str, runner: str) -> float:
    if winner not in probs or runner not in probs:
        return math.nan
    return math.log(probs[winner]) - math.log(probs[runner])


def mean_ci(values: pd.Series) -> tuple[float, float, float, int]:
    clean = values.dropna().astype(float).to_numpy()
    if not len(clean):
        return math.nan, math.nan, math.nan, 0
    mean = float(np.mean(clean))
    if len(clean) == 1:
        return mean, math.nan, math.nan, 1
    se = float(np.std(clean, ddof=1) / math.sqrt(len(clean)))
    return mean, mean - 1.96 * se, mean + 1.96 * se, len(clean)


def prop_ci(k: int, n: int) -> tuple[float, float, float]:
    if not n:
        return math.nan, math.nan, math.nan
    low, high = proportion_confint(k, n, alpha=0.05, method="wilson")
    return k / n, float(low), float(high)


def build_trials(config: dict) -> pd.DataFrame:
    baseline = load_results(config["baseline"])
    game = load_results(config["game"])
    neutral = load_results(config["neutral"])
    common = sorted(set(baseline).intersection(game, neutral))
    rows: list[dict] = []
    for qid in common:
        b, g, n = baseline[qid], game[qid], neutral[qid]
        first = answer(b, baseline=True)
        game_answer = answer(g)
        neutral_answer = answer(n)
        if not first:
            continue
        bp = canonical_probs(b.get("probs"))
        gp = canonical_probs(g.get("probs"))
        np_ = canonical_probs(n.get("probs"))
        alternatives = sorted(
            [letter for letter in bp if letter != first],
            key=lambda letter: (-bp[letter], letter),
        )
        runner = alternatives[0] if alternatives else ""
        # The generated capabilities answer defines rank 1, even if aggregation
        # over returned token variants makes another letter's stored
        # probability fractionally larger.  Rank the three alternatives after
        # that frozen first choice.
        complete_ranks = [first] + alternatives if set(bp) == set(LETTERS) else []
        # Preserve the paper's behavioral convention: malformed/non-letter
        # outputs are recorded as unchanged rather than dropped.  Their
        # probability records still contribute to entropy contrasts.
        game_changed = bool(g.get("answer_changed", False))
        neutral_changed = bool(n.get("answer_changed", False))
        if game_changed and not neutral_changed:
            stratum = "game_only"
        elif neutral_changed and not game_changed:
            stratum = "neutral_only"
        elif game_changed and neutral_changed:
            stratum = "both"
        else:
            stratum = "neither"

        def destination(new_answer: str, changed: bool) -> str:
            if not changed:
                return "unchanged"
            if new_answer not in LETTERS:
                return "invalid_changed_answer"
            if not runner:
                return "runner_censored"
            if new_answer == runner:
                return "runner_up"
            if complete_ranks:
                return f"rank_{complete_ranks.index(new_answer) + 1}"
            return "lower_rank"

        hb = observed_entropy(bp)
        hg = observed_entropy(gp)
        hn = observed_entropy(np_)
        hbc = complete_entropy(bp)
        hgc = complete_entropy(gp)
        hnc = complete_entropy(np_)
        mb = log_margin(bp, first, runner) if runner else math.nan
        mg = log_margin(gp, first, runner) if runner else math.nan
        mn = log_margin(np_, first, runner) if runner else math.nan
        rows.append(
            {
                "run_id": config["id"],
                "model": config["model"],
                "dataset": config["dataset"],
                "qid": qid,
                "first": first,
                "runner": runner,
                "runner_observed": bool(runner),
                "baseline_complete_ad": set(bp) == set(LETTERS),
                "game_complete_ad": set(gp) == set(LETTERS),
                "neutral_complete_ad": set(np_) == set(LETTERS),
                "game_answer": game_answer,
                "neutral_answer": neutral_answer,
                "game_changed": game_changed,
                "neutral_changed": neutral_changed,
                "stratum": stratum,
                "game_destination": destination(game_answer, game_changed),
                "neutral_destination": destination(neutral_answer, neutral_changed),
                "h_baseline_observed": hb,
                "h_game_observed": hg,
                "h_neutral_observed": hn,
                "dh_game_baseline_observed": hg - hb,
                "dh_neutral_baseline_observed": hn - hb,
                "dh_game_neutral_observed": hg - hn,
                "h_baseline_complete": hbc,
                "h_game_complete": hgc,
                "h_neutral_complete": hnc,
                "dh_game_baseline_complete": hgc - hbc,
                "dh_neutral_baseline_complete": hnc - hbc,
                "dh_game_neutral_complete": hgc - hnc,
                "margin_baseline": mb,
                "margin_game": mg,
                "margin_neutral": mn,
                "dmargin_game_baseline": mg - mb,
                "dmargin_neutral_baseline": mn - mb,
                "dmargin_game_neutral": mg - mn,
                "game_runner_leads": bool(math.isfinite(mg) and mg < 0),
                "neutral_runner_leads": bool(math.isfinite(mn) and mn < 0),
            }
        )
    frame = pd.DataFrame(rows)
    expected_n, expected_game, expected_neutral = config["expected"]
    observed = (len(frame), int(frame.game_changed.sum()), int(frame.neutral_changed.sum()))
    if observed != (expected_n, expected_game, expected_neutral):
        raise ValueError(f"{config['id']} validation failed: expected {config['expected']}, got {observed}")
    return frame


def summarize_run(config: dict, trials: pd.DataFrame) -> tuple[dict, list[dict]]:
    counts = Counter(trials.stratum)
    game_only = trials[trials.stratum == "game_only"]
    neutral_only = trials[trials.stratum == "neutral_only"]
    game_runner = game_only[game_only.game_destination == "runner_up"]
    neutral_runner = neutral_only[neutral_only.neutral_destination == "runner_up"]

    game_runner_k = len(game_runner)
    game_runner_n = int(game_only.runner_observed.sum())
    neutral_runner_k = len(neutral_runner)
    neutral_runner_n = int(neutral_only.runner_observed.sum())
    game_prop, game_low, game_high = prop_ci(game_runner_k, game_runner_n)
    neutral_prop, neutral_low, neutral_high = prop_ci(neutral_runner_k, neutral_runner_n)

    entropy_rows: list[dict] = []
    for stratum_name, subset in (
        ("game_only", game_only),
        ("neutral_only", neutral_only),
        ("both", trials[trials.stratum == "both"]),
        ("neither", trials[trials.stratum == "neither"]),
    ):
        for metric in (
            "dh_game_baseline_observed",
            "dh_neutral_baseline_observed",
            "dh_game_neutral_observed",
            "dh_game_baseline_complete",
            "dh_neutral_baseline_complete",
            "dh_game_neutral_complete",
        ):
            mean, low, high, n = mean_ci(subset[metric])
            entropy_rows.append(
                {
                    "run_id": config["id"],
                    "model": config["model"],
                    "dataset": config["dataset"],
                    "stratum": stratum_name,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    all_entropy, all_entropy_low, all_entropy_high, all_entropy_n = mean_ci(
        trials.dh_game_baseline_observed
    )
    complete_entropy_mean, complete_entropy_low, complete_entropy_high, complete_entropy_n = mean_ci(
        trials.dh_game_baseline_complete
    )
    go_entropy, go_entropy_low, go_entropy_high, go_entropy_n = mean_ci(
        game_only.dh_game_baseline_observed
    )
    go_direct, go_direct_low, go_direct_high, go_direct_n = mean_ci(
        game_only.dh_game_neutral_observed
    )
    go_margin, go_margin_low, go_margin_high, go_margin_n = mean_ci(
        game_only.dmargin_game_neutral
    )

    runner_entropy = game_runner.dh_game_baseline_observed.dropna()
    runner_direct_entropy = game_runner.dh_game_neutral_observed.dropna()
    runner_complete_entropy = game_runner.dh_game_baseline_complete.dropna()
    runner_margin = game_runner.dmargin_game_neutral.dropna()
    runner_lead_covered = game_runner.margin_game.notna()

    game_destination_counts = Counter(game_only.game_destination)
    neutral_destination_counts = Counter(neutral_only.neutral_destination)
    net_runner = game_destination_counts["runner_up"] - neutral_destination_counts["runner_up"]
    net_lower = (
        sum(v for k, v in game_destination_counts.items() if k not in {"runner_up", "runner_censored"})
        - sum(v for k, v in neutral_destination_counts.items() if k not in {"runner_up", "runner_censored"})
    )
    net_unknown = game_destination_counts["runner_censored"] - neutral_destination_counts["runner_censored"]
    neutral_runner_entropy = neutral_runner.dh_neutral_baseline_observed.dropna()
    game_low_entropy_runner = int((runner_entropy <= 0.05).sum())
    neutral_low_entropy_runner = int((neutral_runner_entropy <= 0.05).sum())

    normalized_lift = (trials.game_changed.mean() - trials.neutral_changed.mean()) / (
        1 - trials.neutral_changed.mean()
    )
    return (
        {
            "run_id": config["id"],
            "model": config["model"],
            "dataset": config["dataset"],
            "n": len(trials),
            "game_change_rate": float(trials.game_changed.mean()),
            "neutral_change_rate": float(trials.neutral_changed.mean()),
            "normalized_lift": float(normalized_lift),
            "game_only_n": counts["game_only"],
            "neutral_only_n": counts["neutral_only"],
            "both_n": counts["both"],
            "neither_n": counts["neither"],
            "raw_excess_switches": counts["game_only"] - counts["neutral_only"],
            "baseline_runner_coverage_all": float(trials.runner_observed.mean()),
            "baseline_complete_ad_all": float(trials.baseline_complete_ad.mean()),
            "game_only_runner_coverage_n": game_runner_n,
            "game_only_runner_hits": game_runner_k,
            "game_only_runner_rate": game_prop,
            "game_only_runner_ci_low": game_low,
            "game_only_runner_ci_high": game_high,
            "neutral_only_runner_coverage_n": neutral_runner_n,
            "neutral_only_runner_hits": neutral_runner_k,
            "neutral_only_runner_rate": neutral_prop,
            "neutral_only_runner_ci_low": neutral_low,
            "neutral_only_runner_ci_high": neutral_high,
            "net_runner_excess": net_runner,
            "net_lower_excess": net_lower,
            "net_unknown_excess": net_unknown,
            "all_dh_game_baseline": all_entropy,
            "all_dh_game_baseline_ci_low": all_entropy_low,
            "all_dh_game_baseline_ci_high": all_entropy_high,
            "all_dh_game_baseline_n": all_entropy_n,
            "complete_dh_game_baseline": complete_entropy_mean,
            "complete_dh_game_baseline_ci_low": complete_entropy_low,
            "complete_dh_game_baseline_ci_high": complete_entropy_high,
            "complete_dh_game_baseline_n": complete_entropy_n,
            "game_only_dh_game_baseline": go_entropy,
            "game_only_dh_game_baseline_ci_low": go_entropy_low,
            "game_only_dh_game_baseline_ci_high": go_entropy_high,
            "game_only_dh_game_baseline_n": go_entropy_n,
            "game_only_median_dh_game_baseline": float(game_only.dh_game_baseline_observed.median()),
            "game_only_dh_game_neutral": go_direct,
            "game_only_dh_game_neutral_ci_low": go_direct_low,
            "game_only_dh_game_neutral_ci_high": go_direct_high,
            "game_only_dh_game_neutral_n": go_direct_n,
            "game_only_dmargin_game_neutral": go_margin,
            "game_only_dmargin_game_neutral_ci_low": go_margin_low,
            "game_only_dmargin_game_neutral_ci_high": go_margin_high,
            "game_only_dmargin_game_neutral_n": go_margin_n,
            "runner_switch_dh_le_zero": float((runner_entropy <= 0).mean()) if len(runner_entropy) else math.nan,
            "runner_switch_dh_le_005": float((runner_entropy <= 0.05).mean()) if len(runner_entropy) else math.nan,
            "runner_switch_median_dh_game_baseline": float(runner_entropy.median()) if len(runner_entropy) else math.nan,
            "game_low_entropy_runner_n": game_low_entropy_runner,
            "neutral_low_entropy_runner_n": neutral_low_entropy_runner,
            "net_low_entropy_runner_excess": game_low_entropy_runner - neutral_low_entropy_runner,
            "runner_switch_direct_dh_le_zero": float((runner_direct_entropy <= 0).mean()) if len(runner_direct_entropy) else math.nan,
            "runner_switch_direct_dh_le_005": float((runner_direct_entropy <= 0.05).mean()) if len(runner_direct_entropy) else math.nan,
            "runner_switch_complete_entropy_n": len(runner_complete_entropy),
            "runner_switch_complete_dh_le_zero": float((runner_complete_entropy <= 0).mean()) if len(runner_complete_entropy) else math.nan,
            "runner_switch_complete_dh_le_005": float((runner_complete_entropy <= 0.05).mean()) if len(runner_complete_entropy) else math.nan,
            "runner_switch_margin_covered_n": int(runner_lead_covered.sum()),
            "runner_switch_runner_leads_game": float(game_runner.loc[runner_lead_covered, "game_runner_leads"].mean()) if runner_lead_covered.any() else math.nan,
            "runner_switch_mean_dmargin_game_neutral": float(runner_margin.mean()) if len(runner_margin) else math.nan,
            "game_destination_counts": dict(game_destination_counts),
            "neutral_destination_counts": dict(neutral_destination_counts),
        },
        entropy_rows,
    )


def make_figure(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = ["#D55E00", "#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#AA4499", "#882255"]
    x = np.arange(len(summary))

    ax = axes[0]
    ax.scatter(summary.normalized_lift, summary.all_dh_game_baseline, s=80, c=colors)
    annotation_offsets = [(5, -9), (-5, 6), (5, 7), (5, -13), (5, -15), (5, 9), (5, 8)]
    for (_, row), offset in zip(summary.iterrows(), annotation_offsets):
        label = f"{row.model} {row.dataset}"
        ax.annotate(
            label,
            (row.normalized_lift, row.all_dh_game_baseline),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            ha="right" if offset[0] < 0 else "left",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Normalized Game–Neutral switch lift")
    ax.set_ylabel("Game–baseline A–D entropy (bits)")
    ax.set_title("A  Aggregate behavior", loc="left", fontweight="bold")

    ax = axes[1]
    runner = summary.game_only_runner_rate.to_numpy()
    lower = 1 - runner
    ax.bar(x, runner, color="#0072B2", label="Runner-up")
    ax.bar(x, lower, bottom=runner, color="#BDBDBD", label="Other covered option")
    ax.set_xticks(x, [config["label"] for config in RUNS], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of covered Game-only switches")
    ax.set_title("B  Destination of excess switches", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    le_zero = summary.runner_switch_dh_le_zero.to_numpy()
    le_005 = summary.runner_switch_dh_le_005.to_numpy()
    small_pos = le_005 - le_zero
    large = 1 - le_005
    ax.bar(x, le_zero, color="#009E73", label="ΔH ≤ 0")
    ax.bar(x, small_pos, bottom=le_zero, color="#F0E442", label="0 < ΔH ≤ .05")
    ax.bar(x, large, bottom=le_005, color="#D55E00", label="ΔH > .05")
    ax.set_xticks(x, [config["label"] for config in RUNS], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of runner-up Game-only switches")
    ax.set_title("C  Entropy accompanying top-two reversals", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return "—" if not math.isfinite(value) else f"{value:.{digits}f}"


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            f"| {r.model} | {r.dataset} | {r.game_only_n}/{r.neutral_only_n} | "
            f"{int(r.game_only_runner_hits)}/{int(r.game_only_runner_coverage_n)} = {r.game_only_runner_rate:.1%} | "
            f"{fmt(r.game_only_dh_game_baseline)} / {fmt(r.game_only_median_dh_game_baseline)} | "
            f"{fmt(r.runner_switch_median_dh_game_baseline)} | "
            f"{r.runner_switch_dh_le_zero:.1%} | {r.runner_switch_dh_le_005:.1%} |"
        )
    destination_rows = []
    for _, r in summary.iterrows():
        destination_rows.append(
            f"| {r.model} | {r.dataset} | {int(r.raw_excess_switches)} | "
            f"{int(r.net_runner_excess)} | {int(r.net_lower_excess)} | {int(r.net_unknown_excess)} | "
            f"{int(r.game_low_entropy_runner_n)}−{int(r.neutral_low_entropy_runner_n)} = {int(r.net_low_entropy_runner_excess)} | "
            f"{int(r.runner_switch_margin_covered_n)} | {r.runner_switch_runner_leads_game:.1%} | "
            f"{fmt(r.runner_switch_mean_dmargin_game_neutral)} |"
        )
    coverage_rows = []
    for _, r in summary.iterrows():
        coverage_rows.append(
            f"| {r.model} | {r.dataset} | {r.baseline_runner_coverage_all:.1%} | "
            f"{r.baseline_complete_ad_all:.1%} | {int(r.complete_dh_game_baseline_n)}/{int(r.n)} | "
            f"{fmt(r.all_dh_game_baseline)} | {fmt(r.complete_dh_game_baseline)} |"
        )

    gpt = summary.loc[summary.run_id == "gpt41_triviamc"].iloc[0]
    qwen397 = summary.loc[summary.run_id == "qwen35_397b_simplemc"].iloc[0]
    qwen397_trivia = summary.loc[summary.run_id == "qwen35_397b_triviamc"].iloc[0]
    qwen397_pop = summary.loc[summary.run_id == "qwen35_397b_popmc"].iloc[0]

    report = f"""# Selective re-ranking across low-entropy-lift Second Chance runs

Ranks are defined separately on every question from the capabilities-test A–D probabilities. A **Game-only switch** changes under incorrect feedback but not under the neutral redo prompt; a **neutral-only switch** does the reverse. These discordant trials generate the paired behavioral lift.

## Main result

| Model | Dataset | Game-only / neutral-only | Game-only switches to runner-up | Game-only ΔH mean / median | Runner-switch median ΔH | Runner switches with ΔH≤0 | Runner switches with ΔH≤.05 |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

ΔH in this table is Game minus baseline A–D entropy. The final two columns use the all-trial A–D sensitivity measure: captured letter-token variants are aggregated, censored letters receive zero, and observed A–D mass is renormalized. A .05-bit cutoff is descriptive rather than a formal equivalence bound; ΔH≤0 requires no cutoff.

## Destination and top-two margin

| Model | Dataset | Net excess switches | Net runner-up excess | Net lower-rank excess | Net unknown rank | Low-entropy runner switches, Game−neutral | Margin coverage | Runner leads in Game | Mean margin change |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(destination_rows)}

The low-entropy count uses ΔH≤.05 relative to each condition's baseline (Game−baseline for Game-only switches and Neutral−baseline for neutral-only switches). The margin is `log p(original winner) − log p(baseline runner-up)`, so a negative Game−neutral change means selective movement toward the runner-up. “Runner actually leads” checks whether the stored Game probabilities themselves put the runner-up above the original winner on trials where the generated Game answer is the runner-up.

## Coverage and strict complete-A–D sensitivity

| Model | Dataset | Runner identifiable, all trials | Complete baseline A–D | Complete Game/baseline entropy pairs | All-trial ΔH | Complete-pair ΔH |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(coverage_rows)}

The strict column requires all four A–D probabilities in both baseline and Game. It is highly selective for GPT-4.1 because even 20 returned tokens often contain multiple spellings of the same high-confidence answer. The all-trial sensitivity estimate is therefore retained as the primary descriptive entropy measure, with coverage stated explicitly.

## What the decomposition says

GPT-4.1 is the clearest case in this set of a low-entropy top-two reversal. Its all-trial entropy increase is only {gpt.all_dh_game_baseline:.3f} bits. More importantly, the Game-only mean of {gpt.game_only_dh_game_baseline:.3f} bits is highly skewed: the median is {gpt.game_only_median_dh_game_baseline:.3f} bits, and the median among identifiable runner-up switches is {gpt.runner_switch_median_dh_game_baseline:.3f} bits. {int(gpt.game_low_entropy_runner_n)} of {int(gpt.game_only_runner_hits)} identifiable Game-only runner switches have ΔH≤.05, compared with {int(gpt.neutral_low_entropy_runner_n)} neutral-only cases. Requiring complete A–D probability coverage gives the same qualitative result ({gpt.runner_switch_complete_dh_le_005:.1%}, {int(gpt.runner_switch_complete_entropy_n)} trials).

Qwen3.5-397B is the closest partial analogue on SimpleMC: {qwen397.game_only_runner_rate:.1%} of covered Game-only switches go to the runner-up and {qwen397.runner_switch_dh_le_005:.1%} of those have ΔH≤.05. On TriviaMC, the corresponding values are {qwen397_trivia.game_only_runner_rate:.1%} and {qwen397_trivia.runner_switch_dh_le_005:.1%}; on PopMC they are {qwen397_pop.game_only_runner_rate:.1%} and {qwen397_pop.runner_switch_dh_le_005:.1%}. The other Qwen runs more often combine switching with appreciable entropy growth or distribute more of the net excess over lower-ranked options.

Entropy in bits and normalized switch lift are not commensurate quantities, so their numerical ratio is not an explanatory test. The informative result is the trial-level mixture: GPT-4.1 combines many nearly entropy-preserving top-two reversals with a minority of large-entropy changes that raise the mean.

## Interpretation rule

Evidence for selective re-ranking is strongest when the net excess switches predominantly go to the baseline runner-up, the original-winner-versus-runner margin falls specifically in Game, and many such reversals occur with non-increasing or negligible entropy. Conversely, frequent movement to ranks 3–4 together with substantial entropy growth supports broad flattening or noise.

![Cross-model selective re-ranking summary]({(output_dir / 'selective_reranking_summary.png').resolve()})
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_trials: list[pd.DataFrame] = []
    summaries: list[dict] = []
    entropy_rows: list[dict] = []
    for config in RUNS:
        trials = build_trials(config)
        summary, run_entropy = summarize_run(config, trials)
        all_trials.append(trials)
        summaries.append(summary)
        entropy_rows.extend(run_entropy)

    trial_frame = pd.concat(all_trials, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    entropy_frame = pd.DataFrame(entropy_rows)
    trial_frame.to_csv(args.output_dir / "trial_level.csv", index=False)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    entropy_frame.to_csv(args.output_dir / "entropy_by_stratum.csv", index=False)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, allow_nan=True)
    make_figure(summary_frame, args.output_dir / "selective_reranking_summary")
    write_report(summary_frame, args.output_dir)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
