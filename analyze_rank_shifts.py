#!/usr/bin/env python3
"""Analyze rank-conditioned answer-logit movements in the Second Chance game.

The neutral prompt defines ranks 1--4 separately for every question.  For trials
where all four canonical answer letters are present in both the neutral and game
top-logprob records, the script computes (in natural-logit units):

    delta_1 = Delta[z_rank1 - mean(z_rank3, z_rank4)]
    delta_2 = Delta[z_rank2 - mean(z_rank3, z_rank4)]
    delta_12 = Delta[z_rank1 - z_rank2]

Whitespace/punctuation variants of A--D are canonicalized and their probability
mass is summed.  The complete-case restriction is deliberate: the APIs stored
only the top four *tokens*, so duplicate tokenizations can leave a letter
censored.  Coverage and missing-rank diagnostics are emitted alongside results.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


LETTERS = ("A", "B", "C", "D")
CONDITION_CONTRASTS = {
    "neutral_minus_baseline": ("neutral", "source"),
    "incorrect_minus_baseline": ("game", "source"),
    "incorrect_minus_neutral": ("game", "neutral"),
}
LIFT_SECOND_CHOICE_CELLS = (
    ("grok-3-latest", "GPQA"),
    ("grok-3-latest", "SimpleMC"),
    ("qwen3-235b-a22b-2507", "GPQA"),
    ("qwen3-235b-a22b-2507", "SimpleMC"),
    ("gpt-4.1-2025-04-14", "GPQA"),
    ("gpt-4.1-2025-04-14", "SimpleMC"),
    ("gpt-4o-2024-08-06", "GPQA"),
    ("gpt-4o-2024-08-06", "SimpleMC"),
    ("gpt-4o-mini", "GPQA"),
    ("gpt-4o-mini", "SimpleMC"),
    ("gemini-2.5-flash_nothink", "GPQA"),
    ("gemini-2.5-flash_nothink", "SimpleMC"),
    ("gemini-2.5-flash-lite_nothink", "GPQA"),
    ("gemini-2.0-flash-001", "SimpleMC"),
)
LIFT_SECOND_CHOICE_CELL_SET = set(LIFT_SECOND_CHOICE_CELLS)
PAPER_FULL_SUCCESS_CELLS = {
    ("gpt-4.1-2025-04-14", "GPQA"),
    ("gpt-4.1-2025-04-14", "SimpleMC"),
    ("gpt-4o-2024-08-06", "GPQA"),
    ("gpt-4o-2024-08-06", "SimpleMC"),
    ("gpt-4o-mini", "GPQA"),
    ("gpt-4o-mini", "SimpleMC"),
    ("gemini-2.5-flash-lite_nothink", "GPQA"),
}
FILE_RE = re.compile(
    r"^(?P<model>.+)_(?P<dataset>GPQA|SimpleMC)_(?P<condition>.+)_"
    r"temp(?P<temperature>[0-9.]+)_(?P<timestamp>[0-9]+)_game_data\.json$"
)
LETTER_RE = re.compile(r"^\s*([A-D])\s*[\.)\]:,-]?\s*$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/Users/christopherackerman/repos/self_awareness/metacog"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rank_shifts"))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    return parser.parse_args()


def file_info(path: Path) -> dict | None:
    match = FILE_RE.match(path.name)
    if not match:
        return None
    info = match.groupdict()
    condition = info["condition"]
    if "redacted" not in condition or "_pos_" in f"_{condition}_" or "shown" in condition:
        return None
    info["temperature"] = float(info["temperature"])
    info["timestamp"] = int(info["timestamp"])
    info["correctness_set"] = "baseline_correct" if "_cor" in condition else "baseline_incorrect"
    info["path"] = path
    return info


def match_key(path: Path) -> str | None:
    info = file_info(path)
    if not info:
        return None
    stem = re.sub(r"_[0-9]+_game_data\.json$", "", path.name)
    return stem.replace("_neut_", "_")


def choose_game_files(game_dir: Path) -> list[dict]:
    infos = [x for p in game_dir.glob("*_game_data.json") if (x := file_info(p))]
    by_model_dataset: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for info in infos:
        if "neut" not in info["condition"]:
            by_model_dataset[(info["model"], info["dataset"])].append(info)

    chosen: list[dict] = []
    for group in by_model_dataset.values():
        # This reproduces the paper analysis convention: when temperature-1
        # records exist, use them instead of the older temperature-0 records.
        available_temps = {x["temperature"] for x in group}
        preferred_temp = 1.0 if 1.0 in available_temps else max(available_temps)
        chosen.extend(x for x in group if x["temperature"] == preferred_temp)
    return sorted(chosen, key=lambda x: str(x["path"]))


def build_neutral_lookup(neutral_dir: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for path in neutral_dir.glob("*_game_data.json"):
        info = file_info(path)
        key = match_key(path)
        if not info or not key:
            continue
        # Retain the latest record if a duplicated logical key exists.
        if key not in lookup or info["timestamp"] > lookup[key]["timestamp"]:
            lookup[key] = info
    return lookup


def canonical_probs(raw: object) -> tuple[dict[str, float], int, float | None]:
    """Return aggregated A--D probabilities, raw token count, and top-k cutoff."""
    if not isinstance(raw, dict):
        return {}, 0, None
    result = {letter: 0.0 for letter in LETTERS}
    present: set[str] = set()
    positive_raw: list[float] = []
    for token, value in raw.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            continue
        value = float(value)
        positive_raw.append(value)
        match = LETTER_RE.match(str(token))
        if match:
            letter = match.group(1).upper()
            result[letter] += value
            present.add(letter)
    return ({k: result[k] for k in present}, len(raw), min(positive_raw) if positive_raw else None)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_capability_lookups(source_root: Path, game_infos: list[dict]) -> dict[tuple[str, str], dict]:
    """Select the baseline file whose recorded answers best match each game run."""
    original_answers: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    temperatures: dict[tuple[str, str], float] = {}
    for info in game_infos:
        key = (info["model"], info["dataset"])
        temperatures[key] = info["temperature"]
        for qid, result in load_json(info["path"]).get("results", {}).items():
            original_answers[key][qid] = str(result.get("original_answer", "")).strip()

    selected: dict[tuple[str, str], dict] = {}
    for key, answers in original_answers.items():
        model, dataset = key
        directory = source_root / ("completed_results_gpqa" if dataset == "GPQA" else "compiled_results_smc")
        candidates = sorted(directory.glob(f"{model}_phase1_*.json"))
        best = None
        for path in candidates:
            results = load_json(path).get("results", {})
            overlap = 0
            matches = 0
            for qid, original in answers.items():
                if qid not in results:
                    continue
                overlap += 1
                subject = str(results[qid].get("subject_answer", "")).strip()
                matches += int(bool(original) and original == subject)
            rate = matches / overlap if overlap else 0.0
            desired = "t1" if temperatures[key] == 1.0 else "t0"
            temp_match = int(f"_{desired}" in path.stem)
            candidate = {
                "path": path,
                "results": results,
                "overlap": overlap,
                "matches": matches,
                "match_rate": rate,
                "temp_match": temp_match,
                "score": (rate, temp_match, overlap, path.stat().st_mtime),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        if best:
            selected[key] = best
    return selected


def bootstrap_ci(values: np.ndarray, stat: str, draws: int, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    remaining = draws
    while remaining:
        chunk = min(500, remaining)
        sample = values[rng.integers(0, len(values), size=(chunk, len(values)))]
        estimates.append(np.mean(sample, axis=1) if stat == "mean" else np.median(sample, axis=1))
        remaining -= chunk
    combined = np.concatenate(estimates)
    return tuple(np.percentile(combined, [2.5, 97.5]).tolist())


def sign_flip_p(values: np.ndarray, draws: int, seed: int) -> float:
    if len(values) == 0:
        return math.nan
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = draws
    while remaining:
        chunk = min(500, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(chunk, len(values)))
        exceed += int(np.sum(np.abs(np.mean(signs * values, axis=1)) >= observed))
        remaining -= chunk
    return (exceed + 1) / (draws + 1)


def stable_seed(label: str) -> int:
    return zlib.crc32(label.encode("utf-8")) & 0xFFFFFFFF


def summarize_group(group: pd.DataFrame, bootstrap: int, permutations: int, group_label: tuple | str) -> dict:
    exact = group[group["exact_complete"]].copy()
    pair12 = group[group["pair12_complete"]].copy()
    result = {
        "n_paired": len(group),
        "n_source_complete": int(group["source_complete"].sum()),
        "n_neutral_complete": int(group["neutral_complete"].sum()),
        "n_exact_complete": len(exact),
        "exact_coverage": len(exact) / len(group) if len(group) else math.nan,
        "n_pair12_complete": len(pair12),
        "pair12_coverage": len(pair12) / len(group) if len(group) else math.nan,
        "source_tie_n": int(group["source_top_tie"].sum()),
        "game_missing_rank1_n": int(
            ((group["source_complete"]) & (~group["game_has_rank1"])).sum()
        ),
        "game_missing_rank2_n": int(
            ((group["source_complete"]) & (~group["game_has_rank2"])).sum()
        ),
        "game_change_rate": float(group["game_answer_changed"].mean()),
        "neutral_change_rate": float(group["neutral_answer_changed"].mean()),
    }
    for metric in ("delta_1", "delta_2", "delta_12"):
        values = exact[metric].to_numpy(dtype=float)
        labels = group_label if isinstance(group_label, tuple) else (group_label,)
        seed = stable_seed("|".join(map(str, labels)) + metric)
        mean = float(np.mean(values)) if len(values) else math.nan
        median = float(np.median(values)) if len(values) else math.nan
        mean_low, mean_high = bootstrap_ci(values, "mean", bootstrap, seed)
        med_low, med_high = bootstrap_ci(values, "median", bootstrap, seed + 1)
        result.update(
            {
                f"{metric}_mean": mean,
                f"{metric}_mean_ci_low": mean_low,
                f"{metric}_mean_ci_high": mean_high,
                f"{metric}_median": median,
                f"{metric}_median_ci_low": med_low,
                f"{metric}_median_ci_high": med_high,
                f"{metric}_signflip_p": sign_flip_p(values, permutations, seed + 2),
            }
        )

    pair_values = pair12["delta_12_pair"].to_numpy(dtype=float)
    labels = group_label if isinstance(group_label, tuple) else (group_label,)
    pair_seed = stable_seed("|".join(map(str, labels)) + "delta_12_pair")
    pair_low, pair_high = bootstrap_ci(pair_values, "mean", bootstrap, pair_seed)
    result.update(
        {
            "delta_12_pair_mean": float(np.mean(pair_values)) if len(pair_values) else math.nan,
            "delta_12_pair_mean_ci_low": pair_low,
            "delta_12_pair_mean_ci_high": pair_high,
            "delta_12_pair_signflip_p": sign_flip_p(pair_values, permutations, pair_seed + 1),
        }
    )

    first_down = result["delta_1_mean_ci_high"] < 0
    second_up = result["delta_2_mean_ci_low"] > 0
    if first_down and second_up:
        label = "first down + second up"
    elif first_down:
        label = "detectable first down only"
    elif second_up:
        label = "detectable second up only"
    else:
        label = "neither resolved"
    result["directional_pattern"] = label
    return result


def analyze_pairs(
    game_infos: list[dict], neutral_lookup: dict[str, dict], capability_lookups: dict[tuple[str, str], dict]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    manifest: list[dict] = []

    for game_info in game_infos:
        game_path = game_info["path"]
        key = match_key(game_path)
        neutral_info = neutral_lookup.get(key or "")
        capability_info = capability_lookups.get((game_info["model"], game_info["dataset"]))
        manifest_row = {
            "model": game_info["model"],
            "dataset": game_info["dataset"],
            "correctness_set": game_info["correctness_set"],
            "temperature": game_info["temperature"],
            "game_file": str(game_path),
            "neutral_file": str(neutral_info["path"]) if neutral_info else "",
            "capabilities_file": str(capability_info["path"]) if capability_info else "",
            "capabilities_answer_match_rate": capability_info["match_rate"] if capability_info else math.nan,
            "matched": bool(neutral_info),
        }
        manifest.append(manifest_row)
        if not neutral_info:
            continue

        game = load_json(game_path).get("results", {})
        neutral = load_json(neutral_info["path"]).get("results", {})
        for qid in sorted(set(game) & set(neutral)):
            game_result = game[qid]
            neutral_result = neutral[qid]
            game_probs, game_raw_n, game_cutoff = canonical_probs(game_result.get("probs"))
            neutral_probs, neutral_raw_n, neutral_cutoff = canonical_probs(neutral_result.get("probs"))
            capability_result = (capability_info or {}).get("results", {}).get(qid, {})
            baseline_raw = capability_result.get("probs")
            if not isinstance(baseline_raw, dict):
                baseline_raw = (game_result.get("question") or {}).get("probs")
            baseline_probs, baseline_raw_n, baseline_cutoff = canonical_probs(baseline_raw)

            original_answer = str(game_result.get("original_answer", "")).strip()
            for rank_source, source_probs, source_raw_n, source_cutoff, forced_first in (
                ("original_choice", baseline_probs, baseline_raw_n, baseline_cutoff, original_answer),
                ("baseline", baseline_probs, baseline_raw_n, baseline_cutoff, ""),
                ("neutral", neutral_probs, neutral_raw_n, neutral_cutoff, ""),
            ):
                source_complete = all(letter in source_probs for letter in LETTERS)
                source_top_tie = False
                ranks: list[str] = []
                if source_complete:
                    if forced_first in LETTERS:
                        remaining = sorted(
                            [letter for letter in LETTERS if letter != forced_first],
                            key=lambda x: (-source_probs[x], x),
                        )
                        ranks = [forced_first, *remaining]
                        source_top_tie = math.isclose(
                            source_probs[ranks[1]], source_probs[ranks[2]], rel_tol=1e-12, abs_tol=1e-15
                        )
                    else:
                        ranks = sorted(LETTERS, key=lambda x: (-source_probs[x], x))
                        source_top_tie = math.isclose(
                            source_probs[ranks[0]], source_probs[ranks[1]], rel_tol=1e-12, abs_tol=1e-15
                        )

                row = {
                    "model": game_info["model"],
                    "dataset": game_info["dataset"],
                    "rank_source": rank_source,
                    "correctness_set": game_info["correctness_set"],
                    "temperature": game_info["temperature"],
                    "qid": qid,
                    "source_complete": source_complete,
                    "neutral_complete": all(letter in neutral_probs for letter in LETTERS),
                    "game_complete": all(letter in game_probs for letter in LETTERS),
                    "source_top_tie": source_top_tie,
                    "neutral_top_tie": source_top_tie if rank_source == "neutral" else False,
                    "source_raw_token_n": source_raw_n,
                    "source_topk_cutoff": source_cutoff,
                    "neutral_raw_token_n": neutral_raw_n,
                    "game_raw_token_n": game_raw_n,
                    "neutral_topk_cutoff": neutral_cutoff,
                    "game_topk_cutoff": game_cutoff,
                    "neutral_answer_changed": bool(neutral_result.get("answer_changed", False)),
                    "game_answer_changed": bool(game_result.get("answer_changed", False)),
                    "original_answer": original_answer,
                    "game_new_answer": str(game_result.get("new_answer", "")).strip(),
                    "neutral_new_answer": str(neutral_result.get("new_answer", "")).strip(),
                }
                for idx in range(4):
                    row[f"rank{idx + 1}_letter"] = ranks[idx] if ranks else ""
                    row[f"game_has_rank{idx + 1}"] = bool(ranks and ranks[idx] in game_probs)
                    row[f"neutral_has_rank{idx + 1}"] = bool(ranks and ranks[idx] in neutral_probs)
                for letter in LETTERS:
                    row[f"source_p_{letter}"] = source_probs.get(letter, math.nan)
                    row[f"neutral_p_{letter}"] = neutral_probs.get(letter, math.nan)
                    row[f"game_p_{letter}"] = game_probs.get(letter, math.nan)

                pair12_complete = bool(
                    source_complete
                    and not source_top_tie
                    and ranks
                    and ranks[0] in neutral_probs
                    and ranks[1] in neutral_probs
                    and ranks[0] in game_probs
                    and ranks[1] in game_probs
                )
                row["pair12_complete"] = pair12_complete
                row["delta_12_pair"] = math.nan
                for contrast in CONDITION_CONTRASTS:
                    for rank in (1, 2):
                        row[f"{contrast}_rank{rank}"] = math.nan
                if pair12_complete:
                    k, j = ranks[:2]
                    condition_probs = {
                        "source": source_probs,
                        "neutral": neutral_probs,
                        "game": game_probs,
                    }
                    for contrast, (condition, reference) in CONDITION_CONTRASTS.items():
                        for rank, letter in ((1, k), (2, j)):
                            row[f"{contrast}_rank{rank}"] = (
                                math.log(condition_probs[condition][letter])
                                - math.log(condition_probs[reference][letter])
                            )
                    row["delta_12_pair"] = (
                        row["incorrect_minus_neutral_rank1"]
                        - row["incorrect_minus_neutral_rank2"]
                    )

                exact_complete = bool(
                    source_complete
                    and row["neutral_complete"]
                    and row["game_complete"]
                    and not source_top_tie
                    and all(value > 0 for value in source_probs.values())
                    and all(value > 0 for value in neutral_probs.values())
                    and all(value > 0 for value in game_probs.values())
                )
                row["exact_complete"] = exact_complete
                row["delta_1"] = math.nan
                row["delta_2"] = math.nan
                row["delta_12"] = math.nan

                if exact_complete:
                    log_neutral = {letter: math.log(neutral_probs[letter]) for letter in LETTERS}
                    log_game = {letter: math.log(game_probs[letter]) for letter in LETTERS}
                    k, j, r3, r4 = ranks
                    raw_shift = {letter: log_game[letter] - log_neutral[letter] for letter in LETTERS}
                    shift_center = float(np.mean(list(raw_shift.values())))
                    lower_neutral = 0.5 * (log_neutral[r3] + log_neutral[r4])
                    lower_game = 0.5 * (log_game[r3] + log_game[r4])
                    row["delta_1"] = (log_game[k] - lower_game) - (log_neutral[k] - lower_neutral)
                    row["delta_2"] = (log_game[j] - lower_game) - (log_neutral[j] - lower_neutral)
                    row["delta_12"] = (log_game[k] - log_game[j]) - (log_neutral[k] - log_neutral[j])
                    for idx, letter in enumerate(ranks, start=1):
                        row[f"centered_shift_rank{idx}"] = raw_shift[letter] - shift_center
                    selected = str(game_result.get("new_answer", "")).strip()
                    row["game_selected_rank"] = ranks.index(selected) + 1 if selected in ranks else math.nan
                    row["game_only_changed"] = bool(
                        game_result.get("answer_changed", False)
                        and not neutral_result.get("answer_changed", False)
                    )
                rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(manifest)


def summarize(trials: pd.DataFrame, bootstrap: int, permutations: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed_rows = []
    for keys, group in trials.groupby(["model", "dataset", "rank_source"], sort=True):
        detailed_rows.append(
            {
                "model": keys[0],
                "dataset": keys[1],
                "rank_source": keys[2],
                **summarize_group(group, bootstrap, permutations, keys),
            }
        )
    detailed = pd.DataFrame(detailed_rows)

    model_rows = []
    for keys, group in trials.groupby(["model", "rank_source"], sort=True):
        model_rows.append(
            {
                "model": keys[0],
                "dataset": "Pooled",
                "rank_source": keys[1],
                **summarize_group(group, bootstrap, permutations, keys),
            }
        )
    return detailed, pd.DataFrame(model_rows)


def summarize_rank_profiles(trials: pd.DataFrame, bootstrap: int) -> pd.DataFrame:
    exact = trials[trials["exact_complete"]].copy()
    subsets = {
        "all_exact": exact,
        "game_changed": exact[exact["game_answer_changed"]],
        "game_only_changed": exact[exact["game_only_changed"] == True],  # noqa: E712
        "game_selected_rank2": exact[exact["game_selected_rank"] == 2],
    }
    rows: list[dict] = []
    for subset_name, frame in subsets.items():
        for keys, group in frame.groupby(["model", "dataset", "rank_source"], sort=True):
            for rank in range(1, 5):
                metric = f"centered_shift_rank{rank}"
                values = group[metric].dropna().to_numpy(dtype=float)
                seed = stable_seed("|".join(keys) + subset_name + metric)
                low, high = bootstrap_ci(values, "mean", bootstrap, seed)
                rows.append(
                    {
                        "model": keys[0],
                        "dataset": keys[1],
                        "rank_source": keys[2],
                        "subset": subset_name,
                        "rank": rank,
                        "n": len(values),
                        "mean_centered_shift": float(np.mean(values)) if len(values) else math.nan,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def summarize_condition_contrasts(trials: pd.DataFrame, bootstrap: int) -> pd.DataFrame:
    primary = trials[trials["rank_source"] == "original_choice"]
    rows: list[dict] = []
    for keys, group in primary.groupby(["model", "dataset"], sort=True):
        common = group[group["pair12_complete"]]
        for contrast in CONDITION_CONTRASTS:
            for rank in (1, 2):
                metric = f"{contrast}_rank{rank}"
                values = common[metric].dropna().to_numpy(dtype=float)
                seed = stable_seed("|".join(keys) + metric)
                low, high = bootstrap_ci(values, "mean", bootstrap, seed)
                rows.append(
                    {
                        "model": keys[0],
                        "dataset": keys[1],
                        "contrast": contrast,
                        "rank": rank,
                        "n_common_top2": len(values),
                        "coverage": len(values) / len(group) if len(group) else math.nan,
                        "mean_log_shift": float(np.mean(values)) if len(values) else math.nan,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def summarize_performance_correspondence(
    summary: pd.DataFrame, condition_contrasts: pd.DataFrame
) -> pd.DataFrame:
    target = summary[summary["rank_source"] == "original_choice"].copy()
    target["normalized_change_lift"] = (
        (target["game_change_rate"] - target["neutral_change_rate"])
        / (1.0 - target["neutral_change_rate"])
    )
    contrasts = condition_contrasts.pivot(
        index=["model", "dataset"],
        columns=["contrast", "rank"],
        values="mean_log_shift",
    )
    contrasts.columns = [f"{contrast}_rank{rank}" for contrast, rank in contrasts.columns]
    result = target[
        ["model", "dataset", "n_pair12_complete", "pair12_coverage", "normalized_change_lift"]
    ].merge(contrasts.reset_index(), on=["model", "dataset"])
    result["lift_second_choice_cell"] = [
        (model, dataset) in LIFT_SECOND_CHOICE_CELL_SET
        for model, dataset in zip(result["model"], result["dataset"])
    ]
    result["paper_full_success"] = [
        (model, dataset) in PAPER_FULL_SUCCESS_CELLS
        for model, dataset in zip(result["model"], result["dataset"])
    ]
    result["original_suppression_vs_neutral"] = -result["incorrect_minus_neutral_rank1"]
    result["runner_boost_vs_neutral"] = result["incorrect_minus_neutral_rank2"]
    result["runner_change_vs_baseline"] = result["incorrect_minus_baseline_rank2"]
    result["top_two_margin_narrowing"] = (
        result["incorrect_minus_neutral_rank2"]
        - result["incorrect_minus_neutral_rank1"]
    )
    return result.sort_values(
        ["paper_full_success", "lift_second_choice_cell", "dataset", "model"],
        ascending=[False, False, True, True],
    )


def errorbar_svg(summary: pd.DataFrame, output_path: Path) -> None:
    plot = summary[
        (summary["n_exact_complete"] >= 20) & (summary["rank_source"] == "original_choice")
    ].copy()
    if plot.empty:
        return
    plot["label"] = plot["model"] + " — " + plot["dataset"]
    plot = plot.sort_values(["dataset", "model"])
    interval_values = np.concatenate(
        [
            plot["delta_1_mean_ci_low"].to_numpy(),
            plot["delta_1_mean_ci_high"].to_numpy(),
            plot["delta_2_mean_ci_low"].to_numpy(),
            plot["delta_2_mean_ci_high"].to_numpy(),
        ]
    )
    finite = interval_values[np.isfinite(interval_values)]
    bound = max(1.0, float(np.percentile(np.abs(finite), 95)) * 1.15)
    bound = min(bound, 12.0)

    left, right, top, row_h = 340, 50, 80, 28
    width = 1100
    height = top + row_h * len(plot) + 85
    x0, x1 = left, width - right

    def xpos(value: float) -> float:
        value = max(-bound, min(bound, value))
        return x0 + (value + bound) / (2 * bound) * (x1 - x0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.lab{font-size:13px}.tick{font-size:12px}.title{font-size:20px;font-weight:700}.sub{font-size:13px;fill:#555}</style>',
        '<text class="title" x="20" y="30">Second Chance minus neutral: original-choice-ranked relative logit shifts</text>',
        '<text class="sub" x="20" y="53">Dots are means; bars are trial-bootstrap 95% CIs. Natural-logit units. Complete top-token cases only.</text>',
    ]
    for tick in np.linspace(-bound, bound, 7):
        x = xpos(float(tick))
        parts.append(f'<line x1="{x:.1f}" y1="65" x2="{x:.1f}" y2="{height - 42}" stroke="#e4e4e4"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{height - 18}" text-anchor="middle">{tick:.1f}</text>')
    zero = xpos(0)
    parts.append(f'<line x1="{zero:.1f}" y1="65" x2="{zero:.1f}" y2="{height - 42}" stroke="#555" stroke-width="1.5"/>')

    for idx, (_, row) in enumerate(plot.iterrows()):
        y = top + idx * row_h
        parts.append(f'<text class="lab" x="{left - 12}" y="{y + 4}" text-anchor="end">{html.escape(row["label"])}</text>')
        for metric, color, offset in (("delta_1", "#C43C39", -5), ("delta_2", "#2878B5", 5)):
            mean = float(row[f"{metric}_mean"])
            low = float(row[f"{metric}_mean_ci_low"])
            high = float(row[f"{metric}_mean_ci_high"])
            yy = y + offset
            parts.append(f'<line x1="{xpos(low):.1f}" y1="{yy}" x2="{xpos(high):.1f}" y2="{yy}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<circle cx="{xpos(mean):.1f}" cy="{yy}" r="3.8" fill="{color}"/>')
    parts.extend(
        [
            f'<circle cx="{left}" cy="{height - 55}" r="4" fill="#C43C39"/><text class="tick" x="{left + 10}" y="{height - 51}">Δ1: first choice vs ranks 3–4</text>',
            f'<circle cx="{left + 240}" cy="{height - 55}" r="4" fill="#2878B5"/><text class="tick" x="{left + 250}" y="{height - 51}">Δ2: second choice vs ranks 3–4</text>',
            '</svg>',
        ]
    )
    output_path.write_text("\n".join(parts), encoding="utf-8")


def target_profile_svg(profiles: pd.DataFrame, output_path: Path) -> None:
    target_mask = pd.MultiIndex.from_frame(profiles[["model", "dataset"]]).isin(
        LIFT_SECOND_CHOICE_CELL_SET
    )
    plot = profiles[
        target_mask
        & (profiles["rank_source"] == "original_choice")
        & (profiles["subset"] == "all_exact")
    ].copy()
    if plot.empty:
        return
    order = [cell for cell in LIFT_SECOND_CHOICE_CELLS if cell in set(zip(plot["model"], plot["dataset"]))]
    finite = np.concatenate([plot["ci_low"].to_numpy(), plot["ci_high"].to_numpy()])
    bound = max(1.0, float(np.max(np.abs(finite[np.isfinite(finite)]))) * 1.08)
    left, right, top, row_h = 335, 55, 100, 58
    width, height = 1100, top + row_h * len(order) + 90
    x0, x1 = left, width - right

    def xpos(value: float) -> float:
        value = max(-bound, min(bound, value))
        return x0 + (value + bound) / (2 * bound) * (x1 - x0)

    colors = {1: "#C43C39", 2: "#2878B5", 3: "#4C9F70", 4: "#8A62A7"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.lab{font-size:15px}.tick{font-size:12px}.title{font-size:21px;font-weight:700}.sub{font-size:13px;fill:#555}</style>',
        '<text class="title" x="20" y="30">Lift + second-choice cases: centered logit movement by original-choice rank</text>',
        '<text class="sub" x="20" y="53">Second Chance minus neutral. Rank 1 is the recorded original answer; rank 2 is the baseline runner-up.</text>',
        '<text class="sub" x="20" y="72">Table 3 cases passing both behavioral checks. Each trial is centered across A–D. Means and trial-bootstrap 95% CIs.</text>',
    ]
    for tick in np.linspace(-bound, bound, 7):
        x = xpos(float(tick))
        parts.append(f'<line x1="{x:.1f}" y1="82" x2="{x:.1f}" y2="{height - 48}" stroke="#e4e4e4"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{height - 20}" text-anchor="middle">{tick:.1f}</text>')
    zero = xpos(0)
    parts.append(f'<line x1="{zero:.1f}" y1="82" x2="{zero:.1f}" y2="{height - 48}" stroke="#555" stroke-width="1.5"/>')
    offsets = {1: -18, 2: -6, 3: 6, 4: 18}
    for row_idx, (model, dataset) in enumerate(order):
        y = top + row_idx * row_h + 25
        label = f"{model} — {dataset}"
        parts.append(f'<text class="lab" x="{left - 14}" y="{y + 5}" text-anchor="end">{html.escape(label)}</text>')
        subset = plot[(plot["model"] == model) & (plot["dataset"] == dataset)]
        for _, point in subset.iterrows():
            rank = int(point["rank"])
            yy = y + offsets[rank]
            color = colors[rank]
            parts.append(
                f'<line x1="{xpos(float(point["ci_low"])):.1f}" y1="{yy}" '
                f'x2="{xpos(float(point["ci_high"])):.1f}" y2="{yy}" stroke="{color}" stroke-width="2"/>'
            )
            parts.append(
                f'<circle cx="{xpos(float(point["mean_centered_shift"])):.1f}" cy="{yy}" r="4" fill="{color}"/>'
            )
    legend_x = left
    for rank in range(1, 5):
        lx = legend_x + (rank - 1) * 150
        parts.append(f'<circle cx="{lx}" cy="{height - 55}" r="4" fill="{colors[rank]}"/>')
        parts.append(f'<text class="tick" x="{lx + 10}" y="{height - 51}">Rank {rank}</text>')
    parts.append('</svg>')
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_report(
    summary: pd.DataFrame,
    profiles: pd.DataFrame,
    condition_contrasts: pd.DataFrame,
    performance: pd.DataFrame,
    all_performance: pd.DataFrame,
    manifest: pd.DataFrame,
    trials: pd.DataFrame,
    output_path: Path,
) -> None:
    summary_target_mask = pd.MultiIndex.from_frame(summary[["model", "dataset"]]).isin(
        LIFT_SECOND_CHOICE_CELL_SET
    )
    profile_target_mask = pd.MultiIndex.from_frame(profiles[["model", "dataset"]]).isin(
        LIFT_SECOND_CHOICE_CELL_SET
    )
    focus = summary[summary_target_mask & (summary["rank_source"] == "original_choice")].copy()
    focus_profiles = profiles[
        profile_target_mask & (profiles["rank_source"] == "original_choice")
    ].copy()
    overall = focus_profiles[focus_profiles["subset"] == "all_exact"]
    game_only = focus_profiles[focus_profiles["subset"] == "game_only_changed"]
    first_down_cells = int(((overall["rank"] == 1) & (overall["ci_high"] < 0)).sum())
    second_up_cells = int(((overall["rank"] == 2) & (overall["ci_low"] > 0)).sum())
    game_only_second_up = int(((game_only["rank"] == 2) & (game_only["ci_low"] > 0)).sum())
    n_cells = len(focus)
    top_two_down_cells = int((focus["delta_12_pair_mean_ci_high"] < 0).sum())
    contrast_target_mask = pd.MultiIndex.from_frame(
        condition_contrasts[["model", "dataset"]]
    ).isin(LIFT_SECOND_CHOICE_CELL_SET)
    target_contrasts = condition_contrasts[contrast_target_mask].copy()
    runner_raw_up = target_contrasts[
        (target_contrasts["contrast"] == "incorrect_minus_neutral")
        & (target_contrasts["rank"] == 2)
    ]
    runner_raw_up_cells = int((runner_raw_up["mean_log_shift"] > 0).sum())
    runner_raw_up_resolved = int((runner_raw_up["ci_low"] > 0).sum())
    incorrect_vs_baseline = target_contrasts[
        target_contrasts["contrast"] == "incorrect_minus_baseline"
    ]
    incorrect_vs_neutral = target_contrasts[
        target_contrasts["contrast"] == "incorrect_minus_neutral"
    ]
    baseline_first_down = int(
        ((incorrect_vs_baseline["rank"] == 1) & (incorrect_vs_baseline["ci_high"] < 0)).sum()
    )
    baseline_runner_up = int(
        ((incorrect_vs_baseline["rank"] == 2) & (incorrect_vs_baseline["ci_low"] > 0)).sum()
    )
    neutral_first_down = int(
        ((incorrect_vs_neutral["rank"] == 1) & (incorrect_vs_neutral["ci_high"] < 0)).sum()
    )
    suppression_lift_rho = float(
        performance["original_suppression_vs_neutral"].rank().corr(
            performance["normalized_change_lift"].rank()
        )
    )
    margin_lift_rho = float(
        performance["top_two_margin_narrowing"].rank().corr(
            performance["normalized_change_lift"].rank()
        )
    )
    successful = performance[performance["paper_full_success"]]
    other = performance[~performance["paper_full_success"]]
    successful_runner_above_baseline = int((successful["runner_change_vs_baseline"] > 0).sum())
    other_runner_above_baseline = int((other["runner_change_vs_baseline"] > 0).sum())
    analyzable = all_performance[all_performance["n_pair12_complete"] >= 50]
    all_suppression_rho = float(
        analyzable["original_suppression_vs_neutral"].rank().corr(
            analyzable["normalized_change_lift"].rank()
        )
    )
    all_runner_neutral_rho = float(
        analyzable["runner_boost_vs_neutral"].rank().corr(
            analyzable["normalized_change_lift"].rank()
        )
    )
    all_runner_baseline_rho = float(
        analyzable["runner_change_vs_baseline"].rank().corr(
            analyzable["normalized_change_lift"].rank()
        )
    )
    all_margin_rho = float(
        analyzable["top_two_margin_narrowing"].rank().corr(
            analyzable["normalized_change_lift"].rank()
        )
    )
    dataset_suppression_rho = {
        dataset: float(
            group["original_suppression_vs_neutral"].rank().corr(
                group["normalized_change_lift"].rank()
            )
        )
        for dataset, group in analyzable.groupby("dataset")
    }
    unique_pairs = int((trials["rank_source"] == "original_choice").sum())
    lines = [
        "# Rank-shift diagnostic",
        "",
        "The primary rank-1 option is the model's recorded original answer. Rank 2 is the "
        "highest-probability remaining option in the original baseline capabilities distribution. "
        "All effects are Second Chance minus neutral in natural-logit units. Baseline- and "
        "neutral-ranked sensitivity analyses are included in the CSV outputs.",
        "",
        "The primary exact analysis requires all four canonical A–D tokens in both stored top-token "
        "records. Whitespace and punctuation variants are merged. Because only four raw tokens were "
        "stored, duplicate tokenizations can censor a letter; coverage is therefore reported and the "
        "complete-case estimates should be treated as a diagnostic rather than a definitive estimand.",
        "",
        f"Matched files: {int(manifest['matched'].sum())}/{len(manifest)}. Unique paired question-condition "
        f"observations: {unique_pairs}. The trial CSV has {len(trials)} rows because it retains three rank definitions.",
        "",
        "## Main result",
        "",
        f"Across the {n_cells} model-by-dataset cells that show significant lift and pass the second-choice "
        f"check in Table 3, the centered rank-1 movement was detectably negative in {first_down_cells}/{n_cells} "
        f"cells, while the centered runner-up movement was detectably positive in {second_up_cells}/{n_cells}. "
        f"The rank-1-minus-rank-2 margin also narrowed detectably in {top_two_down_cells}/{n_cells} cells using "
        "the higher-coverage top-two analysis. The unconditional signature is therefore robust loss of support "
        "for the original answer, not a dedicated runner-up boost. The two unresolved top-two cases are Qwen; "
        "Qwen GPQA is also the sole cell without detectable rank-1 suppression in the exact four-answer analysis.",
        "",
        f"On the subset of game-only changes—the trials that create behavioral lift—the runner-up movement "
        f"was detectably positive in {game_only_second_up}/{n_cells} cells. Because this conditions on the "
        "behavioral outcome, it is descriptive rather than evidence for a separate causal boosting mechanism.",
        "",
        "## Cases with lift and a passed second-choice check",
        "",
        "The three OpenAI models are the stricter subset that the paper also identifies as passing the "
        "entropy check. They are retained here, but they are not privileged in the primary analysis.",
        "",
        "| Model | Dataset | N exact | Coverage | Δ1 [95% CI] | Δ2 [95% CI] | N top-two | Δ12 top-two [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    focus["cell_order"] = [
        LIFT_SECOND_CHOICE_CELLS.index((model, dataset))
        for model, dataset in zip(focus["model"], focus["dataset"])
    ]
    for _, row in focus.sort_values("cell_order").iterrows():
        def ci(metric: str) -> str:
            return f"{row[f'{metric}_mean']:.2f} [{row[f'{metric}_mean_ci_low']:.2f}, {row[f'{metric}_mean_ci_high']:.2f}]"
        lines.append(
            f"| {row['model']} | {row['dataset']} | {int(row['n_exact_complete'])} | "
            f"{row['exact_coverage']:.1%} | {ci('delta_1')} | {ci('delta_2')} | "
            f"{int(row['n_pair12_complete'])} | {ci('delta_12_pair')} |"
        )
    lines.extend(
        [
            "",
            "## Baseline, neutral, and incorrect-prompt decomposition",
            "",
            "The baseline capabilities test defines rank 1 (the recorded original answer) and rank 2 "
            "(the highest-probability remaining answer). The table reports raw log-probability shifts on "
            "the common top-two sample; each cell is `rank 1 / rank 2`. Bootstrap CIs are in "
            "`condition_contrasts_by_model_dataset.csv`.",
            "",
            f"The runner-up's raw log probability is higher under the incorrect prompt than under the "
            f"neutral prompt in {runner_raw_up_cells}/{n_cells} cells, with a bootstrap CI above zero in "
            f"{runner_raw_up_resolved}/{n_cells}. This is not a selective runner-up "
            "boost: in the centered four-rank analysis, ranks 3 and 4 generally gain more relative support.",
            "",
            f"Relative to baseline, the incorrect prompt lowers the original answer detectably in "
            f"{baseline_first_down}/{n_cells} cells, but raises the runner-up detectably above its baseline "
            f"level in only {baseline_runner_up}/{n_cells}. Relative to neutral, original-answer suppression "
            f"is detectable in {neutral_first_down}/{n_cells} cells and the raw runner-up increase in "
            f"{runner_raw_up_resolved}/{n_cells}. Thus the incorrect instruction often restores or raises the "
            "runner-up relative to the neutral redo context without necessarily pushing it above baseline.",
            "",
            "| Model | Dataset | Top-two coverage | Neutral − baseline | Incorrect − baseline | Incorrect − neutral |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model, dataset in LIFT_SECOND_CHOICE_CELLS:
        cell = target_contrasts[
            (target_contrasts["model"] == model) & (target_contrasts["dataset"] == dataset)
        ]
        if cell.empty:
            continue

        def rank_pair(contrast: str) -> str:
            points = cell[cell["contrast"] == contrast].set_index("rank")["mean_log_shift"]
            return f"{points.loc[1]:+.2f} / {points.loc[2]:+.2f}"

        coverage = float(cell["coverage"].iloc[0])
        lines.append(
            f"| {model} | {dataset} | {coverage:.1%} | "
            f"{rank_pair('neutral_minus_baseline')} | "
            f"{rank_pair('incorrect_minus_baseline')} | "
            f"{rank_pair('incorrect_minus_neutral')} |"
        )
    lines.extend(
        [
            "",
            "## Correspondence with behavioral performance",
            "",
            f"Across all {len(analyzable)} model-dataset cells with at least 50 common top-two trials, normalized "
            f"behavioral lift is strongly rank-correlated with original-answer suppression relative to neutral "
            f"(Spearman ρ={all_suppression_rho:.2f}) and narrowing of the original-versus-runner-up margin "
            f"(ρ={all_margin_rho:.2f}). The suppression relationship holds within GPQA "
            f"(ρ={dataset_suppression_rho.get('GPQA', math.nan):.2f}) and SimpleMC "
            f"(ρ={dataset_suppression_rho.get('SimpleMC', math.nan):.2f}). It is also present within the 14 "
            f"lift-plus-second-choice cells alone (ρ={suppression_lift_rho:.2f}; margin ρ={margin_lift_rho:.2f}).",
            "",
            f"Runner-up increase relative to neutral has a weaker positive relationship with lift "
            f"(ρ={all_runner_neutral_rho:.2f}). Runner-up change relative to baseline has essentially no "
            f"relationship with lift across the broader sample (ρ={all_runner_baseline_rho:.2f}).",
            "",
            "The paper's seven fully successful cells are not cleanly separated from the other seven by "
            "suppression magnitude. They are differentiated by restraint relative to baseline: the runner-up "
            f"exceeds its baseline probability in {successful_runner_above_baseline}/7 fully successful cells, "
            f"versus {other_runner_above_baseline}/7 other cells. The two exceptions among the other cells are "
            "Qwen, whose entropy increase appears to be distributed beyond the runner-up. Because this sign "
            "comparison was identified post hoc, N=14, and does not generalize to continuous lift across all "
            "models, it should be treated as a candidate signature of the no-entropy-increase subgroup only.",
            "",
            "See `all_performance_correspondence.csv` for all cells and `performance_correspondence.csv` for "
            "the 14 lift-plus-second-choice cells.",
        ]
    )
    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- Δ1 = change in rank-1 logit relative to the mean of rank-3 and rank-4 logits.",
            "- Δ2 = change in rank-2 logit relative to the mean of rank-3 and rank-4 logits.",
            "- Δ12 = change in the rank-1 minus rank-2 margin; algebraically, Δ12 = Δ1 − Δ2.",
            "- The top-two Δ12 estimate needs only rank 1 and rank 2 to be observed in both conditions, so it has better coverage than Δ1/Δ2.",
            "- A negative Δ1 is consistent with first-choice suppression; a positive Δ2 is consistent with second-choice boosting.",
            "- The pattern label uses bootstrap CIs and is descriptive, not an equivalence test.",
            "- Complete cases have higher answer-change rates than excluded cases, especially for GPT-4.1; effect magnitudes may therefore be upward-biased.",
            "",
            "See `summary_by_model_dataset.csv`, `summary_by_model.csv`, `trial_rank_shifts.csv`, "
            "`condition_contrasts_by_model_dataset.csv`, `file_manifest.csv`, `rank_shift_means.svg`, and "
            "`lift_second_choice_centered_rank_profile.svg` for full results.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    game_dir = args.source_root / "sc_logs_new"
    neutral_dir = args.source_root / "sc_logs_neutral"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    game_infos = choose_game_files(game_dir)
    neutral_lookup = build_neutral_lookup(neutral_dir)
    capability_lookups = build_capability_lookups(args.source_root, game_infos)
    trials, manifest = analyze_pairs(game_infos, neutral_lookup, capability_lookups)
    if trials.empty:
        raise SystemExit("No matched probability trials found")

    detailed, pooled = summarize(trials, args.bootstrap, args.permutations)
    profiles = summarize_rank_profiles(trials, args.bootstrap)
    condition_contrasts = summarize_condition_contrasts(trials, args.bootstrap)
    all_performance = summarize_performance_correspondence(detailed, condition_contrasts)
    performance = all_performance[all_performance["lift_second_choice_cell"]].copy()
    trials.to_csv(args.output_dir / "trial_rank_shifts.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    manifest.to_csv(args.output_dir / "file_manifest.csv", index=False)
    detailed.to_csv(args.output_dir / "summary_by_model_dataset.csv", index=False)
    pooled.to_csv(args.output_dir / "summary_by_model.csv", index=False)
    profiles.to_csv(args.output_dir / "rank_profile_subsets.csv", index=False)
    condition_contrasts.to_csv(
        args.output_dir / "condition_contrasts_by_model_dataset.csv", index=False
    )
    performance.to_csv(args.output_dir / "performance_correspondence.csv", index=False)
    all_performance.to_csv(args.output_dir / "all_performance_correspondence.csv", index=False)
    errorbar_svg(detailed, args.output_dir / "rank_shift_means.svg")
    target_profile_svg(profiles, args.output_dir / "lift_second_choice_centered_rank_profile.svg")
    write_report(
        detailed,
        profiles,
        condition_contrasts,
        performance,
        all_performance,
        manifest,
        trials,
        args.output_dir / "REPORT.md",
    )

    print(f"Matched file pairs: {int(manifest['matched'].sum())}/{len(manifest)}")
    print(f"Paired trials: {len(trials)}")
    print(f"Exact complete trials: {int(trials['exact_complete'].sum())}")
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
