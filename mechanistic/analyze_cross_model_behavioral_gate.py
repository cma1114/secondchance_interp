from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


LETTERS = "ABCD"


def _entropy_bits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    return -(probs * np.log2(np.clip(probs, 1e-30, None))).sum(axis=-1)


def _ci(
    values: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    draws: int = 10_000,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata)
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty subset")
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    boot = np.zeros(draws, dtype=float)
    for group in groups:
        selected = rng.choice(group, size=(draws, len(group)), replace=True)
        boot += values[selected].sum(axis=1)
    boot /= len(values)
    low, high = np.quantile(boot, (0.025, 0.975))
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(len(values)),
    }


def _fmt(cell: dict[str, Any], scale: float = 1.0, digits: int = 1) -> str:
    return (
        f"{cell['mean'] * scale:+.{digits}f} "
        f"[{cell['ci_low'] * scale:+.{digits}f}, {cell['ci_high'] * scale:+.{digits}f}]"
    )


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def analyze(
    results_path: Path,
    discovery_plan: Path,
    confirmation_plan: Path,
    output_dir: Path,
    figure_path: Path,
    seed: int,
) -> dict[str, Any]:
    payload = json.loads(results_path.read_text())
    model_name = payload["model_id"].split("/")[-1]
    if not payload.get("complete"):
        raise RuntimeError("Behavioral run is not marked complete")
    scenarios = payload["scenarios"]
    required = {
        "baseline",
        "incorrect_again_nonremapped",
        "lost_again_nonremapped",
        "incorrect_again_remapped",
        "lost_again_remapped",
    }
    if not required <= set(scenarios):
        raise RuntimeError("Behavioral run is missing required scenarios")
    qids = list(scenarios["baseline"])
    if any(set(scenarios[name]) != set(qids) for name in required):
        raise RuntimeError("Scenario question sets differ")
    if not all(row["all_logits_finite"] for name in required for row in scenarios[name].values()):
        raise RuntimeError("A scenario contains non-finite outputs")

    discovery_payload = json.loads(discovery_plan.read_text())
    confirmation_payload = json.loads(confirmation_plan.read_text())
    discovery_ids = set(
        discovery_payload.get("question_ids", discovery_payload.get("discovery_question_ids", []))
    )
    confirmation_ids = set(
        confirmation_payload.get(
            "question_ids", confirmation_payload.get("confirmation_question_ids", [])
        )
    )
    if discovery_ids & confirmation_ids or discovery_ids | confirmation_ids != set(qids):
        raise RuntimeError("Frozen 251/249 split does not partition the behavioral questions")
    split_masks = {
        "all": np.ones(len(qids), dtype=bool),
        "discovery": np.asarray([qid in discovery_ids for qid in qids]),
        "confirmation": np.asarray([qid in confirmation_ids for qid in qids]),
    }

    unrestricted = {
        name: np.asarray([scenarios[name][qid]["answer_new_letter"] for qid in qids], dtype=object)
        for name in required
    }
    compliance = {
        name: float(np.mean(np.isin(values, list(LETTERS))))
        for name, values in unrestricted.items()
    }
    use_unrestricted = all(value == 1.0 for value in compliance.values())
    aggregated = {
        name: np.asarray(
            [scenarios[name][qid]["aggregated_ad_answer_new_letter"] for qid in qids]
        )
        for name in required
    }
    answers = unrestricted if use_unrestricted else aggregated
    logits = {
        name: np.asarray(
            [scenarios[name][qid]["aggregated_ad_logits"] for qid in qids], dtype=float
        )
        for name in required
    }
    ties = {
        name: int(np.sum(np.sort(values, axis=1)[:, -1] == np.sort(values, axis=1)[:, -2]))
        for name, values in logits.items()
    }

    baseline_answer = answers["baseline"].astype(str)
    baseline_index = np.asarray([LETTERS.index(letter) for letter in baseline_answer])
    strata = baseline_answer.copy()
    # W1 is the model's primary unrestricted first-pass choice.  OLMo can put
    # slightly different mass on bare versus space-prefixed answer tokens, so
    # the log-sum-exp A-D argmax is not allowed to silently redefine W1.  Rank
    # the remaining three candidates by aggregated Baseline evidence.
    baseline_order = np.empty((len(qids), 4), dtype=int)
    for qi, winner in enumerate(baseline_index):
        remaining = [index for index in range(4) if index != winner]
        remaining.sort(key=lambda index: (-logits["baseline"][qi, index], index))
        baseline_order[qi] = [winner, *remaining]

    def content_answers(name: str) -> np.ndarray:
        if name.endswith("_nonremapped") or name == "baseline":
            return answers[name].astype(str)
        return np.asarray([
            scenarios[name][qid]["new_to_original"][str(answers[name][index])]
            for index, qid in enumerate(qids)
        ])

    content = {name: content_answers(name) for name in required}
    switch = {
        name: content[name] != baseline_answer for name in required if name != "baseline"
    }
    old_letter_avoidance = {
        name: answers[name].astype(str) != baseline_answer
        for name in ("incorrect_again_remapped", "lost_again_remapped")
    }

    content_logits: dict[str, np.ndarray] = {"baseline": logits["baseline"]}
    for name in required - {"baseline"}:
        if name.endswith("_nonremapped"):
            content_logits[name] = logits[name]
        else:
            aligned = np.empty_like(logits[name])
            for qi, qid in enumerate(qids):
                original_to_new = scenarios[name][qid]["original_to_new"]
                aligned[qi] = [
                    logits[name][qi, LETTERS.index(original_to_new[original])]
                    for original in LETTERS
                ]
            content_logits[name] = aligned
    rank_logits = {
        name: np.take_along_axis(values, baseline_order, axis=1)
        for name, values in content_logits.items()
    }
    centered_rank = {name: _center(values) for name, values in rank_logits.items()}

    rng = np.random.default_rng(seed)
    summary: dict[str, Any] = {
        "model_id": payload["model_id"],
        "model_revision": payload["model_revision"],
        "n": len(qids),
        "primary_choice_readout": "unrestricted top token" if use_unrestricted else "aggregated A-D argmax",
        "unrestricted_answer_only_compliance": compliance,
        "exact_aggregated_ad_ties_using_displayed_order": ties,
        "prompt_audit": {
            "game_neutral_only_difference": payload["prompt_audit"]["game_neutral_only_difference"],
            "first_answer_history": "empty assistant turn",
            "chat_serialization": payload["config"]["chat_serialization"],
        },
        "subsets": {},
    }

    for split_name, mask in split_masks.items():
        local_strata = strata[mask]
        split: dict[str, Any] = {"n": int(mask.sum()), "mappings": {}}
        for mapping in ("nonremapped", "remapped"):
            game = f"incorrect_again_{mapping}"
            neutral = f"lost_again_{mapping}"
            game_switch = switch[game][mask]
            neutral_switch = switch[neutral][mask]
            game_minus_neutral = game_switch.astype(float) - neutral_switch.astype(float)
            rank_delta = centered_rank[game][mask] - centered_rank[neutral][mask]
            game_from_baseline = centered_rank[game][mask] - centered_rank["baseline"][mask]
            neutral_from_baseline = centered_rank[neutral][mask] - centered_rank["baseline"][mask]
            split["mappings"][mapping] = {
                "semantic_switch_rate": {
                    "game": float(game_switch.mean()),
                    "neutral": float(neutral_switch.mean()),
                    "game_minus_neutral": _ci(game_minus_neutral, local_strata, rng),
                    "paired_counts": {
                        "game_only": int(np.sum(game_switch & ~neutral_switch)),
                        "neutral_only": int(np.sum(~game_switch & neutral_switch)),
                        "both": int(np.sum(game_switch & neutral_switch)),
                        "neither": int(np.sum(~game_switch & ~neutral_switch)),
                    },
                },
                "centered_rank_logits_game_minus_neutral": {
                    f"W{rank + 1}": _ci(rank_delta[:, rank], local_strata, rng)
                    for rank in range(4)
                },
                "centered_rank_logits_game_minus_baseline": {
                    f"W{rank + 1}": _ci(game_from_baseline[:, rank], local_strata, rng)
                    for rank in range(4)
                },
                "centered_rank_logits_neutral_minus_baseline": {
                    f"W{rank + 1}": _ci(neutral_from_baseline[:, rank], local_strata, rng)
                    for rank in range(4)
                },
                "entropy_bits": {
                    "baseline": float(_entropy_bits(logits["baseline"][mask]).mean()),
                    "game": float(_entropy_bits(logits[game][mask]).mean()),
                    "neutral": float(_entropy_bits(logits[neutral][mask]).mean()),
                    "game_minus_neutral": _ci(
                        _entropy_bits(logits[game][mask]) - _entropy_bits(logits[neutral][mask]),
                        local_strata,
                        rng,
                    ),
                    "game_minus_baseline": _ci(
                        _entropy_bits(logits[game][mask]) - _entropy_bits(logits["baseline"][mask]),
                        local_strata,
                        rng,
                    ),
                },
            }
        game = "incorrect_again_remapped"
        neutral = "lost_again_remapped"
        centered_game_new = _center(logits[game])[mask]
        centered_neutral_new = _center(logits[neutral])[mask]
        q_indices = np.flatnonzero(mask)
        semantic_new_index = []
        literal_old_index = []
        for global_index in q_indices:
            qid = qids[global_index]
            semantic_new = scenarios[game][qid]["original_to_new"][baseline_answer[global_index]]
            semantic_new_index.append(LETTERS.index(semantic_new))
            literal_old_index.append(baseline_index[global_index])
        local_q = np.arange(mask.sum())
        semantic_suppression = (
            centered_neutral_new[local_q, semantic_new_index]
            - centered_game_new[local_q, semantic_new_index]
        )
        literal_suppression = (
            centered_neutral_new[local_q, literal_old_index]
            - centered_game_new[local_q, literal_old_index]
        )
        old_game = old_letter_avoidance[game][mask]
        old_neutral = old_letter_avoidance[neutral][mask]
        split["remapped_semantic_vs_literal"] = {
            "old_literal_letter_avoidance_rate": {
                "game": float(old_game.mean()),
                "neutral": float(old_neutral.mean()),
                "game_minus_neutral": _ci(
                    old_game.astype(float) - old_neutral.astype(float), local_strata, rng
                ),
            },
            "neutral_minus_game_centered_logit_suppression": {
                "semantic_old_winner_at_new_letter": _ci(semantic_suppression, local_strata, rng),
                "old_literal_letter": _ci(literal_suppression, local_strata, rng),
                "semantic_minus_literal": _ci(
                    semantic_suppression - literal_suppression, local_strata, rng
                ),
            },
        }
        summary["subsets"][split_name] = split

    confirmation = summary["subsets"]["confirmation"]
    remap_gap = confirmation["mappings"]["remapped"]["semantic_switch_rate"]["game_minus_neutral"]
    target = confirmation["remapped_semantic_vs_literal"]["neutral_minus_game_centered_logit_suppression"]
    consistent_sign = all(
        summary["subsets"][name]["mappings"]["remapped"]["semantic_switch_rate"]["game_minus_neutral"]["mean"] > 0
        for name in ("discovery", "confirmation")
    )
    summary["mechanistic_followup_gate"] = {
        "positive_remapped_semantic_switch_gap": bool(
            remap_gap["ci_low"] > 0 or (remap_gap["mean"] >= 0.05 and consistent_sign)
        ),
        "positive_semantic_w1_suppression": bool(target["semantic_old_winner_at_new_letter"]["ci_low"] > 0),
        "semantic_target_exceeds_old_literal_letter": bool(target["semantic_minus_literal"]["ci_low"] > 0),
    }
    summary["mechanistic_followup_gate"]["passed"] = all(
        summary["mechanistic_followup_gate"][key]
        for key in (
            "positive_remapped_semantic_switch_gap",
            "positive_semantic_w1_suppression",
            "semantic_target_exceeds_old_literal_letter",
        )
    )

    # Sensitivity analysis for models that occasionally ignore the answer-only
    # instruction.  The prespecified primary readout remains the conditional
    # A-D argmax on all questions; this reports the same paired switch contrast
    # only where Baseline, Game, and Neutral each emitted an unrestricted A-D
    # token, so format failures cannot manufacture the headline effect.
    sensitivity_rng = np.random.default_rng(seed + 10_000)
    sensitivity: dict[str, Any] = {}
    for mapping in ("nonremapped", "remapped"):
        game = f"incorrect_again_{mapping}"
        neutral = f"lost_again_{mapping}"
        game_switch, neutral_switch, sensitivity_strata = [], [], []
        for qi, qid in enumerate(qids):
            baseline_letter = unrestricted["baseline"][qi]
            game_letter = unrestricted[game][qi]
            neutral_letter = unrestricted[neutral][qi]
            if any(
                value is None or value not in LETTERS
                for value in (baseline_letter, game_letter, neutral_letter)
            ):
                continue
            if mapping == "remapped":
                game_answer = scenarios[game][qid]["new_to_original"][str(game_letter)]
                neutral_answer = scenarios[neutral][qid]["new_to_original"][str(neutral_letter)]
            else:
                game_answer = str(game_letter)
                neutral_answer = str(neutral_letter)
            game_switch.append(game_answer != baseline_letter)
            neutral_switch.append(neutral_answer != baseline_letter)
            sensitivity_strata.append(str(baseline_letter))
        game_values = np.asarray(game_switch, dtype=bool)
        neutral_values = np.asarray(neutral_switch, dtype=bool)
        paired = game_values.astype(float) - neutral_values.astype(float)
        sensitivity[mapping] = {
            "n_complete_cases": int(len(paired)),
            "game_semantic_switch_rate": float(game_values.mean()),
            "neutral_semantic_switch_rate": float(neutral_values.mean()),
            "game_minus_neutral": _ci(
                paired,
                np.asarray(sensitivity_strata),
                sensitivity_rng,
            ),
        }
    summary["unrestricted_complete_case_sensitivity"] = sensitivity

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    all_data = summary["subsets"]["all"]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), layout="constrained")
    colors = {"game": "#b23a48", "neutral": "#3f6da8"}
    ax = axes[0, 0]
    x = np.arange(2)
    width = 0.34
    for offset, task in ((-width / 2, "game"), (width / 2, "neutral")):
        vals = [all_data["mappings"][m]["semantic_switch_rate"][task] for m in ("nonremapped", "remapped")]
        ax.bar(x + offset, vals, width, label=task.title(), color=colors[task])
    ax.set_xticks(x, ["Same option order", "All options remapped"])
    ax.set_ylabel("Semantic switch rate")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("A. Behavioral switching")

    ax = axes[0, 1]
    width = 0.34
    for offset, mapping, color in ((-width / 2, "nonremapped", "#777777"), (width / 2, "remapped", "#6a4c93")):
        cells = all_data["mappings"][mapping]["centered_rank_logits_game_minus_neutral"]
        vals = np.asarray([cells[f"W{i}"]["mean"] for i in range(1, 5)])
        lows = vals - np.asarray([cells[f"W{i}"]["ci_low"] for i in range(1, 5)])
        highs = np.asarray([cells[f"W{i}"]["ci_high"] for i in range(1, 5)]) - vals
        ax.bar(np.arange(4) + offset, vals, width, color=color, label=mapping.title())
        ax.errorbar(np.arange(4) + offset, vals, yerr=[lows, highs], fmt="none", color="black", capsize=3, lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(4), ["W1", "W2", "W3", "W4"])
    ax.set_ylabel("Game − Neutral centered logit")
    ax.legend(frameon=False)
    ax.set_title("B. Baseline-rank redistribution")

    ax = axes[1, 0]
    suppression = all_data["remapped_semantic_vs_literal"]["neutral_minus_game_centered_logit_suppression"]
    keys = ("semantic_old_winner_at_new_letter", "old_literal_letter")
    vals = np.asarray([suppression[key]["mean"] for key in keys])
    lows = vals - np.asarray([suppression[key]["ci_low"] for key in keys])
    highs = np.asarray([suppression[key]["ci_high"] for key in keys]) - vals
    ax.bar(np.arange(2), vals, color=["#6a4c93", "#aaaaaa"])
    ax.errorbar(np.arange(2), vals, yerr=[lows, highs], fmt="none", color="black", capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(2), ["Old winner content\n(new letter)", "Old literal letter\n(different content)"])
    ax.set_ylabel("Neutral − Game centered logit")
    ax.set_title("C. Semantic versus letter targeting")

    ax = axes[1, 1]
    labels = ["Baseline", "Game\nsame order", "Neutral\nsame order", "Game\nremapped", "Neutral\nremapped"]
    vals = [
        all_data["mappings"]["nonremapped"]["entropy_bits"]["baseline"],
        all_data["mappings"]["nonremapped"]["entropy_bits"]["game"],
        all_data["mappings"]["nonremapped"]["entropy_bits"]["neutral"],
        all_data["mappings"]["remapped"]["entropy_bits"]["game"],
        all_data["mappings"]["remapped"]["entropy_bits"]["neutral"],
    ]
    ax.bar(np.arange(5), vals, color=["#555555", colors["game"], colors["neutral"], colors["game"], colors["neutral"]])
    ax.set_xticks(np.arange(5), labels)
    ax.set_ylabel("A-D entropy (bits)")
    ax.set_ylim(0, 2)
    ax.set_title("D. Answer-distribution entropy")

    dataset_name = "TriviaMC" if qids and str(qids[0]).startswith("triviamc_") else "SimpleMC"
    fig.suptitle(f"{model_name} — clean {dataset_name} behavioral gate", fontsize=15)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    all_non = all_data["mappings"]["nonremapped"]["semantic_switch_rate"]
    all_remap = all_data["mappings"]["remapped"]["semantic_switch_rate"]
    rank = all_data["mappings"]["remapped"]["centered_rank_logits_game_minus_neutral"]
    report = [
        f"# {model_name} {dataset_name} behavioral gate",
        "",
        "## Design and validity",
        "",
        f"All {len(qids)} frozen {dataset_name} questions were run with the model's native chat template. The first assistant turn was empty. In each mapping, Game and Neutral differed only at `incorrect` versus `lost` in `Your answer was ... Choose the answer again.` Every remapped option moved to a different letter.",
        "",
        f"Primary choice readout: **{summary['primary_choice_readout']}**. Answer-only compliance ranged from **{100 * min(compliance.values()):.1f}% to {100 * max(compliance.values()):.1f}%**; all outputs were finite.",
        "",
        f"As a format-compliance sensitivity check, restricting to questions where Baseline, Game, and Neutral all emitted unrestricted A-D tokens left a same-order gap of **{_fmt(sensitivity['nonremapped']['game_minus_neutral'], 100)} points** (n={sensitivity['nonremapped']['n_complete_cases']}) and a remapped semantic gap of **{_fmt(sensitivity['remapped']['game_minus_neutral'], 100)} points** (n={sensitivity['remapped']['n_complete_cases']}).",
        "",
        "## Behavioral result",
        "",
        f"With the same option order, Game switched semantic answers on **{100 * all_non['game']:.1f}%** of questions and Neutral on **{100 * all_non['neutral']:.1f}%**; the paired Game-minus-Neutral gap was **{_fmt(all_non['game_minus_neutral'], 100)} percentage points**.",
        "",
        f"After every option content moved to a new letter, Game switched semantic answers on **{100 * all_remap['game']:.1f}%** and Neutral on **{100 * all_remap['neutral']:.1f}%**; the semantic gap was **{_fmt(all_remap['game_minus_neutral'], 100)} percentage points**.",
        "",
        "## What Game changes",
        "",
        f"In the remapped presentation, Game-minus-Neutral centered evidence by the model's own first-pass rank was W1/W2/W3/W4 = **{_fmt(rank['W1'], digits=3)} / {_fmt(rank['W2'], digits=3)} / {_fmt(rank['W3'], digits=3)} / {_fmt(rank['W4'], digits=3)} logits**.",
        "",
        f"The old winner's semantic content, now at a new letter, was suppressed by **{_fmt(suppression['semantic_old_winner_at_new_letter'], digits=3)} logits** in Game relative to Neutral. The old literal letter, now attached to different content, changed by **{_fmt(suppression['old_literal_letter'], digits=3)} logits**. Their semantic-minus-letter contrast was **{_fmt(suppression['semantic_minus_literal'], digits=3)} logits**.",
        "",
        "## Mechanistic follow-up decision",
        "",
        f"Prespecified gate passed: **{summary['mechanistic_followup_gate']['passed']}**.",
        "",
        "The gate requires a positive remapped semantic switch gap, positive suppression of the semantic old winner, and stronger targeting of that content than of its former literal letter on the frozen confirmation split.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the clean cross-model behavioral gate")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--confirmation-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(
        args.results,
        args.discovery_plan,
        args.confirmation_plan,
        args.output_dir,
        args.figure,
        args.seed,
    )


if __name__ == "__main__":
    main()
