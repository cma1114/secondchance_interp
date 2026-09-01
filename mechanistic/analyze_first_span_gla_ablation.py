from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


LETTERS = "ABCD"


def _stratified_interval(
    values: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata)
    valid = np.isfinite(values)
    values, strata = values[valid], strata[valid]
    if not len(values):
        return {"n": 0, "mean": None, "ci": [None, None]}
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    boot = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = np.concatenate(
            [group[rng.integers(0, len(group), size=len(group))] for group in groups]
        )
        boot[draw] = values[sampled].mean()
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(boot, [0.025, 0.975]).tolist(),
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _entropy(values: np.ndarray) -> np.ndarray:
    probabilities = _softmax(values)
    return -(probabilities * np.log2(np.clip(probabilities, 1e-12, None))).sum(-1)


def _align(values: np.ndarray, qids: list[str], plan: dict[str, dict]) -> np.ndarray:
    output = np.empty_like(values)
    for qi, qid in enumerate(qids):
        mapping = plan[qid]["original_to_new"]
        for content_index, content in enumerate(LETTERS):
            letter_index = LETTERS.index(mapping[content])
            output[..., qi, content_index] = values[..., qi, letter_index]
    return output


def _semantic_answers(
    raw_values: np.ndarray, qids: list[str], plan: dict[str, dict]
) -> np.ndarray:
    displayed = raw_values.argmax(axis=-1)
    output = np.empty_like(displayed)
    for qi, qid in enumerate(qids):
        mapping = plan[qid]["new_to_original"]
        for new_index, new_letter in enumerate(LETTERS):
            output[..., qi] = np.where(
                displayed[..., qi] == new_index,
                LETTERS.index(mapping[new_letter]),
                output[..., qi],
            )
    return output


def _choice_rates(answers: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> dict[str, float]:
    return {
        "w1": float(np.mean(answers == w1)),
        "w2": float(np.mean(answers == w2)),
        "other": float(np.mean((answers != w1) & (answers != w2))),
    }


def _analyze_split(
    root: Path,
    baseline: dict[str, dict],
    remapped: dict[str, dict],
    manifest: dict[str, dict],
    plan: dict[str, dict],
    historical_game: dict[str, dict],
    historical_neutral: dict[str, dict],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    arrays = dict(np.load(root / "results.npz", allow_pickle=False))
    if not np.all(arrays["completed"]):
        raise ValueError(f"Incomplete result set: {root}")
    qids = arrays["question_ids"].astype(str).tolist()
    scenarios = arrays["scenario_ids"].astype(str).tolist()
    natural = _align(arrays["natural_logits"], qids, plan)
    ablated = _align(arrays["ablated_logits"], qids, plan)
    w1 = np.asarray([LETTERS.index(baseline[qid]["answer"]) for qid in qids])
    w2 = np.asarray([
        LETTERS.index(remapped[qid]["answer_original_content"]) for qid in qids
    ])
    correct = np.asarray([
        LETTERS.index(manifest[qid]["correct_answer"]) for qid in qids
    ])
    discordant = w1 != w2
    selected = np.flatnonzero(discordant)
    strata = w1[discordant]
    row = np.arange(len(qids))

    natural_suppression = natural[1] - natural[0]
    natural_contrast = (
        natural_suppression[row, w1] - natural_suppression[row, w2]
    )
    natural_answers = _semantic_answers(arrays["natural_logits"], qids, plan)
    rng = np.random.default_rng(seed)
    natural_summary = {
        "targeting_contrast": _stratified_interval(
            natural_contrast[discordant], strata, rng, draws
        ),
        "game_choice_rates": _choice_rates(
            natural_answers[0, discordant], w1[discordant], w2[discordant]
        ),
        "neutral_choice_rates": _choice_rates(
            natural_answers[1, discordant], w1[discordant], w2[discordant]
        ),
        "game_accuracy": float(np.mean(natural_answers[0] == correct)),
        "neutral_accuracy": float(np.mean(natural_answers[1] == correct)),
        "game_entropy_bits": float(_entropy(natural[0]).mean()),
        "neutral_entropy_bits": float(_entropy(natural[1]).mean()),
    }

    scenario_rows = []
    for scenario_index, scenario in enumerate(scenarios):
        values = ablated[:, scenario_index]
        suppression = values[1] - values[0]
        contrast = suppression[row, w1] - suppression[row, w2]
        reduction = natural_contrast - contrast
        answers = _semantic_answers(
            arrays["ablated_logits"][:, scenario_index], qids, plan
        )
        game_w1_logit_change = values[0, row, w1] - natural[0, row, w1]
        neutral_w1_logit_change = values[1, row, w1] - natural[1, row, w1]
        game_w2_logit_change = values[0, row, w2] - natural[0, row, w2]
        neutral_w2_logit_change = values[1, row, w2] - natural[1, row, w2]
        game_margin_change = game_w1_logit_change - game_w2_logit_change
        neutral_margin_change = neutral_w1_logit_change - neutral_w2_logit_change
        game_rates = _choice_rates(
            answers[0, discordant], w1[discordant], w2[discordant]
        )
        neutral_rates = _choice_rates(
            answers[1, discordant], w1[discordant], w2[discordant]
        )
        scenario_rows.append(
            {
                "scenario": scenario,
                "ablated_targeting_contrast": _stratified_interval(
                    contrast[discordant], strata, rng, draws
                ),
                "reduction_in_targeting_contrast": _stratified_interval(
                    reduction[discordant], strata, rng, draws
                ),
                "game_w1_logit_change": _stratified_interval(
                    game_w1_logit_change[discordant], strata, rng, draws
                ),
                "neutral_w1_logit_change": _stratified_interval(
                    neutral_w1_logit_change[discordant], strata, rng, draws
                ),
                "game_w2_logit_change": _stratified_interval(
                    game_w2_logit_change[discordant], strata, rng, draws
                ),
                "neutral_w2_logit_change": _stratified_interval(
                    neutral_w2_logit_change[discordant], strata, rng, draws
                ),
                "game_w1_vs_w2_margin_change": _stratified_interval(
                    game_margin_change[discordant], strata, rng, draws
                ),
                "neutral_w1_vs_w2_margin_change": _stratified_interval(
                    neutral_margin_change[discordant], strata, rng, draws
                ),
                "game_choice_rates": game_rates,
                "neutral_choice_rates": neutral_rates,
                "game_w1_choice_rate_change": float(
                    game_rates["w1"] - natural_summary["game_choice_rates"]["w1"]
                ),
                "neutral_w1_choice_rate_change": float(
                    neutral_rates["w1"] - natural_summary["neutral_choice_rates"]["w1"]
                ),
                "game_accuracy_change": float(
                    np.mean(answers[0] == correct) - natural_summary["game_accuracy"]
                ),
                "neutral_accuracy_change": float(
                    np.mean(answers[1] == correct) - natural_summary["neutral_accuracy"]
                ),
                "game_entropy_change_bits": float(
                    _entropy(values[0]).mean() - natural_summary["game_entropy_bits"]
                ),
                "neutral_entropy_change_bits": float(
                    _entropy(values[1]).mean() - natural_summary["neutral_entropy_bits"]
                ),
            }
        )

    raw_natural = arrays["natural_logits"]
    validation = {}
    for condition_index, (condition, historical) in enumerate(
        (("game", historical_game), ("neutral", historical_neutral))
    ):
        matches = sum(
            int(np.argmax(raw_natural[condition_index, qi]) == LETTERS.index(
                historical[qid]["aggregated_ad_answer_new_letter"]
            ))
            for qi, qid in enumerate(qids)
        )
        validation[condition] = {
            "aggregated_ad_winner_matches_historical": int(matches),
            "n": len(qids),
            "rate": float(matches / len(qids)),
        }

    return {
        "root": str(root),
        "n": len(qids),
        "n_discordant_w1_w2": int(discordant.sum()),
        "natural": natural_summary,
        "scenarios": scenario_rows,
        "historical_validation": validation,
    }


def _fmt(value: dict[str, Any], scale: float = 1.0) -> str:
    return (
        f"{value['mean'] * scale:+.3f} "
        f"[{value['ci'][0] * scale:+.3f}, {value['ci'][1] * scale:+.3f}]"
    )


def analyze(
    discovery_root: Path,
    confirmation_root: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
    manifest_path: Path,
    mapping_plan_path: Path,
    historical_root: Path,
    output: Path,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped = json.loads(remapped_baseline_path.read_text())["results"]
    manifest = {
        row["id"]: row for row in json.loads(manifest_path.read_text())["questions"]
    }
    plan = {
        row["question_id"]: row
        for row in json.loads(mapping_plan_path.read_text())["rows"]
    }
    historical_game = json.loads(
        (historical_root / "incorrect_results.json").read_text()
    )["results"]
    historical_neutral = json.loads(
        (historical_root / "neutral_results.json").read_text()
    )["results"]
    discovery = _analyze_split(
        discovery_root, baseline, remapped, manifest, plan,
        historical_game, historical_neutral, draws, seed,
    )
    confirmation = _analyze_split(
        confirmation_root, baseline, remapped, manifest, plan,
        historical_game, historical_neutral, draws, seed + 1,
    )
    summary = {
        "definitions": {
            "targeting_contrast": (
                "(Neutral-Game suppression of W1) minus (Neutral-Game suppression of W2)"
            ),
            "reduction_in_targeting_contrast": (
                "Natural targeting contrast minus ablated targeting contrast; positive "
                "means the source writes are necessary for preferential W1 targeting."
            ),
        },
        "discovery": discovery,
        "confirmation": confirmation,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# First-presentation GLA-memory ablation",
        "",
        "The intervention removes selected first-presentation writes from all 48 GLA layers while preserving each target's exact historical four-question cohort.",
        "",
        "Discrete answers resolve exact ties in displayed A-D order before mapping the winning letter back to semantic content. Continuous quantities are invariant to this tie rule.",
        "",
        "## Natural behavior",
        "",
        f"Discovery discordant W1/W2 questions: **{discovery['n_discordant_w1_w2']}**; natural targeting contrast: **{_fmt(discovery['natural']['targeting_contrast'])} logits**.",
        f"Confirmation discordant questions: **{confirmation['n_discordant_w1_w2']}**; natural targeting contrast: **{_fmt(confirmation['natural']['targeting_contrast'])} logits**.",
        "",
        "## Frozen confirmation",
        "",
        "The most interpretable direct outcome is the change in the W1-minus-W2 margin within each condition. Positive values mean the lesion makes the model more likely to retain the semantic answer it reached on the first presentation rather than choose the answer it would have reached by freshly solving the remapped presentation.",
        "",
        "| Source writes removed | Game W1-W2 margin | Game W1 choice | Neutral W1-W2 margin | Neutral W1 choice | Difference-in-differences reduction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "first_question_content": "Question + options",
        "first_options": "Options only",
        "first_answer_boundary": "First-answer boundary",
        "content_plus_answer_boundary": "Question/options + boundary",
    }
    for row in confirmation["scenarios"]:
        lines.append(
            f"| {labels[row['scenario']]} | "
            f"{_fmt(row['game_w1_vs_w2_margin_change'])} | "
            f"{row['game_w1_choice_rate_change']:+.1%} | "
            f"{_fmt(row['neutral_w1_vs_w2_margin_change'])} | "
            f"{row['neutral_w1_choice_rate_change']:+.1%} | "
            f"{_fmt(row['reduction_in_targeting_contrast'])} |"
        )
    lines.extend(
        [
            "",
            "All margin changes and choice-rate changes are lesion minus natural within the named condition. The difference-in-differences column is secondary because it can change either by weakening Game suppression or by weakening Neutral retention.",
            "",
            "## Interpretation",
            "",
            "The first-answer-boundary lesion is the clean evidence for the route that carries the prior semantic decision into Game. On frozen confirmation it raises Game's W1-minus-W2 margin and W1 selection, while leaving Neutral's W1-minus-W2 margin approximately unchanged. Thus, recurrent GLA writes made while processing the empty first assistant boundary preserve information that the later incorrect-feedback computation uses to disfavor the semantic answer reached on the first presentation.",
            "",
            "The option-token lesion has a different role: it lowers the W1-minus-W2 margin in both conditions, especially Neutral. Those writes primarily support retaining/reconstructing the first answer, rather than implementing Game-specific suppression. The combined lesion therefore should not be described as a unitary suppression mechanism even though it has the largest difference-in-differences effect.",
            "",
            "## Historical-run validation",
            "",
            f"Discovery natural A–D winners matched the saved run on {discovery['historical_validation']['game']['aggregated_ad_winner_matches_historical']}/{discovery['n']} Game and {discovery['historical_validation']['neutral']['aggregated_ad_winner_matches_historical']}/{discovery['n']} Neutral questions.",
            f"Confirmation matched on {confirmation['historical_validation']['game']['aggregated_ad_winner_matches_historical']}/{confirmation['n']} Game and {confirmation['historical_validation']['neutral']['aggregated_ad_winner_matches_historical']}/{confirmation['n']} Neutral questions.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    _plot(summary, output / "first_span_gla_ablation.png")
    return summary


def _plot(summary: dict[str, Any], path: Path) -> None:
    labels = ("Question +\noptions", "Options\nonly", "Answer\nboundary", "Content +\nboundary")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, split_name, title in (
        (axes[0], "discovery", "A  Discovery"),
        (axes[1], "confirmation", "B  Frozen confirmation"),
    ):
        rows = summary[split_name]["scenarios"]
        x = np.arange(len(rows))
        for offset, key, label, color in (
            (-0.08, "game_w1_vs_w2_margin_change", "Game", "#348ce8"),
            (+0.08, "neutral_w1_vs_w2_margin_change", "Neutral", "#ed7d31"),
        ):
            means = np.asarray([row[key]["mean"] for row in rows])
            low = np.asarray([row[key]["ci"][0] for row in rows])
            high = np.asarray([row[key]["ci"][1] for row in rows])
            ax.errorbar(
                x + offset, means, yerr=np.vstack([means - low, high - means]),
                fmt="o", linestyle="none", capsize=4, color=color, label=label,
            )
        ax.axhline(0, color="#555", linewidth=1)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Change in W1 minus W2 margin (logits)")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.suptitle("What do first-presentation GLA-memory lesions do within each condition?")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping-plan", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    analyze(
        args.discovery, args.confirmation, args.baseline,
        args.remapped_baseline, args.manifest, args.mapping_plan,
        args.historical_root, args.output, args.draws, args.seed,
    )


if __name__ == "__main__":
    main()
