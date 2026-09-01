from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon

from .io import shard_path
from .run_decision_semantic_ablation import SCENARIOS


LETTERS = "ABCD"


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _entropy_bits(probabilities: np.ndarray) -> np.ndarray:
    return -(probabilities * np.log2(np.clip(probabilities, 1e-12, None))).sum(axis=-1)


def _bootstrap(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "ci": [None, None]}
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[samples].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def _bootstrap_stat(
    n: int,
    statistic: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    draws: int,
) -> dict:
    observed = np.arange(n)
    values = np.asarray(
        [statistic(rng.integers(0, n, n)) for _ in range(draws)], dtype=float
    )
    return {
        "n": int(n),
        "mean": float(statistic(observed)),
        "ci": np.quantile(values, [0.025, 0.975]).tolist(),
    }


def _load_group(roots: list[Path], group: str, qids_by_root: list[list[str]]) -> np.ndarray:
    rows = []
    for root, qids in zip(roots, qids_by_root):
        rows.extend(
            np.load(shard_path(root, group, qid), allow_pickle=False)[
                "final_canonical_logits"
            ].astype(np.float64)
            for qid in qids
        )
    return np.asarray(rows)


def _paired_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.allclose(values, 0):
        return 1.0
    return float(wilcoxon(values).pvalue)


def _rate_test(successes: int, n: int, chance: float = 1 / 3) -> dict:
    if n == 0:
        return {"n": 0, "successes": 0, "rate": None, "exact_one_sided_p": None}
    return {
        "n": int(n),
        "successes": int(successes),
        "rate": float(successes / n),
        "exact_one_sided_p": float(
            binomtest(successes, n, chance, alternative="greater").pvalue
        ),
    }


def _scenario_metrics(
    logits: np.ndarray,
    baseline_logits: np.ndarray,
    baseline_winner: np.ndarray,
    baseline_runner: np.ndarray,
    correct: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.arange(len(logits))
    answers = logits.argmax(axis=-1)
    probabilities = _softmax(logits)
    centered = logits - logits.mean(axis=-1, keepdims=True)
    baseline_centered = baseline_logits - baseline_logits.mean(axis=-1, keepdims=True)
    winner_values = centered[rows, baseline_winner]
    other_mean = (centered.sum(axis=-1) - winner_values) / 3
    return {
        "answer": answers,
        "switch": (answers != baseline_winner).astype(float),
        "correct": (answers == correct).astype(float),
        "probabilities": probabilities,
        "winner_probability": probabilities[rows, baseline_winner],
        "runner_probability": probabilities[rows, baseline_runner],
        "winner_contrast": winner_values - other_mean,
        "entropy_bits": _entropy_bits(probabilities),
        "centered_logit_delta_from_baseline": centered - baseline_centered,
    }


def _rank_values(values: np.ndarray, rank_order: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, rank_order, axis=1)


def _rank_summary(values: np.ndarray, rng: np.random.Generator, draws: int) -> list[dict]:
    return [
        {"rank": rank + 1, **_bootstrap(values[:, rank], rng, draws)}
        for rank in range(4)
    ]


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _fmt_signed(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:+.{digits}f}"


def _plot(summary: dict, output: Path) -> None:
    labels = {
        "natural": "Natural",
        "erase_winner_semantic": "Erase winner",
        "erase_runner_semantic": "Erase runner",
        "erase_all_option_semantics": "Erase all option content",
        "orthogonal_winner_matched": "Orthogonal control",
    }
    scenarios = list(SCENARIOS)
    x = np.arange(len(scenarios))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.3))
    game = [summary["scenarios"][s]["game"]["change_rate"] for s in scenarios]
    neutral = [summary["scenarios"][s]["neutral"]["change_rate"] for s in scenarios]
    axes[0].bar(x - width / 2, game, width, color="#2f8df3", label="Game")
    axes[0].bar(x + width / 2, neutral, width, color="#f08233", label="Neutral")
    axes[0].set_xticks(x, [labels[s] for s in scenarios], rotation=24, ha="right")
    axes[0].set_ylabel("Change rate")
    axes[0].set_title("A  Behavioral effects", loc="left")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    primary = summary["causal_effects"]["erase_winner_semantic"]
    rank_x = np.arange(1, 5)
    for offset, condition, color in ((-0.07, "game", "#2f8df3"), (0.07, "neutral", "#f08233")):
        rows = primary[condition]["centered_logit_effect_by_baseline_rank"]
        means = np.asarray([row["mean"] for row in rows])
        ci = np.asarray([row["ci"] for row in rows])
        axes[1].errorbar(
            rank_x + offset,
            means,
            yerr=np.vstack([means - ci[:, 0], ci[:, 1] - means]),
            fmt="o",
            capsize=3,
            color=color,
            label=condition.capitalize(),
        )
    axes[1].axhline(0, color="#555555", lw=1)
    axes[1].set_xticks(rank_x, ["Winner", "Runner-up", "Rank 3", "Rank 4"])
    axes[1].set_ylabel("Intervention − natural centered logit")
    axes[1].set_title("B  Effect of erasing winner semantics", loc="left")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("First-answer decision-position semantic ablation", fontsize=17)
    fig.tight_layout()
    fig.savefig(output / "decision_semantic_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(
    roots: list[Path],
    baseline_results: Path,
    output: Path,
    draws: int,
    seed: int,
) -> dict:
    metadata = [json.loads((root / "run_metadata.json").read_text()) for root in roots]
    qids_by_root = [list(row["question_ids"]) for row in metadata]
    qids = [qid for group in qids_by_root for qid in group]
    if len(qids) != len(set(qids)):
        raise ValueError("Result roots contain overlapping question IDs")
    baseline_payload = json.loads(baseline_results.read_text())["results"]
    if not set(qids) <= set(baseline_payload):
        raise ValueError("Baseline results are missing intervention questions")
    baseline_logits = np.asarray(
        [baseline_payload[qid]["aggregated_ad_logits"] for qid in qids], dtype=float
    )
    baseline_winner = baseline_logits.argmax(axis=-1)
    rank_order = np.argsort(-baseline_logits, axis=-1, kind="stable")
    baseline_runner = rank_order[:, 1]
    correct = np.asarray(
        [LETTERS.index(baseline_payload[qid]["correct_answer"]) for qid in qids]
    )
    baseline_probs = _softmax(baseline_logits)
    baseline_entropy = _entropy_bits(baseline_probs)
    baseline_accuracy = float((baseline_winner == correct).mean())

    logits = {
        condition: {
            scenario: _load_group(roots, f"{condition}_{scenario}", qids_by_root)
            for scenario in SCENARIOS
        }
        for condition in ("game", "neutral")
    }
    metrics = {
        condition: {
            scenario: _scenario_metrics(
                values,
                baseline_logits,
                baseline_winner,
                baseline_runner,
                correct,
            )
            for scenario, values in rows.items()
        }
        for condition, rows in logits.items()
    }
    rng = np.random.default_rng(seed)
    summary: dict = {
        "n": len(qids),
        "question_ids": qids,
        "baseline_accuracy": baseline_accuracy,
        "baseline_entropy_bits": _bootstrap(baseline_entropy, rng, draws),
        "vector_definition": metadata[0]["vector_definition"],
        "selected_post_block_readouts": metadata[0]["selected_post_block_readouts"],
        "scenarios": {},
        "causal_effects": {},
    }
    trial_rows = []
    for scenario in SCENARIOS:
        scenario_row = {}
        for condition in ("game", "neutral"):
            row = metrics[condition][scenario]
            changed_wrong = (
                (baseline_winner != correct) & row["switch"].astype(bool)
            )
            acc_hits = int(row["correct"][changed_wrong].sum())
            second_hits = int(
                (row["answer"][row["switch"].astype(bool)]
                 == baseline_runner[row["switch"].astype(bool)]).sum()
            )
            second_n = int(row["switch"].sum())
            entropy_delta = row["entropy_bits"] - baseline_entropy
            scenario_row[condition] = {
                "accuracy": float(row["correct"].mean()),
                "change_rate": float(row["switch"].mean()),
                "winner_probability": _bootstrap(row["winner_probability"], rng, draws),
                "runner_probability": _bootstrap(row["runner_probability"], rng, draws),
                "winner_contrast": _bootstrap(row["winner_contrast"], rng, draws),
                "entropy_bits": _bootstrap(row["entropy_bits"], rng, draws),
                "entropy_minus_baseline": {
                    **_bootstrap(entropy_delta, rng, draws),
                    "wilcoxon_p": _paired_wilcoxon(entropy_delta),
                },
                "accincor_changed_baseline_wrong": _rate_test(
                    acc_hits, int(changed_wrong.sum())
                ),
                "second_choice_among_changed": _rate_test(second_hits, second_n),
                "centered_logit_delta_from_baseline_by_rank": _rank_summary(
                    _rank_values(
                        row["centered_logit_delta_from_baseline"], rank_order
                    ),
                    rng,
                    draws,
                ),
            }
        game_switch = metrics["game"][scenario]["switch"].astype(bool)
        neutral_switch = metrics["neutral"][scenario]["switch"].astype(bool)
        game_only = int((game_switch & ~neutral_switch).sum())
        neutral_only = int((neutral_switch & ~game_switch).sum())
        discordant = game_only + neutral_only
        game_rate = float(game_switch.mean())
        neutral_rate = float(neutral_switch.mean())
        scenario_row["lift"] = {
            "absolute": game_rate - neutral_rate,
            "normalized": (game_rate - neutral_rate) / max(1 - neutral_rate, 1e-12),
            "game_only": game_only,
            "neutral_only": neutral_only,
            "mcnemar_exact_p": float(
                binomtest(game_only, discordant, 0.5).pvalue if discordant else 1.0
            ),
        }
        game_acc = scenario_row["game"]["accincor_changed_baseline_wrong"]
        game_second = scenario_row["game"]["second_choice_among_changed"]
        game_entropy_delta = scenario_row["game"]["entropy_minus_baseline"]
        scenario_row["paper_style_tests"] = {
            "Lift": bool(
                scenario_row["lift"]["absolute"] > 0
                and scenario_row["lift"]["mcnemar_exact_p"] < 0.05
            ),
            "AccIncor": bool(
                game_acc["rate"] is not None
                and game_acc["rate"] > 1 / 3
                and game_acc["exact_one_sided_p"] < 0.05
            ),
            "SecChoice": bool(
                game_second["rate"] is not None
                and game_second["rate"] > 1 / 3
                and game_second["exact_one_sided_p"] < 0.05
            ),
            "NoEntInc": not bool(
                game_entropy_delta["mean"] > 0
                and game_entropy_delta["wilcoxon_p"] < 0.05
            ),
        }
        summary["scenarios"][scenario] = scenario_row

    for scenario in SCENARIOS[1:]:
        causal_row = {}
        for condition in ("game", "neutral"):
            natural = metrics[condition]["natural"]
            intervened = metrics[condition][scenario]
            rank_effect = _rank_values(
                (logits[condition][scenario] - logits[condition][scenario].mean(axis=1, keepdims=True))
                - (logits[condition]["natural"] - logits[condition]["natural"].mean(axis=1, keepdims=True)),
                rank_order,
            )
            causal_row[condition] = {
                "answer_changed_from_natural": float(
                    (intervened["answer"] != natural["answer"]).mean()
                ),
                "switch_effect": _bootstrap(
                    intervened["switch"] - natural["switch"], rng, draws
                ),
                "accuracy_effect": _bootstrap(
                    intervened["correct"] - natural["correct"], rng, draws
                ),
                "winner_probability_effect": _bootstrap(
                    intervened["winner_probability"] - natural["winner_probability"],
                    rng,
                    draws,
                ),
                "runner_probability_effect": _bootstrap(
                    intervened["runner_probability"] - natural["runner_probability"],
                    rng,
                    draws,
                ),
                "winner_contrast_effect": _bootstrap(
                    intervened["winner_contrast"] - natural["winner_contrast"],
                    rng,
                    draws,
                ),
                "entropy_effect": _bootstrap(
                    intervened["entropy_bits"] - natural["entropy_bits"], rng, draws
                ),
                "centered_logit_effect_by_baseline_rank": _rank_summary(
                    rank_effect, rng, draws
                ),
            }
        summary["causal_effects"][scenario] = causal_row

    # A norm-matched intervention can itself perturb an answer near a decision
    # boundary.  The scientifically relevant specificity contrast is therefore
    # semantic erasure versus the orthogonal intervention in the same question,
    # not semantic erasure versus natural execution alone.
    summary["control_adjusted_effects"] = {}
    control_scenario = "orthogonal_winner_matched"
    for scenario in (
        "erase_winner_semantic",
        "erase_runner_semantic",
        "erase_all_option_semantics",
    ):
        control_row = {}
        for condition in ("game", "neutral"):
            intervened = metrics[condition][scenario]
            control = metrics[condition][control_scenario]
            rank_effect = _rank_values(
                (
                    logits[condition][scenario]
                    - logits[condition][scenario].mean(axis=1, keepdims=True)
                )
                - (
                    logits[condition][control_scenario]
                    - logits[condition][control_scenario].mean(axis=1, keepdims=True)
                ),
                rank_order,
            )
            control_row[condition] = {
                "switch_effect": _bootstrap(
                    intervened["switch"] - control["switch"], rng, draws
                ),
                "accuracy_effect": _bootstrap(
                    intervened["correct"] - control["correct"], rng, draws
                ),
                "winner_probability_effect": _bootstrap(
                    intervened["winner_probability"] - control["winner_probability"],
                    rng,
                    draws,
                ),
                "runner_probability_effect": _bootstrap(
                    intervened["runner_probability"] - control["runner_probability"],
                    rng,
                    draws,
                ),
                "winner_contrast_effect": _bootstrap(
                    intervened["winner_contrast"] - control["winner_contrast"],
                    rng,
                    draws,
                ),
                "entropy_effect": _bootstrap(
                    intervened["entropy_bits"] - control["entropy_bits"], rng, draws
                ),
                "centered_logit_effect_by_baseline_rank": _rank_summary(
                    rank_effect, rng, draws
                ),
            }
        summary["control_adjusted_effects"][scenario] = control_row

    for qi, qid in enumerate(qids):
        row = {
            "question_id": qid,
            "baseline_answer": LETTERS[baseline_winner[qi]],
            "baseline_runner": LETTERS[baseline_runner[qi]],
            "correct_answer": LETTERS[correct[qi]],
        }
        for condition in ("game", "neutral"):
            for scenario in SCENARIOS:
                values = metrics[condition][scenario]
                stem = f"{condition}_{scenario}"
                row[f"{stem}_answer"] = LETTERS[int(values["answer"][qi])]
                row[f"{stem}_changed"] = int(values["switch"][qi])
                row[f"{stem}_correct"] = int(values["correct"][qi])
                row[f"{stem}_entropy_bits"] = float(values["entropy_bits"][qi])
        trial_rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "trial_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)

    lines = [
        "# First-answer decision-position semantic ablation",
        "",
        f"Questions: **{len(qids)}**. Baseline accuracy: **{baseline_accuracy:.1%}**. ",
        f"The question-specific semantic projection was removed continuously at "
        f"post-block readouts **{metadata[0]['selected_post_block_readouts'][0]}--"
        f"{metadata[0]['selected_post_block_readouts'][-1]}**.",
        "",
        "## Complete Second Chance battery",
        "",
        "| Scenario | Game change | Neutral change | Normalized lift | Game AccIncor | Game second choice | Game entropy − Baseline | Passes (L/A/S/E) |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    labels = {
        "natural": "Natural",
        "erase_winner_semantic": "Erase winner semantics",
        "erase_runner_semantic": "Erase runner semantics",
        "erase_all_option_semantics": "Erase all option semantics",
        "orthogonal_winner_matched": "Norm-matched orthogonal control",
    }
    for scenario in SCENARIOS:
        row = summary["scenarios"][scenario]
        acc = row["game"]["accincor_changed_baseline_wrong"]
        second = row["game"]["second_choice_among_changed"]
        entropy = row["game"]["entropy_minus_baseline"]
        tests = row["paper_style_tests"]
        passes = "/".join("✓" if tests[key] else "X" for key in ("Lift", "AccIncor", "SecChoice", "NoEntInc"))
        lines.append(
            f"| {labels[scenario]} | {_fmt_pct(row['game']['change_rate'])} | "
            f"{_fmt_pct(row['neutral']['change_rate'])} | {row['lift']['normalized']:.3f} | "
            f"{acc['successes']}/{acc['n']} = {_fmt_pct(acc['rate'])} | "
            f"{second['successes']}/{second['n']} = {_fmt_pct(second['rate'])} | "
            f"{entropy['mean']:+.3f} bits | {passes} |"
        )
    lines.extend(
        [
            "",
            "AccIncor is conditioned on Baseline-wrong trials that changed and is tested against 1/3. Second choice is conditioned on changed trials and is tested against 1/3. Entropy is the exact normalized A--D entropy from the four captured canonical logits.",
            "",
            "## Paired causal effects relative to natural execution",
            "",
            "| Intervention | Game switch | Neutral switch | Game winner probability | Game runner probability | Game entropy | Answers changed from natural (G/N) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS[1:]:
        row = summary["causal_effects"][scenario]
        lines.append(
            f"| {labels[scenario]} | {_fmt_signed(100*row['game']['switch_effect']['mean'])} pp | "
            f"{_fmt_signed(100*row['neutral']['switch_effect']['mean'])} pp | "
            f"{_fmt_signed(row['game']['winner_probability_effect']['mean'], 3)} | "
            f"{_fmt_signed(row['game']['runner_probability_effect']['mean'], 3)} | "
            f"{_fmt_signed(row['game']['entropy_effect']['mean'], 3)} bits | "
            f"{_fmt_pct(row['game']['answer_changed_from_natural'])} / "
            f"{_fmt_pct(row['neutral']['answer_changed_from_natural'])} |"
        )
    lines.extend(
        [
            "",
            "## Semantic specificity relative to the norm-matched orthogonal control",
            "",
            "This paired contrast asks whether removing the intended semantic direction does more than removing an equally large, question-specific orthogonal direction. The interval is a paired question-level 95% bootstrap confidence interval.",
            "",
            "| Intervention | Game switch vs control | Neutral switch vs control | Game winner contrast vs control |",
            "|---|---:|---:|---:|",
        ]
    )
    for scenario in (
        "erase_winner_semantic",
        "erase_runner_semantic",
        "erase_all_option_semantics",
    ):
        row = summary["control_adjusted_effects"][scenario]
        game_switch = row["game"]["switch_effect"]
        neutral_switch = row["neutral"]["switch_effect"]
        game_winner = row["game"]["winner_contrast_effect"]
        lines.append(
            f"| {labels[scenario]} | "
            f"{_fmt_signed(100*game_switch['mean'])} pp "
            f"[{_fmt_signed(100*game_switch['ci'][0])}, {_fmt_signed(100*game_switch['ci'][1])}] | "
            f"{_fmt_signed(100*neutral_switch['mean'])} pp "
            f"[{_fmt_signed(100*neutral_switch['ci'][0])}, {_fmt_signed(100*neutral_switch['ci'][1])}] | "
            f"{_fmt_signed(game_winner['mean'], 3)} "
            f"[{_fmt_signed(game_winner['ci'][0], 3)}, {_fmt_signed(game_winner['ci'][1], 3)}] |"
        )
    primary = summary["causal_effects"]["erase_winner_semantic"]
    lines.extend(
        [
            "",
            "## Rank-resolved primary effect",
            "",
            "Centered-logit change caused by winner-semantic erasure, relative to the matched natural execution:",
            "",
            "| Condition | Winner | Runner-up | Rank 3 | Rank 4 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition in ("game", "neutral"):
        values = primary[condition]["centered_logit_effect_by_baseline_rank"]
        lines.append(
            f"| {condition.capitalize()} | "
            + " | ".join(f"{row['mean']:+.3f}" for row in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "The layer band was selected after inspecting SimpleMC representations, so this is exploratory. A positive result requires replication on a fresh dataset.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    _plot(summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    analyze(args.results, args.baseline_results, args.output, args.draws, args.seed)


if __name__ == "__main__":
    main()
