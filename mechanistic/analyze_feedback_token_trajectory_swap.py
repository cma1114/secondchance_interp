from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analyze_baseline_mixer_function import _bootstrap_indices, _summary
from .data import load_activation_dataset
from .run_feedback_token_trajectory_swap import SCENARIOS


DIRECTIONS = ("Neutral into Game", "Game into Neutral")
TOKENS_GAME = ("Your", "answer", "was", "incorrect", ".", "Choose", "a", "different", "answer", ".")
TOKENS_NEUTRAL = ("Your", "answer", "was", "lost", ".", "Choose", "the", "answer", "again", ".")
DISPLAY_LABELS = (
    "1 Your↔Your",
    "2 answer↔answer",
    "3 was↔was",
    "4 incorrect↔lost",
    "5 .↔. (evaluation end)",
    "6 Choose↔Choose",
    "7 a↔the",
    "8 different↔answer",
    "9 answer↔again",
    "10 .↔. (feedback end)",
    "Evaluation clause",
    "Action clause",
    "Full feedback",
)


def _entropy_bits(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(centered)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -np.sum(
        probabilities * np.log2(np.maximum(probabilities, 1e-300)), axis=-1
    )


def _spread(logits: np.ndarray) -> np.ndarray:
    return np.std(logits - logits.mean(axis=-1, keepdims=True), axis=-1)


def _winner_advantage(logits: np.ndarray, order: np.ndarray) -> np.ndarray:
    aligned = np.take_along_axis(logits, order, axis=-1)
    return aligned[..., 0] - aligned[..., 1:].mean(axis=-1)


def _ci(values: np.ndarray, bootstrap: np.ndarray) -> dict:
    point, low, high = _summary(values, bootstrap)
    return {
        "estimate": float(point),
        "ci": [float(low), float(high)],
    }


def _plot(path: Path, effects: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.2), sharex=True)
    x = np.arange(len(SCENARIOS))
    panels = (
        (axes[0, 0], DIRECTIONS[0], "switch_rate_change_pp", "A  Neutral states into Game", "Change in switch rate (pp)"),
        (axes[0, 1], DIRECTIONS[1], "switch_rate_change_pp", "B  Game states into Neutral", "Change in switch rate (pp)"),
        (axes[1, 0], DIRECTIONS[0], "winner_advantage_change", "C  Neutral states into Game", "Change in Baseline-winner advantage"),
        (axes[1, 1], DIRECTIONS[1], "winner_advantage_change", "D  Game states into Neutral", "Change in Baseline-winner advantage"),
    )
    colors = ("#1689d8", "#e66b19", "#555555", "#555555")
    for (axis, direction, metric, title, ylabel), color in zip(panels, colors):
        records = [effects[direction][name][metric] for name in SCENARIOS]
        point = np.asarray([record["estimate"] for record in records])
        low = np.asarray([record["ci"][0] for record in records])
        high = np.asarray([record["ci"][1] for record in records])
        axis.errorbar(
            x,
            point,
            yerr=np.stack((point - low, high - point)),
            fmt="o",
            markersize=5,
            linestyle="none",
            capsize=3,
            color=color,
        )
        axis.axhline(0, color="#666", lw=1, ls="--")
        axis.axvline(9.5, color="#999", lw=1, ls=":")
        axis.set_title(title, loc="left", weight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[1]:
        axis.set_xticks(x, DISPLAY_LABELS, rotation=58, ha="right")
    fig.suptitle(
        "Aligned feedback-token trajectory swaps — token-matched prompt test",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Each selected token position is clamped to its paired other-condition residual before every block; error bars are paired, Baseline-letter-stratified 95% bootstrap CIs.",
        ha="center",
        fontsize=10,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def analyze(
    results_path: Path,
    metadata_path: Path,
    baseline_root: Path,
    output: Path,
    figure: Path,
    draws: int,
    seed: int,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with np.load(results_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    if not np.all(data["completed"]):
        raise RuntimeError("Token-swap run is incomplete")
    qids = data["question_ids"].astype(str).tolist()
    baseline_dataset = load_activation_dataset(baseline_root, ["baseline"])
    baseline_index = {qid: index for index, qid in enumerate(baseline_dataset.question_ids)}
    if not set(qids) <= set(baseline_index):
        raise RuntimeError("A token-swap question is absent from Baseline")
    selected = [baseline_index[qid] for qid in qids]
    baseline = baseline_dataset.logits[selected, 0, -1].astype(float)
    correct = np.asarray(
        [
            "ABCD".index(baseline_dataset.metadata[(qid, "baseline")]["correct_answer"])
            for qid in qids
        ]
    )
    baseline_answer = baseline.argmax(axis=1)
    order = np.argsort(-baseline, axis=1, kind="stable")
    bootstrap = _bootstrap_indices(baseline_answer, draws, seed)

    natural = data["natural_logits"].astype(float)
    patched = data["patched_logits"].astype(float)
    natural_answer = natural.argmax(axis=2)
    patched_answer = patched.argmax(axis=3)
    natural_switch = natural_answer != baseline_answer[None, :]
    patched_switch = patched_answer != baseline_answer[None, None, :]
    natural_accuracy = natural_answer == correct[None, :]
    patched_accuracy = patched_answer == correct[None, None, :]

    order_natural = np.repeat(order[None, :, :], 2, axis=0)
    order_patched = np.repeat(order[None, None, :, :], 2, axis=0)
    order_patched = np.repeat(order_patched, len(SCENARIOS), axis=1)
    metrics = {
        "switch_rate_change_pp": 100.0 * (
            patched_switch.astype(float) - natural_switch[:, None, :].astype(float)
        ),
        "accuracy_change_pp": 100.0 * (
            patched_accuracy.astype(float) - natural_accuracy[:, None, :].astype(float)
        ),
        "winner_advantage_change": (
            _winner_advantage(patched, order_patched)
            - _winner_advantage(natural, order_natural)[:, None, :]
        ),
        "spread_change": _spread(patched) - _spread(natural)[:, None, :],
        "entropy_change_bits": (
            _entropy_bits(patched) - _entropy_bits(natural)[:, None, :]
        ),
    }
    natural_advantage = _winner_advantage(natural, order_natural)
    natural_spread = _spread(natural)
    natural_entropy = _entropy_bits(natural)

    effects: dict[str, dict] = {}
    for direction_index, direction in enumerate(DIRECTIONS):
        effects[direction] = {}
        for scenario_index, scenario in enumerate(SCENARIOS):
            record = {
                metric: _ci(values[direction_index, scenario_index], bootstrap)
                for metric, values in metrics.items()
            }
            changed = (
                patched_answer[direction_index, scenario_index]
                != natural_answer[direction_index]
            )
            record["choice_transitions"] = {
                "total_choice_flips": int(changed.sum()),
                "new_switches": int(
                    np.sum(
                        ~natural_switch[direction_index]
                        & patched_switch[direction_index, scenario_index]
                    )
                ),
                "prevented_switches": int(
                    np.sum(
                        natural_switch[direction_index]
                        & ~patched_switch[direction_index, scenario_index]
                    )
                ),
                "switch_to_other_alternative": int(
                    np.sum(
                        natural_switch[direction_index]
                        & patched_switch[direction_index, scenario_index]
                        & changed
                    )
                ),
            }
            delta = (
                patched[direction_index, scenario_index]
                - natural[direction_index]
            )
            delta -= delta.mean(axis=1, keepdims=True)
            ranked = np.take_along_axis(delta, order, axis=1)
            point, low, high = _summary(ranked, bootstrap)
            record["baseline_rank_logit_changes"] = {
                f"Rank {rank + 1}": {
                    "estimate": float(point[rank]),
                    "ci": [float(low[rank]), float(high[rank])],
                }
                for rank in range(4)
            }
            effects[direction][scenario] = record

    expected = {
        DIRECTIONS[0]: {
            "switch_rate_change_pp": 100.0 * np.mean(
                natural_switch[1].astype(float) - natural_switch[0].astype(float)
            ),
            "winner_advantage_change": np.mean(
                natural_advantage[1] - natural_advantage[0]
            ),
            "spread_change": np.mean(natural_spread[1] - natural_spread[0]),
            "entropy_change_bits": np.mean(natural_entropy[1] - natural_entropy[0]),
        },
        DIRECTIONS[1]: {
            "switch_rate_change_pp": 100.0 * np.mean(
                natural_switch[0].astype(float) - natural_switch[1].astype(float)
            ),
            "winner_advantage_change": np.mean(
                natural_advantage[0] - natural_advantage[1]
            ),
            "spread_change": np.mean(natural_spread[0] - natural_spread[1]),
            "entropy_change_bits": np.mean(natural_entropy[0] - natural_entropy[1]),
        },
    }
    full_index = SCENARIOS.index("full_feedback")
    full_swap_fidelity = {
        "Neutral_into_Game_max_abs_logit_error": float(
            np.max(np.abs(patched[0, full_index] - natural[1]))
        ),
        "Game_into_Neutral_max_abs_logit_error": float(
            np.max(np.abs(patched[1, full_index] - natural[0]))
        ),
        "Neutral_into_Game_same_batch_identity_error": float(
            np.max(np.abs(data["full_swap_same_batch_logit_error"][0]))
        ),
        "Game_into_Neutral_same_batch_identity_error": float(
            np.max(np.abs(data["full_swap_same_batch_logit_error"][1]))
        ),
    }
    state_distance = data["game_neutral_token_state_distance"].astype(float)
    summary = {
        "status": "prompt_variant_test_only",
        "n_questions": len(qids),
        "tokens": {
            "Game": list(TOKENS_GAME),
            "Neutral": list(TOKENS_NEUTRAL),
            "aligned_pairs": list(DISPLAY_LABELS[:10]),
        },
        "natural": {
            "Game_switch_rate": float(natural_switch[0].mean()),
            "Neutral_switch_rate": float(natural_switch[1].mean()),
            "Game_accuracy": float(natural_accuracy[0].mean()),
            "Neutral_accuracy": float(natural_accuracy[1].mean()),
            "Game_winner_advantage": float(natural_advantage[0].mean()),
            "Neutral_winner_advantage": float(natural_advantage[1].mean()),
            "Game_spread": float(natural_spread[0].mean()),
            "Neutral_spread": float(natural_spread[1].mean()),
            "Game_entropy_bits": float(natural_entropy[0].mean()),
            "Neutral_entropy_bits": float(natural_entropy[1].mean()),
        },
        "expected_complete_condition_moves": {
            direction: {metric: float(value) for metric, value in values.items()}
            for direction, values in expected.items()
        },
        "effects": effects,
        "full_feedback_swap_fidelity": full_swap_fidelity,
        "mean_token_state_distance_by_layer_and_position": state_distance.mean(axis=0).tolist(),
        "run_metadata": json.loads(metadata_path.read_text()),
    }
    summary["fraction_of_condition_gap"] = {
        direction: {
            scenario: {
                metric: (
                    float(effects[direction][scenario][metric]["estimate"] / expected[direction][metric])
                    if abs(expected[direction][metric]) > 1e-12
                    else None
                )
                for metric in expected[direction]
            }
            for scenario in SCENARIOS
        }
        for direction in DIRECTIONS
    }
    summary["sum_of_individual_token_effects"] = {
        direction: {
            metric: float(
                sum(
                    effects[direction][scenario][metric]["estimate"]
                    for scenario in SCENARIOS[:10]
                )
            )
            for metric in expected[direction]
        }
        for direction in DIRECTIONS
    }

    (output / "feedback_token_trajectory_swap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    _plot(figure, effects)

    def fmt(record: dict, metric: str, digits: int = 2) -> str:
        value = record[metric]
        return (
            f"{value['estimate']:+.{digits}f} "
            f"[{value['ci'][0]:+.{digits}f}, {value['ci'][1]:+.{digits}f}]"
        )

    lines = [
        "# Aligned feedback-token trajectory swaps",
        "",
        "This is a **separate token-matched prompt-variant experiment**. It is not pooled with the canonical standard-wording results.",
        "",
        f"Questions: **{len(qids)}**. Each selected feedback-token position was clamped to its paired same-question other-condition residual immediately before every model block.",
        "",
        "## Literal aligned tokens",
        "",
        "```text",
        "Game:    " + " | ".join(TOKENS_GAME),
        "Neutral: " + " | ".join(TOKENS_NEUTRAL),
        "```",
        "",
        "## Effects",
        "",
        "Switch-rate effects are percentage points. Winner-advantage and entropy effects are changes from the target condition's natural output.",
        "",
        "| Swap | N→G switch | G→N switch | N→G winner advantage | G→N winner advantage | N→G entropy (bits) | G→N entropy (bits) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, label in zip(SCENARIOS, DISPLAY_LABELS):
        ng = effects[DIRECTIONS[0]][scenario]
        gn = effects[DIRECTIONS[1]][scenario]
        lines.append(
            f"| {label} | {fmt(ng, 'switch_rate_change_pp')} | {fmt(gn, 'switch_rate_change_pp')} | "
            f"{fmt(ng, 'winner_advantage_change', 3)} | {fmt(gn, 'winner_advantage_change', 3)} | "
            f"{fmt(ng, 'entropy_change_bits', 3)} | {fmt(gn, 'entropy_change_bits', 3)} |"
        )
    lines.extend(
        [
            "",
            "The complete-feedback swap is an implementation sanity check: because the aligned prompts differ only within these ten tokens, swapping the complete feedback trajectory should reproduce the other condition.",
            "",
            "The individual-position interventions are carrier tests, not literal word substitutions: at every block, the selected position receives the state that naturally occurred at that position in the other condition. A large effect therefore means that position carries causally useful condition information; it does not mean the surface word alone is sufficient.",
            "",
            "See the machine-readable summary for spread, accuracy, rank-resolved logit changes, choice-transition counts, fractions of the full condition gap, and natural Game–Neutral state distances by layer and token position.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = analyze(
        args.results,
        args.metadata,
        args.baseline_root,
        args.output,
        args.figure,
        args.draws,
        args.seed,
    )
    print(json.dumps({key: value for key, value in summary.items() if key not in {"effects", "mean_token_state_distance_by_layer_and_position", "run_metadata"}}, indent=2))


if __name__ == "__main__":
    main()
