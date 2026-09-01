from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io import shard_path


SEMANTIC_CONTROL = {
    "erase_winner_continuous": "orthogonal_winner_matched",
    "erase_all_ad_continuous": "orthogonal_all_ad_matched",
}


def _load(root: Path, group: str, qids: list[str]) -> np.ndarray:
    return np.asarray([
        np.load(shard_path(root, group, qid), allow_pickle=False)[
            "final_canonical_logits"
        ].astype(np.float64)
        for qid in qids
    ])


def _softmax(values):
    exp = np.exp(values - values.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def _metrics(values, winners):
    rows = np.arange(len(values))
    probs = _softmax(values)
    selected = values[rows, winners]
    return {
        "switch": (values.argmax(axis=-1) != winners).astype(float),
        "winner_probability": probs[rows, winners],
        "winner_logit_contrast": selected - (values.sum(axis=-1) - selected) / 3,
        "ad_entropy": -(probs * np.log(probs.clip(1e-12))).sum(axis=-1),
    }


def _bootstrap(values, draws, rng):
    n = len(values)
    means = np.asarray([
        values[rng.integers(0, n, n)].mean() for _ in range(draws)
    ])
    return {
        "mean": float(values.mean()),
        "ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def analyze(results: Path, baseline_root: Path, output: Path, draws: int, seed: int):
    metadata = json.loads((results / "run_metadata.json").read_text())
    qids = metadata["question_ids"]
    baseline = _load(baseline_root, "baseline_natural", qids)
    winners = baseline.argmax(axis=-1)
    scenarios = [
        "natural",
        "erase_winner_continuous",
        "erase_all_ad_continuous",
        "orthogonal_winner_matched",
        "orthogonal_all_ad_matched",
    ]
    logits = {
        condition: {
            scenario: _load(results, f"{condition}_{scenario}", qids)
            for scenario in scenarios
        }
        for condition in ("game", "neutral")
    }
    metrics = {
        condition: {
            scenario: _metrics(values, winners)
            for scenario, values in condition_values.items()
        }
        for condition, condition_values in logits.items()
    }
    rng = np.random.default_rng(seed)
    summary = {
        "n": len(qids),
        "natural": {},
        "interventions": {},
    }
    for condition in ("game", "neutral"):
        summary["natural"][condition] = {
            "switch_rate": float(metrics[condition]["natural"]["switch"].mean())
        }
    for semantic, control in SEMANTIC_CONTROL.items():
        row = {"conditions": {}, "predicted_opposite_switch_signature": {}}
        effects = {}
        controls = {}
        for condition in ("game", "neutral"):
            natural = metrics[condition]["natural"]
            effects[condition] = {
                metric: metrics[condition][semantic][metric] - natural[metric]
                for metric in natural
            }
            controls[condition] = {
                metric: metrics[condition][control][metric] - natural[metric]
                for metric in natural
            }
            natural_answers = logits[condition]["natural"].argmax(axis=-1)
            semantic_answers = logits[condition][semantic].argmax(axis=-1)
            control_answers = logits[condition][control].argmax(axis=-1)
            natural_switch = natural_answers != winners
            semantic_switch = semantic_answers != winners
            row["conditions"][condition] = {
                "answer_changed_rate": float(
                    (semantic_answers != natural_answers).mean()
                ),
                "control_answer_changed_rate": float(
                    (control_answers != natural_answers).mean()
                ),
                "semantic_switch_transitions": {
                    "repeat_to_switch": int((~natural_switch & semantic_switch).sum()),
                    "switch_to_repeat": int((natural_switch & ~semantic_switch).sum()),
                    "switch_to_other_switch": int(
                        (natural_switch & semantic_switch
                         & (semantic_answers != natural_answers)).sum()
                    ),
                },
                "semantic_effect": {
                    metric: _bootstrap(values, draws, rng)
                    for metric, values in effects[condition].items()
                },
                "orthogonal_control_effect": {
                    metric: _bootstrap(values, draws, rng)
                    for metric, values in controls[condition].items()
                },
                "semantic_minus_control": {
                    metric: _bootstrap(
                        effects[condition][metric] - controls[condition][metric],
                        draws,
                        rng,
                    )
                    for metric in natural
                },
            }
        for label, source in (("semantic", effects), ("orthogonal_control", controls)):
            row["predicted_opposite_switch_signature"][label] = _bootstrap(
                source["neutral"]["switch"] - source["game"]["switch"],
                draws,
                rng,
            )
        semantic_specific = {
            condition: effects[condition]["switch"] - controls[condition]["switch"]
            for condition in ("game", "neutral")
        }
        row["predicted_opposite_switch_signature"]["semantic_minus_control"] = _bootstrap(
            semantic_specific["neutral"] - semantic_specific["game"],
            draws,
            rng,
        )
        summary["interventions"][semantic] = row

    output.mkdir(parents=True, exist_ok=True)
    (output / "continuous_historical_answer_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    winner = summary["interventions"]["erase_winner_continuous"]
    full = summary["interventions"]["erase_all_ad_continuous"]
    lines = [
        "# Continuous historical-answer identity ablation",
        "",
        f"Held-out SimpleMC questions: **{len(qids)}**. A-D identity was removed at",
        f"the historical first-answer endpoint before every Mixer from layers",
        f"**{metadata['first_user_facing_layer']}–{metadata['last_user_facing_layer']}**.",
        "",
        f"Natural switching: Game **{100*summary['natural']['game']['switch_rate']:.1f}%**; "
        f"Neutral **{100*summary['natural']['neutral']['switch_rate']:.1f}%**.",
        "",
        "| Continuous ablation | Game switch effect | Neutral switch effect | Neutral effect − Game effect | Answers changed (G/N) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in (("Baseline-winner direction", winner), ("Full centered A-D subspace", full)):
        game = item["conditions"]["game"]["semantic_effect"]["switch"]
        neutral = item["conditions"]["neutral"]["semantic_effect"]["switch"]
        signature = item["predicted_opposite_switch_signature"]["semantic"]
        lines.append(
            f"| {label} | {100*game['mean']:+.2f} pp | {100*neutral['mean']:+.2f} pp | "
            f"{100*signature['mean']:+.2f} pp [{100*signature['ci'][0]:+.2f}, {100*signature['ci'][1]:+.2f}] | "
            f"{100*item['conditions']['game']['answer_changed_rate']:.1f}% / "
            f"{100*item['conditions']['neutral']['answer_changed_rate']:.1f}% |"
        )
    lines.extend([
        "",
        "Positive Neutral-minus-Game means the predicted pattern: Neutral switches",
        "more and/or Game switches less after identity removal.",
        "",
        "## Effects relative to norm-matched orthogonal controls",
        "",
        "| Continuous ablation | Game switch | Neutral switch | Predicted opposite-condition signature | Game winner contrast | Neutral winner contrast |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for label, item in (("Baseline-winner direction", winner), ("Full centered A-D subspace", full)):
        game_switch = item["conditions"]["game"]["semantic_minus_control"]["switch"]
        neutral_switch = item["conditions"]["neutral"]["semantic_minus_control"]["switch"]
        signature = item["predicted_opposite_switch_signature"]["semantic_minus_control"]
        game_contrast = item["conditions"]["game"]["semantic_minus_control"]["winner_logit_contrast"]
        neutral_contrast = item["conditions"]["neutral"]["semantic_minus_control"]["winner_logit_contrast"]
        lines.append(
            f"| {label} | {100*game_switch['mean']:+.2f} pp "
            f"[{100*game_switch['ci'][0]:+.2f}, {100*game_switch['ci'][1]:+.2f}] | "
            f"{100*neutral_switch['mean']:+.2f} pp "
            f"[{100*neutral_switch['ci'][0]:+.2f}, {100*neutral_switch['ci'][1]:+.2f}] | "
            f"{100*signature['mean']:+.2f} pp "
            f"[{100*signature['ci'][0]:+.2f}, {100*signature['ci'][1]:+.2f}] | "
            f"{game_contrast['mean']:+.3f} | {neutral_contrast['mean']:+.3f} |"
        )
    lines.extend([
        "",
        "The semantic-minus-control contrast removes the effect of an equally large",
        "perturbation in a direction orthogonal to all centered A-D evidence. The JSON",
        "report additionally contains probability and entropy effects and the exact",
        "switch-transition counts.",
    ])
    (output / "CONTINUOUS_HISTORICAL_ANSWER_ABLATION_REPORT.md").write_text(
        "\n".join(lines) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(args.results, args.baseline_root, args.output, args.draws, args.seed)


if __name__ == "__main__":
    main()
