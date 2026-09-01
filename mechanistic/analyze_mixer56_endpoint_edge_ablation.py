from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io import shard_path
from .sublayer_config import SublayerExperimentConfig


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _bootstrap(values: np.ndarray, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(draws)
    for index in range(draws):
        means[index] = values[rng.integers(0, n, n)].mean()
    return {
        "mean": float(values.mean()),
        "ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def _load(root: Path, group: str, qids: list[str]) -> np.ndarray:
    values = []
    for qid in qids:
        with np.load(shard_path(root, group, qid), allow_pickle=False) as source:
            values.append(source["final_canonical_logits"].astype(np.float64))
    return np.asarray(values)


def _metrics(logits: np.ndarray, baseline: np.ndarray) -> dict[str, np.ndarray]:
    rows = np.arange(len(logits))
    probs = _softmax(logits)
    selected = logits[rows, baseline]
    contrast = selected - (logits.sum(axis=-1) - selected) / 3.0
    entropy = -(probs * np.log(probs.clip(1e-12))).sum(axis=-1)
    return {
        "switch": (logits.argmax(axis=-1) != baseline).astype(float),
        "winner_probability": probs[rows, baseline],
        "winner_logit_contrast": contrast,
        "ad_entropy": entropy,
    }


def analyze(
    config_path: Path,
    results: Path,
    baseline_root: Path,
    output: Path,
    bootstrap: int,
    seed: int,
) -> dict:
    config = SublayerExperimentConfig.load(config_path)
    metadata = json.loads((results / "run_metadata.json").read_text())
    qids = metadata["question_ids"]
    # Use the matched live self-hosted Baseline, not the older provider answer.
    # Those differ on 35/249 questions in this confirmation set.
    baseline_logits = _load(baseline_root, "natural_baseline", qids)
    baseline = baseline_logits.argmax(axis=-1)
    groups = {
        "game_natural": _load(results, "game_natural", qids),
        "game_ablated": _load(results, "game_endpoint_ablated", qids),
        "neutral_natural": _load(results, "neutral_natural", qids),
        "neutral_ablated": _load(results, "neutral_endpoint_ablated", qids),
    }
    metrics = {name: _metrics(values, baseline) for name, values in groups.items()}
    summary: dict = {"n": len(qids), "conditions": {}, "difference_in_differences": {}}
    for offset, condition in enumerate(("game", "neutral")):
        natural = metrics[f"{condition}_natural"]
        ablated = metrics[f"{condition}_ablated"]
        condition_summary = {
            "natural_switch_rate": float(natural["switch"].mean()),
            "ablated_switch_rate": float(ablated["switch"].mean()),
            "natural_answer_changed_by_ablation": float(
                (groups[f"{condition}_natural"].argmax(axis=-1)
                 != groups[f"{condition}_ablated"].argmax(axis=-1)).mean()
            ),
            "effects_ablated_minus_natural": {},
        }
        for metric in natural:
            condition_summary["effects_ablated_minus_natural"][metric] = _bootstrap(
                ablated[metric] - natural[metric], bootstrap, seed + offset * 20
            )
        summary["conditions"][condition] = condition_summary
    for index, metric in enumerate(metrics["game_natural"]):
        game_effect = metrics["game_ablated"][metric] - metrics["game_natural"][metric]
        neutral_effect = metrics["neutral_ablated"][metric] - metrics["neutral_natural"][metric]
        summary["difference_in_differences"][metric] = _bootstrap(
            game_effect - neutral_effect, bootstrap, seed + 100 + index
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "mixer56_endpoint_edge_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    game = summary["conditions"]["game"]
    neutral = summary["conditions"]["neutral"]
    did = summary["difference_in_differences"]["switch"]
    report = f"""# Mixer 56 historical-endpoint edge ablation

This confirmatory intervention uses all **{len(qids)}** held-out SimpleMC
questions. In every Mixer 56 head, the historical answer endpoint is made
unavailable to the final query by setting that one attention logit to negative
infinity before softmax.

| Condition | Natural switch rate | Endpoint-ablated switch rate | Answers changed by ablation |
|---|---:|---:|---:|
| Game | {100*game['natural_switch_rate']:.1f}% | {100*game['ablated_switch_rate']:.1f}% | {100*game['natural_answer_changed_by_ablation']:.1f}% |
| Neutral | {100*neutral['natural_switch_rate']:.1f}% | {100*neutral['ablated_switch_rate']:.1f}% | {100*neutral['natural_answer_changed_by_ablation']:.1f}% |

The ablation changes Game switching by
**{100*game['effects_ablated_minus_natural']['switch']['mean']:+.2f} percentage points**
(95% CI {100*game['effects_ablated_minus_natural']['switch']['ci'][0]:+.2f},
{100*game['effects_ablated_minus_natural']['switch']['ci'][1]:+.2f}) and Neutral
switching by
**{100*neutral['effects_ablated_minus_natural']['switch']['mean']:+.2f} points**
(95% CI {100*neutral['effects_ablated_minus_natural']['switch']['ci'][0]:+.2f},
{100*neutral['effects_ablated_minus_natural']['switch']['ci'][1]:+.2f}).
The Game-minus-Neutral causal difference is
**{100*did['mean']:+.2f} points** (95% CI {100*did['ci'][0]:+.2f},
{100*did['ci'][1]:+.2f}).
"""
    (output / "MIXER56_ENDPOINT_EDGE_ABLATION_REPORT.md").write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(
        args.config,
        args.results,
        args.baseline_root,
        args.output,
        args.bootstrap,
        args.seed,
    )


if __name__ == "__main__":
    main()
