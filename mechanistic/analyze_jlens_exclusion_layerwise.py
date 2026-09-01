from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .all_trial_figures import _style
from .run_jlens_exclusion_layerwise import SOURCE_LAYERS, WINDOWS, _scenarios


def _load(root: Path, scenario: str, qids: list[str]) -> np.ndarray:
    return np.asarray([
        np.load(root / "shards" / scenario / f"{qid}.npz", allow_pickle=False)[
            "final_canonical_logits"
        ]
        for qid in qids
    ], dtype=np.float64)


def _prior_labels(residual_root: Path, qids: list[str]) -> np.ndarray:
    labels = []
    for qid in qids:
        with np.load(
            residual_root / "shards" / "baseline" / f"{qid}.npz", allow_pickle=False
        ) as shard:
            metadata = json.loads(str(shard["metadata"]))
        labels.append("ABCD".index(metadata["full_vocab_top_token"].strip()))
    return np.asarray(labels, dtype=int)


def _metrics(logits: np.ndarray, prior: np.ndarray) -> dict[str, np.ndarray]:
    row = np.arange(len(prior))
    competitors = logits.copy()
    competitors[row, prior] = -np.inf
    choice = logits.argmax(axis=1)
    return {
        "prior_margin": logits[row, prior] - competitors.max(axis=1),
        "ad_spread": logits.std(axis=1),
        "switch": (choice != prior).astype(float),
        "choice": choice,
    }


def _macro(values: np.ndarray, strata: np.ndarray) -> float:
    return float(np.mean([values[strata == label].mean() for label in range(4)]))


def _bootstrap(delta: np.ndarray, strata: np.ndarray, seed: int, draws: int = 5000):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == label) for label in range(4)]
    values = np.empty(draws)
    for draw in range(draws):
        values[draw] = np.mean([
            delta[rng.choice(group, len(group), replace=True)].mean() for group in groups
        ])
    return [float(value) for value in np.quantile(values, (.025, .975))]


def analyze(root: Path, residual_root: Path, output: Path) -> dict:
    metadata = json.loads((root / "run_metadata.json").read_text())
    qids = metadata["question_ids"]
    prior = _prior_labels(residual_root, qids)
    scenarios = _scenarios()
    loaded = {scenario: _load(root, scenario, qids) for scenario in scenarios}
    metrics = {scenario: _metrics(logits, prior) for scenario, logits in loaded.items()}

    rows = []
    effects = {}
    patched = scenarios[2:]
    for si, scenario in enumerate(patched):
        reference = "natural_game" if "into_game" in scenario else "natural_neutral"
        scenario_summary = {}
        for metric in ("prior_margin", "ad_spread", "switch"):
            delta = metrics[scenario][metric] - metrics[reference][metric]
            mean = _macro(delta, prior)
            ci = _bootstrap(delta, prior, 1000 + si * 10 + len(rows))
            scenario_summary[metric] = {"mean_delta": mean, "ci_95": ci}
            rows.append({
                "scenario": scenario,
                "reference": reference,
                "metric": metric,
                "mean_delta": mean,
                "ci_low": ci[0],
                "ci_high": ci[1],
            })
        natural_choice = metrics[reference]["choice"]
        patched_choice = metrics[scenario]["choice"]
        natural_switch = natural_choice != prior
        patched_switch = patched_choice != prior
        scenario_summary["choice_transitions"] = {
            "total_choice_flips": int(np.sum(natural_choice != patched_choice)),
            "new_switches": int(np.sum(~natural_switch & patched_switch)),
            "prevented_switches": int(np.sum(natural_switch & ~patched_switch)),
            "switched_to_different_alternative": int(np.sum(
                natural_switch & patched_switch & (natural_choice != patched_choice)
            )),
        }
        effects[scenario] = scenario_summary

    layer_rows = {
        (row["scenario"], row["metric"]): row
        for row in rows
    }
    expected = {
        "game_ablation": {
            "switch": "negative",
            "prior_margin": "positive",
        },
        "neutral_insertion": {
            "switch": "positive",
            "prior_margin": "negative",
        },
    }
    largest = {}
    for target, stem in (
        ("game_ablation", "exclude_neutral_into_game"),
        ("neutral_insertion", "exclude_game_into_neutral"),
    ):
        largest[target] = {}
        for metric in ("switch", "prior_margin", "ad_spread"):
            candidates = []
            for layer in SOURCE_LAYERS:
                scenario = f"{stem}_L{layer + 1}"
                row = layer_rows[(scenario, metric)]
                candidates.append({
                    "layer": layer + 1,
                    "mean_delta": row["mean_delta"],
                    "ci_95": [row["ci_low"], row["ci_high"]],
                    "total_choice_flips": effects[scenario]["choice_transitions"]["total_choice_flips"],
                })
            largest[target][metric] = max(candidates, key=lambda item: abs(item["mean_delta"]))

    summary = {
        "n_questions": len(qids),
        "source_layers": [layer + 1 for layer in SOURCE_LAYERS],
        "natural": {
            "game_switch_rate": _macro(metrics["natural_game"]["switch"], prior),
            "neutral_switch_rate": _macro(metrics["natural_neutral"]["switch"], prior),
        },
        "expected_causal_signs": expected,
        "largest_single_layer_effects": largest,
        "effects": effects,
    }

    output.mkdir(parents=True, exist_ok=True)
    figure_dir = output / "preserved_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    with (output / "layerwise_effects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "layerwise_summary.json").write_text(json.dumps(summary, indent=2))

    _style()
    import matplotlib.pyplot as plt

    layers = np.asarray([layer + 1 for layer in SOURCE_LAYERS])
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), sharex=True)
    colors = {"game": "#0072B2", "neutral": "#D55E00"}
    for axis, metric, title, ylabel in (
        (axes[0], "switch", "A  Final switching", "Patched − natural switch rate"),
        (axes[1], "prior_margin", "B  Prior-answer margin", "Patched − natural logits"),
        (axes[2], "ad_spread", "C  A–D spread", "Patched − natural logits SD"),
    ):
        for target, stem, label in (
            ("game", "exclude_neutral_into_game", "Remove from Game"),
            ("neutral", "exclude_game_into_neutral", "Insert into Neutral"),
        ):
            selected = [layer_rows[(f"{stem}_L{layer}", metric)] for layer in layers]
            means = np.asarray([row["mean_delta"] for row in selected])
            low = np.asarray([row["ci_low"] for row in selected])
            high = np.asarray([row["ci_high"] for row in selected])
            axis.plot(layers, means, marker="o", ms=2.8, lw=1.3, color=colors[target], label=label)
            axis.fill_between(layers, low, high, color=colors[target], alpha=.13, linewidth=0)
        axis.axhline(0, color="#777777", lw=.8)
        axis.set_title(title, loc="left")
        axis.set_xlabel("Feedback-end post-block readout")
        axis.set_ylabel(ylabel)
        axis.set_xticks((41, 45, 49, 53, 57, 61, 63))
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Qwen3.6-27B SimpleMC: layerwise causal test of the JLens exclusion direction",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        figure_dir / "exclusion_direction_layerwise.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.residual_root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
