from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .all_trial_figures import _style


SCENARIOS = (
    "natural_game", "natural_neutral",
    "exclude_neutral_into_game_L41_48", "exclude_game_into_neutral_L41_48",
    "full_neutral_into_game_L41_48", "full_game_into_neutral_L41_48",
    "full_neutral_into_game_all_layers", "full_game_into_neutral_all_layers",
    "full_neutral_into_game_L1_16", "full_game_into_neutral_L1_16",
    "full_neutral_into_game_L17_32", "full_game_into_neutral_L17_32",
    "full_neutral_into_game_L33_40", "full_game_into_neutral_L33_40",
    "full_neutral_into_game_L49_64", "full_game_into_neutral_L49_64",
)
PATCHED = SCENARIOS[2:]


def _load(root: Path, scenario: str, qids: list[str]):
    logits, alternative = [], []
    for qid in qids:
        with np.load(root / "shards" / scenario / f"{qid}.npz", allow_pickle=False) as shard:
            logits.append(shard["final_canonical_logits"])
            alternative.append(shard["decision_alternative_scores"])
    return np.asarray(logits, dtype=np.float64), np.asarray(alternative, dtype=np.float64)


def _prior_labels(residual_root: Path, qids: list[str]):
    prior = []
    historical = {condition: [] for condition in ("incorrect", "neutral")}
    for qid in qids:
        for condition in ("baseline", "incorrect", "neutral"):
            with np.load(residual_root / "shards" / condition / f"{qid}.npz", allow_pickle=False) as shard:
                metadata = json.loads(str(shard["metadata"]))
            label = "ABCD".index(metadata["full_vocab_top_token"].strip())
            if condition == "baseline":
                prior.append(label)
            else:
                historical[condition].append(label)
    return np.asarray(prior), {key: np.asarray(value) for key, value in historical.items()}


def _metrics(logits: np.ndarray, alternative: np.ndarray, prior: np.ndarray):
    row = np.arange(len(prior))
    competitors = logits.copy(); competitors[row, prior] = -np.inf
    choice = logits.argmax(axis=1)
    return {
        "decision_alternative": alternative.mean(axis=1),
        "prior_margin": logits[row, prior] - competitors.max(axis=1),
        "ad_spread": logits.std(axis=1),
        "switch": (choice != prior).astype(float),
        "choice": choice,
    }


def _macro(values: np.ndarray, strata: np.ndarray) -> float:
    return float(np.mean([values[strata == label].mean() for label in range(4)]))


def _bootstrap(delta: np.ndarray, strata: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    values = np.empty(5000)
    groups = [np.flatnonzero(strata == label) for label in range(4)]
    for draw in range(len(values)):
        values[draw] = np.mean([
            delta[rng.choice(group, len(group), replace=True)].mean() for group in groups
        ])
    return [float(x) for x in np.quantile(values, (.025, .975))]


def analyze(root: Path, residual_root: Path, output: Path) -> dict:
    metadata = json.loads((root / "run_metadata.json").read_text())
    qids = metadata["question_ids"]
    prior, historical = _prior_labels(residual_root, qids)
    loaded = {scenario: _load(root, scenario, qids) for scenario in SCENARIOS}
    metrics = {scenario: _metrics(*loaded[scenario], prior) for scenario in SCENARIOS}
    references = {
        scenario: ("natural_game" if "into_game" in scenario else "natural_neutral")
        for scenario in PATCHED
    }
    rows = []
    summary = {
        "n_questions": len(qids),
        "natural_choice_agreement_with_cached": {
            "game": float(np.mean(metrics["natural_game"]["choice"] == historical["incorrect"])),
            "neutral": float(np.mean(metrics["natural_neutral"]["choice"] == historical["neutral"])),
        },
        "effects": {},
    }
    for si, scenario in enumerate(PATCHED):
        reference = references[scenario]
        scenario_summary = {}
        for metric in ("decision_alternative", "prior_margin", "ad_spread", "switch"):
            delta = metrics[scenario][metric] - metrics[reference][metric]
            mean = _macro(delta, prior)
            ci = _bootstrap(delta, prior, 100 + si)
            scenario_summary[metric] = {"mean_delta": mean, "ci_95": ci}
            rows.append({
                "scenario": scenario, "reference": reference, "metric": metric,
                "mean_delta": mean, "ci_low": ci[0], "ci_high": ci[1],
            })
        natural_choice = metrics[reference]["choice"]
        patched_choice = metrics[scenario]["choice"]
        natural_switch = natural_choice != prior
        patched_switch = patched_choice != prior
        scenario_summary["choice_transitions"] = {
            "total_choice_flips": int(np.sum(natural_choice != patched_choice)),
            "new_switches": int(np.sum(~natural_switch & patched_switch)),
            "prevented_switches": int(np.sum(natural_switch & ~patched_switch)),
            "switched_to_different_alternative": int(np.sum(natural_switch & patched_switch & (natural_choice != patched_choice))),
        }
        summary["effects"][scenario] = scenario_summary

    output.mkdir(parents=True, exist_ok=True)
    with (output / "causal_effects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "causal_summary.json").write_text(json.dumps(summary, indent=2))

    _style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), sharex=True)
    windows = ("L1_16", "L17_32", "L33_40", "L41_48", "L49_64", "all_layers")
    window_labels = ("L1–16", "L17–32", "L33–40", "L41–48", "L49–64", "All")
    colors = {"game": "#0072B2", "neutral": "#D55E00"}
    x = np.arange(len(windows)); width = .36
    for axis, metric, title, ylabel in (
        (axes[0, 0], "decision_alternative", "A  Later alternative representation", "Change in JLens family score"),
        (axes[0, 1], "prior_margin", "B  Final prior-answer margin", "Patched minus natural logits"),
        (axes[1, 0], "switch", "C  Final switching", "Change in switch rate"),
    ):
        for offset, target, source, label in ((-.5, "game", "neutral", "Neutral into Game"), (.5, "neutral", "game", "Game into Neutral")):
            scenarios = [f"full_{source}_into_{target}_{window}" for window in windows]
            values = np.asarray([summary["effects"][scenario][metric]["mean_delta"] for scenario in scenarios])
            low = np.asarray([values[i] - summary["effects"][scenario][metric]["ci_95"][0] for i, scenario in enumerate(scenarios)])
            high = np.asarray([summary["effects"][scenario][metric]["ci_95"][1] - values[i] for i, scenario in enumerate(scenarios)])
            axis.bar(x + offset*width, values, width, yerr=np.stack([low, high]), capsize=2, color=colors[target], label=label)
        axis.axhline(0, color="#555555", linewidth=.7)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, fontsize=7)
    flip_game = [summary["effects"][f"full_neutral_into_game_{window}"]["choice_transitions"]["total_choice_flips"] for window in windows]
    flip_neutral = [summary["effects"][f"full_game_into_neutral_{window}"]["choice_transitions"]["total_choice_flips"] for window in windows]
    axes[1, 1].bar(x-width/2, flip_game, width, color=colors["game"], label="Neutral into Game")
    axes[1, 1].bar(x+width/2, flip_neutral, width, color=colors["neutral"], label="Game into Neutral")
    axes[1, 1].set_title("D  Questions whose A-D choice changed", loc="left", fontweight="bold")
    axes[1, 1].set_ylabel("Count out of 128")
    axes[1, 1].legend(frameon=False, fontsize=7)
    for axis in axes[1]:
        axis.set_xticks(x, window_labels)
        axis.set_xlabel("Feedback-end residual replacement window")
    for axis in axes.flat:
        axis.grid(axis="y", color="#DDDDDD", linewidth=.5); axis.set_axisbelow(True); axis.spines[["top","right"]].set_visible(False)
    fig.suptitle("Qwen3.6-27B SimpleMC: causal test of the feedback-end exclusion bridge", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.96), w_pad=1.3, h_pad=1.5)
    figure_dir = output / "preserved_figures"; figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "exclusion_bridge_causal_test.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.residual_root, args.output), indent=2))


if __name__ == "__main__":
    main()
