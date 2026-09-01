from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .io import read_metadata, shard_path
from .sublayer_config import SublayerExperimentConfig


def _plot_screen(rows: list[dict], selected: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    selected_names = {row["component"] for row in selected}
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    for axis, kind, title in zip(axes, ("mixer", "mlp"), ("Attention/DeltaNet mixer outputs", "MLP outputs")):
        subset = sorted((row for row in rows if row["kind"] == kind), key=lambda row: row["layer"])
        x = np.asarray([row["layer"] for row in subset])
        mean = np.asarray([row["confirmation_baseline_aligned_compression"] for row in subset])
        low = np.asarray([row["confirmation_baseline_aligned_compression_ci_low"] for row in subset])
        high = np.asarray([row["confirmation_baseline_aligned_compression_ci_high"] for row in subset])
        colors = ["#0072B2" if row["mixer_type"] == "deltanet" else "#D55E00" if kind == "mixer" else "#009E73" for row in subset]
        axis.vlines(x, low, high, color=colors, alpha=.55, linewidth=1.2)
        axis.scatter(x, mean, c=colors, s=28, zorder=3)
        chosen = np.asarray([row["component"] in selected_names for row in subset])
        axis.scatter(x[chosen], mean[chosen], facecolors="none", edgecolors="black", s=95, linewidths=1.5, zorder=4)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_ylabel("Game − Neutral\ncompression contribution")
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    axes[-1].set_xlabel("Zero-indexed transformer layer")
    axes[-1].set_xticks(np.arange(0, 64, 4))
    axes[0].scatter([], [], c="#0072B2", label="Gated DeltaNet")
    axes[0].scatter([], [], c="#D55E00", label="Ordinary attention")
    axes[0].scatter([], [], facecolors="none", edgecolors="black", s=95, label="Discovery-selected")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _common_questions(root: Path) -> list[str]:
    sets = [{path.stem for path in (root / "shards" / condition).glob("*.npz")} for condition in ("baseline", "incorrect", "neutral")]
    qids = sorted(set.intersection(*sets))
    if not qids:
        raise FileNotFoundError("No complete baseline/Game/Neutral sublayer shards")
    return qids


def _load(root: Path, condition: str, qids: list[str]) -> tuple[np.ndarray, list[dict]]:
    values, metadata = [], []
    for qid in qids:
        with np.load(shard_path(root, condition, qid), allow_pickle=False) as data:
            values.append(data["boundary_canonical_logits"])
            metadata.append(read_metadata(data))
    return np.asarray(values, dtype=np.float64), metadata


def _stratified_split(labels: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    discovery, confirmation = [], []
    for label in range(4):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        cut = int(round(len(indices) * fraction))
        discovery.extend(indices[:cut])
        confirmation.extend(indices[cut:])
    return np.asarray(sorted(discovery)), np.asarray(sorted(confirmation))


def _macro_mean(values: np.ndarray, labels: np.ndarray) -> float:
    groups = [values[labels == label] for label in range(4) if np.any(labels == label)]
    if not groups:
        raise ValueError("Cannot macro-average an empty split")
    return float(np.mean([group.mean() for group in groups]))


def _macro_bootstrap(values: np.ndarray, labels: np.ndarray, samples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    groups = [values[labels == label] for label in range(4) if np.any(labels == label)]
    if not groups:
        raise ValueError("Cannot bootstrap an empty split")
    for sample in range(samples):
        means[sample] = np.mean([group[rng.integers(0, len(group), len(group))].mean() for group in groups])
    return _macro_mean(values, labels), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _component_values(boundaries: np.ndarray, baseline_final: np.ndarray, winners: np.ndarray, layer: int, kind: str) -> dict[str, np.ndarray]:
    before_index, after_index = (0, 1) if kind == "mixer" else (1, 2)
    before = _center(boundaries[:, layer, before_index])
    after = _center(boundaries[:, layer, after_index])
    delta = after - before
    denominator = np.maximum(np.sum(baseline_final * baseline_final, axis=-1), 1e-8)
    row = np.arange(len(delta))
    winner_update = delta[row, winners] - (delta.sum(axis=-1) - delta[row, winners]) / 3.0
    return {
        "baseline_aligned_compression": -np.sum(delta * baseline_final, axis=-1) / denominator,
        "incoming_antievidence": -np.sum(delta * before, axis=-1) / denominator,
        "winner_suppression": -winner_update,
    }


def analyze(config: SublayerExperimentConfig, output: Path) -> dict:
    root = Path(config.output_dir)
    qids = _common_questions(root)
    baseline, baseline_meta = _load(root, "baseline", qids)
    game, _ = _load(root, "incorrect", qids)
    neutral, _ = _load(root, "neutral", qids)
    baseline_final = _center(baseline[:, -1, -1])
    winners = np.argmax(baseline_final, axis=-1)
    discovery, confirmation = _stratified_split(winners, config.discovery_fraction, config.seed)
    run_meta = json.loads((root / "run_metadata.json").read_text())
    layer_kinds = run_meta["layer_mixer_kinds"]
    rows = []
    component_arrays: dict[str, dict[str, np.ndarray]] = {}
    for layer in range(baseline.shape[1]):
        for kind in ("mixer", "mlp"):
            game_values = _component_values(game, baseline_final, winners, layer, kind)
            neutral_values = _component_values(neutral, baseline_final, winners, layer, kind)
            key = f"{kind}_l{layer}"
            component_arrays[key] = {}
            row = {"component": key, "kind": kind, "layer": layer, "mixer_type": layer_kinds[layer] if kind == "mixer" else "mlp"}
            for metric in game_values:
                difference = game_values[metric] - neutral_values[metric]
                component_arrays[key][metric] = difference
                row[f"discovery_{metric}"] = _macro_mean(difference[discovery], winners[discovery])
                mean, low, high = _macro_bootstrap(difference[confirmation], winners[confirmation], config.bootstrap_samples, config.seed + layer * 7 + (kind == "mlp"))
                row[f"confirmation_{metric}"] = mean
                row[f"confirmation_{metric}_ci_low"] = low
                row[f"confirmation_{metric}_ci_high"] = high
            rows.append(row)

    selected = []
    for kind in ("mixer", "mlp"):
        candidates = [row for row in rows if row["kind"] == kind]
        candidates.sort(key=lambda row: row["discovery_baseline_aligned_compression"], reverse=True)
        selected.extend(candidates[: config.candidates_per_kind])
    selected.sort(key=lambda row: row["discovery_baseline_aligned_compression"], reverse=True)
    targets = [{"layer": int(row["layer"]), "kind": row["kind"], "component": row["component"]} for row in selected]
    scenarios = []
    for target in targets:
        scenarios.extend([
            {"id": f"neutral_into_game__{target['component']}", "source_condition": "neutral", "target_condition": "incorrect", "targets": [target]},
            {"id": f"game_into_neutral__{target['component']}", "source_condition": "incorrect", "target_condition": "neutral", "targets": [target]},
        ])
    for label, subset in (("top_mixers", [x for x in targets if x["kind"] == "mixer"]), ("top_mlps", [x for x in targets if x["kind"] == "mlp"]), ("all_selected", targets)):
        if subset:
            scenarios.extend([
                {"id": f"neutral_into_game__{label}", "source_condition": "neutral", "target_condition": "incorrect", "targets": subset},
                {"id": f"game_into_neutral__{label}", "source_condition": "incorrect", "target_condition": "neutral", "targets": subset},
            ])
    plan = {
        "selection_rule": f"top {config.candidates_per_kind} mixer and MLP components by discovery-split Game-minus-Neutral baseline-aligned compression",
        "discovery_question_ids": [qids[i] for i in discovery],
        "confirmation_question_ids": [qids[i] for i in confirmation],
        "targets": targets,
        "scenarios": scenarios,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "sublayer_screen.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    _plot_screen(rows, selected, output / "sublayer_compression_screen")
    (output / "component_patch_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    summary = {
        "n_questions": len(qids),
        "n_discovery": len(discovery),
        "n_confirmation": len(confirmation),
        "selected_targets": selected,
        "primary_metric": "Each sublayer's immediate centered A-D update projected opposite the same question's final Baseline A-D evidence, Game minus Neutral.",
    }
    (output / "sublayer_screen_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze mixer/MLP compression contributions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(SublayerExperimentConfig.load(args.config), Path(args.output))


if __name__ == "__main__":
    main()
