from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .gdn_config import GDNExperimentConfig
from .io import read_metadata, shard_path


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _entropy(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=-1)


def _metrics(values: np.ndarray, baseline: np.ndarray, winners: np.ndarray) -> dict[str, np.ndarray]:
    values, baseline = _center(values), _center(baseline)
    row = np.arange(len(values)); winner = values[row, winners]
    denominator = np.sum(baseline * baseline, axis=-1)
    return {
        "compression": -np.sum((values - baseline) * baseline, axis=-1) / np.maximum(denominator, 1e-12),
        "winner_advantage": winner - (values.sum(axis=-1) - winner) / 3,
        "ad_entropy": _entropy(values),
        "ad_spread": values.std(axis=-1),
        "switch": (np.argmax(values, axis=-1) != winners).astype(float),
    }


def _contribution_metrics(contribution: np.ndarray, baseline: np.ndarray, winners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contribution, baseline = _center(contribution), _center(baseline)
    intermediate = (1,) * (contribution.ndim - 2)
    expanded_baseline = baseline.reshape((len(baseline),) + intermediate + (4,))
    denominator = np.sum(baseline * baseline, axis=-1).reshape((len(baseline),) + intermediate)
    compression = -np.sum(contribution * expanded_baseline, axis=-1) / np.maximum(denominator, 1e-12)
    selected = np.take_along_axis(
        contribution,
        winners.reshape((len(winners),) + intermediate + (1,)),
        axis=-1,
    )[..., 0]
    winner_advantage = selected - (contribution.sum(axis=-1) - selected) / 3
    return compression, winner_advantage


def _bootstrap(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float, float]:
    means = np.empty(samples)
    for start in range(0, samples, 1000):
        stop = min(start + 1000, samples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _load_final(root: Path, group: str, qids: list[str], key: str) -> np.ndarray:
    rows = []
    for qid in qids:
        with np.load(shard_path(root, group, qid), allow_pickle=False) as data:
            value = data[key]
            rows.append(value[-1] if value.ndim == 2 else value)
    return np.asarray(rows, dtype=np.float64)


def _split(qids: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(range(len(qids)), key=lambda index: hashlib.sha256(f"{seed}:{qids[index]}".encode()).digest())
    midpoint = len(qids) // 2
    discovery = np.zeros(len(qids), dtype=bool); discovery[order[:midpoint]] = True
    return discovery, ~discovery


def analyze(config: GDNExperimentConfig, output: Path) -> dict:
    root = Path(config.output_dir)
    groups = ["gdn_all48__user_incorrect"] + [f"gdn_all48__structural_{i}" for i in range(config.structural_controls)] + ["gdn_screen"]
    sets = [{path.stem for path in (root / "shards" / group).glob("*.npz")} for group in groups]
    sets.extend([
        {path.stem for path in (Path(config.natural_attention_dir) / "shards" / "natural_game_rerun").glob("*.npz")},
        {path.stem for path in (Path(config.mechanistic_dir) / "shards" / "baseline").glob("*.npz")},
    ])
    qids = sorted(set.intersection(*sets))
    if len(qids) != 500:
        raise RuntimeError(f"Expected 500 complete trials across all conditions; found {len(qids)}")
    baseline = _load_final(Path(config.mechanistic_dir), "baseline", qids, "canonical_logits")
    natural = _load_final(Path(config.natural_attention_dir), "natural_game_rerun", qids, "final_canonical_logits")
    winners = np.argmax(baseline, axis=-1)
    natural_metrics = _metrics(natural, baseline, winners)
    rng = np.random.default_rng(config.seed)

    global_logits = np.stack([_load_final(root, group, qids, "final_canonical_logits") for group in groups[:-1]], axis=1)
    global_metrics = [{name: values for name, values in _metrics(global_logits[:, index], baseline, winners).items()} for index in range(global_logits.shape[1])]
    global_rows = []
    for index, group in enumerate(groups[:-1]):
        for metric, natural_values in natural_metrics.items():
            effect = global_metrics[index][metric] - natural_values
            mean, low, high = _bootstrap(effect, config.bootstrap_samples, rng)
            global_rows.append({"scenario": group, "metric": metric, "effect": mean, "ci_low": low, "ci_high": high})
    primary_vs_control = {}
    for metric in natural_metrics:
        contrast = global_metrics[0][metric] - np.mean([values[metric] for values in global_metrics[1:]], axis=0)
        mean, low, high = _bootstrap(contrast, config.bootstrap_samples, rng)
        primary_vs_control[metric] = {"mean": mean, "ci_low": low, "ci_high": high}

    first = np.load(shard_path(root, "gdn_screen", qids[0]), allow_pickle=False)
    shape = first["head_direct_ad"].shape
    direct = np.empty((len(qids), *shape), dtype=np.float32)
    output_norm = np.empty((len(qids), *shape[:-1]), dtype=np.float32)
    for index, qid in enumerate(qids):
        with np.load(shard_path(root, "gdn_screen", qid), allow_pickle=False) as data:
            direct[index] = data["head_direct_ad"]
            output_norm[index] = data["head_output_norm"]
    meta = json.loads((root / "screen_run_metadata.json").read_text())
    layers = meta["linear_attention_layers"]
    sources = meta["screen_sources"]
    target_index = sources.index("user_incorrect")
    control_indices = [index for index, source in enumerate(sources) if source.startswith("structural_")]
    compression, winner_contribution = _contribution_metrics(direct, baseline, winners)
    differential_compression = compression[:, :, target_index] - compression[:, :, control_indices].mean(axis=2)
    differential_winner = winner_contribution[:, :, target_index] - winner_contribution[:, :, control_indices].mean(axis=2)
    differential_norm = output_norm[:, :, target_index] - output_norm[:, :, control_indices].mean(axis=2)
    discovery, confirmation = _split(qids, config.seed)
    discovery_mean = differential_compression[discovery].mean(axis=0)
    confirmation_mean = differential_compression[confirmation].mean(axis=0)

    head_rows = []
    for layer_index, layer in enumerate(layers):
        for head in range(shape[2]):
            values = differential_compression[confirmation, layer_index, head]
            mean, low, high = _bootstrap(values, config.bootstrap_samples, rng)
            head_rows.append({
                "layer": layer, "head": head,
                "discovery_differential_compression": float(discovery_mean[layer_index, head]),
                "confirmation_differential_compression": mean,
                "confirmation_ci_low": low, "confirmation_ci_high": high,
                "confirmation_differential_winner_contribution": float(differential_winner[confirmation, layer_index, head].mean()),
                "confirmation_differential_output_norm": float(differential_norm[confirmation, layer_index, head].mean()),
            })
    ranked_heads = sorted(head_rows, key=lambda row: row["discovery_differential_compression"], reverse=True)

    layer_direct = direct.sum(axis=3)
    layer_compression, layer_winner = _contribution_metrics(layer_direct, baseline, winners)
    layer_differential = layer_compression[:, :, target_index] - layer_compression[:, :, control_indices].mean(axis=2)
    layer_rows = []
    for layer_index, layer in enumerate(layers):
        mean, low, high = _bootstrap(layer_differential[confirmation, layer_index], config.bootstrap_samples, rng)
        layer_rows.append({
            "layer": layer,
            "discovery_differential_compression": float(layer_differential[discovery, layer_index].mean()),
            "confirmation_differential_compression": mean,
            "confirmation_ci_low": low, "confirmation_ci_high": high,
            "confirmation_differential_winner_contribution": float(
                (layer_winner[:, layer_index, target_index] - layer_winner[:, layer_index, control_indices].mean(axis=1))[confirmation].mean()
            ),
        })
    ranked_layers = sorted(layer_rows, key=lambda row: row["discovery_differential_compression"], reverse=True)

    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("gdn_global_effects.csv", global_rows), ("gdn_head_screen.csv", head_rows), ("gdn_layer_screen.csv", layer_rows)):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    bound = np.percentile(np.abs(np.concatenate([discovery_mean.ravel(), confirmation_mean.ravel()])), 99)
    fig, axes = plt.subplots(2, 1, figsize=(12, 11), constrained_layout=True)
    for axis, values, title in zip(axes, (discovery_mean, confirmation_mean), ("Discovery (n=250)", "Held-out confirmation (n=250)")):
        image = axis.imshow(values, aspect="auto", origin="lower", cmap="coolwarm", vmin=-bound, vmax=bound)
        axis.set_yticks(np.arange(len(layers)), labels=layers); axis.set_xlabel("DeltaNet value head"); axis.set_ylabel("Layer")
        axis.set_title(title + ": `incorrect` minus mean structural-control compression contribution")
        fig.colorbar(image, ax=axis, shrink=.8)
    fig.savefig(output / "gdn_head_screen.png", dpi=220); fig.savefig(output / "gdn_head_screen.svg"); plt.close(fig)

    selected_heads = ranked_heads[:8]
    selected_layers = ranked_layers[:3]
    confirmation_qids = [qid for qid, keep in zip(qids, confirmation) if keep]
    head_targets: dict[int, list[int]] = {}
    for row in selected_heads:
        head_targets.setdefault(row["layer"], []).append(row["head"])
    scenarios = [
        {"id": "gdn_confirm__top8_heads__incorrect", "source": "user_incorrect", "targets": [{"layer": layer, "heads": heads} for layer, heads in sorted(head_targets.items())]},
        {"id": "gdn_confirm__top8_heads__structural0", "source": "structural_0", "targets": [{"layer": layer, "heads": heads} for layer, heads in sorted(head_targets.items())]},
        {"id": "gdn_confirm__top3_layers__incorrect", "source": "user_incorrect", "targets": [{"layer": row["layer"], "heads": None} for row in selected_layers]},
        {"id": "gdn_confirm__top3_layers__structural0", "source": "structural_0", "targets": [{"layer": row["layer"], "heads": None} for row in selected_layers]},
    ]
    scenarios.extend({"id": f"gdn_confirm__layer{row['layer']}__incorrect", "source": "user_incorrect", "targets": [{"layer": row["layer"], "heads": None}]} for row in selected_layers)
    plan = {"question_ids": confirmation_qids, "selected_heads": selected_heads, "selected_layers": selected_layers, "scenarios": scenarios}
    (output / "gdn_confirmation_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    summary = {
        "n_questions": len(qids), "discovery_n": int(discovery.sum()), "confirmation_n": int(confirmation.sum()),
        "global_primary_vs_mean_structural_control": primary_vs_control,
        "top_discovery_heads_with_heldout_results": selected_heads,
        "top_discovery_layers_with_heldout_results": selected_layers,
    }
    (output / "gdn_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze global and all-head Gated DeltaNet interventions")
    parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); config = GDNExperimentConfig.load(args.config)
    print(json.dumps(analyze(config, Path(args.output_dir)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
