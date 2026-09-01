from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import load_activation_dataset


def centered(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def rank_align(logits: np.ndarray, baseline_order: np.ndarray) -> np.ndarray:
    return np.take_along_axis(logits, baseline_order[:, None, :], axis=-1)


def trimmed_mean(x: np.ndarray, proportion: float = 0.1) -> float:
    x = np.sort(np.asarray(x))
    cut = int(len(x) * proportion)
    return float(x[cut : len(x) - cut].mean()) if cut and 2 * cut < len(x) else float(x.mean())


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    n = len(values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = values[rng.integers(0, n, n)].mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def analyze(output_dir: str, analysis_dir: str, n_boot: int, seed: int) -> dict:
    conditions = ["baseline", "incorrect", "neutral"]
    data = load_activation_dataset(output_dir, conditions)
    logits = centered(data.logits)
    baseline = logits[:, 0]
    order = np.argsort(-baseline[:, -1], axis=-1)
    rank_logits = np.stack([rank_align(logits[:, ci], order) for ci in range(3)], axis=1)
    n, _, n_layers, _ = rank_logits.shape
    rng = np.random.default_rng(seed)
    out = Path(analysis_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ci, condition in enumerate(conditions):
        for layer in range(n_layers):
            for rank in range(4):
                values = rank_logits[:, ci, layer, rank]
                lo, hi = bootstrap_ci(values, rng, n_boot)
                rows.append({
                    "condition": condition, "layer": layer, "rank": rank + 1,
                    "mean": float(values.mean()), "median": float(np.median(values)),
                    "trimmed_mean_10pct": trimmed_mean(values), "ci_low": lo, "ci_high": hi,
                })
    with (out / "rank_trajectories.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)

    # Quantities that do not acquire an interpretation from within-trial centering.
    strength_rows = []
    for ci, condition in enumerate(conditions):
        r = rank_logits[:, ci]
        metrics = {
            "first_strength": r[:, :, 0] - r[:, :, 1:].mean(axis=-1),
            "runner_strength": r[:, :, 1] - r[:, :, 2:].mean(axis=-1),
            "top_two_gap": r[:, :, 0] - r[:, :, 1],
        }
        for name, values in metrics.items():
            for layer in range(n_layers):
                lo, hi = bootstrap_ci(values[:, layer], rng, n_boot)
                strength_rows.append({
                    "condition": condition, "metric": name, "layer": layer,
                    "mean": float(values[:, layer].mean()), "median": float(np.median(values[:, layer])),
                    "trimmed_mean_10pct": trimmed_mean(values[:, layer]), "ci_low": lo, "ci_high": hi,
                })
    with (out / "strength_trajectories.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=strength_rows[0])
        writer.writeheader(); writer.writerows(strength_rows)

    local_choices = order[:, 0]
    recorded = np.asarray(["ABCD".index(data.metadata[(qid, "baseline")]["baseline_answer"]) for qid in data.question_ids])
    final_choices = np.argmax(logits[:, :, -1], axis=-1)
    summary = {
        "n_questions": n,
        "n_residual_readouts": n_layers,
        "local_baseline_vs_recorded_api_agreement": float(np.mean(local_choices == recorded)),
        "local_change_rate_incorrect": float(np.mean(final_choices[:, 1] != local_choices)),
        "local_change_rate_neutral": float(np.mean(final_choices[:, 2] != local_choices)),
        "mean_final_centered_rank_logits": {
            condition: rank_logits[:, ci, -1].mean(axis=0).tolist() for ci, condition in enumerate(conditions)
        },
        "important_caveat": "Intermediate-layer values are logit-lens pseudo-logits, not behavioral logits.",
    }
    (out / "trajectory_summary.json").write_text(json.dumps(summary, indent=2))
    _plot(rank_logits, conditions, out)
    return summary


def _plot(rank_logits: np.ndarray, conditions: list[str], out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    layers = np.arange(rank_logits.shape[2])
    rank_colors = ["#1b5e20", "#1565c0", "#ef6c00", "#6a1b9a"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ci, (ax, condition) in enumerate(zip(axes, conditions)):
        for rank in range(4):
            values = rank_logits[:, ci, :, rank]
            mean = values.mean(axis=0)
            se = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
            ax.plot(layers, mean, label=f"Baseline final rank {rank + 1}", color=rank_colors[rank])
            ax.fill_between(layers, mean - 1.96 * se, mean + 1.96 * se, color=rank_colors[rank], alpha=.15)
        ax.axhline(0, color="black", lw=.6); ax.set_title(condition.capitalize()); ax.set_xlabel("Residual readout (0 = embedding)")
    axes[0].set_ylabel("Centered A-D pseudo-logit")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "rank_trajectories.svg"); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize layerwise Second Chance logit-lens trajectories")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=2000); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); print(json.dumps(analyze(args.input, args.output, args.bootstrap, args.seed), indent=2))


if __name__ == "__main__":
    main()

