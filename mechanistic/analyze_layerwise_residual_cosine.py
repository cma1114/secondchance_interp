from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _residuals(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as shard:
        return shard["residuals"].astype(np.float32)


def _cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ld,ld->l", first, second, dtype=np.float64)
    first_norm = np.sqrt(np.einsum("ld,ld->l", first, first, dtype=np.float64))
    second_norm = np.sqrt(np.einsum("ld,ld->l", second, second, dtype=np.float64))
    return numerator / np.maximum(first_norm * second_norm, 1e-30)


def _bootstrap(values: np.ndarray, draws: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_questions, n_layers = values.shape
    samples = np.empty((draws, n_layers), dtype=np.float32)
    chunk = 100
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        indices = rng.integers(0, n_questions, size=(stop - start, n_questions))
        samples[start:stop] = values[indices].mean(axis=1)
    return np.quantile(samples, 0.025, axis=0), np.quantile(samples, 0.975, axis=0)


def _first_sustained(lower: np.ndarray, start: int = 1, width: int = 3) -> int | None:
    for layer in range(start, len(lower) - width + 1):
        if np.all(lower[layer : layer + width] > 0):
            return layer
    return None


def analyze(
    baseline_root: Path,
    second_chance_root: Path,
    output: Path,
    draws: int,
    seed: int,
) -> None:
    baseline_dir = baseline_root / "shards" / "baseline"
    game_dir = second_chance_root / "shards" / "incorrect"
    neutral_dir = second_chance_root / "shards" / "neutral"
    qids = sorted(
        {path.stem for path in baseline_dir.glob("*.npz")}
        & {path.stem for path in game_dir.glob("*.npz")}
        & {path.stem for path in neutral_dir.glob("*.npz")}
    )
    if not qids:
        raise RuntimeError("No matched Baseline/Game/Neutral residual shards")

    example = _residuals(baseline_dir / f"{qids[0]}.npz")
    n_layers, width = example.shape
    means = {
        "baseline": np.zeros((n_layers, width), dtype=np.float64),
        "game": np.zeros((n_layers, width), dtype=np.float64),
        "neutral": np.zeros((n_layers, width), dtype=np.float64),
    }
    raw_bg = np.empty((len(qids), n_layers), dtype=np.float32)
    raw_bn = np.empty_like(raw_bg)

    for index, qid in enumerate(qids):
        baseline = _residuals(baseline_dir / f"{qid}.npz")
        game = _residuals(game_dir / f"{qid}.npz")
        neutral = _residuals(neutral_dir / f"{qid}.npz")
        if baseline.shape != example.shape or game.shape != example.shape or neutral.shape != example.shape:
            raise RuntimeError(f"Residual shape mismatch for {qid}")
        means["baseline"] += baseline
        means["game"] += game
        means["neutral"] += neutral
        raw_bg[index] = _cosine(baseline, game)
        raw_bn[index] = _cosine(baseline, neutral)

    for condition in means:
        means[condition] = (means[condition] / len(qids)).astype(np.float32)

    centered_bg = np.empty_like(raw_bg)
    centered_bn = np.empty_like(raw_bg)
    for index, qid in enumerate(qids):
        baseline = _residuals(baseline_dir / f"{qid}.npz") - means["baseline"]
        game = _residuals(game_dir / f"{qid}.npz") - means["game"]
        neutral = _residuals(neutral_dir / f"{qid}.npz") - means["neutral"]
        centered_bg[index] = _cosine(baseline, game)
        centered_bn[index] = _cosine(baseline, neutral)

    arrays = {
        "raw_baseline_game": raw_bg,
        "raw_baseline_neutral": raw_bn,
        "raw_neutral_minus_game": raw_bn - raw_bg,
        "centered_baseline_game": centered_bg,
        "centered_baseline_neutral": centered_bn,
        "centered_neutral_minus_game": centered_bn - centered_bg,
    }
    means_by_metric = {name: values.mean(axis=0) for name, values in arrays.items()}
    intervals = {
        name: _bootstrap(values, draws, seed + index)
        for index, (name, values) in enumerate(arrays.items())
    }

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "layerwise_residual_cosine.npz",
        question_ids=np.asarray(qids),
        layers=np.arange(n_layers),
        **arrays,
        **{f"mean_{name}": values for name, values in means_by_metric.items()},
        **{f"ci_low_{name}": values[0] for name, values in intervals.items()},
        **{f"ci_high_{name}": values[1] for name, values in intervals.items()},
    )

    raw_diff = means_by_metric["raw_neutral_minus_game"]
    centered_diff = means_by_metric["centered_neutral_minus_game"]
    raw_low, raw_high = intervals["raw_neutral_minus_game"]
    centered_low, centered_high = intervals["centered_neutral_minus_game"]
    selected_layers = [layer for layer in (0, 16, 24, 32, 40, 48, 52, 56, 60, 64) if layer < n_layers]
    summary = {
        "n_questions": len(qids),
        "n_readouts": n_layers,
        "residual_width": width,
        "baseline_root": str(baseline_root),
        "second_chance_root": str(second_chance_root),
        "difference_definition": "cos(Baseline, Neutral) - cos(Baseline, Game)",
        "first_three_layer_positive_raw": _first_sustained(raw_low),
        "first_three_layer_positive_question_centered": _first_sustained(centered_low),
        "largest_raw_difference": {
            "layer": int(np.argmax(raw_diff)),
            "mean": float(raw_diff.max()),
        },
        "largest_question_centered_difference": {
            "layer": int(np.argmax(centered_diff)),
            "mean": float(centered_diff.max()),
        },
        "selected_layers": {
            str(layer): {
                "raw_baseline_game": float(means_by_metric["raw_baseline_game"][layer]),
                "raw_baseline_neutral": float(means_by_metric["raw_baseline_neutral"][layer]),
                "raw_neutral_minus_game": float(raw_diff[layer]),
                "raw_difference_ci": [float(raw_low[layer]), float(raw_high[layer])],
                "centered_baseline_game": float(means_by_metric["centered_baseline_game"][layer]),
                "centered_baseline_neutral": float(means_by_metric["centered_baseline_neutral"][layer]),
                "centered_neutral_minus_game": float(centered_diff[layer]),
                "centered_difference_ci": [float(centered_low[layer]), float(centered_high[layer])],
            }
            for layer in selected_layers
        },
    }
    (output / "layerwise_residual_cosine_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    import matplotlib.pyplot as plt

    layers = np.arange(n_layers)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    panels = (
        (axes[0, 0], "raw_baseline_game", "raw_baseline_neutral", "A  Raw residual cosine"),
        (axes[0, 1], "raw_neutral_minus_game", None, "B  Raw paired difference"),
        (axes[1, 0], "centered_baseline_game", "centered_baseline_neutral", "C  Question-centered residual cosine"),
        (axes[1, 1], "centered_neutral_minus_game", None, "D  Centered paired difference"),
    )
    colors = ("#2f90f5", "#ef7f35")
    for axis, first, second, title in panels:
        if second is not None:
            for name, label, color in (
                (first, "Baseline–Game", colors[0]),
                (second, "Baseline–Neutral", colors[1]),
            ):
                mean = means_by_metric[name]
                low, high = intervals[name]
                axis.plot(layers, mean, color=color, linewidth=2, label=label)
                axis.fill_between(layers, low, high, color=color, alpha=0.14, linewidth=0)
            axis.legend(frameon=False)
            axis.set_ylabel("Cosine similarity")
        else:
            mean = means_by_metric[first]
            low, high = intervals[first]
            axis.axhline(0, color="#777777", linewidth=1)
            axis.plot(layers, mean, color="#5abf73", linewidth=2)
            axis.fill_between(layers, low, high, color="#5abf73", alpha=0.16, linewidth=0)
            axis.set_ylabel("Neutral minus Game")
        axis.set_title(title, loc="left")
        axis.grid(axis="y", alpha=0.2)
        axis.set_xlim(0, n_layers - 1)
    for axis in axes[1]:
        axis.set_xlabel("Residual readout")
    # Use matched axes so the raw and question-centered magnitudes are visually
    # comparable rather than being exaggerated by independent autoscaling.
    axes[0, 0].set_ylim(0, 1.02)
    axes[1, 0].set_ylim(0, 1.02)
    difference_low = min(
        float(intervals["raw_neutral_minus_game"][0].min()),
        float(intervals["centered_neutral_minus_game"][0].min()),
    )
    difference_high = max(
        float(intervals["raw_neutral_minus_game"][1].max()),
        float(intervals["centered_neutral_minus_game"][1].max()),
    )
    difference_pad = 0.05 * (difference_high - difference_low)
    for axis in (axes[0, 1], axes[1, 1]):
        axis.set_ylim(difference_low - difference_pad, difference_high + difference_pad)
    fig.suptitle(
        "Paired final-decision residual similarity to Baseline (SimpleMC, n=500)",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(output / "layerwise_residual_cosine.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "layerwise_residual_cosine.svg", bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Layerwise residual cosine similarity",
        "",
        f"Matched SimpleMC questions: **{len(qids)}**.",
        "",
        "The primary paired difference is `cos(Baseline, Neutral) - cos(Baseline, Game)`.",
        "Positive values mean that Game is farther from Baseline than Neutral is.",
        "Question-centered curves subtract the across-question mean residual separately",
        "within each condition and layer before taking the matched cosine.",
        "",
        "| Readout | Raw B–G | Raw B–N | Raw N−G [95% CI] | Centered B–G | Centered B–N | Centered N−G [95% CI] |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in selected_layers:
        row = summary["selected_layers"][str(layer)]
        lines.append(
            f"| {layer} | {row['raw_baseline_game']:.5f} | {row['raw_baseline_neutral']:.5f} | "
            f"{row['raw_neutral_minus_game']:+.5f} [{row['raw_difference_ci'][0]:+.5f}, {row['raw_difference_ci'][1]:+.5f}] | "
            f"{row['centered_baseline_game']:.5f} | {row['centered_baseline_neutral']:.5f} | "
            f"{row['centered_neutral_minus_game']:+.5f} [{row['centered_difference_ci'][0]:+.5f}, {row['centered_difference_ci'][1]:+.5f}] |"
        )
    lines.extend([
        "",
        f"First three-readout run with a positive 95% interval (raw): **{summary['first_three_layer_positive_raw']}**.",
        f"First three-readout run with a positive 95% interval (question-centered): **{summary['first_three_layer_positive_question_centered']}**.",
    ])
    (output / "LAYERWISE_RESIDUAL_COSINE_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--second-chance-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(args.baseline_root, args.second_chance_root, args.output, args.draws, args.seed)


if __name__ == "__main__":
    main()
