from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = values.max(axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


def _answer_scores(scores: np.ndarray, layout: list[dict]) -> np.ndarray:
    answers = []
    for letter in "ABCD":
        indices = [index for index, row in enumerate(layout) if row["family"] == f"answer_{letter}"]
        if not indices:
            raise ValueError(f"No selected JLens tokens for answer {letter}")
        answers.append(_logsumexp(scores[..., indices]))
    return np.stack(answers, axis=-1)


def _entropy_bits(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -(probabilities * np.log2(np.maximum(probabilities, 1e-30))).sum(axis=-1)


def _bootstrap(values: np.ndarray, draws: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_questions, n_layers = values.shape
    samples = np.empty((draws, n_layers), dtype=np.float32)
    for start in range(0, draws, 100):
        stop = min(start + 100, draws)
        indices = rng.integers(0, n_questions, size=(stop - start, n_questions))
        samples[start:stop] = values[indices].mean(axis=1)
    return np.quantile(samples, 0.025, axis=0), np.quantile(samples, 0.975, axis=0)


def analyze(jlens_root: Path, output: Path, draws: int, seed: int) -> None:
    layout = json.loads((jlens_root / "selected_token_layout.json").read_text())
    with np.load(jlens_root / "jlens_scores.npz", allow_pickle=False) as cached:
        scores = _answer_scores(cached["final_scores"].astype(np.float64), layout)
        qids = cached["question_ids"].astype(str)
        conditions = cached["conditions"].astype(str).tolist()
    expected = ["baseline", "incorrect", "neutral"]
    if conditions != expected:
        raise ValueError(f"Unexpected condition order: {conditions}")

    centered = scores - scores.mean(axis=-1, keepdims=True)
    spread = centered.std(axis=-1)
    entropy = _entropy_bits(centered)
    arrays = {
        "spread_baseline": spread[0],
        "spread_game": spread[1],
        "spread_neutral": spread[2],
        "compression_game_vs_baseline": spread[0] - spread[1],
        "compression_neutral_vs_baseline": spread[0] - spread[2],
        "entropy_baseline": entropy[0],
        "entropy_game": entropy[1],
        "entropy_neutral": entropy[2],
        "entropy_increase_game_vs_baseline": entropy[1] - entropy[0],
        "entropy_increase_neutral_vs_baseline": entropy[2] - entropy[0],
    }
    means = {name: values.mean(axis=0) for name, values in arrays.items()}
    intervals = {
        name: _bootstrap(values, draws, seed + index)
        for index, (name, values) in enumerate(arrays.items())
    }
    layers = np.arange(1, scores.shape[2] + 1)

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "jlens_layerwise_compression.npz",
        question_ids=qids,
        layers=layers,
        **arrays,
        **{f"mean_{name}": value for name, value in means.items()},
        **{f"ci_low_{name}": value[0] for name, value in intervals.items()},
        **{f"ci_high_{name}": value[1] for name, value in intervals.items()},
    )

    selected_layers = [layer for layer in (1, 16, 24, 32, 40, 48, 52, 56, 60, 64) if layer <= len(layers)]
    summary = {
        "n_questions": len(qids),
        "layers": layers.tolist(),
        "spread_definition": "standard deviation across the four within-question centered JLens A-D scores",
        "compression_definition": "Baseline spread minus condition spread; positive means flatter than Baseline",
        "entropy_definition": "softmax entropy in bits across the four centered JLens A-D scores",
        "selected_layers": {},
    }
    for layer in selected_layers:
        index = layer - 1
        summary["selected_layers"][str(layer)] = {
            name: {
                "mean": float(means[name][index]),
                "ci": [float(intervals[name][0][index]), float(intervals[name][1][index])],
            }
            for name in arrays
        }
    (output / "jlens_layerwise_compression_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    colors = {"baseline": "#777777", "game": "#2f90f5", "neutral": "#ef7f35"}
    for axis, prefix, title, ylabel in (
        (axes[0, 0], "spread", "A  A-D evidence spread", "JLens score SD"),
        (axes[1, 0], "entropy", "C  A-D entropy", "Entropy (bits)"),
    ):
        for condition, label in (("baseline", "Baseline"), ("game", "Game"), ("neutral", "Neutral")):
            name = f"{prefix}_{condition}"
            axis.plot(layers, means[name], color=colors[condition], linewidth=2, label=label)
            axis.fill_between(layers, *intervals[name], color=colors[condition], alpha=0.12, linewidth=0)
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.2)

    for axis, names, title, ylabel in (
        (
            axes[0, 1],
            (("compression_game_vs_baseline", "Game", colors["game"]),
             ("compression_neutral_vs_baseline", "Neutral", colors["neutral"])),
            "B  Compression relative to Baseline",
            "Baseline spread minus condition spread",
        ),
        (
            axes[1, 1],
            (("entropy_increase_game_vs_baseline", "Game", colors["game"]),
             ("entropy_increase_neutral_vs_baseline", "Neutral", colors["neutral"])),
            "D  Entropy increase relative to Baseline",
            "Condition minus Baseline (bits)",
        ),
    ):
        axis.axhline(0, color="#777777", linewidth=1)
        for name, label, color in names:
            axis.plot(layers, means[name], color=color, linewidth=2, label=label)
            axis.fill_between(layers, *intervals[name], color=color, alpha=0.12, linewidth=0)
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.2)

    for axis in axes.flat:
        axis.set_xlim(1, 64)
        axis.set_xticks((1, 8, 16, 24, 32, 40, 48, 56, 64))
    for axis in axes[1]:
        axis.set_xlabel("Residual readout")
    fig.suptitle("JLens A-D compression at the final decision position (SimpleMC, n=500)", y=0.995)
    fig.tight_layout()
    fig.savefig(output / "jlens_layerwise_compression.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "jlens_layerwise_compression.svg", bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# JLens layerwise A-D compression",
        "",
        f"Matched corrected-prompt SimpleMC questions: **{len(qids)}**.",
        "",
        "Spread is the within-question standard deviation of the four centered JLens",
        "answer scores. Compression is Baseline spread minus condition spread, so",
        "positive values mean the condition is flatter than Baseline.",
        "",
        "| Readout | Baseline spread | Game spread | Neutral spread | Game compression | Neutral compression | Game entropy increase | Neutral entropy increase |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in selected_layers:
        row = summary["selected_layers"][str(layer)]
        value = lambda name: row[name]["mean"]
        lines.append(
            f"| {layer} | {value('spread_baseline'):.3f} | {value('spread_game'):.3f} | "
            f"{value('spread_neutral'):.3f} | {value('compression_game_vs_baseline'):+.3f} | "
            f"{value('compression_neutral_vs_baseline'):+.3f} | "
            f"{value('entropy_increase_game_vs_baseline'):+.3f} | "
            f"{value('entropy_increase_neutral_vs_baseline'):+.3f} |"
        )
    (output / "JLENS_LAYERWISE_COMPRESSION_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jlens-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(args.jlens_root, args.output, args.draws, args.seed)


if __name__ == "__main__":
    main()
