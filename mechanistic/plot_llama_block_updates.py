from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .data import load_activation_dataset
from .leader_event_analysis import _compression, _relative
from .trajectory_analysis import centered


CONDITIONS = ("baseline", "incorrect", "neutral")
DISPLAY_NAMES = {"baseline": "Baseline", "incorrect": "Game", "neutral": "Neutral"}
COLORS = {"baseline": "#4C78A8", "incorrect": "#D62728", "neutral": "#2CA02C"}


def _bootstrap_curve(
    values: np.ndarray, rng: np.random.Generator, repetitions: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the question mean and percentile bootstrap interval by block."""
    mean = values.mean(axis=0)
    n_questions = values.shape[0]
    bootstrap = np.empty((repetitions, values.shape[1]), dtype=np.float64)
    for start in range(0, repetitions, 100):
        stop = min(start + 100, repetitions)
        indices = rng.integers(0, n_questions, size=(stop - start, n_questions))
        bootstrap[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return mean, low, high


def _condition_curves(input_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    data = load_activation_dataset(input_dir, list(CONDITIONS))
    scores = centered(data.logits)
    game = scores[:, CONDITIONS.index("incorrect")]

    # This preserves the estimand used in the report: define the candidate from
    # the Game trajectory immediately before each block, then evaluate that
    # same candidate's relative update in every paired condition.
    game_leader = np.argmax(game[:, :-1], axis=-1)
    curves: dict[str, dict[str, np.ndarray]] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        condition_scores = scores[:, condition_index]
        update = condition_scores[:, 1:] - condition_scores[:, :-1]
        curves[condition] = {
            "leader_update": _relative(update, game_leader),
            "compression": _compression(condition_scores),
        }
    return curves


def _write_curves(
    path: Path,
    dataset_curves: dict[str, dict[str, dict[str, np.ndarray]]],
    repetitions: int,
    seed: int,
) -> None:
    rows: list[dict[str, object]] = []
    for dataset_index, (dataset, condition_curves) in enumerate(dataset_curves.items()):
        rng = np.random.default_rng(seed + dataset_index)
        for condition in CONDITIONS:
            for metric in ("leader_update", "compression"):
                mean, low, high = _bootstrap_curve(
                    condition_curves[condition][metric], rng, repetitions
                )
                for block, (estimate, lo, hi) in enumerate(zip(mean, low, high)):
                    rows.append(
                        {
                            "dataset": dataset,
                            "condition": DISPLAY_NAMES[condition],
                            "metric": metric,
                            "block_from": block,
                            "mean": float(estimate),
                            "ci_low": float(lo),
                            "ci_high": float(hi),
                            "n": int(condition_curves[condition][metric].shape[0]),
                        }
                    )
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    output_path: Path,
    dataset_curves: dict[str, dict[str, dict[str, np.ndarray]]],
    repetitions: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey="col")
    metrics = (
        (
            "leader_update",
            "Update to Game's pre-block leader",
            "Relative A–D pseudo-logit update",
        ),
        (
            "compression",
            "Compression of centered A–D evidence",
            "Proportional compression per block",
        ),
    )

    for dataset_index, (dataset, condition_curves) in enumerate(dataset_curves.items()):
        rng = np.random.default_rng(seed + dataset_index)
        cached: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for condition in CONDITIONS:
            for metric, _, _ in metrics:
                cached[(condition, metric)] = _bootstrap_curve(
                    condition_curves[condition][metric], rng, repetitions
                )

        for metric_index, (metric, title, ylabel) in enumerate(metrics):
            axis = axes[dataset_index, metric_index]
            zoom = inset_axes(axis, width="39%", height="43%", loc="lower right", borderpad=1.3)
            for condition in CONDITIONS:
                mean, low, high = cached[(condition, metric)]
                blocks = np.arange(len(mean))
                label = DISPLAY_NAMES[condition]
                color = COLORS[condition]
                axis.plot(blocks, mean, color=color, lw=1.55, label=label)
                axis.fill_between(blocks, low, high, color=color, alpha=0.13, linewidth=0)
                axis.scatter([39], [mean[39]], color=color, s=25, zorder=5)

                zoom.plot(blocks, mean, color=color, lw=1.45)
                zoom.fill_between(blocks, low, high, color=color, alpha=0.13, linewidth=0)
                zoom.scatter([39], [mean[39]], color=color, s=20, zorder=5)

            axis.axhline(0, color="black", lw=0.65, alpha=0.65)
            axis.axvline(39, color="black", lw=1.0, ls="--", alpha=0.85)
            axis.set_title(f"{dataset}: {title}", loc="left", fontweight="bold")
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", color="#dddddd", lw=0.6)
            if dataset_index == 1:
                axis.set_xlabel("Transformer block (update from block b to b+1)")

            zoom.set_xlim(33, 47)
            zoom_low = min(cached[(condition, metric)][1][33:48].min() for condition in CONDITIONS)
            zoom_high = max(cached[(condition, metric)][2][33:48].max() for condition in CONDITIONS)
            zoom_padding = max(0.08, 0.08 * (zoom_high - zoom_low))
            zoom.set_ylim(zoom_low - zoom_padding, zoom_high + zoom_padding)
            zoom.axhline(0, color="black", lw=0.55, alpha=0.65)
            zoom.axvline(39, color="black", lw=0.8, ls="--", alpha=0.85)
            zoom.set_xticks([34, 39, 44])
            zoom.tick_params(labelsize=7)
            zoom.set_title("Blocks 33–47", fontsize=8, pad=2)
            zoom.grid(axis="y", color="#e4e4e4", lw=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        "Llama 3.1 405B: blockwise leader update and A–D compression",
        y=1.025,
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "Leader is defined from the Game trajectory before each block and the same option is traced in all conditions. "
        "Shading is a question-bootstrap 95% CI; dashed line marks block 39.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(output_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Llama blockwise leader and compression curves")
    parser.add_argument("--simplemc", required=True)
    parser.add_argument("--triviamc", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_curves = {
        "SimpleMC": _condition_curves(args.simplemc),
        "TriviaMC": _condition_curves(args.triviamc),
    }
    _write_curves(
        output_path.with_name(output_path.name + "_data.csv"),
        dataset_curves,
        args.bootstrap,
        args.seed,
    )
    _plot(output_path, dataset_curves, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
