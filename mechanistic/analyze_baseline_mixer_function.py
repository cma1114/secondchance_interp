from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("Mixer 56", "Mixer 63", "Both")
RANKS = ("Rank 1", "Rank 2", "Rank 3", "Rank 4")
COLORS = ("#1689d8", "#e66b19", "#4cae68")


def _entropy(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(centered)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -np.sum(probabilities * np.log(probabilities + 1e-30), axis=-1)


def _spread(logits: np.ndarray) -> np.ndarray:
    return np.std(logits - logits.mean(axis=-1, keepdims=True), axis=-1)


def _winner_advantage(logits: np.ndarray, order: np.ndarray) -> np.ndarray:
    aligned = np.take_along_axis(logits, order, axis=-1)
    return aligned[..., 0] - aligned[..., 1:].mean(axis=-1)


def _rank_delta(patched: np.ndarray, natural: np.ndarray, order: np.ndarray) -> np.ndarray:
    delta = patched - natural[:, None, :]
    delta -= delta.mean(axis=-1, keepdims=True)
    return np.take_along_axis(delta, order[:, None, :], axis=-1)


def _aligned(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    return np.take_along_axis(centered, order[:, None, :], axis=-1)


def _bootstrap_indices(labels: np.ndarray, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.empty((draws, len(labels)), dtype=np.int32)
    groups = [np.flatnonzero(labels == value) for value in range(4)]
    for draw in range(draws):
        parts = [rng.choice(group, len(group), replace=True) for group in groups]
        rows[draw] = np.concatenate(parts)
    return rows


def _summary(values: np.ndarray, indices: np.ndarray):
    point = np.mean(values, axis=0)
    draws = np.mean(values[indices], axis=1)
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return point, low, high


def _metric_effects(
    natural: np.ndarray,
    patched: np.ndarray,
    order: np.ndarray,
    correct: np.ndarray,
    reference_answers: np.ndarray,
):
    natural_answer = np.argmax(natural, axis=1)
    patched_answer = np.argmax(patched, axis=2)
    natural_entropy = _entropy(natural)
    patched_entropy = _entropy(patched)
    natural_spread = _spread(natural)
    patched_spread = _spread(patched)
    natural_advantage = _winner_advantage(natural, order)
    patched_advantage = _winner_advantage(patched, np.repeat(order[:, None, :], 3, axis=1))
    return {
        "answer_change_rate": (patched_answer != reference_answers[:, None]).astype(float),
        "accuracy_change": (patched_answer == correct[:, None]).astype(float)
        - (natural_answer == correct).astype(float)[:, None],
        "entropy_change": patched_entropy - natural_entropy[:, None],
        "spread_change": patched_spread - natural_spread[:, None],
        "winner_advantage_change": patched_advantage - natural_advantage[:, None],
    }


def _format_ci(point: float, low: float, high: float, scale: float = 1.0, suffix: str = ""):
    return f"{point * scale:+.3f}{suffix} [{low * scale:+.3f}, {high * scale:+.3f}]"


def _plot(
    path: Path,
    immediate_stats,
    ablation_stats,
    insertion_stats,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=False)
    x = np.arange(4)

    for component, color, stats in zip(COMPONENTS[:2], COLORS[:2], immediate_stats):
        point, low, high = stats
        axes[0].plot(x, point, marker="o", lw=2.4, color=color, label=component)
        axes[0].fill_between(x, low, high, color=color, alpha=0.22)
    axes[0].axhline(0, color="#555", lw=1, ls="--")
    axes[0].set_title("A  Immediate Baseline JLens write", loc="left", weight="bold")
    axes[0].set_ylabel("Centered A–D contribution (score units)")
    axes[0].legend(frameon=False)

    for component, color, stats in zip(COMPONENTS, COLORS, ablation_stats):
        point, low, high = stats
        axes[1].plot(x, point, marker="o", lw=2.4, color=color, label=component)
        axes[1].fill_between(x, low, high, color=color, alpha=0.20)
    axes[1].axhline(0, color="#555", lw=1, ls="--")
    axes[1].set_title("B  Mean-ablate in Baseline", loc="left", weight="bold")
    axes[1].set_ylabel("Change in final canonical logit")
    axes[1].legend(frameon=False)

    for component, color, stats in zip(COMPONENTS, COLORS, insertion_stats):
        point, low, high = stats
        axes[2].plot(x, point, marker="o", lw=2.4, color=color, label=component)
        axes[2].fill_between(x, low, high, color=color, alpha=0.20)
    axes[2].axhline(0, color="#555", lw=1, ls="--")
    axes[2].set_title("C  Insert Baseline outputs into Game", loc="left", weight="bold")
    axes[2].set_ylabel("Change in final canonical logit")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.set_xticks(x, RANKS)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        -0.01,
        "Ranks are defined by each question's natural Baseline logits; bands are paired, Baseline-letter-stratified 95% bootstrap CIs.",
        ha="center",
        fontsize=10.5,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def analyze(
    results_path: Path,
    metadata_path: Path,
    output: Path,
    figure_path: Path,
    draws: int,
    seed: int,
):
    output.mkdir(parents=True, exist_ok=True)
    with np.load(results_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    if not np.all(data["completed"]):
        raise ValueError("Cannot analyze an incomplete run")
    baseline = data["natural_baseline_logits"].astype(float)
    game = data["natural_game_logits"].astype(float)
    ablation = data["baseline_mean_ablation_logits"].astype(float)
    insertion = data["game_baseline_insertion_logits"].astype(float)
    correct = data["correct_indices"].astype(int)
    baseline_answers = np.argmax(baseline, axis=1)
    order = np.argsort(-baseline, axis=1)
    bootstrap = _bootstrap_indices(baseline_answers, draws, seed)

    immediate = data["baseline_immediate_jlens_write"].astype(float)
    immediate_ranked = _aligned(immediate, order)
    ablation_ranked = _rank_delta(ablation, baseline, order)
    insertion_ranked = _rank_delta(insertion, game, order)
    immediate_stats = [_summary(immediate_ranked[:, i], bootstrap) for i in range(2)]
    ablation_stats = [_summary(ablation_ranked[:, i], bootstrap) for i in range(3)]
    insertion_stats = [_summary(insertion_ranked[:, i], bootstrap) for i in range(3)]

    baseline_metrics = _metric_effects(
        baseline, ablation, order, correct, baseline_answers
    )
    game_metrics = _metric_effects(
        game, insertion, order, correct, baseline_answers
    )
    # For insertion, the behaviorally relevant answer-change outcome is the
    # change in Game switching relative to the natural Baseline answer.
    natural_game_switch = (np.argmax(game, axis=1) != baseline_answers).astype(float)
    patched_game_switch = (
        np.argmax(insertion, axis=2) != baseline_answers[:, None]
    ).astype(float)
    game_metrics["switch_rate_change"] = patched_game_switch - natural_game_switch[:, None]

    metric_summary: dict[str, dict[str, list[dict[str, float]]]] = {
        "baseline_mean_ablation": {},
        "baseline_into_game": {},
    }
    for group, metrics in (
        ("baseline_mean_ablation", baseline_metrics),
        ("baseline_into_game", game_metrics),
    ):
        for name, values in metrics.items():
            point, low, high = _summary(values, bootstrap)
            metric_summary[group][name] = [
                {"estimate": float(point[i]), "ci": [float(low[i]), float(high[i])]}
                for i in range(3)
            ]

    summary = {
        "n_confirmation": int(len(baseline)),
        "bootstrap_draws": int(draws),
        "natural": {
            "baseline_accuracy": float(np.mean(baseline_answers == correct)),
            "game_accuracy": float(np.mean(np.argmax(game, axis=1) == correct)),
            "game_switch_rate": float(np.mean(natural_game_switch)),
            "baseline_full_vocab_top_is_ad": float(
                np.mean(np.isin(data["baseline_full_top_ids"], data["canonical_token_ids"]))
            ),
        },
        "metrics": metric_summary,
        "immediate_rank_writes": {
            COMPONENTS[i]: {
                RANKS[r]: {
                    "estimate": float(immediate_stats[i][0][r]),
                    "ci": [float(immediate_stats[i][1][r]), float(immediate_stats[i][2][r])],
                }
                for r in range(4)
            }
            for i in range(2)
        },
        "baseline_ablation_rank_changes": {
            COMPONENTS[i]: {
                RANKS[r]: {
                    "estimate": float(ablation_stats[i][0][r]),
                    "ci": [float(ablation_stats[i][1][r]), float(ablation_stats[i][2][r])],
                }
                for r in range(4)
            }
            for i in range(3)
        },
        "baseline_into_game_rank_changes": {
            COMPONENTS[i]: {
                RANKS[r]: {
                    "estimate": float(insertion_stats[i][0][r]),
                    "ci": [float(insertion_stats[i][1][r]), float(insertion_stats[i][2][r])],
                }
                for r in range(4)
            }
            for i in range(3)
        },
        "run_metadata": json.loads(metadata_path.read_text()),
    }
    (output / "baseline_mixer_function_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    _plot(figure_path, immediate_stats, ablation_stats, insertion_stats)

    lines = [
        "# Baseline function of Qwen3.6-27B Mixers 56 and 63",
        "",
        f"Confirmation questions: **{len(baseline)}**. The equal-letter mean-ablation source was estimated on the disjoint 251-question discovery set. All prompts use the canonical `baseline_matched_empty_history` explicit ChatML format.",
        "",
        "## Natural behavior",
        "",
        f"- Baseline accuracy: {100 * summary['natural']['baseline_accuracy']:.1f}%",
        f"- Game accuracy: {100 * summary['natural']['game_accuracy']:.1f}%",
        f"- Natural Game switch rate relative to Baseline: {100 * summary['natural']['game_switch_rate']:.1f}%",
        "",
        "## Causal effects",
        "",
        "Mean ablation replaces a Baseline mixer output with the answer-letter-balanced discovery-set mean. Baseline-into-Game insertion replaces a Game output with the paired same-question natural Baseline output. Values below are paired changes from the natural target condition.",
        "",
        "| Intervention | Baseline answer changed | Baseline accuracy | Baseline spread | Baseline winner advantage | Game switch rate | Game spread | Game winner advantage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, label in enumerate(COMPONENTS):
        bm = metric_summary["baseline_mean_ablation"]
        gm = metric_summary["baseline_into_game"]
        row = [
            label,
            _format_ci(**{
                "point": bm["answer_change_rate"][i]["estimate"],
                "low": bm["answer_change_rate"][i]["ci"][0],
                "high": bm["answer_change_rate"][i]["ci"][1],
            }, scale=100, suffix=" pp"),
            _format_ci(
                bm["accuracy_change"][i]["estimate"],
                *bm["accuracy_change"][i]["ci"],
                scale=100,
                suffix=" pp",
            ),
            _format_ci(
                bm["spread_change"][i]["estimate"], *bm["spread_change"][i]["ci"]
            ),
            _format_ci(
                bm["winner_advantage_change"][i]["estimate"],
                *bm["winner_advantage_change"][i]["ci"],
            ),
            _format_ci(
                gm["switch_rate_change"][i]["estimate"],
                *gm["switch_rate_change"][i]["ci"],
                scale=100,
                suffix=" pp",
            ),
            _format_ci(
                gm["spread_change"][i]["estimate"], *gm["spread_change"][i]["ci"]
            ),
            _format_ci(
                gm["winner_advantage_change"][i]["estimate"],
                *gm["winner_advantage_change"][i]["ci"],
            ),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Panel A is a JLens finite-difference direct attribution, not a causal intervention: it asks what answer-aligned write is immediately present in each natural Baseline mixer output while downstream computation is held fixed. Panels B and C are causal final-logit effects.",
        "",
        "### Mixer 56",
        "",
        "Mixer 56 is an ordinary Baseline discrimination/sharpening component, although it is not specifically a winner booster. Its immediate JLens-aligned write is +0.407 to Baseline rank 1, +0.629 to rank 2, approximately zero to rank 3, and -1.018 to rank 4. Replacing its question-specific Baseline output with the discovery-set mean produces the inverse causal pattern: ranks 1 and 2 fall and rank 4 rises. Baseline spread decreases by 0.111 and original-winner advantage by 0.134; 3.6% of Baseline answers change, with no reliable accuracy effect.",
        "",
        "Putting the paired Baseline Mixer-56 output into Game modestly restores sharpening: spread rises by 0.010 and original-winner advantage by 0.030. It does not, however, change the Game switch rate (0.0 percentage points, 95% CI -2.8 to +2.8).",
        "",
        "### Mixer 63",
        "",
        "Mixer 63 performs the opposite operation even during ordinary Baseline answering. Its immediate write is -0.316 to rank 1, -0.151 to rank 2, +0.090 to rank 3, and +0.377 to rank 4. It is therefore a late rank-opposed flattening/rebalancing component, not a normal winner-sharpening component. Mean-ablation removes part of that operation: Baseline spread increases by 0.032 and winner advantage by 0.031; 6.4% of answers change, again with no reliable accuracy effect.",
        "",
        "Inserting the paired Baseline Mixer-63 output into Game makes Game **more**, not less, compressed: spread falls by 0.076 and winner advantage by 0.098. Switching rises by 4.8 percentage points, although its interval narrowly includes zero (-0.4 to +10.0).",
        "",
        "### Consequence for the eight-output mechanism",
        "",
        "Jointly inserting the Baseline outputs of Mixers 56 and 63 into Game does not restore ordinary behavior: the switch-rate change is -1.6 points (95% CI -6.8 to +3.6), spread falls by 0.047, and winner advantage does not change reliably. Their opposing natural roles largely cancel.",
        "",
        "Thus the prior eight-output mediation result should not be described as Game simply suppressing ordinary late answer sharpening. Mixer 56 supplies normal question-specific discrimination, while Mixer 63 supplies normal late flattening. Their importance to the Game–Neutral intervention is contextual and depends on the coordinated sequence of late mixer states. The upstream mechanism and the relevant interacting features remain unidentified.",
        "",
        f"Figure: `{figure_path}`",
        "",
        "Numerical results: `baseline_mixer_function_summary.json` and `baseline_mixer_function_results.npz`.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(
        args.results,
        args.metadata,
        args.output,
        args.figure,
        args.bootstrap,
        args.seed,
    )


if __name__ == "__main__":
    main()
