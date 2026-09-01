from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS


TASKS = ("Game", "Neutral")
SCENARIO_LABELS = {
    "natural": "Natural",
    "layer40_2p_options_swapped": "L40 2P lines\nswapped",
    "layer40_2p_options_ablated": "L40 2P lines\nablated",
    "layers52_56_scaffold_swapped": "L52/56 scaffold\nswapped",
    "layers52_56_scaffold_ablated": "L52/56 scaffold\nablated",
    "all_layers_1p_options_ablated": "All-layer 1P lines\nablated",
}


def _bootstrap_mean(
    values: np.ndarray, mask: np.ndarray, seed: int, draws: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(values[..., mask], dtype=np.float32)
    point = selected.mean(axis=-1)
    rng = np.random.default_rng(seed)
    samples = np.empty((draws,) + point.shape, dtype=np.float32)
    for start in range(0, draws, 100):
        stop = min(start + 100, draws)
        rows = rng.integers(0, selected.shape[-1], size=(stop - start, selected.shape[-1]))
        samples[start:stop] = selected[..., rows].mean(axis=-1).transpose(
            -1, *range(selected.ndim - 1)
        )
    low, high = np.quantile(samples, (0.025, 0.975), axis=0)
    return point, low, high


def _bootstrap_rank(
    values: np.ndarray, mask: np.ndarray, seed: int, draws: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Values are task x scenario x question x rank."""
    selected = np.asarray(values[:, :, mask], dtype=np.float32)
    point = selected.mean(axis=2)
    rng = np.random.default_rng(seed)
    samples = np.empty((draws,) + point.shape, dtype=np.float32)
    for start in range(0, draws, 100):
        stop = min(start + 100, draws)
        rows = rng.integers(0, selected.shape[2], size=(stop - start, selected.shape[2]))
        samples[start:stop] = selected[:, :, rows].mean(axis=3).transpose(2, 0, 1, 3)
    low, high = np.quantile(samples, (0.025, 0.975), axis=0)
    return point, low, high


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def analyze(args: argparse.Namespace) -> None:
    import matplotlib

    causal = np.load(args.causal_results)
    qids = [str(value) for value in causal["question_ids"]]
    if len(qids) != 500 or not causal["completed"].all():
        raise RuntimeError("Complete 500-question causal result is required")
    scenarios = [str(value) for value in causal["scenario_ids"]]
    if scenarios != list(SCENARIO_LABELS):
        raise RuntimeError(f"Causal scenarios changed: {scenarios}")
    logits = causal["scenario_final_logits"].astype(np.float64)
    if not np.isfinite(logits).all():
        raise RuntimeError("Causal logits contain non-finite values")

    baseline = json.loads(args.baseline.read_text())["results"]
    fresh = json.loads(args.remapped_baseline.read_text())["results"]
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    if [int(discovery.sum()), int((~discovery).sum())] != [251, 249]:
        raise RuntimeError("Frozen split changed")

    rank_order = np.empty((500, 4), dtype=np.int64)
    w1_current = np.empty(500, dtype=np.int64)
    w2_current = np.empty(500, dtype=np.int64)
    for qi, qid in enumerate(qids):
        old = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=np.float64)
        semantic_rank = np.argsort(-old, kind="stable")
        current = [
            LETTERS.index(mappings[qid]["original_to_new"][LETTERS[int(index)]])
            for index in semantic_rank
        ]
        rank_order[qi] = current
        w1_current[qi] = current[0]
        w2_current[qi] = int(
            np.argmax(np.asarray(fresh[qid]["aggregated_ad_logits"], dtype=np.float64))
        )
    conflict = w1_current != w2_current

    choices = np.argmax(logits, axis=-1)
    choose_w1 = choices == w1_current[None, None, :]
    switch = ~choose_w1
    choose_w2 = choices == w2_current[None, None, :]
    centered = _center(logits)
    ranked = np.empty_like(centered)
    for qi in range(500):
        ranked[:, :, qi] = np.take(centered[:, :, qi], rank_order[qi], axis=-1)
    bivalent = ranked[..., 3] - ranked[..., :2].mean(axis=-1)

    split_masks = {"discovery": discovery, "confirmation": ~discovery}
    result: dict[str, Any] = {
        "question": (
            "Which incoming final-query sources causally carry the task-specific "
            "old-rank transformation into the final answer?"
        ),
        "evidence_label": "Causal final-query-only ordinary-attention intervention.",
        "coverage": {
            "questions": 500,
            "discovery": 251,
            "confirmation": 249,
            "tasks": list(TASKS),
            "scenarios": scenarios,
            "complete_model_forwards": 1500,
        },
        "definitions": {
            "W1": "winner under the original 1P answer logits, mapped into the remapped 2P display order",
            "W2": "winner under the fresh remapped 2P answer logits",
            "switch": "final answer is not semantic W1",
            "conflict": "W1 and W2 are different semantic candidates",
            "rank_logits": "final A-D logits centered within question and ordered by 1P rank R1-R4",
        },
        "splits": {},
    }
    for split_index, (split_name, mask) in enumerate(split_masks.items()):
        split_rows: dict[str, Any] = {}
        for subset_name, subset in (
            ("all", mask),
            ("conflict", mask & conflict),
            ("no_conflict", mask & ~conflict),
        ):
            if not subset.any():
                raise RuntimeError(f"Empty {split_name} {subset_name} subset")
            sw_point, sw_low, sw_high = _bootstrap_mean(
                switch.astype(np.float32), subset, args.seed + 10 * split_index, args.bootstrap_draws
            )
            w1_point, w1_low, w1_high = _bootstrap_mean(
                choose_w1.astype(np.float32), subset, args.seed + 100 + 10 * split_index, args.bootstrap_draws
            )
            w2_point, w2_low, w2_high = _bootstrap_mean(
                choose_w2.astype(np.float32), subset, args.seed + 200 + 10 * split_index, args.bootstrap_draws
            )
            rank_point, rank_low, rank_high = _bootstrap_rank(
                ranked, subset, args.seed + 300 + 10 * split_index, args.bootstrap_draws
            )
            switch_delta_point, switch_delta_low, switch_delta_high = _bootstrap_mean(
                switch.astype(np.float32) - switch[:, :1].astype(np.float32),
                subset,
                args.seed + 400 + 10 * split_index,
                args.bootstrap_draws,
            )
            bivalent_delta_point, bivalent_delta_low, bivalent_delta_high = _bootstrap_mean(
                bivalent.astype(np.float32) - bivalent[:, :1].astype(np.float32),
                subset,
                args.seed + 500 + 10 * split_index,
                args.bootstrap_draws,
            )
            subset_rows: dict[str, Any] = {"questions": int(subset.sum()), "tasks": {}}
            for task_index, task in enumerate(TASKS):
                task_rows: dict[str, Any] = {}
                for scenario_index, scenario in enumerate(scenarios):
                    task_rows[scenario] = {
                        "switch_rate": float(sw_point[task_index, scenario_index]),
                        "switch_ci": [
                            float(sw_low[task_index, scenario_index]),
                            float(sw_high[task_index, scenario_index]),
                        ],
                        "choose_W1_rate": float(w1_point[task_index, scenario_index]),
                        "choose_W1_ci": [
                            float(w1_low[task_index, scenario_index]),
                            float(w1_high[task_index, scenario_index]),
                        ],
                        "choose_W2_rate": float(w2_point[task_index, scenario_index]),
                        "choose_W2_ci": [
                            float(w2_low[task_index, scenario_index]),
                            float(w2_high[task_index, scenario_index]),
                        ],
                        "rank_logit_mean": rank_point[task_index, scenario_index].tolist(),
                        "rank_logit_ci_low": rank_low[task_index, scenario_index].tolist(),
                        "rank_logit_ci_high": rank_high[task_index, scenario_index].tolist(),
                        "paired_switch_change_vs_natural": float(
                            switch_delta_point[task_index, scenario_index]
                        ),
                        "paired_switch_change_vs_natural_ci": [
                            float(switch_delta_low[task_index, scenario_index]),
                            float(switch_delta_high[task_index, scenario_index]),
                        ],
                        "paired_bivalent_logit_change_vs_natural": float(
                            bivalent_delta_point[task_index, scenario_index]
                        ),
                        "paired_bivalent_logit_change_vs_natural_ci": [
                            float(bivalent_delta_low[task_index, scenario_index]),
                            float(bivalent_delta_high[task_index, scenario_index]),
                        ],
                    }
                subset_rows["tasks"][task] = task_rows
            split_rows[subset_name] = subset_rows
        result["splits"][split_name] = split_rows

    natural_error = float(
        np.max(np.abs(logits[:, 0] - causal["trusted_natural_logits"].astype(float)))
    )
    result["validation"] = {
        "all_questions_completed": True,
        "all_values_finite": True,
        "corrected_natural_max_abs_error": natural_error,
        "source_counts_task_aligned": {
            key: bool(np.array_equal(causal[key][0], causal[key][1]))
            for key in (
                "layer40_source_count",
                "layer52_source_count",
                "layer56_source_count",
                "all_1p_source_count",
            )
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    score = json.loads(args.score_summary.read_text())
    attention = np.load(args.attention_arrays)
    components = np.load(args.component_arrays)
    labels = attention["display_labels"].astype(str).tolist()
    labels[attention["source_names"].astype(str).tolist().index("final_assistant_prefix")] = (
        "Final assistant prefix + final query"
    )
    attention_values = attention["attention_mass"].astype(float)[:, :, ~discovery].mean(axis=2)

    component_values = components["decoded_delta"].astype(float)
    component_ranked = np.empty_like(component_values)
    for qi in range(500):
        component_ranked[:, qi] = np.take(
            component_values[:, qi], rank_order[qi], axis=-1
        )
    component_mean = component_ranked[:, ~discovery].mean(axis=1)
    component_bivalent = component_mean[..., 0, 3] - component_mean[..., 0, :2].mean(axis=-1)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 20), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=(1.0, 1.5, 1.5, 1.15))

    ax = fig.add_subplot(grid[0, 0])
    layers = np.arange(1, 65)
    styles = {
        ("old_unique", "Game"): ("#c44e52", "-"),
        ("old_unique", "Neutral"): ("#4c72b0", "-"),
        ("fresh_unique", "Game"): ("#c44e52", "--"),
        ("fresh_unique", "Neutral"): ("#4c72b0", "--"),
    }
    for target in ("old_unique", "fresh_unique"):
        for task in TASKS:
            values = [
                score["trajectory"][str(layer)][target]["tasks"][task][
                    "confirmation_correlation"
                ]
                for layer in layers
            ]
            color, linestyle = styles[(target, task)]
            ax.plot(
                layers,
                values,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=f"{task}, {'old 1P' if target == 'old_unique' else 'fresh 2P'}",
            )
    ax.set_title("A  Final-position evidence decoding (held-out)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Correlation with independent score")
    ax.legend(ncol=2, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.7)

    ax = fig.add_subplot(grid[0, 1])
    for task_index, task in enumerate(TASKS):
        ax.plot(
            layers,
            component_bivalent[task_index, :, 0],
            linewidth=2.2,
            color=("#c44e52", "#4c72b0")[task_index],
            label=f"{task} mixer",
        )
        ax.plot(
            layers,
            component_bivalent[task_index, :, 1],
            linewidth=1.3,
            linestyle="--",
            color=("#c44e52", "#4c72b0")[task_index],
            label=f"{task} MLP",
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("B  Component writes into old-rank geometry")
    ax.set_xlabel("Layer")
    ax.set_ylabel("R4 − mean(R1,R2) decoded contribution")
    ax.set_ylim(-0.28, 0.12)
    ax.text(
        0.01,
        0.98,
        (
            "L1 is off-scale: mixer Game "
            f"{component_bivalent[0, 0, 0]:.2f}, Neutral "
            f"{component_bivalent[1, 0, 0]:.2f}; MLP "
            f"{component_bivalent[:, 0, 1].mean():+.2f}."
        ),
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )
    ax.legend(ncol=2, fontsize=9)

    vmax = float(attention_values.max() * 100)
    for task_index, task in enumerate(TASKS):
        ax = fig.add_subplot(grid[1 + task_index, :])
        image = ax.imshow(
            100 * attention_values[task_index].T,
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(f"{'C' if task_index == 0 else 'D'}  {task}: absolute final-query attention")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks(np.arange(len(attention["ordinary_layers"])))
        ax.set_xticklabels(attention["ordinary_layers"].astype(int))
        ax.set_xlabel("Ordinary-attention layer")
        fig.colorbar(image, ax=ax, pad=0.01, label="Mean attention mass (%)")

    confirmation = result["splits"]["confirmation"]["all"]
    ax = fig.add_subplot(grid[3, 0])
    x = np.arange(len(scenarios))
    width = 0.36
    for task_index, task in enumerate(TASKS):
        rows = confirmation["tasks"][task]
        means = np.asarray([rows[s]["switch_rate"] for s in scenarios])
        cis = np.asarray([rows[s]["switch_ci"] for s in scenarios])
        ax.bar(
            x + (task_index - 0.5) * width,
            100 * means,
            width,
            color=("#c44e52", "#4c72b0")[task_index],
            label=task,
            yerr=np.stack([100 * (means - cis[:, 0]), 100 * (cis[:, 1] - means)]),
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=8)
    ax.set_ylabel("Switch away from W1 (%)")
    ax.set_title("E  Causal behavior (held-out 249 questions)")
    ax.legend()

    ax = fig.add_subplot(grid[3, 1])
    for task_index, task in enumerate(TASKS):
        rows = confirmation["tasks"][task]
        values = []
        for scenario in scenarios:
            rank = np.asarray(rows[scenario]["rank_logit_mean"])
            values.append(rank[3] - rank[:2].mean())
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2,
            color=("#c44e52", "#4c72b0")[task_index],
            label=task,
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=8)
    ax.set_ylabel("Final logit: R4 − mean(R1,R2)")
    ax.set_title("F  Final old-rank policy shape (held-out)")
    ax.legend()

    fig.suptitle(
        "How the final decision position assembles old evidence, fresh evidence, and policy",
        fontsize=18,
        fontweight="bold",
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result["validation"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--score-summary", type=Path, required=True)
    parser.add_argument("--attention-arrays", type=Path, required=True)
    parser.add_argument("--component-arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=48334074)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
