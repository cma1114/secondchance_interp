from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LETTERS = "ABCD"
CONDITIONS = ("Game", "Neutral")


def _interval(values: np.ndarray, seed: int, draws: int = 5000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "n": int(len(values)),
    }


def _curve_interval(values: np.ndarray, seed: int, draws: int = 3000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return values.mean(axis=0), np.quantile(sampled, 0.025, axis=0), np.quantile(sampled, 0.975, axis=0)


def _aligned_logits(
    displayed: np.ndarray,
    qids: list[str],
    mapping: dict[str, dict[str, str]],
) -> np.ndarray:
    aligned = np.empty_like(displayed)
    for qi, qid in enumerate(qids):
        new_to_original = mapping[qid]["new_to_original"]
        for new_letter, original_content in new_to_original.items():
            aligned[..., qi, LETTERS.index(original_content)] = displayed[..., qi, LETTERS.index(new_letter)]
    return aligned


def analyze(
    discovery_path: Path,
    confirmation_path: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
    mapping_plan_path: Path,
    output: Path,
    figure_path: Path,
) -> dict:
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped = json.loads(remapped_baseline_path.read_text())["results"]
    mapping = {
        row["question_id"]: row
        for row in json.loads(mapping_plan_path.read_text())["rows"]
    }
    split_arrays = {
        "discovery": dict(np.load(discovery_path, allow_pickle=False)),
        "confirmation": dict(np.load(confirmation_path, allow_pickle=False)),
    }
    for name, arrays in split_arrays.items():
        if not bool(arrays["completed"].all()):
            raise ValueError(f"{name} is incomplete")
        if not all(np.all(np.isfinite(arrays[key])) for key in (
            "natural_logits", "lesioned_logits", "natural_projection", "lesioned_projection"
        )):
            raise ValueError(f"{name} contains non-finite values")

    pooled: dict[str, np.ndarray] = {}
    for key in split_arrays["discovery"]:
        if key in {"question_ids", "completed"}:
            continue
        pooled[key] = np.concatenate(
            [split_arrays["discovery"][key], split_arrays["confirmation"][key]], axis=1
        )
    pooled_qids = sum(
        (split_arrays[name]["question_ids"].astype(str).tolist() for name in ("discovery", "confirmation")),
        [],
    )
    split_arrays["pooled"] = {
        "question_ids": np.asarray(pooled_qids),
        "completed": np.ones(len(pooled_qids), dtype=bool),
        **pooled,
    }

    summary: dict[str, object] = {
        "definition": (
            "Effect is first-answer-boundary GLA-write lesion minus natural. "
            "Projection is the final-decision residual dot the exact layer-specific W1 semantic direction."
        ),
        "splits": {},
        "cross_host_validation": {},
    }
    seed = 81201
    curve_effects: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for split_index, (split, arrays) in enumerate(split_arrays.items()):
        qids = arrays["question_ids"].astype(str).tolist()
        w1 = np.asarray([LETTERS.index(baseline[qid]["answer"]) for qid in qids])
        w2 = np.asarray([
            LETTERS.index(remapped[qid]["answer_original_content"]) for qid in qids
        ])
        conflict = w1 != w2
        subsets = {
            "all": np.ones(len(qids), dtype=bool),
            "conflict": conflict,
            "no_conflict": ~conflict,
            "conflict_w1_a": conflict & (w1 == 0),
            "conflict_w1_bcd": conflict & (w1 != 0),
            "no_conflict_w1_a": (~conflict) & (w1 == 0),
            "no_conflict_w1_bcd": (~conflict) & (w1 != 0),
        }
        projection_effect = arrays["lesioned_projection"] - arrays["natural_projection"]
        summary["cross_host_validation"][split] = {
            "natural_max_abs_logit_difference": float(
                np.max(np.abs(arrays["natural_logits"] - arrays["natural_reference_logits"]))
            ),
            "lesion_max_abs_logit_difference": float(
                np.max(np.abs(arrays["lesioned_logits"] - arrays["lesion_reference_logits"]))
            ),
            "natural_ad_argmax_agreement": float(
                np.mean(
                    np.argmax(arrays["natural_logits"], axis=-1)
                    == np.argmax(arrays["natural_reference_logits"], axis=-1)
                )
            ),
            "lesion_ad_argmax_agreement": float(
                np.mean(
                    np.argmax(arrays["lesioned_logits"], axis=-1)
                    == np.argmax(arrays["lesion_reference_logits"], axis=-1)
                )
            ),
        }
        natural_aligned = _aligned_logits(arrays["natural_logits"], qids, mapping)
        lesion_aligned = _aligned_logits(arrays["lesioned_logits"], qids, mapping)
        logit_effect = lesion_aligned - natural_aligned
        split_result: dict[str, object] = {"n": len(qids), "subsets": {}}
        for subset_index, (subset_name, selected) in enumerate(subsets.items()):
            if not selected.any():
                continue
            subset_result: dict[str, object] = {"n": int(selected.sum()), "conditions": {}}
            for condition_index, condition in enumerate(CONDITIONS):
                final_projection = projection_effect[condition_index, selected, -1]
                natural_projection = arrays["natural_projection"][condition_index, selected, -1]
                lesioned_projection = arrays["lesioned_projection"][condition_index, selected, -1]
                w1_logit = logit_effect[condition_index, selected, w1[selected]]
                centered = logit_effect[condition_index, selected]
                w1_centered = w1_logit - centered.mean(axis=-1)
                condition_result = {
                    "final_projection_effect": _interval(
                        final_projection, seed + 1000 * split_index + 100 * subset_index + condition_index
                    ),
                    "final_natural_projection": _interval(
                        natural_projection, seed + 2000 + 1000 * split_index + 100 * subset_index + condition_index
                    ),
                    "final_absolute_projection_effect": _interval(
                        np.abs(lesioned_projection) - np.abs(natural_projection),
                        seed + 3000 + 1000 * split_index + 100 * subset_index + condition_index,
                    ),
                    "final_positive_projection_effect": _interval(
                        np.maximum(lesioned_projection, 0) - np.maximum(natural_projection, 0),
                        seed + 3500 + 1000 * split_index + 100 * subset_index + condition_index,
                    ),
                    "final_negative_magnitude_effect": _interval(
                        np.maximum(-lesioned_projection, 0) - np.maximum(-natural_projection, 0),
                        seed + 3750 + 1000 * split_index + 100 * subset_index + condition_index,
                    ),
                    "final_w1_centered_logit_effect": _interval(
                        w1_centered, seed + 4000 + 1000 * split_index + 100 * subset_index + condition_index
                    ),
                    "projection_effect_fraction_of_absolute_natural": float(
                        final_projection.mean() / np.maximum(np.abs(natural_projection).mean(), 1e-12)
                    ),
                }
                if subset_name.startswith("conflict"):
                    w2_logit = logit_effect[condition_index, selected, w2[selected]]
                    margin_effect = w1_logit - w2_logit
                    condition_result["final_w1_minus_w2_margin_effect"] = _interval(
                        margin_effect, seed + 6000 + 1000 * split_index + 100 * subset_index + condition_index
                    )
                    condition_result["projection_margin_effect_correlation"] = float(
                        np.corrcoef(final_projection, margin_effect)[0, 1]
                    )
                subset_result["conditions"][condition] = condition_result
            interaction = (
                projection_effect[0, selected, -1] - projection_effect[1, selected, -1]
            )
            subset_result["game_minus_neutral_projection_effect"] = _interval(
                interaction, seed + 8000 + 1000 * split_index + 100 * subset_index
            )
            split_result["subsets"][subset_name] = subset_result
        summary["splits"][split] = split_result
        curve_effects[split] = {
            name: {
                condition: projection_effect[ci, selected]
                for ci, condition in enumerate(CONDITIONS)
            }
            for name, selected in subsets.items()
        }

    output.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    layers = np.arange(1, 65)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    colors = {"Game": "#2f8df3", "Neutral": "#ef7d32"}
    panels = (
        (axes[0, 0], "discovery", "conflict", "A  Discovery: W1≠W2"),
        (axes[0, 1], "confirmation", "conflict", "B  Confirmation: W1≠W2"),
        (axes[1, 0], "pooled", "no_conflict", "C  Pooled: W1=W2"),
    )
    for panel_index, (ax, split, subset, title) in enumerate(panels):
        for condition_index, condition in enumerate(CONDITIONS):
            mean, low, high = _curve_interval(
                curve_effects[split][subset][condition], seed + 12000 + panel_index * 10 + condition_index
            )
            ax.plot(layers, mean, color=colors[condition], lw=2, label=condition)
            ax.fill_between(layers, low, high, color=colors[condition], alpha=0.2, linewidth=0)
        ax.axhline(0, color="#777777", lw=1)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Post-block residual readout")
        ax.set_ylabel("Lesioned − natural W1 projection\n(residual units)")
        ax.set_xlim(1, 64)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, loc="lower left")

    ax = axes[1, 1]
    for subset_index, (subset, label, color) in enumerate((
        ("conflict", "W1≠W2", "#8b5cf6"),
        ("no_conflict", "W1=W2", "#2ca25f"),
    )):
        interaction = (
            curve_effects["pooled"][subset]["Game"]
            - curve_effects["pooled"][subset]["Neutral"]
        )
        mean, low, high = _curve_interval(interaction, seed + 12100 + subset_index)
        ax.plot(layers, mean, color=color, lw=2, label=label)
        ax.fill_between(layers, low, high, color=color, alpha=0.2, linewidth=0)
    ax.axhline(0, color="#777777", lw=1)
    ax.set_title("D  Game-specific projection effect", loc="left", fontweight="bold")
    ax.set_xlabel("Post-block residual readout")
    ax.set_ylabel("Game effect − Neutral effect\n(residual units)")
    ax.set_xlim(1, 64)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="lower left")
    fig.suptitle(
        "Boundary GLA lesion has no replicated Game-specific effect on W1 activation",
        fontsize=17,
    )
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    def f(cell: dict[str, float]) -> str:
        return f"{cell['mean']:+.3f} [{cell['ci_low']:+.3f}, {cell['ci_high']:+.3f}]"

    lines = [
        "# First-answer-boundary lesion and final W1 semantic activation",
        "",
        "## Bottom line",
        "",
        "The answer is **no, not in a replicated Game-specific way**. On held-out W1≠W2 questions, the lesion increased Game's W1−W2 output margin while leaving Neutral's margin approximately unchanged, reproducing the earlier causal result. But the final W1 semantic projection decreased by similar, statistically uncertain amounts in both conditions. The Game-minus-Neutral projection interaction was small and crossed zero. Discovery also showed no reliable interaction and did not reproduce the confirmation projection direction.",
        "",
        "Therefore the first-answer-boundary GLA writes affect how the later computation ranks W1, but the effect is **not mediated by simply adding or removing the one-dimensional W1 semantic activation measured at the final decision position**. This narrows the missing mechanism to other residual dimensions or to condition-dependent use of an intact candidate representation.",
        "",
        "The lesion is identical to the prior experiment. Natural and lesioned projections are paired within the present same-host run; saved historical logits are retained as a cross-host numerical validation reference.",
        "",
        "A positive projection effect means that removing the boundary write makes the final decision residual more aligned with W1; a negative effect means it removes W1-aligned activation.",
        "",
        "## Final-readout effects",
        "",
        "| Split | Subset | Game | Neutral | Game minus Neutral |",
        "|---|---|---:|---:|---:|",
    ]
    for split in ("discovery", "confirmation", "pooled"):
        for subset in ("conflict", "no_conflict", "conflict_w1_a", "conflict_w1_bcd"):
            cell = summary["splits"][split]["subsets"][subset]
            lines.append(
                f"| {split.title()} | {subset.replace('_', ' ')} (n={cell['n']}) | "
                f"{f(cell['conditions']['Game']['final_projection_effect'])} | "
                f"{f(cell['conditions']['Neutral']['final_projection_effect'])} | "
                f"{f(cell['game_minus_neutral_projection_effect'])} |"
            )
    lines.extend([
        "",
        "## Held-out conflict-trial dissociation",
        "",
    ])
    held = summary["splits"]["confirmation"]["subsets"]["conflict"]
    lines.extend([
        f"- Game W1−W2 output-margin effect: {f(held['conditions']['Game']['final_w1_minus_w2_margin_effect'])} logits.",
        f"- Neutral W1−W2 output-margin effect: {f(held['conditions']['Neutral']['final_w1_minus_w2_margin_effect'])} logits.",
        f"- Game final W1-projection effect: {f(held['conditions']['Game']['final_projection_effect'])} residual units.",
        f"- Neutral final W1-projection effect: {f(held['conditions']['Neutral']['final_projection_effect'])} residual units.",
        f"- Game-minus-Neutral projection interaction: {f(held['game_minus_neutral_projection_effect'])} residual units.",
        f"- Projection-change/output-margin correlation: Game r={held['conditions']['Game']['projection_margin_effect_correlation']:.3f}; Neutral r={held['conditions']['Neutral']['projection_margin_effect_correlation']:.3f}.",
        "",
        "The lesion also increased absolute projection magnitude on conflict trials in both conditions; this reflected mixtures of stronger positive and negative projections rather than selective removal of W1. Those descriptive sign-resolved quantities are stored in `summary.json`.",
        "",
        "## Cross-host numerical validation",
        "",
        "The present run uses the same model revision, prompts, batch-of-four cohorts, SDPA path, and intervention. Exact low-order bfloat16 equality is not expected across retained hosts. The summary records maximum A-D logit deviations and discrete A-D argmax agreement against both historical natural and lesion outputs.",
        "",
        "## Interpretation limits",
        "",
        "The measured projection is the one-dimensional four-mapping W1 candidate direction. A lesion effect on this projection shows that first-boundary GLA writes help construct or regulate that representation. A null effect would instead imply that the earlier behavioral/logit effect operates through other dimensions or through how an intact W1 representation is used. Neither result by itself proves that the boundary stores a portable semantic memory; the prior donor-state transplants directly tested and rejected that stronger claim.",
        "",
        f"Canonical figure: `{figure_path}`.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--mapping-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.discovery,
        args.confirmation,
        args.baseline,
        args.remapped_baseline,
        args.mapping_plan,
        args.output,
        args.figure,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
