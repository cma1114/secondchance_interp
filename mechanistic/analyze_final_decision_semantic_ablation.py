from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


LETTERS = "ABCD"
COLORS = {"game": "#348ce8", "neutral": "#ed7d31"}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _entropy(values: np.ndarray) -> np.ndarray:
    probabilities = _softmax(values)
    return -(probabilities * np.log2(np.clip(probabilities, 1e-12, None))).sum(-1)


def _align(values: np.ndarray, qids: list[str], plan: dict[str, dict]) -> np.ndarray:
    output = np.empty_like(values)
    for qi, qid in enumerate(qids):
        mapping = plan[qid]["original_to_new"]
        for content_index, content in enumerate(LETTERS):
            output[..., qi, content_index] = values[
                ..., qi, LETTERS.index(mapping[content])
            ]
    return output


def _semantic_answers(
    raw_values: np.ndarray, qids: list[str], plan: dict[str, dict]
) -> np.ndarray:
    """Resolve displayed-letter ties before mapping answers to semantic content."""
    displayed = raw_values.argmax(axis=-1)
    output = np.empty_like(displayed)
    for qi, qid in enumerate(qids):
        mapping = plan[qid]["new_to_original"]
        for new_index, new_letter in enumerate(LETTERS):
            output[..., qi] = np.where(
                displayed[..., qi] == new_index,
                LETTERS.index(mapping[new_letter]),
                output[..., qi],
            )
    return output


def _stratified_indices(
    strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    return np.concatenate(
        [group[rng.integers(0, len(group), size=len(group))] for group in groups]
    )


def _interval(
    values: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    values = values[valid]
    strata = np.asarray(strata)[valid]
    boot = np.empty(draws, dtype=float)
    for draw in range(draws):
        boot[draw] = values[_stratified_indices(strata, rng)].mean()
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(boot, [0.025, 0.975]).tolist(),
    }


def _layer_intervals(
    values: np.ndarray,
    strata: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, values.shape[1]), dtype=np.float32)
    for draw in range(draws):
        boot[draw] = values[_stratified_indices(strata, rng)].mean(axis=0)
    return {
        "mean": values.mean(axis=0).tolist(),
        "ci_low": np.quantile(boot, 0.025, axis=0).tolist(),
        "ci_high": np.quantile(boot, 0.975, axis=0).tolist(),
    }


def _load_split(
    root: Path,
    baseline: dict[str, dict],
    remapped: dict[str, dict],
    manifest: dict[str, dict],
    plan: dict[str, dict],
    historical_game: dict[str, dict],
    historical_neutral: dict[str, dict],
    draws: int,
    layer_draws: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = dict(np.load(root / "results.npz", allow_pickle=False))
    if not np.all(arrays["completed"]):
        raise ValueError(f"Incomplete results: {root}")
    qids = arrays["question_ids"].astype(str).tolist()
    natural = _align(arrays["natural_logits"], qids, plan)
    ablated = _align(arrays["ablated_logits"], qids, plan)
    w1 = np.asarray([LETTERS.index(baseline[qid]["answer"]) for qid in qids])
    w2_values = [remapped[qid].get("answer_original_content") for qid in qids]
    valid_w2 = np.asarray([value in LETTERS for value in w2_values])
    w2 = np.asarray([
        LETTERS.index(value) if value in LETTERS else -1 for value in w2_values
    ])
    correct = np.asarray([
        LETTERS.index(manifest[qid]["correct_answer"]) for qid in qids
    ])
    selected = valid_w2 & (w1 != w2)
    index = np.flatnonzero(selected)
    strata = w1[selected]
    row = np.arange(len(qids))

    natural_answers = _semantic_answers(arrays["natural_logits"], qids, plan)
    ablated_answers = _semantic_answers(arrays["ablated_logits"], qids, plan)
    natural_behavior = (
        (natural_answers[1] == w1).astype(float)
        - (natural_answers[0] == w1).astype(float)
    )
    ablated_behavior = (
        (ablated_answers[1] == w1).astype(float)
        - (ablated_answers[0] == w1).astype(float)
    )
    natural_margin = natural[:, row, w1] - natural[:, row, w2.clip(min=0)]
    ablated_margin = ablated[:, row, w1] - ablated[:, row, w2.clip(min=0)]
    natural_targeting = natural_margin[1] - natural_margin[0]
    ablated_targeting = ablated_margin[1] - ablated_margin[0]
    game_choice_change = (
        (ablated_answers[0] == w1).astype(float)
        - (natural_answers[0] == w1).astype(float)
    )
    neutral_choice_change = (
        (ablated_answers[1] == w1).astype(float)
        - (natural_answers[1] == w1).astype(float)
    )
    game_margin_change = ablated_margin[0] - natural_margin[0]
    neutral_margin_change = ablated_margin[1] - natural_margin[1]

    rng = np.random.default_rng(seed)
    endpoints = {
        "natural_w1_selection_gap": _interval(
            natural_behavior[selected], strata, rng, draws
        ),
        "ablated_w1_selection_gap": _interval(
            ablated_behavior[selected], strata, rng, draws
        ),
        "reduction_in_w1_selection_gap": _interval(
            (natural_behavior - ablated_behavior)[selected], strata, rng, draws
        ),
        "natural_targeting_contrast_logits": _interval(
            natural_targeting[selected], strata, rng, draws
        ),
        "ablated_targeting_contrast_logits": _interval(
            ablated_targeting[selected], strata, rng, draws
        ),
        "reduction_in_targeting_contrast_logits": _interval(
            (natural_targeting - ablated_targeting)[selected], strata, rng, draws
        ),
        "game_w1_selection_change": _interval(
            game_choice_change[selected], strata, rng, draws
        ),
        "neutral_w1_selection_change": _interval(
            neutral_choice_change[selected], strata, rng, draws
        ),
        "game_w1_vs_w2_margin_change": _interval(
            game_margin_change[selected], strata, rng, draws
        ),
        "neutral_w1_vs_w2_margin_change": _interval(
            neutral_margin_change[selected], strata, rng, draws
        ),
    }
    natural_gap = endpoints["natural_w1_selection_gap"]["mean"]
    natural_contrast = endpoints["natural_targeting_contrast_logits"]["mean"]
    endpoints["behavioral_fraction_explained"] = (
        endpoints["reduction_in_w1_selection_gap"]["mean"] / natural_gap
        if abs(natural_gap) > 1e-12 else None
    )
    endpoints["logit_fraction_explained"] = (
        endpoints["reduction_in_targeting_contrast_logits"]["mean"]
        / natural_contrast if abs(natural_contrast) > 1e-12 else None
    )

    activation = {}
    for condition_index, condition in enumerate(("game", "neutral")):
        natural_projection = arrays["natural_projection"][condition_index, selected]
        natural_norm = arrays["natural_residual_norm"][condition_index, selected]
        ablated_projection = arrays["ablated_pre_projection"][condition_index, selected]
        ablated_norm = arrays["ablated_residual_norm"][condition_index, selected]
        after = arrays["ablated_projection_after"][condition_index, selected]
        activation[condition] = {
            "natural_signed_projection": _layer_intervals(
                natural_projection, strata, seed + 101 + condition_index, layer_draws
            ),
            "natural_absolute_projection": _layer_intervals(
                np.abs(natural_projection), strata,
                seed + 111 + condition_index, layer_draws,
            ),
            "natural_energy_fraction": _layer_intervals(
                (natural_projection / np.clip(natural_norm, 1e-12, None)) ** 2,
                strata, seed + 121 + condition_index, layer_draws,
            ),
            "ablated_pre_signed_projection": _layer_intervals(
                ablated_projection, strata,
                seed + 131 + condition_index, layer_draws,
            ),
            "ablated_pre_absolute_projection": _layer_intervals(
                np.abs(ablated_projection), strata,
                seed + 141 + condition_index, layer_draws,
            ),
            "ablated_pre_energy_fraction": _layer_intervals(
                (ablated_projection / np.clip(ablated_norm, 1e-12, None)) ** 2,
                strata, seed + 151 + condition_index, layer_draws,
            ),
            "max_abs_projection_after": float(np.abs(after).max()),
        }

    condition_summary = {}
    for scenario_name, values, answers in (
        ("natural", natural, natural_answers),
        ("ablated", ablated, ablated_answers),
    ):
        condition_summary[scenario_name] = {}
        for condition_index, condition in enumerate(("game", "neutral")):
            condition_summary[scenario_name][condition] = {
                "w1_selection_discordant": float(
                    np.mean(answers[condition_index, selected] == w1[selected])
                ),
                "w2_selection_discordant": float(
                    np.mean(answers[condition_index, selected] == w2[selected])
                ),
                "accuracy_all": float(np.mean(answers[condition_index] == correct)),
                "entropy_bits_all": float(_entropy(values[condition_index]).mean()),
            }

    validation = {}
    for condition_index, (condition, historical) in enumerate(
        (("game", historical_game), ("neutral", historical_neutral))
    ):
        expected = np.asarray([
            historical[qid]["aggregated_ad_logits"] for qid in qids
        ], dtype=np.float32)
        difference = np.abs(arrays["natural_logits"][condition_index] - expected)
        validation[condition] = {
            "max_absolute_logit_difference": float(difference.max()),
            "exact_logit_rows": int(np.all(difference == 0, axis=1).sum()),
            "n": len(qids),
        }

    summary = {
        "root": str(root),
        "n_questions": len(qids),
        "n_discordant_w1_w2": int(selected.sum()),
        "endpoints": endpoints,
        "conditions": condition_summary,
        "activation": activation,
        "historical_validation": validation,
    }
    raw = {
        "qids": np.asarray(qids),
        "w1": w1,
        "w2": w2,
        "selected": selected,
        **arrays,
    }
    return summary, raw


def _combine_raw(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = [key for key in first if key not in ("completed",)]
    output = {}
    for key in keys:
        axis = 1 if first[key].ndim >= 2 and first[key].shape[0] == 2 else 0
        output[key] = np.concatenate([first[key], second[key]], axis=axis)
    output["completed"] = np.ones(len(output["qids"]), dtype=bool)
    return output


def _fmt(value: dict[str, Any], scale: float = 1.0) -> str:
    return (
        f"{value['mean'] * scale:+.3f} "
        f"[{value['ci'][0] * scale:+.3f}, {value['ci'][1] * scale:+.3f}]"
    )


def _plot(summary: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    layers = np.arange(1, 65)
    for column, split in enumerate(("discovery", "confirmation")):
        for row, (key, ylabel, prefix) in enumerate((
            ("natural_signed_projection", "Natural W1 projection", "Natural execution"),
            ("ablated_pre_signed_projection", "W1 projection removed", "Continuously ablated execution"),
        )):
            ax = axes[row, column]
            for condition in ("game", "neutral"):
                values = summary[split]["activation"][condition][key]
                mean = np.asarray(values["mean"])
                low = np.asarray(values["ci_low"])
                high = np.asarray(values["ci_high"])
                ax.plot(layers, mean, color=COLORS[condition], label=condition.title())
                ax.fill_between(layers, low, high, color=COLORS[condition], alpha=0.14)
            ax.axhline(0, color="#666", linewidth=0.8)
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"{'ABCD'[row * 2 + column]}  {prefix}: {split}"
            )
            ax.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
            ax.legend(frameon=False)
    axes[1, 0].set_xlabel("Post-block residual readout")
    axes[1, 1].set_xlabel("Post-block residual readout")
    fig.suptitle(
        "W1 semantic activation at the second-answer decision position",
        fontsize=17,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _report(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# Continuous W1-semantic ablation at the final decision position",
        "",
        "W1 is the semantic answer selected in the original Baseline presentation. W2 is the semantic answer selected by a fresh Baseline solution of the remapped second presentation. Primary analyses use questions where W1 and W2 differ.",
        "",
        "Discrete answers resolve exact ties in displayed A-D order before mapping the winning letter back to semantic content. Continuous quantities are invariant to this tie rule.",
        "",
        "At every post-block readout 1–64, the experiment measured the final-position projection onto the question- and layer-specific four-mapping W1 semantic vector. In the intervention pass it immediately subtracted that projection. Natural and intervention passes used the same exact historical four-question cohorts.",
        "",
        "## Primary causal results",
        "",
        "| Split | Natural Neutral–Game W1 gap | Ablated gap | Gap removed | Natural targeting contrast | Ablated contrast | Contrast removed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("discovery", "confirmation", "pooled"):
        row = summary[split]["endpoints"]
        lines.append(
            f"| {split.title()} | {_fmt(row['natural_w1_selection_gap'], 100)} pp | "
            f"{_fmt(row['ablated_w1_selection_gap'], 100)} pp | "
            f"{_fmt(row['reduction_in_w1_selection_gap'], 100)} pp | "
            f"{_fmt(row['natural_targeting_contrast_logits'])} | "
            f"{_fmt(row['ablated_targeting_contrast_logits'])} | "
            f"{_fmt(row['reduction_in_targeting_contrast_logits'])} |"
        )
    lines.extend([
        "",
        "Positive `gap removed` means the ablation eliminates part of the behavioral evidence for semantic recollection and suppression. Positive `contrast removed` means it eliminates part of the corresponding W1-versus-W2 logit effect.",
        "",
        "## Direct effects on the discordant questions",
        "",
        "| Split | Game W1 selection | Neutral W1 selection | Game W1–W2 margin | Neutral W1–W2 margin | Behavioral fraction explained | Logit fraction explained |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for split in ("discovery", "confirmation", "pooled"):
        row = summary[split]["endpoints"]
        lines.append(
            f"| {split.title()} | {_fmt(row['game_w1_selection_change'], 100)} pp | "
            f"{_fmt(row['neutral_w1_selection_change'], 100)} pp | "
            f"{_fmt(row['game_w1_vs_w2_margin_change'])} | "
            f"{_fmt(row['neutral_w1_vs_w2_margin_change'])} | "
            f"{row['behavioral_fraction_explained']:.1%} | "
            f"{row['logit_fraction_explained']:.1%} |"
        )

    confirmation = summary["confirmation"]
    lines.extend([
        "",
        "## How much activation was removed?",
        "",
    ])
    for condition in ("game", "neutral"):
        activation = confirmation["activation"][condition]
        natural_abs = np.asarray(activation["natural_absolute_projection"]["mean"])
        natural_energy = np.asarray(activation["natural_energy_fraction"]["mean"])
        removed_abs = np.asarray(activation["ablated_pre_absolute_projection"]["mean"])
        peak = int(np.argmax(natural_abs))
        removed_peak = int(np.argmax(removed_abs))
        lines.append(
            f"- **{condition.title()}:** natural mean absolute W1 projection peaks at readout {peak + 1} with {natural_abs[peak]:.4f} residual units, equal to {natural_energy[peak] * 100:.6f}% of squared residual magnitude. During continuous ablation, regenerated activation removed at each layer peaks at readout {removed_peak + 1} with {removed_abs[removed_peak]:.4f} units. Maximum residual projection after subtraction was {activation['max_abs_projection_after']:.6f}."
        )

    lines.extend([
        "",
        "## Standard outcome checks",
        "",
        "| Scenario | Condition | W1 selection | W2 selection | Accuracy | A–D entropy |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for scenario in ("natural", "ablated"):
        for condition in ("game", "neutral"):
            row = confirmation["conditions"][scenario][condition]
            lines.append(
                f"| {scenario.title()} | {condition.title()} | "
                f"{row['w1_selection_discordant']:.1%} | {row['w2_selection_discordant']:.1%} | "
                f"{row['accuracy_all']:.1%} | {row['entropy_bits_all']:.3f} bits |"
            )

    lines.extend([
        "",
        "## Historical-run validation",
        "",
    ])
    for split in ("discovery", "confirmation"):
        values = summary[split]["historical_validation"]
        lines.append(
            f"- {split.title()}: natural logits exactly matched the saved historical logits on {values['game']['exact_logit_rows']}/{values['game']['n']} Game and {values['neutral']['exact_logit_rows']}/{values['neutral']['n']} Neutral questions; maximum absolute difference was {max(values['game']['max_absolute_logit_difference'], values['neutral']['max_absolute_logit_difference']):.6g}."
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def analyze(
    discovery_root: Path,
    confirmation_root: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
    manifest_path: Path,
    mapping_plan_path: Path,
    historical_root: Path,
    output: Path,
    draws: int,
    layer_draws: int,
    seed: int,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped = json.loads(remapped_baseline_path.read_text())["results"]
    manifest = {
        row["id"]: row for row in json.loads(manifest_path.read_text())["questions"]
    }
    plan = {
        row["question_id"]: row
        for row in json.loads(mapping_plan_path.read_text())["rows"]
    }
    historical_game = json.loads(
        (historical_root / "incorrect_results.json").read_text()
    )["results"]
    historical_neutral = json.loads(
        (historical_root / "neutral_results.json").read_text()
    )["results"]
    discovery, discovery_raw = _load_split(
        discovery_root, baseline, remapped, manifest, plan,
        historical_game, historical_neutral, draws, layer_draws, seed,
    )
    confirmation, confirmation_raw = _load_split(
        confirmation_root, baseline, remapped, manifest, plan,
        historical_game, historical_neutral, draws, layer_draws, seed + 1000,
    )

    pooled_raw = _combine_raw(discovery_raw, confirmation_raw)
    pooled_root = output / ".pooled"
    temporary = pooled_root / "results.npz"
    output.mkdir(parents=True, exist_ok=True)
    pooled_root.mkdir(parents=True, exist_ok=True)
    atomic_values = {
        key: value for key, value in pooled_raw.items()
        if key in {
            "question_ids", "completed", "natural_logits", "ablated_logits",
            "natural_projection", "natural_residual_norm",
            "ablated_pre_projection", "ablated_residual_norm",
            "ablated_projection_after",
        }
    }
    # Reuse the split analyzer without persisting a second large result tree.
    np.savez_compressed(temporary, **atomic_values)
    try:
        pooled, _ = _load_split(
            pooled_root, baseline, remapped, manifest, plan,
            historical_game, historical_neutral, draws, layer_draws, seed + 2000,
        )
    finally:
        if temporary.exists():
            temporary.unlink()
        if pooled_root.exists():
            pooled_root.rmdir()

    summary = {
        "definitions": {
            "W1": "semantic answer chosen in the original Baseline presentation",
            "W2": "semantic answer chosen by fresh Baseline solution of the remapped presentation",
            "behavioral_gap": "Neutral W1-selection rate minus Game W1-selection rate on W1!=W2 questions",
            "targeting_contrast": "(Neutral W1-W2 logit margin) minus (Game W1-W2 logit margin)",
        },
        "discovery": discovery,
        "confirmation": confirmation,
        "pooled": pooled,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(summary, output / "final_decision_semantic_ablation.png")
    _report(summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping-plan", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--layer-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    analyze(
        args.discovery, args.confirmation, args.baseline,
        args.remapped_baseline, args.manifest, args.mapping_plan,
        args.historical_root, args.output, args.draws, args.layer_draws, args.seed,
    )


if __name__ == "__main__":
    main()
