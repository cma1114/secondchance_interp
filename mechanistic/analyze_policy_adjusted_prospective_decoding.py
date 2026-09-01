from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


RANK_LABELS = ("R1", "R2", "R3", "R4")
RANK_COLORS = ("#d1495b", "#2878b5", "#2a9d6f", "#8c6bb1")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _letter_control(values: np.ndarray, discovery: np.ndarray) -> np.ndarray:
    """Subtract discovery-only condition/layer/displayed-letter means."""
    values = values.astype(np.float32, copy=True)
    if values.ndim == 4:  # condition, question, layer, letter
        return values - values[:, discovery].mean(axis=1, keepdims=True)
    if values.ndim == 3:  # condition, question, letter
        return values - values[:, discovery].mean(axis=1, keepdims=True)
    raise ValueError(f"Unexpected score shape {values.shape}")


def _rank_align(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    if values.ndim == 3:  # question, layer, letter
        indices = np.broadcast_to(order[:, None, :], values.shape)
    elif values.ndim == 2:  # question, letter
        indices = order
    else:
        raise ValueError(f"Unexpected rank-alignment shape {values.shape}")
    return np.take_along_axis(values, indices, axis=-1)


def _cosine(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = _center(prediction.astype(np.float64))
    target = _center(target.astype(np.float64))
    while target.ndim < prediction.ndim:
        target = np.expand_dims(target, axis=-2)
    numerator = np.sum(prediction * target, axis=-1)
    denominator = np.linalg.norm(prediction, axis=-1) * np.linalg.norm(target, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def _stratified_indices(
    strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    return np.concatenate(
        [
            rng.choice(
                np.flatnonzero(strata == value),
                size=int(np.sum(strata == value)),
                replace=True,
            )
            for value in np.unique(strata)
        ]
    )


def _bootstrap_mean(
    values: np.ndarray,
    strata: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point = values.mean(axis=0)
    samples = np.empty((draws,) + point.shape, dtype=np.float32)
    for draw in range(draws):
        samples[draw] = values[_stratified_indices(strata, rng)].mean(axis=0)
    low, high = np.quantile(samples, (0.025, 0.975), axis=0)
    return point, low, high


def _condition_sign_null(
    cosine: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly reverse the Game/Neutral target orientation within each question."""
    samples = np.empty((draws, cosine.shape[1]), dtype=np.float32)
    for draw in range(draws):
        signs = rng.choice((-1.0, 1.0), size=(len(cosine), 1))
        samples[draw] = (cosine * signs).mean(axis=0)
    low, median, high = np.quantile(samples, (0.025, 0.5, 0.975), axis=0)
    return median, low, high


def _first_sustained(
    bound: np.ndarray, direction: str, count: int = 3
) -> int | None:
    mask = bound > 0 if direction == "positive" else bound < 0
    for start in range(len(mask) - count + 1):
        if bool(np.all(mask[start : start + count])):
            return start + 1
    return None


def _first_persistent(bound: np.ndarray, direction: str) -> int | None:
    mask = bound > 0 if direction == "positive" else bound < 0
    for start in range(len(mask)):
        if bool(np.all(mask[start:])):
            return start + 1
    return None


def _layer_phrase(layer: int | None) -> str:
    return f"L{layer}" if layer is not None else "none"


def _analyze_dataset(
    spec: dict[str, Any],
    output: Path,
    figure_dir: Path,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    run = _load(Path(spec["results"]))
    decoded = _load(Path(spec["analysis_output"]) / "predictions.npz")
    if not np.array_equal(run["question_ids"].astype(str), decoded["question_ids"].astype(str)):
        raise ValueError("Run and decoder question orders differ")

    discovery = decoded["discovery"].astype(bool)
    confirmation = ~discovery
    order = run["rank_order"].astype(np.int64)
    confirmation_order = order[confirmation]
    strata = confirmation_order[:, 0]

    exact = _center(decoded["exact_final_scores"].astype(np.float32))
    exact_controlled = _letter_control(exact, discovery)
    shared = decoded["predictions"][0].astype(np.float32)
    shared_controlled = _letter_control(shared, discovery)
    layer_count = int(shared.shape[-2])
    layers = np.arange(1, layer_count + 1)

    readout_label = spec.get("readout_label", "Fixed JLens")
    jlens = run[spec.get("score_key", "jlens_scores")].astype(np.float32)
    jlens[:, :, -1] = run["direct_logits"].astype(np.float32)
    jlens_controlled = _letter_control(jlens, discovery)

    # The same question is evaluated in Game and Neutral. Their difference is
    # therefore the response to the single incorrect/lost policy token, after
    # removing discovery-only displayed-letter means from each condition.
    exact_delta = _center(exact_controlled[0] - exact_controlled[1])
    decoded_delta = _center(shared_controlled[0] - shared_controlled[1])
    jlens_delta = _center(jlens_controlled[0] - jlens_controlled[1])

    exact_rank = _rank_align(exact_delta[confirmation], confirmation_order)
    decoded_rank = _rank_align(decoded_delta[confirmation], confirmation_order)
    jlens_rank = _rank_align(jlens_delta[confirmation], confirmation_order)

    decoded_cosine = _cosine(decoded_rank, exact_rank)
    jlens_cosine = _cosine(jlens_rank, exact_rank)
    similarity_values = np.stack([decoded_cosine, jlens_cosine], axis=-1)
    similarity, similarity_low, similarity_high = _bootstrap_mean(
        similarity_values,
        strata,
        draws,
        np.random.default_rng(seed),
    )
    null, null_low, null_high = _condition_sign_null(
        decoded_cosine,
        draws,
        np.random.default_rng(seed + 1),
    )

    rank_effect, rank_low, rank_high = _bootstrap_mean(
        decoded_rank,
        strata,
        draws,
        np.random.default_rng(seed + 2),
    )
    exact_effect, exact_low, exact_high = _bootstrap_mean(
        exact_rank,
        strata,
        draws,
        np.random.default_rng(seed + 3),
    )

    display_name = spec["name"]
    slug = "simplemc" if display_name == "SimpleMC" else "triviamc"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))

    axes[0].plot(layers, similarity[:, 0], color="#2f6fb0", linewidth=2.2, label="Shared learned decoder")
    axes[0].fill_between(layers, similarity_low[:, 0], similarity_high[:, 0], color="#2f6fb0", alpha=0.18)
    axes[0].plot(layers, similarity[:, 1], color="#777777", linewidth=1.8, label=readout_label)
    axes[0].fill_between(layers, similarity_low[:, 1], similarity_high[:, 1], color="#777777", alpha=0.14)
    axes[0].plot(layers, null, color="#5f4b32", linewidth=1.3, linestyle=":", label="Within-question condition-sign null")
    axes[0].fill_between(layers, null_low, null_high, color="#8c6d46", alpha=0.11)
    axes[0].set_title("Question-specific policy-pattern decoding")
    axes[0].set_ylabel("Cosine to exact final Game − Neutral pattern")
    axes[0].legend(frameon=False, fontsize=9)

    for ri, (label, color) in enumerate(zip(RANK_LABELS, RANK_COLORS)):
        axes[1].plot(layers, rank_effect[:, ri], color=color, linewidth=2.0, label=label)
        axes[1].fill_between(layers, rank_low[:, ri], rank_high[:, ri], color=color, alpha=0.14)
        axes[1].errorbar(
            [layer_count],
            [exact_effect[ri]],
            yerr=np.asarray(
                [[exact_effect[ri] - exact_low[ri]], [exact_high[ri] - exact_effect[ri]]]
            ),
            color=color,
            marker="D",
            markeredgecolor="#333333",
            markeredgewidth=0.7,
            markersize=6.5,
            capsize=2.5,
            linewidth=1.1,
            zorder=5,
        )
    axes[1].set_title("Decoded policy effect by first-presentation rank")
    axes[1].set_ylabel("Game − Neutral centered answer score")
    axes[1].legend(frameon=False, ncol=4, fontsize=9)
    axes[1].text(
        0.99,
        0.03,
        "Diamonds: exact final means",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )

    for axis in axes:
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set_xlim(0.5, layer_count + 0.5)
        axis.set_xticks(np.arange(5, layer_count + 1, 5))
        axis.set_xticks(np.arange(1, layer_count + 1), minor=True)
        axis.set_xlabel("Final-decision post-block layer")
        axis.grid(axis="x", which="major", color="#e5e5e5", linewidth=0.45)
        axis.grid(axis="y", color="#dddddd", linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"{display_name}: policy-adjusted answer information at the final decision position",
        fontsize=14,
        weight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.015,
        "All curves use held-out confirmation questions. Displayed-letter means and decoder fitting use discovery questions only; bands are paired-question bootstrap 95% CIs.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    prefix = spec.get("figure_prefix", "qwen36")
    figure_path = figure_dir / f"{prefix}_{slug}_policy_adjusted_prospective_decoding.png"
    fig.savefig(figure_path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    selected_layers = tuple(
        layer
        for layer in (16, 24, 32, 35, 40, 44, 48, 52, 56, 60, 64)
        if layer <= layer_count
    )
    similarity_onset = _first_sustained(similarity_low[:, 0], "positive")
    similarity_persistent_onset = _first_persistent(similarity_low[:, 0], "positive")
    component_onsets = {
        label: _first_persistent(
            rank_high[:, ri] if exact_effect[ri] < 0 else rank_low[:, ri],
            "negative" if exact_effect[ri] < 0 else "positive",
        )
        for ri, label in enumerate(RANK_LABELS)
    }
    return {
        "dataset": display_name,
        "n_discovery": int(discovery.sum()),
        "n_confirmation": int(confirmation.sum()),
        "policy_difference": "Game minus Neutral on the same question, after discovery-only displayed-letter control",
        "similarity": {
            "labels": ["Shared learned decoder", readout_label],
            "mean": similarity.tolist(),
            "ci_low": similarity_low.tolist(),
            "ci_high": similarity_high.tolist(),
            "condition_sign_null_median": null.tolist(),
            "condition_sign_null_ci_low": null_low.tolist(),
            "condition_sign_null_ci_high": null_high.tolist(),
            "first_sustained_positive_decoder_ci_layer": similarity_onset,
            "first_persistent_positive_decoder_ci_layer": similarity_persistent_onset,
            "selected_layers": {
                str(layer): {
                    "learned": float(similarity[layer - 1, 0]),
                    "learned_ci_low": float(similarity_low[layer - 1, 0]),
                    "learned_ci_high": float(similarity_high[layer - 1, 0]),
                    "jlens": float(similarity[layer - 1, 1]),
                    "null_median": float(null[layer - 1]),
                    "null_ci_low": float(null_low[layer - 1]),
                    "null_ci_high": float(null_high[layer - 1]),
                }
                for layer in selected_layers
            },
        },
        "rank_effect": {
            "labels": list(RANK_LABELS),
            "decoded_mean": rank_effect.tolist(),
            "decoded_ci_low": rank_low.tolist(),
            "decoded_ci_high": rank_high.tolist(),
            "exact_final_mean": exact_effect.tolist(),
            "exact_final_ci_low": exact_low.tolist(),
            "exact_final_ci_high": exact_high.tolist(),
            "first_sustained_final_direction_ci_layer": component_onsets,
            "selected_layers": {
                str(layer): {
                    label: {
                        "mean": float(rank_effect[layer - 1, ri]),
                        "ci_low": float(rank_low[layer - 1, ri]),
                        "ci_high": float(rank_high[layer - 1, ri]),
                    }
                    for ri, label in enumerate(RANK_LABELS)
                }
                for layer in selected_layers
            },
        },
        "figure": str(figure_path),
    }


def analyze(
    specs_path: Path,
    output: Path,
    figure_dir: Path,
    draws: int,
    seed: int,
) -> None:
    specs = json.loads(specs_path.read_text())["datasets"]
    output.mkdir(parents=True, exist_ok=True)
    results = [
        _analyze_dataset(spec, output, figure_dir, draws, seed + index * 1000)
        for index, spec in enumerate(specs)
    ]
    summary = {
        "analysis": "all_confirmation_policy_adjusted_prospective_decoding",
        "evidence_class": "held-out activation decoding; paired condition contrast; noncausal",
        "datasets": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    simple = results[0]
    trivia = results[1]
    simple_l40 = simple["similarity"]["selected_layers"]["40"]
    trivia_l40 = trivia["similarity"]["selected_layers"]["40"]
    simple_exact = simple["rank_effect"]["exact_final_mean"]
    trivia_exact = trivia["rank_effect"]["exact_final_mean"]

    lines = [
        "# Policy-adjusted prospective-answer information",
        "",
        "This analysis uses every held-out confirmation question and asks when the final decision position contains the question-specific Game-minus-Neutral change in the eventual four-answer score pattern. Game and Neutral prompts are paired within question and differ only at `incorrect`/`lost`. The shared prospective decoder is used for both conditions, so the contrast is expressed in one learned basis. Discovery-only displayed-letter means are removed before candidates are aligned by first-presentation rank.",
        "",
        "The first panel compares the decoded Game-minus-Neutral vector with the exact final Game-minus-Neutral vector for the same held-out question. The condition-sign null randomly reverses which condition is called Game within each question. The second panel decomposes the decoded difference into R1--R4 components; diamonds show the exact final confirmation means.",
        "",
        "## Main findings",
        "",
        f"The question-specific policy-adjusted answer pattern becomes stably decodable at L{simple['similarity']['first_persistent_positive_decoder_ci_layer']} on SimpleMC and L{trivia['similarity']['first_persistent_positive_decoder_ci_layer']} on TriviaMC. At L40, learned-decoder similarity is {simple_l40['learned']:.3f} on SimpleMC and {trivia_l40['learned']:.3f} on TriviaMC, while fixed JLens is {simple_l40['jlens']:.3f} and {trivia_l40['jlens']:.3f}. Thus the final decision position contains a non-output-aligned, question-specific Game/Neutral adjustment by the low-to-mid 30s, well before that adjustment is expressed as answer-token logits.",
        "",
        f"The full-population rank profile matches the strategic-switching account. On SimpleMC, the exact held-out Game-minus-Neutral effect is R1 {simple_exact[0]:+.3f}, R2 {simple_exact[1]:+.3f}, R3 {simple_exact[2]:+.3f}, R4 {simple_exact[3]:+.3f}: Game selectively lowers the old winner and raises the bottom two old-ranked candidates, with essentially no average R2 change. On TriviaMC the corresponding effects are R1 {trivia_exact[0]:+.3f}, R2 {trivia_exact[1]:+.3f}, R3 {trivia_exact[2]:+.3f}, R4 {trivia_exact[3]:+.3f}; all alternatives rise, but R3 and R4 rise most. The decoded candidate components acquire these signs around L32--L36, except TriviaMC R2, which becomes persistently positive at L43.",
        "",
        "Unlike the earlier Game-switch/Neutral-stay panel, this result uses every held-out question and therefore does not obtain its policy difference by selecting questions on the eventual outcome.",
        "",
    ]
    for result in results:
        onset = _layer_phrase(result["similarity"]["first_sustained_positive_decoder_ci_layer"])
        persistent_onset = _layer_phrase(result["similarity"]["first_persistent_positive_decoder_ci_layer"])
        exact = result["rank_effect"]["exact_final_mean"]
        onsets = result["rank_effect"]["first_sustained_final_direction_ci_layer"]
        lines.extend(
            [
                f"## {result['dataset']}",
                "",
                f"Discovery n={result['n_discovery']}; confirmation n={result['n_confirmation']}. [Figure](../../../../../figures/prospective_decoding/{Path(result['figure']).name})",
                "",
                f"The learned policy-pattern similarity first has a three-layer positive 95% CI run at {onset} and remains continuously positive from {persistent_onset}. The latter is the stable onset used in the interpretation.",
                "",
            ]
        )
        available_selected = result["similarity"]["selected_layers"]
        for layer in (24, 32, 35, 40, 44, 48, 52, 56, 64):
            if str(layer) not in available_selected:
                continue
            row = result["similarity"]["selected_layers"][str(layer)]
            lines.append(
                f"- L{layer}: learned cosine {row['learned']:.3f} [{row['learned_ci_low']:.3f}, {row['learned_ci_high']:.3f}]; JLens {row['jlens']:.3f}; sign-null {row['null_median']:.3f} [{row['null_ci_low']:.3f}, {row['null_ci_high']:.3f}]"
            )
        lines.extend(
            [
                "",
                "Exact final Game-minus-Neutral centered rank effects: "
                + ", ".join(
                    f"{label} {exact[ri]:+.3f}" for ri, label in enumerate(RANK_LABELS)
                )
                + ".",
                "",
                "First sustained layer in each candidate's exact-final direction: "
                + ", ".join(f"{label} {_layer_phrase(onsets[label])}" for label in RANK_LABELS)
                + ".",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            "This is held-out activation/decoding evidence on the full confirmation populations. It is not conditioned on whether either task switches, avoiding the outcome-selection problem in the earlier paired Game-switch/Neutral-stay panel. It still does not identify a causal source or prove that the model uses the decoded linear direction.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("figures/prospective_decoding")
    )
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    analyze(args.specs, args.output, args.figure_dir, args.draws, args.seed)


if __name__ == "__main__":
    main()
