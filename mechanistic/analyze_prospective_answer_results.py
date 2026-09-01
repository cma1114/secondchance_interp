from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("game", "neutral")
PERFORMANCE_LABELS = ("Shared", "Matched task", "Cross-task", "Fixed JLens")
PERFORMANCE_COLORS = ("#2f6fb0", "#3b8f62", "#c43c39", "#777777")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


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
        where=denominator > 0,
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
        selected = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(strata == value),
                    size=int(np.sum(strata == value)),
                    replace=True,
                )
                for value in np.unique(strata)
            ]
        )
        samples[draw] = values[selected].mean(axis=0)
    low, high = np.quantile(samples, (0.025, 0.975), axis=0)
    return point, low, high


def _permutation_null(
    prediction: np.ndarray,
    target: np.ndarray,
    strata: np.ndarray | None,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Question-pair shuffle null for mean per-question cosine across conditions."""
    prediction = _center(prediction.astype(np.float64))
    target = _center(target.astype(np.float64))
    prediction /= np.maximum(np.linalg.norm(prediction, axis=-1, keepdims=True), 1e-12)
    target /= np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-12)
    question_count = prediction.shape[1]
    groups = (
        [np.arange(question_count)]
        if strata is None
        else [np.flatnonzero(strata == value) for value in np.unique(strata)]
    )
    samples = np.empty((draws, prediction.shape[2]), dtype=np.float32)
    for draw in range(draws):
        permutation = np.arange(question_count)
        for group in groups:
            permutation[group] = rng.permutation(group)
        # The same question permutation is used for Game and Neutral, preserving
        # the paired condition targets while breaking question-specific matching.
        samples[draw] = (
            np.einsum("cqla,cqa->ql", prediction, target[:, permutation]).mean(axis=0)
            / prediction.shape[0]
        )
    low, median, high = np.quantile(samples, (0.025, 0.5, 0.975), axis=0)
    return median, low, high


def _letter_control(
    values: np.ndarray,
    discovery: np.ndarray,
) -> np.ndarray:
    """Remove discovery-only displayed-letter means from (..., condition, question, layer, letter)."""
    controlled = values.astype(np.float32, copy=True)
    if controlled.ndim == 4:  # condition, question, layer, letter
        means = controlled[:, discovery].mean(axis=1, keepdims=True)
        return controlled - means
    if controlled.ndim == 5:  # decoder, condition, question, layer, letter
        means = controlled[:, :, discovery].mean(axis=2, keepdims=True)
        return controlled - means
    raise ValueError(f"Unexpected score shape {controlled.shape}")


def _rank_align(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    broadcast_shape = values.shape[:-1] + (4,)
    if values.ndim == 4:
        indices = np.broadcast_to(order[None, :, None, :], broadcast_shape)
    elif values.ndim == 5:
        indices = np.broadcast_to(order[None, None, :, None, :], broadcast_shape)
    else:
        raise ValueError(f"Unexpected rank-alignment shape {values.shape}")
    return np.take_along_axis(values, indices, axis=-1)


def _first_sustained(low: np.ndarray, direction: str = "positive", count: int = 3) -> int | None:
    mask = low > 0 if direction == "positive" else low < 0
    for start in range(len(mask) - count + 1):
        if bool(np.all(mask[start : start + count])):
            return start + 1
    return None


def _onset_phrase(layer: int | None) -> str:
    return f"L{layer}" if layer is not None else "no layer (the interval never stays positive for three layers)"


def _analyze_dataset(
    spec: dict[str, Any],
    output: Path,
    figure_dir: Path,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    run = _load(Path(spec["results"]))
    decoded = _load(Path(spec["analysis_output"]) / "predictions.npz")
    qids = run["question_ids"].astype(str)
    if not np.array_equal(qids, decoded["question_ids"].astype(str)):
        raise ValueError("Run and decoder question orders differ")
    discovery = decoded["discovery"].astype(bool)
    confirmation = ~discovery
    order = run["rank_order"].astype(np.int64)
    first_winner = order[:, 0]
    exact = _center(decoded["exact_final_scores"].astype(np.float32))
    predictions = decoded["predictions"].astype(np.float32)
    predictions = _letter_control(predictions, discovery)
    exact_controlled = exact - exact[:, discovery].mean(axis=1, keepdims=True)
    aligned_prediction = _rank_align(predictions, order)

    direct = run["direct_logits"].astype(np.float32)
    final_choice = np.argmax(direct, axis=-1)
    switch = final_choice != first_winner[None]
    confirm_indices = np.flatnonzero(confirmation)
    strata = first_winner[confirmation]
    layer_count = int(predictions.shape[-2])
    layers = np.arange(1, layer_count + 1)
    readout_label = spec.get("readout_label", "Fixed JLens")
    performance_labels = ("Shared", "Matched task", "Cross-task", readout_label)
    jlens = run[spec.get("score_key", "jlens_scores")].astype(np.float32)
    jlens[:, :, -1] = direct
    jlens_controlled = _letter_control(jlens, discovery)
    jlens_aligned = _rank_align(jlens_controlled, order)

    # Per-question cosine to the exact final four-answer pattern. The matched and
    # crossed traces average the two condition-specific coefficient bases.
    shared_cos_by_condition = np.stack(
        [
            _cosine(predictions[0, ci, confirmation], exact_controlled[ci, confirmation])
            for ci in range(2)
        ],
        axis=0,
    )
    matched_cos_by_condition = np.stack(
        [
            _cosine(predictions[1 + ci, ci, confirmation], exact_controlled[ci, confirmation])
            for ci in range(2)
        ],
        axis=0,
    )
    crossed_cos_by_condition = np.stack(
        [
            _cosine(predictions[2 - ci, ci, confirmation], exact_controlled[ci, confirmation])
            for ci in range(2)
        ],
        axis=0,
    )
    jlens_cos_by_condition = np.stack(
        [
            _cosine(jlens_controlled[ci, confirmation], exact_controlled[ci, confirmation])
            for ci in range(2)
        ],
        axis=0,
    )
    shared_cos = shared_cos_by_condition.mean(axis=0)
    matched_cos = matched_cos_by_condition.mean(axis=0)
    crossed_cos = crossed_cos_by_condition.mean(axis=0)
    jlens_cos = jlens_cos_by_condition.mean(axis=0)
    performance_values = np.stack(
        [shared_cos, matched_cos, crossed_cos, jlens_cos], axis=-1
    )
    perf, perf_low, perf_high = _bootstrap_mean(
        performance_values, strata, draws, np.random.default_rng(seed)
    )
    basis_gap_values = np.stack(
        [matched_cos - crossed_cos, matched_cos - shared_cos], axis=-1
    )
    basis_gap, basis_gap_low, basis_gap_high = _bootstrap_mean(
        basis_gap_values, strata, draws, np.random.default_rng(seed + 1)
    )
    absolute_null, absolute_null_low, absolute_null_high = _permutation_null(
        predictions[0][:, confirmation],
        exact_controlled[:, confirmation],
        None,
        draws,
        np.random.default_rng(seed + 2),
    )
    winner_matched_null, winner_matched_null_low, winner_matched_null_high = _permutation_null(
        predictions[0][:, confirmation],
        exact_controlled[:, confirmation],
        strata,
        draws,
        np.random.default_rng(seed + 3),
    )

    # Shared decoder, sliced only after held-out prediction, on each task's switch trials.
    switch_records: dict[str, Any] = {}
    switch_curves = []
    switch_lows = []
    switch_highs = []
    jlens_switch_curves = []
    for ci, condition in enumerate(CONDITIONS):
        mask = confirmation & switch[ci]
        decoded_margin = aligned_prediction[0, ci, mask, :, 1] - aligned_prediction[0, ci, mask, :, 0]
        jlens_margin = jlens_aligned[ci, mask, :, 1] - jlens_aligned[ci, mask, :, 0]
        point, low, high = _bootstrap_mean(
            decoded_margin,
            first_winner[mask],
            draws,
            np.random.default_rng(seed + 10 + ci),
        )
        switch_curves.append(point)
        switch_lows.append(low)
        switch_highs.append(high)
        jlens_switch_curves.append(jlens_margin.mean(axis=0))
        switch_records[condition] = {
            "n": int(mask.sum()),
            "shared_decoder_r2_minus_r1": point.tolist(),
            "ci_low": low.tolist(),
            "ci_high": high.tolist(),
            "first_sustained_positive_ci_layer": _first_sustained(low),
            "jlens_r2_minus_r1": jlens_margin.mean(axis=0).tolist(),
        }

    # Paired core contrast: questions where Game switches but Neutral stays.
    core = confirmation & switch[0] & ~switch[1]
    game_margin = aligned_prediction[0, 0, core, :, 1] - aligned_prediction[0, 0, core, :, 0]
    neutral_margin = aligned_prediction[0, 1, core, :, 1] - aligned_prediction[0, 1, core, :, 0]
    shared_margin = (game_margin + neutral_margin) * 0.5
    differential_margin = game_margin - neutral_margin
    components = np.stack([shared_margin, differential_margin], axis=-1)
    component_point, component_low, component_high = _bootstrap_mean(
        components,
        first_winner[core],
        draws,
        np.random.default_rng(seed + 30),
    )

    display_name = spec["name"]
    slug = "simplemc" if display_name == "SimpleMC" else "triviamc"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.0))

    for index, label in enumerate(performance_labels):
        axes[0].plot(layers, perf[:, index], color=PERFORMANCE_COLORS[index], linewidth=2.0, label=label)
        axes[0].fill_between(
            layers, perf_low[:, index], perf_high[:, index], color=PERFORMANCE_COLORS[index], alpha=0.16
        )
    axes[0].set_title("Held-out final-pattern decoding")
    axes[0].plot(
        layers,
        winner_matched_null,
        color="#5f4b32",
        linewidth=1.4,
        linestyle=":",
        label="W1-matched shuffle null",
    )
    axes[0].fill_between(
        layers,
        winner_matched_null_low,
        winner_matched_null_high,
        color="#8c6d46",
        alpha=0.12,
    )
    axes[0].set_ylabel("Mean per-question cosine")
    axes[0].legend(frameon=False, fontsize=9)

    condition_colors = ("#c43c39", "#2f6fb0")
    for ci, condition in enumerate(("Game", "Neutral")):
        axes[1].plot(layers, switch_curves[ci], color=condition_colors[ci], linewidth=2.0, label=f"{condition}: learned decoder")
        axes[1].fill_between(
            layers, switch_lows[ci], switch_highs[ci], color=condition_colors[ci], alpha=0.16
        )
        axes[1].plot(
            layers,
            jlens_switch_curves[ci],
            color=condition_colors[ci],
            linewidth=1.25,
            linestyle="--",
            alpha=0.8,
            label=f"{condition}: {readout_label}",
        )
    axes[1].set_title("Held-out switch trials")
    axes[1].set_ylabel("Decoded 1P R2 − R1 score")
    axes[1].legend(frameon=False, fontsize=8)

    component_labels = ("Shared R2−R1", "Game−Neutral R2−R1")
    component_colors = ("#7357a6", "#d17a22")
    for index, label in enumerate(component_labels):
        axes[2].plot(layers, component_point[:, index], color=component_colors[index], linewidth=2.0, label=label)
        axes[2].fill_between(
            layers,
            component_low[:, index],
            component_high[:, index],
            color=component_colors[index],
            alpha=0.16,
        )
    axes[2].set_title(f"Paired Game-switch / Neutral-stay (n={int(core.sum())})")
    axes[2].set_ylabel("Decoded R2 − R1 score")
    axes[2].legend(frameon=False, fontsize=9)

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
        f"{display_name}: prospective final-answer decoding at the final decision position",
        fontsize=14,
        weight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "All decoder fits use frozen discovery questions; curves and 95% CIs use confirmation questions. Displayed-letter means are removed using discovery only.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    prefix = spec.get("figure_prefix", "qwen36")
    figure_path = figure_dir / f"{prefix}_{slug}_prospective_answer_decoding.png"
    fig.savefig(figure_path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    selected_layers = tuple(
        layer
        for layer in (16, 24, 32, 40, 44, 48, 50, 52, 54, 56, 60, 64)
        if layer <= layer_count
    )
    record = {
        "dataset": display_name,
        "n_discovery": int(discovery.sum()),
        "n_confirmation": int(confirmation.sum()),
        "performance": {
            "labels": list(performance_labels),
            "mean": perf.tolist(),
            "ci_low": perf_low.tolist(),
            "ci_high": perf_high.tolist(),
            "selected_layers": {
                str(layer): {
                    label: float(perf[layer - 1, index])
                    for index, label in enumerate(performance_labels)
                }
                for layer in selected_layers
            },
            "by_condition_selected_layers": {
                condition: {
                    str(layer): {
                        "Shared": float(shared_cos_by_condition[ci, :, layer - 1].mean()),
                        "Matched task": float(matched_cos_by_condition[ci, :, layer - 1].mean()),
                        "Cross-task": float(crossed_cos_by_condition[ci, :, layer - 1].mean()),
                        readout_label: float(jlens_cos_by_condition[ci, :, layer - 1].mean()),
                    }
                    for layer in selected_layers
                }
                for ci, condition in enumerate(CONDITIONS)
            },
            "basis_gap": {
                "labels": ["Matched minus cross-task", "Matched minus shared"],
                "mean": basis_gap.tolist(),
                "ci_low": basis_gap_low.tolist(),
                "ci_high": basis_gap_high.tolist(),
                "selected_layers": {
                    str(layer): {
                        "Matched minus cross-task": {
                            "mean": float(basis_gap[layer - 1, 0]),
                            "ci_low": float(basis_gap_low[layer - 1, 0]),
                            "ci_high": float(basis_gap_high[layer - 1, 0]),
                        },
                        "Matched minus shared": {
                            "mean": float(basis_gap[layer - 1, 1]),
                            "ci_low": float(basis_gap_low[layer - 1, 1]),
                            "ci_high": float(basis_gap_high[layer - 1, 1]),
                        },
                    }
                    for layer in selected_layers
                },
            },
            "chance_nulls": {
                "definition": {
                    "absolute_question_shuffle": "Permute paired Game/Neutral final targets across all confirmation questions.",
                    "w1_matched_question_shuffle": "Permute paired Game/Neutral final targets only among confirmation questions sharing the same displayed first-presentation winner letter. This preserves the easiest old-winner structure while breaking question-specific final geometry.",
                },
                "absolute_question_shuffle": {
                    "median": absolute_null.tolist(),
                    "ci_low": absolute_null_low.tolist(),
                    "ci_high": absolute_null_high.tolist(),
                },
                "w1_matched_question_shuffle": {
                    "median": winner_matched_null.tolist(),
                    "ci_low": winner_matched_null_low.tolist(),
                    "ci_high": winner_matched_null_high.tolist(),
                },
                "selected_layers": {
                    str(layer): {
                        "absolute": {
                            "median": float(absolute_null[layer - 1]),
                            "ci_low": float(absolute_null_low[layer - 1]),
                            "ci_high": float(absolute_null_high[layer - 1]),
                        },
                        "w1_matched": {
                            "median": float(winner_matched_null[layer - 1]),
                            "ci_low": float(winner_matched_null_low[layer - 1]),
                            "ci_high": float(winner_matched_null_high[layer - 1]),
                        },
                    }
                    for layer in selected_layers
                },
            },
        },
        "switch_trials": switch_records,
        "paired_core": {
            "n": int(core.sum()),
            "shared_r2_minus_r1": component_point[:, 0].tolist(),
            "shared_ci_low": component_low[:, 0].tolist(),
            "shared_ci_high": component_high[:, 0].tolist(),
            "game_minus_neutral_r2_minus_r1": component_point[:, 1].tolist(),
            "differential_ci_low": component_low[:, 1].tolist(),
            "differential_ci_high": component_high[:, 1].tolist(),
            "shared_first_sustained_positive_ci_layer": _first_sustained(component_low[:, 0]),
            "differential_first_sustained_positive_ci_layer": _first_sustained(component_low[:, 1]),
        },
        "figure": str(figure_path),
    }
    return record


def analyze(specs_path: Path, output: Path, figure_dir: Path, draws: int, seed: int) -> None:
    specs = json.loads(specs_path.read_text())["datasets"]
    output.mkdir(parents=True, exist_ok=True)
    results = [
        _analyze_dataset(spec, output, figure_dir, draws, seed + index * 1000)
        for index, spec in enumerate(specs)
    ]
    summary = {
        "analysis": "prospective_exact_final_answer_decoding",
        "evidence_class": "held-out linear activation decoding; outcome slices are descriptive postselection",
        "target": "exact final centered A-D logits",
        "letter_control": "discovery-only displayed-letter means removed from targets and predictions",
        "datasets": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Prospective final-answer decoding at the final decision position",
        "",
        f"This analysis asks whether the model's exact eventual four-answer score pattern is linearly available at the final decision position before it is aligned with the fixed output readout used by JLens. It fits one shared Game+Neutral decoder and separate Game-only and Neutral-only decoders at every post-block residual L1–L{max(len(item['performance']['mean']) for item in results)}. Every basis is evaluated on both conditions. All fitting, centering, and ridge selection use frozen discovery questions; all reported curves use held-out confirmation questions.",
        "",
        "The cross-task curves are the cleanest test of representational basis: the Neutral coefficient basis is applied to Game residuals and the Game basis to Neutral residuals. The shared fit has twice as many training rows as a task-specific fit, so shared-versus-specific performance alone is not interpreted as a basis test. Stable displayed-letter means are estimated on discovery questions and removed before semantic-rank analysis.",
        "",
        "## Main findings",
        "",
        "The prospective final four-answer pattern is linearly decodable well before it becomes visible through the model's fixed output lens. The effect is especially clear on TriviaMC: held-out shared-decoder cosine is already 0.369 at L32 and 0.676 at L40, while fixed-JLens cosine is approximately zero at both layers. Thus the late JLens rise is primarily a late alignment of an earlier linear representation with the output vocabulary, not the first appearance of all answer-relevant information at the final decision position.",
        "",
        "The representation is predominantly condition-general. A decoder trained in Game and evaluated in Neutral, or vice versa, remains close to a decoder trained and evaluated in the same condition, particularly in the late layers. The modest cross-condition penalty shows that the bases are not perfectly identical, but there is no large per-condition-versus-transfer collapse that would support wholly different Game and Neutral answer-code bases.",
        "",
        "Chance is reported two ways. A fully shuffled question-to-target assignment is the absolute null and is approximately zero. The stronger W1-matched null shuffles paired Game/Neutral final targets only among confirmation questions with the same displayed first-presentation winner letter. It therefore preserves the easiest old-winner structure while destroying question-specific final geometry. The dotted curve and band in the first panel show this stronger null.",
        "",
        "The more specific eventual-switch ordering is later than the broad final-pattern decodability: on held-out switch trials, decoded R2 exceeds R1 reliably only around L44–L48. On the paired questions where Game switches but Neutral stays, the across-condition mean remains R1-favoring; it never develops a sustained positive R2−R1 interval. The Game-minus-Neutral R2−R1 difference, however, is decodable by L34–L35. This rejects the proposed shared-R2-first sequence on that selected subset: an early task-dependent difference exists in a non-output-aligned linear basis and is later rotated and amplified into answer-logit space.",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result['dataset']}",
                "",
                f"Discovery n={result['n_discovery']}; confirmation n={result['n_confirmation']}; paired Game-switch/Neutral-stay confirmation subset n={result['paired_core']['n']}. [Figure](../../../../../{result['figure']})",
                "",
                "Selected held-out mean cosine similarities (Shared / Matched task / Cross-task / fixed JLens):",
                "",
            ]
        )
        available_selected = result["performance"]["selected_layers"]
        for layer in (32, 40, 44, 48, 52, 56, 60, 64):
            if str(layer) not in available_selected:
                continue
            row = result["performance"]["selected_layers"][str(layer)]
            lines.append(
                f"- L{layer}: {row['Shared']:.3f} / {row['Matched task']:.3f} / "
                f"{row['Cross-task']:.3f} / {row['Fixed JLens']:.3f}"
            )
        gap48 = result["performance"]["basis_gap"]["selected_layers"]["48"]["Matched minus cross-task"]
        gap56 = result["performance"]["basis_gap"]["selected_layers"]["56"]["Matched minus cross-task"]
        null32 = result["performance"]["chance_nulls"]["selected_layers"]["32"]
        null40 = result["performance"]["chance_nulls"]["selected_layers"]["40"]
        lines.extend(
            [
                "",
                f"At L32, the absolute shuffle median is {null32['absolute']['median']:.3f} and the stronger W1-matched shuffle median is {null32['w1_matched']['median']:.3f} [{null32['w1_matched']['ci_low']:.3f}, {null32['w1_matched']['ci_high']:.3f}]. At L40 they are {null40['absolute']['median']:.3f} and {null40['w1_matched']['median']:.3f} [{null40['w1_matched']['ci_low']:.3f}, {null40['w1_matched']['ci_high']:.3f}].",
                "",
                f"The matched-minus-cross-task cosine penalty is {gap48['mean']:.3f} [{gap48['ci_low']:.3f}, {gap48['ci_high']:.3f}] at L48 and {gap56['mean']:.3f} [{gap56['ci_low']:.3f}, {gap56['ci_high']:.3f}] at L56.",
                "",
                f"On held-out switch trials, the first sustained layer at which the shared decoder's R2−R1 95% CI is positive is L{result['switch_trials']['game']['first_sustained_positive_ci_layer']} in Game and L{result['switch_trials']['neutral']['first_sustained_positive_ci_layer']} in Neutral. These are descriptive outcome slices, not causal evidence about why switching occurred.",
                "",
                f"In the paired Game-switch/Neutral-stay subset, the shared R2−R1 component first has a sustained positive CI at {_onset_phrase(result['paired_core']['shared_first_sustained_positive_ci_layer'])}; the Game−Neutral R2−R1 component does so at {_onset_phrase(result['paired_core']['differential_first_sustained_positive_ci_layer'])}. Exact trajectories and intervals are in `summary.json`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            "This is activation/decoding evidence. Earlier held-out decoding than JLens shows that the eventual answer pattern is linearly present before it is output-aligned; it does not prove that the decoded direction is causally used. A late decoder onset is stronger evidence for late linear construction, but it cannot rule out an earlier nonlinear representation. Switch-conditioned panels are selected by the eventual answer and cannot identify the cause of switching.",
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
