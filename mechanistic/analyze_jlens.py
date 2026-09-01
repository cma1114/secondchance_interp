from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .all_trial_figures import CONDITION_COLORS, CONDITION_LABELS, CONDITION_STYLES, _style
from .answer_emergence_figures import Z_975, macro_mean_and_se
from .data import decision_letter, load_activation_dataset


CONDITIONS = ("baseline", "incorrect", "neutral")


def _logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = values.max(axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


def _family_scores(scores: np.ndarray, layout: list[dict], prefix: str) -> np.ndarray:
    indices = [i for i, row in enumerate(layout) if row["family"] == prefix]
    if not indices:
        raise ValueError(f"No selected tokens for {prefix}")
    return _logsumexp(scores[..., indices])


def _concept_family_score(scores: np.ndarray, layout: list[dict], family: str) -> np.ndarray:
    concepts: dict[str, list[int]] = {}
    for index, row in enumerate(layout):
        if row["family"] == family:
            concepts.setdefault(row["text"].strip().lower(), []).append(index)
    if not concepts:
        raise ValueError(f"No selected tokens for strategy family {family}")
    values = [_logsumexp(scores[..., indices]) for indices in concepts.values()]
    return np.stack(values, axis=-1).mean(axis=-1)


def _concept_scores(scores: np.ndarray, layout: list[dict], family: str) -> dict[str, np.ndarray]:
    concepts: dict[str, list[int]] = {}
    for index, row in enumerate(layout):
        if row["family"] == family:
            concepts.setdefault(row["text"].strip().lower(), []).append(index)
    return {concept: _logsumexp(scores[..., indices]) for concept, indices in concepts.items()}


def _answer_scores(scores: np.ndarray, layout: list[dict]) -> np.ndarray:
    return np.stack([_family_scores(scores, layout, f"answer_{letter}") for letter in "ABCD"], axis=-1)


def _labels(data, condition: str) -> np.ndarray:
    labels = []
    for qid in data.question_ids:
        answer = decision_letter(data.metadata[(qid, condition)])
        if answer not in "ABCD":
            raise ValueError(f"Non-A-D answer for {condition}/{qid}: {answer!r}")
        labels.append("ABCD".index(answer))
    return np.asarray(labels, dtype=int)


def _summary(values: np.ndarray, strata: np.ndarray):
    mean, se = macro_mean_and_se(values, strata)
    return mean, Z_975 * se


def _balanced_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean([np.mean(prediction[target == label] == label) for label in range(4)]))


def _macro_auc(target: np.ndarray, score: np.ndarray, strata: np.ndarray) -> tuple[float, dict[str, float]]:
    from sklearn.metrics import roc_auc_score

    by_letter = {
        "ABCD"[letter]: float(roc_auc_score(target[strata == letter], score[strata == letter]))
        for letter in range(4)
    }
    return float(np.mean(list(by_letter.values()))), by_letter


def _bootstrap_macro_auc(
    target: np.ndarray, score: np.ndarray, strata: np.ndarray, seed: int = 42, draws: int = 5000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for draw in range(draws):
        aucs = []
        for letter in range(4):
            letter_ids = np.flatnonzero(strata == letter)
            positive = letter_ids[target[letter_ids]]
            negative = letter_ids[~target[letter_ids]]
            selected = np.concatenate((
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ))
            from sklearn.metrics import roc_auc_score
            aucs.append(roc_auc_score(target[selected], score[selected]))
        values[draw] = np.mean(aucs)
    low, high = np.quantile(values, (0.025, 0.975))
    return float(low), float(high)


def analyze(jlens_root: str | Path, residual_root: str | Path, output_root: str | Path) -> dict:
    jlens_root = Path(jlens_root)
    output = Path(output_root)
    figure_dir = output / "preserved_figures"
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    layout = json.loads((jlens_root / "selected_token_layout.json").read_text())
    with np.load(jlens_root / "jlens_scores.npz", allow_pickle=False) as cached:
        final_selected = cached["final_scores"].astype(np.float64)
        position_selected = cached["position_scores"].astype(np.float64)
        question_ids = cached["question_ids"].astype(str).tolist()
        position_qids = cached["position_question_ids"].astype(str).tolist()
        anchors = cached["anchors"].astype(str).tolist()
        position_availability = cached["position_availability"].astype(bool)
    data = load_activation_dataset(residual_root, list(CONDITIONS))
    if data.question_ids != question_ids:
        raise ValueError("JLens and residual question orders differ")
    layers = np.arange(1, 65)
    answer = _answer_scores(final_selected, layout)
    answer -= answer.mean(axis=-1, keepdims=True)
    prior = _labels(data, "baseline")
    generated = {condition: _labels(data, condition) for condition in CONDITIONS}
    final_logits = data.logits[:, 0, -1].copy()
    final_logits[np.arange(len(prior)), prior] = -np.inf
    runner = final_logits.argmax(axis=-1)
    row = np.arange(len(prior))

    strategy_final = {
        family: _concept_family_score(final_selected, layout, family)
        for family in ("switch", "repeat", "incorrect", "lost")
    }
    strategy_position = {
        family: _concept_family_score(position_selected, layout, family)
        for family in ("switch", "repeat", "incorrect", "lost")
    }
    switch_signal_final = strategy_final["switch"] - strategy_final["repeat"]
    switch_signal_position = strategy_position["switch"] - strategy_position["repeat"]

    metric_rows = []
    summaries = {}
    accuracy_rows = []
    for ci, condition in enumerate(CONDITIONS):
        values = answer[ci]
        competitors = values.copy()
        competitors[row, :, prior] = -np.inf
        prior_margin = values[row, :, prior] - competitors.max(axis=-1)
        competitors = values.copy()
        competitors[row, :, runner] = -np.inf
        runner_margin = values[row, :, runner] - competitors.max(axis=-1)
        spread = values.std(axis=-1)
        strategy = switch_signal_final[ci]
        for metric, array in (
            ("prior_answer_margin", prior_margin),
            ("prior_runner_margin", runner_margin),
            ("ad_spread", spread),
            ("switch_minus_repeat", strategy),
        ):
            mean, half = _summary(array, prior)
            summaries[(condition, metric)] = mean, half
            for layer, value, width in zip(layers, mean, half):
                metric_rows.append({
                    "condition": condition, "metric": metric, "layer": int(layer),
                    "mean": float(value), "ci_low": float(value - width),
                    "ci_high": float(value + width),
                })
        for li, layer in enumerate(layers):
            prediction = values[:, li].argmax(axis=-1)
            accuracy_rows.append({
                "condition": condition,
                "layer": int(layer),
                "accuracy_vs_condition_output": float(np.mean(prediction == generated[condition])),
                "balanced_accuracy_vs_condition_output": _balanced_accuracy(prediction, generated[condition]),
            })

    qid_to_index = {qid: index for index, qid in enumerate(question_ids)}
    position_prior = prior[[qid_to_index[qid] for qid in position_qids]]
    paired_position = switch_signal_position[0] - switch_signal_position[1]
    paired_anchor_indices = np.flatnonzero(position_availability.all(axis=0)).tolist()
    paired_anchors = [anchors[index] for index in paired_anchor_indices]
    position_mean = np.empty((len(paired_anchors), len(layers)))
    position_half = np.empty_like(position_mean)
    position_rows = []
    for output_index, ai in enumerate(paired_anchor_indices):
        anchor = anchors[ai]
        mean, half = _summary(paired_position[:, ai], position_prior)
        position_mean[output_index] = mean
        position_half[output_index] = half
        for layer, value, width in zip(layers, mean, half):
            position_rows.append({
                "anchor": anchor, "metric": "game_minus_neutral_switch_minus_repeat",
                "layer": int(layer), "mean": float(value),
                "ci_low": float(value - width), "ci_high": float(value + width),
            })

    # The decision anchor is exactly the already-cached final prompt position,
    # so use all 500 questions there rather than only the 128-question position
    # sample used for the other semantic anchors.
    decision_values = switch_signal_final[1] - switch_signal_final[2]
    decision_mean, decision_half = _summary(decision_values, prior)

    concept_rows = []
    concept_contrasts: dict[str, np.ndarray] = {}
    for family in ("switch", "repeat", "incorrect", "lost"):
        for concept, values in _concept_scores(final_selected, layout, family).items():
            contrast = values[1] - values[2]
            concept_contrasts[f"{family}/{concept}"] = contrast
            mean, half = _summary(contrast, prior)
            for layer, value, width in zip(layers, mean, half):
                concept_rows.append({
                    "family": family, "concept": concept, "layer": int(layer),
                    "game_minus_neutral_mean": float(value),
                    "ci_low": float(value - width), "ci_high": float(value + width),
                })

    from sklearn.metrics import roc_auc_score

    switch_rows = []
    switch_diagnostics = {}
    for ci, condition in ((1, "incorrect"), (2, "neutral")):
        switched = generated[condition] != prior
        signal = switch_signal_final[ci]
        for li, layer in enumerate(layers):
            auc = float(roc_auc_score(switched, signal[:, li])) if len(np.unique(switched)) == 2 else np.nan
            switch_rows.append({
                "condition": condition,
                "layer": int(layer),
                "n_switch": int(switched.sum()),
                "n_non_switch": int((~switched).sum()),
                "switch_mean": float(signal[switched, li].mean()),
                "non_switch_mean": float(signal[~switched, li].mean()),
                "switch_minus_non_switch": float(signal[switched, li].mean() - signal[~switched, li].mean()),
                "auc_predicting_switch": auc,
            })
        switch_diagnostics[condition] = (switched, signal)

    with (output / "jlens_final_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader(); writer.writerows(metric_rows)
    with (output / "jlens_accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(accuracy_rows[0]))
        writer.writeheader(); writer.writerows(accuracy_rows)
    with (output / "jlens_position_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(position_rows[0]))
        writer.writeheader(); writer.writerows(position_rows)
    with (output / "jlens_strategy_concepts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(concept_rows[0]))
        writer.writeheader(); writer.writerows(concept_rows)
    with (output / "jlens_strategy_switch_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(switch_rows[0]))
        writer.writeheader(); writer.writerows(switch_rows)

    top_tokens = json.loads((jlens_root / "top_tokens.json").read_text())
    selected_layers = (23, 31, 39, 47, 55, 63)
    top_rows = []
    for layer in selected_layers:
        for anchor in anchors:
            key = f"game_minus_neutral/{anchor}/L{layer}"
            if key not in top_tokens["positions"]:
                continue
            for direction in ("top", "bottom"):
                for rank, token in enumerate(top_tokens["positions"][key][direction], 1):
                    top_rows.append({
                        "layer": layer + 1, "anchor": anchor, "direction": direction,
                        "rank": rank, **token,
                    })
    with (output / "jlens_game_neutral_top_tokens.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(top_rows[0]))
        writer.writeheader(); writer.writerows(top_rows)

    _style()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(8.5, 6.1), sharex="col")
    for condition in CONDITIONS:
        for axis, metric in ((axes[0, 0], "prior_answer_margin"), (axes[0, 1], "ad_spread")):
            mean, half = summaries[(condition, metric)]
            color = CONDITION_COLORS[condition]
            axis.fill_between(layers, mean - half, mean + half, color=color, alpha=0.14, linewidth=0)
            axis.plot(
                layers, mean, color=color, linestyle=CONDITION_STYLES[condition],
                linewidth=1.55, label=CONDITION_LABELS[condition],
            )
    axes[0, 0].axhline(0, color="#555555", linewidth=0.7)
    axes[0, 0].set_title("A  Original-answer advantage", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("JLens score minus strongest competitor")
    axes[0, 0].legend(frameon=False, loc="upper left")
    axes[0, 1].set_title("B  Total A-D spread", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("Within-question JLens score SD")

    axes[1, 0].fill_between(
        layers, decision_mean - decision_half, decision_mean + decision_half,
        color="#7B3294", alpha=0.18, linewidth=0,
    )
    axes[1, 0].plot(layers, decision_mean, color="#7B3294", linewidth=1.65)
    axes[1, 0].axhline(0, color="#555555", linewidth=0.7)
    axes[1, 0].set_title("C  Game-specific switch-versus-repeat content", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("Game minus Neutral JLens contrast\nat final decision position")
    axes[1, 0].set_xlabel("Post-block residual readout")

    bound = float(np.nanmax(np.abs(position_mean)))
    image = axes[1, 1].imshow(
        position_mean, aspect="auto", origin="upper", cmap="RdBu_r",
        vmin=-bound, vmax=bound, extent=(0.5, 64.5, len(paired_anchors) - 0.5, -0.5),
    )
    axes[1, 1].set_yticks(np.arange(len(paired_anchors)))
    axes[1, 1].set_yticklabels([anchor.replace("_", " ") for anchor in paired_anchors])
    axes[1, 1].set_title("D  Where that content appears", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Post-block residual readout")
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.03, label="Game − Neutral contrast")
    for axis in axes[0]:
        axis.set_xlim(1, 64)
        axis.set_xticks(np.arange(8, 65, 8))
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlim(1, 64)
    axes[1, 0].set_xticks(np.arange(8, 65, 8))
    axes[1, 0].grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axes[1, 0].spines[["top", "right"]].set_visible(False)
    figure.suptitle("Qwen3.6-27B SimpleMC: Jacobian-lens readouts", fontsize=10.5, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96), w_pad=1.7, h_pad=1.6)
    figure.savefig(figure_dir / "jlens_condition_representations.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    peak_layer_index = int(np.argmax(np.abs(decision_mean)))
    game_switched, game_signal = switch_diagnostics["incorrect"]
    neutral_switched, neutral_signal = switch_diagnostics["neutral"]
    concept_at_peak = {
        key: float(values[:, peak_layer_index].mean())
        for key, values in concept_contrasts.items()
    }
    paired_macro_auc, paired_auc_by_letter = _macro_auc(
        game_switched, decision_values[:, peak_layer_index], prior
    )
    paired_macro_auc_ci = _bootstrap_macro_auc(
        game_switched, decision_values[:, peak_layer_index], prior
    )
    metadata = json.loads((jlens_root / "run_metadata.json").read_text())
    summary = {
        "preflight": metadata["preflight"],
        "n_final_trials": len(question_ids),
        "n_position_trials": len(position_qids),
        "final_layer_balanced_accuracy": {
            condition: _balanced_accuracy(answer[ci, :, -1].argmax(-1), generated[condition])
            for ci, condition in enumerate(CONDITIONS)
        },
        "decision_switch_minus_repeat_game_neutral": {
            "largest_absolute_layer": int(layers[peak_layer_index]),
            "value": float(decision_mean[peak_layer_index]),
            "ci_low": float(decision_mean[peak_layer_index] - decision_half[peak_layer_index]),
            "ci_high": float(decision_mean[peak_layer_index] + decision_half[peak_layer_index]),
        },
        "decision_signal_behavior_at_peak": {
            "game_switch_mean": float(game_signal[game_switched, peak_layer_index].mean()),
            "game_non_switch_mean": float(game_signal[~game_switched, peak_layer_index].mean()),
            "game_auc_predicting_switch": float(roc_auc_score(game_switched, game_signal[:, peak_layer_index])),
            "neutral_switch_mean": float(neutral_signal[neutral_switched, peak_layer_index].mean()),
            "neutral_non_switch_mean": float(neutral_signal[~neutral_switched, peak_layer_index].mean()),
            "neutral_auc_predicting_switch": float(roc_auc_score(neutral_switched, neutral_signal[:, peak_layer_index])),
            "paired_game_neutral_delta_auc_predicting_game_switch": float(
                roc_auc_score(game_switched, decision_values[:, peak_layer_index])
            ),
            "paired_delta_macro_auc_controlling_prior_letter": paired_macro_auc,
            "paired_delta_macro_auc_by_prior_letter": paired_auc_by_letter,
            "paired_delta_macro_auc_ci_95": list(paired_macro_auc_ci),
            "switch_contingency": {
                "game_only": int(np.sum(game_switched & ~neutral_switched)),
                "neutral_only": int(np.sum(~game_switched & neutral_switched)),
                "both": int(np.sum(game_switched & neutral_switched)),
                "neither": int(np.sum(~game_switched & ~neutral_switched)),
            },
        },
        "decision_concept_game_neutral_contrasts_at_peak": dict(
            sorted(concept_at_peak.items(), key=lambda item: -abs(item[1]))
        ),
        "anchor_peak_layers": {
            anchor: {
                "layer": int(layers[int(np.argmax(np.abs(position_mean[ai])))]),
                "value": float(position_mean[ai, int(np.argmax(np.abs(position_mean[ai])))]),
            }
            for ai, anchor in enumerate(paired_anchors)
        },
    }
    (output / "jlens_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Second Chance Jacobian-lens scores")
    parser.add_argument("--jlens-root", required=True)
    parser.add_argument("--residual-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.jlens_root, args.residual_root, args.output), indent=2))


if __name__ == "__main__":
    main()
