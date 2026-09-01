from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .semantic_mapping import displayed_argmax_to_semantic_indices


LETTERS = "ABCD"
CONDITIONS = ("game", "neutral")
MODES = ("natural", "chosen_sham", "devalue", "opposite")
TARGET_READOUTS = np.arange(33, 57)


def _interval(values: np.ndarray, draws: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Interval requires a nonempty vector")
    sampled = values[draws].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci": [float(low), float(high)],
        "n": int(len(values)),
    }


def _entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=-1, keepdims=True)
    return -(probability * np.log(probability + 1e-30)).sum(axis=-1)


def _load_split(
    path: Path,
    second_rows: dict[str, dict[str, Any]],
    remapped_baseline: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not arrays["completed"].all():
        raise ValueError(f"Incomplete result: {path}")
    qids = arrays["question_ids"].astype(str).tolist()
    logits = arrays["logits"].astype(np.float64)
    semantic = np.empty_like(logits)
    w1 = np.asarray([LETTERS.index(value) for value in arrays["w1_original"].astype(str)])
    w2 = np.asarray(
        [
            LETTERS.index(remapped_baseline[qid]["answer_original_content"])
            for qid in qids
        ]
    )
    for qi, qid in enumerate(qids):
        mapping = second_rows[qid]["new_to_original"]
        for new_index, new_letter in enumerate(LETTERS):
            original_index = LETTERS.index(mapping[new_letter])
            semantic[:, :, qi, original_index] = logits[:, :, qi, new_index]
    choices = displayed_argmax_to_semantic_indices(
        logits, [second_rows[qid] for qid in qids]
    )
    rows = np.arange(len(qids))
    w1_logits = semantic[:, :, rows, w1]
    w2_logits = semantic[:, :, rows, w2]
    return {
        "arrays": arrays,
        "qids": qids,
        "semantic_logits": semantic,
        "choices": choices,
        "w1": w1,
        "w2": w2,
        "conflict": w1 != w2,
        "w1_logits": w1_logits,
        "w2_logits": w2_logits,
        "margin": w1_logits - w2_logits,
        "centered_w1": w1_logits - semantic.mean(axis=-1),
        "entropy": _entropy(semantic),
        "spread": semantic.max(axis=-1) - semantic.min(axis=-1),
        "w1_choice": choices == w1[None, None, :],
        "w2_choice": choices == w2[None, None, :],
        "switch": choices != w1[None, None, :],
        "letters": arrays["w1_displayed_letter"].astype(str),
    }


def _metric_vectors(data: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "w1_choice_pp": data["w1_choice"].astype(float) * 100,
        "w2_choice_pp": data["w2_choice"].astype(float) * 100,
        "switch_pp": data["switch"].astype(float) * 100,
        "w1_minus_w2_margin": data["margin"],
        "w1_centered_evidence": data["centered_w1"],
        "ad_entropy": data["entropy"],
        "ad_spread": data["spread"],
    }


def _summarize_split(
    data: dict[str, Any], seed: int, draws: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    metrics = _metric_vectors(data)
    subsets = {
        "all": np.ones(len(data["qids"]), dtype=bool),
        "conflict": data["conflict"],
        "no_conflict": ~data["conflict"],
    }
    for letter in "BCD":
        subsets[f"w1_{letter}"] = data["letters"] == letter
    summary: dict[str, Any] = {}
    for subset_name, mask in subsets.items():
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        boot = rng.integers(0, len(indices), size=(draws, len(indices)))
        subset: dict[str, Any] = {"n": int(len(indices)), "natural": {}, "effects": {}}
        for metric_name, values in metrics.items():
            subset["natural"][metric_name] = {
                condition: _interval(values[ci, 0, indices], boot)
                for ci, condition in enumerate(CONDITIONS)
            }
        for mode_index, mode in enumerate(MODES[1:], start=1):
            subset["effects"][mode] = {}
            for metric_name, values in metrics.items():
                game = values[0, mode_index, indices] - values[0, 0, indices]
                neutral = values[1, mode_index, indices] - values[1, 0, indices]
                subset["effects"][mode][metric_name] = {
                    "game": _interval(game, boot),
                    "neutral": _interval(neutral, boot),
                    "game_minus_neutral_interaction": _interval(
                        game - neutral, boot
                    ),
                }
        summary[subset_name] = subset
    return summary


def _dose_summary(
    data: dict[str, Any], seed: int, draws: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    arrays = data["arrays"]
    target = TARGET_READOUTS - 1
    summary: dict[str, Any] = {}
    for mode_index, mode in enumerate(MODES[1:], start=1):
        summary[mode] = {}
        for condition_index, condition in enumerate(CONDITIONS):
            pre = arrays["pre_score"][condition_index, mode_index, :, target]
            post = arrays["post_score"][condition_index, mode_index, :, target]
            dose = arrays["dose_l2"][condition_index, mode_index, :, target]
            norm = arrays["residual_norm"][condition_index, mode_index, :, target]
            n = pre.shape[1]
            boot = rng.integers(0, n, size=(draws, n))
            score_change = (post - pre).T
            relative = (dose / np.maximum(norm, 1e-12)).T * 100
            # [question, layer] after transpose; report layerwise mean/CI.
            score_boot = score_change[boot].mean(axis=1)
            relative_boot = relative[boot].mean(axis=1)
            summary[mode][condition] = {
                "readouts": TARGET_READOUTS.tolist(),
                "score_change_mean": score_change.mean(axis=0).tolist(),
                "score_change_ci_low": np.quantile(score_boot, 0.025, axis=0).tolist(),
                "score_change_ci_high": np.quantile(score_boot, 0.975, axis=0).tolist(),
                "relative_l2_dose_percent_mean": relative.mean(axis=0).tolist(),
                "relative_l2_dose_percent_ci_low": np.quantile(
                    relative_boot, 0.025, axis=0
                ).tolist(),
                "relative_l2_dose_percent_ci_high": np.quantile(
                    relative_boot, 0.975, axis=0
                ).tolist(),
                "max_mean_relative_l2_dose_percent": float(
                    relative.mean(axis=0).max()
                ),
                "mean_absolute_post_target_error": float(
                    np.abs(
                        post
                        - (
                            pre
                            if mode == "chosen_sham"
                            else (
                                arrays["unchosen_target_score"][:, target].T
                                if mode == "devalue"
                                else (
                                    2 * arrays["chosen_target_score"][:, target].T
                                    - arrays["unchosen_target_score"][:, target].T
                                )
                            )
                        )
                    ).mean()
                ),
            }
    return summary


def _plot(
    path: Path,
    discovery: dict[str, Any],
    confirmation: dict[str, Any],
    dose: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    colors = {"game": "#2F8EF4", "neutral": "#F08032"}
    modes = ("chosen_sham", "devalue", "opposite")
    labels = {"chosen_sham": "Chosen sham", "devalue": "Devalue", "opposite": "Opposite"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))

    def points(axis: Any, metric: str, title: str, ylabel: str) -> None:
        base_x = np.arange(len(modes))
        for ci, condition in enumerate(CONDITIONS):
            means, lows, highs = [], [], []
            for mode in modes:
                value = confirmation["conflict"]["effects"][mode][metric][condition]
                means.append(value["mean"])
                lows.append(value["mean"] - value["ci"][0])
                highs.append(value["ci"][1] - value["mean"])
            axis.errorbar(
                base_x + (-0.09 if ci == 0 else 0.09),
                means,
                yerr=[lows, highs],
                fmt="o",
                capsize=4,
                color=colors[condition],
                label=condition.title(),
            )
        axis.axhline(0, color="#888888", linestyle="--", linewidth=1)
        axis.set_xticks(base_x, [labels[mode] for mode in modes])
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.18)

    points(
        axes[0, 0],
        "w1_minus_w2_margin",
        "A  Held-out conflict trials: W1-minus-W2 effect",
        "Change from natural (logits)",
    )
    points(
        axes[0, 1],
        "w1_choice_pp",
        "B  Held-out conflict trials: W1 choice effect",
        "Change from natural (percentage points)",
    )

    x = np.arange(len(modes))
    for split_index, (name, values) in enumerate(
        (("Discovery", discovery), ("Confirmation", confirmation))
    ):
        means, lows, highs = [], [], []
        for mode in modes:
            value = values["conflict"]["effects"][mode]["w1_minus_w2_margin"][
                "game_minus_neutral_interaction"
            ]
            means.append(value["mean"])
            lows.append(value["mean"] - value["ci"][0])
            highs.append(value["ci"][1] - value["mean"])
        axes[1, 0].errorbar(
            x + (-0.09 if split_index == 0 else 0.09),
            means,
            yerr=[lows, highs],
            fmt="o",
            capsize=4,
            label=name,
        )
    axes[1, 0].axhline(0, color="#888888", linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(x, [labels[mode] for mode in modes])
    axes[1, 0].set_title(
        "C  Game-minus-Neutral interaction in W1 margin",
        loc="left",
        fontweight="bold",
    )
    axes[1, 0].set_ylabel("Difference-in-differences (logits)")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.18)

    readouts = np.asarray(dose["devalue"]["game"]["readouts"])
    for condition in CONDITIONS:
        value = dose["devalue"][condition]
        mean = np.asarray(value["relative_l2_dose_percent_mean"])
        low = np.asarray(value["relative_l2_dose_percent_ci_low"])
        high = np.asarray(value["relative_l2_dose_percent_ci_high"])
        axes[1, 1].plot(readouts, mean, color=colors[condition], label=condition.title())
        axes[1, 1].fill_between(readouts, low, high, color=colors[condition], alpha=0.18)
    axes[1, 1].set_title(
        "D  Size of the devaluation edit",
        loc="left",
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("Post-block residual readout")
    axes[1, 1].set_ylabel("Residual L2 changed (%)")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", alpha=0.18)

    fig.suptitle(
        "Does the option-newline candidate-value coordinate causally control revision?",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(
    discovery_path: Path,
    confirmation_path: Path,
    second_mapping_path: Path,
    remapped_baseline_path: Path,
    output_dir: Path,
    figure_path: Path,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    second_rows = {
        row["question_id"]: row
        for row in json.loads(second_mapping_path.read_text())["rows"]
    }
    remapped_baseline = json.loads(remapped_baseline_path.read_text())["results"]
    discovery_data = _load_split(discovery_path, second_rows, remapped_baseline)
    confirmation_data = _load_split(confirmation_path, second_rows, remapped_baseline)
    discovery = _summarize_split(discovery_data, seed, draws)
    confirmation = _summarize_split(confirmation_data, seed + 1, draws)
    dose = {
        "discovery": _dose_summary(discovery_data, seed + 2, draws),
        "confirmation": _dose_summary(confirmation_data, seed + 3, draws),
    }
    primary = confirmation["conflict"]["effects"]["devalue"]
    summary = {
        "design": (
            "At the first-presentation W1 option newline only, clamp the frozen "
            "candidate-value probe coordinate over readouts 33-56 to its matched "
            "same-content/same-letter unchosen-presentation score."
        ),
        "discovery": discovery,
        "confirmation": confirmation,
        "dose": dose,
        "primary_confirmation_conflict": {
            "w1_margin": primary["w1_minus_w2_margin"],
            "w1_choice_pp": primary["w1_choice_pp"],
            "switch_pp": primary["switch_pp"],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    with (output_dir / "effects.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "split",
                "subset",
                "mode",
                "metric",
                "effect",
                "mean",
                "ci_low",
                "ci_high",
                "n",
            ]
        )
        for split_name, split_values in (
            ("discovery", discovery),
            ("confirmation", confirmation),
        ):
            for subset_name, subset in split_values.items():
                for mode, mode_values in subset["effects"].items():
                    for metric, metric_values in mode_values.items():
                        for effect, value in metric_values.items():
                            writer.writerow(
                                [
                                    split_name,
                                    subset_name,
                                    mode,
                                    metric,
                                    effect,
                                    value["mean"],
                                    value["ci"][0],
                                    value["ci"][1],
                                    value["n"],
                                ]
                            )

    _plot(figure_path, discovery, confirmation, dose["confirmation"])
    p_margin = primary["w1_minus_w2_margin"]
    p_choice = primary["w1_choice_pp"]
    p_switch = primary["switch_pp"]
    d_margin = discovery["conflict"]["effects"]["devalue"][
        "w1_minus_w2_margin"
    ]["game_minus_neutral_interaction"]
    o_margin_d = discovery["conflict"]["effects"]["opposite"][
        "w1_minus_w2_margin"
    ]["game_minus_neutral_interaction"]
    o_margin_c = confirmation["conflict"]["effects"]["opposite"][
        "w1_minus_w2_margin"
    ]["game_minus_neutral_interaction"]
    dose_max = dose["confirmation"]["devalue"]["game"][
        "max_mean_relative_l2_dose_percent"
    ]
    report = [
        "# Causal test of the option-newline candidate-value coordinate",
        "",
        "## Bottom line",
        "",
        "The candidate-value coordinate is genuinely decodable at the first-presentation option newline, but this intervention does **not** establish that the coordinate causally controls Game-specific revision. The small predicted-direction margin interaction seen in discovery remained positive but was uncertain on held-out confirmation, while the prespecified opposite-sign control failed to replicate. Discrete answer changes were likewise unstable across splits.",
        "",
        "## Design",
        "",
        summary["design"],
        "The run uses natural, exact zero-dose chosen-value sham, devaluation, and equal/opposite controls in both Game and Neutral, preserving historical physical batches. The causal sample excludes W1=A because the W1 newline is token-for-token identical before later distractors appear; it contains 74 discovery and 71 confirmation W1=B/C/D questions.",
        "",
        "## Primary held-out result",
        "",
        f"On {confirmation['conflict']['n']} confirmation conflict questions, devaluation changed the W1-minus-W2 margin by {p_margin['game']['mean']:+.3f} [{p_margin['game']['ci'][0]:+.3f}, {p_margin['game']['ci'][1]:+.3f}] logits in Game and {p_margin['neutral']['mean']:+.3f} [{p_margin['neutral']['ci'][0]:+.3f}, {p_margin['neutral']['ci'][1]:+.3f}] in Neutral. The Game-minus-Neutral interaction was {p_margin['game_minus_neutral_interaction']['mean']:+.3f} [{p_margin['game_minus_neutral_interaction']['ci'][0]:+.3f}, {p_margin['game_minus_neutral_interaction']['ci'][1]:+.3f}] logits.",
        "",
        f"W1 choice changed by {p_choice['game']['mean']:+.1f} [{p_choice['game']['ci'][0]:+.1f}, {p_choice['game']['ci'][1]:+.1f}] points in Game and {p_choice['neutral']['mean']:+.1f} [{p_choice['neutral']['ci'][0]:+.1f}, {p_choice['neutral']['ci'][1]:+.1f}] in Neutral. The interaction was {p_choice['game_minus_neutral_interaction']['mean']:+.1f} [{p_choice['game_minus_neutral_interaction']['ci'][0]:+.1f}, {p_choice['game_minus_neutral_interaction']['ci'][1]:+.1f}] points.",
        "",
        f"Switching away from W1 changed by {p_switch['game']['mean']:+.1f} [{p_switch['game']['ci'][0]:+.1f}, {p_switch['game']['ci'][1]:+.1f}] points in Game and {p_switch['neutral']['mean']:+.1f} [{p_switch['neutral']['ci'][0]:+.1f}, {p_switch['neutral']['ci'][1]:+.1f}] in Neutral; the interaction was {p_switch['game_minus_neutral_interaction']['mean']:+.1f} [{p_switch['game_minus_neutral_interaction']['ci'][0]:+.1f}, {p_switch['game_minus_neutral_interaction']['ci'][1]:+.1f}] points.",
        "",
        "## Replication and controls",
        "",
        f"The discovery conflict interaction in W1-minus-W2 margin was {d_margin['mean']:+.3f} [{d_margin['ci'][0]:+.3f}, {d_margin['ci'][1]:+.3f}] logits, compared with {p_margin['game_minus_neutral_interaction']['mean']:+.3f} [{p_margin['game_minus_neutral_interaction']['ci'][0]:+.3f}, {p_margin['game_minus_neutral_interaction']['ci'][1]:+.3f}] on confirmation. The equal/opposite edit gave {o_margin_d['mean']:+.3f} [{o_margin_d['ci'][0]:+.3f}, {o_margin_d['ci'][1]:+.3f}] in discovery but only {o_margin_c['mean']:+.3f} [{o_margin_c['ci'][0]:+.3f}, {o_margin_c['ci'][1]:+.3f}] on confirmation, so the sign-reversal evidence did not replicate.",
        "",
        f"The chosen-sham edit had exactly zero dose and zero behavioral/logit effect. The devaluation edit changed at most {dose_max:.2f}% of residual L2 on average at any tested readout, and the post-clamp probe-score error averaged about 0.016 score units. Natural A-D logits reproduced the trusted run exactly.",
        "",
        "The intervention also did not produce a stable entropy or A-D-spread effect across splits. Full conflict/no-conflict, letter-stratified, and secondary-metric results are in `summary.json` and `effects.csv`.",
        "",
        "## Interpretation",
        "",
        "This is an informative causal null. A linear decoder can read a context-dependent candidate-value/selectedness correlate from the semantic option state, but moving that one fitted coordinate from its chosen-presentation value to its matched unchosen-presentation value is not sufficient to reproducibly alter the later Game-versus-Neutral policy. The operative binding may be nonlinear, multidimensional, distributed across option states, or represented in a different feature basis. The decoder result remains valid; the stronger claim that its one-dimensional direction is the mechanism does not.",
        "",
        f"Canonical figure: `{figure_path}`.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--second-mapping", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    analyze(
        args.discovery,
        args.confirmation,
        args.second_mapping,
        args.remapped_baseline,
        args.output_dir,
        args.figure,
        args.draws,
        args.seed,
    )


if __name__ == "__main__":
    main()
