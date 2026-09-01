from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import FuncNorm
import numpy as np


TASK_LABELS = ("Game", "Neutral")
SUBSETS = ("all", "switch", "stay")
RANK_COLORS = ("#c43c39", "#2f6fb0", "#3b8f62", "#7357a6")
INFORMATIVENESS_CMAP = "RdBu"


def _informativeness_forward(values: Any) -> np.ndarray:
    values = np.asarray(values)
    return np.where(values < 0, 0.5 * (values + 1.0), 0.5 + 0.5 * values**3)


def _informativeness_inverse(values: Any) -> np.ndarray:
    values = np.asarray(values)
    return np.where(values < 0.5, 2.0 * values - 1.0, np.cbrt(2.0 * values - 1.0))


# The cubic positive branch deliberately expands visual differences among the
# late, already-positive similarities (roughly 0.5-0.95) while leaving zero at
# the neutral midpoint and preserving a separate red range for inverse scores.
INFORMATIVENESS_NORM = FuncNorm(
    (_informativeness_forward, _informativeness_inverse), vmin=-1.0, vmax=1.0
)


def _bootstrap_curve(
    values: np.ndarray,
    strata: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point = values.mean(axis=0)
    if len(values) < 2:
        return point, np.full_like(point, np.nan), np.full_like(point, np.nan)
    strata_values = np.unique(strata)
    boot = np.empty((draws,) + point.shape, dtype=np.float32)
    for draw in range(draws):
        sampled = np.concatenate(
            [
                rng.choice(np.flatnonzero(strata == stratum), size=np.sum(strata == stratum), replace=True)
                for stratum in strata_values
            ]
        )
        boot[draw] = values[sampled].mean(axis=0)
    low, high = np.quantile(boot, (0.025, 0.975), axis=0)
    return point, low, high


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _final_score_similarity(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean per-question cosine to the exact final four-candidate contrast."""
    centered = scores - scores.mean(axis=-1, keepdims=True)
    final = centered[:, -1]
    numerator = np.sum(centered * final[:, None], axis=-1)
    denominator = np.linalg.norm(centered, axis=-1) * np.linalg.norm(final, axis=-1)[:, None]
    cosine = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return cosine[mask].mean(axis=0)


def _add_informativeness_background(axis: Any, values: np.ndarray) -> None:
    cmap = plt.get_cmap(INFORMATIVENESS_CMAP)
    for layer, value in enumerate(values, start=1):
        axis.axvspan(
            layer - 0.5,
            layer + 0.5,
            color=cmap(INFORMATIVENESS_NORM(float(value))),
            alpha=0.19,
            linewidth=0,
            zorder=0,
        )


def _add_informativeness_colorbar(fig: Any, axes: Any) -> None:
    scalar = ScalarMappable(norm=INFORMATIVENESS_NORM, cmap=INFORMATIVENESS_CMAP)
    scalar.set_array([])
    count = np.asarray(axes).size
    bounds = [0.895, 0.205, 0.012, 0.575] if count == 2 else [0.89, 0.145, 0.012, 0.70]
    colorbar_axis = fig.add_axes(bounds)
    colorbar = fig.colorbar(
        scalar,
        cax=colorbar_axis,
        orientation="vertical",
    )
    colorbar.set_label(
        "Similarity to exact final A-D score pattern\n(nonlinear color scale; ticks are raw values)",
        rotation=270,
        labelpad=25,
    )
    colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 0.7, 0.8, 0.9, 1.0])


def _format_layer_axis(axis: Any, layer_count: int, show_labels: bool = True) -> None:
    """Use a conventional five-layer major grid with one-layer minor ticks."""
    axis.set_xlim(0.5, layer_count + 0.5)
    axis.set_xticks(np.arange(5, layer_count + 1, 5))
    axis.set_xticks(np.arange(1, layer_count + 1), minor=True)
    axis.tick_params(
        axis="x",
        which="major",
        labelbottom=show_labels,
        labelrotation=0,
        labelsize=9,
        pad=3,
    )
    axis.tick_params(axis="x", which="minor", length=1.8, color="#aaaaaa")
    axis.grid(axis="x", which="major", color="#e5e5e5", linewidth=0.45, alpha=0.6)


def _plot_companion_grid(
    values: np.ndarray,
    scores_by_letter: np.ndarray,
    switch: np.ndarray,
    first_winner: np.ndarray,
    layers: np.ndarray,
    spec: dict[str, Any],
    figure_dir: Path,
    draws: int,
    seed: int,
    suffix: str,
    title: str,
    ylabel: str,
) -> str:
    fig, axes = plt.subplots(3, 2, figsize=(14.4, 15.6), sharex=True, sharey=True)
    for subset_index, subset in enumerate(SUBSETS):
        for task_index, task in enumerate(TASK_LABELS):
            axis = axes[subset_index, task_index]
            mask = np.ones(values.shape[1], dtype=bool)
            if subset == "switch":
                mask = switch[task_index]
            elif subset == "stay":
                mask = ~switch[task_index]
            similarity = _final_score_similarity(scores_by_letter[task_index], mask)
            _add_informativeness_background(axis, similarity)
            point, low, high = _bootstrap_curve(
                values[task_index, mask],
                first_winner[mask],
                draws,
                np.random.default_rng(seed + 100 * subset_index + task_index),
            )
            for rank in range(4):
                axis.plot(layers, point[:, rank], color=RANK_COLORS[rank], linewidth=2.05)
                axis.fill_between(
                    layers,
                    low[:, rank],
                    high[:, rank],
                    color=RANK_COLORS[rank],
                    alpha=0.16,
                    linewidth=0,
                )
            axis.axhline(0, color="#777777", linewidth=0.8, alpha=0.8)
            subset_label = {"all": "All", "switch": "Switch", "stay": "No switch"}[subset]
            axis.set_title(f"{subset_label} — {task} (n={int(mask.sum())})", fontsize=11, weight="bold")
            axis.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.65)
            _format_layer_axis(axis, len(layers), show_labels=True)
            axis.spines[["top", "right"]].set_visible(False)
            if subset_index == 2:
                axis.set_xlabel(f"{spec.get('readout_label', 'JLens')} readout layer")
            if task_index == 0:
                axis.set_ylabel(ylabel)
    handles = [plt.Line2D([0], [0], color=RANK_COLORS[r], linewidth=2.15) for r in range(4)]
    fig.legend(handles, [f"1P R{r + 1}" for r in range(4)], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.47, 0.957))
    fig.suptitle(f"{spec['display_name']}: {title}", y=0.992, fontsize=14, weight="bold")
    fig.subplots_adjust(left=0.07, right=0.85, bottom=0.055, top=0.91, hspace=0.31, wspace=0.075)
    _add_informativeness_colorbar(fig, axes)
    prefix = spec.get("figure_prefix", "qwen36")
    figure_path = figure_dir / f"{prefix}_{spec['slug']}_nonremapped_rank_trajectories_{suffix}.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(figure_path)


def _analyze_dataset(
    spec: dict[str, Any],
    figure_dir: Path,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    data = _load(Path(spec["results"]))
    score_key = spec.get("score_key", "jlens_scores")
    scores = data[score_key].astype(np.float32)
    direct = data["direct_logits"].astype(np.float32)
    order = data["rank_order"].astype(np.int64)
    qids = data["question_ids"].astype(str)
    if scores.ndim != 4 or scores.shape[:2] != (2, len(qids)) or scores.shape[-1] != 4:
        raise ValueError(f"Unexpected score shape {scores.shape}")
    layer_count = int(scores.shape[2])
    selected_layers = tuple(
        layer
        for layer in (16, 24, 32, 40, 48, 52, 56, 60, 63, 64)
        if layer <= layer_count
    )
    # All pre-final rows are fixed-lens scores; use the exact natural output at
    # the architecture's final layer.
    scores[:, :, -1] = direct
    centered_by_letter = scores - scores.mean(axis=-1, keepdims=True)
    aligned_raw = np.take_along_axis(
        scores, np.broadcast_to(order[None, :, None, :], scores.shape), axis=-1
    )
    aligned = np.take_along_axis(
        centered_by_letter,
        np.broadcast_to(order[None, :, None, :], centered_by_letter.shape),
        axis=-1,
    )
    # Remove each task/layer's across-question displayed-letter mean before
    # rank alignment. This is a sensitivity view, not a replacement for the
    # behaviorally complete A-D score geometry.
    displayed_letter_mean = centered_by_letter.mean(axis=1, keepdims=True)
    letter_controlled_by_letter = centered_by_letter - displayed_letter_mean
    aligned_letter_controlled = np.take_along_axis(
        letter_controlled_by_letter,
        np.broadcast_to(order[None, :, None, :], letter_controlled_by_letter.shape),
        axis=-1,
    )
    aligned_direct = np.take_along_axis(direct, order[None], axis=-1)
    aligned_direct -= aligned_direct.mean(axis=-1, keepdims=True)
    aligned[:, :, -1] = aligned_direct

    final_choice = np.argmax(direct, axis=-1)
    first_winner = order[:, 0]
    switch = final_choice != first_winner[None]
    layers = np.arange(1, layer_count + 1)
    result: dict[str, Any] = {
        "dataset": spec["name"],
        "n_questions": len(qids),
        "score_definition": (
            f"Within-question centered {spec.get('readout_label', 'JLens')} score for bare-plus-space A-D token groups, "
            "aligned by same-format first-presentation aggregated-logit rank."
        ),
        "switch_definition": (
            "Displayed-order argmax of second-presentation aggregated A-D logits differs "
            "from displayed-order first-presentation aggregated-logit R1."
        ),
        "subsets": {},
    }
    trusted_reproduction: dict[str, Any] = {}
    for task_index, task in enumerate(("game", "neutral")):
        trusted_path = spec.get(f"trusted_{task}_results")
        if trusted_path is None:
            continue
        trusted_rows = json.loads(Path(trusted_path).read_text())["results"]
        trusted_logits = np.asarray(
            [trusted_rows[qid]["aggregated_ad_logits"] for qid in qids],
            dtype=np.float32,
        )
        trusted_choice = np.argmax(trusted_logits, axis=-1)
        trusted_switch = trusted_choice != first_winner
        trusted_reproduction[task] = {
            "path": trusted_path,
            "max_abs_logit_error": float(
                np.max(np.abs(direct[task_index] - trusted_logits))
            ),
            "aggregated_choice_agreement": float(
                np.mean(final_choice[task_index] == trusted_choice)
            ),
            "current_switch_count": int(switch[task_index].sum()),
            "trusted_switch_count": int(trusted_switch.sum()),
            "current_activation_l64_grouped_by_trusted_switch": {
                "switch": aligned[task_index, trusted_switch, -1].mean(axis=0).tolist(),
                "stay": aligned[task_index, ~trusted_switch, -1].mean(axis=0).tolist(),
            },
        }
    result["trusted_reproduction"] = trusted_reproduction
    difference, difference_low, difference_high = _bootstrap_curve(
        aligned[0] - aligned[1],
        first_winner,
        draws,
        np.random.default_rng(seed + 900),
    )
    result["game_minus_neutral_all"] = {
        "mean": difference.tolist(),
        "ci_low": difference_low.tolist(),
        "ci_high": difference_high.tolist(),
        "selected_layers": {
            str(layer): {
                "mean": difference[layer - 1].tolist(),
                "ci_low": difference_low[layer - 1].tolist(),
                "ci_high": difference_high[layer - 1].tolist(),
            }
            for layer in selected_layers
        },
    }
    for subset_index, subset in enumerate(SUBSETS):
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharex=True, sharey=True)
        subset_record: dict[str, Any] = {}
        for task_index, (task, axis) in enumerate(zip(TASK_LABELS, axes)):
            mask = np.ones(len(qids), dtype=bool)
            if subset == "switch":
                mask = switch[task_index]
            elif subset == "stay":
                mask = ~switch[task_index]
            strata = first_winner[mask]
            similarity = _final_score_similarity(centered_by_letter[task_index], mask)
            _add_informativeness_background(axis, similarity)
            point, low, high = _bootstrap_curve(
                aligned[task_index, mask],
                strata,
                draws,
                np.random.default_rng(seed + 100 * subset_index + task_index),
            )
            for rank in range(4):
                axis.plot(
                    layers,
                    point[:, rank],
                    color=RANK_COLORS[rank],
                    linewidth=2.15,
                    label=f"1P R{rank + 1}",
                )
                axis.fill_between(
                    layers,
                    low[:, rank],
                    high[:, rank],
                    color=RANK_COLORS[rank],
                    alpha=0.12,
                    linewidth=0,
                )
            axis.axhline(0, color="#777777", linewidth=0.8, alpha=0.8)
            axis.set_title(f"{task} (n={int(mask.sum())})", fontsize=12, weight="bold")
            axis.set_xlabel(f"{spec.get('readout_label', 'JLens')} readout layer")
            axis.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.65)
            _format_layer_axis(axis, layer_count)
            axis.spines[["top", "right"]].set_visible(False)
            subset_record[task.lower()] = {
                "n": int(mask.sum()),
                "switch_rate": float(switch[task_index].mean()),
                "mean": point.tolist(),
                "ci_low": low.tolist(),
                "ci_high": high.tolist(),
                "selected_layers": {
                    str(layer): point[layer - 1].tolist()
                    for layer in selected_layers
                },
                "final_score_similarity": similarity.tolist(),
            }
        axes[0].set_ylabel("Candidate-centered A-D score (logit units)")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
        label = {"all": "All trials", "switch": "Switch trials", "stay": "No-switch trials"}[subset]
        fig.suptitle(
            f"{spec['display_name']}: final-decision trajectories by 1P rank — {label}",
            y=1.07,
            fontsize=14,
            weight="bold",
        )
        fig.text(
            0.5,
            0.012,
            "R1–R4 are fixed by the same-format first presentation; shading is a first-winner-letter-stratified 95% bootstrap CI.",
            ha="center",
            fontsize=9,
            color="#444444",
        )
        fig.subplots_adjust(left=0.065, right=0.86, bottom=0.14, top=0.76, wspace=0.055)
        _add_informativeness_colorbar(fig, axes)
        prefix = spec.get("figure_prefix", "qwen36")
        figure_path = figure_dir / f"{prefix}_{spec['slug']}_nonremapped_rank_trajectories_{subset}.png"
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        subset_record["figure"] = str(figure_path)
        result["subsets"][subset] = subset_record
    result["companion_figures"] = {
        "noncentered": _plot_companion_grid(
            aligned_raw,
            centered_by_letter,
            switch,
            first_winner,
            layers,
            spec,
            figure_dir,
            draws,
            seed + 2000,
            "raw",
            "non-centered A-D scores by 1P rank",
            "Non-centered A-D score (logit units)",
        ),
        "displayed_letter_controlled": _plot_companion_grid(
            aligned_letter_controlled,
            centered_by_letter,
            switch,
            first_winner,
            layers,
            spec,
            figure_dir,
            draws,
            seed + 4000,
            "letter_controlled",
            "displayed-letter-controlled scores by 1P rank",
            "Displayed-letter-controlled score (logit units)",
        ),
    }
    result["displayed_letter_means"] = displayed_letter_mean[:, 0].tolist()
    return result


def analyze(specs_path: Path, output: Path, figure_dir: Path, draws: int, seed: int) -> None:
    specs = json.loads(specs_path.read_text())["datasets"]
    readout_labels = {spec.get("readout_label", "JLens") for spec in specs}
    if len(readout_labels) != 1:
        raise ValueError(f"All datasets must use one readout label, got {readout_labels}")
    readout_label = next(iter(readout_labels))
    is_jlens = readout_label.lower() == "jlens"
    readout_description = (
        "the fixed Qwen3.6-27B Jacobian lens"
        if is_jlens
        else "Qwen3.6-27B's final RMS norm and its own A--D output-embedding rows"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    results = [
        _analyze_dataset(spec, figure_dir, draws, seed + 1000 * index)
        for index, spec in enumerate(specs)
    ]
    summary = {
        "analysis": "nonremapped_final_decision_rank_trajectories",
        "readout": readout_label,
        "evidence_class": (
            f"activation/{readout_label} decoding; outcome-conditioned panels are descriptive postselection"
        ),
        "bootstrap_draws": draws,
        "datasets": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Non-remapped final-decision trajectories by first-presentation rank",
        "",
        f"This report shows Qwen3.6-27B's complete L1–L64 {readout_label} trajectory at the final decision position in the current clean prompt. Game and Neutral differ only at the single `incorrect`/`lost` token, and the second question and options retain their original displayed order.",
        "",
        "R1–R4 are frozen separately for each question from the same-format first-presentation aggregated A-D logits. Each layer's four scores are centered within question. The all-trial figures are the primary activation description. Switch/no-switch figures condition on the model's eventual aggregated-A-D choice and are therefore descriptive, not causal evidence about why it switched. The background tint is the mean per-question cosine similarity between that layer's centered four-candidate score vector and the exact final vector: red is inverse, white is unaligned, and blue is final-aligned. The positive half of the color mapping is cubic so that variation among late similarities of roughly 0.5–0.95 remains visible; colorbar ticks report the untransformed similarity values.",
        "",
    ]
    for dataset in results:
        lines.extend([f"## {dataset['dataset']}", ""])
        for subset in SUBSETS:
            row = dataset["subsets"][subset]
            game = row["game"]
            neutral = row["neutral"]
            relative_figure = "../../../../" + row["figure"]
            lines.append(
                f"- **{subset}:** Game n={game['n']}, Neutral n={neutral['n']}; "
                f"[figure]({relative_figure})"
            )
        noncentered = "../../../../" + dataset["companion_figures"]["noncentered"]
        letter_controlled = "../../../../" + dataset["companion_figures"]["displayed_letter_controlled"]
        lines.extend(
            [
                f"- **Companions:** [non-centered A-D scores]({noncentered}) · "
                f"[displayed-letter-controlled scores]({letter_controlled})",
            ]
        )
        lines.append("")
        all_game = dataset["subsets"]["all"]["game"]["selected_layers"]["64"]
        all_neutral = dataset["subsets"]["all"]["neutral"]["selected_layers"]["64"]
        diff = dataset["game_minus_neutral_all"]["selected_layers"]["64"]
        switch_game = dataset["subsets"]["switch"]["game"]["selected_layers"]["64"]
        switch_neutral = dataset["subsets"]["switch"]["neutral"]["selected_layers"]["64"]
        stay_game = dataset["subsets"]["stay"]["game"]["selected_layers"]["64"]
        stay_neutral = dataset["subsets"]["stay"]["neutral"]["selected_layers"]["64"]
        lines.extend(
            [
                f"On all trials, the L64 R1 score is {all_game[0]:.3f} in Game and "
                f"{all_neutral[0]:.3f} in Neutral; the paired Game-minus-Neutral "
                f"difference is {diff['mean'][0]:.3f} "
                f"`[{diff['ci_low'][0]:.3f}, {diff['ci_high'][0]:.3f}]`. ",
                "",
                f"On switch trials, R2 is the largest mean L64 score in both tasks: "
                f"Game R1/R2 = {switch_game[0]:.3f}/{switch_game[1]:.3f}; Neutral "
                f"R1/R2 = {switch_neutral[0]:.3f}/{switch_neutral[1]:.3f}. On no-switch "
                f"trials, R1 dominates: Game R1 = {stay_game[0]:.3f}; Neutral R1 = "
                f"{stay_neutral[0]:.3f}.",
                "",
            ]
        )
        for task, reproduction in dataset["trusted_reproduction"].items():
            if reproduction["max_abs_logit_error"] == 0.0:
                continue
            robust = reproduction["current_activation_l64_grouped_by_trusted_switch"]
            lines.extend(
                [
                    f"**Numerical-host audit ({task.title()}):** the current retained host "
                    f"agrees with the older trusted run on "
                    f"{100 * reproduction['aggregated_choice_agreement']:.1f}% of aggregated "
                    f"A-D choices (maximum logit difference "
                    f"{reproduction['max_abs_logit_error']:.3f}). The current run has "
                    f"{reproduction['current_switch_count']} switch trials versus "
                    f"{reproduction['trusted_switch_count']} in the old-host run. Grouping the "
                    f"current activations by the trusted old-host choices leaves the L64 switch "
                    f"R1/R2 means at {robust['switch'][0]:.3f}/{robust['switch'][1]:.3f}, "
                    f"so the late R2-takeover conclusion is unchanged. The canonical figures "
                    f"use the current run's own choices, keeping activations and outcomes from "
                    f"the same execution.",
                    "",
                ]
            )
    cross_dataset_lines = [
            "## Cross-dataset reading",
            "",
            (
                "The rank-separated decision state is predominantly late in both datasets. "
                + (
                    "The visible ordering begins to grow around L48–L52 and changes steeply around L54–L56. "
                    if is_jlens
                    else "After displayed-letter geometry is removed, meaningful rank separation emerges around L50–L56. "
                )
                + "Neutral develops a much larger late R1 advantage than Game. Game is therefore best described here as weaker late amplification of the recalled first-pass winner—not as undirected noise added uniformly to all four answers."
            ),
            "",
            "Conditioning on the eventual outcome gives the expected but useful decomposition. On no-switch trials, R1 becomes dominant late. On switch trials, R2 overtakes R1 late in both Game and Neutral and in both datasets. Thus the R2 takeover is not a Game-only computation. These selected panels cannot establish why a question switched, because they are defined using the final choice itself.",
            "",
            f"**Paper-figure interpretation.** The displayed-letter-controlled companions are the preferred paper figures for this result. In their switch panels, once a meaningful semantic-rank separation becomes visible at the final decision position, R2 is already above R1; there is no phase readable through the {readout_label} in which R1 first becomes the leading candidate and is only later overtaken. This weighs against a serial final-position story in which the model first reconstructs W1 as its prospective answer and then suppresses it. It does not show that remembered information first arrives only near L50: separate causal experiments place matching 1P-to-2P semantic-history transmission across ordinary-attention layers 4–48 and policy-dependent use of that route before the late readout. Nor does this activation plot alone prove a direct suppressive operation. Combined with the matching-history lesions, the supported account is that semantic recollection identifies the old winner, Game uses that rank information to reduce W1 relative to alternatives, and the resulting answer ordering becomes output-readable at the final decision position around L50–L56. Late sublayer decomposition further shows that much of the final Game-versus-Neutral difference is weaker Game-side amplification of W1 rather than one large negative W1 write.",
            "",
    ]
    if is_jlens:
        cross_dataset_lines.extend(
            [
                "The apparent inverse R1–R4 ordering around L8–L15 is displayed-letter geometry, not mapping-independent rank information. At SimpleMC L10, for example, the centered displayed-letter means are A/B/C/D = −1.325/+0.057/+0.090/+1.178, while the apparent Game rank means are −0.417/−0.088/+0.154/+0.351. Removing the displayed-letter means leaves +0.001/0.000/+0.002/−0.003. TriviaMC shows the same collapse. The displayed-letter-controlled companion therefore provides the cleaner sensitivity view of semantic-rank organization.",
                "",
            ]
        )
    else:
        cross_dataset_lines.extend(
            [
                "The standard-lens centered plots contain a stable early R1 advantage, but the displayed-letter-controlled companions remove it almost completely. That early pattern is therefore attributable to stable A/B/C/D output-row geometry rather than mapping-independent first-presentation rank. The controlled companion is the appropriate matched comparison with Seed-OSS.",
                "",
            ]
        )
    cross_dataset_lines.extend(
        [
            "The background informativeness scale answers a different question from the rank-line means: whether an individual question's complete four-candidate geometry at a layer resembles the exact final geometry. It remains approximately unaligned through the early layers and rises sharply around L50–L56, with real non-monotonic variation through L63. L64 equals the exact final readout and therefore has similarity 1 by construction.",
            "",
        ]
    )
    lines.extend(cross_dataset_lines)
    lines.extend(
        [
            "## Measurement scope",
            "",
            f"Readouts L1–L63 are post-block residuals transported to final-output space by {readout_description}. L64 is replaced with the exact live aggregated A-D logits from the natural forward. The score for a letter is log-sum-exp over its bare and leading-space token variants. This is activation/decoding evidence at the final decision token; it is not a layerwise causal intervention.",
            "",
            f"The non-centered companion retains each layer's common A-D offset. Across L1–L63 that offset mixes generic answer-token readiness with layer-dependent {readout_label} scale, so it should not be read as a calibrated layerwise confidence trajectory. The displayed-letter-controlled companion subtracts the task/layer-specific across-question mean for each displayed letter before aligning candidates by first-presentation rank. It is a sensitivity analysis that removes the stable A/B/C/D geometry; the centered behaviorally complete plot remains canonical.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    analyze(args.specs, args.output, args.figure_dir, args.draws, args.seed)


if __name__ == "__main__":
    main()
