from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CELLS = ("evaluation_x", "neutral_x", "evaluation_y", "neutral_y")


def _interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def _entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -(probabilities * np.log2(np.clip(probabilities, 1e-12, None))).sum(axis=-1)


def _load(root: Path) -> dict[str, np.ndarray]:
    arrays = dict(np.load(root / "results.npz", allow_pickle=False))
    if not np.all(arrays["completed"]):
        raise ValueError(f"Incomplete result: {root}")
    return arrays


def _metrics(root: Path, draws: int, seed: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = _load(root)
    natural = arrays["natural_logits"]
    patched = arrays["patched_batch_logits"]
    readouts = arrays["readouts"].astype(int)
    x = np.asarray(["ABCD".index(value) for value in arrays["x_second_letter"].astype(str)])
    y = np.asarray(["ABCD".index(value) for value in arrays["y_second_letter"].astype(str)])
    q = np.arange(len(x))

    natural_suppression_x = (
        natural[1, q, x] - natural[0, q, x]
        - natural[3, q, x] + natural[2, q, x]
    )
    natural_suppression_y = (
        natural[3, q, y] - natural[2, q, y]
        - natural[1, q, y] + natural[0, q, y]
    )
    natural_interaction = 0.5 * (natural_suppression_x + natural_suppression_y)

    game_x_delta = (
        patched[0, :, 0][:, q, x] - patched[0, :, 0][:, q, y]
        - (natural[0, q, x] - natural[0, q, y])[None, :]
    )
    game_y_delta = (
        patched[0, :, 2][:, q, y] - patched[0, :, 2][:, q, x]
        - (natural[2, q, y] - natural[2, q, x])[None, :]
    )
    neutral_x_delta = (
        patched[1, :, 1][:, q, x] - patched[1, :, 1][:, q, y]
        - (natural[1, q, x] - natural[1, q, y])[None, :]
    )
    neutral_y_delta = (
        patched[1, :, 3][:, q, y] - patched[1, :, 3][:, q, x]
        - (natural[3, q, y] - natural[3, q, x])[None, :]
    )
    game = 0.5 * (game_x_delta + game_y_delta)
    neutral = 0.5 * (neutral_x_delta + neutral_y_delta)
    divergence = game - neutral

    natural_answers = natural.argmax(axis=-1)
    patched_answers = patched.argmax(axis=-1)
    natural_game_x = (natural_answers[0] == x).astype(float) - (natural_answers[0] == y).astype(float)
    natural_game_y = (natural_answers[2] == y).astype(float) - (natural_answers[2] == x).astype(float)
    natural_neutral_x = (natural_answers[1] == x).astype(float) - (natural_answers[1] == y).astype(float)
    natural_neutral_y = (natural_answers[3] == y).astype(float) - (natural_answers[3] == x).astype(float)
    selection_game = 0.5 * (
        (patched_answers[0, :, 0] == x).astype(float)
        - (patched_answers[0, :, 0] == y).astype(float)
        - natural_game_x[None, :]
        + (patched_answers[0, :, 2] == y).astype(float)
        - (patched_answers[0, :, 2] == x).astype(float)
        - natural_game_y[None, :]
    )
    selection_neutral = 0.5 * (
        (patched_answers[1, :, 1] == x).astype(float)
        - (patched_answers[1, :, 1] == y).astype(float)
        - natural_neutral_x[None, :]
        + (patched_answers[1, :, 3] == y).astype(float)
        - (patched_answers[1, :, 3] == x).astype(float)
        - natural_neutral_y[None, :]
    )

    entropy_natural_game = 0.5 * (_entropy(natural[0]) + _entropy(natural[2]))
    entropy_natural_neutral = 0.5 * (_entropy(natural[1]) + _entropy(natural[3]))
    entropy_game = 0.5 * (
        _entropy(patched[0, :, 0]) + _entropy(patched[0, :, 2])
    ) - entropy_natural_game[None, :]
    entropy_neutral = 0.5 * (
        _entropy(patched[1, :, 1]) + _entropy(patched[1, :, 3])
    ) - entropy_natural_neutral[None, :]

    identity_logit_errors = []
    for li in range(len(readouts)):
        identity_logit_errors.extend(
            np.abs(patched[0, li, [1, 3]] - natural[[1, 3]]).ravel().tolist()
        )
        identity_logit_errors.extend(
            np.abs(patched[1, li, [0, 2]] - natural[[0, 2]]).ravel().tolist()
        )

    rng = np.random.default_rng(seed)
    layerwise = []
    for li, readout in enumerate(readouts):
        layerwise.append(
            {
                "readout": int(readout),
                "game_margin_transfer": _interval(game[li], rng, draws),
                "neutral_margin_transfer": _interval(neutral[li], rng, draws),
                "game_minus_neutral_margin_transfer": _interval(divergence[li], rng, draws),
                "game_selection_transfer": _interval(selection_game[li], rng, draws),
                "neutral_selection_transfer": _interval(selection_neutral[li], rng, draws),
                "game_entropy_change_bits": _interval(entropy_game[li], rng, draws),
                "neutral_entropy_change_bits": _interval(entropy_neutral[li], rng, draws),
            }
        )
    selected_index = int(np.nanargmax(divergence.mean(axis=1)))
    summary = {
        "root": str(root),
        "n": int(len(x)),
        "readouts": readouts.tolist(),
        "natural_semantic_targeting": _interval(natural_interaction, rng, draws),
        "selected_readout": int(readouts[selected_index]),
        "selection_rule": "maximum discovery Game-minus-Neutral signed semantic margin transfer",
        "layerwise": layerwise,
        "max_identity_patch_ad_logit_error": float(max(identity_logit_errors)),
        "max_identity_source_error_norm": float(np.nanmax(arrays["identity_source_error_norm"])),
        "mean_donor_identity_delta_norm_at_selected": float(
            np.nanmean(arrays["donor_identity_delta_norm"][:, selected_index])
        ),
    }
    cache = {
        "readouts": readouts,
        "game": game,
        "neutral": neutral,
        "divergence": divergence,
        "selection_game": selection_game,
        "selection_neutral": selection_neutral,
    }
    return summary, cache


def _plot(
    discovery: dict[str, Any],
    confirmation: dict[str, Any] | None,
    output: Path,
) -> None:
    readouts = np.asarray(discovery["readouts"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, metric, title, ylabel in (
        (axes[0], "margin_transfer", "Semantic suppression-target transfer", "Logit-margin change"),
        (axes[1], "selection_transfer", "Answer-selection transfer", "Probability-point change"),
    ):
        for condition, color in (("game", "#2f91f3"), ("neutral", "#f0803c")):
            key = f"{condition}_{metric}"
            means = np.asarray([row[key]["mean"] for row in discovery["layerwise"]])
            cis = np.asarray([row[key]["ci"] for row in discovery["layerwise"]])
            if metric == "selection_transfer":
                means *= 100
                cis *= 100
            ax.plot(readouts, means, color=color, marker="o", markersize=3, label=condition.title())
            ax.fill_between(readouts, cis[:, 0], cis[:, 1], color=color, alpha=0.18)
            if confirmation is not None:
                row = confirmation["layerwise"][0]
                mean = row[key]["mean"] * (100 if metric == "selection_transfer" else 1)
                ci = np.asarray(row[key]["ci"]) * (100 if metric == "selection_transfer" else 1)
                x = confirmation["selected_readout"]
                ax.errorbar([x], [mean], yerr=[[mean - ci[0]], [ci[1] - mean]], fmt="D", color=color, capsize=4)
        ax.axhline(0, color="#555", linewidth=1)
        ax.axvline(discovery["selected_readout"], color="#777", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Post-block residual readout")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
    fig.suptitle("Fixed-A first-decision residual swap", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze(
    discovery_root: Path,
    confirmation_root: Path | None,
    output_dir: Path,
    figure: Path,
    draws: int,
) -> None:
    discovery, _ = _metrics(discovery_root, draws, 1701)
    confirmation = None
    if confirmation_root is not None:
        confirmation, _ = _metrics(confirmation_root, draws, 2701)
        if confirmation["readouts"] != [discovery["selected_readout"]]:
            raise ValueError("Confirmation did not use the frozen selected readout")
        confirmation["selected_readout"] = discovery["selected_readout"]
    payload = {"discovery": discovery, "confirmation": confirmation}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _plot(discovery, confirmation, figure)

    decisive = confirmation or discovery
    row = decisive["layerwise"][0] if confirmation else next(
        row for row in discovery["layerwise"] if row["readout"] == discovery["selected_readout"]
    )
    label = "Confirmation" if confirmation else "Discovery selection only"
    report = [
        "# Fixed-A first-decision semantic-state transfer",
        "",
        "The literal first decision is `A` in both histories, but A names different semantic answers X and Y. The complete first-decision residual is exchanged between those histories while feedback and the second presentation remain fixed.",
        "",
        "## Result",
        "",
        f"Discovery selected post-block readout **{discovery['selected_readout']}** by the frozen Game-minus-Neutral semantic-transfer rule.",
        "",
        f"### {label}",
        "",
        "| Outcome | Estimate [95% CI] |",
        "|---|---:|",
    ]
    for label_text, key, scale, suffix in (
        ("Game semantic-target margin transfer", "game_margin_transfer", 1, " logits"),
        ("Neutral semantic-target margin transfer", "neutral_margin_transfer", 1, " logits"),
        ("Game minus Neutral transfer", "game_minus_neutral_margin_transfer", 1, " logits"),
        ("Game answer-selection transfer", "game_selection_transfer", 100, " pp"),
        ("Neutral answer-selection transfer", "neutral_selection_transfer", 100, " pp"),
    ):
        value = row[key]
        report.append(
            f"| {label_text} | {scale*value['mean']:+.3f} [{scale*value['ci'][0]:+.3f}, {scale*value['ci'][1]:+.3f}]{suffix} |"
        )
    report.extend(
        [
            "",
            "Positive Game transfer means exchanging the first-decision state moved later suppression toward the donor semantic answer. Negative Neutral transfer means it moved repetition toward the donor semantic answer.",
            "",
            "## Validation",
            "",
            f"- Discovery natural semantic-targeting interaction: {discovery['natural_semantic_targeting']['mean']:.3f} [{discovery['natural_semantic_targeting']['ci'][0]:.3f}, {discovery['natural_semantic_targeting']['ci'][1]:.3f}] logits.",
            f"- Maximum identity-patch A-D logit error: {decisive['max_identity_patch_ad_logit_error']:.8f}.",
            f"- Maximum identity source residual error: {decisive['max_identity_source_error_norm']:.8f}.",
            "",
            f"Canonical figure: `{figure}`.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    analyze(args.discovery, args.confirmation, args.output_dir, args.figure, args.draws)


if __name__ == "__main__":
    main()
