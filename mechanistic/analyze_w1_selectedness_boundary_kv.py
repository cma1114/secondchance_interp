from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


LETTERS = "ABCD"


def _interval(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def _centered(logits: np.ndarray) -> np.ndarray:
    return logits - logits.mean(axis=-1, keepdims=True)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "results.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not arrays["completed"].all():
        raise ValueError(f"Incomplete result: {path}")
    return arrays


def _metrics(arrays: dict[str, np.ndarray], draws: int, seed: int) -> dict[str, Any]:
    exact = arrays["exact_eligible"].astype(bool)
    if not np.any(exact):
        raise ValueError("No exact-regime eligible questions")
    identity = arrays["cached_identity_logits"][:, exact].astype(float)
    boundary = arrays["boundary_kv_logits"][:, exact].astype(float)
    natural = arrays["natural_logits"][:, exact].astype(float)
    target = np.asarray(
        [LETTERS.index(value) for value in arrays["target_second_letter"][exact].astype(str)]
    )
    recipient = np.asarray(
        [
            LETTERS.index(value)
            for value in arrays["exact_recipient_winner_second_letter"][exact].astype(str)
        ]
    )
    q = np.arange(len(target))
    centered_identity = _centered(identity)
    centered_boundary = _centered(boundary)

    natural_target = {
        "game": centered_identity[0, q, target] - centered_identity[2, q, target],
        "neutral": centered_identity[1, q, target] - centered_identity[3, q, target],
    }
    causal_target = {
        "game": centered_boundary[2, q, target] - centered_identity[2, q, target],
        "neutral": centered_boundary[3, q, target] - centered_identity[3, q, target],
    }
    natural_margin = {
        "game": (
            identity[0, q, target] - identity[0, q, recipient]
            - identity[2, q, target] + identity[2, q, recipient]
        ),
        "neutral": (
            identity[1, q, target] - identity[1, q, recipient]
            - identity[3, q, target] + identity[3, q, recipient]
        ),
    }
    causal_margin = {
        "game": (
            boundary[2, q, target] - boundary[2, q, recipient]
            - identity[2, q, target] + identity[2, q, recipient]
        ),
        "neutral": (
            boundary[3, q, target] - boundary[3, q, recipient]
            - identity[3, q, target] + identity[3, q, recipient]
        ),
    }
    identity_answers = identity.argmax(axis=-1)
    boundary_answers = boundary.argmax(axis=-1)
    natural_choice = {
        "game": (identity_answers[0] == target).astype(float)
        - (identity_answers[2] == target).astype(float),
        "neutral": (identity_answers[1] == target).astype(float)
        - (identity_answers[3] == target).astype(float),
    }
    causal_choice = {
        "game": (boundary_answers[2] == target).astype(float)
        - (identity_answers[2] == target).astype(float),
        "neutral": (boundary_answers[3] == target).astype(float)
        - (identity_answers[3] == target).astype(float),
    }

    raw = {
        "natural_target_centered": natural_target,
        "causal_target_centered": causal_target,
        "natural_target_vs_recipient_margin": natural_margin,
        "causal_target_vs_recipient_margin": causal_margin,
        "natural_target_choice": natural_choice,
        "causal_target_choice": causal_choice,
    }
    summarized: dict[str, Any] = {}
    for metric_index, (metric, conditions) in enumerate(raw.items()):
        summarized[metric] = {
            condition: _interval(
                values,
                draws,
                seed + 100 * metric_index + condition_index,
            )
            for condition_index, (condition, values) in enumerate(conditions.items())
        }
        interaction = conditions["neutral"] - conditions["game"]
        summarized[metric]["neutral_minus_game"] = _interval(
            interaction, draws, seed + 100 * metric_index + 20
        )

    natural_interaction = summarized["natural_target_centered"][
        "neutral_minus_game"
    ]["mean"]
    causal_interaction = summarized["causal_target_centered"][
        "neutral_minus_game"
    ]["mean"]
    summary = {
        "n_frozen": int(len(exact)),
        "n_exact_eligible": int(exact.sum()),
        "n_screened_out": int((~exact).sum()),
        "metrics": summarized,
        "descriptive_centered_interaction_fraction": float(
            causal_interaction / natural_interaction
        )
        if abs(natural_interaction) > 1e-12
        else None,
        "validation": {
            "complete_cache_donor_max_abs_error": float(
                np.nanmax(arrays["complete_cache_donor_max_abs_error"][exact])
            ),
            "boundary_donor_control_max_abs_error": float(
                np.nanmax(arrays["boundary_donor_control_max_abs_error"][exact])
            ),
            "cached_identity_vs_unsplit_natural_mean_abs_error": float(
                np.mean(np.abs(identity - natural))
            ),
            "cached_identity_vs_unsplit_natural_max_abs_error": float(
                np.max(np.abs(identity - natural))
            ),
            "cached_identity_vs_unsplit_natural_answer_differences": int(
                np.sum(identity.argmax(axis=-1) != natural.argmax(axis=-1))
            ),
            "mean_boundary_kv_delta_l2": float(
                np.nanmean(arrays["boundary_kv_delta_l2"][exact])
            ),
            "mean_boundary_kv_delta_relative": float(
                np.nanmean(arrays["boundary_kv_delta_relative"][exact])
            ),
        },
    }
    causal = summarized["causal_target_centered"]
    summary["strict_selectedness_signature"] = bool(
        causal["game"]["mean"] < 0
        and causal["neutral"]["mean"] > 0
        and causal["neutral_minus_game"]["ci"][0] > 0
    )
    return summary


def _fmt(row: dict[str, Any], scale: float = 1.0, suffix: str = "") -> str:
    mean = row["mean"] * scale
    low, high = np.asarray(row["ci"]) * scale
    return f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]{suffix}"


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = (
        ("natural_target_centered", "A  Natural donor − recipient", "Centered target logit"),
        ("causal_target_centered", "B  Boundary K/V transplant", "Centered target-logit change"),
        ("causal_target_choice", "C  Boundary K/V transplant", "Target-choice change (pp)"),
    )
    conditions = ("game", "neutral", "neutral_minus_game")
    labels = ("Game", "Neutral", "Neutral − Game")
    colors = ("#2f8ef4", "#f28a35", "#555555")
    splits = ("discovery", "confirmation")
    offsets = {"discovery": -0.12, "confirmation": 0.12}
    markers = {"discovery": "o", "confirmation": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8))
    x = np.arange(3)
    for axis, (metric, title, ylabel) in zip(axes, panels):
        for split in splits:
            rows = summary[split]["metrics"][metric]
            means = np.asarray([rows[condition]["mean"] for condition in conditions])
            cis = np.asarray([rows[condition]["ci"] for condition in conditions])
            scale = 100.0 if metric.endswith("choice") else 1.0
            axis.errorbar(
                x + offsets[split],
                means * scale,
                yerr=np.vstack(
                    ((means - cis[:, 0]) * scale, (cis[:, 1] - means) * scale)
                ),
                fmt=markers[split],
                linestyle="none",
                capsize=4,
                color="#777777" if split == "discovery" else "#111111",
                label=split.title(),
                zorder=3,
            )
        axis.axhline(0, color="#999999", linewidth=1, linestyle="--")
        axis.set_xticks(x, labels)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.18)
        for tick, color in zip(axis.get_xticklabels(), colors):
            tick.set_color(color)
    split_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[split],
            linestyle="none",
            color="#777777" if split == "discovery" else "#111111",
            label=split.title(),
        )
        for split in splits
    ]
    fig.legend(
        handles=split_handles,
        labels=[split.title() for split in splits],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Does the empty first-answer boundary carry ‘A was selected’?",
        y=0.985,
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(
    discovery_path: Path,
    confirmation_path: Path,
    output_dir: Path,
    figure_path: Path,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    summary = {
        "definition": (
            "Recipient semantic A is fixed at the same first-presentation letter and "
            "position but is not selected. Ordinary-attention K/V at only the empty "
            "first-answer decision boundary is imported from the matched A-selected "
            "donor history."
        ),
        "discovery": _metrics(_load(discovery_path), draws, seed),
        "confirmation": _metrics(_load(confirmation_path), draws, seed + 10000),
    }
    summary["joint_gate_passed"] = bool(
        summary["discovery"]["strict_selectedness_signature"]
        and summary["confirmation"]["strict_selectedness_signature"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# First-decision selectedness K/V transplant",
        "",
        "## Bottom line",
        "",
        "This experiment asks whether ordinary-attention K/V at the final token of the empty first-answer scaffold carries the missing ‘A was selected’ signal. Semantic A and its displayed position are identical in donor and recipient; only the ordering of B-D changes whether A wins.",
        "",
        f"Exact-regime eligibility retained {summary['discovery']['n_exact_eligible']}/{summary['discovery']['n_frozen']} discovery and {summary['confirmation']['n_exact_eligible']}/{summary['confirmation']['n_frozen']} confirmation questions.",
        "",
        f"The strict opposite-direction selectedness signature {'passed' if summary['joint_gate_passed'] else 'did not pass'} on both frozen splits.",
        "",
        "## Natural donor-minus-recipient effect on semantic A",
        "",
        "Positive Neutral-minus-Game values mean that making A the first-pass winner favors its semantic content more under `lost` than under `incorrect`.",
        "",
        "| Split | Game centered A | Neutral centered A | Neutral minus Game |",
        "|---|---:|---:|---:|",
    ]
    for split in ("discovery", "confirmation"):
        rows = summary[split]["metrics"]["natural_target_centered"]
        lines.append(
            f"| {split.title()} | {_fmt(rows['game'])} | {_fmt(rows['neutral'])} | {_fmt(rows['neutral_minus_game'])} |"
        )
    lines.extend(
        [
            "",
            "## Causal effect of importing donor boundary K/V into recipient",
            "",
            "| Split | Game centered A | Neutral centered A | Neutral minus Game | Interaction fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in ("discovery", "confirmation"):
        rows = summary[split]["metrics"]["causal_target_centered"]
        fraction = summary[split]["descriptive_centered_interaction_fraction"]
        fraction_text = "--" if fraction is None else f"{100*fraction:.1f}%"
        lines.append(
            f"| {split.title()} | {_fmt(rows['game'])} | {_fmt(rows['neutral'])} | {_fmt(rows['neutral_minus_game'])} | {fraction_text} |"
        )
    lines.extend(
        [
            "",
            "### Target-versus-recipient-winner margin",
            "",
            "| Split | Game | Neutral | Neutral minus Game |",
            "|---|---:|---:|---:|",
        ]
    )
    for split in ("discovery", "confirmation"):
        rows = summary[split]["metrics"]["causal_target_vs_recipient_margin"]
        lines.append(
            f"| {split.title()} | {_fmt(rows['game'])} | {_fmt(rows['neutral'])} | {_fmt(rows['neutral_minus_game'])} |"
        )
    lines.extend(
        [
            "",
            "### Semantic-A choice rate",
            "",
            "| Split | Game | Neutral | Neutral minus Game |",
            "|---|---:|---:|---:|",
        ]
    )
    for split in ("discovery", "confirmation"):
        rows = summary[split]["metrics"]["causal_target_choice"]
        lines.append(
            f"| {split.title()} | {_fmt(rows['game'], 100, ' pp')} | {_fmt(rows['neutral'], 100, ' pp')} | {_fmt(rows['neutral_minus_game'], 100, ' pp')} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
        ]
    )
    for split in ("discovery", "confirmation"):
        validation = summary[split]["validation"]
        lines.extend(
            [
                f"- {split.title()} complete-cache donor maximum A-D error: {validation['complete_cache_donor_max_abs_error']:.6g}.",
                f"- {split.title()} untouched donor-row maximum A-D error: {validation['boundary_donor_control_max_abs_error']:.6g}.",
                f"- {split.title()} mean relative donor-recipient boundary K/V difference: {validation['mean_boundary_kv_delta_relative']:.4f}.",
            ]
        )
    lines.extend(
        [
            "",
            f"Canonical figure: `{figure_path}`.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines))
    _plot(summary, figure_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(
        args.discovery,
        args.confirmation,
        args.output_dir,
        args.figure,
        args.draws,
        args.seed,
    )


if __name__ == "__main__":
    main()
