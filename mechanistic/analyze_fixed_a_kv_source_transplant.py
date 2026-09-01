from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .run_fixed_a_kv_source_transplant import SOURCE_CELLS


CELL_LABELS = {
    "identity": "Identity",
    "selected_option": "Selected A option line",
    "question_without_selected": "Other first-question tokens",
    "first_question": "Entire first question",
    "decision_boundary": "First-decision boundary",
    "post_question_without_boundary": "Post-question cue/header",
    "selected_plus_boundary": "Selected option + boundary",
    "informative_prefix": "Entire informative prefix",
    "all_attention_kv": "All conventional-attention K/V",
    "complete_causal_cache": "Complete causal cache",
}


def _entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -(probabilities * np.log2(np.clip(probabilities, 1e-12, None))).sum(axis=-1)


def _interval(values: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    means = values[indices].mean(axis=1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def _load(root: Path) -> dict[str, np.ndarray]:
    arrays = dict(np.load(root / "results.npz", allow_pickle=False))
    if not np.all(arrays["completed"]):
        raise ValueError(f"Incomplete result: {root}")
    if arrays["source_cells"].astype(str).tolist() != list(SOURCE_CELLS):
        raise ValueError(f"Unexpected source cells in {root}")
    return arrays


def _transfer_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    identity = arrays["source_logits"][0]
    patched = arrays["source_logits"]
    x = np.asarray(["ABCD".index(v) for v in arrays["x_second_letter"].astype(str)])
    y = np.asarray(["ABCD".index(v) for v in arrays["y_second_letter"].astype(str)])
    q = np.arange(len(x))
    identity_answers = identity.argmax(axis=-1)
    identity_entropy = _entropy(identity)

    output: dict[str, list[np.ndarray]] = {
        "game_margin": [],
        "neutral_margin": [],
        "game_minus_neutral_margin": [],
        "game_selection_index": [],
        "neutral_selection_index": [],
        "game_donor_selection": [],
        "neutral_donor_selection": [],
        "game_recipient_selection": [],
        "neutral_recipient_selection": [],
        "game_entropy_bits": [],
        "neutral_entropy_bits": [],
        "game_answer_changed": [],
        "neutral_answer_changed": [],
    }
    for cell_index in range(len(SOURCE_CELLS)):
        cell = patched[cell_index]
        game_x = (
            cell[0, q, x] - cell[0, q, y]
            - identity[0, q, x] + identity[0, q, y]
        )
        game_y = (
            cell[2, q, y] - cell[2, q, x]
            - identity[2, q, y] + identity[2, q, x]
        )
        neutral_x = (
            cell[1, q, x] - cell[1, q, y]
            - identity[1, q, x] + identity[1, q, y]
        )
        neutral_y = (
            cell[3, q, y] - cell[3, q, x]
            - identity[3, q, y] + identity[3, q, x]
        )
        game = 0.5 * (game_x + game_y)
        neutral = 0.5 * (neutral_x + neutral_y)
        output["game_margin"].append(game)
        output["neutral_margin"].append(neutral)
        output["game_minus_neutral_margin"].append(game - neutral)

        answers = cell.argmax(axis=-1)
        output["game_selection_index"].append(
            0.5
            * (
                (answers[0] == x).astype(float)
                - (answers[0] == y).astype(float)
                - (identity_answers[0] == x).astype(float)
                + (identity_answers[0] == y).astype(float)
                + (answers[2] == y).astype(float)
                - (answers[2] == x).astype(float)
                - (identity_answers[2] == y).astype(float)
                + (identity_answers[2] == x).astype(float)
            )
        )
        output["neutral_selection_index"].append(
            0.5
            * (
                (answers[1] == x).astype(float)
                - (answers[1] == y).astype(float)
                - (identity_answers[1] == x).astype(float)
                + (identity_answers[1] == y).astype(float)
                + (answers[3] == y).astype(float)
                - (answers[3] == x).astype(float)
                - (identity_answers[3] == y).astype(float)
                + (identity_answers[3] == x).astype(float)
            )
        )
        output["game_donor_selection"].append(
            0.5
            * (
                (answers[0] == y).astype(float)
                - (identity_answers[0] == y).astype(float)
                + (answers[2] == x).astype(float)
                - (identity_answers[2] == x).astype(float)
            )
        )
        output["neutral_donor_selection"].append(
            0.5
            * (
                (answers[1] == y).astype(float)
                - (identity_answers[1] == y).astype(float)
                + (answers[3] == x).astype(float)
                - (identity_answers[3] == x).astype(float)
            )
        )
        output["game_recipient_selection"].append(
            0.5
            * (
                (answers[0] == x).astype(float)
                - (identity_answers[0] == x).astype(float)
                + (answers[2] == y).astype(float)
                - (identity_answers[2] == y).astype(float)
            )
        )
        output["neutral_recipient_selection"].append(
            0.5
            * (
                (answers[1] == x).astype(float)
                - (identity_answers[1] == x).astype(float)
                + (answers[3] == y).astype(float)
                - (identity_answers[3] == y).astype(float)
            )
        )
        entropy = _entropy(cell) - identity_entropy
        output["game_entropy_bits"].append(0.5 * (entropy[0] + entropy[2]))
        output["neutral_entropy_bits"].append(0.5 * (entropy[1] + entropy[3]))
        output["game_answer_changed"].append(
            0.5
            * (
                (answers[0] != identity_answers[0]).astype(float)
                + (answers[2] != identity_answers[2]).astype(float)
            )
        )
        output["neutral_answer_changed"].append(
            0.5
            * (
                (answers[1] != identity_answers[1]).astype(float)
                + (answers[3] != identity_answers[3]).astype(float)
            )
        )
    return {key: np.stack(values) for key, values in output.items()}


def _summarize(root: Path, draws: int, seed: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    complete = _load(root)
    eligible = complete["first_decision_valid"].astype(bool)
    if not np.any(eligible):
        raise ValueError(f"No exact-regime fixed-A questions in {root}")
    arrays = {
        "question_ids": complete["question_ids"][eligible],
        "x_second_letter": complete["x_second_letter"][eligible],
        "y_second_letter": complete["y_second_letter"][eligible],
        "natural_logits": complete["natural_logits"][:, eligible],
        "first_decision_logits": complete["first_decision_logits"][:, eligible],
        "source_logits": complete["source_logits"][:, :, eligible],
        "source_position_counts": complete["source_position_counts"][:, :, eligible],
        "complete_cache_donor_max_abs_error": complete[
            "complete_cache_donor_max_abs_error"
        ][eligible],
        "informative_prefix_vs_all_kv_max_abs_error": complete[
            "informative_prefix_vs_all_kv_max_abs_error"
        ][eligible],
    }
    transfers = _transfer_arrays(arrays)
    n = len(arrays["question_ids"])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(draws, n))
    cells = []
    for cell_index, cell in enumerate(SOURCE_CELLS):
        cells.append(
            {
                "cell": cell,
                "label": CELL_LABELS[cell],
                "mean_source_positions": float(
                    arrays["source_position_counts"][cell_index].mean()
                ),
                **{
                    metric: _interval(values[cell_index], indices)
                    for metric, values in transfers.items()
                },
            }
        )

    identity = arrays["source_logits"][0]
    natural = arrays["natural_logits"]
    x = np.asarray(["ABCD".index(v) for v in arrays["x_second_letter"].astype(str)])
    y = np.asarray(["ABCD".index(v) for v in arrays["y_second_letter"].astype(str)])
    q = np.arange(n)
    natural_targeting = 0.5 * (
        (natural[1, q, x] - natural[0, q, x] - natural[3, q, x] + natural[2, q, x])
        + (natural[3, q, y] - natural[2, q, y] - natural[1, q, y] + natural[0, q, y])
    )
    cached_targeting = 0.5 * (
        (identity[1, q, x] - identity[0, q, x] - identity[3, q, x] + identity[2, q, x])
        + (identity[3, q, y] - identity[2, q, y] - identity[1, q, y] + identity[0, q, y])
    )
    return (
        {
            "root": str(root),
            "n": n,
            "n_historical_cohort": int(eligible.size),
            "n_exact_regime_excluded": int(np.sum(~eligible)),
            "natural_semantic_targeting": _interval(natural_targeting, indices),
            "cached_identity_semantic_targeting": _interval(cached_targeting, indices),
            "cells": cells,
            "validation": {
                "complete_cache_donor_max_abs_error": float(
                    arrays["complete_cache_donor_max_abs_error"].max()
                ),
                "informative_prefix_vs_all_kv_max_abs_error": float(
                    arrays["informative_prefix_vs_all_kv_max_abs_error"].max()
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
            },
        },
        transfers,
    )


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    display_cells = list(SOURCE_CELLS[1:])
    labels = [CELL_LABELS[cell] for cell in display_cells]
    y = np.arange(len(display_cells))
    cell_indices = [SOURCE_CELLS.index(cell) for cell in display_cells]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = (
        (axes[0, 0], "game_margin", "A  Game semantic-memory transfer", "Logit-margin change"),
        (axes[0, 1], "neutral_margin", "B  Neutral semantic-memory transfer", "Logit-margin change"),
        (axes[1, 0], "game_minus_neutral_margin", "C  Game minus Neutral", "Differential logit-margin change"),
    )
    offsets = {"discovery": -0.11, "confirmation": 0.11}
    colors = {"discovery": "#777777", "confirmation": "#2674d9"}
    for axis, metric, title, xlabel in panels:
        for split in ("discovery", "confirmation"):
            rows = summary[split]["cells"]
            means = np.asarray([rows[i][metric]["mean"] for i in cell_indices])
            cis = np.asarray([rows[i][metric]["ci"] for i in cell_indices])
            axis.errorbar(
                means,
                y + offsets[split],
                xerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
                fmt="o",
                capsize=3,
                color=colors[split],
                label=split.title(),
            )
        axis.axvline(0, color="#999999", linestyle="--", linewidth=1)
        axis.set_title(title, loc="left")
        axis.set_xlabel(xlabel)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.18)

    axis = axes[1, 1]
    rows = summary["confirmation"]["cells"]
    for condition, offset, color in (
        ("game_donor_selection", -0.11, "#2f91f3"),
        ("neutral_donor_selection", 0.11, "#f0803c"),
    ):
        means = 100 * np.asarray([rows[i][condition]["mean"] for i in cell_indices])
        cis = 100 * np.asarray([rows[i][condition]["ci"] for i in cell_indices])
        axis.errorbar(
            means,
            y + offset,
            xerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
            fmt="o",
            capsize=3,
            color=color,
            label="Game" if condition.startswith("game") else "Neutral",
        )
    axis.axvline(0, color="#999999", linestyle="--", linewidth=1)
    axis.set_title("D  Held-out donor-answer selection", loc="left")
    axis.set_xlabel("Change in donor semantic answer chosen (percentage points)")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.18)
    axes[0, 0].legend(frameon=False, loc="best")
    axis.legend(frameon=False, loc="best")
    fig.suptitle("Where conventional-attention K/V stores the first semantic answer")
    fig.text(
        0.5,
        -0.012,
        "Negative = movement toward the donor history's semantic answer; positive Game-minus-Neutral = weaker donor reinstatement in Game. Points are means; bars are 95% question-bootstrap CIs.",
        ha="center",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fmt(value: dict[str, Any], scale: float = 1.0, suffix: str = "") -> str:
    mean = value["mean"] * scale
    lo, hi = np.asarray(value["ci"]) * scale
    return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]{suffix}"


def analyze(
    discovery_root: Path,
    confirmation_root: Path,
    output_dir: Path,
    figure_path: Path,
    draws: int,
    seed: int,
) -> None:
    discovery, _ = _summarize(discovery_root, draws, seed)
    confirmation, _ = _summarize(confirmation_root, draws, seed + 1)
    summary = {
        "design": {
            "question": (
                "Which first-presentation token region stores the semantic-history "
                "information carried by conventional-attention K/V?"
            ),
            "source_cells": list(SOURCE_CELLS),
            "primary_metric": (
                "Symmetric donor-versus-recipient semantic-answer margin transfer, "
                "separately in Game and Neutral and as Game minus Neutral."
            ),
        },
        "discovery": discovery,
        "confirmation": confirmation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _plot(summary, figure_path)

    lines = [
        "# Fixed-A conventional-attention K/V source localization",
        "",
        "## Metric",
        "",
        "The visible recipient prompt is unchanged. For each fixed-A X/Y pair, the "
        "intervention replaces conventional-attention K/V entries from a specified "
        "first-presentation token region with entries from the opposite semantic "
        "history. Negative margin transfer means the final answer moves toward the "
        "donor history's previous semantic answer. Positive Game-minus-Neutral means "
        "that donor-answer reinstatement is weaker in Game than in Neutral.",
        "",
        "This candidate-specific crossover is the primary endpoint. Aggregate switching "
        "is not a valid primary endpoint because the symmetric X↔Y complete-cache "
        "positive control permutes histories by construction.",
        "",
        "## Validation",
        "",
    ]
    for split in ("discovery", "confirmation"):
        row = summary[split]
        validation = row["validation"]
        lines.extend(
            [
                f"- {split.title()} exact-regime sample: {row['n']}/{row['n_historical_cohort']} "
                f"({row['n_exact_regime_excluded']} excluded before feedback).",
                f"- Complete causal-cache donor reproduction maximum A-D error: "
                f"{validation['complete_cache_donor_max_abs_error']:.6g} logits.",
                f"- Informative-prefix versus all-attention-K/V maximum A-D error: "
                f"{validation['informative_prefix_vs_all_kv_max_abs_error']:.6g} logits.",
                f"- Cached identity versus unsplit natural answer differences: "
                f"{validation['cached_identity_vs_unsplit_natural_answer_differences']}.",
                f"- Natural semantic targeting: {_fmt(row['natural_semantic_targeting'], suffix=' logits')}.",
                f"- Cached-identity semantic targeting: {_fmt(row['cached_identity_semantic_targeting'], suffix=' logits')}.",
            ]
        )

    for split in ("discovery", "confirmation"):
        lines.extend(
            [
                "",
                f"## {split.title()} source-region transfer",
                "",
                "| K/V source transplanted | Positions | Game margin | Neutral margin | Game − Neutral | Game donor chosen | Neutral donor chosen | Game entropy | Neutral entropy |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary[split]["cells"]:
            lines.append(
                f"| {row['label']} | {row['mean_source_positions']:.1f} | "
                f"{_fmt(row['game_margin'])} | {_fmt(row['neutral_margin'])} | "
                f"{_fmt(row['game_minus_neutral_margin'])} | "
                f"{_fmt(row['game_donor_selection'], 100, ' pp')} | "
                f"{_fmt(row['neutral_donor_selection'], 100, ' pp')} | "
                f"{_fmt(row['game_entropy_bits'], suffix=' bits')} | "
                f"{_fmt(row['neutral_entropy_bits'], suffix=' bits')} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation status",
            "",
            "The table is intentionally source-localization-first. A region should be "
            "treated as carrying semantic history only if its signed effect replicates "
            "across discovery and confirmation, approaches the complete-K/V control, "
            "and cannot be explained by entropy alone. Layer-band localization should "
            "be attempted only for such a region.",
            "",
            f"Canonical figure: `{figure_path}`.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=8675309)
    args = parser.parse_args()
    analyze(
        args.discovery,
        args.confirmation,
        args.output,
        args.figure,
        args.draws,
        args.seed,
    )


if __name__ == "__main__":
    main()
