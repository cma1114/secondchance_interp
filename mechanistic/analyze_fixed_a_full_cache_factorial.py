from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FAMILIES = ("attention_kv", "gla_conv", "gla_recurrent")
MASK_LABELS = (
    "Identity",
    "Attention K/V",
    "GLA convolution",
    "K/V + convolution",
    "GLA recurrent matrix",
    "K/V + recurrent",
    "Convolution + recurrent",
    "Complete causal cache",
)


def _load(path: Path) -> dict[str, np.ndarray]:
    arrays = dict(np.load(path / "results.npz", allow_pickle=False))
    if not np.all(arrays["completed"]):
        raise ValueError(f"Incomplete result: {path}")
    return arrays


def _interval(values: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    means = values[indices].mean(axis=1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def _transfer_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    identity = arrays["cached_identity_logits"]
    factorial = arrays["factorial_logits"]
    x = np.asarray(["ABCD".index(v) for v in arrays["x_second_letter"].astype(str)])
    y = np.asarray(["ABCD".index(v) for v in arrays["y_second_letter"].astype(str)])
    q = np.arange(len(x))
    outputs: dict[str, list[np.ndarray]] = {
        "game": [],
        "neutral": [],
        "game_minus_neutral": [],
        "game_selection": [],
        "neutral_selection": [],
    }
    identity_answers = identity.argmax(axis=-1)
    for mask in range(8):
        patched = factorial[mask]
        game_x = (
            patched[0, q, x] - patched[0, q, y]
            - identity[0, q, x] + identity[0, q, y]
        )
        game_y = (
            patched[2, q, y] - patched[2, q, x]
            - identity[2, q, y] + identity[2, q, x]
        )
        neutral_x = (
            patched[1, q, x] - patched[1, q, y]
            - identity[1, q, x] + identity[1, q, y]
        )
        neutral_y = (
            patched[3, q, y] - patched[3, q, x]
            - identity[3, q, y] + identity[3, q, x]
        )
        game = 0.5 * (game_x + game_y)
        neutral = 0.5 * (neutral_x + neutral_y)
        outputs["game"].append(game)
        outputs["neutral"].append(neutral)
        outputs["game_minus_neutral"].append(game - neutral)

        answers = patched.argmax(axis=-1)
        outputs["game_selection"].append(
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
        outputs["neutral_selection"].append(
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
    return {key: np.stack(value) for key, value in outputs.items()}


def _shapley(metric: np.ndarray) -> np.ndarray:
    """Three-factor Shapley values per question; metric is [8, question]."""
    n_factors = 3
    output = np.zeros((n_factors, metric.shape[1]), dtype=float)
    for factor in range(n_factors):
        bit = 1 << factor
        for mask in range(8):
            if mask & bit:
                continue
            size = int(mask.bit_count())
            weight = (
                math.factorial(size)
                * math.factorial(n_factors - size - 1)
                / math.factorial(n_factors)
            )
            output[factor] += weight * (metric[mask | bit] - metric[mask])
    return output


def _summarize(
    root: Path, draws: int, seed: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    all_arrays = _load(root)
    if "first_decision_valid" in all_arrays:
        eligible = all_arrays["first_decision_valid"].astype(bool)
    else:
        eligible = np.all(
            all_arrays["first_decision_logits"].argmax(axis=-1) == 0, axis=0
        )
    if not np.any(eligible):
        raise ValueError(f"No exact-regime fixed-A questions in {root}")
    arrays = {
        "question_ids": all_arrays["question_ids"][eligible],
        "x_second_letter": all_arrays["x_second_letter"][eligible],
        "y_second_letter": all_arrays["y_second_letter"][eligible],
        "natural_logits": all_arrays["natural_logits"][:, eligible],
        "first_decision_logits": all_arrays["first_decision_logits"][:, eligible],
        "cached_identity_logits": all_arrays["cached_identity_logits"][:, eligible],
        "factorial_logits": all_arrays["factorial_logits"][:, :, eligible],
        "full_cache_donor_max_abs_error": all_arrays[
            "full_cache_donor_max_abs_error"
        ][eligible],
    }
    transfers = _transfer_arrays(arrays)
    n = len(arrays["question_ids"])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(draws, n))
    factorial_summary = []
    for mask in range(8):
        factorial_summary.append(
            {
                "mask": mask,
                "label": MASK_LABELS[mask],
                "families": [FAMILIES[i] for i in range(3) if mask & (1 << i)],
                **{
                    metric: _interval(values[mask], indices)
                    for metric, values in transfers.items()
                },
            }
        )

    shapley_summary: dict[str, Any] = {}
    for metric in ("game", "neutral", "game_minus_neutral"):
        values = _shapley(transfers[metric])
        shapley_summary[metric] = {
            family: _interval(values[i], indices)
            for i, family in enumerate(FAMILIES)
        }

    identity = arrays["cached_identity_logits"]
    natural = arrays["natural_logits"]
    first_answers = arrays["first_decision_logits"].argmax(axis=-1)
    x = np.asarray(["ABCD".index(v) for v in arrays["x_second_letter"].astype(str)])
    y = np.asarray(["ABCD".index(v) for v in arrays["y_second_letter"].astype(str)])
    q = np.arange(n)
    identity_answers = identity.argmax(axis=-1)
    prior_answer_margin = {
        "game": 0.5
        * (
            identity[0, q, x]
            - identity[0, q, y]
            + identity[2, q, y]
            - identity[2, q, x]
        ),
        "neutral": 0.5
        * (
            identity[1, q, x]
            - identity[1, q, y]
            + identity[3, q, y]
            - identity[3, q, x]
        ),
    }
    natural_prior_answer_margin = {
        "game": 0.5
        * (
            natural[0, q, x]
            - natural[0, q, y]
            + natural[2, q, y]
            - natural[2, q, x]
        ),
        "neutral": 0.5
        * (
            natural[1, q, x]
            - natural[1, q, y]
            + natural[3, q, y]
            - natural[3, q, x]
        ),
    }
    prior_answer_selection = {
        "game": 0.5
        * (
            (identity_answers[0] == x).astype(float)
            + (identity_answers[2] == y).astype(float)
        ),
        "neutral": 0.5
        * (
            (identity_answers[1] == x).astype(float)
            + (identity_answers[3] == y).astype(float)
        ),
    }
    return (
        {
            "root": str(root),
            "n": n,
            "n_historical_cohort": int(eligible.size),
            "n_exact_regime_excluded": int(np.sum(~eligible)),
            "factorial": factorial_summary,
            "shapley": shapley_summary,
            "cached_regime_prior_answer": {
                "margin": {
                    condition: _interval(values, indices)
                    for condition, values in prior_answer_margin.items()
                },
                "selection": {
                    condition: _interval(values, indices)
                    for condition, values in prior_answer_selection.items()
                },
            },
            "validation": {
                "full_cache_donor_max_abs_error": float(
                    arrays["full_cache_donor_max_abs_error"].max()
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
                "non_a_first_decisions": int(np.sum(first_answers != 0)),
                "cached_vs_unsplit_game_neutral_prior_margin_gap_difference": float(
                    np.mean(prior_answer_margin["neutral"] - prior_answer_margin["game"])
                    - np.mean(
                        natural_prior_answer_margin["neutral"]
                        - natural_prior_answer_margin["game"]
                    )
                ),
            },
        },
        transfers,
    )


def _fmt(value: dict[str, Any], scale: float = 1.0, suffix: str = "") -> str:
    mean = value["mean"] * scale
    lo, hi = np.asarray(value["ci"]) * scale
    return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]{suffix}"


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    splits = ("discovery", "confirmation")
    metrics = (
        ("game", "Game: donor-history transfer"),
        ("neutral", "Neutral: donor-history transfer"),
        ("game_minus_neutral", "Game minus Neutral"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    y = np.arange(8)
    offsets = {"discovery": -0.12, "confirmation": 0.12}
    colors = {"discovery": "#777777", "confirmation": "#2674d9"}
    for axis, (metric, title) in zip(axes, metrics):
        for split in splits:
            rows = summary[split]["factorial"]
            means = np.asarray([row[metric]["mean"] for row in rows])
            cis = np.asarray([row[metric]["ci"] for row in rows])
            axis.errorbar(
                means,
                y + offsets[split],
                xerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
                fmt="o",
                capsize=3,
                color=colors[split],
                label=split.title(),
            )
        axis.axvline(0, color="#999999", linewidth=1, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("Change in recipient-W1 vs donor-W1 margin (logits)")
        axis.grid(axis="x", alpha=0.18)
    axes[0].set_yticks(y, MASK_LABELS)
    axes[0].invert_yaxis()
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Fixed-A complete causal-cache transplant and decomposition", y=0.995)
    fig.text(
        0.5,
        0.012,
        "Negative = final evidence moves toward the donor history's previous semantic answer; 95% bootstrap CIs.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
                "Which persistent causal-state family at the first-decision boundary "
                "carries the fixed-A semantic-history effect?"
            ),
            "families": list(FAMILIES),
            "factorial_masks": {str(i): MASK_LABELS[i] for i in range(8)},
            "positive_control": (
                "The complete-cache transplant must reproduce the donor-history "
                "continuation because post-boundary tokens are identical."
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
        "# Fixed-A complete causal-cache transplant",
        "",
        "The first-decision causal cache is decomposed into conventional-attention K/V, "
        "GLA causal-convolution state, and GLA delta-rule recurrent matrices. All eight "
        "factorial combinations are compared with recipient-cache continuation.",
        "",
        "## Bottom line",
        "",
        "The causal influence of which semantic answer was selected on the first "
        "presentation is transmitted primarily through the conventional-attention "
        "K/V cache. In both splits, transplanting K/V alone "
        "reproduces the complete-cache semantic-history transfer, while neither GLA "
        "convolutional state nor GLA recurrent matrices transfer the donor answer's "
        "semantic identity without K/V. K/V alone is sufficient, and the continuous "
        "semantic-margin transfer without K/V is negligible. This localizes the "
        "persistent-state family, but not a token position: the K/V transplant includes "
        "every prefix position through the first-decision boundary and could contain "
        "either an explicit previous-answer record or distributed first-presentation "
        "information. It also does not yet identify the "
        "later computation that makes Game discount that memory more strongly than Neutral.",
        "",
        "## Metric definition",
        "",
        "For an X-history recipient, the intervention inserts state from the Y-history "
        "donor and measures the change in the X-minus-Y final logit margin; the symmetric "
        "Y-history comparison is averaged with it. **Negative transfer therefore means "
        "that the final answer moved toward the donor history's previous semantic answer.** "
        "A positive Game-minus-Neutral value means donor-history dependence is weaker in "
        "Game than in Neutral. The selection metric is the analogous change in which "
        "semantic answer wins the A-D argmax.",
        "",
        "This symmetric X↔Y crossover is not a test of whether deleting memory reduces "
        "overall switching. The complete-cache intervention exactly exchanges the two "
        "histories' continuations, so their aggregate switch rate is unchanged by "
        "construction. It replaces recipient answer X with donor answer Y rather than "
        "removing the existence of a previous answer.",
        "",
        "## Validation",
        "",
    ]
    for split in ("discovery", "confirmation"):
        val = summary[split]["validation"]
        lines.extend(
            [
                f"- {split.title()} complete-cache donor reproduction maximum error: "
                f"{val['full_cache_donor_max_abs_error']:.6g} logits.",
                f"- {split.title()} exact-regime fixed-A sample: "
                f"{summary[split]['n']}/{summary[split]['n_historical_cohort']} questions "
                f"({summary[split]['n_exact_regime_excluded']} screened out before intervention).",
                f"- {split.title()} non-A first decisions in the analyzed sample: "
                f"{val['non_a_first_decisions']}.",
                f"- {split.title()} cached identity versus unsplit natural answer differences: "
                f"{val['cached_identity_vs_unsplit_natural_answer_differences']}.",
                f"- {split.title()} cached versus unsplit difference in the "
                f"Neutral-minus-Game prior-answer margin gap: "
                f"{val['cached_vs_unsplit_game_neutral_prior_margin_gap_difference']:+.3f} logits.",
            ]
        )

    lines.extend(["", "## Factorial semantic transfer", ""])
    for split in ("discovery", "confirmation"):
        lines.extend(
            [
                f"### {split.title()}",
                "",
                "| Cache families transplanted | Game | Neutral | Game − Neutral | Game selection | Neutral selection |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary[split]["factorial"]:
            lines.append(
                f"| {row['label']} | {_fmt(row['game'], suffix=' logits')} | "
                f"{_fmt(row['neutral'], suffix=' logits')} | "
                f"{_fmt(row['game_minus_neutral'], suffix=' logits')} | "
                f"{_fmt(row['game_selection'], 100, ' pp')} | "
                f"{_fmt(row['neutral_selection'], 100, ' pp')} |"
            )
        lines.append("")

    lines.extend(["## Cached-regime natural dependence on the prior semantic answer", ""])
    lines.extend(
        [
            "These values are measured before transplantation. Positive margins and "
            "selection rates mean the final decision favors the semantic answer selected "
            "on the first presentation over the paired alternative.",
            "",
            "| Split | Game margin | Neutral margin | Game selection | Neutral selection |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in ("discovery", "confirmation"):
        values = summary[split]["cached_regime_prior_answer"]
        lines.append(
            f"| {split.title()} | {_fmt(values['margin']['game'], suffix=' logits')} | "
            f"{_fmt(values['margin']['neutral'], suffix=' logits')} | "
            f"{_fmt(values['selection']['game'], 100, ' pp')} | "
            f"{_fmt(values['selection']['neutral'], 100, ' pp')} |"
        )
    lines.append("")

    lines.extend(["## Shapley allocation of the complete-cache effect", ""])
    for split in ("discovery", "confirmation"):
        lines.extend(
            [
                f"### {split.title()}",
                "",
                "| State family | Game | Neutral | Game − Neutral |",
                "|---|---:|---:|---:|",
            ]
        )
        for family in FAMILIES:
            lines.append(
                f"| {family} | {_fmt(summary[split]['shapley']['game'][family], suffix=' logits')} | "
                f"{_fmt(summary[split]['shapley']['neutral'][family], suffix=' logits')} | "
                f"{_fmt(summary[split]['shapley']['game_minus_neutral'][family], suffix=' logits')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation discipline",
            "",
            "The complete-cache cell is an implementation positive control and a localization "
            "upper bound, not by itself a mechanistic discovery. The family-only cells and their "
            "interactions determine whether semantic history is carried primarily by ordinary "
            "attention memory, GLA convolutional history, GLA recurrent matrices, or a distributed "
            "combination.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1661)
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
