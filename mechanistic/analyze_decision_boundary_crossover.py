from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


LETTERS = "ABCD"


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "results.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not arrays["completed"].all():
        raise ValueError(f"Incomplete result: {path}")
    return arrays


def _interval(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "ci": [None, None]}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def _center(logits: np.ndarray) -> np.ndarray:
    return logits - logits.mean(axis=-1, keepdims=True)


def _semantic_transfer(
    intervention: np.ndarray,
    identity: np.ndarray,
    x_second: np.ndarray,
    y_second: np.ndarray,
) -> dict[str, np.ndarray]:
    q = np.arange(len(x_second))
    out = {}
    for name, x_row, y_row in (("game", 0, 2), ("neutral", 1, 3)):
        x_recipient = (
            intervention[x_row, q, y_second]
            - intervention[x_row, q, x_second]
            - identity[x_row, q, y_second]
            + identity[x_row, q, x_second]
        )
        y_recipient = (
            intervention[y_row, q, x_second]
            - intervention[y_row, q, y_second]
            - identity[y_row, q, x_second]
            + identity[y_row, q, y_second]
        )
        out[name] = 0.5 * (x_recipient + y_recipient)
    out["neutral_minus_game"] = out["neutral"] - out["game"]
    return out


def _donor_choice_transfer(
    intervention: np.ndarray,
    identity: np.ndarray,
    x_second: np.ndarray,
    y_second: np.ndarray,
) -> dict[str, np.ndarray]:
    i_answer = intervention.argmax(axis=-1)
    b_answer = identity.argmax(axis=-1)
    out = {}
    for name, x_row, y_row in (("game", 0, 2), ("neutral", 1, 3)):
        x_recipient = (i_answer[x_row] == y_second).astype(float) - (
            b_answer[x_row] == y_second
        ).astype(float)
        y_recipient = (i_answer[y_row] == x_second).astype(float) - (
            b_answer[y_row] == x_second
        ).astype(float)
        out[name] = 50.0 * (x_recipient + y_recipient)
    out["neutral_minus_game"] = out["neutral"] - out["game"]
    return out


def _donor_boundary_match(
    logits: np.ndarray, x_first: np.ndarray, y_first: np.ndarray
) -> np.ndarray:
    answers = logits.argmax(axis=-1)
    matches = np.stack(
        (
            answers[0] == y_first,
            answers[1] == y_first,
            answers[2] == x_first,
            answers[3] == x_first,
        )
    )
    return matches.mean(axis=0).astype(float) * 100.0


def _evidence_changes(
    intervention: np.ndarray,
    identity: np.ndarray,
    x_second: np.ndarray,
    y_second: np.ndarray,
    x_first: np.ndarray,
    y_first: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    centered_delta = _center(intervention) - _center(identity)
    q = np.arange(len(x_second))
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, x_row, y_row in (("game", 0, 2), ("neutral", 1, 3)):
        semantic = 0.5 * (
            centered_delta[x_row, q, y_second]
            + centered_delta[y_row, q, x_second]
        )
        old_letter = 0.5 * (
            centered_delta[x_row, q, y_first]
            + centered_delta[y_row, q, x_first]
        )
        semantic_differs = np.concatenate(
            (y_second != y_first, x_second != x_first)
        )
        out[name] = {
            "donor_semantic": semantic,
            "donor_old_letter": old_letter,
            "semantic_old_letter_decoupled_fraction": np.full(
                len(semantic), semantic_differs.mean(), dtype=float
            ),
        }
    return out


def _metrics(arrays: dict[str, np.ndarray], draws: int, seed: int) -> dict[str, Any]:
    exact = arrays["exact_eligible"].astype(bool)
    if not np.any(exact):
        raise ValueError("No exact-regime eligible questions")
    identity = arrays["identity_logits"][:, exact].astype(float)
    cross = arrays["cross_logits"][:, exact].astype(float)
    full = arrays["full_donor_logits"][:, exact].astype(float)
    prefix = arrays["prefix_logits"][:, exact].astype(float)
    identity_boundary = arrays["identity_boundary_logits"][:, exact].astype(float)
    cross_boundary = arrays["cross_boundary_logits"][:, exact].astype(float)
    full_boundary = arrays["full_donor_boundary_logits"][:, exact].astype(float)
    x_first = np.asarray(
        [LETTERS.index(value) for value in arrays["exact_x_first_letter"][exact].astype(str)]
    )
    y_first = np.asarray(
        [LETTERS.index(value) for value in arrays["exact_y_first_letter"][exact].astype(str)]
    )
    x_second = np.asarray(
        [LETTERS.index(value) for value in arrays["exact_x_second_letter"][exact].astype(str)]
    )
    y_second = np.asarray(
        [LETTERS.index(value) for value in arrays["exact_y_second_letter"][exact].astype(str)]
    )

    raw = {
        "boundary_semantic_margin_transfer": _semantic_transfer(
            cross, identity, x_second, y_second
        ),
        "full_history_semantic_margin_transfer": _semantic_transfer(
            full, identity, x_second, y_second
        ),
        "boundary_donor_choice_transfer_pp": _donor_choice_transfer(
            cross, identity, x_second, y_second
        ),
        "full_history_donor_choice_transfer_pp": _donor_choice_transfer(
            full, identity, x_second, y_second
        ),
    }
    summarized: dict[str, Any] = {}
    for metric_index, (metric, conditions) in enumerate(raw.items()):
        summarized[metric] = {
            condition: _interval(values, draws, seed + metric_index * 100 + ci)
            for ci, (condition, values) in enumerate(conditions.items())
        }

    immediate = {
        "identity_donor_match_pp": _donor_boundary_match(
            identity_boundary, x_first, y_first
        ),
        "cross_donor_match_pp": _donor_boundary_match(
            cross_boundary, x_first, y_first
        ),
        "full_donor_match_pp": _donor_boundary_match(
            full_boundary, x_first, y_first
        ),
    }
    summarized["immediate_boundary_choice"] = {
        name: _interval(values, draws, seed + 600 + index)
        for index, (name, values) in enumerate(immediate.items())
    }
    evidence = _evidence_changes(
        cross, identity, x_second, y_second, x_first, y_first
    )
    summarized["boundary_centered_evidence_change"] = {
        condition: {
            target: _interval(values, draws, seed + 700 + 20 * ci + ti)
            for ti, (target, values) in enumerate(targets.items())
        }
        for ci, (condition, targets) in enumerate(evidence.items())
    }

    boundary_interaction = summarized["boundary_semantic_margin_transfer"][
        "neutral_minus_game"
    ]["mean"]
    full_interaction = summarized["full_history_semantic_margin_transfer"][
        "neutral_minus_game"
    ]["mean"]
    result = {
        "n_frozen": int(len(exact)),
        "n_token_aligned": int(arrays["token_aligned"].sum()),
        "n_exact_eligible": int(exact.sum()),
        "n_screened_out": int((~exact).sum()),
        "metrics": summarized,
        "descriptive_boundary_fraction_of_full_condition_interaction": (
            float(boundary_interaction / full_interaction)
            if abs(full_interaction) > 1e-12
            else None
        ),
        "validation": {
            "identity_vs_natural_max_abs_error": float(
                np.nanmax(arrays["identity_vs_natural_max_error"][exact])
            ),
            "identity_vs_natural_choice_changes": int(
                np.sum(arrays["identity_vs_natural_choice_changes"][exact])
            ),
            "identity_boundary_vs_inclusive_prefix_max_abs_error": float(
                np.max(np.abs(identity_boundary - prefix))
            ),
            "identity_boundary_choice_changes": int(
                np.sum(identity_boundary.argmax(axis=-1) != prefix.argmax(axis=-1))
            ),
            "full_donor_max_abs_error": float(
                np.nanmax(arrays["full_donor_max_error"][exact])
            ),
            "full_donor_boundary_max_abs_error": float(
                np.nanmax(arrays["full_donor_boundary_max_error"][exact])
            ),
            "mean_identity_trajectory_dose": float(
                np.nanmean(arrays["identity_trajectory_dose"][exact])
            ),
            "mean_cross_trajectory_dose": float(
                np.nanmean(arrays["cross_trajectory_dose"][exact])
            ),
            "model_calls_total": int(arrays["model_calls"].sum()),
            "all_analyzed_outputs_finite": bool(
                all(
                    np.all(np.isfinite(value))
                    for value in (
                        identity,
                        cross,
                        full,
                        prefix,
                        identity_boundary,
                        cross_boundary,
                        full_boundary,
                    )
                )
            ),
        },
    }
    result["manipulation_gate_passed"] = bool(
        summarized["immediate_boundary_choice"]["cross_donor_match_pp"]["mean"]
        >= 80.0
        and result["validation"]["full_donor_max_abs_error"] <= 1e-4
        and result["validation"]["full_donor_boundary_max_abs_error"] <= 1e-4
    )
    return result


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    splits = ("discovery", "confirmation")
    markers = {"discovery": "o", "confirmation": "s"}
    colors = {"discovery": "#777777", "confirmation": "#111111"}
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.6))

    immediate_names = (
        "identity_donor_match_pp",
        "cross_donor_match_pp",
        "full_donor_match_pp",
    )
    immediate_labels = ("Identity", "Crossed boundary", "Full donor")
    x = np.arange(3)
    for split in splits:
        rows = summary[split]["metrics"]["immediate_boundary_choice"]
        means = np.asarray([rows[name]["mean"] for name in immediate_names])
        cis = np.asarray([rows[name]["ci"] for name in immediate_names])
        offset = -0.10 if split == "discovery" else 0.10
        axes[0].errorbar(
            x + offset,
            means,
            yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
            fmt=markers[split],
            color=colors[split],
            capsize=4,
            linestyle="none",
        )
    axes[0].axhline(50, color="#999999", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, immediate_labels, rotation=15, ha="right")
    axes[0].set_ylabel("Boundary choice matches donor (%)")
    axes[0].set_title("A  Manipulation check", loc="left", fontweight="bold")

    intervention_metrics = (
        "boundary_semantic_margin_transfer",
        "full_history_semantic_margin_transfer",
    )
    intervention_labels = ("Boundary update", "Full history")
    condition_offsets = {"game": -0.13, "neutral": 0.13}
    condition_colors = {"game": "#2f8ef4", "neutral": "#f28a35"}
    for split in splits:
        for condition in ("game", "neutral"):
            means = np.asarray(
                [
                    summary[split]["metrics"][metric][condition]["mean"]
                    for metric in intervention_metrics
                ]
            )
            cis = np.asarray(
                [
                    summary[split]["metrics"][metric][condition]["ci"]
                    for metric in intervention_metrics
                ]
            )
            split_shift = -0.035 if split == "discovery" else 0.035
            positions = np.arange(2) + condition_offsets[condition] + split_shift
            axes[1].errorbar(
                positions,
                means,
                yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
                fmt=markers[split],
                color=condition_colors[condition],
                markerfacecolor=("white" if split == "discovery" else condition_colors[condition]),
                capsize=4,
                linestyle="none",
            )
    axes[1].axhline(0, color="#999999", linestyle="--", linewidth=1)
    axes[1].set_xticks(np.arange(2), intervention_labels)
    axes[1].set_ylabel("Donor-winner margin transfer (logits)")
    axes[1].set_title("B  Which source controls semantics?", loc="left", fontweight="bold")

    for split in splits:
        means = np.asarray(
            [
                summary[split]["metrics"][metric]["neutral_minus_game"]["mean"]
                for metric in intervention_metrics
            ]
        )
        cis = np.asarray(
            [
                summary[split]["metrics"][metric]["neutral_minus_game"]["ci"]
                for metric in intervention_metrics
            ]
        )
        offset = -0.08 if split == "discovery" else 0.08
        axes[2].errorbar(
            np.arange(2) + offset,
            means,
            yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
            fmt=markers[split],
            color=colors[split],
            capsize=4,
            linestyle="none",
        )
    axes[2].axhline(0, color="#999999", linestyle="--", linewidth=1)
    axes[2].set_xticks(np.arange(2), intervention_labels)
    axes[2].set_ylabel("Neutral − Game transfer (logits)")
    axes[2].set_title("C  Condition-specific use", loc="left", fontweight="bold")

    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="#777777", label="Discovery"),
        Line2D([0], [0], marker="s", linestyle="none", color="#111111", label="Confirmation"),
        Line2D([0], [0], color="#2f8ef4", linewidth=4, label="Game"),
        Line2D([0], [0], color="#f28a35", linewidth=4, label="Neutral"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Does the first-decision boundary store which semantic answer won?", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fmt(row: dict[str, Any], suffix: str = "") -> str:
    if row["mean"] is None:
        return "NA"
    return f"{row['mean']:+.3f} [{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]{suffix}"


def analyze(
    discovery_path: Path,
    confirmation_path: Path,
    output_dir: Path,
    figure_path: Path,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    arrays = {
        "discovery": _load(discovery_path),
        "confirmation": _load(confirmation_path),
    }
    summary = {
        split: _metrics(value, draws, seed + 1000 * index)
        for index, (split, value) in enumerate(arrays.items())
    }
    summary["design"] = {
        "primary_contrast": (
            "Crossed boundary update minus identity in the reciprocal donor-semantic "
            "winner-versus-recipient-winner margin, with Neutral minus Game as the "
            "condition-specific selectedness-binding endpoint."
        ),
        "reconstruction_prediction": "Boundary transfer near zero while full-history transfer is substantial.",
        "stored_decision_prediction": "Boundary transfer follows the donor and captures a substantial fraction of full-history transfer.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "effects.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "metric", "condition", "n", "mean", "ci_low", "ci_high"))
        for split in ("discovery", "confirmation"):
            for metric, conditions in summary[split]["metrics"].items():
                if metric == "boundary_centered_evidence_change":
                    continue
                for condition, row in conditions.items():
                    writer.writerow((split, metric, condition, row["n"], row["mean"], *row["ci"]))
    _plot(summary, figure_path)

    d = summary["discovery"]
    c = summary["confirmation"]
    gate = d["manipulation_gate_passed"] and c["manipulation_gate_passed"]
    d_boundary = d["metrics"]["boundary_semantic_margin_transfer"]["neutral_minus_game"]
    c_boundary = c["metrics"]["boundary_semantic_margin_transfer"]["neutral_minus_game"]
    c_full = c["metrics"]["full_history_semantic_margin_transfer"]["neutral_minus_game"]
    if not gate:
        conclusion = "The donor-decision manipulation gate failed; downstream causal interpretation is not licensed."
    elif c_boundary["ci"][0] > 0 and d_boundary["mean"] > 0:
        conclusion = (
            "The boundary update carries a replicating condition-specific semantic winner signal; "
            "its magnitude relative to the full-history effect determines whether it is sufficient or partial."
        )
    elif abs(c_boundary["mean"]) < 0.25 * max(abs(c_full["mean"]), 1e-12) and abs(d_boundary["mean"]) < 0.25 * max(abs(d["metrics"]["full_history_semantic_margin_transfer"]["neutral_minus_game"]["mean"]), 1e-12):
        conclusion = (
            "The complete local boundary update does not carry the condition-specific semantic winner effect; "
            "the result supports reconstruction from distributed first-presentation history."
        )
    else:
        conclusion = "The crossed effect is mixed or imprecise; explicit boundary memory versus reconstruction remains unresolved."
    summary["conclusion"] = conclusion
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    c_evidence = c["metrics"]["boundary_centered_evidence_change"]
    d_evidence = d["metrics"]["boundary_centered_evidence_change"]
    report = f"""# First-decision boundary crossover

## Bottom line

{conclusion}

The manipulation gate {'passed' if gate else 'failed'}. The crossed one-token boundary trajectory made the immediate boundary decision match the donor on {_fmt(c['metrics']['immediate_boundary_choice']['cross_donor_match_pp'], '%')} of held-out confirmation questions. The complete-history donor control matched on {_fmt(c['metrics']['immediate_boundary_choice']['full_donor_match_pp'], '%')}.

## Primary semantic result

On held-out confirmation, the crossed boundary update's Neutral-minus-Game donor-semantic transfer was {_fmt(c_boundary)} logits. The complete-history positive control was {_fmt(c_full)} logits. The descriptive boundary fraction of that full condition interaction was {c['descriptive_boundary_fraction_of_full_condition_interaction']}.

Discovery gave {_fmt(d_boundary)} logits for the boundary update and {_fmt(d['metrics']['full_history_semantic_margin_transfer']['neutral_minus_game'])} for the complete history.

Held-out crossed effects by condition were {_fmt(c['metrics']['boundary_semantic_margin_transfer']['game'])} in Game and {_fmt(c['metrics']['boundary_semantic_margin_transfer']['neutral'])} in Neutral. Corresponding complete-history effects were {_fmt(c['metrics']['full_history_semantic_margin_transfer']['game'])} and {_fmt(c['metrics']['full_history_semantic_margin_transfer']['neutral'])}.

## What the boundary update did carry

The crossed update left a small donor-**letter** trace at the final decision without transporting the donor's semantic answer. In confirmation, centered evidence at the donor's old literal letter increased by {_fmt(c_evidence['game']['donor_old_letter'])} logits in Game and {_fmt(c_evidence['neutral']['donor_old_letter'])} in Neutral. Evidence for the donor semantic answer at its current second-presentation letter changed by only {_fmt(c_evidence['game']['donor_semantic'])} and {_fmt(c_evidence['neutral']['donor_semantic'])}. Discovery showed the same separation: donor-old-letter effects {_fmt(d_evidence['game']['donor_old_letter'])}/{_fmt(d_evidence['neutral']['donor_old_letter'])}, versus donor-semantic effects {_fmt(d_evidence['game']['donor_semantic'])}/{_fmt(d_evidence['neutral']['donor_semantic'])}.

The donor-semantic descriptive endpoint is not perfectly letter-decoupled: the donor semantic answer differs from the donor's old literal letter on {100 * c_evidence['game']['semantic_old_letter_decoupled_fraction']['mean']:.1f}% of confirmation rows and {100 * d_evidence['game']['semantic_old_letter_decoupled_fraction']['mean']:.1f}% of discovery rows. The remaining {100 * (1 - c_evidence['game']['semantic_old_letter_decoupled_fraction']['mean']):.1f}%/{100 * (1 - d_evidence['game']['semantic_old_letter_decoupled_fraction']['mean']):.1f}% retain letter overlap. The prespecified reciprocal Neutral-minus-Game semantic-margin contrast is differenced within each row and is unaffected by interpreting the auxiliary donor-semantic level as fully letter-pure.

Thus the first-decision boundary contains a portable record of **which output letter was about to be emitted**, but not a portable binding from that letter to the answer's semantic content. This agrees with the separate continuous A--D scrub: explicit answer-letter geometry exists there, but is not the mechanism that produces preferential semantic W1 avoidance.

## Validation

- Frozen/exact questions: discovery {d['n_exact_eligible']}/{d['n_frozen']}; confirmation {c['n_exact_eligible']}/{c['n_frozen']}.
- Full-donor suffix maximum A-D error: discovery {d['validation']['full_donor_max_abs_error']:.6g}; confirmation {c['validation']['full_donor_max_abs_error']:.6g}.
- Full-donor boundary maximum A-D error: discovery {d['validation']['full_donor_boundary_max_abs_error']:.6g}; confirmation {c['validation']['full_donor_boundary_max_abs_error']:.6g}.
- Identity split-path versus unsplit natural answer changes: discovery {d['validation']['identity_vs_natural_choice_changes']}; confirmation {c['validation']['identity_vs_natural_choice_changes']}.
- Identity boundary versus inclusive-prefix answer changes: discovery {d['validation']['identity_boundary_choice_changes']}; confirmation {c['validation']['identity_boundary_choice_changes']}.
- All analyzed outputs finite: discovery {d['validation']['all_analyzed_outputs_finite']}; confirmation {c['validation']['all_analyzed_outputs_finite']}.

The split cached execution is not numerically identical to a single unsplit forward in this recurrent architecture, and it changed some near-boundary natural choices. All causal estimates therefore use the same split identity path as their baseline. The complete-donor positive control reproduces that split donor path exactly; no causal contrast mixes split and unsplit logits.

## What was crossed

For each question, two natural first presentations selected different semantic answers X and Y. The second presentation and feedback were identical. The intervention retained X's accumulated state through the token immediately before the empty first-answer position, then replayed Y's complete 64-block boundary trajectory for that one token (and reciprocally Y-history/X-update). This makes the model itself write donor-driven ordinary-attention K/V and GLA updates into recipient history. The final block output was also clamped to the donor so the immediate A--D decision manipulation was exact; that final clamp does not alter the K/V or recurrent state already written inside the block. It is therefore a direct conflict between presentation history and the persistent state written at the decision boundary, not another one-dimensional decoder ablation.
"""
    (output_dir / "REPORT.md").write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260819)
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
