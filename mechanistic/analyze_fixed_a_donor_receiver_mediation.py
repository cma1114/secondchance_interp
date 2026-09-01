from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .run_fixed_a_donor_receiver_mediation import SCENARIOS
from .run_fixed_a_kv_source_transplant import SOURCE_CELLS


def _load(root: Path) -> dict[str, np.ndarray]:
    with np.load(root / "results.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"].astype(bool)):
        raise RuntimeError(f"Incomplete result: {root}")
    if arrays["scenarios"].astype(str).tolist() != list(SCENARIOS):
        raise RuntimeError(f"Unexpected scenarios: {root}")
    return arrays


def _ci(values: np.ndarray, draws: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    means = values[draws].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(means, (0.025, 0.975)).tolist(),
    }


def _oriented_margin(logits: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Donor-semantic minus recipient-semantic margin for each X/Y row."""
    q = np.arange(len(x))
    out = np.empty((4, len(x)), dtype=float)
    out[0] = logits[0, q, y] - logits[0, q, x]
    out[1] = logits[1, q, y] - logits[1, q, x]
    out[2] = logits[2, q, x] - logits[2, q, y]
    out[3] = logits[3, q, x] - logits[3, q, y]
    return out


def _oriented_donor_choice(logits: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    answers = logits.argmax(axis=-1)
    out = np.empty((4, len(x)), dtype=float)
    out[0] = answers[0] == y
    out[1] = answers[1] == y
    out[2] = answers[2] == x
    out[3] = answers[3] == x
    return out


def _condition_average(values: np.ndarray, condition: str) -> np.ndarray:
    rows = (0, 2) if condition == "Game" else (1, 3)
    return 0.5 * (values[rows[0]] + values[rows[1]])


def _split_summary(
    root: Path,
    prior_root: Path,
    seed: int,
    bootstrap: int,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    raw = _load(root)
    valid = raw["first_decision_valid"].astype(bool)
    if not np.any(valid):
        raise RuntimeError(f"No valid fixed-A rows: {root}")
    logits = raw["scenario_logits"][:, :, valid].astype(float)
    x = np.array(["ABCD".index(v) for v in raw["matching_query_letters"][2, valid].astype(str)])
    y = np.array(["ABCD".index(v) for v in raw["matching_query_letters"][0, valid].astype(str)])
    # Rows 0/1 are X recipients and therefore target donor Y; rows 2/3 are Y
    # recipients and target donor X. The stored matching letters must reflect that.
    if not np.array_equal(
        raw["matching_query_letters"][0, valid], raw["matching_query_letters"][1, valid]
    ) or not np.array_equal(
        raw["matching_query_letters"][2, valid], raw["matching_query_letters"][3, valid]
    ):
        raise RuntimeError("Game and Neutral use different matching receivers")
    margins = np.stack([_oriented_margin(row, x, y) for row in logits])
    choices = np.stack([_oriented_donor_choice(row, x, y) for row in logits])
    scenario = {name: i for i, name in enumerate(SCENARIOS)}

    raw_metrics: dict[str, dict[str, np.ndarray]] = {}
    for condition in ("Game", "Neutral"):
        open_transfer = _condition_average(
            margins[scenario["donor_open"]] - margins[scenario["recipient_open"]], condition
        )
        matching_transfer = _condition_average(
            margins[scenario["donor_matching_block"]]
            - margins[scenario["recipient_matching_block"]],
            condition,
        )
        control_transfer = _condition_average(
            margins[scenario["donor_control_block"]]
            - margins[scenario["recipient_control_block"]],
            condition,
        )
        open_choice = _condition_average(
            choices[scenario["donor_open"]] - choices[scenario["recipient_open"]], condition
        )
        matching_choice = _condition_average(
            choices[scenario["donor_matching_block"]]
            - choices[scenario["recipient_matching_block"]],
            condition,
        )
        raw_metrics[condition] = {
            "open_transfer": open_transfer,
            "matching_blocked_transfer": matching_transfer,
            "control_blocked_transfer": control_transfer,
            "matching_mediation": open_transfer - matching_transfer,
            "control_mediation": open_transfer - control_transfer,
            "matching_specific_mediation": control_transfer - matching_transfer,
            "open_donor_choice_transfer": open_choice,
            "matching_blocked_donor_choice_transfer": matching_choice,
            "donor_choice_mediation": open_choice - matching_choice,
        }
    raw_metrics["Game minus Neutral"] = {
        key: raw_metrics["Game"][key] - raw_metrics["Neutral"][key]
        for key in raw_metrics["Game"]
    }

    n = int(valid.sum())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(bootstrap, n))
    summarized = {
        group: {metric: _ci(values, draws) for metric, values in rows.items()}
        for group, rows in raw_metrics.items()
    }

    prior = dict(np.load(prior_root / "results.npz", allow_pickle=False))
    prior_valid = prior["first_decision_valid"].astype(bool)
    prior_ids = prior["question_ids"].astype(str).tolist()
    current_ids = raw["question_ids"][valid].astype(str).tolist()
    prior_index = {qid: i for i, qid in enumerate(prior_ids)}
    if any(qid not in prior_index for qid in current_ids):
        raise RuntimeError("Current fixed-A rows are absent from prior source run")
    comparable = np.asarray(
        [prior_valid[prior_index[qid]] for qid in current_ids], dtype=bool
    )
    selected_all = prior["source_logits"][SOURCE_CELLS.index("selected_option")]
    selected = selected_all[:, [prior_index[qid] for qid in current_ids]]
    donor_open = logits[scenario["donor_open"]]
    recipient_open = logits[scenario["recipient_open"]]
    natural = raw["natural_logits"][:, valid]
    validation = {
        "historical_rows": int(len(valid)),
        "valid_rows": n,
        "excluded_non_A_first_decision": int((~valid).sum()),
        "all_position_counts_positive": bool(
            np.all(raw["source_position_counts"][:, valid] > 0)
            and np.all(raw["matching_query_counts"][:, valid] > 0)
            and np.all(raw["control_query_counts"][:, valid] > 0)
        ),
        "recipient_open_vs_natural_mean_abs_logit_error": float(
            np.mean(np.abs(recipient_open - natural))
        ),
        "recipient_open_vs_natural_answer_agreement": float(
            (recipient_open.argmax(-1) == natural.argmax(-1)).mean()
        ),
        "prior_selected_line_comparable_rows": int(comparable.sum()),
        "donor_open_vs_prior_selected_line_mean_abs_logit_error": float(
            np.mean(np.abs(donor_open[:, comparable] - selected[:, comparable]))
        ),
        "donor_open_vs_prior_selected_line_answer_agreement": float(
            (
                donor_open[:, comparable].argmax(-1)
                == selected[:, comparable].argmax(-1)
            ).mean()
        ),
    }
    return {"root": str(root), "n": n, "metrics": summarized, "validation": validation}, raw_metrics


def _fmt(row: dict[str, Any], scale: float = 1.0) -> str:
    lo, hi = row["ci"]
    return f"{row['mean']*scale:+.3f} [{lo*scale:+.3f}, {hi*scale:+.3f}]"


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = ("open_transfer", "matching_blocked_transfer", "control_blocked_transfer")
    labels = ("Open", "Matching receiver blocked", "Nonmatching receiver blocked")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, split in zip(axes, ("discovery", "confirmation")):
        x = np.arange(3)
        for offset, condition, color in ((-0.12, "Game", "#2878b5"), (0.12, "Neutral", "#e07a2f")):
            rows = summary[split]["metrics"][condition]
            means = np.array([rows[m]["mean"] for m in metrics])
            cis = np.array([rows[m]["ci"] for m in metrics])
            axis.errorbar(x + offset, means, yerr=np.vstack((means-cis[:, 0], cis[:, 1]-means)),
                          fmt="o", capsize=4, color=color, label=condition)
        axis.axhline(0, color="#888888", linewidth=1)
        axis.set_xticks(x, labels, rotation=14, ha="right")
        axis.set_ylabel("Donor-semantic logit-margin transfer")
        axis.set_title(split.title())
        axis.grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False)
    fig.suptitle("Does the matching repeated option mediate first-answer semantic transfer?")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--prior-discovery", type=Path, required=True)
    parser.add_argument("--prior-confirmation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    discovery, _ = _split_summary(args.discovery, args.prior_discovery, args.seed, args.bootstrap)
    confirmation, _ = _split_summary(
        args.confirmation, args.prior_confirmation, args.seed + 1, args.bootstrap
    )
    summary = {
        "design": {
            "primary_question": "Does the semantically matching repeated option line mediate the fixed-A donor-history effect?",
            "positive_transfer": "movement toward the donor history's previous semantic answer",
            "primary_mediation": "open donor transfer minus donor transfer when the matching repeated receiver is blocked",
            "specificity_control": "same calculation while blocking a token-count-matched nonmatching receiver",
        },
        "discovery": discovery,
        "confirmation": confirmation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _plot(summary, args.figure)
    lines = [
        "# Fixed-A donor-to-repeated-option mediation",
        "",
        "Positive values mean that transplanting the first selected-option line moves the final answer toward the donor history's semantic answer.",
        "",
    ]
    for split in ("discovery", "confirmation"):
        lines += [f"## {split.title()}", ""]
        for condition in ("Game", "Neutral", "Game minus Neutral"):
            rows = summary[split]["metrics"][condition]
            lines.append(
                f"- {condition}: open transfer {_fmt(rows['open_transfer'])}; "
                f"matching-blocked transfer {_fmt(rows['matching_blocked_transfer'])}; "
                f"matching mediation {_fmt(rows['matching_mediation'])}; "
                f"matching-specific mediation {_fmt(rows['matching_specific_mediation'])}."
            )
        lines += ["", f"Validation: `{json.dumps(summary[split]['validation'], sort_keys=True)}`", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
