from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _fmt(row: dict[str, Any], scale: float = 1.0) -> str:
    return (
        f"{row['mean'] * scale:+.3f} "
        f"[{row['ci'][0] * scale:+.3f}, {row['ci'][1] * scale:+.3f}]"
    )


def _combine_letters(a: dict[str, Any], bcd: dict[str, Any], split: str) -> dict[str, Any]:
    rows = dict(bcd[split]["letters"])
    rows.update(a[split]["letters"])
    return {letter: rows[letter] for letter in "ABCD"}


def _errorbar(axis: Any, x: np.ndarray, rows: list[dict[str, Any]], offset: float,
              label: str, color: str) -> None:
    means = np.asarray([row["mean"] for row in rows])
    cis = np.asarray([row["ci"] for row in rows])
    axis.errorbar(
        x + offset,
        means,
        yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
        fmt="o-",
        capsize=3,
        lw=1.5,
        color=color,
        label=label,
    )


def plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"Game": "#2878b5", "Neutral": "#e07a2f"}
    for column, split in enumerate(("discovery", "confirmation")):
        axis = axes[0, column]
        letters = summary["same_pipeline"][split]
        x = np.arange(4)
        for offset, condition in ((-0.08, "Game"), (0.08, "Neutral")):
            _errorbar(
                axis,
                x,
                [letters[letter][f"{condition}_margin"] for letter in "ABCD"],
                offset,
                condition,
                colors[condition],
            )
        axis.axhline(0, color="#777", lw=1)
        axis.set_xticks(x, list("ABCD"))
        axis.set_title(f"{split.title()}: selected-line semantic transfer")
        axis.set_ylabel("Movement toward donor semantic answer (logits)")
        axis.grid(axis="y", alpha=0.18)
        axis.legend(frameon=False)

        axis = axes[1, column]
        x = np.arange(3)
        labels = ("Open", "Match blocked", "Control blocked")
        metrics = ("open_transfer", "matching_blocked_transfer", "control_blocked_transfer")
        for offset, condition in ((-0.08, "Game"), (0.08, "Neutral")):
            rows = summary["a_mediation"][split]["letters"]["A"][condition]
            _errorbar(
                axis,
                x,
                [rows[metric] for metric in metrics],
                offset,
                condition,
                colors[condition],
            )
        axis.axhline(0, color="#777", lw=1)
        axis.set_xticks(x, labels, rotation=12)
        axis.set_title(f"{split.title()}: fixed-A repeated-line mediation")
        axis.set_ylabel("Donor-semantic margin transfer (logits)")
        axis.grid(axis="y", alpha=0.18)
        axis.legend(frameon=False)

    fig.suptitle("Fixed A under the same 24-ordering pipeline as fixed B–D")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-transplant", type=Path, required=True)
    parser.add_argument("--bcd-transplant", type=Path, required=True)
    parser.add_argument("--a-mediation", type=Path, required=True)
    parser.add_argument("--old-a-mediation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    a = json.loads(args.a_transplant.read_text())
    bcd = json.loads(args.bcd_transplant.read_text())
    mediation = json.loads(args.a_mediation.read_text())
    old = json.loads(args.old_a_mediation.read_text())
    summary = {
        "same_pipeline": {
            split: _combine_letters(a, bcd, split)
            for split in ("discovery", "confirmation")
        },
        "a_mediation": {
            split: mediation[split] for split in ("discovery", "confirmation")
        },
        "old_two_mapping_a_context": {
            condition: old["confirmation"]["metrics"][condition]["open_transfer"]
            for condition in ("Game", "Neutral")
        },
        "validation": {
            "a_transplant_discovery": a["discovery"]["validation"],
            "a_transplant_confirmation": a["confirmation"]["validation"],
            "a_mediation_discovery": mediation["discovery"]["validation"],
            "a_mediation_confirmation": mediation["confirmation"]["validation"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    plot(summary, args.figure)

    held = summary["same_pipeline"]["confirmation"]["A"]
    med = summary["a_mediation"]["confirmation"]["letters"]["A"]
    lines = [
        "# Fixed-A calibration under the complete 24-ordering pipeline",
        "",
        "## Result",
        "",
        "The earlier fixed-A Game transfer does not replicate when A is put through the "
        "same 24-ordering cohort and causal pipeline as B-D. The selected A line transfers "
        "its donor semantic answer strongly in Neutral but not in Game.",
        "",
        f"- Held-out Game margin transfer: {_fmt(held['Game_margin'])}; donor choice "
        f"{_fmt(held['Game_donor_choice'], 100)} percentage points.",
        f"- Held-out Neutral margin transfer: {_fmt(held['Neutral_margin'])}; donor choice "
        f"{_fmt(held['Neutral_donor_choice'], 100)} percentage points.",
        f"- Held-out Game-minus-Neutral transfer: {_fmt(held['Game_minus_Neutral_margin'])}.",
        "",
        "This matches the B-D same-pipeline pattern. The apparent fixed-A Game exception "
        "was therefore contingent on the old two-mapping cohort/matching design, not a "
        "robust property of literal A.",
        "",
        "## Repeated-line mediation",
        "",
    ]
    for condition in ("Game", "Neutral"):
        row = med[condition]
        lines.append(
            f"- {condition}: open {_fmt(row['open_transfer'])}; matching blocked "
            f"{_fmt(row['matching_blocked_transfer'])}; control blocked "
            f"{_fmt(row['control_blocked_transfer'])}; matching-specific mediation "
            f"{_fmt(row['matching_specific_mediation'])}."
        )
    lines += [
        "",
        "A positive matching-specific mediation would require the matching blockade to "
        "reduce transfer more than the nonmatching control blockade. Interpret the values "
        "above using that criterion, not merely whether matching blockade changes transfer.",
        "",
        "## Relation to the old fixed-A cohort",
        "",
        f"The old held-out two-mapping estimate was {_fmt(summary['old_two_mapping_a_context']['Game'])} "
        f"in Game and {_fmt(summary['old_two_mapping_a_context']['Neutral'])} in Neutral. "
        "Those estimates are retained as historical results from a different cohort "
        "construction; they should not be pooled with this calibration.",
        "",
        "## Validation",
        "",
        f"`{json.dumps(summary['validation'], sort_keys=True)}`",
        "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
