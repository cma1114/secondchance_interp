from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_final_decision_positive_only import (
    _fmt_change,
    analyze_split,
)


def _negative_dose(results: Path, base: dict[str, Any]) -> dict[str, Any]:
    arrays = dict(np.load(results, allow_pickle=False))
    qids = arrays["question_ids"].astype(str).tolist()
    # The subset counts and order come from analyze_split. Reconstructing masks
    # here only requires W1/W2, which the caller stores alongside the results.
    return {
        condition: {
            "natural_negative_fraction_by_layer": (
                arrays["natural_projection"][ci] < 0
            ).mean(0).tolist(),
            "natural_negative_magnitude_by_layer": np.maximum(
                -arrays["natural_projection"][ci], 0
            ).mean(0).tolist(),
            "live_negative_fraction_by_layer": (
                arrays["ablated_pre_projection"][ci] < 0
            ).mean(0).tolist(),
            "live_negative_magnitude_removed_by_layer": np.maximum(
                -arrays["ablated_pre_projection"][ci], 0
            ).mean(0).tolist(),
            "n_questions": len(qids),
        }
        for ci, condition in enumerate(("game", "neutral"))
    }


def _rename(summary: dict[str, Any]) -> dict[str, Any]:
    for subset in summary["subsets"].values():
        for condition in subset["conditions"].values():
            condition["negative_only"] = condition.pop("positive_only")
    summary["negative_projection_dose_all_questions"] = _negative_dose(
        Path(summary["root"]) / "results.npz", summary
    )
    summary.pop("projection_dose", None)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--confirmation-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summary = {
        "definitions": {
            "negative_only": (
                "At every readout, if h dot v_W1 is negative, add back exactly "
                "enough v_W1 to make it zero; leave positive projections unchanged."
            ),
            "W1": "semantic answer selected in the original Baseline presentation",
            "W2": (
                "semantic answer selected by a fresh Baseline solution of the "
                "remapped second presentation"
            ),
            "conflict": "questions where W1 differs from W2",
            "agreement": "questions where W1 equals W2",
        },
        "discovery": _rename(analyze_split(
            args.discovery / "results.npz", args.baseline,
            args.remapped_baseline, args.manifest, args.mapping_plan,
            args.discovery_plan, seed=args.seed, draws=args.draws,
        )),
        "confirmation": _rename(analyze_split(
            args.confirmation / "results.npz", args.baseline,
            args.remapped_baseline, args.manifest, args.mapping_plan,
            args.confirmation_plan, seed=args.seed + 1000, draws=args.draws,
        )),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    lines = [
        "# Negative-only W1 semantic projection ablation",
        "",
        "This experiment removes only negative projection onto the layer-specific "
        "semantic vector for W1 at the final decision position. At every readout, "
        "negative `h · v_W1` is moved to zero; positive projection is untouched. "
        "The frozen primary subset is conflict trials (`W1 != W2`). Agreement "
        "trials (`W1 = W2`) are the prespecified control.",
        "",
    ]
    for split in ("discovery", "confirmation"):
        lines.extend([
            f"## {split.title()}",
            "",
            "| Subset | Condition | Natural W1 | Negative-only W1 | W1 change (95% CI) | Natural W2 | Negative-only W2 | Centered-W1 logit change (95% CI) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for subset in ("conflict", "agreement", "all"):
            row = summary[split]["subsets"][subset]
            for condition in ("game", "neutral"):
                values = row["conditions"][condition]
                natural = values["natural"]
                negative = values["negative_only"]
                lines.append(
                    f"| {subset.title()} (n={row['n']}) | {condition.title()} | "
                    f"{natural['w1_selection']:.1%} | "
                    f"{negative['w1_selection']:.1%} | "
                    f"{_fmt_change(values['w1_selection_change_ci'], 100)} pp | "
                    f"{natural.get('w2_selection', float('nan')):.1%} | "
                    f"{negative.get('w2_selection', float('nan')):.1%} | "
                    f"{_fmt_change(values['centered_w1_change_ci'])} |"
                )
        lines.extend([
            "",
            "The Game-minus-Neutral entry below is not the primary outcome; it "
            "only reports whether the two conditions respond differently. The "
            "within-Game change above is the direct test of whether removing "
            "negative W1 projection reduces or increases Game's W1 choices.",
            "",
            "| Subset | Game-minus-Neutral W1-selection change (95% CI) | Game-minus-Neutral centered-W1 change (95% CI) |",
            "|---|---:|---:|",
        ])
        for subset in ("conflict", "agreement", "all"):
            row = summary[split]["subsets"][subset]
            contrast = row["game_minus_neutral_change"]
            lines.append(
                f"| {subset.title()} (n={row['n']}) | "
                f"{_fmt_change(contrast['w1_selection_change'], 100)} pp | "
                f"{_fmt_change(contrast['centered_w1_change'])} |"
            )
        lines.append("")

    lines.extend([
        "## Interpretation rule",
        "",
        "On conflict trials, evidence for a causal `not-W1` signal would require "
        "negative-only removal to increase W1 selection in Game reliably, with "
        "a corresponding movement of W1 logits toward W2. A similar effect in "
        "Neutral would show that the representation is not Game-specific. Null "
        "or unstable effects across the frozen split rule out this particular "
        "one-dimensional negative-projection account; they do not rule out a "
        "distributed semantic representation.",
        "",
        "## Reproducibility",
        "",
        "The runner preserved the historical physical batch-of-four cohorts and "
        "SDPA kernels. Natural logits, projections, and residual norms were reused "
        "from the bit-exact 500-question positive-only companion. Semantic "
        "directions were reconstructed from the same four counterbalanced option "
        "mappings and cached per cohort as float32 arrays. `summary.json` contains "
        "the complete layerwise negative-projection dose and all reported confidence "
        "intervals.",
    ])
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
