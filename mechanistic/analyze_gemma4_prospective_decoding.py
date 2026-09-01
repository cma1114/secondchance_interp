from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .analyze_policy_adjusted_prospective_decoding import (
    _analyze_dataset as analyze_policy_dataset,
)
from .analyze_prospective_answer_results import (
    _analyze_dataset as analyze_answer_dataset,
)


def _selected_table(dataset: dict[str, Any]) -> list[str]:
    rows = [
        "| Layer | Shared decoder | Matched task | Cross-task | Fixed lens |",
        "|---:|---:|---:|---:|---:|",
    ]
    selected = dataset["performance"]["selected_layers"]
    for layer in (32, 40, 48, 56, 60):
        if str(layer) not in selected:
            continue
        row = selected[str(layer)]
        rows.append(
            f"| L{layer} | {row['Shared']:.3f} | {row['Matched task']:.3f} | "
            f"{row['Cross-task']:.3f} | {row['Standard logit lens']:.3f} |"
        )
    return rows


def analyze(
    specs_path: Path,
    output: Path,
    figure_dir: Path,
    draws: int,
    seed: int,
) -> None:
    specs = json.loads(specs_path.read_text())["datasets"]
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    answer = [
        analyze_answer_dataset(spec, output, figure_dir, draws, seed + 1000 * index)
        for index, spec in enumerate(specs)
    ]
    policy = [
        analyze_policy_dataset(spec, output, figure_dir, draws, seed + 10_000 + 1000 * index)
        for index, spec in enumerate(specs)
    ]
    summary = {
        "analysis": "gemma4_31b_prospective_and_policy_adjusted_decoding",
        "model": "google/gemma-4-31B-it",
        "evidence_class": "held-out linear activation decoding; noncausal",
        "answer_pattern": answer,
        "policy_adjusted_pattern": policy,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Gemma 4 31B prospective final-answer decoding",
        "",
        "Frozen discovery questions fit one shared Game+Neutral decoder and separate Game-only and Neutral-only decoders at every final-position post-block residual L1--L60. Frozen confirmation questions evaluate all bases. Targets are the exact eventual centered A--D logits; displayed-letter means are estimated and removed using discovery only.",
        "",
        "A second paired analysis decodes the same-question Game-minus-Neutral change in the eventual four-answer score vector. These are activation/decoding results: they establish linear availability and basis transfer, not causal use.",
        "",
    ]
    for answer_row, policy_row in zip(answer, policy):
        lines.extend(
            [
                f"## {answer_row['dataset']}",
                "",
                *_selected_table(answer_row),
                "",
                f"Stable positive held-out policy-pattern decoding begins at L{policy_row['similarity']['first_persistent_positive_decoder_ci_layer']}. Exact final Game-minus-Neutral centered rank effects (R1--R4) are "
                + ", ".join(f"{value:+.3f}" for value in policy_row["rank_effect"]["exact_final_mean"])
                + ".",
                "",
                f"Prospective-answer figure: `{answer_row['figure']}`. Policy-adjusted figure: `{policy_row['figure']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            "Switch/no-switch slices use the eventual answer and are descriptive postselection. Earlier linear decodability than the fixed lens means that answer information is present in another linear basis; it does not prove that the model reads the fitted direction or rule out still-earlier nonlinear information.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("figures/model_replications")
    )
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    analyze(args.specs, args.output, args.figure_dir, args.draws, args.seed)


if __name__ == "__main__":
    main()
