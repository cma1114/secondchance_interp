from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_policy_adjusted_prospective_decoding import _analyze_dataset as _analyze_policy
from .analyze_prospective_answer_results import _analyze_dataset as _analyze_prospective


def _layer_row(record: dict, layer: int) -> dict:
    return record["performance"]["selected_layers"][str(layer)]


def analyze(specs_path: Path, output: Path, figure_dir: Path, draws: int, seed: int) -> None:
    specs = json.loads(specs_path.read_text())["datasets"]
    output.mkdir(parents=True, exist_ok=True)
    prospective = [
        _analyze_prospective(spec, output, figure_dir, draws, seed + 1000 * index)
        for index, spec in enumerate(specs)
    ]
    policy = [
        _analyze_policy(spec, output, figure_dir, draws, seed + 5000 + 1000 * index)
        for index, spec in enumerate(specs)
    ]
    summary = {
        "analysis": "seed_oss_36b_prospective_and_policy_adjusted_final_answer_decoding",
        "model": "ByteDance-Seed/Seed-OSS-36B-Instruct",
        "evidence_class": "held-out activation/linear decoding; not a causal intervention",
        "decoder_fitting": "discovery questions only; all reported curves use frozen confirmation questions",
        "fixed_readout": "standard Seed logit lens",
        "prospective": prospective,
        "policy_adjusted": policy,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Seed-OSS 36B prospective final-answer decoding",
        "",
        "At every final-decision post-block residual L1--L64, ridge decoders predict the exact eventual centered A--D score vector. Shared Game+Neutral, Game-only, and Neutral-only bases are fit and tuned using discovery questions only and evaluated on the frozen confirmation questions. Cross-condition evaluation asks whether Game and Neutral encode the prospective answer in a shared linear basis. A W1-matched question shuffle preserves the easiest displayed old-winner structure while destroying question-specific final geometry.",
        "",
        "The fixed comparison is Seed's standard logit lens, because no compatible Seed Jacobian lens exists. Earlier held-out decoding than the logit lens means that the eventual answer pattern is linearly accessible before it is directly output-readable; it does not by itself establish causal use.",
        "",
    ]
    for record in prospective:
        label = record["dataset"]
        lines.extend([f"## {label}", ""])
        lines.append(
            f"Discovery n={record['n_discovery']}; confirmation n={record['n_confirmation']}. "
            f"[figure](../../../../{record['figure']})"
        )
        lines.extend(["", "Selected held-out mean cosine similarities:", ""])
        labels = record["performance"]["labels"]
        for layer in (24, 32, 40, 44, 48, 52, 56, 60, 64):
            row = _layer_row(record, layer)
            lines.append(
                f"- L{layer}: " + " / ".join(f"{name} {row[name]:.3f}" for name in labels)
            )
        lines.append("")
        game = record["switch_trials"]["game"]
        neutral = record["switch_trials"]["neutral"]
        lines.append(
            "On held-out eventual-switch trials, the first three-layer-sustained positive "
            f"shared-decoder R2-minus-R1 interval begins at {game['first_sustained_positive_ci_layer']} "
            f"in Game and {neutral['first_sustained_positive_ci_layer']} in Neutral."
        )
        lines.append("")

    lines.extend(
        [
            "## All-question policy-adjusted timing",
            "",
            "For every paired confirmation question, the shared decoder is applied to both conditions and the decoded Neutral vector is subtracted from Game. This measures when the question-specific final Game-versus-Neutral answer adjustment is linearly available without selecting on eventual switching.",
            "",
        ]
    )
    for record in policy:
        onset = record["similarity"]["first_persistent_positive_decoder_ci_layer"]
        l40 = record["similarity"]["selected_layers"]["40"]
        exact = record["rank_effect"]["exact_final_mean"]
        lines.extend(
            [
                f"### {record['dataset']}",
                "",
                f"The held-out policy-pattern cosine becomes persistently positive at {onset}. At L40 the learned cosine is {l40['learned']:.3f} `[{l40['learned_ci_low']:.3f},{l40['learned_ci_high']:.3f}]`, versus standard-logit-lens {l40['jlens']:.3f}. The exact final Game-minus-Neutral R1--R4 effects are " + "/".join(f"{value:+.3f}" for value in exact) + f". [figure](../../../../{record['figure']})",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            "These are held-out linear decoding results at the exact final decision position. They establish timing and cross-condition representational accessibility, not causal mediation. Switch-conditioned margins are descriptive postselection. The completed all-layer matching-history blockade remains the causal evidence that semantic 1P recollection affects Seed's preferential Game avoidance of the old winner.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/prospective_decoding"))
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    analyze(args.specs, args.output, args.figure_dir, args.draws, args.seed)


if __name__ == "__main__":
    main()
