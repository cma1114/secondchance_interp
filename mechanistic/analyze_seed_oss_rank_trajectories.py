from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_nonremapped_rank_trajectories import SUBSETS, _analyze_dataset


def analyze(specs_path: Path, output: Path, figure_dir: Path, draws: int, seed: int) -> None:
    specs = json.loads(specs_path.read_text())["datasets"]
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        _analyze_dataset(spec, figure_dir, draws, seed + 1000 * index)
        for index, spec in enumerate(specs)
    ]
    summary = {
        "analysis": "seed_oss_36b_nonremapped_final_decision_rank_trajectories",
        "model": "ByteDance-Seed/Seed-OSS-36B-Instruct",
        "readout": (
            "Standard logit lens using Seed's exact final RMS norm and A-D unembedding rows. "
            "No compatible Seed Jacobian lens exists."
        ),
        "evidence_class": (
            "activation/logit-lens decoding; outcome-conditioned panels are descriptive postselection"
        ),
        "bootstrap_draws": draws,
        "datasets": datasets,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Seed-OSS 36B final-decision trajectories by first-presentation rank",
        "",
        "This report measures every post-block residual L1--L64 at the final prompt token immediately before Seed's second answer. Game and Neutral prompts are non-remapped and differ only at the single `incorrect`/`lost` token. R1--R4 are frozen from Seed's same-format first-presentation aggregated A--D logits.",
        "",
        "No published or local Jacobian lens exists for this Seed revision. The figures therefore use the standard Seed logit lens: each post-block residual passes through Seed's exact final RMS norm and A--D unembedding rows. The separate held-out prospective-decoder analysis tests for answer information that is linearly present before this fixed output readout can expose it.",
        "",
        "All-question panels are primary activation descriptions. Switch/no-switch panels are selected by the model's eventual answer and are descriptive, not causal evidence about why it switched. Confidence bands use a first-presentation-winner-letter-stratified question bootstrap. Background tint is the mean per-question cosine similarity between the layer's complete centered A--D pattern and the exact final pattern.",
        "",
    ]
    for dataset in datasets:
        lines.extend([f"## {dataset['dataset']}", ""])
        for subset in SUBSETS:
            row = dataset["subsets"][subset]
            lines.append(
                f"- **{subset}:** Game n={row['game']['n']}, Neutral n={row['neutral']['n']}; "
                f"[figure](../../../../{row['figure']})"
            )
        lines.append(
            f"- **Companions:** [raw scores](../../../../{dataset['companion_figures']['noncentered']}) · "
            f"[displayed-letter-controlled](../../../../{dataset['companion_figures']['displayed_letter_controlled']})"
        )
        lines.append("")
        game_all = dataset["subsets"]["all"]["game"]["selected_layers"]["64"]
        neutral_all = dataset["subsets"]["all"]["neutral"]["selected_layers"]["64"]
        game_switch = dataset["subsets"]["switch"]["game"]["selected_layers"]["64"]
        neutral_switch = dataset["subsets"]["switch"]["neutral"]["selected_layers"]["64"]
        game_stay = dataset["subsets"]["stay"]["game"]["selected_layers"]["64"]
        neutral_stay = dataset["subsets"]["stay"]["neutral"]["selected_layers"]["64"]
        difference = dataset["game_minus_neutral_all"]["selected_layers"]["64"]
        lines.extend(
            [
                f"At L64 on all questions, centered R1 is {game_all[0]:.3f} in Game and {neutral_all[0]:.3f} in Neutral; the paired Game-minus-Neutral difference is {difference['mean'][0]:.3f} `[{difference['ci_low'][0]:.3f}, {difference['ci_high'][0]:.3f}]`.",
                "",
                f"On eventual-switch trials, L64 R1/R2 is {game_switch[0]:.3f}/{game_switch[1]:.3f} in Game and {neutral_switch[0]:.3f}/{neutral_switch[1]:.3f} in Neutral. On no-switch trials, L64 R1 is {game_stay[0]:.3f} in Game and {neutral_stay[0]:.3f} in Neutral.",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            "The standard logit lens measures when a candidate ordering is directly readable by Seed's own output norm and A--D rows. A flat early lens trajectory does not establish absence from the residual stream; that question belongs to the held-out prospective decoders. Displayed-letter-controlled figures subtract each condition/layer's across-question mean for displayed A, B, C, and D before aligning candidates by 1P rank. Raw-score companions retain the common A--D offset, which mixes generic answer-token readiness with layer-dependent readout scale.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/model_replications"))
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    analyze(args.specs, args.output, args.figure_dir, args.draws, args.seed)


if __name__ == "__main__":
    main()
