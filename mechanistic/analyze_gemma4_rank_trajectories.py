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
        "analysis": "gemma4_31b_nonremapped_final_decision_rank_trajectories",
        "model": "google/gemma-4-31B-it",
        "readout": "Standard logit lens through Gemma 4's exact final norm, softcap, and A-D unembedding rows.",
        "evidence_class": "activation/logit-lens decoding; switch panels are descriptive postselection",
        "bootstrap_draws": draws,
        "datasets": datasets,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Gemma 4 31B final-decision trajectories",
        "",
        "Every post-block residual L1--L60 is measured at the final prompt token immediately before the second answer. Game and Neutral use the clean non-remapped prompt and differ only at `incorrect`/`lost`. R1--R4 are frozen from Gemma's same-format first-presentation A--D scores.",
        "",
        "The standard logit lens passes each residual through Gemma's native final norm, final-logit softcap, and answer-token rows. All-question panels are primary activation descriptions; switch/no-switch panels are outcome-conditioned descriptions, not causal tests.",
        "",
    ]
    for dataset in datasets:
        lines.extend([f"## {dataset['dataset']}", ""])
        for subset in SUBSETS:
            row = dataset["subsets"][subset]
            lines.append(
                f"- **{subset}:** Game n={row['game']['n']}, Neutral n={row['neutral']['n']}; [figure](../../../../{row['figure']})"
            )
        lines.append(
            f"- **Companions:** [raw](../../../../{dataset['companion_figures']['noncentered']}) · [displayed-letter-controlled](../../../../{dataset['companion_figures']['displayed_letter_controlled']})"
        )
        lines.append("")
    lines.extend([
        "## Scope",
        "",
        "This is activation/decoding evidence. A flat fixed-lens trajectory does not establish absence of information; the held-out prospective decoder analysis separately measures linearly accessible eventual-answer and Game-minus-Neutral information.",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/model_replications"))
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    analyze(args.specs, args.output, args.figure_dir, args.draws, args.seed)


if __name__ == "__main__":
    main()
