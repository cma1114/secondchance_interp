from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .hypothesis_analysis import analyze as analyze_hypotheses
from .probes import run as run_probes
from .trajectory_analysis import analyze as analyze_trajectories


def main() -> None:
    p = argparse.ArgumentParser(description="Run all observational analyses from an experiment config")
    p.add_argument("--config", required=True)
    p.add_argument("--output")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--run-probes", action="store_true")
    args = p.parse_args()
    cfg = ExperimentConfig.load(args.config)
    output = args.output or str(Path(cfg.output_dir) / "analysis")
    result = {
        "trajectories": analyze_trajectories(cfg.output_dir, output, args.bootstrap, cfg.seed),
        "hypotheses": analyze_hypotheses(cfg.output_dir, output, args.folds, cfg.seed),
    }
    if args.run_probes:
        run_probes(cfg.output_dir, output, "centroid", 1, args.folds, cfg.seed, .01)
        run_probes(cfg.output_dir, output, "logistic", 1, args.folds, cfg.seed, .01)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

