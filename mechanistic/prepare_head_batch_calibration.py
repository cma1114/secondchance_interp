from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prompts import load_trials
from .sublayer_config import SublayerExperimentConfig


def prepare(
    config: SublayerExperimentConfig,
    discovery_plan_path: Path,
    patch_root: Path,
    output: Path,
    unbatched_count: int,
) -> dict:
    discovery = json.loads(discovery_plan_path.read_text())
    scenario_ids = [row["id"] for row in discovery["scenarios"]]
    completion_group = patch_root / "shards" / scenario_ids[-1]
    completion = sorted(
        (path.stat().st_mtime, path.stem) for path in completion_group.glob("*.npz")
    )
    if len(completion) != len(discovery["question_ids"]):
        raise RuntimeError("Discovery head sweep is incomplete")
    unbatched_qids = {qid for _mtime, qid in completion[:unbatched_count]}
    grid_start = (patch_root / "run_metadata.json").stat().st_mtime

    batch24_by_qid: dict[str, list[str]] = {}
    grid_by_qid: dict[str, list[str]] = {}
    for qid in discovery["question_ids"]:
        if qid in unbatched_qids:
            continue
        for scenario in scenario_ids:
            path = patch_root / "shards" / scenario / f"{qid}.npz"
            destination = grid_by_qid if path.stat().st_mtime >= grid_start else batch24_by_qid
            destination.setdefault(qid, []).append(scenario)

    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        discovery["question_ids"],
        None,
    )
    grid_qids = [trial.question_id for trial in trials if trial.question_id in grid_by_qid]
    grid_batches = [grid_qids[start:start + 2] for start in range(0, len(grid_qids), 2)]
    plan = {
        "stage": "head_discovery_batch_calibration",
        "question_ids": discovery["question_ids"],
        "unbatched_qids": sorted(unbatched_qids),
        "batch24_scenarios_by_qid": batch24_by_qid,
        "grid_scenarios_by_qid": grid_by_qid,
        "grid_batches": grid_batches,
        "scenario_ids": scenario_ids,
        "grid_start_mtime": grid_start,
        "unbatched_count": unbatched_count,
        "discovery_plan": str(discovery_plan_path),
        "patch_root": str(patch_root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover batch provenance for head discovery")
    parser.add_argument("--config", required=True)
    parser.add_argument("--discovery-plan", required=True)
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unbatched-count", type=int, default=13)
    args = parser.parse_args()
    prepare(
        SublayerExperimentConfig.load(args.config),
        Path(args.discovery_plan),
        Path(args.patch_root),
        Path(args.output),
        args.unbatched_count,
    )


if __name__ == "__main__":
    main()
