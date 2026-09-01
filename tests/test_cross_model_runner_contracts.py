from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mechanistic.config import ExperimentConfig
from mechanistic import collect_seed_oss_nonremapped_rank_trajectories as trajectories
from mechanistic import run_seed_oss_matching_history_blockade as matching
from mechanistic import run_seed_oss_policy_recollection_factorial as policy


ROOT = Path(__file__).resolve().parents[1]
SEED_ROOT = ROOT / "outputs/model_replications/seed_oss_36b_mechanistic_replication"


@pytest.mark.parametrize("dataset", ["simplemc", "triviamc"])
def test_frozen_seed_policy_checkpoint_remains_resumable(dataset: str) -> None:
    path = SEED_ROOT / dataset / "policy_recollection_factorial/run/results.npz"
    with np.load(path, allow_pickle=False) as loaded:
        qids = loaded["question_ids"].astype(str).tolist()
        frozen_keys = set(loaded.files)
    assert "mixed_batch_natural_drift" not in frozen_keys

    arrays = policy._initialize(path, qids)

    assert arrays["completed"].all()
    assert "mixed_batch_natural_drift" not in arrays


def test_fresh_seed_policy_checkpoint_does_not_create_gemma_drift(
    tmp_path: Path,
) -> None:
    arrays = policy._initialize(tmp_path / "results.npz", ["q1", "q2", "q3", "q4"])
    assert "mixed_batch_natural_drift" not in arrays


def test_mixed_batch_binding_refuses_missing_or_nonfinite_drift() -> None:
    with pytest.raises(RuntimeError, match="must be present and finite"):
        policy._mixed_batch_natural_max_abs_drift({})
    with pytest.raises(RuntimeError, match="must be present and finite"):
        policy._mixed_batch_natural_max_abs_drift(
            {"mixed_batch_natural_drift": np.full((2, 1), np.nan)}
        )
    assert policy._mixed_batch_natural_max_abs_drift(
        {"mixed_batch_natural_drift": np.asarray([[1.0], [2.5]])}
    ) == 2.5


def _seed_config() -> ExperimentConfig:
    return ExperimentConfig.load(ROOT / "configs/seed_oss_36b_simplemc_clean_gate.json")


@pytest.mark.parametrize(
    "assertion",
    [
        trajectories._assert_binding_config,
        matching._assert_binding_config,
        policy._assert_binding_config,
    ],
)
def test_seed_bindings_require_exact_model_revision_and_batch(assertion) -> None:
    config = _seed_config()
    assertion(config)
    with pytest.raises(ValueError, match="pinned configured model revision"):
        assertion(dataclasses.replace(config, model_revision="wrong"))
    with pytest.raises(ValueError, match="canonical batch_size=4"):
        assertion(dataclasses.replace(config, batch_size=1))


def test_seed_trajectory_binding_requires_64_layers() -> None:
    trajectories._assert_layer_count(64)
    with pytest.raises(RuntimeError, match="Expected 64"):
        trajectories._assert_layer_count(60)


def test_seed_provenance_contract_matches_frozen_artifacts() -> None:
    assert trajectories.EXPERIMENT_MODEL_NAME == "Seed-OSS 36B"
    assert matching.EXPERIMENT_MODEL_NAME == "Seed-OSS 36B"
    assert policy.EXPERIMENT_MODEL_NAME == "Seed-OSS 36B"
    assert policy.FORWARDS_PER_COHORT_METADATA_KEY == (
        "complete_model_forwards_per_canonical_cohort"
    )
    assert not policy.TRACK_MIXED_BATCH_NATURAL_DRIFT
    frozen = {
        "trajectory": (
            ROOT
            / "outputs/model_replications/seed_oss_36b_final_position_trajectories/run/simplemc/run_metadata.json"
        ),
        "matching": SEED_ROOT / "simplemc/matching_history/run/run_metadata.json",
        "policy": (
            SEED_ROOT
            / "simplemc/policy_recollection_factorial/run/run_metadata.json"
        ),
    }
    import json

    assert json.loads(frozen["trajectory"].read_text())["experiment"] == (
        trajectories._experiment_name("SimpleMC")
    )
    assert json.loads(frozen["matching"].read_text())["experiment"] == (
        matching._experiment_name("SimpleMC")
    )
    policy_metadata = json.loads(frozen["policy"].read_text())
    assert policy_metadata["experiment"] == policy._experiment_name("SimpleMC")
    assert policy.FORWARDS_PER_COHORT_METADATA_KEY in policy_metadata
    assert "complete_model_forwards_per_configured_cohort" not in policy_metadata
    assert "mixed_batch_natural_max_abs_drift" not in policy_metadata


def test_gemma_bindings_install_their_own_fail_closed_contracts() -> None:
    code = """
from pathlib import Path
import numpy as np
from mechanistic import collect_gemma4_nonremapped_rank_trajectories
from mechanistic import collect_seed_oss_nonremapped_rank_trajectories as trajectories
from mechanistic import run_gemma4_matching_history_blockade
from mechanistic import run_gemma4_policy_recollection_factorial
from mechanistic import run_seed_oss_matching_history_blockade as matching
from mechanistic import run_seed_oss_policy_recollection_factorial as policy
from mechanistic.config import ExperimentConfig
assert trajectories.MODEL_ID == 'google/gemma-4-31B-it'
assert trajectories.CANONICAL_BATCH_SIZE == 1
assert trajectories.EXPECTED_LAYER_COUNT == 60
assert matching.MODEL_ID == 'google/gemma-4-31B-it'
assert matching.CANONICAL_BATCH_SIZE == 1
assert matching.ATTENTION_LAYERS_ONE_BASED == tuple(range(1, 61))
assert policy.MODEL_ID == 'google/gemma-4-31B-it'
assert policy.CANONICAL_BATCH_SIZE == 1
assert policy.FORWARDS_PER_COHORT_METADATA_KEY == 'complete_model_forwards_per_configured_cohort'
assert policy.TRACK_MIXED_BATCH_NATURAL_DRIFT
config = ExperimentConfig.load('configs/gemma4_31b_simplemc_clean_gate.json')
trajectories._assert_binding_config(config)
trajectories._assert_layer_count(60)
matching._assert_binding_config(config)
policy._assert_binding_config(config)
root = Path('outputs/model_replications/gemma4_31b_negative_model_comparison/simplemc')
with np.load(root / 'policy_recollection/run/results.npz', allow_pickle=False) as loaded:
    qids = loaded['question_ids'].astype(str).tolist()
arrays = policy._initialize(root / 'policy_recollection/run/results.npz', qids)
assert arrays['completed'].all()
assert np.isfinite(arrays['mixed_batch_natural_drift']).all()
assert policy._experiment_name('SimpleMC') == 'google/gemma-4-31B-it SimpleMC direct policy by recollection factorial'
assert matching._experiment_name('SimpleMC') == 'google/gemma-4-31B-it SimpleMC all-candidate matching-history blockade'
assert trajectories._experiment_name('SimpleMC') == 'google/gemma-4-31B-it SimpleMC non-remapped final-position trajectories'
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
