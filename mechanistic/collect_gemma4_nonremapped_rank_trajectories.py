"""Gemma-facing entry point for the generic complete-layer trajectory collector."""

from . import collect_seed_oss_nonremapped_rank_trajectories as implementation


implementation.MODEL_ID = "google/gemma-4-31B-it"
implementation.MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
implementation.EXPERIMENT_MODEL_NAME = implementation.MODEL_ID
implementation.CANONICAL_BATCH_SIZE = 1
implementation.EXPECTED_LAYER_COUNT = 60


if __name__ == "__main__":
    implementation.main()
