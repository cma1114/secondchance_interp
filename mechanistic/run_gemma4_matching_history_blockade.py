"""Gemma 4 31B binding for the validated matching-history runner."""

from . import run_seed_oss_matching_history_blockade as implementation


implementation.MODEL_ID = "google/gemma-4-31B-it"
implementation.MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
implementation.EXPERIMENT_MODEL_NAME = implementation.MODEL_ID
implementation.CANONICAL_BATCH_SIZE = 1
implementation.ATTENTION_LAYERS_ONE_BASED = tuple(range(1, 61))


if __name__ == "__main__":
    implementation.main()
