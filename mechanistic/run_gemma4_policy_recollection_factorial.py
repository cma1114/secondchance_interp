"""Gemma 4 31B binding for the direct policy-by-recollection factorial."""

from . import run_seed_oss_matching_history_blockade as matching
from . import run_seed_oss_policy_recollection_factorial as implementation


MODEL_ID = "google/gemma-4-31B-it"
MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
LAYERS = tuple(range(1, 61))

matching.MODEL_ID = MODEL_ID
matching.MODEL_REVISION = MODEL_REVISION
matching.EXPERIMENT_MODEL_NAME = MODEL_ID
matching.CANONICAL_BATCH_SIZE = 1
matching.ATTENTION_LAYERS_ONE_BASED = LAYERS
implementation.MODEL_ID = MODEL_ID
implementation.MODEL_REVISION = MODEL_REVISION
implementation.EXPERIMENT_MODEL_NAME = MODEL_ID
implementation.CANONICAL_BATCH_SIZE = 1
implementation.ATTENTION_LAYERS_ONE_BASED = LAYERS
implementation.FORWARDS_PER_COHORT_METADATA_KEY = (
    "complete_model_forwards_per_configured_cohort"
)
implementation.TRACK_MIXED_BATCH_NATURAL_DRIFT = True


if __name__ == "__main__":
    implementation.run(implementation.build_parser().parse_args())
