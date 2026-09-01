"""Gemma 4 31B binding for the validated feedback-suffix crossover."""

from . import run_seed_oss_feedback_suffix_crossover as implementation
from . import run_seed_oss_matching_history_blockade as matching


MODEL_ID = "google/gemma-4-31B-it"
MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
LAYERS = tuple(range(1, 61))

matching.MODEL_ID = MODEL_ID
matching.MODEL_REVISION = MODEL_REVISION
matching.ATTENTION_LAYERS_ONE_BASED = LAYERS
implementation.MODEL_ID = MODEL_ID
implementation.MODEL_REVISION = MODEL_REVISION
implementation.ATTENTION_LAYERS_ONE_BASED = LAYERS


if __name__ == "__main__":
    implementation.run(implementation.build_parser().parse_args())
