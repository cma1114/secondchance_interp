"""Pinned Gemma 4 31B binding for the canonical history/decision factorial."""

from . import run_canonical_history_decision_factorial as implementation


implementation.MODEL_ID = "google/gemma-4-31B-it"
implementation.MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
implementation.EXPERIMENT_MODEL_NAME = "Gemma 4 31B"
implementation.CANONICAL_BATCH_SIZE = 1
implementation.ATTENTION_LAYERS_ONE_BASED = tuple(range(1, 61))
implementation.EXPECTED_GLA_LAYERS = 0
implementation.FIRST_DECISION_OPENER = "<|turn>model\n"
implementation.ALLOWED_SERIALIZATIONS = ("hf_template", "hf_template_direct_assistant")


if __name__ == "__main__":
    implementation.main()
