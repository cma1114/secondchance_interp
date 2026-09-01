"""Pinned Qwen3.6-27B binding for MCQ uncertainty-state collection."""

from . import collect_mcq_uncertainty_states as implementation


implementation.MODEL_ID = "Qwen/Qwen3.6-27B"
implementation.MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
implementation.EXPERIMENT_MODEL_NAME = "Qwen3.6-27B"
implementation.CANONICAL_BATCH_SIZE = 4
implementation.EXPECTED_LAYER_COUNT = 64
implementation.FIRST_DECISION_OPENER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
implementation.ALLOWED_SERIALIZATIONS = ("raw_qwen_chatml",)


if __name__ == "__main__":
    implementation.main()
