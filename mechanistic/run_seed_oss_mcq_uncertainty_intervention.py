"""Pinned Seed-OSS 36B binding for the 2P uncertainty intervention."""

from . import run_mcq_uncertainty_intervention as implementation


implementation.MODEL_ID = "ByteDance-Seed/Seed-OSS-36B-Instruct"
implementation.MODEL_REVISION = "497f1dca95ebdec98e41d517b9f060ee753c902f"
implementation.EXPERIMENT_MODEL_NAME = "Seed-OSS 36B"
implementation.CANONICAL_BATCH_SIZE = 4
implementation.EXPECTED_LAYER_COUNT = 64
implementation.ALLOWED_SERIALIZATIONS = ("hf_template",)


if __name__ == "__main__":
    implementation.main()
