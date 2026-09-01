from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .collect_action_matched_period_jlens import _build_readouts, _write_json
from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-rank-results", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=24)
    args = parser.parse_args()
    with np.load(args.run_dir / "results.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"]):
        raise ValueError("Cannot finalize an incomplete collection")
    config = ExperimentConfig.load(args.config)
    _model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    _build_readouts(
        arrays,
        tokenizer,
        parts,
        args.run_dir / "top_tokens_with_baseline_ranks.json",
        args.baseline_rank_results,
        args.remapping_plan,
        args.top_k,
    )
    metadata_path = args.run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update({
        "complete": True,
        "completed_questions": int(arrays["completed"].sum()),
        "top_token_summary_complete": True,
        "instrumentation_note": (
            "Complete residual readout itself changes low-order bfloat16/SDPA numerics: "
            "instrumented A-D argmax agrees with the trusted uninstrumented run on 94.4% "
            "of Evaluation and 96.6% of Matched Neutral questions. Prompt hashes match exactly. "
            "A no-hook output_hidden_states control produced the same changed logits, showing "
            "that the effect is inherent to requesting intermediate states rather than JLens preloading."
        ),
    })
    _write_json(metadata_path, metadata)


if __name__ == "__main__":
    main()
