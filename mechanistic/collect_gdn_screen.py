from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

from .attention_spans import attention_span_indices
from .config import config_arg_parser
from .gdn_config import GDNExperimentConfig
from .gdn_intervention import linear_attention_layers
from .gdn_screen import DeltaNetScreenCollector
from .gdn_tokens import source_positions
from .io import atomic_save_npz, json_array, shard_path
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages, load_trials, prompt_hash


def collect(config: GDNExperimentConfig) -> None:
    import torch
    import transformers

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = load_trials(
        config.manifest_path, config.baseline_results_path, config.question_ids, config.max_questions
    )
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    layers = linear_attention_layers(parts)
    metadata = {
        "config": config.as_dict(),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "linear_attention_layers": layers,
        "screen_sources": config.screen_sources,
        "method": "exact within-module recurrence replay with source beta set to zero",
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "platform": platform.platform(),
    }
    (output_dir / "screen_run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    pending = [trial for trial in trials if not shard_path(output_dir, "gdn_screen", trial.question_id).exists()]
    print(f"gdn_screen: {len(pending)} pending / {len(trials)} total", flush=True)
    for completed, trial in enumerate(pending, 1):
        messages = build_messages(trial.question, "incorrect", config.prompt_mode)
        prompt = render_chat(processor, messages, config.disable_thinking)
        annotated_ids, spans = attention_span_indices(tokenizer, prompt, "incorrect", trial.question)
        input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, [prompt])
        positions = {
            source: source_positions(
                source, tokenizer, annotated_ids, spans, trial.question_id,
                config.structural_controls, config.seed,
            )
            for source in config.screen_sources
        }
        device = model_input_device(parts)
        collector = DeltaNetScreenCollector(parts, positions, canonical_ids)
        try:
            with torch.inference_mode():
                kwargs = {
                    "input_ids": input_ids.to(device),
                    "attention_mask": attention_mask.to(device),
                    "use_cache": False,
                    "return_dict": True,
                }
                try:
                    output = model(**kwargs, logits_to_keep=1)
                except TypeError:
                    output = model(**kwargs)
            arrays = collector.arrays()
        finally:
            collector.close()
        final_logits = output.logits.detach().float().cpu()[0, -1, canonical_ids].numpy()
        trial_meta = {
            "question_id": trial.question_id,
            "source_positions": positions,
            "source_tokens": {
                source: tokenizer.convert_ids_to_tokens([annotated_ids[position] for position in values])
                for source, values in positions.items()
            },
            "query_position": last_indices[0],
            "baseline_answer": trial.baseline_answer,
            "baseline_correct": trial.baseline_correct,
            "correct_answer": trial.question["correct_answer"],
            "prompt_hash": prompt_hash(prompt),
        }
        atomic_save_npz(
            shard_path(output_dir, "gdn_screen", trial.question_id),
            **arrays,
            final_canonical_logits=final_logits.astype(np.float32),
            metadata=json_array(trial_meta),
        )
        del output, collector
        if completed == 1 or completed % 10 == 0 or completed == len(pending):
            print(f"gdn_screen: saved {completed}/{len(pending)}", flush=True)


def main() -> None:
    args = config_arg_parser("Screen all Qwen Gated DeltaNet layer/head writes").parse_args()
    collect(GDNExperimentConfig.load(args.config))


if __name__ == "__main__":
    main()

