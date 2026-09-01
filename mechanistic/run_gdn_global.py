from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

from .attention_spans import attention_span_indices
from .config import config_arg_parser
from .gdn_config import GDNExperimentConfig
from .gdn_intervention import BetaWriteAblator, GDNTarget, linear_attention_layers
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


def run(config: GDNExperimentConfig) -> None:
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
    if len(layers) != 48:
        raise RuntimeError(f"Expected 48 Gated DeltaNet layers; found {len(layers)}")
    scenarios = ["user_incorrect"] + [
        f"structural_{index}" for index in range(config.structural_controls)
    ]
    metadata = {
        "config": config.as_dict(),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "linear_attention_layers": layers,
        "intervention": "set beta=0 for the selected token in every value head of all 48 Gated DeltaNet layers",
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    targets = [GDNTarget(layer) for layer in layers]

    for scenario in scenarios:
        scenario_id = f"gdn_all48__{scenario}"
        pending = [trial for trial in trials if not shard_path(output_dir, scenario_id, trial.question_id).exists()]
        print(f"{scenario_id}: {len(pending)} pending / {len(trials)} total", flush=True)
        for completed, trial in enumerate(pending, 1):
            messages = build_messages(trial.question, "incorrect", config.prompt_mode)
            prompt = render_chat(processor, messages, config.disable_thinking)
            annotated_ids, spans = attention_span_indices(tokenizer, prompt, "incorrect", trial.question)
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, [prompt])
            if annotated_ids != input_ids[0, : len(annotated_ids)].tolist():
                raise RuntimeError("Offset-aware and model tokenizations disagree")
            positions = source_positions(
                scenario, tokenizer, annotated_ids, spans, trial.question_id,
                config.structural_controls, config.seed,
            )
            device = model_input_device(parts)
            with BetaWriteAblator(parts, targets, positions), torch.inference_mode():
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
            final_logits = output.logits.detach().float().cpu()[0, -1, canonical_ids].numpy()
            trial_meta = {
                "question_id": trial.question_id,
                "scenario_id": scenario_id,
                "source_positions": positions,
                "source_tokens": tokenizer.convert_ids_to_tokens([annotated_ids[position] for position in positions]),
                "query_position": last_indices[0],
                "baseline_answer": trial.baseline_answer,
                "baseline_correct": trial.baseline_correct,
                "correct_answer": trial.question["correct_answer"],
                "prompt_hash": prompt_hash(prompt),
            }
            atomic_save_npz(
                shard_path(output_dir, scenario_id, trial.question_id),
                final_canonical_logits=final_logits.astype(np.float32),
                metadata=json_array(trial_meta),
            )
            del output
            if completed == 1 or completed % 25 == 0 or completed == len(pending):
                print(f"{scenario_id}: saved {completed}/{len(pending)}", flush=True)


def main() -> None:
    args = config_arg_parser("Ablate Gated DeltaNet writes across all layers").parse_args()
    run(GDNExperimentConfig.load(args.config))


if __name__ == "__main__":
    main()

