from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

from .config import config_arg_parser
from .io import atomic_save_npz, json_array, shard_path
from .modeling import cpu_lens, get_tokenizer, load_model_and_processor, model_input_device, render_chat, resolve_answer_tokens, tokenize_batch
from .prompts import build_messages, load_trials, prompt_hash
from .sublayer import SublayerBoundaryCollector, mixer_module
from .sublayer_config import SublayerExperimentConfig


def _chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def collect(config: SublayerExperimentConfig) -> None:
    import torch
    import transformers

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trials = load_trials(config.manifest_path, config.baseline_results_path, config.question_ids, config.max_questions)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    layer_kinds = ["attention" if getattr(layer, "self_attn", None) is not None else "deltanet" for layer in parts.layers]
    for layer in parts.layers:
        mixer_module(layer)
    metadata = {
        "config": config.as_dict(),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "n_layers": len(parts.layers),
        "layer_mixer_kinds": layer_kinds,
        "boundary_order": ["pre_mixer", "pre_mlp", "post_mlp"],
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    for condition in config.conditions:
        pending = [trial for trial in trials if not shard_path(output, condition, trial.question_id).exists()]
        print(f"{condition}: {len(pending)} pending / {len(trials)} total", flush=True)
        done = 0
        for batch in _chunks(pending, config.batch_size):
            messages = [build_messages(trial.question, condition, config.prompt_mode) for trial in batch]
            prompts = [render_chat(processor, value, config.disable_thinking) for value in messages]
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            collector = SublayerBoundaryCollector(parts, last_indices)
            try:
                with torch.inference_mode():
                    device = model_input_device(parts)
                    kwargs = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device), "use_cache": False, "return_dict": True}
                    try:
                        model_output = model(**kwargs, logits_to_keep=1)
                    except TypeError:
                        model_output = model(**kwargs)
                boundaries = collector.stacked()
            finally:
                collector.close()
            shape = boundaries.shape
            lens = cpu_lens(parts, boundaries.reshape(shape[0], shape[1] * shape[2], shape[3]), canonical_ids)
            lens = lens.reshape(shape[0], shape[1], shape[2], 4).numpy()
            final_logits = model_output.logits.detach().float().cpu()
            actual = final_logits[:, 0 if final_logits.shape[1] == 1 else -1, canonical_ids].numpy()
            error = np.max(np.abs(lens[:, -1, -1] - actual), axis=-1)
            if np.any(error > 0.12):
                raise RuntimeError(f"Final sublayer lens mismatch: {error.tolist()}")
            for index, trial in enumerate(batch):
                meta = {
                    "question_id": trial.question_id,
                    "condition": condition,
                    "baseline_answer": trial.baseline_answer,
                    "baseline_correct": trial.baseline_correct,
                    "correct_answer": trial.question["correct_answer"],
                    "prompt_hash": prompt_hash(prompts[index]),
                    "canonical_ad_choice": "ABCD"[int(actual[index].argmax())],
                    "final_lens_max_abs_error": float(error[index]),
                }
                atomic_save_npz(
                    shard_path(output, condition, trial.question_id),
                    boundary_canonical_logits=lens[index].astype(np.float32),
                    final_canonical_logits=actual[index].astype(np.float32),
                    metadata=json_array(meta),
                )
            done += len(batch)
            print(f"{condition}: saved {done}/{len(pending)} pending trials", flush=True)


def main() -> None:
    args = config_arg_parser("Collect mixer/MLP residual-boundary A-D lenses").parse_args()
    collect(SublayerExperimentConfig.load(args.config))


if __name__ == "__main__":
    main()
