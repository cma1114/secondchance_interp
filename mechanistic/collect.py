from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

from .config import ExperimentConfig, config_arg_parser
from .io import atomic_save_npz, json_array, shard_path
from .modeling import (
    ResidualCollector,
    cpu_lens,
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
    variant_layout,
)
from .prompts import build_messages, load_trials, prompt_hash


FINAL_LENS_ABS_TOL = 0.12
FINAL_LENS_REL_TOL = 0.005


def _chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _scheduled_batches(config: ExperimentConfig, trials: list, output_dir: Path):
    """Yield resumable work in condition-major or matched trial-major order."""
    if config.trial_major:
        for trial_batch in _chunks(trials, config.batch_size):
            for condition in config.conditions:
                pending = [
                    trial
                    for trial in trial_batch
                    if not shard_path(output_dir, condition, trial.question_id).exists()
                ]
                if pending:
                    yield condition, pending
        return
    for condition in config.conditions:
        pending = [
            trial
            for trial in trials
            if not shard_path(output_dir, condition, trial.question_id).exists()
        ]
        for batch in _chunks(pending, config.batch_size):
            yield condition, batch


def collect(config: ExperimentConfig) -> None:
    import torch
    import transformers

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        config.question_ids,
        config.max_questions,
        config.skip_missing_baseline,
    )
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    selected_ids, variant_meta = variant_layout(resolved)
    canonical_positions = [next(i for i, x in enumerate(variant_meta) if x["letter"] == c and x["text"] == c) for c in "ABCD"]

    run_meta = {
        "config": config.as_dict(),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "resolved_answer_tokens": resolved,
        "variant_layout": variant_meta,
        "n_text_layers": len(parts.layers),
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True))

    pending_counts = {
        condition: sum(
            not shard_path(output_dir, condition, trial.question_id).exists()
            for trial in trials
        )
        for condition in config.conditions
    }
    saved_counts = {condition: 0 for condition in config.conditions}
    for condition, count in pending_counts.items():
        print(f"{condition}: {count} pending / {len(trials)} total", flush=True)

    for condition, batch in _scheduled_batches(config, trials, output_dir):
            messages = [build_messages(t.question, condition, config.prompt_mode) for t in batch]
            prompts = [
                render_chat(
                    processor,
                    m,
                    config.disable_thinking,
                    config.chat_serialization,
                )
                for m in messages
            ]
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            collector = ResidualCollector(parts, last_indices)
            try:
                with torch.inference_mode():
                    device = model_input_device(parts)
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
                residuals = collector.stacked()
            finally:
                collector.close()
            all_variant_logits = cpu_lens(parts, residuals, selected_ids).numpy()
            canonical_logits = all_variant_logits[:, :, canonical_positions]
            final_model_logits = output.logits.detach().float().cpu()
            if final_model_logits.shape[1] == 1:
                actual_selected = final_model_logits[:, 0, selected_ids].numpy()
                actual_vocab = final_model_logits[:, 0]
            else:
                rows = np.arange(len(batch))
                actual_selected = final_model_logits[rows, last_indices][:, selected_ids].numpy()
                actual_vocab = final_model_logits[rows, last_indices]
            max_error = np.max(np.abs(all_variant_logits[:, -1] - actual_selected), axis=1)
            logit_scale = np.maximum(np.max(np.abs(actual_selected), axis=1), 1.0)
            relative_error = max_error / logit_scale
            invalid = (max_error > FINAL_LENS_ABS_TOL) & (relative_error > FINAL_LENS_REL_TOL)
            if np.any(invalid):
                raise RuntimeError(
                    "Final-layer lens validation failed; "
                    f"max errors={max_error.tolist()}, relative errors={relative_error.tolist()}"
                )

            residual_np = residuals.numpy().astype(config.residual_dtype) if config.save_residuals else None
            for i, trial in enumerate(batch):
                top_id = int(actual_vocab[i].argmax())
                meta = {
                    "question_id": trial.question_id,
                    "condition": condition,
                    "baseline_answer": trial.baseline_answer,
                    "baseline_correct": trial.baseline_correct,
                    "correct_answer": trial.question["correct_answer"],
                    "prompt_hash": prompt_hash(prompts[i]),
                    "prompt_length": int(attention_mask[i].sum().item()),
                    "rendered_prompt": prompts[i],
                    "messages": messages[i],
                    "full_vocab_top_token_id": top_id,
                    "full_vocab_top_token": tokenizer.decode([top_id]),
                    "canonical_ad_choice": "ABCD"[int(canonical_logits[i, -1].argmax())],
                    "final_lens_max_abs_error": float(max_error[i]),
                    "final_lens_max_rel_error": float(relative_error[i]),
                }
                arrays = {
                    "canonical_logits": canonical_logits[i].astype(np.float32),
                    "variant_logits": all_variant_logits[i].astype(np.float32),
                    "metadata": json_array(meta),
                }
                if residual_np is not None:
                    arrays["residuals"] = residual_np[i]
                atomic_save_npz(shard_path(output_dir, condition, trial.question_id), **arrays)
            saved_counts[condition] += len(batch)
            print(
                f"{condition}: saved {saved_counts[condition]}/{pending_counts[condition]} pending trials",
                flush=True,
            )


def main() -> None:
    args = config_arg_parser("Collect final-position residuals and A-D logit-lens scores").parse_args()
    collect(ExperimentConfig.load(args.config))


if __name__ == "__main__":
    main()
