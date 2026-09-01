from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

from .attention_spans import SPAN_NAMES, attention_span_indices
from .config import ExperimentConfig, config_arg_parser
from .io import atomic_save_npz, json_array, shard_path
from .modeling import (
    FinalHeadContextCollector,
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages, load_trials, prompt_hash


def _attention_tuple(output):
    attentions = getattr(output, "attentions", None)
    if attentions is None:
        attentions = getattr(output, "decoder_attentions", None)
    if attentions is None:
        raise RuntimeError(
            "The model returned no attention matrices. Use an eager attention implementation."
        )
    return attentions


def _canonical_token_ids(tokenizer, variants):
    resolved = resolve_answer_tokens(tokenizer, variants)
    return [resolved[letter][0][1] for letter in "ABCD"], resolved


def _head_ad_projection(parts, canonical_ids, n_heads, head_dim, layer_indices):
    """Map each pre-o_proj head context to its direct, unnormalized A-D write."""
    import torch

    rows = parts.output_head.weight.detach()[canonical_ids].float()
    projections = []
    with torch.inference_mode():
        for layer_index in layer_indices:
            layer = parts.layers[layer_index]
            weight = layer.self_attn.o_proj.weight.detach().float()
            projected = weight.T @ rows.T
            projections.append(projected.reshape(n_heads, head_dim, 4).cpu())
    return torch.stack(projections)


def collect_attention(config: ExperimentConfig) -> None:
    import torch
    import transformers

    if config.batch_size != 1:
        raise ValueError("Attention collection currently requires batch_size=1")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        config.question_ids,
        config.max_questions,
    )
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    canonical_ids, resolved = _canonical_token_ids(tokenizer, config.answer_variants)
    full_attention_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    ]
    if not full_attention_layers:
        raise RuntimeError("The model has no conventional self-attention layers")
    projection = None
    architecture = None

    run_meta = {
        "config": config.as_dict(),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "resolved_answer_tokens": resolved,
        "span_names": list(SPAN_NAMES),
        "n_text_layers": len(parts.layers),
        "full_attention_layers": full_attention_layers,
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True))

    for condition in config.conditions:
        pending = [trial for trial in trials if not shard_path(output_dir, condition, trial.question_id).exists()]
        print(f"{condition}: {len(pending)} pending / {len(trials)} total", flush=True)
        for completed, trial in enumerate(pending, 1):
            messages = build_messages(trial.question, condition, config.prompt_mode)
            prompt = render_chat(processor, messages, config.disable_thinking)
            annotated_ids, spans = attention_span_indices(tokenizer, prompt, condition, trial.question)
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, [prompt])
            if annotated_ids != input_ids[0, : len(annotated_ids)].tolist():
                raise RuntimeError("Offset-aware and model input tokenizations disagree")

            context_collector = FinalHeadContextCollector(parts, last_indices, full_attention_layers)
            try:
                with torch.inference_mode():
                    device = model_input_device(parts)
                    kwargs = {
                        "input_ids": input_ids.to(device),
                        "attention_mask": attention_mask.to(device),
                        "use_cache": False,
                        "return_dict": True,
                        "output_attentions": True,
                    }
                    try:
                        output = model(**kwargs, logits_to_keep=1)
                    except TypeError:
                        output = model(**kwargs)
                head_context = context_collector.stacked()[0].float()
            finally:
                context_collector.close()

            all_attentions = _attention_tuple(output)
            if len(all_attentions) == len(parts.layers):
                attentions = [all_attentions[index] for index in full_attention_layers]
            else:
                attentions = [value for value in all_attentions if value is not None]
            if len(attentions) != len(full_attention_layers) or any(value is None for value in attentions):
                raise RuntimeError(
                    f"Expected {len(full_attention_layers)} full-attention matrices; got {len(attentions)}"
                )
            final_rows = torch.stack(
                [layer_attention[0, :, last_indices[0], : len(annotated_ids)].detach().float().cpu() for layer_attention in attentions]
            )
            n_layers, n_heads, sequence_length = final_rows.shape
            if n_layers != len(full_attention_layers):
                raise RuntimeError(f"Got {n_layers} attention matrices for {len(full_attention_layers)} full-attention blocks")
            if head_context.shape[-1] % n_heads:
                raise RuntimeError("Pre-output attention width is not divisible by the number of heads")
            head_dim = head_context.shape[-1] // n_heads
            head_context = head_context.reshape(n_layers, n_heads, head_dim)

            if projection is None:
                projection = _head_ad_projection(
                    parts, canonical_ids, n_heads, head_dim, full_attention_layers
                )
                architecture = {"n_heads": n_heads, "head_dim": head_dim}
                run_meta["architecture"] = architecture
                (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True))

            attention_mass = torch.zeros((n_layers, n_heads, len(SPAN_NAMES)), dtype=torch.float32)
            attention_peak = torch.zeros_like(attention_mass)
            for span_index, name in enumerate(SPAN_NAMES):
                positions = spans[name]
                if positions:
                    values = final_rows[:, :, positions]
                    attention_mass[:, :, span_index] = values.sum(dim=-1)
                    attention_peak[:, :, span_index] = values.max(dim=-1).values

            top_weight, top_position = torch.topk(final_rows, k=min(8, sequence_length), dim=-1)
            direct_ad = torch.einsum("lhd,lhdc->lhc", head_context, projection)
            direct_ad -= direct_ad.mean(dim=-1, keepdim=True)
            logits = output.logits.detach().float().cpu()
            final_logits = logits[0, -1, canonical_ids].numpy()
            metadata = {
                "question_id": trial.question_id,
                "condition": condition,
                "baseline_answer": trial.baseline_answer,
                "baseline_correct": trial.baseline_correct,
                "correct_answer": trial.question["correct_answer"],
                "prompt_hash": prompt_hash(prompt),
                "prompt_length": len(annotated_ids),
                "rendered_prompt": prompt,
                "messages": messages,
                "tokens": tokenizer.convert_ids_to_tokens(annotated_ids),
                "span_token_positions": spans,
                "span_names": list(SPAN_NAMES),
                "full_attention_layers": full_attention_layers,
            }
            atomic_save_npz(
                shard_path(output_dir, condition, trial.question_id),
                attention_mass=attention_mass.numpy().astype(np.float16),
                attention_peak=attention_peak.numpy().astype(np.float16),
                top_attention_weight=top_weight.numpy().astype(np.float16),
                top_attention_position=top_position.numpy().astype(np.int16),
                head_context_norm=torch.linalg.vector_norm(head_context, dim=-1).numpy().astype(np.float16),
                head_direct_ad=direct_ad.numpy().astype(np.float16),
                final_canonical_logits=final_logits.astype(np.float32),
                input_ids=np.asarray(annotated_ids, dtype=np.int32),
                metadata=json_array(metadata),
            )
            del output, all_attentions, attentions, final_rows
            if completed == 1 or completed % 25 == 0 or completed == len(pending):
                print(f"{condition}: saved {completed}/{len(pending)} pending trials", flush=True)


def main() -> None:
    args = config_arg_parser("Collect final-position feedback attention by layer and head").parse_args()
    collect_attention(ExperimentConfig.load(args.config))


if __name__ == "__main__":
    main()
