from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .attention_spans import attention_span_indices
from .collect_attention import _attention_tuple
from .io import atomic_save_npz, json_array, shard_path
from .jlens_collect import ANCHORS, _anchor_positions
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages, load_trials, prompt_hash
from .sublayer_config import SublayerExperimentConfig


def run(config_path: Path, plan_path: Path, output: Path, first_layer: int) -> None:
    import torch

    config = SublayerExperimentConfig.load(config_path)
    if config.attn_implementation != "eager":
        raise ValueError("Exact token attention requires eager attention")
    plan = json.loads(plan_path.read_text())
    qids = plan.get("question_ids", plan.get("confirmation_question_ids"))
    if not qids:
        raise ValueError("Plan has no question IDs")
    trials = load_trials(config.manifest_path, config.baseline_results_path, qids, None)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    all_attention_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    ]
    # User-facing layer numbers are one-based: Mixer 56 is model index 55.
    late_layers = [index for index in all_attention_layers if index + 1 >= first_layer]
    if not late_layers:
        raise RuntimeError("No ordinary-attention layers matched the requested range")
    late_indices = [all_attention_layers.index(layer) for layer in late_layers]
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metadata.json").write_text(json.dumps({
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "question_ids": qids,
        "first_user_facing_layer": first_layer,
        "ordinary_attention_model_indices_zero_based": all_attention_layers,
        "selected_model_indices_zero_based": late_layers,
        "selected_user_facing_layers_one_based": [value + 1 for value in late_layers],
        "measurement": (
            "final-decision query attention to the historical first-answer "
            "generation endpoint"
        ),
    }, indent=2, sort_keys=True))

    for completed, trial in enumerate(trials, 1):
        for condition in ("incorrect", "neutral"):
            path = shard_path(output, condition, trial.question_id)
            if path.exists():
                continue
            messages = build_messages(trial.question, condition, config.prompt_mode)
            prompt = render_chat(
                processor,
                messages,
                config.disable_thinking,
                config.chat_serialization,
            )
            token_ids, spans = attention_span_indices(
                tokenizer, prompt, condition, trial.question
            )
            anchors = _anchor_positions(
                tokenizer,
                prompt,
                condition,
                spans,
                messages[0]["content"],
                messages[-1]["content"],
            )
            endpoint = int(anchors[ANCHORS.index("historical_answer_end")])
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, [prompt])
            if token_ids != input_ids[0].tolist():
                raise RuntimeError("Offset-aware and model tokenizations disagree")
            with torch.inference_mode():
                kwargs = {
                    "input_ids": input_ids.to(model_input_device(parts)),
                    "attention_mask": attention_mask.to(model_input_device(parts)),
                    "use_cache": False,
                    "return_dict": True,
                    "output_attentions": True,
                }
                try:
                    result = model(**kwargs, logits_to_keep=1)
                except TypeError:
                    result = model(**kwargs)
            raw = _attention_tuple(result)
            if len(raw) == len(parts.layers):
                ordinary = [raw[index] for index in all_attention_layers]
            else:
                ordinary = [value for value in raw if value is not None]
            if len(ordinary) != len(all_attention_layers):
                raise RuntimeError("Unexpected number of ordinary-attention matrices")
            weights = torch.stack([
                ordinary[index][0, :, int(last_indices[0]), endpoint]
                for index in late_indices
            ]).detach().float().cpu().numpy()
            logits = result.logits[0, -1, canonical_ids].detach().float().cpu().numpy()
            atomic_save_npz(
                path,
                endpoint_attention=weights.astype(np.float16),
                final_canonical_logits=logits.astype(np.float32),
                metadata=json_array({
                    "question_id": trial.question_id,
                    "condition": condition,
                    "historical_endpoint": endpoint,
                    "historical_endpoint_token": tokenizer.decode([token_ids[endpoint]]),
                    "final_query": int(last_indices[0]),
                    "final_query_token": tokenizer.decode([token_ids[int(last_indices[0])]]),
                    "prompt_hash": prompt_hash(prompt),
                }),
            )
            del result, raw, ordinary
        if completed == 1 or completed % 10 == 0 or completed == len(trials):
            print(f"Late endpoint-attention scan: {completed}/{len(trials)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, default=47)
    args = parser.parse_args()
    run(args.config, args.plan, args.output, args.first_layer)


if __name__ == "__main__":
    main()

