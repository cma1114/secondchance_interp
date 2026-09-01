from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .attention_intervention import AttentionEdgeAblator, EdgeTarget
from .attention_spans import attention_span_indices
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


MIXER_LAYER = 55  # zero-based model index: Mixer 56


def run(config_path: Path, plan_path: Path, output: Path) -> None:
    import torch
    import transformers

    config = SublayerExperimentConfig.load(config_path)
    if config.attn_implementation != "eager":
        raise ValueError("Exact attention-edge ablation requires eager attention")
    plan = json.loads(plan_path.read_text())
    qids = plan.get("question_ids", plan.get("confirmation_question_ids"))
    if not qids:
        raise ValueError("Plan has no question IDs")
    trials = load_trials(
        config.manifest_path, config.baseline_results_path, qids, None
    )
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    attention = parts.layers[MIXER_LAYER].self_attn
    num_heads = getattr(attention, "num_heads", None)
    if num_heads is None:
        num_heads = int(attention.q_proj.out_features // (2 * attention.head_dim))
    all_heads = tuple(range(int(num_heads)))

    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "config": config.as_dict(),
                "plan_path": str(plan_path),
                "question_ids": qids,
                "model_layer_index_zero_based": MIXER_LAYER,
                "component_name": "Mixer 56",
                "heads": list(all_heads),
                "intervention": (
                    "set the historical-answer-endpoint attention logit to "
                    "negative infinity for the final query in every Mixer 56 head"
                ),
                "software": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                },
                "platform": platform.platform(),
            },
            indent=2,
            sort_keys=True,
        )
    )

    def forward(trial: object, condition: str, ablate: bool):
        messages = build_messages(trial.question, condition, config.prompt_mode)
        prompt = render_chat(
            processor,
            messages,
            config.disable_thinking,
            config.chat_serialization,
        )
        annotated_ids, spans = attention_span_indices(
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
        if annotated_ids != input_ids[0].tolist():
            raise RuntimeError("Offset-aware and model tokenizations disagree")
        ablator = None
        if ablate:
            ablator = AttentionEdgeAblator(
                parts,
                [EdgeTarget(MIXER_LAYER, all_heads)],
                [int(last_indices[0])],
                [endpoint],
            )
        try:
            with torch.inference_mode():
                kwargs = {
                    "input_ids": input_ids.to(model_input_device(parts)),
                    "attention_mask": attention_mask.to(model_input_device(parts)),
                    "use_cache": False,
                    "return_dict": True,
                }
                try:
                    result = model(**kwargs, logits_to_keep=1)
                except TypeError:
                    result = model(**kwargs)
        finally:
            if ablator is not None:
                ablator.close()
        logits = result.logits[0, -1, canonical_ids].detach().float().cpu().numpy()
        return logits, prompt, annotated_ids, endpoint, int(last_indices[0])

    audit_path = output / "position_audit.json"
    groups = (
        ("game_natural", "incorrect", False),
        ("game_endpoint_ablated", "incorrect", True),
        ("neutral_natural", "neutral", False),
        ("neutral_endpoint_ablated", "neutral", True),
    )
    for completed, trial in enumerate(trials, 1):
        for group, condition, ablate in groups:
            path = shard_path(output, group, trial.question_id)
            if path.exists():
                continue
            logits, prompt, token_ids, endpoint, query = forward(
                trial, condition, ablate
            )
            atomic_save_npz(
                path,
                final_canonical_logits=logits.astype(np.float32),
                historical_endpoint=np.asarray(endpoint, dtype=np.int32),
                final_query=np.asarray(query, dtype=np.int32),
                metadata=json_array(
                    {
                        "question_id": trial.question_id,
                        "scenario": group,
                        "condition": condition,
                        "ablated": ablate,
                        "historical_endpoint_token": tokenizer.decode(
                            [token_ids[endpoint]]
                        ),
                        "prompt_hash": prompt_hash(prompt),
                    }
                ),
            )
            if not audit_path.exists():
                audit_path.write_text(
                    json.dumps(
                        {
                            "question_id": trial.question_id,
                            "condition": condition,
                            "historical_endpoint": endpoint,
                            "historical_endpoint_token": tokenizer.decode(
                                [token_ids[endpoint]]
                            ),
                            "historical_endpoint_token_id": token_ids[endpoint],
                            "final_query": query,
                            "final_query_token": tokenizer.decode([token_ids[query]]),
                            "heads": list(all_heads),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
        if completed == 1 or completed % 10 == 0 or completed == len(trials):
            print(f"Mixer 56 endpoint-edge ablation: {completed}/{len(trials)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.plan, args.output)


if __name__ == "__main__":
    main()

