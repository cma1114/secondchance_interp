from __future__ import annotations

import json
import platform
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .attention_ablation_config import AttentionAblationConfig
from .attention_intervention import AttentionEdgeAblator, EdgeTarget
from .attention_spans import attention_span_indices
from .config import config_arg_parser
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


def _source_positions(spans: dict[str, list[int]], selector: str) -> list[int]:
    keyword = set(spans["condition_keyword"])
    if selector == "user_incorrect":
        positions = keyword & set(spans["feedback_sentence"])
    elif selector == "system_incorrect":
        positions = keyword & set(spans["system_condition"])
    else:
        raise ValueError(f"Unknown source selector: {selector}")
    if not positions:
        raise RuntimeError(f"No tokens matched {selector}")
    return sorted(positions)


def _targets(scenario: dict) -> list[EdgeTarget]:
    return [
        EdgeTarget(int(target["layer"]), tuple(int(head) for head in target["heads"]))
        for target in scenario["targets"]
    ]


def run(config: AttentionAblationConfig) -> None:
    import torch
    import transformers

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
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    full_attention_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    ]
    requested_layers = {
        int(target["layer"])
        for scenario in config.scenarios
        for target in scenario["targets"]
    }
    missing = requested_layers - set(full_attention_layers)
    if missing:
        raise ValueError(f"Requested non-softmax-attention layers: {sorted(missing)}")

    run_meta = {
        "config": config.as_dict(),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "resolved_answer_tokens": resolved,
        "full_attention_layers": full_attention_layers,
        "intervention": "mask selected final-query-to-source attention logits before softmax",
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True))

    for scenario in config.scenarios:
        scenario_id = scenario["id"]
        pending = [
            trial for trial in trials
            if not shard_path(output_dir, scenario_id, trial.question_id).exists()
        ]
        print(f"{scenario_id}: {len(pending)} pending / {len(trials)} total", flush=True)
        for completed, trial in enumerate(pending, 1):
            messages = build_messages(trial.question, "incorrect", config.prompt_mode)
            prompt = render_chat(processor, messages, config.disable_thinking)
            annotated_ids, spans = attention_span_indices(
                tokenizer, prompt, "incorrect", trial.question
            )
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, [prompt])
            if annotated_ids != input_ids[0, : len(annotated_ids)].tolist():
                raise RuntimeError("Offset-aware and model input tokenizations disagree")
            source_positions = (
                [] if scenario["source"] == "none"
                else _source_positions(spans, scenario["source"])
            )
            device = model_input_device(parts)
            intervention = (
                nullcontext()
                if scenario["source"] == "none"
                else AttentionEdgeAblator(
                    parts,
                    _targets(scenario),
                    last_indices,
                    source_positions,
                )
            )
            with intervention, torch.inference_mode():
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
            metadata = {
                "question_id": trial.question_id,
                "scenario_id": scenario_id,
                "source": scenario["source"],
                "targets": scenario["targets"],
                "source_token_positions": source_positions,
                "query_position": last_indices[0],
                "source_tokens": tokenizer.convert_ids_to_tokens(
                    [annotated_ids[position] for position in source_positions]
                ),
                "baseline_answer": trial.baseline_answer,
                "baseline_correct": trial.baseline_correct,
                "correct_answer": trial.question["correct_answer"],
                "prompt_hash": prompt_hash(prompt),
                "prompt_length": len(annotated_ids),
            }
            atomic_save_npz(
                shard_path(output_dir, scenario_id, trial.question_id),
                final_canonical_logits=final_logits.astype(np.float32),
                metadata=json_array(metadata),
            )
            del output
            if completed == 1 or completed % 25 == 0 or completed == len(pending):
                print(f"{scenario_id}: saved {completed}/{len(pending)}", flush=True)


def main() -> None:
    args = config_arg_parser("Causally ablate final-query attention to feedback tokens").parse_args()
    run(AttentionAblationConfig.load(args.config))


if __name__ == "__main__":
    main()
