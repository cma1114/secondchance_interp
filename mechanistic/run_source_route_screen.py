from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

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
from .source_partition import SOURCE_NAMES, prompt_source_partition
from .source_route_collectors import (
    AttentionSourceRouteCollector,
    DeltaNetSourceRouteCollector,
)
from .sublayer_config import SublayerExperimentConfig


ATTENTION_LAYER = 55
GDN_LAYER = 62


def _forward(model, parts, input_ids, attention_mask):
    import torch

    device = model_input_device(parts)
    kwargs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "use_cache": False,
        "return_dict": True,
    }
    with torch.inference_mode():
        try:
            return model(**kwargs, logits_to_keep=1)
        except TypeError:
            return model(**kwargs)


def run(
    config: SublayerExperimentConfig,
    plan_path: Path,
    output: Path,
) -> None:
    import torch
    import transformers

    if config.attn_implementation != "eager":
        raise ValueError("Source-route screen requires eager attention")
    plan = json.loads(plan_path.read_text())
    qids = plan.get("question_ids", plan.get("confirmation_question_ids"))
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        qids,
        None,
    )
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    if getattr(parts.layers[ATTENTION_LAYER], "self_attn", None) is None:
        raise RuntimeError(f"Layer {ATTENTION_LAYER} is not ordinary attention")
    if getattr(parts.layers[GDN_LAYER], "linear_attn", None) is None:
        raise RuntimeError(f"Layer {GDN_LAYER} is not Gated DeltaNet")

    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "attention_layer": ATTENTION_LAYER,
        "gdn_layer": GDN_LAYER,
        "source_names": SOURCE_NAMES,
        "method": (
            "exhaustive disjoint semantic source partition; exact final-query "
            "attention-edge renormalization and exact DeltaNet recurrence replay"
        ),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )

    groups = ("baseline", "incorrect", "neutral")
    for completed, trial in enumerate(trials, 1):
        for condition in groups:
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
            input_ids, attention_mask, last_indices = tokenize_batch(
                tokenizer, [prompt]
            )
            if condition == "baseline":
                result = _forward(model, parts, input_ids, attention_mask)
                final_logits = (
                    result.logits.detach().float().cpu()[0, -1, canonical_ids].numpy()
                )
                atomic_save_npz(
                    path,
                    final_canonical_logits=final_logits.astype(np.float32),
                    metadata=json_array(
                        {
                            "question_id": trial.question_id,
                            "condition": condition,
                            "query_position": last_indices[0],
                            "prompt_hash": prompt_hash(prompt),
                        }
                    ),
                )
                del result
                continue

            token_ids, sources = prompt_source_partition(
                tokenizer, prompt, messages, trial.question, condition
            )
            if token_ids != input_ids[0].tolist():
                raise RuntimeError("Source-partition and model tokenizations disagree")
            attention = AttentionSourceRouteCollector(
                parts, ATTENTION_LAYER, sources, canonical_ids
            )
            gdn = DeltaNetSourceRouteCollector(
                parts, GDN_LAYER, sources, canonical_ids
            )
            try:
                result = _forward(model, parts, input_ids, attention_mask)
                arrays = {**attention.arrays(), **gdn.arrays()}
            finally:
                gdn.close()
                attention.close()
            final_logits = (
                result.logits.detach().float().cpu()[0, -1, canonical_ids].numpy()
            )
            atomic_save_npz(
                path,
                **arrays,
                final_canonical_logits=final_logits.astype(np.float32),
                token_ids=np.asarray(token_ids, dtype=np.int32),
                token_source_index=np.asarray(
                    [
                        next(
                            index
                            for index, name in enumerate(SOURCE_NAMES)
                            if position in sources[name]
                        )
                        for position in range(len(token_ids))
                    ],
                    dtype=np.int16,
                ),
                metadata=json_array(
                    {
                        "question_id": trial.question_id,
                        "condition": condition,
                        "source_positions": sources,
                        "source_tokens": {
                            name: tokenizer.convert_ids_to_tokens(
                                [token_ids[position] for position in positions]
                            )
                            for name, positions in sources.items()
                        },
                        "query_position": last_indices[0],
                        "prompt_hash": prompt_hash(prompt),
                    }
                ),
            )
            del result, attention, gdn
        if completed == 1 or completed % 5 == 0 or completed == len(trials):
            print(f"source-route screen: completed {completed}/{len(trials)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen complete prompt-source routes into Mixers 56 and 63"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(
        SublayerExperimentConfig.load(args.config),
        Path(args.plan),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
