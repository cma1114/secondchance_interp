from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .head_patching import BatchedSingleHeadContextPatcher
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
from .sublayer_config import SublayerExperimentConfig


def run(config: SublayerExperimentConfig, plan_path: Path, output: Path) -> None:
    import torch

    plan = json.loads(plan_path.read_text())
    trials = load_trials(
        config.manifest_path, config.baseline_results_path, plan["question_ids"], None
    )
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    layers = sorted({int(target["layer"]) for target in plan["targets"]})
    grouped = defaultdict(list)
    for scenario in plan["scenarios"]:
        if len(scenario["targets"]) != 1 or len(scenario["targets"][0]["heads"]) != 1:
            raise ValueError("Batched discovery requires one head per scenario")
        grouped[(scenario["target_condition"], scenario["source_condition"], scenario["targets"][0]["layer"])].append(scenario)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metadata.json").write_text(json.dumps({
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "intervention": "one paired head-context replacement per batch row before o_proj",
        "batching": "all individual heads at one mixer share a target-prompt batch",
    }, indent=2, sort_keys=True))

    def natural_forward(condition: str, trial: object):
        prompt = render_chat(processor, build_messages(trial.question, condition, config.prompt_mode), config.disable_thinking)
        input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, [prompt])
        collector = FinalHeadContextCollector(parts, last_indices, layers)
        try:
            with torch.inference_mode():
                device = model_input_device(parts)
                kwargs = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device), "use_cache": False, "return_dict": True}
                try: result = model(**kwargs, logits_to_keep=1)
                except TypeError: result = model(**kwargs)
        finally:
            collector.close()
        logits = result.logits.detach().float().cpu()[0, -1, canonical_ids].numpy()
        stacked = collector.stacked()[0]
        return logits, {layer: stacked[index : index + 1] for index, layer in enumerate(layers)}, prompt

    def patched_batch(condition: str, source: dict, layer: int, scenarios: list[dict], trial: object):
        prompt = render_chat(processor, build_messages(trial.question, condition, config.prompt_mode), config.disable_thinking)
        prompts = [prompt] * len(scenarios)
        input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
        heads = [int(row["targets"][0]["heads"][0]) for row in scenarios]
        patcher = BatchedSingleHeadContextPatcher(parts, layer, heads, source[layer], last_indices)
        try:
            with torch.inference_mode():
                device = model_input_device(parts)
                kwargs = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device), "use_cache": False, "return_dict": True}
                try: result = model(**kwargs, logits_to_keep=1)
                except TypeError: result = model(**kwargs)
        finally:
            patcher.close()
        return result.logits.detach().float().cpu()[:, -1, canonical_ids].numpy(), prompt

    all_groups = ["natural_game", "natural_neutral", *[row["id"] for row in plan["scenarios"]]]
    for completed, trial in enumerate(trials, 1):
        pending = {group for group in all_groups if not shard_path(output, group, trial.question_id).exists()}
        if not pending:
            continue
        game_logits, game_source, game_prompt = natural_forward("incorrect", trial)
        neutral_logits, neutral_source, neutral_prompt = natural_forward("neutral", trial)
        sources = {"incorrect": game_source, "neutral": neutral_source}
        for group, condition, logits, prompt in (
            ("natural_game", "incorrect", game_logits, game_prompt),
            ("natural_neutral", "neutral", neutral_logits, neutral_prompt),
        ):
            if group in pending:
                atomic_save_npz(shard_path(output, group, trial.question_id), final_canonical_logits=logits.astype(np.float32), metadata=json_array({"question_id": trial.question_id, "scenario_id": group, "target_condition": condition, "prompt_hash": prompt_hash(prompt)}))
        for (target_condition, source_condition, layer), scenarios in grouped.items():
            needed = [row for row in scenarios if row["id"] in pending]
            if not needed:
                continue
            logits, prompt = patched_batch(target_condition, sources[source_condition], int(layer), needed, trial)
            for row, values in zip(needed, logits):
                atomic_save_npz(shard_path(output, row["id"], trial.question_id), final_canonical_logits=values.astype(np.float32), metadata=json_array({"question_id": trial.question_id, "scenario_id": row["id"], "source_condition": source_condition, "target_condition": target_condition, "targets": row["targets"], "prompt_hash": prompt_hash(prompt)}))
        if completed == 1 or completed % 10 == 0 or completed == len(trials):
            print(f"batched head discovery: completed {completed}/{len(trials)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch individual head discovery interventions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(SublayerExperimentConfig.load(args.config), Path(args.plan), Path(args.output))


if __name__ == "__main__":
    main()
