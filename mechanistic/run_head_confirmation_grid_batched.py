from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .head_patching import BatchedScenarioHeadContextPatcher, HeadTarget
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


def _targets(values: list[dict]) -> list[HeadTarget]:
    return [
        HeadTarget(int(value["layer"]), tuple(int(head) for head in value["heads"]))
        for value in values
    ]


def _chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def run(
    config: SublayerExperimentConfig,
    plan_path: Path,
    output: Path,
    question_batch_size: int,
    scenario_batch_size: int,
) -> None:
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
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for scenario in plan["scenarios"]:
        grouped[(scenario["target_condition"], scenario["source_condition"])].append(scenario)
    groups = ["natural_game", "natural_neutral", *[row["id"] for row in plan["scenarios"]]]
    trials = [
        trial for trial in trials
        if not all(shard_path(output, group, trial.question_id).exists() for group in groups)
    ]

    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metadata.json").write_text(json.dumps({
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "intervention": "question-by-scenario grid of paired head-context replacements before o_proj",
        "question_batch_size": question_batch_size,
        "scenario_batch_size": scenario_batch_size,
    }, indent=2, sort_keys=True))

    def prompts_for(condition: str, batch: list[object]) -> list[str]:
        return [
            render_chat(
                processor,
                build_messages(trial.question, condition, config.prompt_mode),
                config.disable_thinking,
            )
            for trial in batch
        ]

    def natural_forward(condition: str, batch: list[object]):
        prompts = prompts_for(condition, batch)
        input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
        collector = FinalHeadContextCollector(parts, last_indices, layers)
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
                    result = model(**kwargs, logits_to_keep=1)
                except TypeError:
                    result = model(**kwargs)
        finally:
            collector.close()
        logits = result.logits.detach().float().cpu()[:, -1, canonical_ids].numpy()
        stacked = collector.stacked()
        source = {layer: stacked[:, index] for index, layer in enumerate(layers)}
        return logits, source, prompts

    def scenario_grid(
        condition: str,
        scenarios: list[dict],
        batch: list[object],
        source: dict | None = None,
    ):
        question_prompts = prompts_for(condition, batch)
        scenario_count = len(scenarios)
        prompts = [prompt for prompt in question_prompts for _ in scenarios]
        input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
        targets_by_row = [
            _targets(scenario["targets"])
            for _trial in batch
            for scenario in scenarios
        ]
        patcher = None
        if source is not None:
            expanded_source = {
                layer: values.repeat_interleave(scenario_count, dim=0)
                for layer, values in source.items()
            }
            patcher = BatchedScenarioHeadContextPatcher(
                parts, targets_by_row, expanded_source, last_indices
            )
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
                    result = model(**kwargs, logits_to_keep=1)
                except TypeError:
                    result = model(**kwargs)
        finally:
            if patcher is not None:
                patcher.close()
        logits = result.logits.detach().float().cpu()[:, -1, canonical_ids].numpy()
        return logits.reshape(len(batch), scenario_count, 4), question_prompts

    completed = 0
    for batch in _chunks(trials, question_batch_size):
        game_logits, game_source, game_prompts = natural_forward("incorrect", batch)
        neutral_logits, neutral_source, neutral_prompts = natural_forward("neutral", batch)
        sources = {"incorrect": game_source, "neutral": neutral_source}
        for index, trial in enumerate(batch):
            for group, condition, logits, prompts in (
                ("natural_game", "incorrect", game_logits, game_prompts),
                ("natural_neutral", "neutral", neutral_logits, neutral_prompts),
            ):
                path = shard_path(output, group, trial.question_id)
                if not path.exists():
                    atomic_save_npz(
                        path,
                        final_canonical_logits=logits[index].astype(np.float32),
                        metadata=json_array({
                            "question_id": trial.question_id,
                            "scenario_id": group,
                            "target_condition": condition,
                            "prompt_hash": prompt_hash(prompts[index]),
                        }),
                    )
        for (target_condition, source_condition), scenarios in grouped.items():
            for scenario_chunk in _chunks(scenarios, scenario_batch_size):
                matched, _ = scenario_grid(
                    target_condition, scenario_chunk, batch, source=None
                )
                logits, prompts = scenario_grid(
                    target_condition, scenario_chunk, batch,
                    source=sources[source_condition],
                )
                for question_index, trial in enumerate(batch):
                    for scenario_index, scenario in enumerate(scenario_chunk):
                        path = shard_path(output, scenario["id"], trial.question_id)
                        if path.exists():
                            continue
                        atomic_save_npz(
                            path,
                            final_canonical_logits=logits[question_index, scenario_index].astype(np.float32),
                            matched_natural_logits=matched[question_index, scenario_index].astype(np.float32),
                            metadata=json_array({
                                "question_id": trial.question_id,
                                "scenario_id": scenario["id"],
                                "source_condition": source_condition,
                                "target_condition": target_condition,
                                "targets": scenario["targets"],
                                "prompt_hash": prompt_hash(prompts[question_index]),
                            }),
                        )
        completed += len(batch)
        if completed == len(batch) or completed % 10 == 0 or completed == len(trials):
            print(f"grid-batched head confirmation: completed {completed}/{len(trials)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch head confirmation over questions and scenarios")
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--question-batch-size", type=int, default=2)
    parser.add_argument("--scenario-batch-size", type=int, default=24)
    args = parser.parse_args()
    run(
        SublayerExperimentConfig.load(args.config),
        Path(args.plan),
        Path(args.output),
        args.question_batch_size,
        args.scenario_batch_size,
    )


if __name__ == "__main__":
    main()
