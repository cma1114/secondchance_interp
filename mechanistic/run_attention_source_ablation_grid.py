from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .attention_intervention import BatchedAttentionEdgeAblator, EdgeTarget
from .attention_spans import attention_span_indices
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
from .run_attention_source_ablation import source_positions
from .sublayer_config import SublayerExperimentConfig


def _targets(values: list[dict]) -> list[EdgeTarget]:
    return [
        EdgeTarget(int(value["layer"]), tuple(int(head) for head in value["heads"]))
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

    if config.attn_implementation != "eager":
        raise ValueError("Source-edge ablation requires eager attention")
    plan = json.loads(plan_path.read_text())
    trials = load_trials(
        config.manifest_path, config.baseline_results_path, plan["question_ids"], None
    )
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for scenario in plan["scenarios"]:
        grouped[scenario["target_condition"]].append(scenario)
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
        "intervention": "question-by-source grid of final-query attention-edge ablations",
        "question_batch_size": question_batch_size,
        "scenario_batch_size": scenario_batch_size,
    }, indent=2, sort_keys=True))

    def annotated_prompts(condition: str, batch: list[object]):
        rows = []
        for trial in batch:
            prompt = render_chat(
                processor,
                build_messages(trial.question, condition, config.prompt_mode),
                config.disable_thinking,
            )
            ids, spans = attention_span_indices(
                tokenizer, prompt, condition, trial.question
            )
            rows.append((prompt, ids, spans))
        return rows

    def natural_forward(rows):
        prompts = [row[0] for row in rows]
        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
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
        return result.logits.detach().float().cpu()[:, -1, canonical_ids].numpy()

    def scenario_grid(rows, scenarios, intervene: bool):
        scenario_count = len(scenarios)
        prompts = [row[0] for row in rows for _ in scenarios]
        input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
        width = input_ids.shape[1]
        key_positions = []
        for _prompt, ids, spans in rows:
            offset = width - len(ids)
            for scenario in scenarios:
                key_positions.append([
                    offset + position
                    for position in source_positions(spans, scenario["source"])
                ])
        targets_by_row = [
            _targets(scenario["targets"])
            for _row in rows
            for scenario in scenarios
        ]
        ablator = BatchedAttentionEdgeAblator(
            parts, targets_by_row, last_indices, key_positions
        ) if intervene else None
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
            if ablator is not None:
                ablator.close()
        logits = result.logits.detach().float().cpu()[:, -1, canonical_ids].numpy()
        return logits.reshape(len(rows), scenario_count, 4), key_positions

    completed = 0
    for batch in _chunks(trials, question_batch_size):
        prompt_rows = {
            condition: annotated_prompts(condition, batch)
            for condition in ("incorrect", "neutral")
        }
        for group, condition in (("natural_game", "incorrect"), ("natural_neutral", "neutral")):
            logits = natural_forward(prompt_rows[condition])
            for index, trial in enumerate(batch):
                path = shard_path(output, group, trial.question_id)
                if path.exists():
                    continue
                atomic_save_npz(
                    path,
                    final_canonical_logits=logits[index].astype(np.float32),
                    metadata=json_array({
                        "question_id": trial.question_id,
                        "scenario_id": group,
                        "target_condition": condition,
                        "prompt_hash": prompt_hash(prompt_rows[condition][index][0]),
                    }),
                )
        for condition, scenarios in grouped.items():
            for scenario_chunk in _chunks(scenarios, scenario_batch_size):
                matched, _ = scenario_grid(
                    prompt_rows[condition], scenario_chunk, intervene=False
                )
                logits, padded_positions = scenario_grid(
                    prompt_rows[condition], scenario_chunk, intervene=True
                )
                scenario_count = len(scenario_chunk)
                for question_index, trial in enumerate(batch):
                    prompt, ids, spans = prompt_rows[condition][question_index]
                    for scenario_index, scenario in enumerate(scenario_chunk):
                        path = shard_path(output, scenario["id"], trial.question_id)
                        if path.exists():
                            continue
                        original_positions = source_positions(spans, scenario["source"])
                        atomic_save_npz(
                            path,
                            final_canonical_logits=logits[question_index, scenario_index].astype(np.float32),
                            matched_natural_logits=matched[question_index, scenario_index].astype(np.float32),
                            metadata=json_array({
                                "question_id": trial.question_id,
                                "scenario_id": scenario["id"],
                                "target_condition": condition,
                                "source": scenario["source"],
                                "targets": scenario["targets"],
                                "source_positions": original_positions,
                                "source_tokens": tokenizer.convert_ids_to_tokens(
                                    [ids[index] for index in original_positions]
                                ),
                                "prompt_hash": prompt_hash(prompt),
                            }),
                        )
        completed += len(batch)
        if completed == len(batch) or completed % 10 == 0 or completed == len(trials):
            print(f"grid-batched source ablation: completed {completed}/{len(trials)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch source ablations over questions and scenarios")
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--question-batch-size", type=int, default=2)
    parser.add_argument("--scenario-batch-size", type=int, default=8)
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
