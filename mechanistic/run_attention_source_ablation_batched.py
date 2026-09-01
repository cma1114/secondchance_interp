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


def _chunks(values: list[dict], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def run(
    config: SublayerExperimentConfig,
    plan_path: Path,
    output: Path,
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
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metadata.json").write_text(json.dumps({
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "intervention": "one final-query-to-semantic-source attention-edge ablation per batch row",
        "scenario_batch_size": scenario_batch_size,
    }, indent=2, sort_keys=True))

    def annotated_prompt(condition: str, trial: object):
        prompt = render_chat(
            processor,
            build_messages(trial.question, condition, config.prompt_mode),
            config.disable_thinking,
        )
        annotated_ids, spans = attention_span_indices(
            tokenizer, prompt, condition, trial.question
        )
        return prompt, annotated_ids, spans

    def natural_forward(prompt: str):
        input_ids, attention_mask, _ = tokenize_batch(tokenizer, [prompt])
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
        return result.logits.detach().float().cpu()[0, -1, canonical_ids].numpy()

    def patched_batch(
        prompt: str,
        annotated_ids: list[int],
        spans: dict[str, list[int]],
        scenarios: list[dict],
    ):
        input_ids, attention_mask, last_indices = tokenize_batch(
            tokenizer, [prompt] * len(scenarios)
        )
        if annotated_ids != input_ids[0, :len(annotated_ids)].tolist():
            raise RuntimeError("Offset-aware and model tokenizations disagree")
        positions = [source_positions(spans, row["source"]) for row in scenarios]
        ablator = BatchedAttentionEdgeAblator(
            parts,
            [_targets(row["targets"]) for row in scenarios],
            last_indices,
            positions,
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
            ablator.close()
        return result.logits.detach().float().cpu()[:, -1, canonical_ids].numpy(), positions

    groups = ["natural_game", "natural_neutral", *[row["id"] for row in plan["scenarios"]]]
    for completed, trial in enumerate(trials, 1):
        pending = {group for group in groups if not shard_path(output, group, trial.question_id).exists()}
        if not pending:
            continue
        prompts = {}
        for group, condition in (("natural_game", "incorrect"), ("natural_neutral", "neutral")):
            prompt, ids, spans = annotated_prompt(condition, trial)
            prompts[condition] = (prompt, ids, spans)
            logits = natural_forward(prompt)
            if group in pending:
                atomic_save_npz(
                    shard_path(output, group, trial.question_id),
                    final_canonical_logits=logits.astype(np.float32),
                    metadata=json_array({
                        "question_id": trial.question_id,
                        "scenario_id": group,
                        "target_condition": condition,
                        "prompt_hash": prompt_hash(prompt),
                    }),
                )
        for condition, scenarios in grouped.items():
            needed = [row for row in scenarios if row["id"] in pending]
            prompt, ids, spans = prompts[condition]
            for chunk in _chunks(needed, scenario_batch_size):
                logits, positions = patched_batch(prompt, ids, spans, chunk)
                for row, values, source_indices in zip(chunk, logits, positions):
                    atomic_save_npz(
                        shard_path(output, row["id"], trial.question_id),
                        final_canonical_logits=values.astype(np.float32),
                        metadata=json_array({
                            "question_id": trial.question_id,
                            "scenario_id": row["id"],
                            "target_condition": condition,
                            "source": row["source"],
                            "targets": row["targets"],
                            "source_positions": source_indices,
                            "source_tokens": tokenizer.convert_ids_to_tokens(
                                [ids[index] for index in source_indices]
                            ),
                            "prompt_hash": prompt_hash(prompt),
                        }),
                    )
        if completed == 1 or completed % 10 == 0 or completed == len(trials):
            print(f"batched source-edge ablation: completed {completed}/{len(trials)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch semantic attention-source ablations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario-batch-size", type=int, default=8)
    args = parser.parse_args()
    run(
        SublayerExperimentConfig.load(args.config),
        Path(args.plan),
        Path(args.output),
        args.scenario_batch_size,
    )


if __name__ == "__main__":
    main()
