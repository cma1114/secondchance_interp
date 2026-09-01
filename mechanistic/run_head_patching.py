from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .head_patching import FinalHeadContextPatcher, HeadTarget
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


def run(config: SublayerExperimentConfig, plan_path: Path, output: Path) -> None:
    import torch
    import transformers

    plan = json.loads(plan_path.read_text())
    question_ids = plan.get("question_ids")
    if not question_ids:
        raise ValueError("Head-patching plan must contain non-empty question_ids")
    trials = load_trials(
        config.manifest_path, config.baseline_results_path, question_ids, None
    )
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    layers = sorted({int(target["layer"]) for target in plan["targets"]})
    for layer in layers:
        if getattr(parts.layers[layer], "self_attn", None) is None:
            raise ValueError(f"Layer {layer} is not an ordinary attention block")

    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "intervention": (
            "replace selected final-position pre-o_proj attention-head contexts "
            "with paired same-question contexts from the other condition"
        ),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    def forward(
        condition: str,
        trial: object,
        collect: bool = False,
        patch_targets: list[HeadTarget] | None = None,
        source: dict[int, object] | None = None,
    ):
        messages = build_messages(trial.question, condition, config.prompt_mode)
        prompt = render_chat(processor, messages, config.disable_thinking)
        input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, [prompt])
        collector = (
            FinalHeadContextCollector(parts, last_indices, layers) if collect else None
        )
        patcher = (
            FinalHeadContextPatcher(parts, patch_targets or [], source or {}, last_indices)
            if patch_targets
            else None
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
            if collector is not None:
                collector.close()
        logits = result.logits.detach().float().cpu()[0, -1, canonical_ids].numpy()
        source_values = None
        if collector is not None:
            stacked = collector.stacked()[0]
            source_values = {layer: stacked[index : index + 1] for index, layer in enumerate(layers)}
        return logits, source_values, prompt

    groups = ["natural_game", "natural_neutral", *[row["id"] for row in plan["scenarios"]]]
    for completed, trial in enumerate(trials, 1):
        pending = [group for group in groups if not shard_path(output, group, trial.question_id).exists()]
        if not pending:
            continue
        game_logits, game_source, game_prompt = forward("incorrect", trial, collect=True)
        neutral_logits, neutral_source, neutral_prompt = forward("neutral", trial, collect=True)
        natural = {
            "incorrect": (game_logits, game_prompt),
            "neutral": (neutral_logits, neutral_prompt),
        }
        sources = {"incorrect": game_source, "neutral": neutral_source}
        for group, condition in (("natural_game", "incorrect"), ("natural_neutral", "neutral")):
            if group in pending:
                logits, prompt = natural[condition]
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
        for scenario in plan["scenarios"]:
            if scenario["id"] not in pending:
                continue
            target_condition = scenario["target_condition"]
            logits, _, prompt = forward(
                target_condition,
                trial,
                patch_targets=_targets(scenario["targets"]),
                source=sources[scenario["source_condition"]],
            )
            atomic_save_npz(
                shard_path(output, scenario["id"], trial.question_id),
                final_canonical_logits=logits.astype(np.float32),
                metadata=json_array({
                    "question_id": trial.question_id,
                    "scenario_id": scenario["id"],
                    "source_condition": scenario["source_condition"],
                    "target_condition": target_condition,
                    "targets": scenario["targets"],
                    "prompt_hash": prompt_hash(prompt),
                }),
            )
        if completed == 1 or completed % 10 == 0 or completed == len(trials):
            print(f"head patching: completed {completed}/{len(trials)} planned questions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired attention-head context patches")
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(SublayerExperimentConfig.load(args.config), Path(args.plan), Path(args.output))


if __name__ == "__main__":
    main()
