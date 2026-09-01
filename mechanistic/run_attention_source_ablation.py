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


SOURCE_SELECTORS = (
    "condition_keyword",
    "action_keyword",
    "feedback_sentence",
    "system_condition",
    "first_question",
    "repeated_question",
    "redacted_answer",
    "local_answer_cue",
)


def source_positions(spans: dict[str, list[int]], selector: str) -> list[int]:
    if selector == "condition_keyword":
        # `condition_keyword` also records the same word in the system prefix
        # when present. This selector is specifically the user feedback event.
        positions = sorted(set(spans["condition_keyword"]) & set(spans["feedback_sentence"]))
    elif selector == "local_answer_cue":
        positions = spans["previous_8"]
    elif selector in SOURCE_SELECTORS:
        positions = spans[selector]
    else:
        raise ValueError(f"Unknown source selector: {selector}")
    if not positions:
        raise RuntimeError(f"No positions matched {selector}")
    return sorted(set(int(value) for value in positions))


def _targets(values: list[dict]) -> list[EdgeTarget]:
    return [
        EdgeTarget(int(value["layer"]), tuple(int(head) for head in value["heads"]))
        for value in values
    ]


def run(config: SublayerExperimentConfig, plan_path: Path, output: Path) -> None:
    import torch
    import transformers

    if config.attn_implementation != "eager":
        raise ValueError("Source-edge ablation requires eager attention")
    plan = json.loads(plan_path.read_text())
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        plan["question_ids"],
        None,
    )
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "intervention": (
            "mask selected final-query-to-semantic-source attention logits before softmax"
        ),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    def forward(condition: str, trial: object, scenario: dict | None = None):
        messages = build_messages(trial.question, condition, config.prompt_mode)
        prompt = render_chat(processor, messages, config.disable_thinking)
        annotated_ids, spans = attention_span_indices(tokenizer, prompt, condition, trial.question)
        input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, [prompt])
        if annotated_ids != input_ids[0, : len(annotated_ids)].tolist():
            raise RuntimeError("Offset-aware and model tokenizations disagree")
        ablator = None
        positions = []
        if scenario is not None:
            positions = source_positions(spans, scenario["source"])
            ablator = AttentionEdgeAblator(
                parts,
                _targets(scenario["targets"]),
                [last_indices[0]],
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
            if ablator is not None:
                ablator.close()
        logits = result.logits.detach().float().cpu()[0, -1, canonical_ids].numpy()
        return logits, prompt, positions, annotated_ids

    groups = ["natural_game", "natural_neutral", *[row["id"] for row in plan["scenarios"]]]
    for completed, trial in enumerate(trials, 1):
        pending = [group for group in groups if not shard_path(output, group, trial.question_id).exists()]
        if not pending:
            continue
        natural = {}
        for group, condition in (("natural_game", "incorrect"), ("natural_neutral", "neutral")):
            logits, prompt, _, _ = forward(condition, trial)
            natural[condition] = (logits, prompt)
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
        for scenario in plan["scenarios"]:
            if scenario["id"] not in pending:
                continue
            condition = scenario["target_condition"]
            logits, prompt, positions, ids = forward(condition, trial, scenario)
            atomic_save_npz(
                shard_path(output, scenario["id"], trial.question_id),
                final_canonical_logits=logits.astype(np.float32),
                metadata=json_array({
                    "question_id": trial.question_id,
                    "scenario_id": scenario["id"],
                    "target_condition": condition,
                    "source": scenario["source"],
                    "targets": scenario["targets"],
                    "source_positions": positions,
                    "source_tokens": tokenizer.convert_ids_to_tokens([ids[index] for index in positions]),
                    "prompt_hash": prompt_hash(prompt),
                }),
            )
        if completed == 1 or completed % 10 == 0 or completed == len(trials):
            print(f"source-edge ablation: completed {completed}/{len(trials)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate attention edges to semantic source spans")
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(SublayerExperimentConfig.load(args.config), Path(args.plan), Path(args.output))


if __name__ == "__main__":
    main()
