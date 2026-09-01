from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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
from .sublayer import (
    BatchedPositionComponentOutputPatcher,
    PositionComponentOutputCollector,
    PositionComponentTarget,
)
from .sublayer_config import SublayerExperimentConfig


def _targets(values: list[dict]) -> list[PositionComponentTarget]:
    return [
        PositionComponentTarget(
            layer=int(value["layer"]),
            kind=str(value["kind"]),
            anchor=str(value.get("anchor", "decision")),
        )
        for value in values
    ]


def _chunks(values: list[dict], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run(
    config: SublayerExperimentConfig,
    plan_path: Path,
    output: Path,
    batch_size_override: int | None = None,
) -> None:
    import torch
    import transformers

    plan = json.loads(plan_path.read_text())
    question_ids = plan.get("question_ids", plan.get("confirmation_question_ids"))
    if not question_ids:
        raise ValueError("Patching plan must contain non-empty question_ids")
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        question_ids,
        None,
    )
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    unique_targets = _targets(plan["targets"])
    anchor_names = list(dict.fromkeys(target.anchor for target in unique_targets))
    unknown = sorted(set(anchor_names) - set(ANCHORS))
    if unknown:
        raise ValueError(f"Unknown semantic anchors: {unknown}")
    batch_size = int(batch_size_override or plan.get("batch_size", 4))
    if batch_size < 1:
        raise ValueError("Batch size must be positive")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for scenario in plan["scenarios"]:
        grouped[(scenario["target_condition"], scenario["source_condition"])].append(scenario)

    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "intervention": (
            "replace complete mixer/MLP outputs at semantic prompt anchors with "
            "paired same-question outputs from the other condition"
        ),
        "batch_control": (
            "every patched forward is padded to batch_size intervention slots plus "
            "an unpatched final control row; saved logits equal single-trial natural "
            "logits plus patched-minus-control batch logits"
        ),
        "anchors": anchor_names,
        "batch_size": batch_size,
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

    def prompt_positions(condition: str, trial: object):
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
        all_positions = _anchor_positions(
            tokenizer,
            prompt,
            condition,
            spans,
            messages[0]["content"],
            messages[-1]["content"],
        )
        mapping = dict(zip(ANCHORS, all_positions))
        positions = {}
        for anchor in anchor_names:
            if mapping[anchor] is None:
                raise ValueError(f"Anchor {anchor!r} is absent from {condition}")
            positions[anchor] = int(mapping[anchor])
        input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, [prompt])
        if annotated_ids != input_ids[0].tolist():
            raise RuntimeError("Offset-aware and model tokenizations disagree")
        return prompt, positions, annotated_ids, input_ids, attention_mask, last_indices

    def natural_forward(condition: str, trial: object):
        prompt, positions, token_ids, input_ids, attention_mask, _ = prompt_positions(
            condition, trial
        )
        collector = PositionComponentOutputCollector(
            parts, unique_targets, positions
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
            collector.close()
        logits = result.logits.detach().float().cpu()[0, -1, canonical_ids].numpy()
        audit = {
            anchor: {
                "position": position,
                "token": tokenizer.decode([token_ids[position]]),
            }
            for anchor, position in positions.items()
        }
        return logits, collector.values, prompt, audit

    def patched_batch(
        condition: str,
        source: dict,
        scenarios: list[dict],
        trial: object,
        natural_logits: np.ndarray,
    ):
        prompt, positions, _, _, _, _ = prompt_positions(condition, trial)
        # The final unpatched row controls the model's substantial batch-size
        # numerical drift.  We save the causal delta relative to that matched
        # row, recentered on the single-trial natural logits.  Pad partial
        # chunks so every forward has exactly the same physical batch size;
        # Qwen's bf16 logits otherwise change measurably with batch size.
        if len(scenarios) > batch_size:
            raise ValueError("Patched scenario chunk exceeds fixed batch size")
        padding = batch_size - len(scenarios)
        prompts = [prompt] * (batch_size + 1)
        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        targets_by_row = [
            *[_targets(row["targets"]) for row in scenarios],
            *([[]] * padding),
            [],
        ]
        patcher = BatchedPositionComponentOutputPatcher(
            parts, targets_by_row, source, positions
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
            patcher.close()
        raw_logits = result.logits.detach().float().cpu()[:, -1, canonical_ids].numpy()
        control = raw_logits[-1]
        patched = raw_logits[: len(scenarios)]
        corrected = natural_logits[None, :] + (patched - control[None, :])
        return corrected, prompt

    all_groups = [
        "natural_game",
        "natural_neutral",
        *[row["id"] for row in plan["scenarios"]],
    ]
    audit_path = output / "position_audit.json"
    for completed, trial in enumerate(trials, 1):
        pending = {
            group
            for group in all_groups
            if not shard_path(output, group, trial.question_id).exists()
        }
        if not pending:
            continue
        game_logits, game_source, game_prompt, game_audit = natural_forward(
            "incorrect", trial
        )
        neutral_logits, neutral_source, neutral_prompt, neutral_audit = natural_forward(
            "neutral", trial
        )
        sources = {"incorrect": game_source, "neutral": neutral_source}
        natural_logits_by_condition = {
            "incorrect": game_logits,
            "neutral": neutral_logits,
        }
        if completed == 1 and not audit_path.exists():
            audit_path.write_text(
                json.dumps(
                    {
                        "question_id": trial.question_id,
                        "incorrect": game_audit,
                        "neutral": neutral_audit,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        for group, condition, logits, prompt in (
            ("natural_game", "incorrect", game_logits, game_prompt),
            ("natural_neutral", "neutral", neutral_logits, neutral_prompt),
        ):
            if group in pending:
                atomic_save_npz(
                    shard_path(output, group, trial.question_id),
                    final_canonical_logits=logits.astype(np.float32),
                    metadata=json_array(
                        {
                            "question_id": trial.question_id,
                            "scenario_id": group,
                            "target_condition": condition,
                            "prompt_hash": prompt_hash(prompt),
                        }
                    ),
                )
        for (target_condition, source_condition), scenarios in grouped.items():
            needed = [row for row in scenarios if row["id"] in pending]
            for chunk in _chunks(needed, batch_size):
                logits, prompt = patched_batch(
                    target_condition,
                    sources[source_condition],
                    chunk,
                    trial,
                    natural_logits_by_condition[target_condition],
                )
                for row, values in zip(chunk, logits):
                    atomic_save_npz(
                        shard_path(output, row["id"], trial.question_id),
                        final_canonical_logits=values.astype(np.float32),
                        metadata=json_array(
                            {
                                "question_id": trial.question_id,
                                "scenario_id": row["id"],
                                "source_condition": source_condition,
                                "target_condition": target_condition,
                                "targets": row["targets"],
                                "prompt_hash": prompt_hash(prompt),
                            }
                        ),
                    )
        if completed == 1 or completed % 5 == 0 or completed == len(trials):
            print(
                f"position component patching: completed {completed}/{len(trials)}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired component-output replacements at semantic positions"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    run(
        SublayerExperimentConfig.load(args.config),
        Path(args.plan),
        Path(args.output),
        args.batch_size,
    )


if __name__ == "__main__":
    main()
