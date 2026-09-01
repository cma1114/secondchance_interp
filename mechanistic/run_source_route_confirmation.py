from __future__ import annotations

import argparse
import json
import platform
import sys
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
from .source_partition import prompt_source_partition
from .source_route_collectors import (
    AttentionSourceRouteCollector,
    DeltaNetSourceRouteCollector,
)
from .sublayer import (
    BatchedRowSourcePositionComponentOutputPatcher,
    PositionComponentOutputCollector,
    PositionComponentTarget,
)
from .sublayer_config import SublayerExperimentConfig


ATTENTION_LAYER = 55
GDN_LAYER = 62
CONDITION_LABEL = {"incorrect": "game", "neutral": "neutral"}


def _targets(values: list[dict]) -> list[PositionComponentTarget]:
    return [
        PositionComponentTarget(
            layer=int(value["layer"]),
            kind=str(value["kind"]),
            anchor=str(value["anchor"]),
        )
        for value in values
    ]


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


def _chunks(values: list[dict], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run(
    config: SublayerExperimentConfig,
    plan_path: Path,
    output: Path,
    batch_size_override: int | None = None,
    max_questions: int | None = None,
) -> None:
    import torch
    import transformers

    if config.attn_implementation != "eager":
        raise ValueError("Source-route confirmation requires eager attention")
    plan = json.loads(plan_path.read_text())
    qids = plan["question_ids"]
    routes = []
    for index, value in enumerate(plan["selected_routes"], 1):
        route = dict(value)
        route["id"] = f"route_{index:02d}"
        routes.append(route)
    if not routes:
        raise ValueError("No selected source routes in confirmation plan")
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        qids,
        None,
    )
    if max_questions is not None:
        trials = trials[: int(max_questions)]
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    component_targets = _targets(plan["component_targets"])
    anchor_names = list(dict.fromkeys(target.anchor for target in component_targets))
    unknown = sorted(set(anchor_names) - set(ANCHORS))
    if unknown:
        raise ValueError(f"Unknown semantic anchors: {unknown}")
    batch_size = int(batch_size_override or plan.get("batch_size", 8))
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    source_names = list(dict.fromkeys(route["source"] for route in routes))

    output.mkdir(parents=True, exist_ok=True)
    scenario_metadata = {
        route["id"]: {
            key: route[key]
            for key in ("kind", "component", "source", "head")
        }
        for route in routes
    }
    groups = [
        "natural_baseline",
        "natural_game",
        "natural_neutral",
        *[
            f"{direction}__{suffix}"
            for direction in ("game_into_neutral", "neutral_into_game")
            for suffix in ("full", *scenario_metadata)
        ],
    ]
    metadata = {
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "routes": scenario_metadata,
        "component_targets": plan["component_targets"],
        "intervention": (
            "patch all eight paired source-condition component outputs, then "
            "remove one selected source/head route from Mixer 56 or Mixer 63"
        ),
        "batch_control": (
            "fixed intervention batch plus an unpatched control row; causal "
            "deltas recentered on single-trial natural target logits"
        ),
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
        return (
            messages,
            prompt,
            positions,
            annotated_ids,
            input_ids,
            attention_mask,
            last_indices,
        )

    def baseline_forward(trial: object):
        messages = build_messages(trial.question, "baseline", config.prompt_mode)
        prompt = render_chat(
            processor,
            messages,
            config.disable_thinking,
            config.chat_serialization,
        )
        input_ids, attention_mask, _ = tokenize_batch(tokenizer, [prompt])
        result = _forward(model, parts, input_ids, attention_mask)
        logits = result.logits.detach().float().cpu()[0, -1, canonical_ids].numpy()
        del result
        return logits, prompt

    def source_forward(condition: str, trial: object):
        (
            messages,
            prompt,
            positions,
            token_ids,
            input_ids,
            attention_mask,
            _,
        ) = prompt_positions(condition, trial)
        partition_ids, all_sources = prompt_source_partition(
            tokenizer, prompt, messages, trial.question, condition
        )
        if partition_ids != token_ids:
            raise RuntimeError("Source-partition and model tokenizations disagree")
        selected_sources = {name: all_sources[name] for name in source_names}
        component_collector = PositionComponentOutputCollector(
            parts, component_targets, positions
        )
        attention = AttentionSourceRouteCollector(
            parts, ATTENTION_LAYER, selected_sources, canonical_ids
        )
        gdn = DeltaNetSourceRouteCollector(
            parts, GDN_LAYER, selected_sources, canonical_ids
        )
        try:
            result = _forward(model, parts, input_ids, attention_mask)
            route_deltas = {}
            for route in routes:
                collector = attention if route["kind"] == "attention" else gdn
                route_deltas[route["id"]] = (
                    collector.hidden_delta(route["source"], int(route["head"]))
                    .detach()
                    .to("cpu", dtype=torch.float16)
                )
            route_arrays = {**attention.arrays(), **gdn.arrays()}
        finally:
            gdn.close()
            attention.close()
            component_collector.close()
        logits = result.logits.detach().float().cpu()[0, -1, canonical_ids].numpy()
        audit = {
            anchor: {
                "position": position,
                "token": tokenizer.decode([token_ids[position]]),
            }
            for anchor, position in positions.items()
        }
        del result
        return {
            "logits": logits,
            "component_source": component_collector.values,
            "route_deltas": route_deltas,
            "route_arrays": route_arrays,
            "prompt": prompt,
            "positions": positions,
            "audit": audit,
            "source_positions": selected_sources,
        }

    def patched_batch(
        target_condition: str,
        source_result: dict,
        scenarios: list[dict],
        trial: object,
        natural_target_logits: np.ndarray,
    ):
        _, prompt, positions, _, _, _, _ = prompt_positions(target_condition, trial)
        if len(scenarios) > batch_size:
            raise ValueError("Patched scenario chunk exceeds fixed batch size")
        padding = batch_size - len(scenarios)
        prompts = [prompt] * (batch_size + 1)
        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        sources_by_row = []
        for scenario in scenarios:
            source = dict(source_result["component_source"])
            route = scenario.get("route")
            if route is not None:
                component = route["component"]
                original = source[component]
                delta = source_result["route_deltas"][route["id"]]
                source[component] = original - delta.to(original.dtype).unsqueeze(0)
            sources_by_row.append(source)
        sources_by_row.extend({} for _ in range(padding + 1))
        targets_by_row = [
            *([component_targets] * len(scenarios)),
            *([[]] * padding),
            [],
        ]
        patcher = BatchedRowSourcePositionComponentOutputPatcher(
            parts, targets_by_row, sources_by_row, positions
        )
        try:
            result = _forward(model, parts, input_ids, attention_mask)
        finally:
            patcher.close()
        raw = result.logits.detach().float().cpu()[:, -1, canonical_ids].numpy()
        corrected = natural_target_logits[None, :] + (
            raw[: len(scenarios)] - raw[-1][None, :]
        )
        del result
        return corrected, prompt

    audit_path = output / "position_audit.json"
    for completed, trial in enumerate(trials, 1):
        pending = {
            group
            for group in groups
            if not shard_path(output, group, trial.question_id).exists()
        }
        if not pending:
            continue
        baseline_logits, baseline_prompt = baseline_forward(trial)
        condition_results = {
            condition: source_forward(condition, trial)
            for condition in ("incorrect", "neutral")
        }
        if completed == 1 and not audit_path.exists():
            audit_path.write_text(
                json.dumps(
                    {
                        "question_id": trial.question_id,
                        "incorrect": condition_results["incorrect"]["audit"],
                        "neutral": condition_results["neutral"]["audit"],
                        "selected_source_positions": {
                            condition: value["source_positions"]
                            for condition, value in condition_results.items()
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        natural_rows = (
            ("natural_baseline", "baseline", baseline_logits, baseline_prompt, None),
            (
                "natural_game",
                "incorrect",
                condition_results["incorrect"]["logits"],
                condition_results["incorrect"]["prompt"],
                condition_results["incorrect"],
            ),
            (
                "natural_neutral",
                "neutral",
                condition_results["neutral"]["logits"],
                condition_results["neutral"]["prompt"],
                condition_results["neutral"],
            ),
        )
        for group, condition, logits, prompt, source_result in natural_rows:
            if group not in pending:
                continue
            extras = {}
            if source_result is not None:
                extras = source_result["route_arrays"]
            atomic_save_npz(
                shard_path(output, group, trial.question_id),
                **extras,
                final_canonical_logits=np.asarray(logits, dtype=np.float32),
                metadata=json_array(
                    {
                        "question_id": trial.question_id,
                        "scenario_id": group,
                        "condition": condition,
                        "prompt_hash": prompt_hash(prompt),
                    }
                ),
            )

        directions = (
            ("game_into_neutral", "incorrect", "neutral"),
            ("neutral_into_game", "neutral", "incorrect"),
        )
        for direction, source_condition, target_condition in directions:
            scenarios = [
                {"id": f"{direction}__full", "route": None},
                *[
                    {"id": f"{direction}__{route['id']}", "route": route}
                    for route in routes
                ],
            ]
            needed = [scenario for scenario in scenarios if scenario["id"] in pending]
            for chunk in _chunks(needed, batch_size):
                corrected, target_prompt = patched_batch(
                    target_condition,
                    condition_results[source_condition],
                    chunk,
                    trial,
                    condition_results[target_condition]["logits"],
                )
                for scenario, logits in zip(chunk, corrected):
                    route = scenario["route"]
                    atomic_save_npz(
                        shard_path(output, scenario["id"], trial.question_id),
                        final_canonical_logits=logits.astype(np.float32),
                        metadata=json_array(
                            {
                                "question_id": trial.question_id,
                                "scenario_id": scenario["id"],
                                "source_condition": source_condition,
                                "target_condition": target_condition,
                                "removed_route": None
                                if route is None
                                else scenario_metadata[route["id"]],
                                "prompt_hash": prompt_hash(target_prompt),
                            }
                        ),
                    )
        if completed == 1 or completed % 5 == 0 or completed == len(trials):
            print(
                f"source-route confirmation: completed {completed}/{len(trials)}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm source/head routes conditionally inside eight-component patches"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    run(
        SublayerExperimentConfig.load(args.config),
        Path(args.plan),
        Path(args.output),
        args.batch_size,
        args.max_questions,
    )


if __name__ == "__main__":
    main()
