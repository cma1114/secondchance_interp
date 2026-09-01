from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .collect_remapped_behavior import _messages, _remap_question
from .config import ExperimentConfig
from .downstream_source_intervention import BatchedGDNSourceWriteAblator
from .io import atomic_save_npz
from .jlens_collect import _token_offsets
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import present_question, prompt_hash
from .run_first_decision_cross_order_patching import (
    CONDITIONS,
    LETTERS,
    _aggregate_logits,
    _question_ids,
)
from .run_historical_answer_intervention import _forward


SCENARIOS = (
    "first_question_content",
    "first_options",
    "first_answer_boundary",
    "content_plus_answer_boundary",
)


def _overlap(offset: tuple[int, int], interval: tuple[int, int]) -> bool:
    left, right = offset
    start, end = interval
    return right > left and right > start and left < end


def _find_after(prompt: str, text: str, start: int) -> tuple[int, int]:
    index = prompt.find(text, start)
    if index < 0:
        raise RuntimeError(f"Could not locate prompt text after {start}: {text!r}")
    return index, index + len(text)


def _source_positions(
    tokenizer: Any,
    prompt: str,
    messages: list[dict[str, str]],
    question: dict[str, Any],
) -> dict[str, list[int]]:
    offsets = _token_offsets(tokenizer, prompt)
    system = _find_after(prompt, messages[0]["content"], 0)
    first_user = _find_after(prompt, messages[1]["content"], system[1])
    second_user = _find_after(prompt, messages[3]["content"], first_user[1])
    question_range = _find_after(prompt, present_question(question), first_user[0])
    if question_range[1] > first_user[1]:
        raise RuntimeError("First question escaped the first user turn")

    content = [
        index for index, offset in enumerate(offsets)
        if _overlap(offset, question_range)
    ]
    options = []
    for letter in LETTERS:
        option_range = _find_after(
            prompt, f"  {letter}: {question['options'][letter]}", question_range[0]
        )
        if option_range[1] > question_range[1]:
            raise RuntimeError(f"First option {letter} escaped the question")
        options.extend(
            index for index, offset in enumerate(offsets)
            if _overlap(offset, option_range)
        )
    options = sorted(set(options))

    first_user_tokens = [
        index for index, offset in enumerate(offsets)
        if _overlap(offset, first_user)
    ]
    second_user_tokens = [
        index for index, offset in enumerate(offsets)
        if _overlap(offset, second_user)
    ]
    if not first_user_tokens or not second_user_tokens:
        raise RuntimeError("Could not identify user-turn token boundaries")
    boundary = list(
        range(max(first_user_tokens) + 1, min(second_user_tokens))
    )
    if not content or not options or not boundary:
        raise RuntimeError("One or more first-presentation source spans is empty")
    return {
        "first_question_content": content,
        "first_options": options,
        "first_answer_boundary": boundary,
        "content_plus_answer_boundary": sorted(set(content + boundary)),
    }


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        arrays = dict(np.load(path, allow_pickle=False))
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Question IDs changed")
        if arrays["scenario_ids"].astype(str).tolist() != list(SCENARIOS):
            raise ValueError("Scenario IDs changed")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "scenario_ids": np.asarray(SCENARIOS),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "ablated_logits": np.full((2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32),
    }


def run(
    config_path: Path,
    plan_path: Path,
    second_mapping_plan_path: Path,
    output: Path,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires the historical batch-size-4 SDPA regime")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")

    qids = _question_ids(plan_path)
    qid_set = set(qids)
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    all_qids = [row["id"] for row in manifest["questions"]]
    second_plan = {
        row["question_id"]: row
        for row in json.loads(second_mapping_plan_path.read_text())["rows"]
    }
    if not set(all_qids) <= set(second_plan):
        raise ValueError("Second-presentation mapping plan is incomplete")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    gla_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if len(gla_layers) != 48:
        raise RuntimeError(f"Expected 48 GLA layers, found {len(gla_layers)}")

    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.npz"
    arrays = _initialize(result_path, qids)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = output / "prompt_audit.json"

    for group_start in range(0, len(all_qids), config.batch_size):
        group_qids = all_qids[group_start : group_start + config.batch_size]
        targets = [
            qid for qid in group_qids
            if qid in qid_set and not bool(arrays["completed"][qid_index[qid]])
        ]
        if not targets:
            continue

        condition_batches: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            messages = []
            prompts = []
            unpadded_spans = []
            token_rows = []
            for qid in group_qids:
                remapped = _remap_question(
                    questions[qid], second_plan[qid]["new_to_original"]
                )
                row_messages = _messages(
                    config, questions[qid], remapped, condition
                )
                prompt = render_chat(
                    processor,
                    row_messages,
                    config.disable_thinking,
                    config.chat_serialization,
                )
                ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                messages.append(row_messages)
                prompts.append(prompt)
                token_rows.append([int(value) for value in ids])
                unpadded_spans.append(
                    _source_positions(tokenizer, prompt, row_messages, questions[qid])
                )
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
            width = int(input_ids.shape[1])
            physical_spans = []
            for row, (ids, spans) in enumerate(zip(token_rows, unpadded_spans)):
                left_pad = width - len(ids)
                if input_ids[row, left_pad:].tolist() != ids:
                    raise RuntimeError("Historical cohort tokenization changed a prompt")
                physical_spans.append(
                    {
                        scenario: [left_pad + position for position in positions]
                        for scenario, positions in spans.items()
                    }
                )
            natural_output = _forward(model, parts, input_ids, attention_mask)
            natural_logits = _aggregate_logits(
                natural_output.logits[:, -1].float(), variant_ids
            ).detach().cpu().numpy()
            condition_batches[condition] = {
                "messages": messages,
                "prompts": prompts,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "spans": physical_spans,
                "natural_logits": natural_logits,
            }

        for qid in targets:
            qi = qid_index[qid]
            row = group_qids.index(qid)
            for condition_index, condition in enumerate(CONDITIONS):
                batch = condition_batches[condition]
                arrays["natural_logits"][condition_index, qi] = batch[
                    "natural_logits"
                ][row]
                for scenario_index, scenario in enumerate(SCENARIOS):
                    ablator = BatchedGDNSourceWriteAblator(
                        parts, {row: batch["spans"][row][scenario]}
                    )
                    try:
                        ablated_output = _forward(
                            model,
                            parts,
                            batch["input_ids"],
                            batch["attention_mask"],
                        )
                        ablator.assert_fired()
                    finally:
                        ablator.close()
                    ablated = _aggregate_logits(
                        ablated_output.logits[row, -1].float(), variant_ids
                    ).detach().cpu().numpy()
                    arrays["ablated_logits"][
                        condition_index, scenario_index, qi
                    ] = ablated

                if not audit_path.exists():
                    audit_path.write_text(
                        json.dumps(
                            {
                                "question_id": qid,
                                "condition": condition,
                                "historical_group_qids": group_qids,
                                "target_row": row,
                                "prompt_hash": prompt_hash(batch["prompts"][row]),
                                "source_positions_physical": batch["spans"][row],
                                "source_tokens": {
                                    scenario: tokenizer.convert_ids_to_tokens(
                                        batch["input_ids"][
                                            row, batch["spans"][row][scenario]
                                        ].tolist()
                                    )
                                    for scenario in SCENARIOS
                                },
                                "rendered_prompt": batch["prompts"][row],
                            },
                            indent=2,
                            sort_keys=True,
                        ) + "\n"
                    )

            arrays["completed"][qi] = True
            atomic_save_npz(result_path, **arrays)
            done = int(arrays["completed"].sum())
            if done == 1 or done % 5 == 0 or done == len(qids):
                print(f"first-span GLA ablation: {done}/{len(qids)}", flush=True)

    metadata = {
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "second_mapping_plan_path": str(second_mapping_plan_path),
        "scenario_ids": list(SCENARIOS),
        "gla_layers_zero_based": gla_layers,
        "n_questions": len(qids),
        "intervention": (
            "Set beta=0 at the selected first-presentation source positions in all "
            "48 Gated DeltaNet layers for only the target row, preserving its exact "
            "historical four-question cohort."
        ),
        "resolved_answer_tokens": resolved,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--second-mapping-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.plan, args.second_mapping_plan, args.output)


if __name__ == "__main__":
    main()
