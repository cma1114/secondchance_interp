from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .modeling import (
    get_tokenizer,
    forward_runtime_kwargs,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import (
    build_factorial_messages,
    build_messages,
    prompt_hash,
    repeated_question_turn,
)


SCENARIOS = (
    "baseline",
    "incorrect_again_nonremapped",
    "lost_again_nonremapped",
    "incorrect_again_remapped",
    "lost_again_remapped",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _remap_question(question: dict[str, Any], new_to_original: dict[str, str]) -> dict[str, Any]:
    if set(new_to_original) != set(LETTERS) or set(new_to_original.values()) != set(LETTERS):
        raise ValueError("Remapping must be a permutation of A-D")
    if any(new_to_original[letter] == letter for letter in LETTERS):
        raise ValueError("Behavioral-gate remappings must move every option")
    remapped = copy.deepcopy(question)
    remapped["options"] = {
        new: question["options"][original] for new, original in new_to_original.items()
    }
    original_to_new = {original: new for new, original in new_to_original.items()}
    remapped["correct_answer"] = original_to_new[question["correct_answer"]]
    return remapped


def _second_chance_messages(
    question: dict[str, Any],
    condition: str,
    remapped: dict[str, Any] | None,
) -> list[dict[str, str]]:
    messages = build_factorial_messages(
        question,
        condition,
        prompt_mode="baseline_matched_empty_history",
    )
    if remapped is None:
        return messages
    original_repeat = repeated_question_turn(question)
    if not messages[-1]["content"].endswith(original_repeat):
        raise RuntimeError("Could not locate the repeated question in the final user turn")
    messages[-1]["content"] = (
        messages[-1]["content"][: -len(original_repeat)]
        + repeated_question_turn(remapped)
    )
    return messages


def _scenario_messages(
    scenario: str,
    question: dict[str, Any],
    new_to_original: dict[str, str],
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    if scenario in {"baseline", "baseline_duplicate"}:
        return build_messages(question, "baseline"), None
    condition = "incorrect_again" if scenario.startswith("incorrect") else "lost_again"
    remapped = (
        _remap_question(question, new_to_original)
        if scenario.endswith("_remapped") and not scenario.endswith("_nonremapped")
        else None
    )
    return _second_chance_messages(question, condition, remapped), remapped


def _assert_prompt_pair(game: str, neutral: str) -> None:
    game_feedback = "Your answer was incorrect. Choose the answer again."
    neutral_feedback = "Your answer was lost. Choose the answer again."
    expected = game.replace(game_feedback, neutral_feedback, 1)
    if expected != neutral or game.count(game_feedback) != 1 or neutral.count(neutral_feedback) != 1:
        raise RuntimeError(
            "Game and Neutral rendered prompts do not differ only at incorrect/lost"
        )


def collect(
    config_path: Path,
    remapping_plan_path: Path,
    output_path: Path,
    include_duplicate_control: bool,
) -> None:
    import torch

    config = ExperimentConfig.load(config_path)
    if config.model_loader not in {"causal_lm", "multimodal"} or config.chat_serialization not in {
        "hf_template", "hf_template_direct_assistant"
    }:
        raise ValueError(
            "Cross-model behavioral gate requires a supported causal text or "
            "multimodal loader plus the native HF template"
        )
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    ordered_qids = [row["id"] for row in manifest["questions"]]
    if config.question_ids is not None:
        wanted = set(config.question_ids)
        ordered_qids = [qid for qid in ordered_qids if qid in wanted]
        if set(ordered_qids) != wanted:
            raise ValueError("Configured question IDs are not all present in the manifest")
    if config.max_questions is not None:
        ordered_qids = ordered_qids[: config.max_questions]

    plan = json.loads(remapping_plan_path.read_text())
    plan_rows = {row["question_id"]: row for row in plan["rows"]}
    if not set(ordered_qids) <= set(plan_rows):
        raise ValueError("Remapping plan is missing requested questions")
    for qid in ordered_qids:
        _remap_question(questions[qid], plan_rows[qid]["new_to_original"])

    scenarios = list(SCENARIOS)
    if include_duplicate_control:
        scenarios.append("baseline_duplicate")
    if output_path.exists():
        payload = json.loads(output_path.read_text())
        if payload["model_id"] != config.model_id or payload["model_revision"] != config.model_revision:
            raise ValueError("Existing checkpoint belongs to a different model revision")
    else:
        payload = {
            "schema_version": 1,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "manifest_path": config.manifest_path,
            "remapping_plan": str(remapping_plan_path),
            "config": config.as_dict(),
            "scenarios": {name: {} for name in scenarios},
            "prompt_audit": {},
            "timing": {"model_load_seconds": None, "forward_batches": []},
            "complete": False,
        }
    for scenario in scenarios:
        payload["scenarios"].setdefault(scenario, {})
    if all(len(payload["scenarios"][scenario]) == len(ordered_qids) for scenario in scenarios):
        print("All requested scenarios are already complete", flush=True)
        return

    load_start = time.monotonic()
    model, processor, parts = load_model_and_processor(config)
    payload["timing"]["model_load_seconds"] = time.monotonic() - load_start
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]}) for letter in LETTERS
    }
    all_ad_ids = sorted({token_id for ids in variant_ids.values() for token_id in ids})
    device = model_input_device(parts)
    print(f"MODEL_LOADED seconds={payload['timing']['model_load_seconds']:.3f}", flush=True)

    total_cohorts = (len(ordered_qids) + config.batch_size - 1) // config.batch_size
    for cohort_index, batch_qids in enumerate(_chunks(ordered_qids, config.batch_size), 1):
        for scenario in scenarios:
            pending_qids = [qid for qid in batch_qids if qid not in payload["scenarios"][scenario]]
            if not pending_qids:
                continue
            messages_and_remaps = [
                _scenario_messages(
                    scenario,
                    questions[qid],
                    plan_rows[qid]["new_to_original"],
                )
                for qid in pending_qids
            ]
            messages = [item[0] for item in messages_and_remaps]
            remapped_questions = [item[1] for item in messages_and_remaps]
            prompts = [
                render_chat(
                    processor,
                    row,
                    config.disable_thinking,
                    config.chat_serialization,
                    config.chat_template_kwargs,
                )
                for row in messages
            ]
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            forward_start = time.monotonic()
            with torch.inference_mode():
                kwargs = {
                    "input_ids": input_ids.to(device),
                    "attention_mask": attention_mask.to(device),
                    "return_dict": True,
                }
                kwargs.update(forward_runtime_kwargs(model, input_ids, device))
                try:
                    output = model(**kwargs, logits_to_keep=1)
                except TypeError:
                    output = model(**kwargs)
            duration = time.monotonic() - forward_start
            logits = output.logits.detach().float().cpu()
            final_logits = logits[:, 0] if logits.shape[1] == 1 else logits[np.arange(len(pending_qids)), last_indices]
            if not bool(torch.isfinite(final_logits).all()):
                raise RuntimeError(f"Non-finite logits in {scenario} cohort {cohort_index}")
            probabilities = torch.softmax(final_logits, dim=-1)
            top_values, top_ids = torch.topk(probabilities, k=10, dim=-1)
            payload["timing"]["forward_batches"].append({
                "cohort": cohort_index,
                "scenario": scenario,
                "questions": len(pending_qids),
                "seconds": duration,
            })

            for index, (qid, remapped, message, prompt) in enumerate(
                zip(pending_qids, remapped_questions, messages, prompts)
            ):
                aggregated = torch.stack([
                    torch.logsumexp(final_logits[index, variant_ids[letter]], dim=0)
                    for letter in LETTERS
                ])
                token_ids = [int(value) for value in top_ids[index].tolist()]
                top_token = tokenizer.decode([token_ids[0]])
                unrestricted_answer = top_token.strip()
                if unrestricted_answer not in LETTERS:
                    unrestricted_answer = None
                is_remapped = scenario.endswith("_remapped") and not scenario.endswith("_nonremapped")
                mapping = plan_rows[qid]["new_to_original"] if is_remapped else {
                    letter: letter for letter in LETTERS
                }
                payload["scenarios"][scenario][qid] = {
                    "question_id": qid,
                    "correct_original_content": questions[qid]["correct_answer"],
                    "new_to_original": mapping,
                    "original_to_new": {original: new for new, original in mapping.items()},
                    "remapped_correct_letter": None if remapped is None else remapped["correct_answer"],
                    "answer_new_letter": unrestricted_answer,
                    "answer_original_content": None if unrestricted_answer is None else mapping[unrestricted_answer],
                    "aggregated_ad_answer_new_letter": LETTERS[int(aggregated.argmax())],
                    "aggregated_ad_logits": [float(value) for value in aggregated.tolist()],
                    "full_vocab_top_token_id": token_ids[0],
                    "full_vocab_top_token": top_token,
                    "full_vocab_top10": [
                        {
                            "rank": rank + 1,
                            "token_id": token_id,
                            "token": tokenizer.decode([token_id]),
                            "probability": float(top_values[index, rank]),
                        }
                        for rank, token_id in enumerate(token_ids)
                    ],
                    "ad_probability_mass": float(probabilities[index, all_ad_ids].sum()),
                    "all_logits_finite": True,
                    "prompt_hash": prompt_hash(prompt),
                    "prompt_token_count": int(attention_mask[index].sum()),
                }

            if not payload["prompt_audit"] and cohort_index == 1:
                audit_messages = {}
                audit_prompts = {}
                qid = pending_qids[0]
                for name in SCENARIOS:
                    m, _ = _scenario_messages(name, questions[qid], plan_rows[qid]["new_to_original"])
                    audit_messages[name] = m
                    audit_prompts[name] = render_chat(
                        processor,
                        m,
                        config.disable_thinking,
                        config.chat_serialization,
                        config.chat_template_kwargs,
                    )
                _assert_prompt_pair(
                    audit_prompts["incorrect_again_nonremapped"],
                    audit_prompts["lost_again_nonremapped"],
                )
                _assert_prompt_pair(
                    audit_prompts["incorrect_again_remapped"],
                    audit_prompts["lost_again_remapped"],
                )
                payload["prompt_audit"] = {
                    "question_id": qid,
                    "messages": audit_messages,
                    "rendered_prompts": audit_prompts,
                    "game_neutral_only_difference": "incorrect/lost",
                    "answer_token_variants": {
                        letter: [{"text": text, "token_id": token_id} for text, token_id in values]
                        for letter, values in resolved.items()
                    },
                }
            _write_json(output_path, payload)
            print(
                f"FORWARD cohort={cohort_index}/{total_cohorts} scenario={scenario} "
                f"questions={len(pending_qids)} seconds={duration:.3f}",
                flush=True,
            )

        complete_qids = set(ordered_qids)
        for scenario in scenarios:
            complete_qids &= set(payload["scenarios"][scenario])
        print(
            f"PROGRESS completed_cohorts={cohort_index}/{total_cohorts} "
            f"completed_questions={len(complete_qids)}/{len(ordered_qids)}",
            flush=True,
        )

    payload["complete"] = all(
        len(payload["scenarios"][scenario]) == len(ordered_qids) for scenario in scenarios
    )
    payload["n_questions"] = len(ordered_qids)
    if include_duplicate_control:
        errors = []
        for qid in ordered_qids:
            left = np.asarray(payload["scenarios"]["baseline"][qid]["aggregated_ad_logits"])
            right = np.asarray(payload["scenarios"]["baseline_duplicate"][qid]["aggregated_ad_logits"])
            errors.append(float(np.max(np.abs(left - right))))
        payload["duplicate_baseline_max_absolute_ad_logit_error"] = max(errors, default=0.0)
    _write_json(output_path, payload)
    print(f"COMPLETE questions={len(ordered_qids)} output={output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clean cross-model behavioral gate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-duplicate-control", action="store_true")
    args = parser.parse_args()
    collect(args.config, args.remapping_plan, args.output, args.include_duplicate_control)


if __name__ == "__main__":
    main()
