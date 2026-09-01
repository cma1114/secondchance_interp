#!/usr/bin/env python3
"""Parallel, resumable Second Chance runs for candidate models.

By default this uses the frozen 500-question/option SimpleMC manifest from the Qwen3-235B run; model configs may instead point to another frozen manifest.  It uses
the same API message construction as the historical CapabilitiesTest and
SecondChanceGame classes.  Each request is still made by BaseGameClass; only
the orchestration is parallelized.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from base_game_class import BaseGameClass


REFERENCE_BASELINE = Path(
    "compiled_results_smc/qwen3-235b-a22b-2507_phase1_compiled.json"
)
N_QUESTIONS = 500
LETTERS = ("A", "B", "C", "D")

MODELS = {
    "deepseek_v4_pro": {
        "subject_name": "deepseek-v4-pro",
        "provider": "DeepSeek API",
        "direct_provider": "DeepSeek",
        "output_dir": Path("outputs/reproduction/simplemc_deepseek_v4_pro_official"),
        "compiled_dir": Path("compiled_results_simplemc_deepseek_v4_pro_official"),
    },
    "deepseek_v4_pro_or": {
        "subject_name": "deepseek-v4-pro",
        "provider": "StreamLake",
        "output_dir": Path("outputs/reproduction/simplemc_deepseek_v4_pro"),
        "compiled_dir": Path("compiled_results_simplemc_deepseek_v4_pro"),
    },
    "deepseek_v4_pro_ionstream": {
        "subject_name": "deepseek-v4-pro",
        "provider": "Ionstream",
        "output_dir": Path("outputs/reproduction/simplemc_deepseek_v4_pro_ionstream"),
        "compiled_dir": Path("compiled_results_simplemc_deepseek_v4_pro_ionstream"),
    },
    "mimo25_pro": {
        "subject_name": "mimo-v2.5-pro",
        "provider": "DigitalOcean",
        "output_dir": Path("outputs/reproduction/simplemc_mimo_v2_5_pro"),
        "compiled_dir": Path("compiled_results_simplemc_mimo_v2_5_pro"),
    },
    "glm52": {
        "subject_name": "glm-5.2",
        "provider": "StreamLake",
        "output_dir": Path("outputs/reproduction/simplemc_glm_5_2"),
        "compiled_dir": Path("compiled_results_simplemc_glm_5_2"),
    },
    "kimi26": {
        "subject_name": "kimi-k2.6",
        "provider": "StreamLake",
        "output_dir": Path("outputs/reproduction/simplemc_kimi_k2_6"),
        "compiled_dir": Path("compiled_results_simplemc_kimi_k2_6"),
    },
    "deepseek32": {
        "subject_name": "deepseek-v3.2",
        "provider": "StreamLake",
        "output_dir": Path("outputs/reproduction/simplemc_deepseek_v3_2"),
        "compiled_dir": Path("compiled_results_simplemc_deepseek_v3_2"),
    },
    "deepseek32_triviamc": {
        "subject_name": "deepseek-v3.2",
        "provider": "DigitalOcean",
        "dataset": "TriviaMC",
        "manifest_path": Path("outputs/reproduction/triviamc/stimulus_manifest.json"),
        "output_dir": Path("outputs/reproduction/triviamc_deepseek_v3_2"),
        "compiled_dir": Path("compiled_results_triviamc_deepseek_v3_2"),
    },
    "minimax_m3": {
        "subject_name": "minimax-m3",
        "provider": "Morph",
        "output_dir": Path("outputs/reproduction/simplemc_minimax_m3"),
        "compiled_dir": Path("compiled_results_simplemc_minimax_m3"),
    },
    "minimax_m3_triviamc": {
        "subject_name": "minimax-m3",
        "provider": "Morph",
        "dataset": "TriviaMC",
        "manifest_path": Path("outputs/reproduction/triviamc/stimulus_manifest.json"),
        "output_dir": Path("outputs/reproduction/triviamc_minimax_m3"),
        "compiled_dir": Path("compiled_results_triviamc_minimax_m3"),
    },
    "inkling_small": {
        "subject_name": "inkling-small",
        "provider": "Together",
        "output_dir": Path("outputs/reproduction/simplemc_inkling_small"),
        "compiled_dir": Path("compiled_results_simplemc_inkling_small"),
    },
    "inkling_small_triviamc": {
        "subject_name": "inkling-small",
        "provider": "Together",
        "dataset": "TriviaMC",
        "manifest_path": Path("outputs/reproduction/triviamc/stimulus_manifest.json"),
        "output_dir": Path("outputs/reproduction/triviamc_inkling_small"),
        "compiled_dir": Path("compiled_results_triviamc_inkling_small"),
    },
    "gpt41_triviamc": {
        "subject_name": "gpt-4.1-2025-04-14",
        "provider": "OpenAI API",
        "direct_provider": "OpenAI",
        "dataset": "TriviaMC",
        "manifest_path": Path("outputs/reproduction/triviamc/stimulus_manifest.json"),
        "output_dir": Path("outputs/reproduction/triviamc_gpt_4_1"),
        "compiled_dir": Path("compiled_results_triviamc_gpt_4_1"),
    },
    "llama4_maverick": {
        "subject_name": "llama-4-maverick",
        "provider": "DigitalOcean",
        "output_dir": Path("outputs/reproduction/simplemc_llama4_maverick"),
        "compiled_dir": Path("compiled_results_simplemc_llama4_maverick"),
    },
    "llama4_maverick_triviamc": {
        "subject_name": "llama-4-maverick",
        "provider": "DigitalOcean",
        "dataset": "TriviaMC",
        "manifest_path": Path("outputs/reproduction/triviamc/stimulus_manifest.json"),
        "output_dir": Path("outputs/reproduction/triviamc_llama4_maverick"),
        "compiled_dir": Path("compiled_results_triviamc_llama4_maverick"),
    },
    "llama33_70b": {
        "subject_name": "llama-3.3-70b-instruct",
        "provider": "Novita",
        "output_dir": Path("outputs/reproduction/simplemc_llama33_70b"),
        "compiled_dir": Path("compiled_results_simplemc_llama33_70b"),
    },
    "llama33_70b_triviamc": {
        "subject_name": "llama-3.3-70b-instruct",
        "provider": "Novita",
        "dataset": "TriviaMC",
        "manifest_path": Path("outputs/reproduction/triviamc/stimulus_manifest.json"),
        "output_dir": Path("outputs/reproduction/triviamc_llama33_70b"),
        "compiled_dir": Path("compiled_results_triviamc_llama33_70b"),
    },
    "nemotron3_super": {
        "subject_name": "nemotron-3-super-120b-a12b",
        "provider": "DigitalOcean",
        "output_dir": Path("outputs/reproduction/simplemc_nemotron3_super"),
        "compiled_dir": Path("compiled_results_simplemc_nemotron3_super"),
    },
    "nemotron3_super_triviamc": {
        "subject_name": "nemotron-3-super-120b-a12b",
        "provider": "DigitalOcean",
        "dataset": "TriviaMC",
        "manifest_path": Path("outputs/reproduction/triviamc/stimulus_manifest.json"),
        "output_dir": Path("outputs/reproduction/triviamc_nemotron3_super"),
        "compiled_dir": Path("compiled_results_triviamc_nemotron3_super"),
    },
    "deepseek_v4_flash": {
        "subject_name": "deepseek-v4-flash",
        "provider": "DigitalOcean",
        "output_dir": Path("outputs/reproduction/simplemc_deepseek_v4_flash"),
        "compiled_dir": Path("compiled_results_simplemc_deepseek_v4_flash"),
    },
    "deepseek_v4_flash_triviamc": {
        "subject_name": "deepseek-v4-flash",
        "provider": "DigitalOcean",
        "dataset": "TriviaMC",
        "manifest_path": Path("outputs/reproduction/triviamc/stimulus_manifest.json"),
        "output_dir": Path("outputs/reproduction/triviamc_deepseek_v4_flash"),
        "compiled_dir": Path("compiled_results_triviamc_deepseek_v4_flash"),
    },
    "qwen35_122b_a10b": {
        "subject_name": "qwen3.5-122b-a10b",
        "provider": "Novita",
        "output_dir": Path("outputs/reproduction/simplemc_qwen35_122b_a10b"),
        "compiled_dir": Path("compiled_results_simplemc_qwen35_122b_a10b"),
    },
    "qwen35_397b_a17b": {
        "subject_name": "qwen3.5-397b-a17b",
        "provider": "Alibaba",
        "output_dir": Path("outputs/reproduction/simplemc_qwen35_397b_a17b"),
        "compiled_dir": Path("compiled_results_simplemc_qwen35_397b_a17b"),
    },
    "qwen35_397b_a17b_triviamc": {
        "subject_name": "qwen3.5-397b-a17b",
        "provider": "Alibaba",
        "dataset": "TriviaMC",
        "manifest_path": Path("outputs/reproduction/triviamc/stimulus_manifest.json"),
        "output_dir": Path("outputs/reproduction/triviamc_qwen35_397b_a17b"),
        "compiled_dir": Path("compiled_results_triviamc_qwen35_397b_a17b"),
    },
    "qwen35_397b_a17b_popmc": {
        "subject_name": "qwen3.5-397b-a17b",
        "provider": "Alibaba",
        "dataset": "PopMC",
        "manifest_path": Path("outputs/reproduction/popmc/stimulus_manifest.json"),
        "output_dir": Path("outputs/reproduction/popmc_qwen35_397b_a17b"),
        "compiled_dir": Path("compiled_results_popmc_qwen35_397b_a17b"),
    },
    "gemma4_26b_a4b_it": {
        "subject_name": "gemma-4-26b-a4b-it",
        "provider": "Novita",
        "output_dir": Path("outputs/reproduction/simplemc_gemma4_26b_a4b_it"),
        "compiled_dir": Path("compiled_results_simplemc_gemma4_26b_a4b_it"),
    },
    "gemma4_31b_it": {
        "subject_name": "gemma-4-31b-it",
        "provider": "CoreWeave",
        "output_dir": Path("outputs/reproduction/simplemc_gemma4_31b_it"),
        "compiled_dir": Path("compiled_results_simplemc_gemma4_31b_it"),
    },
}

_thread_local = threading.local()


def read_json(path: Path, default: object) -> object:
    if not path.exists():
        return copy.deepcopy(default)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def load_questions(config: dict) -> list[dict]:
    manifest_path = config.get("manifest_path")
    if manifest_path:
        manifest = read_json(manifest_path, {})
        questions = manifest.get("questions", [])
    else:
        reference = read_json(REFERENCE_BASELINE, {})
        questions = [trial["question"] for trial in reference["results"].values()]
    if len(questions) != N_QUESTIONS:
        raise RuntimeError(f"Expected {N_QUESTIONS} frozen questions; got {len(questions)}")
    return questions


def game_for_thread(subject_name: str) -> BaseGameClass:
    game = getattr(_thread_local, "game", None)
    if game is None or game.subject_name != subject_name:
        game = BaseGameClass(
            f"{subject_name}_parallel_worker", subject_name, False, None
        )
        _thread_local.game = game
    return game


def capability_user_prompt(game: BaseGameClass, question: dict) -> str:
    setup = (
        "I'm going to ask you a series of multiple choice questions. For each one, "
        "select the answer you think is best. Respond only with the letter of your "
        "choice; do NOT output any other text."
    )
    q_text = game._present_question(question)
    return setup + "\n\n" + q_text + "\nYour choice (A, B, C, or D): "


def second_chance_parts(
    game: BaseGameClass, question: dict, condition: str
) -> tuple[str, list[dict], str]:
    q_text = game._present_question(question)
    setup = (
        "I'm going to ask you a series of multiple choice questions. For each one, "
        "select the answer you think is best."
    )
    answer_prompt = (
        q_text
        + "\nRespond only with the letter of your choice; do NOT output any other text."
        + "\nYour choice (A, B, C, or D): "
    )
    history = [
        {"role": "user", "content": setup + "\n\n" + answer_prompt},
        {"role": "assistant", "content": "[redacted]"},
    ]
    if condition == "game":
        user_prompt = (
            "Your answer was incorrect. Choose a different answer.\n\n" + answer_prompt
        )
        setup_text = (
            "Your previous answer was incorrect. Respond ONLY with your new answer\n"
        )
    elif condition == "neutral":
        user_prompt = (
            "Your response was lost in transmission. Choose again.\n\n" + answer_prompt
        )
        setup_text = ""
    else:
        raise ValueError(condition)
    return user_prompt, history, setup_text


def parse_answer(response: str, options: tuple[str, ...] = LETTERS) -> str:
    stripped = response.rstrip(string.whitespace + string.punctuation)
    if not stripped:
        return response
    candidate = stripped.upper()
    if candidate[0] in options:
        return candidate[0]
    if candidate[-1] in options:
        return candidate[-1]
    return response


def raw_answer_consistency_error(call: dict, condition: str) -> str | None:
    """Reject serving responses whose actual text is not the scored A-D answer.

    The historical analysis uses the maximum-probability first token as the
    response. At temperature zero, that must agree with the provider's actual
    completion. A mismatch means the serving endpoint's text and logprobs are
    internally inconsistent; prose output also violates the answer-only task.
    """
    metadata = call.get("call_metadata") or {}
    raw = metadata.get("raw_response_content")
    match = re.fullmatch(r"\s*([ABCD])(?:[.)])?\s*", raw or "", flags=re.IGNORECASE)
    if not match:
        return f"raw response was not answer-only A-D: {raw!r}"
    raw_answer = match.group(1).upper()
    answer_key = "subject_answer" if condition == "baseline" else "new_answer"
    scored_answer = call.get(answer_key)
    if raw_answer != scored_answer:
        return (
            f"raw response {raw_answer!r} disagreed with first-token logprob "
            f"argmax {scored_answer!r}"
        )
    return None


def call_one(
    subject_name: str,
    question: dict,
    condition: str,
    top_logprobs: int,
    baseline_answer: str | None = None,
) -> dict:
    game = game_for_thread(subject_name)
    if condition == "baseline":
        # Inkling Small's hosted chat template consumes four completion tokens
        # while exposing a single visible answer token.  A one-token cap returns
        # content=null and logprobs=null even though the model has generated.
        # Other model families in this reproduction emit the visible A-D token
        # immediately and retain the historical one-token cap.
        baseline_max_tokens = 4 if subject_name == "inkling-small" else 1
        response, _, probabilities = game._get_llm_answer(
            list(LETTERS),
            capability_user_prompt(game, question),
            [],
            keep_appending=False,
            MAX_TOKENS=baseline_max_tokens,
            temp=0.0,
            top_logprobs_count=top_logprobs,
        )
        answer = parse_answer(response)
        return {
            "question": question,
            "subject_answer": answer,
            "is_correct": answer == question["correct_answer"],
            "probs": probabilities,
            "call_metadata": copy.deepcopy(game.last_call_metadata),
        }

    if baseline_answer is None:
        raise ValueError("Second-chance requests require a baseline answer")
    user_prompt, history, setup_text = second_chance_parts(game, question, condition)
    response, _, probabilities = game._get_llm_answer(
        list(LETTERS),
        user_prompt,
        history,
        keep_appending=False,
        setup_text=setup_text,
        MAX_TOKENS=None,
        temp=0.0,
        accept_any=True,
        top_logprobs_count=top_logprobs,
    )
    answer = parse_answer(response)
    return {
        "question": question,
        "original_answer": baseline_answer,
        "new_answer": answer,
        "correct_answer": question["correct_answer"],
        "answer_changed": answer != baseline_answer and answer in LETTERS,
        "is_correct": answer == question["correct_answer"],
        "probs": probabilities,
        "call_metadata": copy.deepcopy(game.last_call_metadata),
    }


def raw_condition_path(output_dir: Path, condition: str) -> Path:
    return output_dir / f"{condition}_results.json"


def run_condition(
    config: dict,
    questions: list[dict],
    condition: str,
    top_logprobs: int,
    workers: int,
    baseline: dict[str, dict] | None = None,
) -> dict[str, dict]:
    path = raw_condition_path(config["output_dir"], condition)
    payload = read_json(path, {"condition": condition, "results": {}})
    results = payload["results"]

    def needs_retry(question: dict) -> bool:
        trial = results.get(question["id"])
        if not isinstance(trial, dict):
            return True
        answer_key = "subject_answer" if condition == "baseline" else "new_answer"
        if trial.get(answer_key) not in LETTERS:
            return True
        if (
            condition != "baseline"
            and baseline is not None
            and trial.get("original_answer")
            != baseline[question["id"]].get("subject_answer")
        ):
            return True
        if raw_answer_consistency_error(trial, condition):
            return True
        metadata = trial.get("call_metadata") or {}
        returned = metadata.get("top_logprobs_returned")
        positive_probabilities = [
            value
            for value in (trial.get("probs") or {}).values()
            if isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        ]
        return bool(
            not isinstance(returned, int)
            or returned < top_logprobs
            or metadata.get("reasoning_present")
            or metadata.get("reasoning_tokens") not in (None, 0)
            or len(positive_probabilities) < 4
        )

    pending = [question for question in questions if needs_retry(question)]
    print(f"{config['subject_name']} {condition}: {len(pending)} pending/{len(questions)}")
    if not pending:
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_question = {}
        for question in pending:
            baseline_answer = (
                baseline[question["id"]]["subject_answer"] if baseline else None
            )
            future = executor.submit(
                call_one,
                config["subject_name"],
                question,
                condition,
                top_logprobs,
                baseline_answer,
            )
            future_to_question[future] = question

        completed_since_save = 0
        for index, future in enumerate(as_completed(future_to_question), 1):
            question = future_to_question[future]
            results[question["id"]] = future.result()
            completed_since_save += 1
            if completed_since_save >= 10:
                payload["updated_at"] = time.time()
                write_json(path, payload)
                completed_since_save = 0
            if index % 25 == 0 or index == len(pending):
                print(
                    f"{config['subject_name']} {condition}: "
                    f"completed {index}/{len(pending)}"
                )
    payload["complete"] = len(results) == len(questions)
    payload["updated_at"] = time.time()
    write_json(path, payload)
    return results


def request_messages_for_audit(game: BaseGameClass, question: dict, condition: str) -> list[dict]:
    if condition == "baseline":
        return [
            {"role": "system", "content": "Output ONLY the letter of your choice: A, B, C, or D.\n"},
            {"role": "user", "content": capability_user_prompt(game, question)},
        ]
    user_prompt, history, setup_text = second_chance_parts(game, question, condition)
    history[-1]["content"] = [
        {
            "type": "text",
            "text": history[-1]["content"],
            "cache_control": {"type": "ephemeral"},
        }
    ]
    setup_prefix = f"{setup_text}\n" if setup_text else ""
    history.append(
        {
            "role": "system",
            "content": setup_prefix + "Output ONLY the letter of your choice: A, B, C, or D.\n",
        }
    )
    history.append({"role": "user", "content": user_prompt})
    return history


def validate_call(call: dict, expected_top_logprobs: int) -> None:
    metadata = call.get("call_metadata") or {}
    returned = metadata.get("top_logprobs_returned")
    if not isinstance(returned, int) or returned < 4:
        raise RuntimeError(f"Insufficient logprob coverage: {metadata}")
    if returned < expected_top_logprobs:
        raise RuntimeError(
            f"Requested {expected_top_logprobs} logprobs but received {returned}: {metadata}"
        )
    if metadata.get("reasoning_present"):
        raise RuntimeError(f"Reasoning was present despite disabling it: {metadata}")
    if metadata.get("reasoning_tokens") not in (None, 0):
        raise RuntimeError(f"Nonzero reasoning-token count: {metadata}")
    positive_probabilities = [
        value
        for value in (call.get("probs") or {}).values()
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    ]
    if len(positive_probabilities) < 4:
        raise RuntimeError(
            "Provider returned fewer than four finite, nonzero token probabilities; "
            "the advertised logprobs are numerically unusable"
        )


def run_preflight(
    config: dict, requested: int, validation_questions: int, provider: str | None
) -> None:
    selected_provider = provider or config["provider"]
    BaseGameClass._provider_override = config.get("direct_provider")
    BaseGameClass._openrouter_provider_only = selected_provider
    BaseGameClass._openrouter_request_interval = 0.0
    questions = load_questions(config)[:validation_questions]
    calls = []
    for question in questions:
        baseline = call_one(config["subject_name"], question, "baseline", requested)
        validate_call(baseline, requested)
        consistency_error = raw_answer_consistency_error(baseline, "baseline")
        if baseline.get("subject_answer") not in LETTERS or consistency_error:
            raise RuntimeError(
                f"Invalid baseline response on {question['id']}: "
                f"{consistency_error or baseline.get('subject_answer')!r}"
            )
        calls.append({"question_id": question["id"], "condition": "baseline", **baseline})
        for condition in ("game", "neutral"):
            result = call_one(
                config["subject_name"],
                question,
                condition,
                requested,
                baseline["subject_answer"],
            )
            validate_call(result, requested)
            consistency_error = raw_answer_consistency_error(result, condition)
            if result.get("new_answer") not in LETTERS or consistency_error:
                raise RuntimeError(
                    f"Invalid {condition} response on {question['id']}: "
                    f"{consistency_error or result.get('new_answer')!r}"
                )
            calls.append({"question_id": question["id"], "condition": condition, **result})

    audit_game = BaseGameClass("prompt_audit", config["subject_name"], False, None)
    preflight = {
        "model": config["subject_name"],
        "provider": selected_provider,
        "requested_top_logprobs": requested,
        "selected_top_logprobs": min(
            call["call_metadata"]["top_logprobs_returned"] for call in calls
        ),
        "reasoning_disabled": True,
        "calls": [
            {
                "question_id": call["question_id"],
                "condition": call["condition"],
                "answer": call.get("subject_answer", call.get("new_answer")),
                "call_metadata": call["call_metadata"],
                "probability_entries": len(call.get("probs") or {}),
            }
            for call in calls
        ],
        "prompt_audit": {
            condition: request_messages_for_audit(audit_game, questions[0], condition)
            for condition in ("baseline", "game", "neutral")
        },
    }
    write_json(config["output_dir"] / "preflight.json", preflight)
    print(json.dumps(preflight, indent=2))


def split_second_chance(
    output_dir: Path,
    condition: str,
    results: dict[str, dict],
    baseline: dict[str, dict],
) -> dict[str, Path]:
    paths = {}
    for correctness, is_correct in (("incorrect", False), ("correct", True)):
        selected = {
            qid: trial
            for qid, trial in results.items()
            if bool(baseline[qid]["is_correct"]) is is_correct
        }
        path = output_dir / f"{condition}_baseline_{correctness}.json"
        write_json(path, {"condition": condition, "results": selected})
        paths[correctness] = path
    return paths


def complete_run(config: dict, workers: int) -> None:
    preflight_path = config["output_dir"] / "preflight.json"
    preflight = read_json(preflight_path, None)
    if not preflight:
        raise RuntimeError(f"Run preflight first: {preflight_path}")
    top_logprobs = int(preflight["selected_top_logprobs"])
    selected_provider = str(preflight["provider"])
    BaseGameClass._provider_override = config.get("direct_provider")
    BaseGameClass._openrouter_provider_only = selected_provider
    BaseGameClass._openrouter_request_interval = 0.0
    questions = load_questions(config)

    baseline = run_condition(
        config, questions, "baseline", top_logprobs, workers
    )
    if len(baseline) != N_QUESTIONS:
        raise RuntimeError(f"Incomplete baseline: {len(baseline)}/{N_QUESTIONS}")

    # Game and neutral are independent after the frozen baseline is available.
    with ThreadPoolExecutor(max_workers=2) as condition_executor:
        futures = {
            condition_executor.submit(
                run_condition,
                config,
                questions,
                condition,
                top_logprobs,
                workers,
                baseline,
            ): condition
            for condition in ("game", "neutral")
        }
        condition_results = {
            futures[future]: future.result() for future in as_completed(futures)
        }
    game = condition_results["game"]
    neutral = condition_results["neutral"]
    if not (len(game) == len(neutral) == N_QUESTIONS):
        raise RuntimeError(
            f"Incomplete second chance data: game={len(game)}, neutral={len(neutral)}"
        )

    # A valid raw A-D answer that disagrees with the scored first-token argmax
    # makes the provider's text and logprobs internally inconsistent and is
    # fatal.  A non-answer raw response is instead an ordinary format
    # exclusion, handled below together with parsed non-A-D responses.
    consistency_failures = {
        condition: sorted(
            qid
            for qid, trial in results.items()
            if "disagreed with first-token logprob argmax"
            in (raw_answer_consistency_error(trial, condition) or "")
        )
        for condition, results in (
            ("baseline", baseline),
            ("game", game),
            ("neutral", neutral),
        )
    }
    if any(consistency_failures.values()):
        raise RuntimeError(
            "Provider returned answer/logprob-inconsistent trials: "
            + json.dumps(consistency_failures)
        )

    def invalid_response(trial: dict, condition: str) -> bool:
        answer_key = "subject_answer" if condition == "baseline" else "new_answer"
        consistency_error = raw_answer_consistency_error(trial, condition)
        metadata = trial.get("call_metadata") or {}
        returned = metadata.get("top_logprobs_returned")
        positive_probabilities = [
            value
            for value in (trial.get("probs") or {}).values()
            if isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        ]
        return bool(
            trial.get(answer_key) not in LETTERS
            or not isinstance(returned, int)
            or returned < top_logprobs
            or len(positive_probabilities) < 4
            or metadata.get("reasoning_present")
            or metadata.get("reasoning_tokens") not in (None, 0)
            or (
                consistency_error
                and consistency_error.startswith("raw response was not answer-only A-D")
            )
        )

    invalid = {
        "baseline": sorted(
            qid
            for qid, trial in baseline.items()
            if invalid_response(trial, "baseline")
        ),
        "game": sorted(
            qid for qid, trial in game.items() if invalid_response(trial, "game")
        ),
        "neutral": sorted(
            qid
            for qid, trial in neutral.items()
            if invalid_response(trial, "neutral")
        ),
    }
    excluded_ids = set().union(*map(set, invalid.values()))
    usable_ids = set(baseline) - excluded_ids
    analysis_baseline = {qid: baseline[qid] for qid in baseline if qid in usable_ids}
    analysis_game = {qid: game[qid] for qid in game if qid in usable_ids}
    analysis_neutral = {qid: neutral[qid] for qid in neutral if qid in usable_ids}
    write_json(
        config["output_dir"] / "format_exclusions.json",
        {
            "n_attempted": N_QUESTIONS,
            "n_usable_for_behavioral_tests": len(usable_ids),
            "definition": "A paired trial is excluded if any condition lacks an answer-only A-D response, the requested usable logprobs, or reasoning-off compliance.",
            "invalid_ids_by_condition": invalid,
            "excluded_union": sorted(excluded_ids),
        },
    )

    compiled_path = (
        config["compiled_dir"] / f"{config['subject_name']}_phase1_compiled.json"
    )
    write_json(
        compiled_path,
        {"condition": "baseline", "results": analysis_baseline},
    )
    game_paths = split_second_chance(
        config["output_dir"], "game", analysis_game, analysis_baseline
    )
    neutral_paths = split_second_chance(
        config["output_dir"], "neutral", analysis_neutral, analysis_baseline
    )
    state = {
        "model": config["subject_name"],
        "dataset": config.get("dataset", "SimpleMC"),
        "n_questions": len(usable_ids),
        "n_attempted": N_QUESTIONS,
        "format_exclusions": len(excluded_ids),
        "temperature": 0.0,
        "top_logprobs": top_logprobs,
        "provider": selected_provider,
        "reasoning_disabled": True,
        "baseline_compiled_path": str(compiled_path),
        "second_chance": {
            "incorrect_baseline_incorrect": {
                "complete": True,
                "path": str(game_paths["incorrect"]),
            },
            "incorrect_baseline_correct": {
                "complete": True,
                "path": str(game_paths["correct"]),
            },
            "neutral_baseline_incorrect": {
                "complete": True,
                "path": str(neutral_paths["incorrect"]),
            },
            "neutral_baseline_correct": {
                "complete": True,
                "path": str(neutral_paths["correct"]),
            },
        },
        "complete": True,
        "completed_at": time.time(),
    }
    write_json(config["output_dir"] / "run_state.json", state)
    print(f"Completed {config['subject_name']}: {config['output_dir']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODELS))
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--requested-top-logprobs", type=int, default=20)
    preflight.add_argument("--validation-questions", type=int, default=3)
    preflight.add_argument("--provider")
    run = subparsers.add_parser("run")
    run.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MODELS[args.model]
    if args.command == "preflight":
        run_preflight(
            config,
            args.requested_top_logprobs,
            args.validation_questions,
            args.provider,
        )
    else:
        complete_run(config, args.workers)


if __name__ == "__main__":
    main()
