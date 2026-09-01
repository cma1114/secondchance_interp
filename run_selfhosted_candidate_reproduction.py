#!/usr/bin/env python3
"""Resumable Second Chance screen against a local OpenAI endpoint.

This deliberately has only one non-stdlib dependency (``openai``), so it can
run inside a vLLM serving container without installing the paper repository's
    full analysis environment. It preserves a frozen 500-question manifest and
    writes the same result/state schema as the API reproduction runners.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


REFERENCE_BASELINE = Path(
    "compiled_results_smc/qwen3-235b-a22b-2507_phase1_compiled.json"
)
LETTERS = ("A", "B", "C", "D")
N_QUESTIONS = 500
MIN_PREFLIGHT_VALID_RATE = 0.90
MIN_PREFLIGHT_CONDITION_RATE = 0.80
MIN_FULL_USABLE_RATE = 0.90
_thread_local = threading.local()


@dataclass(frozen=True)
class RunConfig:
    subject_name: str
    served_model_name: str
    endpoint: str
    output_dir: Path
    compiled_dir: Path
    provider_label: str
    chat_template_kwargs: dict
    dataset: str
    questions_file: Path | None


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


def load_questions(config: RunConfig) -> list[dict]:
    if config.questions_file is not None:
        manifest = read_json(config.questions_file, {})
        questions = manifest["questions"]
    else:
        reference = read_json(REFERENCE_BASELINE, {})
        questions = [trial["question"] for trial in reference["results"].values()]
    if len(questions) != N_QUESTIONS:
        raise RuntimeError(
            f"Expected {N_QUESTIONS} frozen {config.dataset} questions; "
            f"got {len(questions)}"
        )
    return questions


def present_question(question: dict) -> str:
    lines = ["-" * 30, "Question:", question["question"], "-" * 10]
    lines.extend(f"  {key}: {value}" for key, value in question["options"].items())
    lines.append("-" * 30)
    return "\n".join(lines)


def capability_user_prompt(question: dict) -> str:
    setup = (
        "I'm going to ask you a series of multiple choice questions. For each one, "
        "select the answer you think is best. Respond only with the letter of your "
        "choice; do NOT output any other text."
    )
    return (
        setup
        + "\n\n"
        + present_question(question)
        + "\nYour choice (A, B, C, or D): "
    )


def second_chance_messages(question: dict, condition: str) -> list[dict]:
    question_text = present_question(question)
    setup = (
        "I'm going to ask you a series of multiple choice questions. For each one, "
        "select the answer you think is best."
    )
    answer_prompt = (
        question_text
        + "\nRespond only with the letter of your choice; do NOT output any other text."
        + "\nYour choice (A, B, C, or D): "
    )
    if condition == "game":
        current_user = (
            "Your answer was incorrect. Choose a different answer.\n\n" + answer_prompt
        )
        setup_text = (
            "Your previous answer was incorrect. Respond ONLY with your new answer\n"
        )
    elif condition == "neutral":
        current_user = (
            "Your response was lost in transmission. Choose again.\n\n"
            + answer_prompt
        )
        setup_text = ""
    else:
        raise ValueError(condition)

    setup_prefix = f"{setup_text}\n" if setup_text else ""
    system_prompt = (
        setup_prefix + "Output ONLY the letter of your choice: A, B, C, or D.\n"
    )
    # OpenAI-compatible providers treat the system message as conversation setup.
    # Put it in the normal first position while preserving every content string
    # used by the historical BaseGameClass implementation.
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": setup + "\n\n" + answer_prompt},
        {"role": "assistant", "content": "[redacted]"},
        {"role": "user", "content": current_user},
    ]


def baseline_messages(question: dict) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "Output ONLY the letter of your choice: A, B, C, or D.\n",
        },
        {"role": "user", "content": capability_user_prompt(question)},
    ]


def parse_answer(response: str) -> str:
    stripped = response.rstrip(string.whitespace + string.punctuation)
    if not stripped:
        return response
    candidate = stripped.upper()
    if candidate[0] in LETTERS:
        return candidate[0]
    if candidate[-1] in LETTERS:
        return candidate[-1]
    return response


def client_for_thread(config: RunConfig) -> OpenAI:
    key = (config.endpoint, config.served_model_name)
    if getattr(_thread_local, "client_key", None) != key:
        _thread_local.client = OpenAI(
            base_url=config.endpoint,
            api_key="EMPTY",
            timeout=120.0,
            max_retries=2,
        )
        _thread_local.client_key = key
    return _thread_local.client


def model_call(
    config: RunConfig,
    messages: list[dict],
    top_logprobs: int,
) -> tuple[str, dict[str, float], dict]:
    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            completion = client_for_thread(config).chat.completions.create(
                model=config.served_model_name,
                messages=messages,
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=top_logprobs,
                extra_body={"chat_template_kwargs": config.chat_template_kwargs},
            )
            choice = completion.choices[0]
            raw_content = (choice.message.content or "").strip()
            logprob_content = getattr(choice.logprobs, "content", None)
            if not logprob_content:
                raise RuntimeError("Server returned no token logprobs")
            first = logprob_content[0]
            top = first.top_logprobs or []
            if len(top) < top_logprobs:
                raise RuntimeError(
                    f"Server returned {len(top)}/{top_logprobs} top logprobs"
                )
            token_probs = {entry.token: math.exp(entry.logprob) for entry in top}
            scored_token = max(token_probs, key=token_probs.get)
            reasoning = getattr(choice.message, "reasoning", None)
            reasoning_content = getattr(choice.message, "reasoning_content", None)
            usage = completion.usage
            metadata = {
                "top_logprobs_requested": top_logprobs,
                "top_logprobs_returned": len(top),
                "serving_provider": config.provider_label,
                "response_model": getattr(completion, "model", None),
                "reasoning_present": bool(reasoning or reasoning_content),
                "reasoning_tokens": None,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "raw_response_content": raw_content,
                "generated_token_trace": [entry.token for entry in logprob_content],
                "message_roles": [message["role"] for message in messages],
                "system_messages": [
                    message["content"]
                    for message in messages
                    if message["role"] == "system"
                ],
                "chat_template_kwargs": config.chat_template_kwargs,
            }
            return scored_token, token_probs, metadata
        except Exception as exc:
            last_error = exc
            if attempt == 7:
                break
            time.sleep(delay)
            delay = min(delay * 2, 15.0)
    raise RuntimeError(f"Model call failed after retries: {last_error}")


def call_one(
    config: RunConfig,
    question: dict,
    condition: str,
    top_logprobs: int,
    baseline_answer: str | None = None,
) -> dict:
    messages = (
        baseline_messages(question)
        if condition == "baseline"
        else second_chance_messages(question, condition)
    )
    token, probabilities, metadata = model_call(
        config, messages, top_logprobs
    )
    answer = parse_answer(token)
    if condition == "baseline":
        return {
            "question": question,
            "subject_answer": answer,
            "is_correct": answer == question["correct_answer"],
            "probs": probabilities,
            "call_metadata": metadata,
        }
    if baseline_answer is None:
        raise ValueError("Second-chance requests require a baseline answer")
    return {
        "question": question,
        "original_answer": baseline_answer,
        "new_answer": answer,
        "correct_answer": question["correct_answer"],
        "answer_changed": answer != baseline_answer and answer in LETTERS,
        "is_correct": answer == question["correct_answer"],
        "probs": probabilities,
        "call_metadata": metadata,
    }


def consistency_error(call: dict, condition: str) -> str | None:
    metadata = call.get("call_metadata") or {}
    raw = metadata.get("raw_response_content")
    match = re.fullmatch(r"\s*([ABCD])(?:[.)])?\s*", raw or "", re.IGNORECASE)
    if not match:
        return f"raw response was not answer-only A-D: {raw!r}"
    answer_key = "subject_answer" if condition == "baseline" else "new_answer"
    if match.group(1).upper() != call.get(answer_key):
        return "raw response disagreed with first-token logprob argmax"
    return None


def validate_call(
    call: dict,
    condition: str,
    expected: int,
    *,
    require_answer_only: bool = True,
) -> None:
    metadata = call.get("call_metadata") or {}
    if metadata.get("top_logprobs_returned", 0) < expected:
        raise RuntimeError(f"Insufficient logprobs: {metadata}")
    if metadata.get("reasoning_present"):
        raise RuntimeError(f"Reasoning was present: {metadata}")
    error = consistency_error(call, condition)
    if require_answer_only and error:
        raise RuntimeError(error)
    finite = [
        value
        for value in (call.get("probs") or {}).values()
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    ]
    if len(finite) < 4:
        raise RuntimeError("Fewer than four finite nonzero token probabilities")


def run_preflight(config: RunConfig, requested: int, n_questions: int) -> None:
    questions = load_questions(config)[:n_questions]
    calls = []
    for question in questions:
        baseline = call_one(config, question, "baseline", requested)
        validate_call(
            baseline, "baseline", requested, require_answer_only=False
        )
        calls.append((question["id"], "baseline", baseline))
        for condition in ("game", "neutral"):
            result = call_one(
                config,
                question,
                condition,
                requested,
                baseline["subject_answer"],
            )
            validate_call(
                result, condition, requested, require_answer_only=False
            )
            calls.append((question["id"], condition, result))
    format_failures = [
        {
            "question_id": qid,
            "condition": condition,
            "error": consistency_error(call, condition),
            "raw_response": call["call_metadata"].get("raw_response_content"),
        }
        for qid, condition, call in calls
        if consistency_error(call, condition)
    ]
    valid_rate = 1.0 - len(format_failures) / len(calls)
    rates_by_condition = {}
    for condition in ("baseline", "game", "neutral"):
        selected = [item for item in calls if item[1] == condition]
        valid = sum(
            consistency_error(call, condition) is None
            for _, _, call in selected
        )
        rates_by_condition[condition] = valid / len(selected)
    if valid_rate < MIN_PREFLIGHT_VALID_RATE or any(
        rate < MIN_PREFLIGHT_CONDITION_RATE
        for rate in rates_by_condition.values()
    ):
        raise RuntimeError(
            "Preflight answer-only coverage was too low: "
            f"overall={valid_rate:.3f}, by_condition={rates_by_condition}, "
            f"failures={format_failures}"
        )
    preflight = {
        "model": config.subject_name,
        "served_model_name": config.served_model_name,
        "provider": config.provider_label,
        "requested_top_logprobs": requested,
        "selected_top_logprobs": min(
            call["call_metadata"]["top_logprobs_returned"] for _, _, call in calls
        ),
        "reasoning_disabled": True,
        "answer_only_valid_rate": valid_rate,
        "answer_only_valid_rate_by_condition": rates_by_condition,
        "format_failures": format_failures,
        "calls": [
            {
                "question_id": qid,
                "condition": condition,
                "answer": call.get("subject_answer", call.get("new_answer")),
                "call_metadata": call["call_metadata"],
                "probability_entries": len(call.get("probs") or {}),
            }
            for qid, condition, call in calls
        ],
        "prompt_audit": {
            "baseline": baseline_messages(questions[0]),
            "game": second_chance_messages(questions[0], "game"),
            "neutral": second_chance_messages(questions[0], "neutral"),
        },
    }
    write_json(config.output_dir / "preflight.json", preflight)
    print(json.dumps(preflight, indent=2))


def condition_path(config: RunConfig, condition: str) -> Path:
    return config.output_dir / f"{condition}_results.json"


def run_condition(
    config: RunConfig,
    questions: list[dict],
    condition: str,
    top_logprobs: int,
    workers: int,
    baseline: dict[str, dict] | None = None,
) -> dict[str, dict]:
    path = condition_path(config, condition)
    payload = read_json(path, {"condition": condition, "results": {}})
    results = payload["results"]

    def needs_retry(question: dict) -> bool:
        trial = results.get(question["id"])
        if not isinstance(trial, dict):
            return True
        return bool(
            (trial.get("call_metadata") or {}).get("top_logprobs_returned", 0)
            < top_logprobs
        )

    pending = [question for question in questions if needs_retry(question)]
    print(f"{condition}: {len(pending)} pending/{len(questions)}", flush=True)
    if not pending:
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for question in pending:
            baseline_answer = (
                baseline[question["id"]]["subject_answer"] if baseline else None
            )
            future = executor.submit(
                call_one,
                config,
                question,
                condition,
                top_logprobs,
                baseline_answer,
            )
            futures[future] = question
        unsaved = 0
        for index, future in enumerate(as_completed(futures), 1):
            question = futures[future]
            results[question["id"]] = future.result()
            unsaved += 1
            if unsaved >= 10:
                payload["updated_at"] = time.time()
                write_json(path, payload)
                unsaved = 0
            if index % 25 == 0 or index == len(pending):
                print(f"{condition}: completed {index}/{len(pending)}", flush=True)
    payload["complete"] = len(results) == len(questions)
    payload["updated_at"] = time.time()
    write_json(path, payload)
    return results


def split_condition(
    config: RunConfig,
    condition: str,
    results: dict[str, dict],
    baseline: dict[str, dict],
) -> dict[str, Path]:
    paths = {}
    for label, correctness in (("incorrect", False), ("correct", True)):
        selected = {
            qid: trial
            for qid, trial in results.items()
            if bool(baseline[qid]["is_correct"]) is correctness
        }
        path = config.output_dir / f"{condition}_baseline_{label}.json"
        write_json(path, {"condition": condition, "results": selected})
        paths[label] = path
    return paths


def complete_run(config: RunConfig, workers: int) -> None:
    preflight = read_json(config.output_dir / "preflight.json", None)
    if not preflight:
        raise RuntimeError("Run preflight first")
    top_logprobs = int(preflight["selected_top_logprobs"])
    questions = load_questions(config)
    baseline = run_condition(
        config, questions, "baseline", top_logprobs, workers
    )
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
    if not (len(baseline) == len(game) == len(neutral) == N_QUESTIONS):
        raise RuntimeError("Incomplete results")
    for condition, results in (
        ("baseline", baseline),
        ("game", game),
        ("neutral", neutral),
    ):
        for trial in results.values():
            validate_call(
                trial,
                condition,
                top_logprobs,
                require_answer_only=False,
            )

    invalid = {
        "baseline": sorted(
            qid for qid, trial in baseline.items()
            if trial.get("subject_answer") not in LETTERS
            or consistency_error(trial, "baseline")
        ),
        "game": sorted(
            qid for qid, trial in game.items()
            if trial.get("new_answer") not in LETTERS
            or consistency_error(trial, "game")
        ),
        "neutral": sorted(
            qid for qid, trial in neutral.items()
            if trial.get("new_answer") not in LETTERS
            or consistency_error(trial, "neutral")
        ),
    }
    excluded = set().union(*map(set, invalid.values()))
    usable = set(baseline) - excluded
    if len(usable) / N_QUESTIONS < MIN_FULL_USABLE_RATE:
        raise RuntimeError(
            "Full-run answer-only coverage was too low: "
            f"{len(usable)}/{N_QUESTIONS} usable trials"
        )
    baseline = {qid: baseline[qid] for qid in baseline if qid in usable}
    game = {qid: game[qid] for qid in game if qid in usable}
    neutral = {qid: neutral[qid] for qid in neutral if qid in usable}
    write_json(
        config.output_dir / "format_exclusions.json",
        {
            "n_attempted": N_QUESTIONS,
            "n_usable_for_behavioral_tests": len(usable),
            "definition": "A trial is excluded if any generated response is not A-D.",
            "invalid_ids_by_condition": invalid,
            "excluded_union": sorted(excluded),
        },
    )
    compiled_path = (
        config.compiled_dir / f"{config.subject_name}_phase1_compiled.json"
    )
    write_json(compiled_path, {"condition": "baseline", "results": baseline})
    game_paths = split_condition(config, "game", game, baseline)
    neutral_paths = split_condition(config, "neutral", neutral, baseline)
    state = {
        "model": config.subject_name,
        "dataset": config.dataset,
        "n_questions": len(usable),
        "n_attempted": N_QUESTIONS,
        "format_exclusions": len(excluded),
        "temperature": 0.0,
        "top_logprobs": top_logprobs,
        "provider": config.provider_label,
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
    write_json(config.output_dir / "run_state.json", state)
    print(f"Completed {config.subject_name}: {config.output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--subject-name", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiled-dir", type=Path, required=True)
    parser.add_argument("--provider-label", default="SelfHosted-Vast")
    parser.add_argument("--dataset", default="SimpleMC")
    parser.add_argument("--questions-file", type=Path)
    parser.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}')
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--validation-questions", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        subject_name=args.subject_name,
        served_model_name=args.served_model_name,
        endpoint=args.endpoint,
        output_dir=args.output_dir,
        compiled_dir=args.compiled_dir,
        provider_label=args.provider_label,
        chat_template_kwargs=json.loads(args.chat_template_kwargs),
        dataset=args.dataset,
        questions_file=args.questions_file,
    )
    if args.command == "preflight":
        run_preflight(config, args.top_logprobs, args.validation_questions)
    else:
        complete_run(config, args.workers)


if __name__ == "__main__":
    main()
