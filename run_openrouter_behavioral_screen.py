#!/usr/bin/env python3
"""Resumable no-logprob Second Chance behavioral screen through OpenRouter.

This is a cheap triage for providers that expose answer text but not token
logprobs. It can test Lift and change-conditioned AccIncor. SecChoice and
NoEntInc require logits and must be evaluated in a subsequent self-hosted run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportions_ztest


REFERENCE_BASELINE = Path(
    "compiled_results_smc/qwen3-235b-a22b-2507_phase1_compiled.json"
)
LETTERS = ("A", "B", "C", "D")
N_QUESTIONS = 500
ANSWER_RE = re.compile(r"^\s*([A-D])(?:[.)])?\s*$", re.IGNORECASE)
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


def load_questions() -> list[dict]:
    reference = read_json(REFERENCE_BASELINE, {})
    questions = [trial["question"] for trial in reference["results"].values()]
    if len(questions) != N_QUESTIONS:
        raise RuntimeError(f"Expected {N_QUESTIONS} frozen questions; got {len(questions)}")
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
    return setup + "\n\n" + present_question(question) + "\nYour choice (A, B, C, or D): "


def messages_for(question: dict, condition: str) -> list[dict]:
    system_base = "Output ONLY the letter of your choice: A, B, C, or D.\n"
    if condition == "baseline":
        return [
            {"role": "system", "content": system_base},
            {"role": "user", "content": capability_user_prompt(question)},
        ]

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
        current_user = "Your answer was incorrect. Choose a different answer.\n\n" + answer_prompt
        system = (
            "Your previous answer was incorrect. Respond ONLY with your new answer\n\n"
            + system_base
        )
    elif condition == "neutral":
        current_user = "Your response was lost in transmission. Choose again.\n\n" + answer_prompt
        system = system_base
    else:
        raise ValueError(condition)

    # Preserve the historical OpenRouter request order exactly. Providers may
    # internally normalize the system role before applying their chat template.
    return [
        {"role": "user", "content": setup + "\n\n" + answer_prompt},
        {"role": "assistant", "content": "[redacted]"},
        {"role": "system", "content": system},
        {"role": "user", "content": current_user},
    ]


def client_for_thread() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=120.0,
            max_retries=0,
        )
        _thread_local.client = client
    return client


def parse_answer(raw: str | None) -> str | None:
    match = ANSWER_RE.fullmatch(raw or "")
    return match.group(1).upper() if match else None


def call_model(model: str, provider: str, messages: list[dict]) -> dict:
    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(30):
        try:
            completion = client_for_thread().chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                # Leave enough room to diagnose format failures rather than
                # mistaking a truncated prose response for a short refusal.
                max_tokens=32,
                extra_body={
                    "reasoning": {"enabled": False},
                    "include_reasoning": False,
                    "provider": {
                        "only": [provider],
                        "allow_fallbacks": False,
                        "require_parameters": True,
                    },
                },
            )
            choice = completion.choices[0]
            raw = choice.message.content
            reasoning = getattr(choice.message, "reasoning", None)
            reasoning_details = getattr(choice.message, "reasoning_details", None)
            usage = completion.usage
            completion_details = getattr(usage, "completion_tokens_details", None)
            return {
                "raw_response_content": raw,
                "answer": parse_answer(raw),
                "serving_provider": getattr(completion, "provider", None),
                "response_model": getattr(completion, "model", None),
                "reasoning_present": bool(reasoning or reasoning_details),
                "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        except Exception as exc:
            last_error = exc
            if attempt == 29:
                break
            if "429" in str(exc) or "rate-limit" in str(exc).lower():
                delay = max(delay, 15.0)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"OpenRouter request failed after retries: {last_error}")


def run_call(model: str, provider: str, question: dict, condition: str, baseline_answer: str | None = None) -> dict:
    result = call_model(model, provider, messages_for(question, condition))
    answer = result.pop("answer")
    trial = {
        "question": question,
        "probs": {},
        "call_metadata": result,
    }
    if condition == "baseline":
        trial.update(
            subject_answer=answer,
            is_correct=answer == question["correct_answer"],
        )
    else:
        trial.update(
            original_answer=baseline_answer,
            new_answer=answer,
            correct_answer=question["correct_answer"],
            answer_changed=answer in LETTERS and answer != baseline_answer,
            is_correct=answer == question["correct_answer"],
        )
    return trial


def validate_preflight_trial(trial: dict, condition: str, provider: str) -> None:
    key = "subject_answer" if condition == "baseline" else "new_answer"
    metadata = trial["call_metadata"]
    if trial.get(key) not in LETTERS:
        raise RuntimeError(f"{condition} did not return answer-only A-D: {metadata}")
    if metadata.get("serving_provider") != provider:
        raise RuntimeError(f"Expected provider {provider!r}, got {metadata}")
    if metadata.get("reasoning_present") or metadata.get("reasoning_tokens") not in (None, 0):
        raise RuntimeError(f"Reasoning was not disabled: {metadata}")


def preflight(args: argparse.Namespace) -> None:
    questions = load_questions()[: args.validation_questions]
    calls = []
    failures = []
    for question in questions:
        baseline = run_call(args.model, args.provider, question, "baseline")
        try:
            validate_preflight_trial(baseline, "baseline", args.provider)
        except RuntimeError as exc:
            failures.append({"question_id": question["id"], "condition": "baseline", "error": str(exc)})
        calls.append({"question_id": question["id"], "condition": "baseline", **baseline})
        for condition in ("game", "neutral"):
            trial = run_call(
                args.model,
                args.provider,
                question,
                condition,
                baseline["subject_answer"],
            )
            try:
                validate_preflight_trial(trial, condition, args.provider)
            except RuntimeError as exc:
                failures.append({"question_id": question["id"], "condition": condition, "error": str(exc)})
            calls.append({"question_id": question["id"], "condition": condition, **trial})
    payload = {
        "model": args.model,
        "provider": args.provider,
        "reasoning_disabled": True,
        "logprobs_available": False,
        "behavioral_tests_available": ["Lift", "AccIncor"],
        "validation_failures": failures,
        "valid_rate": (len(calls) - len(failures)) / len(calls),
        "valid_rate_by_condition": {
            condition: sum(
                call["condition"] == condition
                and call.get("subject_answer", call.get("new_answer")) in LETTERS
                for call in calls
            ) / sum(call["condition"] == condition for call in calls)
            for condition in ("baseline", "game", "neutral")
        },
        "calls": calls,
        "prompt_audit": {
            condition: messages_for(questions[0], condition)
            for condition in ("baseline", "game", "neutral")
        },
    }
    write_json(args.output_dir / "preflight.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    if payload["valid_rate"] < 0.90 or min(payload["valid_rate_by_condition"].values()) < 0.80:
        raise RuntimeError(
            f"Preflight format compliance was too low: {payload['valid_rate']:.1%} overall, "
            f"{payload['valid_rate_by_condition']}"
        )


def run_condition(
    args: argparse.Namespace,
    condition: str,
    questions: list[dict],
    baseline: dict[str, dict] | None = None,
) -> dict[str, dict]:
    path = args.output_dir / f"{condition}_results.json"
    payload = read_json(path, {"condition": condition, "results": {}})
    results = payload["results"]
    key = "subject_answer" if condition == "baseline" else "new_answer"
    pending = [question for question in questions if results.get(question["id"], {}).get(key) not in LETTERS]
    print(f"{condition}: {len(pending)} pending/{len(questions)}", flush=True)
    if not pending:
        return results

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for question in pending:
            baseline_answer = baseline[question["id"]]["subject_answer"] if baseline else None
            future = executor.submit(
                run_call,
                args.model,
                args.provider,
                question,
                condition,
                baseline_answer,
            )
            futures[future] = question
        completed_since_save = 0
        for index, future in enumerate(as_completed(futures), 1):
            question = futures[future]
            results[question["id"]] = future.result()
            completed_since_save += 1
            if completed_since_save >= 10:
                payload["updated_at"] = time.time()
                write_json(path, payload)
                completed_since_save = 0
            if index % 25 == 0 or index == len(pending):
                print(f"{condition}: completed {index}/{len(pending)}", flush=True)
    payload["complete"] = len(results) == len(questions)
    payload["updated_at"] = time.time()
    write_json(path, payload)
    return results


def mcnemar_exact(game_switched: list[bool], neutral_switched: list[bool]) -> tuple[int, int, float]:
    game_only = sum(g and not n for g, n in zip(game_switched, neutral_switched))
    neutral_only = sum(n and not g for g, n in zip(game_switched, neutral_switched))
    discordant = game_only + neutral_only
    p_value = 1.0 if discordant == 0 else binomtest(game_only, discordant, 0.5).pvalue
    return game_only, neutral_only, float(p_value)


def analyze(args: argparse.Namespace, baseline: dict[str, dict], game: dict[str, dict], neutral: dict[str, dict]) -> dict:
    usable_ids = [
        qid for qid in baseline
        if baseline[qid].get("subject_answer") in LETTERS
        and game.get(qid, {}).get("new_answer") in LETTERS
        and neutral.get(qid, {}).get("new_answer") in LETTERS
        and not baseline[qid]["call_metadata"].get("reasoning_present")
        and not game[qid]["call_metadata"].get("reasoning_present")
        and not neutral[qid]["call_metadata"].get("reasoning_present")
    ]
    if len(usable_ids) < 450:
        raise RuntimeError(f"Only {len(usable_ids)}/500 trials were usable")
    game_switches = [bool(game[qid]["answer_changed"]) for qid in usable_ids]
    neutral_switches = [bool(neutral[qid]["answer_changed"]) for qid in usable_ids]
    game_rate = sum(game_switches) / len(usable_ids)
    neutral_rate = sum(neutral_switches) / len(usable_ids)
    game_only, neutral_only, lift_p = mcnemar_exact(game_switches, neutral_switches)

    changed_wrong = [
        qid for qid in usable_ids
        if not baseline[qid]["is_correct"] and game[qid]["answer_changed"]
    ]
    corrected = sum(game[qid]["is_correct"] for qid in changed_wrong)
    acc_rate = corrected / len(changed_wrong) if changed_wrong else math.nan
    if changed_wrong:
        _, acc_z_p = proportions_ztest(corrected, len(changed_wrong), value=1 / 3)
        acc_exact_p = binomtest(corrected, len(changed_wrong), 1 / 3, alternative="greater").pvalue
    else:
        acc_z_p = acc_exact_p = math.nan

    prompt_tokens = sum(
        int(trial["call_metadata"].get("prompt_tokens") or 0)
        for results in (baseline, game, neutral)
        for trial in results.values()
    )
    completion_tokens = sum(
        int(trial["call_metadata"].get("completion_tokens") or 0)
        for results in (baseline, game, neutral)
        for trial in results.values()
    )
    estimated_cost = prompt_tokens * 0.09 / 1_000_000 + completion_tokens * 0.18 / 1_000_000
    summary = {
        "model": args.model,
        "provider": args.provider,
        "n_attempted": N_QUESTIONS,
        "n_usable": len(usable_ids),
        "baseline_accuracy": sum(baseline[qid]["is_correct"] for qid in usable_ids) / len(usable_ids),
        "lift": {
            "game_rate": game_rate,
            "neutral_rate": neutral_rate,
            "absolute_lift": game_rate - neutral_rate,
            "normalized_lift": (game_rate - neutral_rate) / (1 - neutral_rate),
            "game_only_discordant": game_only,
            "neutral_only_discordant": neutral_only,
            "mcnemar_exact_p": lift_p,
        },
        "accincor": {
            "changed_baseline_incorrect": len(changed_wrong),
            "corrected": corrected,
            "rate": acc_rate,
            "two_sided_z_p": float(acc_z_p),
            "exact_one_sided_p": float(acc_exact_p),
        },
        "paper_tests": {
            "Lift": bool(game_rate > neutral_rate and lift_p < 0.05),
            "AccIncor": bool(acc_rate > 1 / 3 and acc_z_p < 0.05),
            "SecChoice": None,
            "NoEntInc": None,
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated_openrouter_cost_usd": estimated_cost,
        },
    }
    write_json(args.output_dir / "behavioral_summary.json", summary)
    report = f"""# Laguna S 2.1 SimpleMC hosted behavioral screen

| Test | Result | Pass |
|---|---:|:---:|
| Lift | Game {game_rate:.1%}; neutral {neutral_rate:.1%}; normalized lift {summary['lift']['normalized_lift']:.3f}; paired p={lift_p:.3g} | {'✓' if summary['paper_tests']['Lift'] else 'X'} |
| AccIncor | {corrected}/{len(changed_wrong)} changed baseline-incorrect trials = {acc_rate:.1%}; two-sided z p vs 1/3={acc_z_p:.3g}; exact one-sided p={acc_exact_p:.3g} | {'✓' if summary['paper_tests']['AccIncor'] else 'X'} |
| SecChoice | Requires baseline logits | — |
| NoEntInc | Requires logits | — |

Baseline accuracy was {summary['baseline_accuracy']:.1%} on {len(usable_ids)} paired usable trials. Estimated stored-call cost was ${estimated_cost:.3f}.
"""
    (args.output_dir / "BEHAVIORAL_SCREEN_REPORT.md").write_text(report, encoding="utf-8")
    return summary


def full_run(args: argparse.Namespace) -> None:
    if not (args.output_dir / "preflight.json").exists():
        raise RuntimeError("Run preflight first")
    questions = load_questions()
    baseline = run_condition(args, "baseline", questions)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_condition, args, condition, questions, baseline): condition
            for condition in ("game", "neutral")
        }
        condition_results = {futures[future]: future.result() for future in as_completed(futures)}
    summary = analyze(args, baseline, condition_results["game"], condition_results["neutral"])
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--model", default="poolside/laguna-s-2.1")
    parser.add_argument("--provider", default="Poolside")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reproduction/simplemc_laguna_s_2_1_hosted"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validation-questions", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "preflight":
        preflight(arguments)
    else:
        full_run(arguments)
