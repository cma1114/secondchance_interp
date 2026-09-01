#!/usr/bin/env python3
"""Run the Qwen TriviaMC Second Chance replication with resumable stages."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import time
from pathlib import Path

from base_game_class import BaseGameClass
from capabilities_test import CapabilitiesTest
from load_and_format_datasets import load_and_format_dataset
from second_chance_game import SecondChanceGame


MODEL = "qwen3-235b-a22b-2507"
DATASET = "TriviaMC"
N_QUESTIONS = 500
SEED = 42
OUTPUT_DIR = Path("outputs/reproduction/triviamc")
PREFLIGHT_PATH = OUTPUT_DIR / "preflight.json"
MANIFEST_PATH = OUTPUT_DIR / "stimulus_manifest.json"
STATE_PATH = OUTPUT_DIR / "run_state.json"
COMPILED_PATH = Path("compiled_results_triviamc") / f"{MODEL}_phase1_compiled.json"
PREFLIGHT_REQUEST_INTERVAL = 3.4
PIN_PREFLIGHT_PROVIDER = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--requested-top-logprobs", type=int, default=20)
    preflight.add_argument("--validation-calls", type=int, default=3)
    subparsers.add_parser("run")
    return parser.parse_args()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path: Path, default: object) -> object:
    if not path.exists():
        return copy.deepcopy(default)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        return read_json(MANIFEST_PATH, {})["questions"]
    questions = load_and_format_dataset(DATASET, N_QUESTIONS)
    if not questions or len(questions) != N_QUESTIONS:
        raise RuntimeError(f"Expected {N_QUESTIONS} valid {DATASET} questions")
    rng = random.Random(SEED)
    rng.shuffle(questions)
    manifest = {
        "dataset": DATASET,
        "source": "TriviaMC_difficulty_filtered.jsonl",
        "seed": SEED,
        "n_questions": len(questions),
        "options_preserved_from_source": True,
        "questions": questions,
    }
    write_json(MANIFEST_PATH, manifest)
    return questions


def capability_prompt(game: BaseGameClass, question: dict) -> str:
    q_text = game._present_question(question)
    setup = (
        "I'm going to ask you a series of multiple choice questions. For each one, "
        "select the answer you think is best. Respond only with the letter of your "
        "choice; do NOT output any other text."
    )
    return setup + "\n\n" + q_text + "\nYour choice (A, B, C, or D): "


def run_preflight(requested: int, validation_calls: int) -> None:
    questions = load_manifest()
    game = BaseGameClass("qwen_triviamc_preflight", MODEL, False, "game_logs")
    BaseGameClass._openrouter_request_interval = PREFLIGHT_REQUEST_INTERVAL
    BaseGameClass._openrouter_provider_only = None
    calls = []
    for question in questions[:validation_calls]:
        started = time.monotonic()
        answer, _, probabilities = game._get_llm_answer(
            list(question["options"]),
            capability_prompt(game, question),
            [],
            keep_appending=False,
            MAX_TOKENS=1,
            temp=0.0,
            top_logprobs_count=requested,
        )
        metadata = dict(game.last_call_metadata or {})
        metadata.update({
            "question_id": question["id"],
            "answer": answer,
            "probability_entries": len(probabilities or {}),
            "wall_seconds": time.monotonic() - started,
        })
        calls.append(metadata)
        if PIN_PREFLIGHT_PROVIDER and BaseGameClass._openrouter_provider_only is None:
            BaseGameClass._openrouter_provider_only = metadata.get("serving_provider")

    returned = [
        call.get("top_logprobs_returned")
        for call in calls
        if isinstance(call.get("top_logprobs_returned"), int)
    ]
    if len(returned) != validation_calls or min(returned) < 4:
        raise RuntimeError(f"Preflight did not reliably return at least four logprobs: {calls}")
    selected = min(requested, min(returned))
    output = {
        "model": MODEL,
        "dataset": DATASET,
        "requested_top_logprobs": requested,
        "selected_top_logprobs": selected,
        "request_interval_seconds": BaseGameClass._openrouter_request_interval,
        "selected_provider": (
            calls[0].get("serving_provider")
            if len({call.get("serving_provider") for call in calls}) == 1
            else None
        ),
        "calls": calls,
    }
    write_json(PREFLIGHT_PATH, output)
    print(json.dumps(output, indent=2))


def save_state(state: dict) -> None:
    state["updated_at"] = time.time()
    write_json(STATE_PATH, state)


def run_baseline(questions: list[dict], top_logprobs: int, state: dict) -> None:
    if state.get("baseline_complete") and COMPILED_PATH.exists():
        print(f"Baseline already complete: {COMPILED_PATH}")
        return
    previous_path = state.get("baseline_work_path")
    resume_from = previous_path if previous_path and Path(previous_path).exists() else None
    game = CapabilitiesTest(
        subject_id=f"{MODEL}_{DATASET}_{N_QUESTIONS}",
        subject_name=MODEL,
        questions=questions,
        n_questions=N_QUESTIONS,
        temperature=0.0,
        resume_from=resume_from,
        top_logprobs_count=top_logprobs,
    )
    work_path = f"{game.log_base_name}{game.log_suffix}.json"
    state["baseline_work_path"] = work_path
    save_state(state)
    success, result_path = game.run_capabilities_measurement()
    if not success:
        raise RuntimeError("Baseline capabilities run failed")
    COMPILED_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result_path, COMPILED_PATH)
    state["baseline_work_path"] = result_path
    state["baseline_compiled_path"] = str(COMPILED_PATH)
    state["baseline_complete"] = True
    save_state(state)


def run_second_chance_stage(
    stage_name: str,
    prompt_variant: str,
    use_correct_answers: bool,
    top_logprobs: int,
    state: dict,
) -> None:
    stages = state.setdefault("second_chance", {})
    stage = stages.setdefault(stage_name, {})
    if stage.get("complete") and stage.get("path") and Path(stage["path"]).exists():
        print(f"Stage already complete: {stage_name}")
        return
    resume_from = stage.get("path") if stage.get("path") and Path(stage["path"]).exists() else None
    settings_suffix = f"{prompt_variant}_redacted"
    if use_correct_answers:
        settings_suffix += "_cor"
    settings_suffix += "_temp0.0"
    game = SecondChanceGame(
        subject_id=f"{MODEL}_{DATASET}{settings_suffix}",
        subject_name=MODEL,
        dataset=DATASET,
        capabilities_file_path=str(COMPILED_PATH),
        num_questions=None,
        show_original_answer=False,
        use_correct_answers=use_correct_answers,
        PROMPT_VARIANT=prompt_variant,
        seed=SEED,
        temperature=0.0,
        resample=False,
        resume_from=resume_from,
        top_logprobs_count=top_logprobs,
    )
    stage["path"] = resume_from or game.game_data_filename
    stage["complete"] = False
    save_state(state)
    if not game.run_game():
        raise RuntimeError(f"Second Chance stage failed: {stage_name}")
    stage["complete"] = True
    save_state(state)


def run_replication() -> None:
    preflight = read_json(PREFLIGHT_PATH, None)
    if not preflight:
        raise RuntimeError("Run the preflight command before the full replication")
    top_logprobs = int(preflight["selected_top_logprobs"])
    questions = load_manifest()
    state = read_json(STATE_PATH, {
        "model": MODEL,
        "dataset": DATASET,
        "temperature": 0.0,
        "seed": SEED,
        "top_logprobs": top_logprobs,
    })
    BaseGameClass._openrouter_request_interval = float(
        preflight.get("request_interval_seconds", 3.4)
    )
    BaseGameClass._openrouter_provider_only = preflight.get("selected_provider")
    run_baseline(questions, top_logprobs, state)
    for stage_name, prompt_variant, correct in [
        ("incorrect_baseline_incorrect", "", False),
        ("incorrect_baseline_correct", "", True),
        ("neutral_baseline_incorrect", "_neut", False),
        ("neutral_baseline_correct", "_neut", True),
    ]:
        run_second_chance_stage(
            stage_name, prompt_variant, correct, top_logprobs, state
        )
    state["complete"] = True
    save_state(state)
    print(f"{DATASET} replication complete. State: {STATE_PATH}")


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        run_preflight(args.requested_top_logprobs, args.validation_calls)
    else:
        run_replication()


if __name__ == "__main__":
    main()
