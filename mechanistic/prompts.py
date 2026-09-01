from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import LETTERS


CAPABILITY_SETUP = (
    "I'm going to ask you a series of multiple choice questions. For each one, "
    "select the answer you think is best. Respond only with the letter of your "
    "choice; do NOT output any other text."
)
ANSWER_ONLY_INSTRUCTION = (
    "Respond only with the letter of your choice; do NOT output any other text."
)
CHOICE_CUE = "Your choice (A, B, C, or D): "
HISTORY_SETUP = (
    "I'm going to ask you a series of multiple choice questions. For each one, "
    "select the answer you think is best."
)
GAME_FEEDBACK = "Your answer was incorrect. Choose a different answer."
NEUTRAL_FEEDBACK = "Your response was lost in transmission. Choose again."
TOKEN_MATCHED_TEST_GAME_FEEDBACK = (
    "Your answer was incorrect. Choose a different answer."
)
TOKEN_MATCHED_TEST_NEUTRAL_FEEDBACK = (
    "Your answer was lost. Choose the answer again."
)
FACTORIAL_FEEDBACK = {
    "incorrect_different": "Your answer was incorrect. Choose a different answer.",
    "incorrect_again": "Your answer was incorrect. Choose the answer again.",
    "lost_different": "Your answer was lost. Choose a different answer.",
    "lost_again": "Your answer was lost. Choose the answer again.",
}
GAME_SETUP = "Your previous answer was incorrect. Respond ONLY with your new answer\n"


@dataclass(frozen=True)
class Trial:
    question_id: str
    question: dict[str, Any]
    baseline_answer: str
    baseline_correct: bool


def present_question(question: dict[str, Any]) -> str:
    out = "-" * 30 + "\nQuestion:\n" + question["question"] + "\n" + "-" * 10 + "\n"
    for key, value in question["options"].items():
        out += f"  {key}: {value}\n"
    return out + "-" * 30


def baseline_question_turn(question: dict[str, Any]) -> str:
    """The exact user message used for a capabilities-test question."""
    return (
        CAPABILITY_SETUP
        + "\n\n"
        + present_question(question)
        + "\n"
        + CHOICE_CUE
    )


def repeated_question_turn(question: dict[str, Any]) -> str:
    """Repeat the question with Baseline-matched answer-only instruction ordering."""
    return (
        ANSWER_ONLY_INSTRUCTION
        + "\n\n"
        + present_question(question)
        + "\n"
        + CHOICE_CUE
    )


def _options_system(setup_text: str | None, faithful: bool) -> str:
    # `None` means that no condition-specific setup instruction is present.
    # It must never be stringified into the model-visible prompt.
    if setup_text is None:
        setup_text = ""
    setup_prefix = f"{setup_text}\n" if setup_text else ""
    return f"{setup_prefix}Output ONLY the letter of your choice: A, B, C, or D.\n"


def build_messages(
    question: dict[str, Any],
    condition: str,
    prompt_mode: str = "faithful",
    feedback_variant: str = "standard",
) -> list[dict[str, str]]:
    if list(question["options"]) != list(LETTERS):
        raise ValueError(f"Question {question.get('id')} does not have ordered A-D options")
    faithful = prompt_mode == "faithful"
    q_text = present_question(question)
    if condition == "baseline":
        return [
            {"role": "system", "content": _options_system("", faithful)},
            {"role": "user", "content": baseline_question_turn(question)},
        ]
    baseline_matched = prompt_mode in {
        "baseline_matched", "baseline_matched_empty_history"
    }
    if baseline_matched:
        # The first presentation exactly matches Baseline through the initial
        # assistant decision. The second presentation retains the original
        # Second Chance ordering and does not repeat the conversational preamble.
        first_turn = baseline_question_turn(question)
        repeated = repeated_question_turn(question)
    else:
        first_turn = (
            HISTORY_SETUP
            + "\n\n"
            + q_text
            + "\nRespond only with the letter of your choice; do NOT output any other text.\n"
            + "Your choice (A, B, C, or D): "
        )
        repeated = repeated_question_turn(question)
    if feedback_variant == "standard":
        game_feedback = GAME_FEEDBACK
        neutral_feedback = NEUTRAL_FEEDBACK
    elif feedback_variant == "token_matched_test":
        game_feedback = TOKEN_MATCHED_TEST_GAME_FEEDBACK
        neutral_feedback = TOKEN_MATCHED_TEST_NEUTRAL_FEEDBACK
    else:
        raise ValueError(f"Unknown feedback variant: {feedback_variant}")
    if condition == "incorrect":
        feedback = game_feedback
        setup = None if prompt_mode in {
            "no_system_incorrect", "baseline_matched",
            "baseline_matched_empty_history",
        } else GAME_SETUP
    elif condition == "incorrect_no_system_setup":
        # Prompt ablation: retain the ordinary Game feedback in the final user
        # turn, but remove the advance condition-specific system instruction.
        feedback, setup = game_feedback, None
    elif condition == "neutral":
        feedback, setup = neutral_feedback, None
    else:
        raise ValueError(f"Unknown condition: {condition}")
    # The historical client payload appended the system-role object after the
    # saved history, but OpenRouter successfully served Qwen and the published
    # prompt shows the model-visible system instruction at the beginning. The
    # native Qwen template likewise requires the system role to be first.
    historical_answer = (
        "" if prompt_mode == "baseline_matched_empty_history" else "[redacted]"
    )
    return [
        {"role": "system", "content": _options_system(setup, faithful)},
        {"role": "user", "content": first_turn},
        {"role": "assistant", "content": historical_answer},
        {"role": "user", "content": feedback + "\n\n" + repeated},
    ]


def build_factorial_messages(
    question: dict[str, Any],
    condition: str,
    prompt_mode: str = "baseline_matched_empty_history",
) -> list[dict[str, str]]:
    """Build the canonical 2x2 evaluation-by-action feedback prompt.

    All four conditions use the same token-matched, empty-history format.  The
    only model-visible differences are `incorrect` versus `lost` and
    `different` versus `again` in the feedback sentence.
    """
    if condition not in FACTORIAL_FEEDBACK:
        raise ValueError(f"Unknown factorial condition: {condition}")
    messages = build_messages(
        question,
        "incorrect" if condition.startswith("incorrect_") else "neutral",
        prompt_mode,
        "token_matched_test",
    )
    canonical = (
        TOKEN_MATCHED_TEST_GAME_FEEDBACK
        if condition.startswith("incorrect_")
        else TOKEN_MATCHED_TEST_NEUTRAL_FEEDBACK
    )
    content = messages[-1]["content"]
    if not content.startswith(canonical + "\n\n"):
        raise RuntimeError("Could not locate canonical feedback prefix")
    messages[-1]["content"] = (
        FACTORIAL_FEEDBACK[condition] + content[len(canonical):]
    )
    return messages


def prompt_hash(rendered_prompt: str) -> str:
    return hashlib.sha256(rendered_prompt.encode()).hexdigest()


def load_trials(
    manifest_path: str | Path,
    baseline_results_path: str | Path,
    question_ids: Iterable[str] | None = None,
    max_questions: int | None = None,
    skip_missing_baseline: bool = False,
) -> list[Trial]:
    manifest = json.loads(Path(manifest_path).read_text())
    baseline = json.loads(Path(baseline_results_path).read_text())["results"]
    wanted = set(question_ids) if question_ids else None
    trials: list[Trial] = []
    for q in manifest["questions"]:
        qid = q["id"]
        if wanted is not None and qid not in wanted:
            continue
        if qid not in baseline:
            if skip_missing_baseline:
                continue
            raise KeyError(f"Question {qid} is absent from baseline results")
        row = baseline[qid]
        answer = row["subject_answer"]
        if answer not in LETTERS:
            raise ValueError(f"Baseline answer for {qid} is not A-D: {answer!r}")
        trials.append(Trial(qid, q, answer, bool(row["is_correct"])))
        if max_questions is not None and len(trials) >= max_questions:
            break
    if wanted is not None and {t.question_id for t in trials} != wanted:
        missing = sorted(wanted - {t.question_id for t in trials})
        raise KeyError(f"Requested question IDs not found: {missing}")
    return trials
