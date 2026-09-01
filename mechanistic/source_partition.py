from __future__ import annotations

from typing import Any

from . import LETTERS
from .prompts import (
    CHOICE_CUE,
    GAME_FEEDBACK,
    NEUTRAL_FEEDBACK,
    present_question,
)


SOURCE_NAMES = (
    "system",
    "first_instruction",
    "first_question_stem",
    "first_option_A",
    "first_option_B",
    "first_option_C",
    "first_option_D",
    "first_choice_cue",
    "historical_assistant",
    "feedback_subject",
    "feedback_condition",
    "feedback_action",
    "second_instruction",
    "repeated_question_stem",
    "repeated_option_A",
    "repeated_option_B",
    "repeated_option_C",
    "repeated_option_D",
    "second_choice_cue",
    "final_assistant_prefix",
    "other_structure",
)


def _find_after(text: str, needle: str, start: int) -> tuple[int, int]:
    index = text.find(needle, start)
    if index < 0:
        raise RuntimeError(f"Could not locate prompt substring after {start}: {needle!r}")
    return index, index + len(needle)


def _overlaps(offset: tuple[int, int], interval: tuple[int, int]) -> bool:
    start, end = offset
    left, right = interval
    return end > start and start < right and end > left


def prompt_source_partition(
    tokenizer: Any,
    prompt: str,
    messages: list[dict[str, str]],
    question: dict[str, Any],
    condition: str,
) -> tuple[list[int], dict[str, list[int]]]:
    """Partition every prompt token into a fixed, interpretable source span.

    The partition is exhaustive and disjoint. Question text is grouped by stem
    and option, while condition feedback is divided by grammatical role rather
    than by a hand-selected keyword.
    """

    if condition not in {"incorrect", "neutral"}:
        raise ValueError("Source partition is defined for Game and Neutral prompts")
    if len(messages) != 4 or [message["role"] for message in messages] != [
        "system",
        "user",
        "assistant",
        "user",
    ]:
        raise ValueError("Expected the four-turn empty-history Second Chance prompt")

    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]

    cursor = 0
    system_range = _find_after(prompt, messages[0]["content"], cursor)
    cursor = system_range[1]
    first_user_range = _find_after(prompt, messages[1]["content"], cursor)
    cursor = first_user_range[1]
    second_user_range = _find_after(prompt, messages[3]["content"], cursor)

    question_text = present_question(question)
    first_question = _find_after(prompt, question_text, first_user_range[0])
    repeated_question = _find_after(prompt, question_text, second_user_range[0])
    first_choice = _find_after(prompt, CHOICE_CUE, first_question[1])
    second_choice = _find_after(prompt, CHOICE_CUE, repeated_question[1])

    intervals: dict[str, list[tuple[int, int]]] = {
        name: [] for name in SOURCE_NAMES
    }
    intervals["system"] = [system_range]
    intervals["first_instruction"] = [
        (first_user_range[0], first_question[0])
    ]
    intervals["first_question_stem"] = [first_question]
    intervals["first_choice_cue"] = [
        (first_question[1], first_user_range[1])
    ]
    intervals["historical_assistant"] = [
        (first_user_range[1], second_user_range[0])
    ]

    feedback = GAME_FEEDBACK if condition == "incorrect" else NEUTRAL_FEEDBACK
    feedback_range = _find_after(prompt, feedback, second_user_range[0])
    if condition == "incorrect":
        feedback_parts = {
            "feedback_subject": "Your answer",
            "feedback_condition": "was incorrect.",
            "feedback_action": "Choose a different answer.",
        }
    else:
        feedback_parts = {
            "feedback_subject": "Your response",
            "feedback_condition": "was lost in transmission.",
            "feedback_action": "Choose again.",
        }
    part_cursor = feedback_range[0]
    for name, text in feedback_parts.items():
        interval = _find_after(prompt, text, part_cursor)
        if interval[1] > feedback_range[1]:
            raise RuntimeError(f"Feedback part {name} escaped feedback sentence")
        intervals[name] = [interval]
        part_cursor = interval[1]

    intervals["second_instruction"] = [
        (feedback_range[1], repeated_question[0])
    ]
    intervals["repeated_question_stem"] = [repeated_question]
    intervals["second_choice_cue"] = [
        (repeated_question[1], second_user_range[1])
    ]
    intervals["final_assistant_prefix"] = [
        (second_user_range[1], len(prompt))
    ]

    for occurrence_name, occurrence in (
        ("first", first_question),
        ("repeated", repeated_question),
    ):
        for letter in LETTERS:
            option_text = f"  {letter}: {question['options'][letter]}"
            option_range = _find_after(prompt, option_text, occurrence[0])
            if option_range[1] > occurrence[1]:
                raise RuntimeError(
                    f"Option {letter} escaped the {occurrence_name} question"
                )
            intervals[f"{occurrence_name}_option_{letter}"] = [option_range]

    # Specific regions override their enclosing question-stem regions.
    priority = [
        "first_option_A",
        "first_option_B",
        "first_option_C",
        "first_option_D",
        "repeated_option_A",
        "repeated_option_B",
        "repeated_option_C",
        "repeated_option_D",
        "feedback_subject",
        "feedback_condition",
        "feedback_action",
        *[
            name
            for name in SOURCE_NAMES
            if name
            not in {
                "first_option_A",
                "first_option_B",
                "first_option_C",
                "first_option_D",
                "repeated_option_A",
                "repeated_option_B",
                "repeated_option_C",
                "repeated_option_D",
                "feedback_subject",
                "feedback_condition",
                "feedback_action",
                "other_structure",
            }
        ],
    ]
    positions = {name: [] for name in SOURCE_NAMES}
    for token_index, offset in enumerate(offsets):
        assigned = None
        for name in priority:
            if any(_overlaps(offset, interval) for interval in intervals[name]):
                assigned = name
                break
        positions[assigned or "other_structure"].append(token_index)

    missing = [name for name, values in positions.items() if not values]
    if missing:
        raise RuntimeError(f"Empty source-partition spans: {missing}")
    covered = sorted(position for values in positions.values() for position in values)
    if covered != list(range(len(token_ids))):
        raise RuntimeError("Source partition is not exhaustive and disjoint")
    return token_ids, positions
