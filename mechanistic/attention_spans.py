from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .prompts import GAME_FEEDBACK, GAME_SETUP, NEUTRAL_FEEDBACK, present_question


SPAN_NAMES = (
    "condition_keyword",
    "action_keyword",
    "feedback_sentence",
    "system_condition",
    "redacted_answer",
    "first_question",
    "repeated_question",
    "query_self",
    "previous_4",
    "previous_8",
)


def _character_intervals(text: str, needles: Iterable[str]) -> list[tuple[int, int]]:
    lowered = text.lower()
    intervals: list[tuple[int, int]] = []
    for needle in needles:
        start = 0
        lowered_needle = needle.lower()
        while True:
            index = lowered.find(lowered_needle, start)
            if index < 0:
                break
            intervals.append((index, index + len(needle)))
            start = index + len(needle)
    return intervals


def _token_indices(
    offsets: list[tuple[int, int]], intervals: Iterable[tuple[int, int]]
) -> list[int]:
    intervals = list(intervals)
    return [
        token_index
        for token_index, (start, end) in enumerate(offsets)
        if end > start and any(start < interval_end and end > interval_start for interval_start, interval_end in intervals)
    ]


def attention_span_indices(
    tokenizer: Any,
    prompt: str,
    condition: str,
    question: dict[str, Any],
) -> tuple[list[int], dict[str, list[int]]]:
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    question_text = present_question(question)
    question_occurrences = _character_intervals(prompt, [question_text])
    if len(question_occurrences) != (1 if condition == "baseline" else 2):
        raise RuntimeError(
            f"Expected rendered question occurrence count for {condition}; got {len(question_occurrences)}"
        )

    if condition in {"incorrect", "incorrect_no_system_setup"}:
        condition_terms = ["incorrect"]
        action_terms = ["different answer"]
        feedback = GAME_FEEDBACK
        system_condition = (
            GAME_SETUP.strip() if GAME_SETUP.strip() in prompt else ""
        )
    elif condition == "neutral":
        condition_terms = ["lost", "transmission"]
        action_terms = ["again"]
        feedback = NEUTRAL_FEEDBACK
        system_condition = ""
    else:
        condition_terms = []
        action_terms = []
        feedback = ""
        system_condition = ""

    last = len(input_ids) - 1
    intervals = {
        "condition_keyword": _character_intervals(prompt, condition_terms),
        "action_keyword": _character_intervals(prompt, action_terms),
        "feedback_sentence": _character_intervals(prompt, [feedback]) if feedback else [],
        "system_condition": _character_intervals(prompt, [system_condition]) if system_condition else [],
        "redacted_answer": _character_intervals(prompt, ["[redacted]"]),
        "first_question": question_occurrences[:1],
        "repeated_question": question_occurrences[-1:] if condition != "baseline" else [],
    }
    spans = {name: _token_indices(offsets, value) for name, value in intervals.items()}
    spans["query_self"] = [last]
    spans["previous_4"] = list(range(max(0, last - 4), last))
    spans["previous_8"] = list(range(max(0, last - 8), last))
    return input_ids, {name: spans.get(name, []) for name in SPAN_NAMES}
