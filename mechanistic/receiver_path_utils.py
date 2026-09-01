from __future__ import annotations

from typing import Any

import numpy as np

from . import LETTERS
from .prompts import CHOICE_CUE, FACTORIAL_FEEDBACK, present_question
from .run_fixed_a_final_query_edge_ablation import _option_line_positions


SLOT_LIMITS = {
    "first_choice_cue": 24,
    "historical_assistant": 32,
    "feedback": 24,
    "second_instruction": 40,
    "second_choice_cue": 24,
    "final_assistant": 24,
}

ROLE_NAMES = (
    "ignored_before_all_first_options",
    *(f"first_choice_cue_slot_{i:02d}" for i in range(SLOT_LIMITS["first_choice_cue"])),
    *(f"historical_assistant_slot_{i:02d}" for i in range(SLOT_LIMITS["historical_assistant"])),
    *(f"feedback_slot_{i:02d}" for i in range(SLOT_LIMITS["feedback"])),
    *(f"second_instruction_slot_{i:02d}" for i in range(SLOT_LIMITS["second_instruction"])),
    "second_question_stem",
    "second_option_w1",
    "second_option_w2",
    "second_option_other",
    *(f"second_choice_cue_slot_{i:02d}" for i in range(SLOT_LIMITS["second_choice_cue"])),
    *(f"final_assistant_slot_{i:02d}" for i in range(SLOT_LIMITS["final_assistant"])),
    "final_decision_query_known_null",
    "other_downstream",
)
ROLE_INDEX = {name: index for index, name in enumerate(ROLE_NAMES)}
IGNORED_ROLE = ROLE_INDEX["ignored_before_all_first_options"]


def _find_after(text: str, needle: str, start: int) -> tuple[int, int]:
    index = text.find(needle, start)
    if index < 0:
        raise RuntimeError(f"Could not locate {needle!r} after character {start}")
    return index, index + len(needle)


def _positions_for_interval(
    offsets: list[tuple[int, int]], interval: tuple[int, int]
) -> list[int]:
    left, right = interval
    return [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < right and end > left
    ]


def _assign_slots(
    roles: np.ndarray,
    offsets: list[tuple[int, int]],
    interval: tuple[int, int],
    prefix: str,
) -> None:
    positions = _positions_for_interval(offsets, interval)
    limit = SLOT_LIMITS[prefix]
    if len(positions) > limit:
        raise RuntimeError(
            f"{prefix} uses {len(positions)} token slots; configured limit is {limit}"
        )
    for slot, position in enumerate(positions):
        roles[position] = ROLE_INDEX[f"{prefix}_slot_{slot:02d}"]


def matched_control(option_positions: dict[str, list[int]], selected: str) -> str:
    selected_count = len(option_positions[selected])
    alternatives = [letter for letter in LETTERS if letter != selected]
    return min(
        alternatives,
        key=lambda letter: (
            abs(len(option_positions[letter]) - selected_count),
            letter,
        ),
    )


def locate_receiver_roles(
    tokenizer: Any,
    prompt: str,
    messages: list[dict[str, str]],
    original_question: dict[str, Any],
    remapped_question: dict[str, Any],
    condition: str,
    w1: str,
    w2: str,
    plan_row: dict[str, Any],
) -> dict[str, Any]:
    """Locate causal source lines and structurally aligned receiver roles."""

    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    roles = np.full(len(ids), IGNORED_ROLE, dtype=np.int16)

    cursor = 0
    system_range = _find_after(prompt, messages[0]["content"], cursor)
    cursor = system_range[1]
    first_user_range = _find_after(prompt, messages[1]["content"], cursor)
    cursor = first_user_range[1]
    second_user_range = _find_after(prompt, messages[3]["content"], cursor)
    first_question_range = _find_after(
        prompt, present_question(original_question), first_user_range[0]
    )
    second_question_range = _find_after(
        prompt, present_question(remapped_question), second_user_range[0]
    )
    first_choice_range = _find_after(prompt, CHOICE_CUE, first_question_range[1])
    second_choice_range = _find_after(prompt, CHOICE_CUE, second_question_range[1])
    feedback = FACTORIAL_FEEDBACK[condition]
    feedback_range = _find_after(prompt, feedback, second_user_range[0])
    if prompt.find(feedback, feedback_range[1]) >= 0:
        raise RuntimeError(f"Expected one {condition} feedback clause")

    _assign_slots(roles, offsets, first_choice_range, "first_choice_cue")
    _assign_slots(
        roles,
        offsets,
        (first_user_range[1], second_user_range[0]),
        "historical_assistant",
    )
    _assign_slots(roles, offsets, feedback_range, "feedback")
    _assign_slots(
        roles,
        offsets,
        (feedback_range[1], second_question_range[0]),
        "second_instruction",
    )
    _assign_slots(roles, offsets, second_choice_range, "second_choice_cue")
    _assign_slots(
        roles,
        offsets,
        (second_user_range[1], len(prompt)),
        "final_assistant",
    )
    # The terminal prompt token is the final decision query. Its direct edge to
    # the first W1 option line has already been causally falsified in the
    # canonical remapped run. Keep it in the observational screen for context,
    # but give it a dedicated role so candidate selection cannot rediscover the
    # known-null edge.
    roles[-1] = ROLE_INDEX["final_decision_query_known_null"]

    for position in _positions_for_interval(offsets, second_question_range):
        roles[position] = ROLE_INDEX["second_question_stem"]
    for displayed in LETTERS:
        semantic = plan_row["new_to_original"][displayed]
        option_range = _find_after(
            prompt,
            f"  {displayed}: {remapped_question['options'][displayed]}",
            second_question_range[0],
        )
        if semantic == w1:
            role = "second_option_w1"
        elif semantic == w2:
            role = "second_option_w2"
        else:
            role = "second_option_other"
        for position in _positions_for_interval(offsets, option_range):
            roles[position] = ROLE_INDEX[role]

    option_positions, option_audit = _option_line_positions(
        tokenizer, prompt, original_question
    )
    control = matched_control(option_positions, w1)
    all_options_end = max(max(positions) for positions in option_positions.values())
    # Candidate reads begin only after all four first-presentation option lines.
    # This ensures the W1 line and matched control are both causally available.
    roles[: all_options_end + 1] = IGNORED_ROLE
    unassigned = np.where(
        (np.arange(len(ids)) > all_options_end) & (roles == IGNORED_ROLE)
    )[0]
    roles[unassigned] = ROLE_INDEX["other_downstream"]

    return {
        "ids": ids,
        "offsets": offsets,
        "roles": roles,
        "selected_positions": option_positions[w1],
        "control_positions": option_positions[control],
        "control_letter": control,
        "all_first_options_end": all_options_end,
        "selected_audit": option_audit[w1],
        "control_audit": option_audit[control],
    }


def role_positions(roles: np.ndarray, role_name: str) -> list[int]:
    code = ROLE_INDEX[role_name]
    return np.flatnonzero(roles == code).astype(int).tolist()
