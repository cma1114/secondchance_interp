from __future__ import annotations

import hashlib
import re
from typing import Any


def user_incorrect_positions(spans: dict[str, list[int]]) -> list[int]:
    positions = sorted(
        set(spans["condition_keyword"]) & set(spans["feedback_sentence"])
    )
    if not positions:
        raise RuntimeError("No user-turn `incorrect` token was found")
    return positions


def _is_structural(text: str) -> bool:
    """Formatting/punctuation token with no letters or numbers."""
    return bool(text) and re.search(r"[^\s]", text) is not None and re.search(r"[\w]", text, re.UNICODE) is None


def structural_control_positions(
    tokenizer: Any,
    input_ids: list[int],
    spans: dict[str, list[int]],
    question_id: str,
    count: int,
    seed: int,
) -> list[int]:
    source = user_incorrect_positions(spans)[0]
    final = spans["query_self"][0]
    source_distance = final - source
    excluded = set(spans["feedback_sentence"]) | set(spans["system_condition"])
    excluded |= set(spans["redacted_answer"]) | set(spans["previous_8"])
    candidates = []
    for position, token_id in enumerate(input_ids):
        if position in excluded or position == final:
            continue
        if token_id in set(getattr(tokenizer, "all_special_ids", [])):
            continue
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if not _is_structural(text):
            continue
        distance = final - position
        if distance <= 8 or not (0.55 * source_distance <= distance <= 1.45 * source_distance):
            continue
        digest = hashlib.sha256(f"{seed}:{question_id}:{position}".encode()).digest()
        candidates.append((digest, position))
    if len(candidates) < count:
        raise RuntimeError(
            f"Only {len(candidates)} eligible structural controls for {question_id}; need {count}"
        )
    candidates.sort()
    return [position for _, position in candidates[:count]]


def source_positions(
    selector: str,
    tokenizer: Any,
    input_ids: list[int],
    spans: dict[str, list[int]],
    question_id: str,
    count: int,
    seed: int,
) -> list[int]:
    if selector == "user_incorrect":
        return user_incorrect_positions(spans)
    if selector.startswith("structural_"):
        index = int(selector.split("_", 1)[1])
        controls = structural_control_positions(
            tokenizer, input_ids, spans, question_id, count, seed
        )
        if index >= len(controls):
            raise ValueError(f"Structural-control index out of range: {selector}")
        return [controls[index]]
    raise ValueError(f"Unknown GDN source selector: {selector}")

