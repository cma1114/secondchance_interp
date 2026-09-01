from __future__ import annotations

import numpy as np

from mechanistic.semantic_mapping import (
    align_displayed_logits_to_semantic,
    displayed_argmax_to_semantic_indices,
)


def test_displayed_tie_is_resolved_before_semantic_mapping():
    # Displayed A and B tie. Displayed-order argmax selects A, which maps to
    # semantic D. Reordering first would put semantic A before semantic D and
    # incorrectly select semantic A.
    logits = np.asarray([[[1.0, 1.0, 0.0, 0.0]]])
    mappings = [{"new_to_original": {"A": "D", "B": "A", "C": "B", "D": "C"}}]
    answer = displayed_argmax_to_semantic_indices(logits, mappings)
    assert answer.tolist() == [[3]]
    aligned = align_displayed_logits_to_semantic(logits, mappings)
    assert aligned.argmax(axis=-1).tolist() == [[0]]


def test_helper_accepts_original_to_new_and_second_mapping_rows():
    logits = np.asarray(
        [
            [[0.0, 2.0, 1.0, -1.0], [3.0, 1.0, 0.0, 2.0]],
            [[4.0, 2.0, 1.0, -1.0], [0.0, 1.0, 5.0, 2.0]],
        ]
    )
    mappings = [
        {"original_to_new": {"A": "B", "B": "C", "C": "D", "D": "A"}},
        {
            "second_mapping": {
                "original_to_new": {"A": "D", "B": "A", "C": "B", "D": "C"}
            }
        },
    ]
    answers = displayed_argmax_to_semantic_indices(logits, mappings)
    assert answers.tolist() == [[0, 1], [3, 3]]
