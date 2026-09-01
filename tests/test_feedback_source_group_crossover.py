from __future__ import annotations

import numpy as np

from mechanistic.analyze_feedback_source_group_crossover import _decomposition
from mechanistic.run_feedback_source_group_crossover import selected_group_positions


def test_grouped_feedback_positions_are_exact_and_exhaustive():
    positions = list(range(100, 107))
    assert selected_group_positions(positions, "feedback_sentence") == [100, 101]
    assert selected_group_positions(positions, "following_instruction") == [102, 103, 104, 105, 106]
    assert selected_group_positions(positions, "complete_suffix") == positions


def test_grouped_feedback_positions_reject_noncanonical_inventory():
    try:
        selected_group_positions([1, 2], "feedback_sentence")
    except ValueError:
        return
    raise AssertionError("Expected noncanonical source inventory to be rejected")


def test_grouped_decomposition_uses_paired_complete_minus_components():
    feedback = np.asarray([1.0, 2.0, 3.0])
    instruction = np.asarray([2.0, 1.0, 1.0])
    complete = np.asarray([4.0, 4.0, 5.0])
    rows = _decomposition(
        {
            "feedback_sentence": feedback,
            "following_instruction": instruction,
            "additive_sum": feedback + instruction,
            "complete_suffix": complete,
            "nonlinear_interaction": complete - feedback - instruction,
        },
        np.asarray([10.0, 10.0, 10.0]),
        np.arange(3),
        np.random.default_rng(1),
        100,
    )
    assert np.isclose(rows["feedback_sentence"]["transfer_fraction"], 0.2)
    assert np.isclose(rows["following_instruction"]["transfer_fraction"], 4 / 30)
    assert np.isclose(rows["complete_suffix"]["transfer_fraction"], 13 / 30)
    assert np.isclose(rows["nonlinear_interaction"]["transfer_fraction"], 0.1)
