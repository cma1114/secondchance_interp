from __future__ import annotations

from mechanistic.run_candidate_history_policy_binding import (
    SCENARIOS,
    _cross_row_cache,
)


def test_stage_c_inventory_is_small_complete_and_convolution_safe():
    assert len(SCENARIOS) == 11
    assert len(set(SCENARIOS)) == len(SCENARIOS)
    assert SCENARIOS[:3] == (
        "natural",
        "identity_pre_prefix",
        "feedback_suffix_swapped",
    )
    assert {
        "relay_task_swapped_R1",
        "relay_task_swapped_R2",
        "relay_task_swapped_R3",
        "relay_task_swapped_R4",
    }.issubset(SCENARIOS)
    assert all("prefix" not in name or "pre_prefix" in name for name in SCENARIOS)


def test_cross_row_cache_rekeys_reciprocal_policy_donors():
    rows = {
        0: ("p", "game-q0"),
        1: ("p", "game-q1"),
        2: ("p", "neutral-q0"),
        3: ("p", "neutral-q1"),
    }
    cache = {3: rows, 7: rows}
    donors = {0: 2, 1: 3, 2: 0, 3: 1}
    crossed = _cross_row_cache(cache, donors)
    assert crossed[3][0] is rows[2]
    assert crossed[3][1] is rows[3]
    assert crossed[7][2] is rows[0]
    assert crossed[7][3] is rows[1]
