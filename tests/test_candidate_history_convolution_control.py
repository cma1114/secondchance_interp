from __future__ import annotations

from mechanistic.run_candidate_history_convolution_control import (
    CONSERVATIVE_FREE_SUFFIX,
    CONTROL_SCENARIOS,
    EXACT_FREE_SUFFIX,
    RESTORE_ALL_EXCEPT_LAST3_MASK,
    RESTORE_ALL_EXCEPT_LAST4_MASK,
    _control_mask_label,
    _convolution_safe_positions,
)


def _groups() -> dict[str, list[int]]:
    return {
        "second_option_semantics": [10, 11],
        "second_option_newlines": [12],
        "second_option_structure": [13],
        "post_list_cue_and_query": [14, 15],
        "final_assistant_prefix": list(range(16, 24)),
    }


def test_control_inventory_is_small_and_complete():
    assert len(CONTROL_SCENARIOS) == 8
    assert len(set(CONTROL_SCENARIOS)) == 8
    assert EXACT_FREE_SUFFIX == 3
    assert CONSERVATIVE_FREE_SUFFIX == 4


def test_exact_and_conservative_suffixes_are_left_free():
    groups = _groups()
    exact = _convolution_safe_positions(
        [groups], RESTORE_ALL_EXCEPT_LAST3_MASK
    )[0]
    conservative = _convolution_safe_positions(
        [groups], RESTORE_ALL_EXCEPT_LAST4_MASK
    )[0]
    all_positions = set(range(10, 24))
    assert set(exact) == all_positions - {21, 22, 23}
    assert set(conservative) == all_positions - {20, 21, 22, 23}
    assert _control_mask_label(RESTORE_ALL_EXCEPT_LAST3_MASK).endswith(
        "last_3_prefix_tokens"
    )
    assert _control_mask_label(RESTORE_ALL_EXCEPT_LAST4_MASK).endswith(
        "last_4_prefix_tokens"
    )
