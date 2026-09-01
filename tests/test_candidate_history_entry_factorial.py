from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mechanistic.analyze_candidate_history_entry_factorial import _shapley
from mechanistic.run_candidate_history_entry_factorial import (
    ALL_OPEN_MASK,
    TOKEN_CLASSES,
    _availability_label,
    _blocked_class_indices,
    _factorial_specs,
    _initialize,
    _partition_option_line,
    _validate_completed,
    _wrong_source_rank,
)


def test_canonical_option_line_partition_is_exact_and_disjoint():
    positions = list(range(100, 108))
    tokens = ["Ġ", "ĠA", ":", "ĠBy", "g", "ĠBy", "rd", "Ċ"]
    classes, audit = _partition_option_line(positions, tokens)

    assert classes == [[100], [101], [102], [103, 104, 105, 106], [107]]
    assert [value for group in classes for value in group] == positions
    assert set(audit) == set(TOKEN_CLASSES)
    assert audit["semantic"]["tokens"] == ["ĠBy", "g", "ĠBy", "rd"]


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        (["X", "ĠA", ":", "Ġx", "Ċ"], "leading-space"),
        (["Ġ", "ĠA", ";", "Ġx", "Ċ"], "colon"),
        (["Ġ", "ĠA", ":", "Ġx", "X"], "newline"),
    ],
)
def test_option_line_partition_rejects_noncanonical_boundaries(tokens, message):
    with pytest.raises(ValueError, match=message):
        _partition_option_line(list(range(len(tokens))), tokens)


def test_all_32_masks_are_complete_and_labels_are_unambiguous():
    blocked = {_blocked_class_indices(mask) for mask in range(32)}
    assert len(blocked) == 32
    assert _blocked_class_indices(0) == (0, 1, 2, 3, 4)
    assert _blocked_class_indices(ALL_OPEN_MASK) == ()
    assert _availability_label(0) == "none"
    assert _availability_label(ALL_OPEN_MASK) == "+".join(TOKEN_CLASSES)


def test_wrong_source_assignment_never_matches_and_is_globally_balanced():
    offsets = []
    for question_index in range(500):
        for target_rank in range(4):
            wrong = _wrong_source_rank(question_index, target_rank)
            assert wrong != target_rank
            offsets.append((wrong - target_rank) % 4)
    counts = np.bincount(offsets, minlength=4)
    assert counts[0] == 0
    assert counts[1:].max() - counts[1:].min() <= 1


def test_factorial_specs_block_exact_destination_classes_and_matched_sources():
    sources = [
        [[0, 1], [2, 3], [4, 5], [6, 7]],
        [[10], [11], [12], [13]],
    ]
    queries = [
        [
            [[20], [21], [22], [23, 24], [25]],
            [[30], [31], [32], [33], [34]],
            [[40], [41], [42], [43], [44]],
            [[50], [51], [52], [53], [54]],
        ],
        [
            [[120], [121], [122], [123], [124]],
            [[130], [131], [132], [133], [134]],
            [[140], [141], [142], [143], [144]],
            [[150], [151], [152], [153], [154]],
        ],
    ]
    wrong = np.asarray([[1, 2, 3, 0], [2, 3, 0, 1]])
    # Only semantic receivers remain open: every structural destination is blocked.
    mask = 1 << TOKEN_CLASSES.index("semantic")
    specs = _factorial_specs((3, 7), sources, queries, wrong, mask, "matching")
    assert set(specs) == {3, 7}
    assert specs[3][0][20] == [0, 1]
    assert specs[3][0][21] == [0, 1]
    assert 23 not in specs[3][0]
    assert 24 not in specs[3][0]
    assert specs[3][0][30] == [2, 3]

    wrong_specs = _factorial_specs(
        (3,), sources, queries, wrong, mask, "balanced_wrong"
    )
    assert wrong_specs[3][0][20] == [2, 3]
    assert wrong_specs[3][1][120] == [12]


def test_all_open_mask_is_not_a_fake_intervention():
    with pytest.raises(ValueError, match="all-open"):
        _factorial_specs(
            (3,),
            [[[0], [1], [2], [3]]],
            [[[[4], [5], [6], [7], [8]]] * 4],
            np.asarray([[1, 2, 3, 0]]),
            ALL_OPEN_MASK,
            "matching",
        )


def test_checkpoint_validation_executes_real_all_open_identity(tmp_path: Path):
    arrays = _initialize(tmp_path / "new.npz", ["q0"])
    arrays["completed"][0] = True
    arrays["baseline_logits"][0] = [1, 2, 3, 4]
    natural = np.asarray([[[1, 2, 3, 4]], [[4, 3, 2, 1]]], dtype=np.float32)
    arrays["trusted_natural_logits"][:] = natural
    arrays["natural_logits"][:] = natural
    arrays["factorial_logits"][:] = 0.0
    arrays["factorial_logits"][:, :, ALL_OPEN_MASK, 0] = natural[:, None, 0]
    assert _validate_completed(arrays)["all_open_identity_max_abs_error"] == 0.0

    arrays["factorial_logits"][0, 1, ALL_OPEN_MASK, 0, 0] += 0.25
    with pytest.raises(RuntimeError, match="identity error"):
        _validate_completed(arrays)


def test_five_factor_shapley_is_exact_for_main_effects_and_interaction():
    metric = np.empty((32, 3), dtype=float)
    for mask in range(32):
        bits = np.asarray([(mask >> index) & 1 for index in range(5)])
        metric[mask] = (
            bits @ np.arange(1.0, 6.0)
            + 4.0 * bits[1] * bits[3]
            + np.asarray([0.0, 10.0, -2.0])
        )
    values = _shapley(metric)
    np.testing.assert_allclose(values.sum(0), metric[31] - metric[0], atol=1e-12)
    np.testing.assert_allclose(values[:, 0], [1, 4, 3, 6, 5], atol=1e-12)
