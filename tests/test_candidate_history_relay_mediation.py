from __future__ import annotations

import json

import numpy as np
import pytest

from mechanistic.analyze_candidate_history_relay_mediation import (
    _load_canonical_remapped_baseline,
    _ratio_interval,
)
from mechanistic.analyze_cue_attention_distribution import SOURCE_NAMES
from mechanistic.run_candidate_history_relay_mediation import (
    IDENTITY_MECHANISMS,
    JOINT_RELAY_MASK,
    MECHANISMS,
    RELAY_GROUPS,
    SCENARIO_IDS,
    SCENARIOS,
    _relay_groups,
    _selected_relay_positions,
)


def _rank_classes(start: int) -> list[list[int]]:
    return [[start], [start + 1], [start + 2], [start + 3, start + 4], [start + 5]]


def test_scenario_inventory_is_complete_and_unique():
    assert len(SCENARIOS) == 30
    assert len(set(SCENARIOS)) == len(SCENARIOS)
    assert len(set(SCENARIO_IDS)) == len(SCENARIO_IDS)
    assert MECHANISMS == ("none", "ordinary", "gla", "both")
    assert IDENTITY_MECHANISMS == ("ordinary", "gla", "both")
    assert SCENARIOS[:4] == (
        ("none", 0, "none"),
        ("complete_matching_block", 0, "none"),
        ("complete_balanced_wrong_block", 0, "none"),
        ("complete_balanced_wrong_block", JOINT_RELAY_MASK, "both"),
    )

    singles = {1 << index for index in range(5)}
    complements = {JOINT_RELAY_MASK ^ value for value in singles}
    pairs = {
        (1 << RELAY_GROUPS.index("second_option_newlines"))
        | (1 << RELAY_GROUPS.index("post_list_cue_and_query")),
        (1 << RELAY_GROUPS.index("second_option_newlines"))
        | (1 << RELAY_GROUPS.index("final_assistant_prefix")),
        (1 << RELAY_GROUPS.index("post_list_cue_and_query"))
        | (1 << RELAY_GROUPS.index("final_assistant_prefix")),
    }
    matching = [row for row in SCENARIOS if row[0] == "complete_matching_block"]
    both_masks = {mask for _source, mask, mechanism in matching if mechanism == "both"}
    assert both_masks == singles | complements | pairs | {JOINT_RELAY_MASK}
    for mechanism in ("ordinary", "gla"):
        masks = {mask for _source, mask, mode in matching if mode == mechanism}
        assert masks == singles | {JOINT_RELAY_MASK}


def test_relay_groups_are_disjoint_exhaustive_and_selectable():
    query_classes = [_rank_classes(10 + 6 * rank) for rank in range(4)]
    partition = [[] for _name in SOURCE_NAMES]
    partition[SOURCE_NAMES.index("final_assistant_prefix")] = list(range(42, 49))
    groups = _relay_groups(query_classes, partition, left_pad=0, final_query=50)
    assert tuple(groups) == RELAY_GROUPS
    flat = [position for name in RELAY_GROUPS for position in groups[name]]
    assert len(flat) == len(set(flat))
    assert set(flat) == set(range(10, 50))
    assert groups["second_option_semantics"] == [13, 14, 19, 20, 25, 26, 31, 32]
    assert groups["second_option_newlines"] == [15, 21, 27, 33]
    assert groups["post_list_cue_and_query"] == list(range(34, 42)) + [49]
    selected = _selected_relay_positions([groups, groups], relay_mask=0b10011)
    expected = sorted(
        groups["second_option_semantics"]
        + groups["second_option_newlines"]
        + groups["final_assistant_prefix"]
    )
    assert selected == {0: expected, 1: expected}


def test_canonical_remapped_baseline_rejects_second_chance_results(tmp_path):
    qids = ["q1", "q2"]
    canonical = {
        "results": {
            qid: {
                "answer_original_content": answer,
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "question"},
                ],
            }
            for qid, answer in zip(qids, ("A", "B"), strict=True)
        }
    }
    path = tmp_path / "remapped_baseline_results.json"
    path.write_text(json.dumps(canonical))
    assert set(_load_canonical_remapped_baseline(path, qids)) == set(qids)

    canonical["results"]["q2"]["messages"].extend(
        [
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "second chance"},
        ]
    )
    path.write_text(json.dumps(canonical))
    with pytest.raises(RuntimeError, match="one-presentation canonical"):
        _load_canonical_remapped_baseline(path, qids)


def test_discrete_ratio_with_many_zero_bootstrap_denominators_is_unstable():
    denominator = np.asarray([1.0, -1.0])
    numerator = np.asarray([0.5, -0.5])
    row = _ratio_interval(
        numerator,
        denominator,
        np.asarray([True, True]),
        seed=1,
        draws=1000,
    )
    assert row["stable_denominator"] is False
    assert row["ratio"] is None
    assert row["zero_denominator_bootstrap_fraction"] > 0.0
