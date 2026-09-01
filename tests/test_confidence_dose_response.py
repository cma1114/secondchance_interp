from pathlib import Path

import numpy as np
import pytest

from mechanistic.analyze_confidence_dose_response import (
    CELLS,
    _load_cell,
    _load_split_ids,
    _univariate_bundle,
    derive_quantities,
    stable_descending_order,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stable_descending_order_uses_displayed_order_for_ties():
    logits = np.asarray([[3.0, 3.0, 2.0, 1.0], [1.0, 2.0, 2.0, 0.0]])
    expected = np.asarray([[0, 1, 2, 3], [1, 2, 0, 3]])
    np.testing.assert_array_equal(stable_descending_order(logits), expected)


def test_derived_push_margin_amplitude_and_targeting_have_declared_meaning():
    baseline = np.asarray([[4.0, 3.0, 2.0, 1.0]])
    direct = np.asarray(
        [
            [[8.0, 10.0, 9.0, 8.0]],  # Game
            [[10.0, 9.0, 8.0, 7.0]],  # Neutral
        ]
    )
    rank = np.asarray([[0, 1, 2, 3]])
    q = derive_quantities(baseline, direct, rank)

    np.testing.assert_allclose(q["confidence_c1"], [1.0])
    np.testing.assert_allclose(q["delta_rank"], [[-2.25, 0.75, 0.75, 0.75]])
    np.testing.assert_allclose(q["push_r1"], [2.25])
    np.testing.assert_allclose(q["policy_amplitude"], [np.sqrt(6.75)])
    np.testing.assert_allclose(q["targeting_cosine"], [1.0])
    np.testing.assert_allclose(q["neutral_old_w1_margin"], [1.0])
    np.testing.assert_allclose(q["margin_push"], [3.0])
    np.testing.assert_allclose(q["switch_game"], [1.0])
    np.testing.assert_allclose(q["switch_neutral"], [0.0])
    np.testing.assert_allclose(q["differential_switching"], [1.0])


def test_policy_quantities_are_invariant_to_conditionwise_logit_offsets():
    baseline = np.asarray([[4.0, 3.0, 2.0, 1.0]])
    direct = np.asarray(
        [
            [[8.0, 10.0, 9.0, 8.0]],
            [[10.0, 9.0, 8.0, 7.0]],
        ]
    )
    shifted = direct.copy()
    shifted[0] += 113.0
    shifted[1] -= 57.0
    rank = np.asarray([[0, 1, 2, 3]])
    original = derive_quantities(baseline, direct, rank)
    changed = derive_quantities(baseline, shifted, rank)
    for key in (
        "delta_rank",
        "push_r1",
        "margin_push",
        "policy_amplitude",
        "targeting_cosine",
        "neutral_old_w1_margin",
        "switch_game",
        "switch_neutral",
    ):
        np.testing.assert_allclose(original[key], changed[key])


def test_univariate_bundle_reports_logits_per_sd_and_standardized_beta():
    x = np.linspace(-2.0, 2.0, 40)
    y = 5.0 + 3.0 * ((x - x.mean()) / x.std(ddof=0))
    result = _univariate_bundle(
        x,
        {"y": y},
        draws=200,
        rng=np.random.default_rng(20260901),
    )["y"]
    assert result["raw_outcome_per_1sd_c1"]["value"] == pytest.approx(3.0)
    assert result["standardized_beta"]["value"] == pytest.approx(1.0)
    assert result["raw_outcome_per_1sd_c1"]["ci_low"] < 3.0
    assert result["raw_outcome_per_1sd_c1"]["ci_high"] > 3.0
    assert result["standardized_beta"]["ci_low"] == pytest.approx(1.0)
    assert result["standardized_beta"]["ci_high"] == pytest.approx(1.0)


def test_all_six_canonical_cells_pass_provenance_rank_and_split_gates():
    for spec in CELLS:
        validation, quantities = _load_cell(ROOT, spec)
        assert validation["n"] == 500
        assert validation["all_inputs_finite"]
        assert validation["rank_order_gate_passed"]
        assert validation["rank_order_exact_matches"] == 500
        splits = _load_split_ids(ROOT, spec["dataset_key"], quantities["question_ids"])
        assert int(splits["full"].sum()) == 500
        assert not np.any(splits["discovery"] & splits["confirmation"])
        assert np.all(splits["discovery"] | splits["confirmation"])
