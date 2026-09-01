from __future__ import annotations

import numpy as np

from mechanistic.analyze_strategic_switching_evidence import (
    _destination_summary,
    _ratio_interval,
)


def test_destination_summary_separates_fixed_and_switch_denominators() -> None:
    summary = _destination_summary(
        choices=np.asarray(list("CBAD")),
        old_winner=np.asarray(list("AAAA")),
        old_runner_up=np.asarray(list("BBBB")),
        fresh_winner=np.asarray(list("CCCC")),
        mask=np.ones(4, dtype=bool),
        rng=np.random.default_rng(4),
        draws=100,
    )

    assert summary["n_questions"] == 4
    assert summary["n_switches"] == 3
    fixed = summary["fixed_denominator"]
    assert fixed["fresh_winner_choice"]["mean"] == 0.25
    assert fixed["old_runner_up_choice"]["mean"] == 0.25
    assert fixed["old_winner_choice"]["mean"] == 0.25
    assert fixed["other_choice"]["mean"] == 0.25
    switched = summary["among_switches"]
    assert switched["fresh_winner_choice"]["mean"] == 1 / 3
    assert switched["old_runner_up_choice"]["mean"] == 1 / 3
    assert switched["fresh_minus_old_runner_up"]["mean"] == 0.0


def test_ratio_interval_uses_paired_question_mass() -> None:
    result = _ratio_interval(
        numerator=np.asarray([0.5, 0.0, 1.0]),
        denominator=np.asarray([1.0, 0.5, 1.0]),
        rng=np.random.default_rng(8),
        draws=100,
    )

    assert result["mean"] == 0.6
    assert result["n"] == 3
