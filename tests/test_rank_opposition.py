import numpy as np

from mechanistic.analyze_rank_opposition import RANK_AXIS, _rank_r2, _rank_slope


def test_rank_slope_recovers_axis_scale() -> None:
    aligned = np.stack([RANK_AXIS, 2.0 * RANK_AXIS])[:, None, :]
    np.testing.assert_allclose(_rank_slope(aligned), [[1.0], [2.0]])


def test_rank_axis_r2_is_one_for_pure_rank_opposition() -> None:
    aligned = np.stack([RANK_AXIS, -0.5 * RANK_AXIS])[:, None, :]
    labels = np.asarray([0, 1])
    # _rank_r2 requires all four answer-letter strata for its macro weighting.
    aligned = np.concatenate([aligned, aligned], axis=0)
    labels = np.asarray([0, 1, 2, 3])
    np.testing.assert_allclose(_rank_r2(aligned, labels), [1.0])


def test_rank_slope_sign_matches_progressively_less_suppression() -> None:
    progressively_less_suppressed = np.asarray([[-3.0, -1.0, 1.0, 3.0]])
    progressively_more_suppressed = progressively_less_suppressed[:, ::-1]
    assert _rank_slope(progressively_less_suppressed)[0] > 0
    assert _rank_slope(progressively_more_suppressed)[0] < 0
