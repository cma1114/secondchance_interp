import unittest

import numpy as np

from mechanistic.compare_answer_readouts import summarize_method


class ReadoutComparisonTests(unittest.TestCase):
    def test_paired_centered_fixed_rank_contrast(self):
        # Four questions, one per prior-answer stratum, and two layers.
        scores = np.zeros((3, 8, 2, 4), dtype=float)
        order = np.tile(np.arange(4), (8, 1))
        prior = np.repeat(np.arange(4), 2)
        # Game adds a centered [-3, -1, +1, +3] redistribution;
        # Neutral is zero. The macro summary should recover it exactly.
        scores[1] = np.asarray([-3.0, -1.0, 1.0, 3.0])[None, None, :]
        rows = summarize_method(scores, order, prior)
        for row, expected in zip(rows, (-3.0, -1.0, 1.0, 3.0)):
            np.testing.assert_allclose(row["mean"], expected)
            np.testing.assert_allclose(row["ci_low"], expected)
            np.testing.assert_allclose(row["ci_high"], expected)


if __name__ == "__main__":
    unittest.main()
