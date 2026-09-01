import unittest

import numpy as np

from mechanistic.analyze_jlens_answer_content import _align_and_center, baseline_rank_order


class FakeData:
    question_ids = ["q1"]
    metadata = {("q1", "baseline"): {"full_vocab_top_token": "C"}}

    def condition(self, name):
        assert name == "baseline"
        # A has the largest reconstructed canonical logit, but C was the
        # actually generated answer and must remain the fixed original winner.
        return np.asarray([[[0.0, 0.0, 0.0, 0.0], [9.0, 2.0, 3.0, 1.0]]])


class AnswerContentAnalysisTests(unittest.TestCase):
    def test_generated_baseline_answer_defines_rank_one(self):
        order, prior = baseline_rank_order(FakeData())
        np.testing.assert_array_equal(prior, [2])
        np.testing.assert_array_equal(order, [[2, 0, 1, 3]])

    def test_center_then_align(self):
        scores = np.asarray([[[1.0, 2.0, 3.0, 4.0]]])
        order = np.asarray([[3, 2, 1, 0]])
        result = _align_and_center(scores, order)
        np.testing.assert_allclose(result, [[[1.5, 0.5, -0.5, -1.5]]])


if __name__ == "__main__":
    unittest.main()
