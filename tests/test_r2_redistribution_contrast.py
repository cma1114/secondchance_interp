from __future__ import annotations

import numpy as np

from mechanistic.analyze_r2_redistribution_contrast import contrast, paired_rank_effects


def test_rank_alignment_and_paired_contrast(tmp_path):
    # Two questions. Displayed final logits are constructed so that, after
    # within-question centering and rank alignment, the paired Game-minus-
    # Neutral rank effects are exactly [-2, +1, 0, +1] for question 0 and
    # [0, 0, 0, 0] for question 1.
    direct = np.zeros((2, 2, 4), dtype=np.float32)
    # Question 0: rank order maps displayed letters (B, D, A, C) -> R1..R4.
    rank_order = np.asarray([[1, 3, 0, 2], [0, 1, 2, 3]], dtype=np.int64)
    # Game question 0 (displayed A..D): centered values chosen directly.
    direct[0, 0] = [0.0, -2.0, 1.0, 1.0]  # R1=B=-2, R2=D=+1, R3=A=0, R4=C=+1
    direct[1, 0] = [0.0, 0.0, 0.0, 0.0]
    direct[0, 1] = [3.0, 3.0, 3.0, 3.0]  # centers to zero
    direct[1, 1] = [-1.0, -1.0, -1.0, -1.0]  # centers to zero

    path = tmp_path / "results.npz"
    np.savez(
        path,
        direct_logits=direct,
        rank_order=rank_order,
        question_ids=np.asarray(["q0", "q1"]),
    )
    effects, question_ids = paired_rank_effects(path)
    assert question_ids == ["q0", "q1"]
    assert np.allclose(effects[0], [-2.0, 1.0, 0.0, 1.0])
    assert np.allclose(effects[1], [0.0, 0.0, 0.0, 0.0])

    # Contrast: R2 gain minus mean(R3, R4) gain = 1 - 0.5 = 0.5 for question 0.
    values = contrast(effects)
    assert np.allclose(values, [0.5, 0.0])


def test_common_offset_is_removed_by_centering(tmp_path):
    # Adding a shared constant to all four displayed logits of one condition
    # must not change any rank effect.
    rng = np.random.default_rng(3)
    base = rng.normal(size=(2, 3, 4)).astype(np.float32)
    shifted = base.copy()
    shifted[0] += 7.25  # common offset on every Game logit
    rank_order = np.tile(np.arange(4), (3, 1)).astype(np.int64)
    ids = np.asarray(["a", "b", "c"])

    p1 = tmp_path / "base.npz"
    p2 = tmp_path / "shifted.npz"
    np.savez(p1, direct_logits=base, rank_order=rank_order, question_ids=ids)
    np.savez(p2, direct_logits=shifted, rank_order=rank_order, question_ids=ids)
    e1, _ = paired_rank_effects(p1)
    e2, _ = paired_rank_effects(p2)
    assert np.allclose(e1, e2, atol=1e-6)
