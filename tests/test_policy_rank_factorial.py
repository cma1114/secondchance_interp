from __future__ import annotations

import numpy as np

from mechanistic.analyze_policy_rank_factorial import _ranked_choices


def test_policy_rank_choices_resolve_displayed_tie_before_rank_mapping():
    # Displayed A and B tie, so the model selects displayed A. Under this
    # remapping displayed A is semantic D, which is R1 in the synthetic rank
    # order. Reordering first would incorrectly choose semantic A (R2).
    logits = np.asarray([[[[1.0, 1.0, 0.0, 0.0]]]])
    mappings = [
        {"new_to_original": {"A": "D", "B": "A", "C": "B", "D": "C"}}
    ]
    rank_indices = np.asarray([[3, 0, 1, 2]])
    ranked = _ranked_choices(logits, mappings, rank_indices)
    assert ranked.tolist() == [[[[1.0, 0.0, 0.0, 0.0]]]]
