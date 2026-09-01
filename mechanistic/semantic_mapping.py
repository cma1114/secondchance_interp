from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import LETTERS


def _mapping_dict(row: Mapping[str, Any]) -> Mapping[str, str]:
    """Return the displayed-letter mapping from any frozen-plan row shape."""

    nested = row.get("second_mapping", row)
    if "new_to_original" in nested:
        return nested["new_to_original"]
    if "original_to_new" in nested:
        return {new: original for original, new in nested["original_to_new"].items()}
    raise KeyError("Mapping row has neither new_to_original nor original_to_new")


def displayed_argmax_to_semantic_indices(
    displayed_logits: np.ndarray,
    mapping_rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Resolve displayed A-D ties before mapping winners to semantic content.

    ``displayed_logits`` must have shape ``(..., question, 4)`` and
    ``mapping_rows`` must follow that question axis. NumPy's displayed-order
    argmax supplies the model's A-before-B-before-C-before-D exact-tie rule;
    only the selected displayed letter is then mapped back to original content.
    """

    values = np.asarray(displayed_logits)
    if values.ndim < 2 or values.shape[-1] != len(LETTERS):
        raise ValueError(f"Expected (..., question, 4) logits, got {values.shape}")
    if values.shape[-2] != len(mapping_rows):
        raise ValueError(
            f"Question axis {values.shape[-2]} does not match {len(mapping_rows)} mappings"
        )
    displayed = values.argmax(axis=-1)
    semantic = np.empty(displayed.shape, dtype=np.int64)
    for question_index, row in enumerate(mapping_rows):
        new_to_original = _mapping_dict(row)
        lookup = np.asarray(
            [LETTERS.index(new_to_original[letter]) for letter in LETTERS],
            dtype=np.int64,
        )
        semantic[..., question_index] = lookup[displayed[..., question_index]]
    return semantic


def align_displayed_logits_to_semantic(
    displayed_logits: np.ndarray,
    mapping_rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Reorder logits into original semantic A-D coordinates without argmax."""

    values = np.asarray(displayed_logits)
    if values.ndim < 2 or values.shape[-1] != len(LETTERS):
        raise ValueError(f"Expected (..., question, 4) logits, got {values.shape}")
    if values.shape[-2] != len(mapping_rows):
        raise ValueError(
            f"Question axis {values.shape[-2]} does not match {len(mapping_rows)} mappings"
        )
    aligned = np.empty_like(values)
    for question_index, row in enumerate(mapping_rows):
        new_to_original = _mapping_dict(row)
        original_to_new = {original: new for new, original in new_to_original.items()}
        for semantic_index, semantic_letter in enumerate(LETTERS):
            aligned[..., question_index, semantic_index] = values[
                ..., question_index, LETTERS.index(original_to_new[semantic_letter])
            ]
    return aligned
