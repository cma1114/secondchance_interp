from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mechanistic.analyze_contextual_option_representations import (
    OPTION_ANCHORS,
    _aligned_remapped_indices,
    _candidate_array,
    _load,
    _retrieval_curve,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--remapped", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original, original_meta = _load(args.original)
    remapped, remapped_meta = _load(args.remapped)
    qids = list(original_meta["question_ids"])
    if qids != list(remapped_meta["question_ids"]):
        raise ValueError("Original/remapped question orders differ")
    arrays = {}
    for anchor in OPTION_ANCHORS:
        _, per_question = _retrieval_curve(
            _candidate_array(original, original_meta, anchor),
            _candidate_array(remapped, remapped_meta, anchor),
            _aligned_remapped_indices(remapped_meta, qids),
        )
        arrays[anchor] = per_question.astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, question_ids=np.asarray(qids), **arrays)


if __name__ == "__main__":
    main()
