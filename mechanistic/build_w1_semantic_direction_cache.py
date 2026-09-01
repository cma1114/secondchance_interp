from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .collect_contextual_option_representations import ANCHORS
from .io import atomic_save_npz


LETTERS = "ABCD"


def _load(path: Path) -> tuple[np.memmap, dict]:
    return (
        np.load(path / "position_residuals.npy", mmap_mode="r"),
        json.loads((path / "metadata.json").read_text()),
    )


def build(
    manifest_path: Path,
    baseline_path: Path,
    split_specs: list[tuple[Path, Path, Path, Path]],
    output: Path,
) -> None:
    all_qids = [row["id"] for row in json.loads(manifest_path.read_text())["questions"]]
    baseline = json.loads(baseline_path.read_text())["results"]
    line_indices = [ANCHORS.index(f"line_end_{letter}") for letter in LETTERS]
    direction_by_qid: dict[str, np.ndarray] = {}

    for original_path, map1_path, map2_path, map3_path in split_specs:
        loaded = [_load(path) for path in (original_path, map1_path, map2_path, map3_path)]
        qids = loaded[0][1]["question_ids"]
        if any(item[1]["question_ids"] != qids for item in loaded[1:]):
            raise ValueError("Question IDs differ across the four mappings")
        for qi, qid in enumerate(qids):
            aligned = []
            for mapping_index, (values, metadata) in enumerate(loaded):
                option_values = values[qi][:, line_indices].astype(np.float32)
                if mapping_index == 0:
                    indices = list(range(4))
                else:
                    mapping = metadata["mappings"][qid]["original_to_new"]
                    indices = [LETTERS.index(mapping[content]) for content in LETTERS]
                aligned.append(option_values[:, indices])
            average = np.stack(aligned).mean(axis=0, dtype=np.float32)
            centered = average - average.mean(axis=1, keepdims=True)
            w1 = LETTERS.index(baseline[qid]["answer"])
            vector = centered[:, w1]
            norm = np.linalg.norm(vector, axis=-1, keepdims=True)
            if np.any(norm <= 1e-8):
                raise ValueError(f"Degenerate W1 vector: {qid}")
            direction_by_qid[qid] = vector / norm

    if set(direction_by_qid) != set(all_qids):
        raise ValueError(f"Directions cover {len(direction_by_qid)}/{len(all_qids)} questions")
    output.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(all_qids), 4):
        group = all_qids[start : start + 4]
        directions = np.stack([direction_by_qid[qid] for qid in group], axis=1)
        atomic_save_npz(
            output / f"cohort_{start:04d}.npz",
            question_ids=np.asarray(group),
            directions=directions.astype(np.float32),
            target_answer=np.asarray("w1"),
        )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "definition": (
                    "Exact four-mapping, semantic-content-aligned, within-question-centered "
                    "option-closing-newline direction for W1, normalized separately at each layer."
                ),
                "n_questions": len(all_qids),
                "n_layers": 64,
                "target_answer": "w1",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--split", action="append", nargs=4, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(
        args.manifest,
        args.baseline,
        [tuple(row) for row in args.split],
        args.output,
    )


if __name__ == "__main__":
    main()
