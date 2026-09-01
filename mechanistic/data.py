from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .io import read_metadata, shard_path


@dataclass
class ActivationDataset:
    question_ids: list[str]
    conditions: list[str]
    logits: np.ndarray  # question, condition, layer, option
    metadata: dict[tuple[str, str], dict]

    def condition(self, name: str) -> np.ndarray:
        return self.logits[:, self.conditions.index(name)]


def available_question_ids(output_dir: str | Path, conditions: list[str]) -> list[str]:
    root = Path(output_dir) / "shards"
    sets = [{p.stem for p in (root / c).glob("*.npz")} for c in conditions]
    return sorted(set.intersection(*sets)) if sets else []


def load_activation_dataset(output_dir: str | Path, conditions: list[str]) -> ActivationDataset:
    output_dir = Path(output_dir)
    run_metadata_path = output_dir / "run_metadata.json"
    decision_mode = "unrestricted"
    variant_indices = None
    if run_metadata_path.exists():
        run_metadata = json.loads(run_metadata_path.read_text())
        decision_mode = run_metadata.get("config", {}).get("decision_mode", "unrestricted")
        if "variant_layout" in run_metadata:
            layout = run_metadata["variant_layout"]
            variant_indices = {
                letter: [index for index, row in enumerate(layout) if row["letter"] == letter]
                for letter in "ABCD"
            }
    qids = available_question_ids(output_dir, conditions)
    if not qids:
        raise FileNotFoundError(f"No complete question shards for {conditions} in {output_dir}")
    rows, metadata = [], {}
    expected_shape = None
    for qid in qids:
        condition_rows = []
        for condition in conditions:
            with np.load(shard_path(output_dir, condition, qid), allow_pickle=False) as z:
                if variant_indices is not None:
                    variants = z["variant_logits"].astype(np.float64)
                    family_logits = []
                    for letter in "ABCD":
                        selected = variants[:, variant_indices[letter]]
                        maximum = selected.max(axis=-1, keepdims=True)
                        family_logits.append(
                            (maximum + np.log(np.exp(selected - maximum).sum(axis=-1, keepdims=True)))[:, 0]
                        )
                    logits = np.stack(family_logits, axis=-1)
                else:
                    logits = z["canonical_logits"].astype(np.float64)
                row_metadata = read_metadata(z)
                if decision_mode == "ad_constrained":
                    row_metadata["analysis_answer"] = "ABCD"[int(logits[-1].argmax())]
                    row_metadata["analysis_decision_mode"] = "ad_constrained"
                metadata[(qid, condition)] = row_metadata
            if expected_shape is None:
                expected_shape = logits.shape
            if logits.shape != expected_shape:
                raise ValueError(f"Inconsistent logit shape for {condition}/{qid}: {logits.shape}")
            condition_rows.append(logits)
        rows.append(condition_rows)
    return ActivationDataset(qids, conditions, np.asarray(rows), metadata)


def decision_letter(metadata: dict) -> str:
    """Return the answer used by analysis, honoring explicit constrained runs."""
    return metadata.get("analysis_answer", metadata["full_vocab_top_token"].strip())


def load_residual_layer(output_dir: str | Path, condition: str, qids: list[str], layer: int) -> np.ndarray:
    values = []
    for qid in qids:
        with np.load(shard_path(output_dir, condition, qid), allow_pickle=False) as z:
            if "residuals" not in z.files:
                raise KeyError(f"Residuals were not saved in {condition}/{qid}")
            values.append(z["residuals"][layer].astype(np.float32))
    return np.asarray(values)
