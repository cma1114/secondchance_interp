from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mechanistic.analyze_fresh_history_double_dissociation import (
    _fresh_winner_indices,
)


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_emitted_answer_uses_archived_schema() -> None:
    mappings = {
        "q1": {
            "new_to_original": {"A": "C", "B": "A", "C": "D", "D": "B"}
        }
    }
    remapped = {
        "q1": {
            "answer_new_letter": "B",
            "answer_original_content": "A",
        }
    }
    result = _fresh_winner_indices(
        remapped, ["q1"], mappings, seed_step5=False
    )
    assert result.tolist() == [0]


def test_qwen_emitted_answer_rejects_inconsistent_mapping() -> None:
    mappings = {
        "q1": {
            "new_to_original": {"A": "C", "B": "A", "C": "D", "D": "B"}
        }
    }
    remapped = {
        "q1": {
            "answer_new_letter": "B",
            "answer_original_content": "D",
        }
    }
    with pytest.raises(RuntimeError, match="internally inconsistent"):
        _fresh_winner_indices(remapped, ["q1"], mappings, seed_step5=False)


def test_seed_fresh_winner_uses_bare_remapped_score_argmax() -> None:
    mappings = {
        "q1": {
            "new_to_original": {"A": "C", "B": "A", "C": "D", "D": "B"}
        }
    }
    remapped = {"q1": {"aggregated_ad_logits": [0.0, 3.0, 1.0, 2.0]}}
    result = _fresh_winner_indices(remapped, ["q1"], mappings, seed_step5=True)
    assert result.tolist() == [0]


def test_archived_qwen_conflict_denominator_remains_273() -> None:
    option_root = (
        ROOT / "outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping"
    )
    remapped = json.loads(
        (option_root / "remapped_baseline_results.json").read_text()
    )["results"]
    mappings = {
        row["question_id"]: row
        for row in json.loads((option_root / "plan.json").read_text())["rows"]
    }
    with np.load(
        option_root / "fresh_history_double_dissociation/run/results.npz",
        allow_pickle=False,
    ) as arrays:
        qids = arrays["question_ids"].astype(str).tolist()
        old_winner = np.argsort(
            -arrays["baseline_logits"], axis=-1, kind="stable"
        )[:, 0]
    fresh_winner = _fresh_winner_indices(
        remapped, qids, mappings, seed_step5=False
    )
    assert int(np.sum(old_winner != fresh_winner)) == 273
