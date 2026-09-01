from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSDPAFinalQueryAttentionAblator
from .io import atomic_save_npz
from .modeling import get_tokenizer, load_model_and_processor, resolve_answer_tokens
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_fixed_a_final_query_edge_ablation import (
    BLOCK_SETS,
    INTERVENTION_CELLS,
    _option_line_positions,
)


def _matched_control(
    option_positions: dict[str, list[int]], selected: str
) -> str:
    selected_count = len(option_positions[selected])
    alternatives = [letter for letter in LETTERS if letter != selected]
    return min(
        alternatives,
        key=lambda letter: (
            abs(len(option_positions[letter]) - selected_count),
            letter,
        ),
    )


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Question IDs changed")
        if arrays["intervention_cells"].astype(str).tolist() != list(INTERVENTION_CELLS):
            raise ValueError("Intervention cells changed")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "intervention_cells": np.asarray(INTERVENTION_CELLS),
        "completed": np.zeros(n, dtype=bool),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "intervention_logits": np.full(
            (2, len(INTERVENTION_CELLS), n, 4), np.nan, dtype=np.float32
        ),
        "selected_position_counts": np.zeros((2, n), dtype=np.int16),
        "control_position_counts": np.zeros((2, n), dtype=np.int16),
        "control_letters": np.full((2, n), "", dtype="<U1"),
    }


def run(
    config_path: Path,
    remapping_plan_path: Path,
    baseline_path: Path,
    trusted_game_path: Path,
    trusted_neutral_path: Path,
    output_dir: Path,
    max_cohorts: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires exact historical batch-size-4 SDPA execution")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw_qwen_chatml serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    if max_cohorts is not None:
        qids = qids[: int(max_cohorts) * config.batch_size]
    mappings = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    baseline = json.loads(baseline_path.read_text())["results"]
    trusted = [
        json.loads(trusted_game_path.read_text())["results"],
        json.loads(trusted_neutral_path.read_text())["results"],
    ]

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    ordinary_blocks = tuple(
        index + 1
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if ordinary_blocks != tuple(range(4, 65, 4)):
        raise RuntimeError(f"Unexpected ordinary-attention blocks: {ordinary_blocks}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = output_dir / "prompt_audit.json"
    started = time.monotonic()

    for start in range(0, len(qids), config.batch_size):
        cohort = qids[start : start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if np.all(arrays["completed"][indices]):
            continue
        cohort_started = time.monotonic()
        batches = [
            _build_batch(config, processor, tokenizer, questions, mappings, cohort, condition)
            for condition in CONDITIONS
        ]

        cohort_audit: dict[str, Any] = {"question_ids": cohort, "conditions": {}}
        for ci, batch in enumerate(batches):
            natural = _aggregate_logits(
                _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                variant_ids,
            )
            width = int(batch["input_ids"].shape[1])
            selected_positions: list[list[int]] = []
            control_positions: list[list[int]] = []
            controls: list[str] = []
            audits: list[dict[str, Any]] = []
            for row, qid in enumerate(cohort):
                ids = batch["token_rows"][row]
                left_pad = width - len(ids)
                option_positions, option_audit = _option_line_positions(
                    tokenizer, batch["prompts"][row], questions[qid]
                )
                selected = baseline[qid].get("answer", baseline[qid].get("subject_answer"))
                if selected not in LETTERS:
                    raise RuntimeError(f"Invalid W1 for {qid}: {selected!r}")
                control = _matched_control(option_positions, selected)
                selected_positions.append([left_pad + pos for pos in option_positions[selected]])
                control_positions.append([left_pad + pos for pos in option_positions[control]])
                controls.append(control)
                audits.append({
                    "W1_original_letter": selected,
                    "selected_line": option_audit[selected],
                    "control_letter": control,
                    "control_line": option_audit[control],
                })

            for local, qid in enumerate(cohort):
                qi = qid_index[qid]
                arrays["same_batch_natural_logits"][ci, qi] = natural[local]
                arrays["trusted_natural_logits"][ci, qi] = np.asarray(
                    trusted[ci][qid]["aggregated_ad_logits"], dtype=np.float32
                )
                arrays["selected_position_counts"][ci, qi] = len(selected_positions[local])
                arrays["control_position_counts"][ci, qi] = len(control_positions[local])
                arrays["control_letters"][ci, qi] = controls[local]

            for cell_index, cell in enumerate(INTERVENTION_CELLS):
                if cell.endswith("_matched_control"):
                    block_name = cell.removesuffix("_matched_control")
                    positions = control_positions
                elif cell.endswith("_selected"):
                    block_name = cell.removesuffix("_selected")
                    positions = selected_positions
                else:
                    raise RuntimeError(f"Unknown cell {cell}")
                layer_specs = {
                    block - 1: {
                        row: source_positions
                        for row, source_positions in enumerate(positions)
                    }
                    for block in BLOCK_SETS[block_name]
                }
                with BatchedSDPAFinalQueryAttentionAblator(parts, layer_specs):
                    intervened = _aggregate_logits(
                        _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                        variant_ids,
                    )
                for local, qid in enumerate(cohort):
                    arrays["intervention_logits"][ci, cell_index, qid_index[qid]] = intervened[local]

            cohort_audit["conditions"][CONDITIONS[ci]] = {
                "rendered_prompt": batch["prompts"][0],
                "rows": audits,
                "final_query_position_physical": width - 1,
                "final_query_token": tokenizer.decode([int(batch["input_ids"][0, -1])]),
            }

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        print(
            f"remapped final-query edge: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={time.monotonic() - cohort_started:.1f}",
            flush=True,
        )
        if not audit_path.exists():
            audit_path.write_text(json.dumps(cohort_audit, indent=2, ensure_ascii=False) + "\n")

    metadata = {
        "experiment": "canonical remapped final-query W1 option-line attention-edge ablation",
        "config": config.as_dict(),
        "config_path": str(config_path),
        "remapping_plan": str(remapping_plan_path),
        "baseline": str(baseline_path),
        "trusted_game": str(trusted_game_path),
        "trusted_neutral": str(trusted_neutral_path),
        "n_questions": len(qids),
        "conditions": list(CONDITIONS),
        "intervention_cells": list(INTERVENTION_CELLS),
        "block_sets_one_based": {key: list(value) for key, value in BLOCK_SETS.items()},
        "historical_batch_size": config.batch_size,
        "complete_model_forwards_per_cohort": 2 * (1 + len(INTERVENTION_CELLS)),
        "intervention": (
            "At only the final pre-answer query, block every head in the selected ordinary-attention "
            "blocks from reading the complete first-presentation option line containing W1. "
            "Matched controls block an unselected option line with nearest token count."
        ),
        "elapsed_seconds_after_load": time.monotonic() - started,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.remapping_plan,
        args.baseline,
        args.trusted_game,
        args.trusted_neutral,
        args.output_dir,
        args.max_cohorts,
    )


if __name__ == "__main__":
    main()
