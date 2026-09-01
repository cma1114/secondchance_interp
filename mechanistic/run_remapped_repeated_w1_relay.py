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
from .downstream_source_intervention import BatchedSDPAQuerySourceAttentionAblator
from .io import atomic_save_npz
from .modeling import get_tokenizer, load_model_and_processor, resolve_answer_tokens
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_remapped_final_query_repeated_option_ablation import (
    ORDINARY_BLOCKS,
    _second_option_line_positions,
)


BLOCK_BANDS = {
    "04_16": (4, 8, 12, 16),
    "20_32": (20, 24, 28, 32),
    "36_48": (36, 40, 44, 48),
    "52_64": (52, 56, 60, 64),
}


def _cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = [
        {
            "id": "w1_all_later_pre_final__all_blocks",
            "source": "w1",
            "query_region": "all_later_pre_final",
            "blocks": list(ORDINARY_BLOCKS),
        },
        {
            "id": "w1_later_options__all_blocks",
            "source": "w1",
            "query_region": "later_options",
            "blocks": list(ORDINARY_BLOCKS),
        },
        {
            "id": "w1_post_options_pre_final__all_blocks",
            "source": "w1",
            "query_region": "post_options_pre_final",
            "blocks": list(ORDINARY_BLOCKS),
        },
    ]
    for label, blocks in BLOCK_BANDS.items():
        cells.append(
            {
                "id": f"w1_all_later_pre_final__blocks_{label}",
                "source": "w1",
                "query_region": "all_later_pre_final",
                "blocks": list(blocks),
            }
        )
    cells.append(
        {
            "id": "matched_control_post_options_pre_final__all_blocks",
            "source": "matched_control",
            "query_region": "post_options_pre_final",
            "blocks": list(ORDINARY_BLOCKS),
        }
    )
    return cells


INTERVENTION_CELLS = _cells()


def _stage_cells(stage: str) -> list[dict[str, Any]]:
    if stage == "all":
        return INTERVENTION_CELLS
    if stage == "prerequisite":
        return [
            cell
            for cell in INTERVENTION_CELLS
            if cell["id"] in {
                "w1_all_later_pre_final__all_blocks",
                "w1_later_options__all_blocks",
                "w1_post_options_pre_final__all_blocks",
                "matched_control_post_options_pre_final__all_blocks",
            }
        ]
    if stage == "bands":
        return [
            cell
            for cell in INTERVENTION_CELLS
            if "__blocks_" in cell["id"]
        ]
    raise ValueError(f"Unknown stage {stage!r}")


def _matched_control(
    positions: dict[str, list[int]], selected_displayed: str
) -> str:
    selected_count = len(positions[selected_displayed])
    return min(
        (letter for letter in LETTERS if letter != selected_displayed),
        key=lambda letter: (
            abs(len(positions[letter]) - selected_count),
            letter,
        ),
    )


def _locate_row(
    tokenizer: Any,
    prompt: str,
    remapped_question: dict[str, Any],
    mapping: dict[str, Any],
    w1: str,
) -> dict[str, Any]:
    ids = [
        int(value)
        for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ]
    positions, audit = _second_option_line_positions(
        tokenizer, prompt, remapped_question
    )
    selected_displayed = mapping["original_to_new"][w1]
    control_displayed = _matched_control(positions, selected_displayed)
    selected = positions[selected_displayed]
    control = positions[control_displayed]
    selected_end = max(selected)
    all_options_end = max(max(values) for values in positions.values())
    final_query = len(ids) - 1
    later_options = sorted(
        position
        for values in positions.values()
        for position in values
        if position > selected_end
    )
    post_options = list(range(all_options_end + 1, final_query))
    all_later = list(range(selected_end + 1, final_query))
    if set(later_options) & set(post_options):
        raise RuntimeError("Later-option and post-option regions overlap")
    if sorted(set(later_options) | set(post_options)) != all_later:
        raise RuntimeError("Later query regions do not partition all pre-final queries")
    return {
        "ids": ids,
        "source_positions": {
            "w1": selected,
            "matched_control": control,
        },
        "query_positions": {
            "all_later_pre_final": all_later,
            "later_options": later_options,
            "post_options_pre_final": post_options,
        },
        "w1_displayed_letter": selected_displayed,
        "control_displayed_letter": control_displayed,
        "selected_line": audit[selected_displayed],
        "control_line": audit[control_displayed],
        "all_options_end": all_options_end,
        "final_query": final_query,
    }


def _initialize(
    path: Path, qids: list[str], intervention_cells: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    cell_ids = [cell["id"] for cell in intervention_cells]
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Question IDs changed")
        if arrays["intervention_cells"].astype(str).tolist() != cell_ids:
            raise ValueError("Intervention cells changed")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "intervention_cells": np.asarray(cell_ids),
        "completed": np.zeros(n, dtype=bool),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "intervention_logits": np.full(
            (2, len(cell_ids), n, 4), np.nan, dtype=np.float32
        ),
        "query_position_counts": np.zeros(
            (2, len(cell_ids), n), dtype=np.int16
        ),
        "source_position_counts": np.zeros(
            (2, len(cell_ids), n), dtype=np.int16
        ),
        "w1_displayed_letters": np.full(n, "", dtype="<U1"),
        "control_displayed_letters": np.full(n, "", dtype="<U1"),
    }


def run(
    config_path: Path,
    remapping_plan_path: Path,
    baseline_path: Path,
    trusted_game_path: Path,
    trusted_neutral_path: Path,
    output_dir: Path,
    max_cohorts: int | None,
    cohort_start: int,
    cohort_stop: int | None,
    stage: str,
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
    all_qids = [row["id"] for row in manifest["questions"]]
    if cohort_start < 0:
        raise ValueError("cohort_start must be nonnegative")
    total_cohorts = len(all_qids) // config.batch_size
    stop = total_cohorts if cohort_stop is None else int(cohort_stop)
    if stop < cohort_start or stop > total_cohorts:
        raise ValueError(
            f"Invalid cohort range {cohort_start}:{stop} for {total_cohorts} cohorts"
        )
    qids = all_qids[
        cohort_start * config.batch_size : stop * config.batch_size
    ]
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
    if ordinary_blocks != ORDINARY_BLOCKS:
        raise RuntimeError(f"Unexpected ordinary-attention blocks: {ordinary_blocks}")

    intervention_cells = _stage_cells(stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids, intervention_cells)
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
            _build_batch(
                config, processor, tokenizer, questions, mappings, cohort, condition
            )
            for condition in CONDITIONS
        ]
        cohort_audit: dict[str, Any] = {"question_ids": cohort, "conditions": {}}

        for ci, (condition, batch) in enumerate(zip(CONDITIONS, batches)):
            natural = _aggregate_logits(
                _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                variant_ids,
            )
            width = int(batch["input_ids"].shape[1])
            located_rows: list[dict[str, Any]] = []
            for row, qid in enumerate(cohort):
                mapping = mappings[qid]
                remapped_question = {
                    **questions[qid],
                    "options": {
                        new: questions[qid]["options"][old]
                        for new, old in mapping["new_to_original"].items()
                    },
                }
                w1 = baseline[qid].get(
                    "answer", baseline[qid].get("subject_answer")
                )
                located = _locate_row(
                    tokenizer,
                    batch["prompts"][row],
                    remapped_question,
                    mapping,
                    w1,
                )
                if located["ids"] != batch["token_rows"][row]:
                    raise RuntimeError("Repeated-line tokenization changed")
                located_rows.append(located)
                qi = qid_index[qid]
                arrays["same_batch_natural_logits"][ci, qi] = natural[row]
                arrays["trusted_natural_logits"][ci, qi] = np.asarray(
                    trusted[ci][qid]["aggregated_ad_logits"], dtype=np.float32
                )
                arrays["w1_displayed_letters"][qi] = located[
                    "w1_displayed_letter"
                ]
                arrays["control_displayed_letters"][qi] = located[
                    "control_displayed_letter"
                ]

            for cell_index, cell in enumerate(intervention_cells):
                specs_by_layer: dict[int, dict[int, dict[int, list[int]]]] = {}
                for block in cell["blocks"]:
                    row_specs: dict[int, dict[int, list[int]]] = {}
                    for row, located in enumerate(located_rows):
                        left_pad = width - len(located["ids"])
                        queries = [
                            left_pad + value
                            for value in located["query_positions"][
                                cell["query_region"]
                            ]
                        ]
                        sources = [
                            left_pad + value
                            for value in located["source_positions"][cell["source"]]
                        ]
                        if queries:
                            row_specs[row] = {query: sources for query in queries}
                        qi = qid_index[cohort[row]]
                        arrays["query_position_counts"][ci, cell_index, qi] = len(
                            queries
                        )
                        arrays["source_position_counts"][ci, cell_index, qi] = len(
                            sources
                        )
                    if row_specs:
                        specs_by_layer[int(block) - 1] = row_specs
                if specs_by_layer:
                    with BatchedSDPAQuerySourceAttentionAblator(
                        parts, specs_by_layer
                    ):
                        intervened = _aggregate_logits(
                            _forward(
                                model,
                                parts,
                                batch["input_ids"],
                                batch["attention_mask"],
                            ),
                            variant_ids,
                        )
                else:
                    intervened = natural.copy()
                for row, qid in enumerate(cohort):
                    arrays["intervention_logits"][
                        ci, cell_index, qid_index[qid]
                    ] = intervened[row]

            cohort_audit["conditions"][condition] = {
                "rendered_prompt": batch["prompts"][0],
                "w1_displayed_letter": located_rows[0]["w1_displayed_letter"],
                "selected_line": located_rows[0]["selected_line"],
                "matched_control_displayed_letter": located_rows[0][
                    "control_displayed_letter"
                ],
                "matched_control_line": located_rows[0]["control_line"],
                "query_regions": {
                    name: {
                        "positions": values,
                        "tokens": [
                            tokenizer.decode([located_rows[0]["ids"][position]])
                            for position in values
                        ],
                    }
                    for name, values in located_rows[0]["query_positions"].items()
                },
                "cells": intervention_cells,
            }

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        print(
            f"repeated-W1 relay: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={time.monotonic() - cohort_started:.1f}",
            flush=True,
        )
        if not audit_path.exists():
            audit_path.write_text(
                json.dumps(cohort_audit, indent=2, ensure_ascii=False) + "\n"
            )

    metadata = {
        "experiment": "canonical remapped repeated-W1 downstream relay localization",
        "config": config.as_dict(),
        "n_questions": len(qids),
        "cohort_start": cohort_start,
        "cohort_stop": stop,
        "conditions": list(CONDITIONS),
        "stage": stage,
        "intervention_cells": intervention_cells,
        "ordinary_blocks": list(ORDINARY_BLOCKS),
        "historical_batch_size": config.batch_size,
        "complete_model_forwards_per_cohort": 2 * (1 + len(intervention_cells)),
        "complete_model_forwards_total": (
            (len(qids) // config.batch_size)
            * 2
            * (1 + len(intervention_cells))
        ),
        "intervention": (
            "Block ordinary-attention reads from the complete repeated W1 option "
            "line at every causally later pre-final query, decomposed by later-option "
            "versus post-option query region and by four ordinary-attention block bands. "
            "The already-completed final-query edge test is excluded."
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
    parser.add_argument("--cohort-start", type=int, default=0)
    parser.add_argument("--cohort-stop", type=int)
    parser.add_argument(
        "--stage", choices=("prerequisite", "bands", "all"), default="all"
    )
    args = parser.parse_args()
    run(
        args.config,
        args.remapping_plan,
        args.baseline,
        args.trusted_game,
        args.trusted_neutral,
        args.output_dir,
        args.max_cohorts,
        args.cohort_start,
        args.cohort_stop,
        args.stage,
    )


if __name__ == "__main__":
    main()
