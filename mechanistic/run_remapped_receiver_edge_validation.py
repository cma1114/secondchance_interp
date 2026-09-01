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
from .receiver_path_utils import ROLE_NAMES, locate_receiver_roles, role_positions
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward


def _cells(candidates: list[dict[str, Any]]) -> list[str]:
    return [
        f"{candidate['id']}__{source}"
        for candidate in candidates
        for source in ("selected", "matched_control")
    ]


def _initialize(
    path: Path, qids: list[str], cells: list[str]
) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Question IDs changed")
        if arrays["intervention_cells"].astype(str).tolist() != cells:
            raise ValueError("Intervention cells changed")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "intervention_cells": np.asarray(cells),
        "completed": np.zeros(n, dtype=bool),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "intervention_logits": np.full((2, len(cells), n, 4), np.nan, dtype=np.float32),
        "query_position_counts": np.zeros((2, len(cells), n), dtype=np.int16),
        "source_position_counts": np.zeros((2, len(cells), n), dtype=np.int16),
        "control_letters": np.full((2, n), "", dtype="<U1"),
    }


def run(
    config_path: Path,
    remapping_plan_path: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
    trusted_game_path: Path,
    trusted_neutral_path: Path,
    candidate_plan_path: Path,
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
    remapped_baseline = json.loads(remapped_baseline_path.read_text())["results"]
    trusted = [
        json.loads(trusted_game_path.read_text())["results"],
        json.loads(trusted_neutral_path.read_text())["results"],
    ]
    candidate_plan = json.loads(candidate_plan_path.read_text())
    candidates = candidate_plan["candidates"]
    if not candidates:
        raise ValueError("Candidate plan is empty")
    for candidate in candidates:
        if candidate["role"] not in ROLE_NAMES:
            raise ValueError(f"Unknown receiver role {candidate['role']!r}")
        if not candidate["blocks"]:
            raise ValueError(f"Candidate {candidate['id']} has no blocks")
        if any(int(block) not in range(4, 65, 4) for block in candidate["blocks"]):
            raise ValueError(f"Candidate {candidate['id']} has invalid blocks")
    cells = _cells(candidates)

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
    arrays = _initialize(result_path, qids, cells)
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
        for ci, (condition, batch) in enumerate(zip(CONDITIONS, batches)):
            natural = _aggregate_logits(
                _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                variant_ids,
            )
            width = int(batch["input_ids"].shape[1])
            located_rows: list[dict[str, Any]] = []
            for row, qid in enumerate(cohort):
                remapped_question = {
                    **questions[qid],
                    "options": {
                        new: questions[qid]["options"][old]
                        for new, old in mappings[qid]["new_to_original"].items()
                    },
                }
                w1 = baseline[qid].get("answer", baseline[qid].get("subject_answer"))
                w2 = remapped_baseline[qid]["answer_original_content"]
                located = locate_receiver_roles(
                    tokenizer,
                    batch["prompts"][row],
                    batch["messages"][row],
                    questions[qid],
                    remapped_question,
                    condition,
                    w1,
                    w2,
                    mappings[qid],
                )
                if located["ids"] != batch["token_rows"][row]:
                    raise RuntimeError("Receiver-role tokenization changed")
                located_rows.append(located)
                arrays["control_letters"][ci, qid_index[qid]] = located["control_letter"]
                arrays["same_batch_natural_logits"][ci, qid_index[qid]] = natural[row]
                arrays["trusted_natural_logits"][ci, qid_index[qid]] = np.asarray(
                    trusted[ci][qid]["aggregated_ad_logits"], dtype=np.float32
                )

            for candidate_index, candidate in enumerate(candidates):
                unpadded_queries = [
                    role_positions(located["roles"], candidate["role"])
                    for located in located_rows
                ]
                for source_offset, source_name in enumerate(("selected", "matched_control")):
                    cell_index = candidate_index * 2 + source_offset
                    specs_by_layer: dict[int, dict[int, dict[int, list[int]]]] = {}
                    for block in candidate["blocks"]:
                        rows: dict[int, dict[int, list[int]]] = {}
                        for row, located in enumerate(located_rows):
                            ids = located["ids"]
                            left_pad = width - len(ids)
                            queries = [left_pad + value for value in unpadded_queries[row]]
                            source_key = (
                                "selected_positions"
                                if source_name == "selected"
                                else "control_positions"
                            )
                            sources = [
                                left_pad + value for value in located[source_key]
                            ]
                            if queries:
                                rows[row] = {query: sources for query in queries}
                            qi = qid_index[cohort[row]]
                            arrays["query_position_counts"][ci, cell_index, qi] = len(queries)
                            arrays["source_position_counts"][ci, cell_index, qi] = len(sources)
                        if rows:
                            specs_by_layer[int(block) - 1] = rows
                    if specs_by_layer:
                        with BatchedSDPAQuerySourceAttentionAblator(parts, specs_by_layer):
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
                        arrays["intervention_logits"][ci, cell_index, qid_index[qid]] = intervened[row]

            cohort_audit["conditions"][condition] = {
                "rendered_prompt": batch["prompts"][0],
                "selected_line": located_rows[0]["selected_audit"],
                "control_line": located_rows[0]["control_audit"],
                "candidates": [
                    {
                        "id": candidate["id"],
                        "role": candidate["role"],
                        "blocks": candidate["blocks"],
                        "query_positions": role_positions(
                            located_rows[0]["roles"], candidate["role"]
                        ),
                        "query_tokens": [
                            tokenizer.decode([located_rows[0]["ids"][position]])
                            for position in role_positions(
                                located_rows[0]["roles"], candidate["role"]
                            )
                        ],
                    }
                    for candidate in candidates
                ],
            }

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        print(
            f"receiver edge validation: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={time.monotonic() - cohort_started:.1f}",
            flush=True,
        )
        if not audit_path.exists():
            audit_path.write_text(
                json.dumps(cohort_audit, indent=2, ensure_ascii=False) + "\n"
            )

    metadata = {
        "experiment": "canonical remapped downstream receiver causal edge validation",
        "config": config.as_dict(),
        "candidate_plan": str(candidate_plan_path),
        "candidates": candidates,
        "intervention_cells": cells,
        "n_questions": len(qids),
        "conditions": list(CONDITIONS),
        "historical_batch_size": config.batch_size,
        "complete_model_forwards_per_cohort": 2 * (1 + len(cells)),
        "intervention": (
            "Block only the candidate downstream query positions from reading either "
            "the complete first-presentation W1 option line or its matched unselected-line control "
            "in the candidate ordinary-attention blocks."
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
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.remapping_plan,
        args.baseline,
        args.remapped_baseline,
        args.trusted_game,
        args.trusted_neutral,
        args.candidate_plan,
        args.output_dir,
        args.max_cohorts,
    )


if __name__ == "__main__":
    main()

