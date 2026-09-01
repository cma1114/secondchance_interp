from __future__ import annotations

import argparse
import hashlib
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
from .run_all_candidate_matched_relay import _specs
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_fixed_a_final_query_edge_ablation import _option_line_positions


BAND_NAMES = ("late_52_64", "full_04_64")
BAND_BLOCKS = {
    "late_52_64": (52, 56, 60, 64),
    "full_04_64": tuple(range(4, 65, 4)),
}
RANKS = ("R1", "R2", "R3", "R4")


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses a different question order")
        if arrays["bands"].astype(str).tolist() != list(BAND_NAMES):
            raise ValueError("Existing checkpoint uses different layer bands")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "bands": np.asarray(BAND_NAMES),
        "completed": np.zeros(n, dtype=bool),
        "rank_contents": np.full((n, 4), "", dtype="<U1"),
        "baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "matched_logits": np.full((2, 2, 4, n, 4), np.nan, dtype=np.float32),
        "control_logits": np.full((2, 2, 4, n, 4), np.nan, dtype=np.float32),
        "joint_matched_logits": np.full((2, 2, n, 4), np.nan, dtype=np.float32),
        "joint_control_logits": np.full((2, 2, n, 4), np.nan, dtype=np.float32),
        "source_position_counts": np.zeros((n, 4), dtype=np.int16),
        "query_position_counts": np.zeros((n, 4), dtype=np.int16),
        "control_position_counts": np.zeros((n, 4), dtype=np.int16),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
    }


def _run_lesion(
    model: Any,
    parts: Any,
    batch: dict[str, Any],
    variant_ids: dict[str, list[int]],
    layer_indices: tuple[int, ...],
    sources: list[list[list[int]]],
    queries: list[list[list[int]]],
    ranks: tuple[int, ...],
    controls: bool,
) -> np.ndarray:
    specs = _specs(layer_indices, sources, queries, ranks, controls)
    with BatchedSDPAQuerySourceAttentionAblator(parts, specs):
        logits = _aggregate_logits(
            _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
            variant_ids,
        )
    if not np.all(np.isfinite(logits)):
        raise RuntimeError("Non-finite matching-edge intervention logits")
    return logits


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
        raise ValueError("Requires action-matched incorrect/lost feedback")
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

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids)
    qid_index = {qid: index for index, qid in enumerate(qids)}

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    ordinary = tuple(
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if tuple(index + 1 for index in ordinary) != tuple(range(4, 65, 4)):
        raise RuntimeError("Unexpected ordinary-attention layer inventory")
    band_indices = {
        name: tuple(block - 1 for block in BAND_BLOCKS[name]) for name in BAND_NAMES
    }

    started = time.monotonic()
    durations: list[float] = []
    audit_path = output_dir / "prompt_audit.json"
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

        for qid in cohort:
            qi = qid_index[qid]
            logits = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=float)
            order = np.argsort(-logits, kind="stable")
            arrays["baseline_logits"][qi] = logits
            arrays["rank_contents"][qi] = np.asarray([LETTERS[int(i)] for i in order])

        for ci, (condition, batch) in enumerate(zip(CONDITIONS, batches)):
            width = int(batch["input_ids"].shape[1])
            source_physical: list[list[list[int]]] = []
            query_physical: list[list[list[int]]] = []
            audits: list[dict[str, Any]] = []
            for row, qid in enumerate(cohort):
                qi = qid_index[qid]
                left_pad = width - len(batch["token_rows"][row])
                remapped = {
                    **questions[qid],
                    "options": {
                        new: questions[qid]["options"][old]
                        for new, old in mappings[qid]["new_to_original"].items()
                    },
                }
                first_positions, first_audit = _option_line_positions(
                    tokenizer, batch["prompts"][row], questions[qid]
                )
                second_positions, second_audit = _option_line_positions(
                    tokenizer, batch["prompts"][row], remapped
                )
                ranks = arrays["rank_contents"][qi].astype(str).tolist()
                row_sources: list[list[int]] = []
                row_queries: list[list[int]] = []
                for rank, content in enumerate(ranks):
                    second_letter = mappings[qid]["original_to_new"][content]
                    sources = [left_pad + value for value in first_positions[content]]
                    queries = [left_pad + value for value in second_positions[second_letter]]
                    if not sources or not queries or max(sources) >= min(queries):
                        raise RuntimeError("Invalid matching source/query option lines")
                    row_sources.append(sources)
                    row_queries.append(queries)
                    arrays["source_position_counts"][qi, rank] = len(sources)
                    arrays["query_position_counts"][qi, rank] = len(queries)
                    arrays["control_position_counts"][qi, rank] = len(
                        first_positions[ranks[(rank + 1) % 4]]
                    )
                source_physical.append(row_sources)
                query_physical.append(row_queries)
                arrays["prompt_hashes"][ci, qi] = _hash_prompt(batch["prompts"][row])
                audits.append({"first": first_audit, "second": second_audit, "ranks": ranks})

            natural = _aggregate_logits(
                _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                variant_ids,
            )
            if not np.all(np.isfinite(natural)):
                raise RuntimeError("Non-finite natural logits")
            arrays["natural_logits"][ci, indices] = natural
            for row, qid in enumerate(cohort):
                arrays["trusted_natural_logits"][ci, qid_index[qid]] = np.asarray(
                    trusted[ci][qid]["aggregated_ad_logits"], dtype=np.float32
                )

            for bi, name in enumerate(BAND_NAMES):
                layers = band_indices[name]
                for rank in range(4):
                    arrays["matched_logits"][bi, ci, rank, indices] = _run_lesion(
                        model, parts, batch, variant_ids, layers,
                        source_physical, query_physical, (rank,), False,
                    )
                    arrays["control_logits"][bi, ci, rank, indices] = _run_lesion(
                        model, parts, batch, variant_ids, layers,
                        source_physical, query_physical, (rank,), True,
                    )
                arrays["joint_matched_logits"][bi, ci, indices] = _run_lesion(
                    model, parts, batch, variant_ids, layers,
                    source_physical, query_physical, tuple(range(4)), False,
                )
                arrays["joint_control_logits"][bi, ci, indices] = _run_lesion(
                    model, parts, batch, variant_ids, layers,
                    source_physical, query_physical, tuple(range(4)), True,
                )

            cohort_audit["conditions"][condition] = {
                "rendered_prompt": batch["prompts"][0],
                "prompt_hash": arrays["prompt_hashes"][ci, indices[0]].item(),
                "rank_contents": audits[0]["ranks"],
                "first_option_lines": audits[0]["first"],
                "second_option_lines": audits[0]["second"],
                "bands": {name: list(BAND_BLOCKS[name]) for name in BAND_NAMES},
                "cyclic_control": "For target rank Rr, block source R(r+1), wrapping R4 to R1.",
            }

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"all-candidate full range: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.2f}", flush=True,
        )
        if not audit_path.exists():
            audit_path.write_text(json.dumps(cohort_audit, indent=2) + "\n")

    metadata = {
        "experiment": "all-candidate matching-edge causal extension through layer 64",
        "config": config.as_dict(),
        "n_questions": len(qids),
        "conditions": list(CONDITIONS),
        "ranks": list(RANKS),
        "bands": {name: list(BAND_BLOCKS[name]) for name in BAND_NAMES},
        "complete_model_forwards_per_cohort": 42,
        "complete_model_work": (
            "Per condition: one natural forward; for late 52--64 and full 4--64, "
            "four matching lesions, four cyclic controls, one joint matching lesion, "
            "and one joint cyclic control."
        ),
        "elapsed_seconds_after_load": time.monotonic() - started,
        "cohort_seconds": durations,
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
        args.config, args.remapping_plan, args.baseline,
        args.trusted_game, args.trusted_neutral,
        args.output_dir, args.max_cohorts,
    )


if __name__ == "__main__":
    main()
