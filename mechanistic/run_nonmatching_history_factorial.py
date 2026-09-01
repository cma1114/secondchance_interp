from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSDPAQuerySourceAttentionAblator
from .io import atomic_save_npz
from .modeling import get_tokenizer, load_model_and_processor, resolve_answer_tokens
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_fixed_a_final_query_edge_ablation import _option_line_positions


ORDINARY_LAYERS = tuple(range(4, 65, 4))
RANKS = ("R1", "R2", "R3", "R4")
INTERVENTIONS = (
    "natural",
    "matching_only_blocked",
    "nonmatching_three_blocked",
    "all_four_blocked",
)
LesionMode = Literal[
    "matching_only_blocked", "nonmatching_three_blocked", "all_four_blocked"
]


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses a different question order")
        if arrays["interventions"].astype(str).tolist() != list(INTERVENTIONS):
            raise ValueError("Existing checkpoint uses different interventions")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "ordinary_layers_one_based": np.asarray(ORDINARY_LAYERS, dtype=np.int16),
        "ranks": np.asarray(RANKS),
        "interventions": np.asarray(INTERVENTIONS),
        "completed": np.zeros(n, dtype=bool),
        "rank_contents": np.full((n, 4), "", dtype="<U1"),
        "baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        # condition, intervention, question, current A-D output letter
        "logits": np.full((2, 4, n, 4), np.nan, dtype=np.float32),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "source_position_counts": np.zeros((n, 4), dtype=np.int16),
        "query_position_counts": np.zeros((n, 4), dtype=np.int16),
        "matching_blocked_counts": np.zeros((n, 4), dtype=np.int16),
        "nonmatching_blocked_counts": np.zeros((n, 4), dtype=np.int16),
        "all_four_blocked_counts": np.zeros((n, 4), dtype=np.int16),
    }


def _blocked_source_ranks(target_rank: int, mode: LesionMode) -> tuple[int, ...]:
    if mode == "matching_only_blocked":
        return (target_rank,)
    if mode == "nonmatching_three_blocked":
        return tuple(rank for rank in range(4) if rank != target_rank)
    if mode == "all_four_blocked":
        return tuple(range(4))
    raise ValueError(f"Unknown lesion mode: {mode}")


def _factorial_specs(
    layer_indices: tuple[int, ...],
    source_positions: list[list[list[int]]],
    query_positions: list[list[list[int]]],
    mode: LesionMode,
) -> dict[int, dict[int, dict[int, list[int]]]]:
    specs: dict[int, dict[int, dict[int, list[int]]]] = {}
    for layer_index in layer_indices:
        rows: dict[int, dict[int, list[int]]] = {}
        for row, (row_sources, row_queries) in enumerate(
            zip(source_positions, query_positions)
        ):
            if len(row_sources) != 4 or len(row_queries) != 4:
                raise RuntimeError("Expected four ranked source and query lines")
            queries: dict[int, list[int]] = {}
            for target_rank in range(4):
                blocked_ranks = _blocked_source_ranks(target_rank, mode)
                blocked = sorted(
                    {
                        int(position)
                        for source_rank in blocked_ranks
                        for position in row_sources[source_rank]
                    }
                )
                if not blocked:
                    raise RuntimeError("Every lesion must block at least one source token")
                for query in row_queries[target_rank]:
                    if int(query) in queries:
                        raise RuntimeError("Second-presentation option query lines overlap")
                    queries[int(query)] = blocked
            rows[row] = queries
        specs[int(layer_index)] = rows
    return specs


def _validate_partition(
    row_sources: list[list[int]], row_queries: list[list[int]]
) -> dict[str, list[int]]:
    if len(row_sources) != 4 or len(row_queries) != 4:
        raise RuntimeError("Expected four source and query lines")
    flattened_sources = [position for line in row_sources for position in line]
    flattened_queries = [position for line in row_queries for position in line]
    if len(set(flattened_sources)) != len(flattened_sources):
        raise RuntimeError("First-presentation option source lines overlap")
    if len(set(flattened_queries)) != len(flattened_queries):
        raise RuntimeError("Second-presentation option query lines overlap")
    for target_rank in range(4):
        matching = set(row_sources[target_rank])
        nonmatching = {
            position
            for rank in range(4)
            if rank != target_rank
            for position in row_sources[rank]
        }
        all_four = set(flattened_sources)
        if matching & nonmatching:
            raise RuntimeError("Matching and nonmatching source sets overlap")
        if matching | nonmatching != all_four:
            raise RuntimeError("Matching and nonmatching sets do not partition all sources")
        if max(all_four) >= min(row_queries[target_rank]):
            raise RuntimeError("A first-presentation source does not precede its 2P queries")
    return {
        "all_first_option_positions": sorted(flattened_sources),
        "all_second_option_positions": sorted(flattened_queries),
    }


def _run_lesion(
    model: Any,
    parts: Any,
    batch: dict[str, Any],
    variant_ids: dict[str, list[int]],
    specs: dict[int, dict[int, dict[int, list[int]]]],
) -> np.ndarray:
    with BatchedSDPAQuerySourceAttentionAblator(parts, specs):
        output = _aggregate_logits(
            _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
            variant_ids,
        )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Non-finite intervention logits")
    return output


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
    if set(qids) - set(mappings) or set(qids) - set(baseline):
        raise RuntimeError("Canonical mappings or Baseline results are incomplete")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    layer_indices = tuple(
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if tuple(index + 1 for index in layer_indices) != ORDINARY_LAYERS:
        raise RuntimeError("Unexpected ordinary-attention layer inventory")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = output_dir / "prompt_audit.json"
    started = time.monotonic()
    durations: list[float] = []

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

        for qid in cohort:
            qi = qid_index[qid]
            logits = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=float)
            order = np.argsort(-logits, kind="stable")
            arrays["baseline_logits"][qi] = logits
            arrays["rank_contents"][qi] = np.asarray(
                [LETTERS[int(index)] for index in order]
            )

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
                    sources = [
                        left_pad + int(position)
                        for position in first_positions[content]
                    ]
                    queries = [
                        left_pad + int(position)
                        for position in second_positions[second_letter]
                    ]
                    if not sources or not queries:
                        raise RuntimeError("Empty first- or second-presentation option line")
                    row_sources.append(sources)
                    row_queries.append(queries)
                    arrays["source_position_counts"][qi, rank] = len(sources)
                    arrays["query_position_counts"][qi, rank] = len(queries)
                partition = _validate_partition(row_sources, row_queries)
                all_count = len(partition["all_first_option_positions"])
                for rank in range(4):
                    arrays["matching_blocked_counts"][qi, rank] = len(
                        row_sources[rank]
                    )
                    arrays["nonmatching_blocked_counts"][qi, rank] = sum(
                        len(row_sources[source_rank])
                        for source_rank in range(4)
                        if source_rank != rank
                    )
                    arrays["all_four_blocked_counts"][qi, rank] = all_count
                source_physical.append(row_sources)
                query_physical.append(row_queries)
                arrays["prompt_hashes"][ci, qi] = _hash_prompt(batch["prompts"][row])
                audits.append(
                    {
                        "ranks": ranks,
                        "first_option_lines": first_audit,
                        "second_option_lines": second_audit,
                        "partition": partition,
                    }
                )

            natural = _aggregate_logits(
                _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                variant_ids,
            )
            if not np.all(np.isfinite(natural)):
                raise RuntimeError("Non-finite natural logits")
            arrays["logits"][ci, 0, indices] = natural
            for row, qid in enumerate(cohort):
                qi = qid_index[qid]
                arrays["trusted_natural_logits"][ci, qi] = np.asarray(
                    trusted[ci][qid]["aggregated_ad_logits"], dtype=np.float32
                )

            for intervention_index, mode in enumerate(INTERVENTIONS[1:], start=1):
                specs = _factorial_specs(
                    layer_indices,
                    source_physical,
                    query_physical,
                    mode,  # type: ignore[arg-type]
                )
                arrays["logits"][ci, intervention_index, indices] = _run_lesion(
                    model, parts, batch, variant_ids, specs
                )

            cohort_audit["conditions"][condition] = {
                "rendered_prompt": batch["prompts"][0],
                "prompt_hash": arrays["prompt_hashes"][ci, indices[0]].item(),
                "rank_contents": audits[0]["ranks"],
                "first_option_lines": audits[0]["first_option_lines"],
                "second_option_lines": audits[0]["second_option_lines"],
                "partition": audits[0]["partition"],
                "interventions": {
                    "matching_only_blocked": (
                        "For each complete 2P option line, block its complete "
                        "semantically matching 1P option line."
                    ),
                    "nonmatching_three_blocked": (
                        "For each complete 2P option line, preserve its matching 1P "
                        "line and block the other three complete 1P option lines."
                    ),
                    "all_four_blocked": (
                        "For each complete 2P option line, block all four complete "
                        "1P option lines."
                    ),
                },
            }

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"nonmatching-history factorial: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.2f}",
            flush=True,
        )
        if not audit_path.exists():
            audit_path.write_text(
                json.dumps(cohort_audit, indent=2, ensure_ascii=False) + "\n"
            )

    metadata = {
        "experiment": "canonical remapped matching/nonmatching 1P history factorial",
        "config": config.as_dict(),
        "n_questions": len(qids),
        "conditions": list(CONDITIONS),
        "ranks": list(RANKS),
        "interventions": list(INTERVENTIONS),
        "ordinary_attention_layers_one_based": list(ORDINARY_LAYERS),
        "complete_model_forwards_per_cohort": 8,
        "complete_model_work": (
            "Per condition: one same-batch natural forward, one matching-only "
            "blockade, one all-three-nonmatching blockade preserving the semantic "
            "match, and one all-four-1P-lines blockade."
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
