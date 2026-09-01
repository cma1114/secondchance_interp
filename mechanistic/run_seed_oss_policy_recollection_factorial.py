from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .collect_cross_model_behavioral_gate import _assert_prompt_pair, _scenario_messages
from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSDPAFeedbackHistoryFactorial
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .run_fixed_a_final_query_edge_ablation import _option_line_positions
from .run_seed_oss_feedback_suffix_crossover import (
    _feedback_suffix_positions,
    _pad_to_width,
)
from .run_seed_oss_matching_history_blockade import (
    ATTENTION_LAYERS_ONE_BASED,
    CANONICAL_BATCH_SIZE,
    EXPERIMENT_MODEL_NAME,
    MODEL_ID,
    MODEL_REVISION,
    TASKS,
    TRUSTED_SCENARIOS,
    _aggregate_final_logits,
    _forward_final_logits,
)


POLICIES = ("Game", "Neutral")
ACCESS_LEVELS = ("intact", "matching_block", "cyclic_wrong_block")
LETTERS = ("A", "B", "C", "D")
FORWARDS_PER_COHORT_METADATA_KEY = "complete_model_forwards_per_canonical_cohort"
TRACK_MIXED_BATCH_NATURAL_DRIFT = False


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _initialize(
    path: Path,
    qids: list[str],
    *,
    track_mixed_batch_natural_drift: bool | None = None,
) -> dict[str, np.ndarray]:
    if track_mixed_batch_natural_drift is None:
        track_mixed_batch_natural_drift = TRACK_MIXED_BATCH_NATURAL_DRIFT
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise RuntimeError("Existing checkpoint uses different questions")
        if arrays["access_levels"].astype(str).tolist() != list(ACCESS_LEVELS):
            raise RuntimeError("Existing checkpoint uses different access levels")
        if track_mixed_batch_natural_drift:
            if "mixed_batch_natural_drift" not in arrays:
                if np.any(arrays["completed"]):
                    raise RuntimeError(
                        "Existing mixed-batch checkpoint lacks its required natural-drift array"
                    )
                arrays["mixed_batch_natural_drift"] = np.full(
                    (2, len(qids)), np.nan, dtype=np.float32
                )
        else:
            # Seed's canonical batch-4 path never measures this Gemma-specific
            # diagnostic.  Ignore the transient all-NaN field written by the
            # regressed shared runner rather than fabricating values for it.
            arrays.pop("mixed_batch_natural_drift", None)
        return arrays
    n = len(qids)
    arrays = {
        "question_ids": np.asarray(qids),
        "recipient_tasks": np.asarray(TASKS),
        "installed_policies": np.asarray(POLICIES),
        "access_levels": np.asarray(ACCESS_LEVELS),
        "completed": np.zeros(n, dtype=bool),
        "rank_contents": np.full((n, 4), "", dtype="<U1"),
        "reference_access_logits": np.full((2, 3, n, 4), np.nan, dtype=np.float32),
        "raw_factorial_logits": np.full((2, 2, 3, n, 4), np.nan, dtype=np.float32),
        "factorial_logits": np.full((2, 2, 3, n, 4), np.nan, dtype=np.float32),
        "identity_error": np.full((2, 3, n), np.nan, dtype=np.float32),
        "trusted_natural_error": np.full((2, n), np.nan, dtype=np.float32),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "suffix_position_counts": np.zeros((2, n), dtype=np.int16),
        "history_source_counts": np.zeros((n, 4), dtype=np.int16),
        "history_query_counts": np.zeros((n, 4), dtype=np.int16),
    }
    if track_mixed_batch_natural_drift:
        arrays["mixed_batch_natural_drift"] = np.full(
            (2, n), np.nan, dtype=np.float32
        )
    return arrays


def _assert_binding_config(config: ExperimentConfig) -> None:
    if config.model_id != MODEL_ID or config.model_revision != MODEL_REVISION:
        raise ValueError("Requires the binding's pinned configured model revision")
    if config.batch_size != CANONICAL_BATCH_SIZE:
        raise ValueError(
            f"Requires canonical batch_size={CANONICAL_BATCH_SIZE}, "
            f"found {config.batch_size}"
        )


def _mixed_batch_natural_max_abs_drift(
    arrays: dict[str, np.ndarray],
) -> float:
    drift = arrays.get("mixed_batch_natural_drift")
    if drift is None or not np.all(np.isfinite(drift)):
        raise RuntimeError(
            "Mixed-batch natural drift must be present and finite for this binding"
        )
    return float(np.max(drift))


def _experiment_name(dataset: str) -> str:
    return (
        f"{EXPERIMENT_MODEL_NAME} {dataset} "
        "direct policy by recollection factorial"
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _history_query_sources(
    tokenizer: Any,
    prompt: str,
    question: dict[str, Any],
    remapped: dict[str, Any],
    mapping: dict[str, Any],
    ranks: list[str],
    left_pad: int,
    access: str,
) -> tuple[dict[int, list[int]], dict[str, Any]]:
    if access == "intact":
        return {}, {}
    first_positions, first_audit = _option_line_positions(tokenizer, prompt, question)
    second_positions, second_audit = _option_line_positions(tokenizer, prompt, remapped)
    query_sources: dict[int, list[int]] = {}
    rank_audit: list[dict[str, Any]] = []
    for rank_index, content in enumerate(ranks):
        second_letter = mapping["original_to_new"][content]
        source_content = (
            content if access == "matching_block" else ranks[(rank_index + 1) % 4]
        )
        sources = [left_pad + value for value in first_positions[source_content]]
        queries = [left_pad + value for value in second_positions[second_letter]]
        if not sources or not queries or max(sources) >= min(queries):
            raise RuntimeError("Invalid causal matching-history source/query spans")
        for query in queries:
            query_sources[query] = sources
        rank_audit.append(
            {
                "rank": rank_index + 1,
                "target_content": content,
                "target_second_letter": second_letter,
                "blocked_source_content": source_content,
                "source_count": len(sources),
                "query_count": len(queries),
            }
        )
    return query_sources, {
        "first_option_lines": first_audit,
        "second_option_lines": second_audit,
        "ranks": rank_audit,
    }


def _factorial_forward(
    model: Any,
    parts: Any,
    input_ids: Any,
    attention_mask: Any,
    variant_ids: dict[str, list[int]],
    layers: list[int],
    donors: dict[int, int],
    suffix_positions: list[list[int]],
    history_rows: dict[int, dict[int, list[int]]],
) -> tuple[np.ndarray, dict[str, int]]:
    history_by_layer = (
        {layer: history_rows for layer in layers} if history_rows else None
    )
    intervention = BatchedSDPAFeedbackHistoryFactorial(
        parts,
        {
            row: (donors[row], suffix_positions[row], layers)
            for row in sorted(donors)
        },
        history_by_layer,
    )
    try:
        logits = _aggregate_final_logits(
            _forward_final_logits(model, parts, input_ids, attention_mask),
            variant_ids,
        )
        intervention.assert_fired()
        stats = {
            "sdpa_calls": int(intervention.sdpa_calls),
            "patched_position_count": int(intervention.patched_position_count),
            "edited_edge_count": int(intervention.edited_edge_count),
            "unique_layers_seen": len(intervention.layers_seen),
        }
    finally:
        intervention.close()
    if not np.all(np.isfinite(logits)):
        raise RuntimeError("Non-finite factorial logits")
    return logits, stats


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(args.config)
    _assert_binding_config(config)
    if (
        config.model_loader not in {"causal_lm", "multimodal"}
        or config.chat_serialization != "hf_template"
        or config.attn_implementation != "sdpa"
    ):
        raise ValueError("Requires a supported native HF text path with SDPA")
    if config.prompt_mode != "baseline_matched_empty_history" or config.feedback_variant != "token_matched_test":
        raise ValueError("Requires the clean token-matched prompt")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {str(row["id"]): row for row in manifest["questions"]}
    qids = [str(row["id"]) for row in manifest["questions"]]
    if config.question_ids is not None:
        wanted = set(config.question_ids)
        qids = [qid for qid in qids if qid in wanted]
    if config.max_questions is not None:
        qids = qids[: config.max_questions]
    if args.max_cohorts is not None:
        qids = qids[: int(args.max_cohorts) * config.batch_size]
    if len(qids) % config.batch_size:
        raise RuntimeError("Questions must form complete configured cohorts")
    dataset = "TriviaMC" if qids and qids[0].startswith("triviamc_") else "SimpleMC"
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    trusted_payload = json.loads(args.trusted_behavior.read_text())
    if (
        trusted_payload.get("model_id") != MODEL_ID
        or trusted_payload.get("model_revision") != MODEL_REVISION
        or not trusted_payload.get("complete")
    ):
        raise RuntimeError("Trusted behavior is incomplete or belongs to another model")
    trusted = [trusted_payload["scenarios"][name] for name in TRUSTED_SCENARIOS]
    prior = _load_npz(args.prior_matching_results)
    prior_qids = prior["question_ids"].astype(str).tolist()
    if prior_qids[: len(qids)] != qids:
        raise RuntimeError("Prior matching-history questions differ")
    reference = np.stack(
        (
            prior["natural_logits"],
            prior["joint_matching_logits"],
            prior["joint_cyclic_wrong_logits"],
        ),
        axis=1,
    )[:, :, : len(qids)].astype(np.float32)
    ranks_all = prior["rank_contents"][: len(qids)].astype(str)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.npz"
    arrays = _initialize(
        result_path,
        qids,
        track_mixed_batch_natural_drift=TRACK_MIXED_BATCH_NATURAL_DRIFT,
    )
    arrays["reference_access_logits"] = reference
    arrays["rank_contents"] = ranks_all
    qid_index = {qid: index for index, qid in enumerate(qids)}

    if arrays["completed"].all():
        required = (
            "factorial_logits",
            "identity_error",
            "trusted_natural_error",
        )
        if any(not np.all(np.isfinite(arrays[name])) for name in required):
            raise RuntimeError("Completed checkpoint contains non-finite required outputs")
        print(
            f"{EXPERIMENT_MODEL_NAME} {dataset} policy x recollection: "
            f"checkpoint already complete ({len(qids)}/{len(qids)})",
            flush=True,
        )
        return

    load_started = time.monotonic()
    model, processor, parts = load_model_and_processor(config)
    model_load_seconds = time.monotonic() - load_started
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _text, token_id in resolved[letter]})
        for letter in LETTERS
    }
    layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    ]
    if [index + 1 for index in layers] != list(ATTENTION_LAYERS_ONE_BASED):
        raise RuntimeError("Unexpected attention-layer inventory")
    if any(getattr(layer, "linear_attn", None) is not None for layer in parts.layers):
        raise RuntimeError("Seed unexpectedly exposes recurrent attention")
    print(f"MODEL_LOADED seconds={model_load_seconds:.3f}", flush=True)

    durations: list[float] = []
    stats: list[dict[str, Any]] = []
    first_audit: dict[str, Any] | None = None
    started = time.monotonic()
    for cohort_start in range(0, len(qids), config.batch_size):
        cohort = qids[cohort_start : cohort_start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if arrays["completed"][indices].all():
            continue
        cohort_started = time.monotonic()
        task_prompts: list[list[str]] = [[], []]
        remapped_questions: list[dict[str, dict[str, Any]]] = [{}, {}]
        for task_index, scenario in enumerate(TRUSTED_SCENARIOS):
            for qid in cohort:
                messages, remapped = _scenario_messages(
                    scenario, questions[qid], mappings[qid]["new_to_original"]
                )
                if remapped is None:
                    raise RuntimeError("Policy/recollection factorial requires remapped options")
                prompt = render_chat(
                    processor,
                    messages,
                    config.disable_thinking,
                    config.chat_serialization,
                    config.chat_template_kwargs,
                )
                task_prompts[task_index].append(prompt)
                remapped_questions[task_index][qid] = remapped
                digest = _hash_prompt(prompt)
                if digest != trusted[task_index][qid]["prompt_hash"]:
                    raise RuntimeError("Prompt hash differs from trusted behavior")
                arrays["prompt_hashes"][task_index, qid_index[qid]] = digest
        for local in range(len(cohort)):
            _assert_prompt_pair(task_prompts[0][local], task_prompts[1][local])
        canonical_width = max(
            len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            for prompts in task_prompts for prompt in prompts
        )

        for pair_start in range(0, len(cohort), 2):
            if len(cohort) == 1:
                pair = [cohort[0], cohort[0]]
                pair_local = [0, 0]
            else:
                pair = cohort[pair_start : pair_start + 2]
                if len(pair) != 2:
                    raise RuntimeError("Multi-question cohorts must contain complete pairs")
                pair_local = [pair_start, pair_start + 1]
            pair_indices = [qid_index[qid] for qid in pair]
            mixed_prompts = (
                [task_prompts[0][index] for index in pair_local]
                + [task_prompts[1][index] for index in pair_local]
            )
            mixed_tasks = [0, 0, 1, 1]
            mixed_qids = [pair[0], pair[1], pair[0], pair[1]]
            if len(cohort) == 1:
                # Gemma's low-precision path can change slightly when batch
                # composition changes.  Keep same-policy and opposite-policy
                # suffix donors inside the identical mixed Game/Neutral batch.
                batch_specs = [
                    (
                        "identity_both",
                        mixed_prompts,
                        mixed_tasks,
                        mixed_qids,
                        {0: 1, 1: 0, 2: 3, 3: 2},
                    ),
                    (
                        "reciprocal",
                        mixed_prompts,
                        mixed_tasks,
                        mixed_qids,
                        {0: 2, 1: 3, 2: 0, 3: 1},
                    ),
                ]
            else:
                batch_specs = [
                    (
                        "identity_game",
                        [task_prompts[0][pair_local[0]], task_prompts[0][pair_local[0]], task_prompts[0][pair_local[1]], task_prompts[0][pair_local[1]]],
                        [0, 0, 0, 0],
                        [pair[0], pair[0], pair[1], pair[1]],
                        {0: 1, 1: 0, 2: 3, 3: 2},
                    ),
                    (
                        "identity_neutral",
                        [task_prompts[1][pair_local[0]], task_prompts[1][pair_local[0]], task_prompts[1][pair_local[1]], task_prompts[1][pair_local[1]]],
                        [1, 1, 1, 1],
                        [pair[0], pair[0], pair[1], pair[1]],
                        {0: 1, 1: 0, 2: 3, 3: 2},
                    ),
                    (
                        "reciprocal",
                        mixed_prompts,
                        mixed_tasks,
                        mixed_qids,
                        {0: 2, 1: 3, 2: 0, 3: 1},
                    ),
                ]
            for batch_name, prompts, row_tasks, row_qids, donors in batch_specs:
                input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
                input_ids, attention_mask = _pad_to_width(
                    tokenizer, input_ids, attention_mask, canonical_width
                )
                width = int(input_ids.shape[1])
                suffix_positions: list[list[int]] = []
                token_rows: list[list[int]] = []
                for row, (prompt, task_index, qid) in enumerate(zip(prompts, row_tasks, row_qids)):
                    token_ids = [
                        int(value)
                        for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    ]
                    token_rows.append(token_ids)
                    left_pad = width - len(token_ids)
                    if input_ids[row, left_pad:].tolist() != token_ids:
                        raise RuntimeError("Batched tokenization differs from prompt audit")
                    positions, _suffix_audit = _feedback_suffix_positions(
                        tokenizer, prompt, TASKS[task_index]
                    )
                    physical = [left_pad + value for value in positions]
                    suffix_positions.append(physical)
                    arrays["suffix_position_counts"][task_index, qid_index[qid]] = len(physical)
                for target, donor in donors.items():
                    if suffix_positions[target] != suffix_positions[donor]:
                        raise RuntimeError("Donor and recipient suffix positions are not aligned")

                for access_index, access in enumerate(ACCESS_LEVELS):
                    history_rows: dict[int, dict[int, list[int]]] = {}
                    access_audits: list[dict[str, Any]] = []
                    if access != "intact":
                        for row, (prompt, task_index, qid, token_ids) in enumerate(
                            zip(prompts, row_tasks, row_qids, token_rows)
                        ):
                            left_pad = width - len(token_ids)
                            queries, row_audit = _history_query_sources(
                                tokenizer,
                                prompt,
                                questions[qid],
                                remapped_questions[task_index][qid],
                                mappings[qid],
                                ranks_all[qid_index[qid]].tolist(),
                                left_pad,
                                access,
                            )
                            history_rows[row] = queries
                            access_audits.append(row_audit)
                            # Every question appears in several duplicated rows.  Fill
                            # the span audit for each question (later duplicates must
                            # agree and harmlessly overwrite the same counts).
                            for rank_index, rank_row in enumerate(row_audit["ranks"]):
                                arrays["history_source_counts"][qid_index[qid], rank_index] = rank_row["source_count"]
                                arrays["history_query_counts"][qid_index[qid], rank_index] = rank_row["query_count"]
                    logits, row_stats = _factorial_forward(
                        model,
                        parts,
                        input_ids,
                        attention_mask,
                        variant_ids,
                        layers,
                        donors,
                        suffix_positions,
                        history_rows,
                    )
                    stats.append({"batch": batch_name, "access": access, **row_stats})
                    if batch_name == "identity_both":
                        qi = qid_index[pair[0]]
                        for task_index, (donor_row, target_row) in enumerate(((0, 1), (2, 3))):
                            error = float(np.max(np.abs(logits[target_row] - logits[donor_row])))
                            if error != 0.0:
                                raise RuntimeError(
                                    f"identity_both/{access}/{TASKS[task_index]} identity error {error}"
                                )
                            arrays["identity_error"][task_index, access_index, qi] = error
                            arrays["raw_factorial_logits"][task_index, task_index, access_index, qi] = logits[target_row]
                            arrays["factorial_logits"][task_index, task_index, access_index, qi] = reference[task_index, access_index, qi]
                            if access == "intact":
                                trusted_value = np.asarray(
                                    trusted[task_index][pair[0]]["aggregated_ad_logits"],
                                    dtype=np.float32,
                                )
                                # The factorial cells deliberately share one mixed
                                # Game/Neutral batch so reciprocal and same-policy
                                # suffix patches have identical numerical context.
                                # Gemma's low-precision raw logits can drift with
                                # batch composition; record that drift separately.
                                # Exact trusted-natural validity belongs to the
                                # already validated one-row reference used to anchor
                                # every corrected factorial cell.
                                arrays["mixed_batch_natural_drift"][task_index, qi] = float(
                                    np.max(np.abs(logits[donor_row] - trusted_value))
                                )
                                arrays["trusted_natural_error"][task_index, qi] = float(
                                    np.max(
                                        np.abs(
                                            reference[task_index, 0, qi]
                                            - trusted_value
                                        )
                                    )
                                )
                    elif batch_name.startswith("identity"):
                        task_index = 0 if batch_name == "identity_game" else 1
                        donor_values = logits[[0, 2]]
                        target_values = logits[[1, 3]]
                        errors = np.max(np.abs(target_values - donor_values), axis=-1)
                        if float(np.max(errors)) != 0.0:
                            raise RuntimeError(
                                f"{batch_name}/{access} identity error {float(np.max(errors))}"
                            )
                        arrays["identity_error"][task_index, access_index, pair_indices] = errors
                        arrays["raw_factorial_logits"][task_index, task_index, access_index, pair_indices] = target_values
                        arrays["factorial_logits"][task_index, task_index, access_index, pair_indices] = reference[task_index, access_index, pair_indices]
                        if access == "intact":
                            trusted_values = np.asarray(
                                [trusted[task_index][qid]["aggregated_ad_logits"] for qid in pair],
                                dtype=np.float32,
                            )
                            natural_errors = np.max(np.abs(donor_values - trusted_values), axis=-1)
                            arrays["trusted_natural_error"][task_index, pair_indices] = natural_errors
                    else:
                        for recipient_task in range(2):
                            if len(cohort) == 1:
                                row = 0 if recipient_task == 0 else 2
                                qi = qid_index[pair[0]]
                                opposite_policy = 1 - recipient_task
                                raw = logits[row]
                                same_raw = arrays["raw_factorial_logits"][recipient_task, recipient_task, access_index, qi]
                                if not np.all(np.isfinite(same_raw)):
                                    raise RuntimeError("Identity access cell must run before reciprocal")
                                arrays["raw_factorial_logits"][recipient_task, opposite_policy, access_index, qi] = raw
                                arrays["factorial_logits"][recipient_task, opposite_policy, access_index, qi] = (
                                    reference[recipient_task, access_index, qi] + raw - same_raw
                                )
                                continue
                            rows = [0, 1] if recipient_task == 0 else [2, 3]
                            opposite_policy = 1 - recipient_task
                            raw = logits[rows]
                            same_raw = arrays["raw_factorial_logits"][
                                recipient_task, recipient_task, access_index, pair_indices
                            ]
                            if not np.all(np.isfinite(same_raw)):
                                raise RuntimeError("Identity access cell must run before reciprocal")
                            arrays["raw_factorial_logits"][
                                recipient_task, opposite_policy, access_index, pair_indices
                            ] = raw
                            arrays["factorial_logits"][
                                recipient_task, opposite_policy, access_index, pair_indices
                            ] = (
                                reference[recipient_task, access_index, pair_indices]
                                + raw
                                - same_raw
                            )
                    if first_audit is None and access != "intact":
                        first_audit = {
                            "dataset": dataset,
                            "batch": batch_name,
                            "access": access,
                            "suffix_positions": suffix_positions,
                            "history": access_audits,
                            "layers_one_based": list(ATTENTION_LAYERS_ONE_BASED),
                        }

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"{EXPERIMENT_MODEL_NAME} {dataset} policy x recollection: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.3f}; identity_error={float(np.nanmax(arrays['identity_error'])):.8f}",
            flush=True,
        )
        if first_audit is not None and not (args.output_dir / "prompt_audit.json").exists():
            _atomic_write_json(args.output_dir / "prompt_audit.json", first_audit)

    forwards_per_cohort = (
        6 if config.batch_size == 1 else 9 * ((config.batch_size + 1) // 2)
    )
    metadata = {
        "experiment": _experiment_name(dataset),
        "dataset": dataset,
        "questions": len(qids),
        "recipient_tasks": list(TASKS),
        "installed_policies": list(POLICIES),
        "access_levels": list(ACCESS_LEVELS),
        "attention_layers_one_based": list(ATTENTION_LAYERS_ONE_BASED),
        "complete_model_forwards_total": (
            6 * len(qids)
            if config.batch_size == 1
            else 9 * ((len(qids) + 1) // 2)
        ),
        "identity_max_abs_error": float(np.nanmax(arrays["identity_error"])),
        "trusted_natural_max_abs_error": float(np.nanmax(arrays["trusted_natural_error"])),
        "all_outputs_finite": bool(np.all(np.isfinite(arrays["factorial_logits"]))),
        "same_batch_correction": "opposite-policy cell = prior validated access reference + reciprocal raw - same-policy identity raw",
        "model_load_seconds": model_load_seconds,
        "elapsed_seconds_after_load": time.monotonic() - started,
        "cohort_seconds": durations,
        "intervention_stats": stats,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    metadata[FORWARDS_PER_COHORT_METADATA_KEY] = forwards_per_cohort
    if TRACK_MIXED_BATCH_NATURAL_DRIFT:
        metadata["mixed_batch_natural_max_abs_drift"] = (
            _mixed_batch_natural_max_abs_drift(arrays)
        )
    _atomic_write_json(args.output_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--trusted-behavior", type=Path, required=True)
    parser.add_argument("--prior-matching-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
