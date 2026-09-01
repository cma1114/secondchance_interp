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
from .analyze_cue_attention_distribution import SOURCE_NAMES, _cue_source_partition
from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSDPAQuerySourceAttentionAblator
from .io import atomic_save_npz
from .modeling import get_tokenizer, load_model_and_processor, resolve_answer_tokens
from .relay_interception import (
    BatchedGLACachedRelayDownstreamRestorer,
    BatchedGLARelayWriteCache,
    BatchedSDPACachedRelayDownstreamRestorer,
    BatchedSDPARelayWriteCache,
)
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_candidate_history_entry_factorial import (
    ORDINARY_LAYERS,
    RANKS,
    TOKEN_CLASSES,
    _factorial_specs,
    _hash_prompt,
    _partition_option_line,
    _wrong_source_rank,
)
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_fixed_a_final_query_edge_ablation import _option_line_positions

RELAY_GROUPS = (
    "second_option_semantics",
    "second_option_newlines",
    "second_option_structure",
    "post_list_cue_and_query",
    "final_assistant_prefix",
)
SOURCE_CONDITIONS = (
    "none",
    "complete_matching_block",
    "complete_balanced_wrong_block",
)
MECHANISMS = ("none", "ordinary", "gla", "both")
IDENTITY_MECHANISMS = ("ordinary", "gla", "both")
SEMANTIC_CLASS_INDEX = TOKEN_CLASSES.index("semantic")
JOINT_RELAY_MASK = (1 << len(RELAY_GROUPS)) - 1
EXPERIMENT_NAME = "targeted candidate-history downstream relay mediation"
COMPLETE_MODEL_WORK = (
    "Per task condition: natural, complete matching and balanced-wrong "
    "source baselines, the wrong-source joint specificity control, and 26 "
    "prespecified matching-source relay restorations (five singles, five "
    "complements, joint, three named pairs in both mode; five singles and "
    "joint in each single-carrier mode). Three real no-source joint "
    "restorations run only on the frozen identity sentinel cohorts."
)
RESTORATION_SEMANTICS = (
    "Relay-token local outputs remain source-perturbed; only their outgoing "
    "ordinary-attention K/V and/or recurrent GLA k/v/g/beta are restored."
)
UNINTERCEPTED_CHANNEL = (
    "The short causal GLA q/k/v depthwise-convolution state is not restored."
)


def _relay_bit(name: str) -> int:
    return 1 << RELAY_GROUPS.index(name)


def _scenario_inventory() -> tuple[tuple[str, int, str], ...]:
    """Frozen targeted Stage-B inventory (30 distinct cells per task).

    The matching-edge lesion gets the prespecified singles, complements,
    joint, named-pair, and carrier-mechanism restorations. The balanced wrong
    source is an internal lesion-size anchor; its joint-restoration crossing
    tests whether apparent joint rescue is specific to the semantic match.
    """

    scenarios: list[tuple[str, int, str]] = [
        ("none", 0, "none"),
        ("complete_matching_block", 0, "none"),
        ("complete_balanced_wrong_block", 0, "none"),
        ("complete_balanced_wrong_block", JOINT_RELAY_MASK, "both"),
    ]
    singles = [_relay_bit(name) for name in RELAY_GROUPS]
    complements = [JOINT_RELAY_MASK ^ mask for mask in singles]
    named_pairs = [
        _relay_bit("second_option_newlines")
        | _relay_bit("post_list_cue_and_query"),
        _relay_bit("second_option_newlines")
        | _relay_bit("final_assistant_prefix"),
        _relay_bit("post_list_cue_and_query")
        | _relay_bit("final_assistant_prefix"),
    ]
    scenarios.extend(
        ("complete_matching_block", mask, "both")
        for mask in singles + complements + [JOINT_RELAY_MASK] + named_pairs
    )
    for mechanism in ("ordinary", "gla"):
        scenarios.extend(
            ("complete_matching_block", mask, mechanism)
            for mask in singles + [JOINT_RELAY_MASK]
        )
    if len(scenarios) != 30 or len(set(scenarios)) != 30:
        raise RuntimeError("Frozen targeted Stage-B inventory must have 30 cells")
    return tuple(scenarios)


SCENARIOS = _scenario_inventory()
SCENARIO_IDS = tuple(
    f"{source}__relay_{relay_mask:02d}__{mechanism}"
    for source, relay_mask, mechanism in SCENARIOS
)


def _relay_mask_label(mask: int) -> str:
    if mask < 0 or mask >= 1 << len(RELAY_GROUPS):
        raise ValueError(f"Invalid relay mask {mask}")
    selected = [name for index, name in enumerate(RELAY_GROUPS) if mask & (1 << index)]
    return "+".join(selected) if selected else "none"


def _relay_groups(
    query_classes_by_rank: list[list[list[int]]],
    partition: list[list[int]],
    left_pad: int,
    final_query: int,
) -> dict[str, list[int]]:
    """Build a five-way exact cover from the 2P list through final readout.

    Tokens before the first 2P option line cannot mediate a perturbation first
    written at those lines and are therefore intrinsically outside Stage B.
    The final answer-query position is the outcome and is not a relay.
    """

    semantic = sorted(
        position
        for rank_classes in query_classes_by_rank
        for position in rank_classes[SEMANTIC_CLASS_INDEX]
    )
    structure = sorted(
        position
        for rank_classes in query_classes_by_rank
        for class_index in (0, 1, 2)
        for position in rank_classes[class_index]
    )
    newlines = sorted(
        position
        for rank_classes in query_classes_by_rank
        for position in rank_classes[TOKEN_CLASSES.index("newline")]
    )
    prefix = sorted(
        left_pad + position
        for position in partition[SOURCE_NAMES.index("final_assistant_prefix")]
        if left_pad + position < final_query
    )
    option_positions = semantic + structure + newlines
    if not option_positions or not prefix:
        raise RuntimeError(
            "Canonical prompt has empty option or assistant relay groups"
        )
    causal_tail = set(range(min(option_positions), final_query))
    used = set(semantic) | set(structure) | set(newlines) | set(prefix)
    post_list = sorted(causal_tail - used)
    groups = {
        "second_option_semantics": semantic,
        "second_option_newlines": newlines,
        "second_option_structure": structure,
        "post_list_cue_and_query": post_list,
        "final_assistant_prefix": prefix,
    }
    if any(not values for values in groups.values()):
        raise RuntimeError("Every frozen Stage-B relay group must be nonempty")
    flat = [position for name in RELAY_GROUPS for position in groups[name]]
    if len(flat) != len(set(flat)):
        raise RuntimeError("Stage-B relay groups overlap")
    if set(flat) != causal_tail:
        raise RuntimeError("Stage-B relay groups do not exhaust the causal tail")
    return groups


def _selected_relay_positions(
    groups_by_row: list[dict[str, list[int]]], relay_mask: int
) -> dict[int, list[int]]:
    if relay_mask <= 0 or relay_mask >= 1 << len(RELAY_GROUPS):
        raise ValueError("A restoration scenario needs a nonempty valid relay mask")
    selected: dict[int, list[int]] = {}
    for row, groups in enumerate(groups_by_row):
        positions = sorted(
            position
            for index, name in enumerate(RELAY_GROUPS)
            if relay_mask & (1 << index)
            for position in groups[name]
        )
        if not positions:
            raise RuntimeError("Selected relay mask produced an empty row")
        selected[row] = positions
    return selected


def _initialize(
    path: Path, qids: list[str], identity_qids: list[str]
) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise RuntimeError("Existing Stage-B checkpoint uses different questions")
        if arrays["scenario_ids"].astype(str).tolist() != list(SCENARIO_IDS):
            raise RuntimeError(
                "Existing Stage-B checkpoint uses a different scenario inventory"
            )
        if arrays["identity_question_ids"].astype(str).tolist() != identity_qids:
            raise RuntimeError("Existing Stage-B checkpoint uses different sentinels")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "conditions": np.asarray(CONDITIONS),
        "ranks": np.asarray(RANKS),
        "relay_groups": np.asarray(RELAY_GROUPS),
        "source_conditions": np.asarray(SOURCE_CONDITIONS),
        "mechanisms": np.asarray(MECHANISMS),
        "scenario_ids": np.asarray(SCENARIO_IDS),
        "identity_question_ids": np.asarray(identity_qids),
        "identity_mechanisms": np.asarray(IDENTITY_MECHANISMS),
        "scenario_source_index": np.asarray(
            [
                SOURCE_CONDITIONS.index(source)
                for source, _mask, _mechanism in SCENARIOS
            ],
            dtype=np.int8,
        ),
        "scenario_relay_mask": np.asarray(
            [mask for _source, mask, _mechanism in SCENARIOS], dtype=np.int8
        ),
        "scenario_mechanism_index": np.asarray(
            [MECHANISMS.index(mechanism) for _source, _mask, mechanism in SCENARIOS],
            dtype=np.int8,
        ),
        "completed": np.zeros(n, dtype=bool),
        "rank_contents": np.full((n, 4), "", dtype="<U1"),
        "baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "scenario_logits_raw": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
        "scenario_logits": np.full((2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32),
        "identity_completed": np.zeros(n, dtype=bool),
        "identity_logits_raw": np.full(
            (2, len(IDENTITY_MECHANISMS), n, 4), np.nan, dtype=np.float32
        ),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "wrong_source_ranks": np.full((n, 4), -1, dtype=np.int8),
        "relay_group_counts": np.full((2, n, len(RELAY_GROUPS)), -1, dtype=np.int16),
    }


def _validate_completed(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    complete = arrays["completed"].astype(bool)
    if not np.any(complete):
        return {"n_completed": 0.0}
    qidx = np.flatnonzero(complete)
    natural = np.take(arrays["same_batch_natural_logits"], qidx, axis=1)
    trusted = np.take(arrays["trusted_natural_logits"], qidx, axis=1)
    scenarios = np.take(arrays["scenario_logits_raw"], qidx, axis=2)
    if not all(np.all(np.isfinite(value)) for value in (natural, trusted, scenarios)):
        raise RuntimeError("Completed Stage-B outputs contain non-finite values")
    natural_error = float(np.max(np.abs(natural - trusted)))
    if natural_error != 0.0:
        raise RuntimeError(f"Trusted natural reproduction error is {natural_error}")
    identity_complete = arrays["identity_completed"].astype(bool)
    identity_qidx = np.flatnonzero(identity_complete)
    identity_error = 0.0
    if len(identity_qidx):
        identity = np.take(arrays["identity_logits_raw"], identity_qidx, axis=2)
        identity_natural = np.take(
            arrays["same_batch_natural_logits"], identity_qidx, axis=1
        )
        if not np.all(np.isfinite(identity)):
            raise RuntimeError("Completed identity sentinels contain non-finite values")
        identity_error = float(
            np.max(np.abs(identity - identity_natural[:, None]))
        )
    if identity_error != 0.0:
        raise RuntimeError(f"Real restoration-only identity error is {identity_error}")
    natural_index = SCENARIOS.index(("none", 0, "none"))
    matching_index = SCENARIOS.index(("complete_matching_block", 0, "none"))
    wrong_index = SCENARIOS.index(("complete_balanced_wrong_block", 0, "none"))
    matching_liveness = float(
        np.max(np.abs(scenarios[:, matching_index] - scenarios[:, natural_index]))
    )
    wrong_liveness = float(
        np.max(np.abs(scenarios[:, wrong_index] - scenarios[:, natural_index]))
    )
    return {
        "n_completed": float(complete.sum()),
        "n_identity_completed": float(identity_complete.sum()),
        "natural_max_abs_error": natural_error,
        "restoration_only_max_abs_error": identity_error,
        "complete_matching_source_max_abs_change": matching_liveness,
        "balanced_wrong_source_max_abs_change": wrong_liveness,
    }


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(args.config)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires exact canonical batch-size-4 SDPA execution")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires canonical empty-history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires action-matched incorrect/lost feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw_qwen_chatml serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    all_qids = [str(row["id"]) for row in manifest["questions"]]
    questions = {str(row["id"]): row for row in manifest["questions"]}
    qids = all_qids
    if args.max_cohorts is not None:
        qids = qids[: int(args.max_cohorts) * config.batch_size]
    global_index = {qid: index for index, qid in enumerate(all_qids)}
    local_index = {qid: index for index, qid in enumerate(qids)}
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    baseline = json.loads(args.baseline.read_text())["results"]
    trusted = [
        json.loads(args.trusted_game.read_text())["results"],
        json.loads(args.trusted_neutral.read_text())["results"],
    ]
    identity_plan = json.loads(args.identity_sentinel_plan.read_text())
    planned_identity_qids = [str(value) for value in identity_plan["question_ids"]]
    if len(planned_identity_qids) != len(set(planned_identity_qids)):
        raise RuntimeError("Identity sentinel plan contains duplicate questions")
    if not set(planned_identity_qids).issubset(all_qids):
        raise RuntimeError("Identity sentinel plan contains unknown questions")
    identity_set = set(planned_identity_qids)
    for start in range(0, len(all_qids), config.batch_size):
        membership = [qid in identity_set for qid in all_qids[start : start + 4]]
        if any(membership) and not all(membership):
            raise RuntimeError("Identity sentinels must select complete canonical cohorts")
    identity_qids = [qid for qid in qids if qid in identity_set]

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    ordinary_layers = tuple(
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    gla_layers = tuple(
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    )
    if tuple(index + 1 for index in ordinary_layers) != ORDINARY_LAYERS:
        raise RuntimeError("Unexpected ordinary-attention layer inventory")
    if len(gla_layers) != 48:
        raise RuntimeError("Unexpected GLA layer inventory")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.npz"
    audit_path = args.output_dir / "prompt_audit.json"
    arrays = _initialize(result_path, qids, identity_qids)
    durations: list[float] = []
    started = time.monotonic()

    for start in range(0, len(qids), config.batch_size):
        cohort = qids[start : start + config.batch_size]
        indices = [local_index[qid] for qid in cohort]
        identity_membership = [qid in identity_set for qid in cohort]
        if any(identity_membership) and not all(identity_membership):
            raise RuntimeError("A run cohort partially overlaps the identity sentinels")
        is_identity_cohort = all(identity_membership)
        if arrays["completed"][indices].all():
            continue
        cohort_started = time.monotonic()
        cohort_audit: dict[str, Any] = {"question_ids": cohort, "conditions": {}}
        for qid in cohort:
            qi = local_index[qid]
            old_logits = np.asarray(
                baseline[qid]["aggregated_ad_logits"], dtype=np.float32
            )
            arrays["baseline_logits"][qi] = old_logits
            arrays["rank_contents"][qi] = np.asarray(
                [
                    LETTERS[int(value)]
                    for value in np.argsort(-old_logits, kind="stable")
                ]
            )
            for rank in range(4):
                arrays["wrong_source_ranks"][qi, rank] = _wrong_source_rank(
                    global_index[qid], rank
                )

        for condition_index, condition in enumerate(CONDITIONS):
            batch = _build_batch(
                config, processor, tokenizer, questions, mappings, cohort, condition
            )
            width = int(batch["input_ids"].shape[1])
            final_query = width - 1
            source_positions: list[list[list[int]]] = []
            query_classes: list[list[list[list[int]]]] = []
            groups_by_row: list[dict[str, list[int]]] = []
            wrong_ranks = np.empty((len(cohort), 4), dtype=np.int8)
            row_audits: list[dict[str, Any]] = []

            for row, qid in enumerate(cohort):
                qi = local_index[qid]
                left_pad = width - len(batch["token_rows"][row])
                second_question = {
                    **questions[qid],
                    "options": {
                        new: questions[qid]["options"][old]
                        for new, old in mappings[qid]["new_to_original"].items()
                    },
                }
                first_positions, _first_audit = _option_line_positions(
                    tokenizer, batch["prompts"][row], questions[qid]
                )
                second_positions, second_audit = _option_line_positions(
                    tokenizer, batch["prompts"][row], second_question
                )
                ranks = arrays["rank_contents"][qi].astype(str).tolist()
                row_sources: list[list[int]] = []
                row_classes: list[list[list[int]]] = []
                for target_rank, content in enumerate(ranks):
                    second_letter = mappings[qid]["original_to_new"][content]
                    row_sources.append(
                        [left_pad + value for value in first_positions[content]]
                    )
                    raw_positions = second_positions[second_letter]
                    classes, _class_audit = _partition_option_line(
                        raw_positions, second_audit[second_letter]["tokens"]
                    )
                    physical = [
                        [left_pad + value for value in values] for values in classes
                    ]
                    if max(row_sources[-1]) >= min(
                        value for values in physical for value in values
                    ):
                        raise RuntimeError("1P source does not precede its 2P receiver")
                    row_classes.append(physical)
                    wrong_ranks[row, target_rank] = arrays["wrong_source_ranks"][
                        qi, target_rank
                    ]

                partition, position_audit = _cue_source_partition(
                    tokenizer,
                    batch["prompts"][row],
                    batch["messages"][row],
                    questions[qid],
                    second_question,
                    condition,
                    ranks,
                    mappings[qid]["original_to_new"],
                )
                if width - int(position_audit["prompt_length"]) != left_pad:
                    raise RuntimeError("Independent prompt-position audits disagree")
                groups = _relay_groups(row_classes, partition, left_pad, final_query)
                source_positions.append(row_sources)
                query_classes.append(row_classes)
                groups_by_row.append(groups)
                prompt_hash = _hash_prompt(batch["prompts"][row])
                if prompt_hash != trusted[condition_index][qid]["prompt_hash"]:
                    raise RuntimeError("Prompt hash differs from trusted natural run")
                arrays["prompt_hashes"][condition_index, qi] = prompt_hash
                arrays["trusted_natural_logits"][condition_index, qi] = np.asarray(
                    trusted[condition_index][qid]["aggregated_ad_logits"],
                    dtype=np.float32,
                )
                arrays["relay_group_counts"][condition_index, qi] = np.asarray(
                    [len(groups[name]) for name in RELAY_GROUPS], dtype=np.int16
                )
                row_audits.append(
                    {
                        "question_id": qid,
                        "left_pad": left_pad,
                        "final_query": final_query,
                        "relay_groups": groups,
                        "relay_tokens": {
                            name: [
                                tokenizer.decode(
                                    [int(batch["input_ids"][row, position])]
                                ).replace("\n", "\\n")
                                for position in groups[name]
                            ]
                            for name in RELAY_GROUPS
                        },
                    }
                )

            all_relays = {
                row: sorted(
                    position
                    for name in RELAY_GROUPS
                    for position in groups_by_row[row][name]
                )
                for row in range(len(cohort))
            }
            ordinary_cache = BatchedSDPARelayWriteCache(
                parts, all_relays, list(ordinary_layers)
            )
            gla_cache = BatchedGLARelayWriteCache(parts, all_relays, list(gla_layers))
            try:
                natural = _aggregate_logits(
                    _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                    variant_ids,
                )
            finally:
                gla_cache.close()
                ordinary_cache.close()
            if not np.all(np.isfinite(natural)):
                raise RuntimeError("Non-finite natural logits")
            arrays["same_batch_natural_logits"][condition_index, indices] = natural
            first_scenario = SCENARIOS.index(("none", 0, "none"))
            arrays["scenario_logits_raw"][condition_index, first_scenario, indices] = (
                natural
            )
            for qid in cohort:
                qi = local_index[qid]
                trusted_logits = arrays["trusted_natural_logits"][condition_index, qi]
                arrays["scenario_logits"][condition_index, first_scenario, qi] = (
                    trusted_logits
                )

            source_specs = {
                "complete_matching_block": _factorial_specs(
                    ordinary_layers,
                    source_positions,
                    query_classes,
                    wrong_ranks,
                    0,
                    "matching",
                ),
                "complete_balanced_wrong_block": _factorial_specs(
                    ordinary_layers,
                    source_positions,
                    query_classes,
                    wrong_ranks,
                    0,
                    "balanced_wrong",
                ),
            }

            for scenario_index, (
                source_name,
                relay_mask,
                mechanism,
            ) in enumerate(SCENARIOS):
                if scenario_index == first_scenario:
                    continue
                source_intervention = None
                ordinary_restorer = None
                gla_restorer = None
                try:
                    if source_name != "none":
                        source_intervention = BatchedSDPAQuerySourceAttentionAblator(
                            parts, source_specs[source_name]
                        )
                    if relay_mask:
                        selected = _selected_relay_positions(groups_by_row, relay_mask)
                        if mechanism in {"ordinary", "both"}:
                            ordinary_restorer = (
                                BatchedSDPACachedRelayDownstreamRestorer(
                                    parts,
                                    selected,
                                    list(ordinary_layers),
                                    ordinary_cache.cache,
                                )
                            )
                        if mechanism in {"gla", "both"}:
                            gla_restorer = BatchedGLACachedRelayDownstreamRestorer(
                                parts, selected, list(gla_layers), gla_cache.cache
                            )
                    output = _aggregate_logits(
                        _forward(
                            model,
                            parts,
                            batch["input_ids"],
                            batch["attention_mask"],
                        ),
                        variant_ids,
                    )
                    if ordinary_restorer is not None:
                        ordinary_restorer.assert_fired()
                    if gla_restorer is not None:
                        gla_restorer.assert_fired()
                finally:
                    if gla_restorer is not None:
                        gla_restorer.close()
                    if ordinary_restorer is not None:
                        ordinary_restorer.close()
                    if source_intervention is not None:
                        source_intervention.close()
                if not np.all(np.isfinite(output)):
                    raise RuntimeError(
                        f"Non-finite output in {SCENARIO_IDS[scenario_index]}"
                    )
                arrays["scenario_logits_raw"][
                    condition_index, scenario_index, indices
                ] = output
                for row, qid in enumerate(cohort):
                    qi = local_index[qid]
                    arrays["scenario_logits"][condition_index, scenario_index, qi] = (
                        arrays["trusted_natural_logits"][condition_index, qi]
                        + output[row]
                        - natural[row]
                    )

            if is_identity_cohort:
                selected = _selected_relay_positions(
                    groups_by_row, JOINT_RELAY_MASK
                )
                for identity_index, mechanism in enumerate(IDENTITY_MECHANISMS):
                    ordinary_restorer = None
                    gla_restorer = None
                    try:
                        if mechanism in {"ordinary", "both"}:
                            ordinary_restorer = (
                                BatchedSDPACachedRelayDownstreamRestorer(
                                    parts,
                                    selected,
                                    list(ordinary_layers),
                                    ordinary_cache.cache,
                                )
                            )
                        if mechanism in {"gla", "both"}:
                            gla_restorer = BatchedGLACachedRelayDownstreamRestorer(
                                parts, selected, list(gla_layers), gla_cache.cache
                            )
                        identity_output = _aggregate_logits(
                            _forward(
                                model,
                                parts,
                                batch["input_ids"],
                                batch["attention_mask"],
                            ),
                            variant_ids,
                        )
                        if ordinary_restorer is not None:
                            ordinary_restorer.assert_fired()
                        if gla_restorer is not None:
                            gla_restorer.assert_fired()
                    finally:
                        if gla_restorer is not None:
                            gla_restorer.close()
                        if ordinary_restorer is not None:
                            ordinary_restorer.close()
                    identity_error = float(
                        np.max(np.abs(identity_output - natural))
                    )
                    if identity_error != 0.0:
                        raise RuntimeError(
                            f"Identity sentinel failed for {mechanism}: {identity_error}"
                        )
                    arrays["identity_logits_raw"][
                        condition_index, identity_index, indices
                    ] = identity_output

            cohort_audit["conditions"][condition] = {
                "rendered_prompt": batch["prompts"][0],
                "rows": row_audits,
            }

        arrays["completed"][indices] = True
        if is_identity_cohort:
            arrays["identity_completed"][indices] = True
        validation = _validate_completed(arrays)
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            "candidate-history relay mediation: "
            f"{int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.2f}; identity_error="
            f"{validation['restoration_only_max_abs_error']}",
            flush=True,
        )
        if not audit_path.exists():
            cohort_audit["relay_groups"] = list(RELAY_GROUPS)
            audit_masks = sorted(
                set(range(1 << len(RELAY_GROUPS)))
                | {int(mask) for _source, mask, _mechanism in SCENARIOS}
                | {int(JOINT_RELAY_MASK)}
            )
            cohort_audit["relay_masks"] = {
                str(mask): _relay_mask_label(mask) for mask in audit_masks
            }
            cohort_audit["scenario_ids"] = list(SCENARIO_IDS)
            audit_path.write_text(json.dumps(cohort_audit, indent=2) + "\n")

    validation = _validate_completed(arrays)
    metadata = {
        "experiment": EXPERIMENT_NAME,
        "questions": len(qids),
        "conditions": list(CONDITIONS),
        "relay_groups": list(RELAY_GROUPS),
        "source_conditions": list(SOURCE_CONDITIONS),
        "mechanisms": list(MECHANISMS),
        "scenario_count_per_condition": len(SCENARIOS),
        "main_complete_model_forwards_per_canonical_cohort": 2 * len(SCENARIOS),
        "identity_sentinel_questions": len(identity_qids),
        "identity_sentinel_cohorts": len(identity_qids) // config.batch_size,
        "identity_complete_model_forwards_per_sentinel_cohort": (
            2 * len(IDENTITY_MECHANISMS)
        ),
        "total_complete_model_forwards": (
            (len(qids) // config.batch_size) * 2 * len(SCENARIOS)
            + (len(identity_qids) // config.batch_size)
            * 2
            * len(IDENTITY_MECHANISMS)
        ),
        "complete_model_work": COMPLETE_MODEL_WORK,
        "identity_sentinel_plan": str(args.identity_sentinel_plan),
        "restoration_semantics": RESTORATION_SEMANTICS,
        "unintercepted_channel": UNINTERCEPTED_CHANNEL,
        "validation": validation,
        "ordinary_layers_one_based": [value + 1 for value in ordinary_layers],
        "gla_layers_one_based": [value + 1 for value in gla_layers],
        "elapsed_seconds_after_load": time.monotonic() - started,
        "cohort_seconds": durations,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--identity-sentinel-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
