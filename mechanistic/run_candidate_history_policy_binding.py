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
from .downstream_source_intervention import (
    BatchedSDPADownstreamSourceKVPatcher,
    BatchedSelectiveGDNSourceWritePatcher,
)
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    resolve_answer_tokens,
    tokenize_batch,
)
from .relay_interception import (
    BatchedGLACachedRelayDownstreamRestorer,
    BatchedGLACachedRelayInterceptor,
    BatchedGLARelayWriteCache,
    BatchedSDPACachedRelayDownstreamRestorer,
    BatchedSDPACachedRelayInterceptor,
    BatchedSDPARelayWriteCache,
)
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_candidate_history_entry_factorial import (
    ORDINARY_LAYERS,
    TOKEN_CLASSES,
    _hash_prompt,
    _partition_option_line,
)
from .run_candidate_history_relay_mediation import RELAY_GROUPS, _relay_groups
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_feedback_source_localization import SOURCE_TOKEN_INDICES
from .run_fixed_a_final_query_edge_ablation import _option_line_positions


EXPERIMENT_NAME = "candidate-history policy-binding crossover"
SEMANTIC_CLASS_INDEX = TOKEN_CLASSES.index("semantic")
SCENARIOS = (
    "natural",
    "identity_pre_prefix",
    "feedback_suffix_swapped",
    "relay_task_swapped_R1",
    "relay_task_swapped_R2",
    "relay_task_swapped_R3",
    "relay_task_swapped_R4",
    "relay_task_swapped_all_semantics",
    "relay_task_swapped_all_pre_prefix",
    "feedback_suffix_swapped_restore_semantics",
    "feedback_suffix_swapped_restore_pre_prefix",
)


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise RuntimeError("Existing Stage-C checkpoint uses different questions")
        if arrays["scenario_ids"].astype(str).tolist() != list(SCENARIOS):
            raise RuntimeError("Existing Stage-C checkpoint uses different scenarios")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "conditions": np.asarray(CONDITIONS),
        "scenario_ids": np.asarray(SCENARIOS),
        "completed": np.zeros(n, dtype=bool),
        "rank_contents": np.full((n, 4), "", dtype="<U1"),
        "baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "scenario_logits_raw": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
        "scenario_logits": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
        "semantic_token_counts": np.zeros((2, n, 4), dtype=np.int16),
        "pre_prefix_token_counts": np.zeros((2, n), dtype=np.int16),
        "feedback_source_positions": np.full((2, n, 7), -1, dtype=np.int16),
    }


def _cross_row_cache(
    cache: dict[int, dict[int, tuple[Any, ...]]], donors: dict[int, int]
) -> dict[int, dict[int, tuple[Any, ...]]]:
    """Re-key a clean cache so target rows receive the paired task donor."""

    return {
        layer: {target: rows[donor] for target, donor in donors.items()}
        for layer, rows in cache.items()
    }


def _validate_completed(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    complete = arrays["completed"].astype(bool)
    qidx = np.flatnonzero(complete)
    if not len(qidx):
        return {
            "n_completed": 0.0,
            "natural_max_abs_error": 0.0,
            "identity_max_abs_error": 0.0,
            "feedback_swap_max_abs_change": 0.0,
            "relay_swap_max_abs_change": 0.0,
        }
    raw = np.take(arrays["scenario_logits_raw"], qidx, axis=2)
    corrected = np.take(arrays["scenario_logits"], qidx, axis=2)
    trusted = np.take(arrays["trusted_natural_logits"], qidx, axis=1)
    same = np.take(arrays["same_batch_natural_logits"], qidx, axis=1)
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(trusted)):
        raise RuntimeError("Completed Stage-C rows contain non-finite logits")
    natural_index = SCENARIOS.index("natural")
    identity_index = SCENARIOS.index("identity_pre_prefix")
    feedback_index = SCENARIOS.index("feedback_suffix_swapped")
    joint_index = SCENARIOS.index("relay_task_swapped_all_pre_prefix")
    natural_error = float(np.max(np.abs(raw[:, natural_index] - same)))
    trusted_error = float(np.max(np.abs(corrected[:, natural_index] - trusted)))
    identity_error = float(np.max(np.abs(raw[:, identity_index] - same)))
    if natural_error != 0.0 or trusted_error != 0.0:
        raise RuntimeError(
            f"Stage-C natural identity failed: raw={natural_error}, trusted={trusted_error}"
        )
    if identity_error != 0.0:
        raise RuntimeError(f"Stage-C restoration identity error is {identity_error}")
    return {
        "n_completed": float(complete.sum()),
        "natural_max_abs_error": max(natural_error, trusted_error),
        "identity_max_abs_error": identity_error,
        "feedback_swap_max_abs_change": float(
            np.max(np.abs(raw[:, feedback_index] - raw[:, natural_index]))
        ),
        "relay_swap_max_abs_change": float(
            np.max(np.abs(raw[:, joint_index] - raw[:, natural_index]))
        ),
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
    questions = {str(row["id"]): row for row in manifest["questions"]}
    qids = [str(row["id"]) for row in manifest["questions"]]
    if args.max_cohorts is not None:
        qids = qids[: int(args.max_cohorts) * config.batch_size]
    if len(qids) % config.batch_size:
        raise RuntimeError("Stage C needs complete four-question cohorts")
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    baseline = json.loads(args.baseline.read_text())["results"]
    trusted = [
        json.loads(args.trusted_game.read_text())["results"],
        json.loads(args.trusted_neutral.read_text())["results"],
    ]

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    ordinary_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    ]
    gla_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if tuple(index + 1 for index in ordinary_layers) != ORDINARY_LAYERS:
        raise RuntimeError("Unexpected ordinary-attention layer inventory")
    if len(gla_layers) != 48:
        raise RuntimeError("Unexpected GLA layer inventory")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.npz"
    arrays = _initialize(result_path, qids)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = args.output_dir / "prompt_audit.json"
    durations: list[float] = []
    started = time.monotonic()

    for cohort_start in range(0, len(qids), config.batch_size):
        cohort = qids[cohort_start : cohort_start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if arrays["completed"][indices].all():
            continue
        cohort_started = time.monotonic()
        canonical_batches = [
            _build_batch(config, processor, tokenizer, questions, mappings, cohort, condition)
            for condition in CONDITIONS
        ]
        canonical_width = int(canonical_batches[0]["input_ids"].shape[1])
        cohort_audit: dict[str, Any] = {"question_ids": cohort, "pairs": []}

        for pair_start in range(0, len(cohort), 2):
            pair = cohort[pair_start : pair_start + 2]
            prompts = (
                canonical_batches[0]["prompts"][pair_start : pair_start + 2]
                + canonical_batches[1]["prompts"][pair_start : pair_start + 2]
            )
            messages = (
                canonical_batches[0]["messages"][pair_start : pair_start + 2]
                + canonical_batches[1]["messages"][pair_start : pair_start + 2]
            )
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
            if input_ids.shape[1] > canonical_width:
                raise RuntimeError("Paired Stage-C batch exceeds canonical cohort width")
            if input_ids.shape[1] < canonical_width:
                pad = canonical_width - int(input_ids.shape[1])
                input_ids = torch.nn.functional.pad(
                    input_ids, (pad, 0), value=int(tokenizer.pad_token_id)
                )
                attention_mask = torch.nn.functional.pad(attention_mask, (pad, 0), value=0)
            width = int(input_ids.shape[1])
            final_query = width - 1
            row_sources: list[list[int]] = []
            row_semantics: list[list[list[int]]] = []
            row_groups: list[dict[str, list[int]]] = []
            pair_audit: dict[str, Any] = {"question_ids": pair, "rows": []}

            for condition_index, condition in enumerate(CONDITIONS):
                for local, qid in enumerate(pair):
                    row = 2 * condition_index + local
                    qi = qid_index[qid]
                    prompt_ids = [
                        int(value)
                        for value in tokenizer(prompts[row], add_special_tokens=False)["input_ids"]
                    ]
                    left_pad = width - len(prompt_ids)
                    if input_ids[row, left_pad:].tolist() != prompt_ids:
                        raise RuntimeError("Paired tokenization changed a canonical prompt")
                    old_logits = np.asarray(
                        baseline[qid]["aggregated_ad_logits"], dtype=np.float32
                    )
                    ranks = [
                        LETTERS[int(value)]
                        for value in np.argsort(-old_logits, kind="stable")
                    ]
                    arrays["baseline_logits"][qi] = old_logits
                    arrays["rank_contents"][qi] = np.asarray(ranks)
                    second_question = {
                        **questions[qid],
                        "options": {
                            new: questions[qid]["options"][old]
                            for new, old in mappings[qid]["new_to_original"].items()
                        },
                    }
                    second_positions, second_audit = _option_line_positions(
                        tokenizer, prompts[row], second_question
                    )
                    semantics: list[list[int]] = []
                    rank_classes: list[list[list[int]]] = []
                    for rank, original in enumerate(ranks):
                        second_letter = mappings[qid]["original_to_new"][original]
                        classes, _ = _partition_option_line(
                            second_positions[second_letter],
                            second_audit[second_letter]["tokens"],
                        )
                        physical = [
                            [left_pad + value for value in positions]
                            for positions in classes
                        ]
                        semantic = physical[SEMANTIC_CLASS_INDEX]
                        if not semantic:
                            raise RuntimeError("Every candidate needs semantic wordpieces")
                        semantics.append(semantic)
                        rank_classes.append(physical)
                        arrays["semantic_token_counts"][condition_index, qi, rank] = len(semantic)

                    partition, position_audit = _cue_source_partition(
                        tokenizer,
                        prompts[row],
                        messages[row],
                        questions[qid],
                        second_question,
                        condition,
                        ranks,
                        mappings[qid]["original_to_new"],
                    )
                    if width - int(position_audit["prompt_length"]) != left_pad:
                        raise RuntimeError("Independent Stage-C position audits disagree")
                    feedback = [
                        left_pad
                        + partition[SOURCE_NAMES.index(f"feedback_token_{index}")][0]
                        for index in SOURCE_TOKEN_INDICES
                    ]
                    if feedback != list(range(feedback[0], feedback[0] + 7)):
                        raise RuntimeError("Feedback suffix is not seven aligned contiguous tokens")
                    groups = _relay_groups(rank_classes, partition, left_pad, final_query)
                    pre_prefix = sorted(
                        position
                        for name in RELAY_GROUPS
                        if name != "final_assistant_prefix"
                        for position in groups[name]
                    )
                    arrays["pre_prefix_token_counts"][condition_index, qi] = len(pre_prefix)
                    arrays["feedback_source_positions"][condition_index, qi] = feedback
                    digest = _hash_prompt(prompts[row])
                    if digest != trusted[condition_index][qid]["prompt_hash"]:
                        raise RuntimeError("Prompt hash differs from trusted natural run")
                    arrays["prompt_hashes"][condition_index, qi] = digest
                    arrays["trusted_natural_logits"][condition_index, qi] = np.asarray(
                        trusted[condition_index][qid]["aggregated_ad_logits"],
                        dtype=np.float32,
                    )
                    row_sources.append(feedback)
                    row_semantics.append(semantics)
                    row_groups.append(groups)
                    pair_audit["rows"].append(
                        {
                            "row": row,
                            "condition": condition,
                            "question_id": qid,
                            "prompt_hash": digest,
                            "feedback_positions": feedback,
                            "rank_contents": ranks,
                            "rank_semantic_positions": semantics,
                            "relay_groups": groups,
                        }
                    )

            donors = {row: row + 2 if row < 2 else row - 2 for row in range(4)}
            for target, donor in donors.items():
                if row_sources[target] != row_sources[donor]:
                    raise RuntimeError("Game/Neutral feedback positions are not aligned")
                if row_semantics[target] != row_semantics[donor]:
                    raise RuntimeError("Game/Neutral semantic relay positions are not aligned")
                target_pre = sorted(
                    position
                    for name in RELAY_GROUPS
                    if name != "final_assistant_prefix"
                    for position in row_groups[target][name]
                )
                donor_pre = sorted(
                    position
                    for name in RELAY_GROUPS
                    if name != "final_assistant_prefix"
                    for position in row_groups[donor][name]
                )
                if target_pre != donor_pre:
                    raise RuntimeError("Game/Neutral pre-prefix relays are not aligned")
                if input_ids[target, target_pre].tolist() != input_ids[donor, donor_pre].tolist():
                    raise RuntimeError("Game/Neutral relay token identities differ")

            all_pre_prefix = {
                row: sorted(
                    position
                    for name in RELAY_GROUPS
                    if name != "final_assistant_prefix"
                    for position in row_groups[row][name]
                )
                for row in range(4)
            }
            all_semantics = {
                row: sorted(position for values in row_semantics[row] for position in values)
                for row in range(4)
            }
            ordinary_cache = BatchedSDPARelayWriteCache(
                parts, all_pre_prefix, ordinary_layers
            )
            gla_cache = BatchedGLARelayWriteCache(parts, all_pre_prefix, gla_layers)
            try:
                natural = _aggregate_logits(
                    _forward(model, parts, input_ids, attention_mask), variant_ids
                )
            finally:
                gla_cache.close()
                ordinary_cache.close()
            if set(ordinary_cache.cache) != set(ordinary_layers):
                raise RuntimeError("Stage-C ordinary relay cache is incomplete")
            if set(gla_cache.cache) != set(gla_layers):
                raise RuntimeError("Stage-C GLA relay cache is incomplete")
            donor_ordinary_cache = _cross_row_cache(ordinary_cache.cache, donors)
            donor_gla_cache = _cross_row_cache(gla_cache.cache, donors)

            scenario_outputs: list[np.ndarray] = [natural]
            for scenario in SCENARIOS[1:]:
                ordinary = None
                gla = None
                try:
                    selected: dict[int, list[int]] | None = None
                    if scenario == "identity_pre_prefix":
                        selected = all_pre_prefix
                        ordinary = BatchedSDPACachedRelayDownstreamRestorer(
                            parts, selected, ordinary_layers, ordinary_cache.cache
                        )
                        gla = BatchedGLACachedRelayDownstreamRestorer(
                            parts, selected, gla_layers, gla_cache.cache
                        )
                    elif scenario == "feedback_suffix_swapped":
                        ordinary = BatchedSDPADownstreamSourceKVPatcher(
                            parts,
                            {
                                row: (donors[row], row_sources[row], ordinary_layers)
                                for row in range(4)
                            },
                        )
                        gla = BatchedSelectiveGDNSourceWritePatcher(
                            parts,
                            {
                                row: (donors[row], row_sources[row], gla_layers)
                                for row in range(4)
                            },
                            preserve_source_output=True,
                        )
                    elif scenario.startswith("relay_task_swapped_R"):
                        rank = int(scenario[-1]) - 1
                        selected = {row: row_semantics[row][rank] for row in range(4)}
                    elif scenario == "relay_task_swapped_all_semantics":
                        selected = all_semantics
                    elif scenario == "relay_task_swapped_all_pre_prefix":
                        selected = all_pre_prefix
                    elif scenario.startswith("feedback_suffix_swapped_restore_"):
                        selected = (
                            all_semantics if scenario.endswith("semantics") else all_pre_prefix
                        )
                        ordinary = BatchedSDPACachedRelayInterceptor(
                            parts,
                            {
                                row: (
                                    donors[row], row_sources[row], selected[row], ordinary_layers
                                )
                                for row in range(4)
                            },
                            ordinary_cache.cache,
                        )
                        gla = BatchedGLACachedRelayInterceptor(
                            parts,
                            {
                                row: (donors[row], row_sources[row], selected[row], gla_layers)
                                for row in range(4)
                            },
                            gla_cache.cache,
                        )
                    else:
                        raise RuntimeError(f"Unhandled Stage-C scenario: {scenario}")

                    if selected is not None and ordinary is None and gla is None:
                        ordinary = BatchedSDPACachedRelayDownstreamRestorer(
                            parts, selected, ordinary_layers, donor_ordinary_cache
                        )
                        gla = BatchedGLACachedRelayDownstreamRestorer(
                            parts, selected, gla_layers, donor_gla_cache
                        )
                    output = _aggregate_logits(
                        _forward(model, parts, input_ids, attention_mask), variant_ids
                    )
                    if hasattr(ordinary, "assert_fired"):
                        ordinary.assert_fired()
                    if hasattr(gla, "assert_fired"):
                        gla.assert_fired()
                    if not np.all(np.isfinite(output)):
                        raise RuntimeError(f"Non-finite Stage-C output in {scenario}")
                    scenario_outputs.append(output)
                finally:
                    if gla is not None:
                        gla.close()
                    if ordinary is not None:
                        ordinary.close()

            stacked = np.stack(scenario_outputs, axis=0)
            for condition_index in range(2):
                for local, qid in enumerate(pair):
                    row = 2 * condition_index + local
                    qi = qid_index[qid]
                    trusted_logits = arrays["trusted_natural_logits"][condition_index, qi]
                    arrays["same_batch_natural_logits"][condition_index, qi] = natural[row]
                    arrays["scenario_logits_raw"][condition_index, :, qi] = stacked[:, row]
                    arrays["scenario_logits"][condition_index, :, qi] = (
                        trusted_logits[None, :] + stacked[:, row] - natural[row][None, :]
                    )
            cohort_audit["pairs"].append(pair_audit)

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        validation = _validate_completed(arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"candidate-history policy binding: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.2f}; identity_error={validation['identity_max_abs_error']}; "
            f"feedback_liveness={validation['feedback_swap_max_abs_change']:.6f}; "
            f"relay_liveness={validation['relay_swap_max_abs_change']:.6f}",
            flush=True,
        )
        if not audit_path.exists():
            cohort_audit["scenario_ids"] = list(SCENARIOS)
            cohort_audit["ordinary_layers_one_based"] = [value + 1 for value in ordinary_layers]
            cohort_audit["gla_layers_one_based"] = [value + 1 for value in gla_layers]
            audit_path.write_text(json.dumps(cohort_audit, indent=2) + "\n")

    validation = _validate_completed(arrays)
    metadata = {
        "experiment": EXPERIMENT_NAME,
        "questions": len(qids),
        "conditions": list(CONDITIONS),
        "scenarios": list(SCENARIOS),
        "complete_model_forwards_per_canonical_cohort": 2 * len(SCENARIOS),
        "total_complete_model_forwards": (len(qids) // 4) * 2 * len(SCENARIOS),
        "paired_subbatches_per_canonical_cohort": 2,
        "relay_transplant": (
            "Reciprocal same-question Game/Neutral transplant of complete outgoing "
            "ordinary-attention K/V and GLA k/v/g/beta at all applicable layers; "
            "recipient relay-token local output is preserved."
        ),
        "policy_source_positive_control": (
            "Reciprocal seven-token feedback-suffix outgoing-state swap at all ordinary "
            "and GLA layers with source-token local output preserved."
        ),
        "convolution_safety": (
            "The final assistant prefix is never restored or transplanted; it recomputes "
            "freely from the edited pre-prefix state."
        ),
        "old_evidence_axis_status": (
            "The frozen remapping plan contains one mapping per question and no clean "
            "same-candidate high/low old-evidence donor. Stage C therefore uses frozen "
            "R1-R4 strata and does not fabricate that unavailable factorial axis."
        ),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
