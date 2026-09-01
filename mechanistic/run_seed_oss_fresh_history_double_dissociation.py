from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .collect_cross_model_behavioral_gate import (
    _assert_prompt_pair,
    _scenario_messages,
)
from .collect_seed_oss_fresh_score_directions import _load_discovery
from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSDPAQuerySourceAttentionAblator
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .run_all_candidate_matched_relay import _specs
from .run_fixed_a_final_query_edge_ablation import _option_line_positions
from .run_fresh_history_double_dissociation import (
    FreshOptionLineScrubber,
    _random_control,
    _unit_unique_fresh,
)
from .run_seed_oss_matching_history_blockade import (
    ATTENTION_LAYERS_ONE_BASED,
    MODEL_ID,
    MODEL_REVISION,
    _aggregate_final_logits,
    _forward_final_logits,
)


SCENARIOS = (
    "trusted_natural",
    "complete_path_natural",
    "identity_hook",
    "fresh_scrub",
    "matching_history_blockade",
    "matching_plus_fresh",
    "dose_matched_random",
    "matching_plus_random",
)
TRUSTED_NATURAL, COMPLETE_NATURAL, IDENTITY, FRESH, MATCHING, JOINT, RANDOM, MATCHING_RANDOM = range(8)
CONDITIONS = ("Game", "Neutral")
TRUSTED_SCENARIOS = ("incorrect_again_remapped", "lost_again_remapped")
GROUPS = ("semantic_wordpieces", "option_newline")
LAYERS = tuple(range(64))
OLD_TARGET = 0
FRESH_TARGET = 1


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _direction_geometry(
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if directions.ndim != 4 or tuple(directions.shape[:3]) != (64, 2, 2):
        raise ValueError(f"Unexpected Seed score-direction shape: {directions.shape}")
    width = int(directions.shape[-1])
    old = np.empty((64, 2, width), dtype=np.float32)
    fresh = np.empty_like(old)
    random = np.empty_like(old)
    result: dict[str, Any] = {"model_width": width, "groups": {}}
    for group_index, group_name in enumerate(GROUPS):
        cosines = []
        residual_norms = []
        random_old = []
        random_fresh = []
        for layer in LAYERS:
            raw_old = np.asarray(
                directions[layer, group_index, OLD_TARGET], dtype=np.float32
            )
            raw_fresh = np.asarray(
                directions[layer, group_index, FRESH_TARGET], dtype=np.float32
            )
            old[layer, group_index] = raw_old / max(
                float(np.linalg.norm(raw_old)), 1e-12
            )
            fresh[layer, group_index] = _unit_unique_fresh(
                old[layer, group_index], raw_fresh
            )
            random[layer, group_index] = _random_control(
                old[layer, group_index],
                fresh[layer, group_index],
                seed=20260830 + layer * 101 + group_index * 10007,
            )
            raw_fresh_unit = raw_fresh / max(float(np.linalg.norm(raw_fresh)), 1e-12)
            cosines.append(float(old[layer, group_index] @ raw_fresh_unit))
            residual_norms.append(
                float(
                    np.linalg.norm(
                        raw_fresh_unit
                        - (raw_fresh_unit @ old[layer, group_index])
                        * old[layer, group_index]
                    )
                )
            )
            random_old.append(float(random[layer, group_index] @ old[layer, group_index]))
            random_fresh.append(
                float(random[layer, group_index] @ fresh[layer, group_index])
            )
        result["groups"][group_name] = {
            "raw_old_fresh_cosine_by_layer": cosines,
            "unique_fresh_pre_normalization_norm_by_layer": residual_norms,
            "max_abs_random_old_dot": float(np.max(np.abs(random_old))),
            "max_abs_random_unique_fresh_dot": float(np.max(np.abs(random_fresh))),
        }
    return old, fresh, random, result


def _initialize(path: Path, qids: list[str], split: np.ndarray) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses another question order")
        if arrays["scenarios"].astype(str).tolist() != list(SCENARIOS):
            raise ValueError("Existing checkpoint uses another scenario inventory")
        return arrays
    n = len(qids)
    audit_shape = (2, len(SCENARIOS), n, 64, 2, 4)
    return {
        "question_ids": np.asarray(qids),
        "split": split,
        "scenarios": np.asarray(SCENARIOS),
        "completed": np.zeros(n, dtype=bool),
        "baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "rank_letters": np.full((n, 4), "", dtype="<U1"),
        "logits": np.full((2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32),
        "pre_fresh": np.full(audit_shape, np.nan, dtype=np.float32),
        "post_fresh": np.full(audit_shape, np.nan, dtype=np.float32),
        "pre_old": np.full(audit_shape, np.nan, dtype=np.float32),
        "post_old": np.full(audit_shape, np.nan, dtype=np.float32),
        "dose_l2": np.full(audit_shape, np.nan, dtype=np.float32),
        "trusted_max_abs_error": np.full((2, n), np.nan, dtype=np.float32),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
    }


def _build_batch(
    config: ExperimentConfig,
    processor: Any,
    tokenizer: Any,
    questions: dict[str, dict[str, Any]],
    mappings: dict[str, dict[str, Any]],
    qids: list[str],
    scenario: str,
) -> dict[str, Any]:
    prompts: list[str] = []
    messages: list[list[dict[str, str]]] = []
    remapped_questions: list[dict[str, Any]] = []
    token_rows: list[list[int]] = []
    for qid in qids:
        row_messages, remapped = _scenario_messages(
            scenario, questions[qid], mappings[qid]["new_to_original"]
        )
        if remapped is None:
            raise RuntimeError("Seed fresh-removal runner requires remapped 2P prompts")
        prompt = render_chat(
            processor,
            row_messages,
            config.disable_thinking,
            config.chat_serialization,
            config.chat_template_kwargs,
        )
        ids = [
            int(value)
            for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ]
        prompts.append(prompt)
        messages.append(row_messages)
        remapped_questions.append(remapped)
        token_rows.append(ids)
    input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
    width = int(input_ids.shape[1])
    for row, ids in enumerate(token_rows):
        left_pad = width - len(ids)
        if input_ids[row, left_pad:].tolist() != ids:
            raise RuntimeError("Exact historical-cohort tokenization changed")
    return {
        "prompts": prompts,
        "messages": messages,
        "remapped_questions": remapped_questions,
        "token_rows": token_rows,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def _removal_diagnostics(
    arrays: dict[str, np.ndarray], completed: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, scenario in (("fresh", FRESH), ("joint", JOINT)):
        pre_fresh = arrays["pre_fresh"][:, scenario, completed]
        post_fresh = arrays["post_fresh"][:, scenario, completed]
        pre_old = arrays["pre_old"][:, scenario, completed]
        post_old = arrays["post_old"][:, scenario, completed]
        fresh_fraction = 1.0 - float(np.abs(post_fresh).sum()) / max(
            float(np.abs(pre_fresh).sum()), 1e-12
        )
        old_change = float(np.abs(post_old - pre_old).sum()) / max(
            float(np.abs(pre_old).sum()), 1e-12
        )
        result[label] = {
            "fresh_absolute_coordinate_removed_fraction": fresh_fraction,
            "old_absolute_coordinate_relative_change": old_change,
        }
    result["same_dose_max_abs_error"] = {
        "fresh_vs_random": float(
            np.nanmax(
                np.abs(
                    arrays["dose_l2"][:, FRESH, completed]
                    - arrays["dose_l2"][:, RANDOM, completed]
                )
            )
        ),
        "joint_vs_matching_random": float(
            np.nanmax(
                np.abs(
                    arrays["dose_l2"][:, JOINT, completed]
                    - arrays["dose_l2"][:, MATCHING_RANDOM, completed]
                )
            )
        ),
    }
    return result


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(args.config)
    if config.model_id != MODEL_ID or config.model_revision != MODEL_REVISION:
        raise ValueError("Requires the pinned Seed-OSS 36B revision")
    if (
        config.prompt_mode != "baseline_matched_empty_history"
        or config.feedback_variant != "token_matched_test"
        or config.chat_serialization != "hf_template"
        or config.model_loader != "causal_lm"
        or config.attn_implementation != "sdpa"
        or int(config.batch_size) != 4
    ):
        raise ValueError("Requires the canonical Seed native-template batch-four SDPA regime")

    manifest = json.loads(Path(config.manifest_path).read_text())["questions"]
    qids = [str(row["id"]) for row in manifest]
    questions = {str(row["id"]): row for row in manifest}
    if len(qids) != 500:
        raise ValueError(f"Expected 500 questions, got {len(qids)}")
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    if set(mappings) != set(qids):
        raise ValueError("Remapping plan does not cover the frozen manifest exactly")
    discovery_ids = _load_discovery(args.discovery_plan)
    split = np.asarray(
        ["discovery" if qid in discovery_ids else "confirmation" for qid in qids]
    )
    if not np.any(split == "discovery") or not np.any(split == "confirmation"):
        raise ValueError("Frozen split is degenerate")
    trusted_payload = json.loads(args.trusted_behavior.read_text())
    baseline = trusted_payload["scenarios"]["baseline"]
    trusted = [
        trusted_payload["scenarios"][scenario] for scenario in TRUSTED_SCENARIOS
    ]

    model_load_started = time.monotonic()
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _text, token_id in resolved[letter]})
        for letter in LETTERS
    }
    ordinary_layers = tuple(
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if tuple(index + 1 for index in ordinary_layers) != ATTENTION_LAYERS_ONE_BASED:
        raise RuntimeError("Unexpected Seed attention-layer inventory")
    loaded = torch.load(args.score_directions, map_location="cpu", weights_only=True)
    direction_array = (
        loaded.float().numpy()
        if hasattr(loaded, "numpy")
        else loaded["directions"].float().numpy()
    )
    old_directions, fresh_directions, random_directions, geometry = (
        _direction_geometry(direction_array)
    )
    model_load_seconds = time.monotonic() - model_load_started
    print(f"MODEL_LOADED seconds={model_load_seconds:.3f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.npz"
    arrays = _initialize(result_path, qids, split)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = args.output_dir / "prompt_audit.json"
    started = time.monotonic()
    durations: list[float] = []
    completed_cohorts = 0
    pending = {qid for qid, done in zip(qids, arrays["completed"]) if not bool(done)}

    for start in range(0, len(qids), 4):
        cohort = qids[start : start + 4]
        if not set(cohort) & pending:
            continue
        cohort_started = time.monotonic()
        indices = [qid_index[qid] for qid in cohort]
        for qid, qi in zip(cohort, indices):
            old_logits = np.asarray(
                baseline[qid]["aggregated_ad_logits"], dtype=np.float32
            )
            arrays["baseline_logits"][qi] = old_logits
            arrays["rank_letters"][qi] = np.asarray(
                [LETTERS[int(index)] for index in np.argsort(-old_logits, kind="stable")]
            )

        condition_prompts: list[str] = []
        for condition_index, trusted_scenario in enumerate(TRUSTED_SCENARIOS):
            batch = _build_batch(
                config,
                processor,
                tokenizer,
                questions,
                mappings,
                cohort,
                trusted_scenario,
            )
            condition_prompts.append(batch["prompts"][0])
            width = int(batch["input_ids"].shape[1])
            source_positions: list[list[list[int]]] = []
            query_positions: list[list[list[int]]] = []
            semantic_positions: list[list[list[int]]] = []
            newline_positions: list[list[list[int]]] = []
            row_audits: list[dict[str, Any]] = []
            for row, (qid, qi, remapped) in enumerate(
                zip(cohort, indices, batch["remapped_questions"])
            ):
                left_pad = width - len(batch["token_rows"][row])
                first_lines, first_audit = _option_line_positions(
                    tokenizer, batch["prompts"][row], questions[qid]
                )
                second_lines, second_audit = _option_line_positions(
                    tokenizer, batch["prompts"][row], remapped
                )
                ranks = arrays["rank_letters"][qi].astype(str).tolist()
                sources: list[list[int]] = []
                queries: list[list[int]] = []
                semantics: list[list[int]] = []
                newlines: list[list[int]] = []
                for original in ranks:
                    displayed = mappings[qid]["original_to_new"][original]
                    source = [left_pad + int(value) for value in first_lines[original]]
                    query = [left_pad + int(value) for value in second_lines[displayed]]
                    if len(query) < 5 or not source or max(source) >= min(query):
                        raise RuntimeError(f"{qid}: invalid matching 1P/2P positions")
                    newline_text = tokenizer.decode(
                        [int(batch["input_ids"][row, query[-1]])]
                    )
                    if "\n" not in newline_text:
                        raise RuntimeError(f"{qid}: 2P line lacks standalone newline")
                    sources.append(source)
                    queries.append(query)
                    semantics.append(query[3:-1])
                    newlines.append([query[-1]])
                source_positions.append(sources)
                query_positions.append(queries)
                semantic_positions.append(semantics)
                newline_positions.append(newlines)
                prompt_hash = _hash_prompt(batch["prompts"][row])
                arrays["prompt_hashes"][condition_index, qi] = prompt_hash
                if prompt_hash != trusted[condition_index][qid]["prompt_hash"]:
                    raise RuntimeError(f"{qid}: trusted prompt hash mismatch")
                row_audits.append(
                    {
                        "question_id": qid,
                        "rank_letters": ranks,
                        "first_option_lines": first_audit,
                        "second_option_lines": second_audit,
                        "semantic_positions_padded": semantics,
                        "newline_positions_padded": newlines,
                    }
                )

            matching_specs = _specs(
                ordinary_layers,
                source_positions,
                query_positions,
                tuple(range(4)),
                False,
            )
            dose_schedules: dict[str, np.ndarray] = {}
            for scenario_index, scenario in enumerate(SCENARIOS):
                hook = None
                ablator = None
                with contextlib.ExitStack() as stack:
                    if scenario in {
                        "identity_hook",
                        "fresh_scrub",
                        "matching_plus_fresh",
                        "dose_matched_random",
                        "matching_plus_random",
                    }:
                        mode = (
                            "identity"
                            if scenario == "identity_hook"
                            else "random"
                            if scenario in {"dose_matched_random", "matching_plus_random"}
                            else "fresh"
                        )
                        prescribed = None
                        if mode == "random":
                            prescribed = dose_schedules[
                                "joint" if scenario == "matching_plus_random" else "fresh"
                            ]
                        hook = stack.enter_context(
                            FreshOptionLineScrubber(
                                parts,
                                semantic_positions,
                                newline_positions,
                                old_directions,
                                fresh_directions,
                                random_directions,
                                mode,
                                prescribed_dose=prescribed,
                            )
                        )
                    if scenario in {
                        "matching_history_blockade",
                        "matching_plus_fresh",
                        "matching_plus_random",
                    }:
                        ablator = stack.enter_context(
                            BatchedSDPAQuerySourceAttentionAblator(parts, matching_specs)
                        )
                    output = _aggregate_final_logits(
                        _forward_final_logits(
                            model,
                            parts,
                            batch["input_ids"],
                            batch["attention_mask"],
                        ),
                        variant_ids,
                    )
                if not np.all(np.isfinite(output)):
                    raise RuntimeError(f"Non-finite logits in {CONDITIONS[condition_index]}/{scenario}")
                if ablator is not None and set(ablator.layers_seen) != set(ordinary_layers):
                    raise RuntimeError(f"{scenario}: matching blockade missed Seed layers")
                arrays["logits"][condition_index, scenario_index, indices] = output
                if hook is not None:
                    local = hook.arrays()
                    if scenario == "fresh_scrub":
                        dose_schedules["fresh"] = local["dose_l2"]
                    elif scenario == "matching_plus_fresh":
                        dose_schedules["joint"] = local["dose_l2"]
                    for row, qi in enumerate(indices):
                        for name in (
                            "pre_fresh",
                            "post_fresh",
                            "pre_old",
                            "post_old",
                            "dose_l2",
                        ):
                            arrays[name][condition_index, scenario_index, qi] = local[name][
                                :, row
                            ]

            for row, (qid, qi) in enumerate(zip(cohort, indices)):
                reference = np.asarray(
                    trusted[condition_index][qid]["aggregated_ad_logits"],
                    dtype=np.float32,
                )
                error = float(
                    np.max(
                        np.abs(
                            arrays["logits"][condition_index, TRUSTED_NATURAL, qi]
                            - reference
                        )
                    )
                )
                arrays["trusted_max_abs_error"][condition_index, qi] = error
                if error != 0.0:
                    raise RuntimeError(f"{qid}: trusted natural max error {error}")
            if not audit_path.exists():
                _atomic_json(
                    audit_path,
                    {
                        "condition": CONDITIONS[condition_index],
                        "rendered_prompt": batch["prompts"][0],
                        "messages": batch["messages"][0],
                        "prompt_hash": arrays["prompt_hashes"][
                            condition_index, indices[0]
                        ].item(),
                        "rows": row_audits,
                        "attention_and_scrub_layers_one_based": list(
                            ATTENTION_LAYERS_ONE_BASED
                        ),
                        "groups": list(GROUPS),
                    },
                )
        _assert_prompt_pair(condition_prompts[0], condition_prompts[1])
        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        pending.difference_update(cohort)
        completed_cohorts += 1
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"SEED_FRESH_REMOVAL questions={int(arrays['completed'].sum())}/500 "
            f"cohort_seconds={duration:.3f}",
            flush=True,
        )
        if args.max_cohorts is not None and completed_cohorts >= int(args.max_cohorts):
            break

    completed = arrays["completed"]
    complete_natural_error = float(
        np.nanmax(
            np.abs(
                arrays["logits"][:, COMPLETE_NATURAL, completed]
                - arrays["logits"][:, TRUSTED_NATURAL, completed]
            )
        )
    )
    identity_error = float(
        np.nanmax(
            np.abs(
                arrays["logits"][:, IDENTITY, completed]
                - arrays["logits"][:, COMPLETE_NATURAL, completed]
            )
        )
    )
    removal = _removal_diagnostics(arrays, completed)
    metadata = {
        "experiment": "Seed-OSS fresh-2P by matching-history double dissociation",
        "dataset": "TriviaMC" if qids[0].startswith("triviamc_") else "SimpleMC",
        "config": config.as_dict(),
        "questions": len(qids),
        "conditions": list(CONDITIONS),
        "scenarios": list(SCENARIOS),
        "groups": list(GROUPS),
        "layers_one_based": list(ATTENTION_LAYERS_ONE_BASED),
        "complete_model_forwards_per_cohort": 16,
        "complete": bool(completed.all()),
        "model_load_seconds": model_load_seconds,
        "elapsed_seconds_after_model_load": time.monotonic() - started,
        "cohort_seconds": durations,
        "validity": {
            "trusted_natural_max_abs_error": float(
                np.nanmax(arrays["trusted_max_abs_error"][:, completed])
            ),
            "complete_path_natural_max_abs_error": complete_natural_error,
            "identity_hook_max_abs_error": identity_error,
            "all_finite_completed_logits": bool(
                np.isfinite(arrays["logits"][:, :, completed]).all()
            ),
            "removal": removal,
        },
        "direction_geometry": geometry,
        "score_directions_path": str(args.score_directions),
        "score_directions_sha256": hashlib.sha256(
            args.score_directions.read_bytes()
        ).hexdigest(),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    _atomic_json(args.output_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--trusted-behavior", type=Path, required=True)
    parser.add_argument("--score-directions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
