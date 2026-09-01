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

from . import LETTERS
from .collect_cross_model_behavioral_gate import (
    _assert_prompt_pair,
    _scenario_messages,
)
from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSDPADownstreamSourceKVPatcher
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .run_seed_oss_matching_history_blockade import (
    ATTENTION_LAYERS_ONE_BASED,
    MODEL_ID,
    MODEL_REVISION,
    TASKS,
    TRUSTED_SCENARIOS,
    _aggregate_final_logits,
    _forward_final_logits,
)


SCENARIOS = ("natural", "identity_complete_suffix", "reciprocal_complete_suffix")
FEEDBACK_SUFFIX = {
    "Game": "incorrect. Choose the answer again.",
    "Neutral": "lost. Choose the answer again.",
}


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _feedback_suffix_positions(tokenizer: Any, prompt: str, task: str) -> tuple[list[int], dict[str, Any]]:
    needle = FEEDBACK_SUFFIX[task]
    starts = [index for index in range(len(prompt)) if prompt.startswith(needle, index)]
    if len(starts) != 1:
        raise RuntimeError(f"Expected one {task} feedback suffix, found {starts}")
    interval = (starts[0], starts[0] + len(needle))
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and right > interval[0] and left < interval[1]
    ]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise RuntimeError("Feedback suffix token positions are empty or noncontiguous")
    ids = [int(value) for value in encoded["input_ids"]]
    return positions, {
        "text": needle,
        "character_interval": list(interval),
        "positions_unpadded": positions,
        "tokens": [tokenizer.decode([ids[position]]).replace("\n", "\\n") for position in positions],
    }


def _initialize(path: Path, qids: list[str], source_token_count: int) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise RuntimeError("Existing checkpoint uses different questions")
        if arrays["scenario_ids"].astype(str).tolist() != list(SCENARIOS):
            raise RuntimeError("Existing checkpoint uses different scenarios")
        if arrays["source_positions"].shape[-1] != source_token_count:
            raise RuntimeError("Existing checkpoint uses a different suffix token count")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "scenario_ids": np.asarray(SCENARIOS),
        "completed": np.zeros(n, dtype=bool),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "scenario_final_logits_raw": np.full((2, 3, n, 4), np.nan, dtype=np.float32),
        "scenario_final_logits": np.full((2, 3, n, 4), np.nan, dtype=np.float32),
        "source_positions": np.full((2, n, source_token_count), -1, dtype=np.int16),
        "identity_error_by_question": np.full((2, n), np.nan, dtype=np.float32),
    }


def _pad_to_width(tokenizer: Any, input_ids: Any, attention_mask: Any, width: int):
    import torch

    if input_ids.shape[1] > width:
        raise RuntimeError("Subbatch exceeds canonical cohort width")
    if input_ids.shape[1] < width:
        pad = width - int(input_ids.shape[1])
        input_ids = torch.nn.functional.pad(
            input_ids, (pad, 0), value=int(tokenizer.pad_token_id)
        )
        attention_mask = torch.nn.functional.pad(attention_mask, (pad, 0), value=0)
    return input_ids, attention_mask


def _patched_forward(
    model: Any,
    parts: Any,
    input_ids: Any,
    attention_mask: Any,
    variant_ids: dict[str, list[int]],
    layers: list[int],
    donors: dict[int, int],
    source_positions: list[list[int]],
) -> tuple[np.ndarray, dict[str, int]]:
    patcher = BatchedSDPADownstreamSourceKVPatcher(
        parts,
        {
            row: (donors[row], source_positions[row], layers)
            for row in sorted(donors)
        },
    )
    try:
        output = _aggregate_final_logits(
            _forward_final_logits(model, parts, input_ids, attention_mask),
            variant_ids,
        )
        patcher.assert_fired()
        stats = {
            "sdpa_calls": int(patcher.sdpa_calls),
            "patched_position_count": int(patcher.patched_position_count),
            "unique_layers_seen": len(patcher.layers_seen),
        }
    finally:
        patcher.close()
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Non-finite patched logits")
    return output, stats


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(args.config)
    if config.model_id != MODEL_ID or config.model_revision != MODEL_REVISION:
        raise ValueError("Requires the pinned configured model revision")
    if (
        config.model_loader not in {"causal_lm", "multimodal"}
        or config.chat_serialization != "hf_template"
        or config.attn_implementation != "sdpa"
        or config.batch_size < 1
    ):
        raise ValueError("Requires a supported native HF text path with SDPA")
    if config.prompt_mode != "baseline_matched_empty_history" or config.feedback_variant != "token_matched_test":
        raise ValueError("Requires the canonical clean token-matched prompt")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {str(row["id"]): row for row in manifest["questions"]}
    qids = [str(row["id"]) for row in manifest["questions"]]
    if config.question_ids is not None:
        wanted = set(config.question_ids)
        qids = [qid for qid in qids if qid in wanted]
        if set(qids) != wanted:
            raise RuntimeError("Configured question IDs are absent from the manifest")
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
    trusted = [
        trusted_payload["scenarios"][scenario] for scenario in TRUSTED_SCENARIOS
    ]
    missing = [
        qid for qid in qids
        if qid not in mappings or any(qid not in rows for rows in trusted)
    ]
    if missing:
        raise RuntimeError(f"Frozen inputs are missing {len(missing)} questions")

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
        raise RuntimeError("Seed unexpectedly exposes a GLA/recurrent layer")
    text_config = getattr(model.config, "text_config", model.config)
    query_heads = int(text_config.num_attention_heads)
    key_value_heads = int(text_config.num_key_value_heads)
    if MODEL_ID == "ByteDance-Seed/Seed-OSS-36B-Instruct" and (
        query_heads != 80 or key_value_heads != 8
    ):
        raise RuntimeError("Unexpected Seed grouped-query-attention configuration")
    print(f"MODEL_LOADED seconds={model_load_seconds:.3f}", flush=True)

    first_positions: list[list[int]] = []
    first_prompts: list[str] = []
    first_audits: list[dict[str, Any]] = []
    for task_index, scenario in enumerate(TRUSTED_SCENARIOS):
        messages, _ = _scenario_messages(
            scenario, questions[qids[0]], mappings[qids[0]]["new_to_original"]
        )
        prompt = render_chat(
            processor,
            messages,
            config.disable_thinking,
            config.chat_serialization,
            config.chat_template_kwargs,
        )
        positions, audit = _feedback_suffix_positions(tokenizer, prompt, TASKS[task_index])
        first_positions.append(positions)
        first_prompts.append(prompt)
        first_audits.append(audit)
    _assert_prompt_pair(first_prompts[0], first_prompts[1])
    if len(first_positions[0]) != len(first_positions[1]):
        raise RuntimeError("Game and Neutral feedback suffixes tokenize to different lengths")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.npz"
    arrays = _initialize(result_path, qids, len(first_positions[0]))
    completed_at_resume = arrays["completed"].astype(bool)
    if completed_at_resume.any():
        arrays["scenario_final_logits"][:, 0, completed_at_resume] = arrays[
            "trusted_natural_logits"
        ][:, completed_at_resume]
        arrays["scenario_final_logits"][:, 1, completed_at_resume] = arrays[
            "trusted_natural_logits"
        ][:, completed_at_resume]
        atomic_save_npz(result_path, **arrays)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = args.output_dir / "prompt_audit.json"
    durations: list[float] = []
    patch_stats: list[dict[str, Any]] = []
    started = time.monotonic()

    for cohort_start in range(0, len(qids), config.batch_size):
        cohort = qids[cohort_start : cohort_start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if arrays["completed"][indices].all():
            continue
        cohort_started = time.monotonic()
        task_prompts: list[list[str]] = [[], []]
        task_messages: list[list[list[dict[str, str]]]] = [[], []]
        for task_index, scenario in enumerate(TRUSTED_SCENARIOS):
            for qid in cohort:
                messages, remapped = _scenario_messages(
                    scenario, questions[qid], mappings[qid]["new_to_original"]
                )
                if remapped is None:
                    raise RuntimeError("Feedback crossover requires remapped 2P options")
                prompt = render_chat(
                    processor,
                    messages,
                    config.disable_thinking,
                    config.chat_serialization,
                    config.chat_template_kwargs,
                )
                task_prompts[task_index].append(prompt)
                task_messages[task_index].append(messages)
        for game_prompt, neutral_prompt in zip(task_prompts[0], task_prompts[1]):
            _assert_prompt_pair(game_prompt, neutral_prompt)
        canonical_width = max(
            len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            for prompts in task_prompts for prompt in prompts
        )
        cohort_audit: dict[str, Any] = {"question_ids": cohort, "pairs": []}

        # Gemma 4's current Hugging Face implementation is not numerically safe
        # under variable-length left-padded batches.  With a configured
        # one-question cohort, use four same-length duplicate rows so reciprocal
        # and identity patches retain real cross-row controls without padding.
        if len(cohort) == 1:
            qid = cohort[0]
            qi = qid_index[qid]
            prompts = [
                task_prompts[0][0], task_prompts[0][0],
                task_prompts[1][0], task_prompts[1][0],
            ]
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
            width = int(input_ids.shape[1])
            row_sources: list[list[int]] = []
            row_audit: list[dict[str, Any]] = []
            for row, task_index in enumerate((0, 0, 1, 1)):
                task = TASKS[task_index]
                prompt = prompts[row]
                token_ids = [
                    int(value)
                    for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]
                ]
                left_pad = width - len(token_ids)
                positions, suffix_audit = _feedback_suffix_positions(tokenizer, prompt, task)
                physical = [left_pad + position for position in positions]
                if len(physical) != len(first_positions[0]):
                    raise RuntimeError("Feedback suffix token count changed across questions")
                row_sources.append(physical)
                digest = _hash_prompt(prompt)
                trusted_row = trusted[task_index][qid]
                if digest != trusted_row["prompt_hash"]:
                    raise RuntimeError("Prompt hash differs from trusted behavior")
                arrays["prompt_hashes"][task_index, qi] = digest
                arrays["source_positions"][task_index, qi] = physical
                arrays["trusted_natural_logits"][task_index, qi] = np.asarray(
                    trusted_row["aggregated_ad_logits"], dtype=np.float32
                )
                row_audit.append(
                    {
                        "row": row,
                        "task": task,
                        "question_id": qid,
                        "prompt_hash": digest,
                        "suffix": suffix_audit,
                        "positions_physical": physical,
                    }
                )
            if row_sources[0] != row_sources[2]:
                raise RuntimeError("Game/Neutral source positions are not physically aligned")
            natural = _aggregate_final_logits(
                _forward_final_logits(model, parts, input_ids, attention_mask), variant_ids
            )
            reciprocal, reciprocal_stats = _patched_forward(
                model,
                parts,
                input_ids,
                attention_mask,
                variant_ids,
                layers,
                {0: 2, 1: 3, 2: 0, 3: 1},
                row_sources,
            )
            patch_stats.append({"scenario": "reciprocal", **reciprocal_stats})
            # Test identity in the exact same four-row batch used by natural
            # and reciprocal.  Comparing across a separate two-row batch is
            # invalid for Gemma because changing the batch composition can
            # change low-precision reductions even when prompt rows match.
            identity_output, identity_stats = _patched_forward(
                model,
                parts,
                input_ids,
                attention_mask,
                variant_ids,
                layers,
                {0: 1, 1: 0, 2: 3, 3: 2},
                row_sources,
            )
            patch_stats.append({"scenario": "identity", **identity_stats})
            identity = identity_output[[0, 2]]
            identity_error = np.asarray(
                [
                    np.max(np.abs(identity_output[0] - natural[0])),
                    np.max(np.abs(identity_output[2] - natural[2])),
                ],
                dtype=np.float32,
            )
            if float(np.max(identity_error)) != 0.0:
                raise RuntimeError(
                    "Distinct-row same-batch identity error is "
                    f"{float(np.max(identity_error))}"
                )
            for task_index, row in enumerate((0, 2)):
                trusted_logits = arrays["trusted_natural_logits"][task_index, qi]
                raw = np.stack((natural[row], identity[task_index], reciprocal[row]), axis=0)
                arrays["same_batch_natural_logits"][task_index, qi] = natural[row]
                arrays["scenario_final_logits_raw"][task_index, :, qi] = raw
                arrays["scenario_final_logits"][task_index, :, qi] = (
                    trusted_logits[None, :] + raw - natural[row][None, :]
                )
                arrays["scenario_final_logits"][task_index, 0, qi] = trusted_logits
                arrays["scenario_final_logits"][task_index, 1, qi] = trusted_logits
                arrays["identity_error_by_question"][task_index, qi] = identity_error[task_index]
            cohort_audit["pairs"].append({"question_ids": [qid], "rows": row_audit})
            arrays["completed"][indices] = True
            atomic_save_npz(result_path, **arrays)
            duration = time.monotonic() - cohort_started
            durations.append(duration)
            print(
                f"{MODEL_ID} {dataset} feedback suffix: "
                f"{int(arrays['completed'].sum())}/{len(qids)}; "
                f"cohort_seconds={duration:.3f}; "
                f"identity_error={float(np.nanmax(arrays['identity_error_by_question']))}",
                flush=True,
            )
            if not audit_path.exists():
                cohort_audit["attention_layers_one_based"] = list(ATTENTION_LAYERS_ONE_BASED)
                cohort_audit["scenario_ids"] = list(SCENARIOS)
                cohort_audit["initial_suffix_audits"] = first_audits
                _atomic_write_json(audit_path, cohort_audit)
            continue

        for pair_start in range(0, len(cohort), 2):
            pair = cohort[pair_start : pair_start + 2]
            prompts = (
                task_prompts[0][pair_start : pair_start + 2]
                + task_prompts[1][pair_start : pair_start + 2]
            )
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
            input_ids, attention_mask = _pad_to_width(
                tokenizer, input_ids, attention_mask, canonical_width
            )
            width = int(input_ids.shape[1])
            row_sources: list[list[int]] = []
            pair_audit: dict[str, Any] = {"question_ids": pair, "rows": []}

            for task_index, task in enumerate(TASKS):
                for local, qid in enumerate(pair):
                    row = 2 * task_index + local
                    prompt = prompts[row]
                    token_ids = [
                        int(value)
                        for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    ]
                    left_pad = width - len(token_ids)
                    if input_ids[row, left_pad:].tolist() != token_ids:
                        raise RuntimeError("Paired tokenization changed the prompt")
                    positions, suffix_audit = _feedback_suffix_positions(tokenizer, prompt, task)
                    physical = [left_pad + position for position in positions]
                    if len(physical) != len(first_positions[0]):
                        raise RuntimeError("Feedback suffix token count changed across questions")
                    row_sources.append(physical)
                    digest = _hash_prompt(prompt)
                    trusted_row = trusted[task_index][qid]
                    if digest != trusted_row["prompt_hash"]:
                        raise RuntimeError("Prompt hash differs from trusted Seed behavior")
                    qi = qid_index[qid]
                    arrays["prompt_hashes"][task_index, qi] = digest
                    arrays["source_positions"][task_index, qi] = physical
                    arrays["trusted_natural_logits"][task_index, qi] = np.asarray(
                        trusted_row["aggregated_ad_logits"], dtype=np.float32
                    )
                    pair_audit["rows"].append(
                        {
                            "row": row,
                            "task": task,
                            "question_id": qid,
                            "prompt_hash": digest,
                            "suffix": suffix_audit,
                            "positions_physical": physical,
                        }
                    )

            for local in range(2):
                if row_sources[local] != row_sources[local + 2]:
                    raise RuntimeError("Game/Neutral source positions are not physically aligned")
            natural = _aggregate_final_logits(
                _forward_final_logits(model, parts, input_ids, attention_mask),
                variant_ids,
            )
            if not np.all(np.isfinite(natural)):
                raise RuntimeError("Non-finite natural logits")
            reciprocal, reciprocal_stats = _patched_forward(
                model,
                parts,
                input_ids,
                attention_mask,
                variant_ids,
                layers,
                {row: row + 2 if row < 2 else row - 2 for row in range(4)},
                row_sources,
            )
            patch_stats.append({"scenario": "reciprocal", **reciprocal_stats})

            identity_by_task = np.empty((2, 2, 4), dtype=np.float32)
            identity_error_by_task = np.empty((2, 2), dtype=np.float32)
            for task_index in range(2):
                source_rows = [2 * task_index, 2 * task_index + 1]
                duplicate_prompts = [
                    prompts[source_rows[0]], prompts[source_rows[0]],
                    prompts[source_rows[1]], prompts[source_rows[1]],
                ]
                duplicate_ids, duplicate_mask, _ = tokenize_batch(tokenizer, duplicate_prompts)
                duplicate_ids, duplicate_mask = _pad_to_width(
                    tokenizer, duplicate_ids, duplicate_mask, canonical_width
                )
                duplicate_sources = [
                    row_sources[source_rows[0]], row_sources[source_rows[0]],
                    row_sources[source_rows[1]], row_sources[source_rows[1]],
                ]
                identity_output, identity_stats = _patched_forward(
                    model,
                    parts,
                    duplicate_ids,
                    duplicate_mask,
                    variant_ids,
                    layers,
                    {0: 1, 1: 0, 2: 3, 3: 2},
                    duplicate_sources,
                )
                patch_stats.append(
                    {"scenario": f"identity_{TASKS[task_index]}", **identity_stats}
                )
                selected_identity = identity_output[[0, 2]]
                selected_natural = natural[source_rows]
                errors = np.max(np.abs(selected_identity - selected_natural), axis=-1)
                if float(np.max(errors)) != 0.0:
                    raise RuntimeError(
                        f"Distinct-row {TASKS[task_index]} identity error is {float(np.max(errors))}"
                    )
                identity_by_task[task_index] = selected_identity
                identity_error_by_task[task_index] = errors

            for task_index in range(2):
                for local, qid in enumerate(pair):
                    row = 2 * task_index + local
                    qi = qid_index[qid]
                    trusted_logits = arrays["trusted_natural_logits"][task_index, qi]
                    raw = np.stack(
                        (natural[row], identity_by_task[task_index, local], reciprocal[row]),
                        axis=0,
                    )
                    arrays["same_batch_natural_logits"][task_index, qi] = natural[row]
                    arrays["scenario_final_logits_raw"][task_index, :, qi] = raw
                    arrays["scenario_final_logits"][task_index, :, qi] = (
                        trusted_logits[None, :] + raw - natural[row][None, :]
                    )
                    arrays["scenario_final_logits"][task_index, 0, qi] = trusted_logits
                    arrays["scenario_final_logits"][task_index, 1, qi] = trusted_logits
                    arrays["identity_error_by_question"][task_index, qi] = (
                        identity_error_by_task[task_index, local]
                    )
            cohort_audit["pairs"].append(pair_audit)

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"Seed {dataset} feedback suffix: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.3f}; "
            f"identity_error={float(np.nanmax(arrays['identity_error_by_question']))}",
            flush=True,
        )
        if not audit_path.exists():
            cohort_audit["attention_layers_one_based"] = list(ATTENTION_LAYERS_ONE_BASED)
            cohort_audit["scenario_ids"] = list(SCENARIOS)
            cohort_audit["initial_suffix_audits"] = first_audits
            _atomic_write_json(audit_path, cohort_audit)

    metadata = {
        "experiment": f"{MODEL_ID} {dataset} complete-feedback-suffix K/V crossover",
        "dataset": dataset,
        "questions": len(qids),
        "tasks": list(TASKS),
        "scenarios": list(SCENARIOS),
        "source_token_count": len(first_positions[0]),
        "source_text": FEEDBACK_SUFFIX,
        "complete_model_forwards_per_canonical_cohort": (
            3 if config.batch_size == 1 else 4 * ((config.batch_size + 1) // 2)
        ),
        "paired_subbatches_per_canonical_cohort": (
            1 if config.batch_size == 1 else (config.batch_size + 1) // 2
        ),
        "attention_layers_one_based": list(ATTENTION_LAYERS_ONE_BASED),
        "attention_architecture": {
            "type": "grouped-query causal self-attention",
            "query_heads": query_heads,
            "key_value_heads": key_value_heads,
            "gla_or_recurrent_layers": 0,
        },
        "source_scope": (
            "All contiguous tokens from incorrect/lost through the final period. "
            "Donor ordinary-attention K/V is exposed only to causally later queries "
            f"at all {len(ATTENTION_LAYERS_ONE_BASED)} layers; source-token residual outputs remain recipient-natural."
        ),
        "identity_control": (
            "The full patch path crosses between distinct duplicated rows with identical "
            "prompts and must be bit-exact to same-batch natural."
        ),
        "same_batch_correction": (
            "Crossover = trusted natural + reciprocal same-batch - natural same-batch; "
            "natural and bit-exact identity are stored exactly as trusted natural."
        ),
        "model_load_seconds": model_load_seconds,
        "elapsed_seconds_after_load": time.monotonic() - started,
        "cohort_seconds": durations,
        "identity_max_abs_error": float(np.nanmax(arrays["identity_error_by_question"])),
        "patch_stats": patch_stats,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    _atomic_write_json(args.output_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--trusted-behavior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
