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
from .analyze_cue_attention_distribution import SOURCE_NAMES, _cue_source_partition
from .config import ExperimentConfig
from .downstream_source_intervention import (
    BatchedSDPAFinalQueryAttentionAblator,
    BatchedSDPAFinalQuerySourceKVPatcher,
)
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    resolve_answer_tokens,
    tokenize_batch,
)
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward


SCENARIOS = (
    "natural",
    "layer40_2p_options_swapped",
    "layer40_2p_options_ablated",
    "layers52_56_scaffold_swapped",
    "layers52_56_scaffold_ablated",
    "all_layers_1p_options_ablated",
)
ORDINARY_LAYERS = tuple(range(4, 65, 4))
LAYER40_SOURCES = tuple(f"second_R{rank}_line" for rank in range(1, 5))
LAYER52_SOURCES = (
    "second_answer_instruction",
    "second_question_stem",
    "second_choice_cue_and_query",
)
LAYER56_SOURCES = (
    "second_question_stem",
    "second_choice_cue_and_query",
)
ALL_1P_OPTION_SOURCES = tuple(f"first_R{rank}_line" for rank in range(1, 5))


def _hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _union(partition: list[list[int]], names: tuple[str, ...]) -> list[int]:
    values = sorted(
        position
        for name in names
        for position in partition[SOURCE_NAMES.index(name)]
    )
    if not values or len(values) != len(set(values)):
        raise RuntimeError(f"Invalid source union for {names}")
    return values


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise RuntimeError("Existing checkpoint uses different questions")
        if arrays["scenario_ids"].astype(str).tolist() != list(SCENARIOS):
            raise RuntimeError("Existing checkpoint uses different scenarios")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "scenario_ids": np.asarray(SCENARIOS),
        "completed": np.zeros(n, dtype=bool),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "scenario_final_logits_raw": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
        "scenario_final_logits": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
        "layer40_source_count": np.zeros((2, n), dtype=np.int16),
        "layer52_source_count": np.zeros((2, n), dtype=np.int16),
        "layer56_source_count": np.zeros((2, n), dtype=np.int16),
        "all_1p_source_count": np.zeros((2, n), dtype=np.int16),
    }


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(args.config)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires canonical batch-size-4 SDPA execution")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires canonical empty-history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires action-matched incorrect/lost feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires canonical raw Qwen ChatML serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {str(row["id"]): row for row in manifest["questions"]}
    all_qids = [str(row["id"]) for row in manifest["questions"]]
    qids = (
        all_qids[: int(args.max_cohorts) * config.batch_size]
        if args.max_cohorts is not None
        else all_qids
    )
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
    ordinary_indices = [
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    ]
    if tuple(index + 1 for index in ordinary_indices) != ORDINARY_LAYERS:
        raise RuntimeError(f"Ordinary-attention inventory changed: {ordinary_indices}")

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
        if len(cohort) != config.batch_size:
            raise RuntimeError("Canonical questions must form complete cohorts")
        cohort_started = time.monotonic()
        condition_batches = [
            _build_batch(
                config, processor, tokenizer, questions, mappings, cohort, condition
            )
            for condition in CONDITIONS
        ]
        canonical_width = int(condition_batches[0]["input_ids"].shape[1])
        cohort_audit: dict[str, Any] = {"question_ids": cohort, "pairs": []}

        for pair_start in range(0, len(cohort), 2):
            pair = cohort[pair_start : pair_start + 2]
            prompts = (
                condition_batches[0]["prompts"][pair_start : pair_start + 2]
                + condition_batches[1]["prompts"][pair_start : pair_start + 2]
            )
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
            if input_ids.shape[1] > canonical_width:
                raise RuntimeError("Paired batch exceeds canonical cohort width")
            if input_ids.shape[1] < canonical_width:
                pad = canonical_width - int(input_ids.shape[1])
                input_ids = torch.nn.functional.pad(
                    input_ids, (pad, 0), value=int(tokenizer.pad_token_id)
                )
                attention_mask = torch.nn.functional.pad(
                    attention_mask, (pad, 0), value=0
                )
            width = int(input_ids.shape[1])
            partitions: list[list[list[int]]] = []
            pair_audit: dict[str, Any] = {"question_ids": pair, "rows": []}

            for condition_index, condition in enumerate(CONDITIONS):
                for local, qid in enumerate(pair):
                    row = 2 * condition_index + local
                    mapping = mappings[qid]
                    second_question = {
                        **questions[qid],
                        "options": {
                            new: questions[qid]["options"][old]
                            for new, old in mapping["new_to_original"].items()
                        },
                    }
                    old_logits = np.asarray(
                        baseline[qid]["aggregated_ad_logits"], dtype=np.float64
                    )
                    rank_letters = [
                        LETTERS[int(value)]
                        for value in np.argsort(-old_logits, kind="stable")
                    ]
                    unpadded, position_audit = _cue_source_partition(
                        tokenizer,
                        prompts[row],
                        condition_batches[condition_index]["messages"][pair_start + local],
                        questions[qid],
                        second_question,
                        condition,
                        rank_letters,
                        mapping["original_to_new"],
                    )
                    left_pad = width - int(position_audit["prompt_length"])
                    partition = [
                        [left_pad + value for value in source] for source in unpadded
                    ]
                    partitions.append(partition)
                    prompt_ids = [
                        int(value)
                        for value in tokenizer(
                            prompts[row], add_special_tokens=False
                        )["input_ids"]
                    ]
                    if input_ids[row, left_pad:].tolist() != prompt_ids:
                        raise RuntimeError("Paired tokenization changed the prompt")
                    prompt_digest = _hash(prompts[row])
                    if prompt_digest != trusted[condition_index][qid]["prompt_hash"]:
                        raise RuntimeError("Prompt hash differs from trusted natural run")
                    qi = qid_index[qid]
                    arrays["prompt_hashes"][condition_index, qi] = prompt_digest
                    arrays["trusted_natural_logits"][condition_index, qi] = np.asarray(
                        trusted[condition_index][qid]["aggregated_ad_logits"],
                        dtype=np.float32,
                    )
                    counts = {
                        "layer40": len(_union(partition, LAYER40_SOURCES)),
                        "layer52": len(_union(partition, LAYER52_SOURCES)),
                        "layer56": len(_union(partition, LAYER56_SOURCES)),
                        "all_1p": len(_union(partition, ALL_1P_OPTION_SOURCES)),
                    }
                    arrays["layer40_source_count"][condition_index, qi] = counts["layer40"]
                    arrays["layer52_source_count"][condition_index, qi] = counts["layer52"]
                    arrays["layer56_source_count"][condition_index, qi] = counts["layer56"]
                    arrays["all_1p_source_count"][condition_index, qi] = counts["all_1p"]
                    pair_audit["rows"].append(
                        {
                            "row": row,
                            "condition": condition,
                            "question_id": qid,
                            "prompt_hash": prompt_digest,
                            "left_padding": left_pad,
                            "final_query_position": width - 1,
                            "final_query_token": tokenizer.decode([int(input_ids[row, -1])]),
                            "source_counts": counts,
                        }
                    )

            for local in range(2):
                if any(
                    partitions[local][SOURCE_NAMES.index(name)]
                    != partitions[local + 2][SOURCE_NAMES.index(name)]
                    for name in (
                        *LAYER40_SOURCES,
                        *LAYER52_SOURCES,
                        *LAYER56_SOURCES,
                        *ALL_1P_OPTION_SOURCES,
                    )
                ):
                    raise RuntimeError("Game/Neutral source positions are not aligned")

            donors = {row: row + 2 if row < 2 else row - 2 for row in range(4)}
            layer40_positions = {
                row: _union(partitions[row], LAYER40_SOURCES) for row in range(4)
            }
            layer52_positions = {
                row: _union(partitions[row], LAYER52_SOURCES) for row in range(4)
            }
            layer56_positions = {
                row: _union(partitions[row], LAYER56_SOURCES) for row in range(4)
            }
            first_positions = {
                row: _union(partitions[row], ALL_1P_OPTION_SOURCES) for row in range(4)
            }
            scenario_outputs: list[np.ndarray] = []
            for scenario in SCENARIOS:
                intervention = None
                try:
                    if scenario == "layer40_2p_options_swapped":
                        intervention = BatchedSDPAFinalQuerySourceKVPatcher(
                            parts,
                            {
                                39: {
                                    row: (donors[row], layer40_positions[row])
                                    for row in range(4)
                                }
                            },
                        )
                    elif scenario == "layer40_2p_options_ablated":
                        intervention = BatchedSDPAFinalQueryAttentionAblator(
                            parts, {39: layer40_positions}
                        )
                    elif scenario == "layers52_56_scaffold_swapped":
                        intervention = BatchedSDPAFinalQuerySourceKVPatcher(
                            parts,
                            {
                                51: {
                                    row: (donors[row], layer52_positions[row])
                                    for row in range(4)
                                },
                                55: {
                                    row: (donors[row], layer56_positions[row])
                                    for row in range(4)
                                },
                            },
                        )
                    elif scenario == "layers52_56_scaffold_ablated":
                        intervention = BatchedSDPAFinalQueryAttentionAblator(
                            parts,
                            {51: layer52_positions, 55: layer56_positions},
                        )
                    elif scenario == "all_layers_1p_options_ablated":
                        intervention = BatchedSDPAFinalQueryAttentionAblator(
                            parts,
                            {layer: first_positions for layer in ordinary_indices},
                        )
                    output = _aggregate_logits(
                        _forward(model, parts, input_ids, attention_mask), variant_ids
                    )
                    scenario_outputs.append(output)
                finally:
                    if intervention is not None:
                        intervention.close()

            scenario_logits = np.stack(scenario_outputs, axis=0)
            natural = scenario_logits[0]
            for condition_index in range(2):
                for local, qid in enumerate(pair):
                    row = 2 * condition_index + local
                    qi = qid_index[qid]
                    trusted_logits = arrays["trusted_natural_logits"][condition_index, qi]
                    arrays["same_batch_natural_logits"][condition_index, qi] = natural[row]
                    arrays["scenario_final_logits_raw"][condition_index, :, qi] = (
                        scenario_logits[:, row]
                    )
                    arrays["scenario_final_logits"][condition_index, :, qi] = (
                        trusted_logits[None, :]
                        + scenario_logits[:, row]
                        - natural[row][None, :]
                    )
            cohort_audit["pairs"].append(pair_audit)

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"final-position source causality: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.2f}",
            flush=True,
        )
        if not audit_path.exists():
            cohort_audit["scenario_definitions"] = {
                "layer40_2p_options": list(LAYER40_SOURCES),
                "layer52_scaffold": list(LAYER52_SOURCES),
                "layer56_scaffold": list(LAYER56_SOURCES),
                "all_1p_options": list(ALL_1P_OPTION_SOURCES),
            }
            audit_path.write_text(json.dumps(cohort_audit, indent=2) + "\n")

    metadata = {
        "experiment": "final-position causal source test frozen from cached discovery map",
        "questions": len(qids),
        "conditions": list(CONDITIONS),
        "scenarios": list(SCENARIOS),
        "complete_model_forwards_per_canonical_cohort": 12,
        "paired_subbatches_per_canonical_cohort": 2,
        "ordinary_layers_one_based": list(ORDINARY_LAYERS),
        "same_batch_correction": (
            "trusted natural logits + intervention same-batch logits - natural same-batch logits"
        ),
        "selection": (
            "Frozen on discovery: distributed 2P option sources at L40 and contextualized "
            "2P scaffold sources at L52/L56. The full-range direct 1P-option ablation "
            "completes the previously truncated control through L64."
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
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
