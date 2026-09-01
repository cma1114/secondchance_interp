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
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward


SOURCE_TOKEN_INDICES = tuple(range(3, 10))
SCENARIOS = (
    "natural",
    *(f"feedback_token_{index}_swapped" for index in SOURCE_TOKEN_INDICES),
    "feedback_suffix_3_9_swapped",
)


def _hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


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
        "source_positions": np.full((2, n, 7), -1, dtype=np.int16),
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
    if args.split_plan is None:
        qids = all_qids
    else:
        selected = set(json.loads(args.split_plan.read_text())["question_ids"])
        qids = [qid for qid in all_qids if qid in selected]
    if args.max_cohorts is not None:
        qids = qids[: int(args.max_cohorts) * config.batch_size]
    if len(qids) % config.batch_size:
        raise RuntimeError("Selected questions must form complete canonical cohorts")

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
    if [value + 1 for value in ordinary_layers] != list(range(4, 65, 4)):
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
            row_sources: list[list[int]] = []
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
                    partition, position_audit = _cue_source_partition(
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
                    feedback_positions = [
                        left_pad
                        + partition[SOURCE_NAMES.index(f"feedback_token_{index}")][0]
                        for index in SOURCE_TOKEN_INDICES
                    ]
                    if any(
                        len(partition[SOURCE_NAMES.index(f"feedback_token_{index}")]) != 1
                        for index in SOURCE_TOKEN_INDICES
                    ):
                        raise RuntimeError("Each policy-bearing feedback token must be singular")
                    if feedback_positions != list(
                        range(feedback_positions[0], feedback_positions[0] + 7)
                    ):
                        raise RuntimeError("Policy-bearing feedback tokens are not contiguous")
                    row_sources.append(feedback_positions)
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
                    arrays["source_positions"][condition_index, qi] = feedback_positions
                    arrays["trusted_natural_logits"][condition_index, qi] = np.asarray(
                        trusted[condition_index][qid]["aggregated_ad_logits"],
                        dtype=np.float32,
                    )
                    pair_audit["rows"].append({
                        "row": row,
                        "condition": condition,
                        "question_id": qid,
                        "prompt_hash": prompt_digest,
                        "feedback_positions": feedback_positions,
                        "feedback_tokens": [
                            tokenizer.decode([int(input_ids[row, position])])
                            for position in feedback_positions
                        ],
                    })

            for local in range(2):
                if row_sources[local] != row_sources[local + 2]:
                    raise RuntimeError("Game/Neutral feedback positions are not aligned")
            donors = {row: row + 2 if row < 2 else row - 2 for row in range(4)}
            scenario_outputs: list[np.ndarray] = []
            for scenario in SCENARIOS:
                ordinary = None
                gla = None
                try:
                    if scenario != "natural":
                        if scenario == "feedback_suffix_3_9_swapped":
                            selected_by_row = {
                                row: list(row_sources[row]) for row in range(4)
                            }
                        else:
                            token_index = int(scenario.split("_")[2])
                            offset = SOURCE_TOKEN_INDICES.index(token_index)
                            selected_by_row = {
                                row: [row_sources[row][offset]] for row in range(4)
                            }
                        ordinary = BatchedSDPADownstreamSourceKVPatcher(
                            parts,
                            {
                                row: (
                                    donors[row], selected_by_row[row], ordinary_layers
                                )
                                for row in range(4)
                            },
                        )
                        gla = BatchedSelectiveGDNSourceWritePatcher(
                            parts,
                            {
                                row: (donors[row], selected_by_row[row], gla_layers)
                                for row in range(4)
                            },
                            preserve_source_output=True,
                        )
                    scenario_outputs.append(
                        _aggregate_logits(
                            _forward(model, parts, input_ids, attention_mask),
                            variant_ids,
                        )
                    )
                    ordinary.assert_fired()
                    gla.assert_fired()
                finally:
                    if gla is not None:
                        gla.close()
                    if ordinary is not None:
                        ordinary.close()

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
            f"feedback source localization: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.2f}",
            flush=True,
        )
        if not audit_path.exists():
            cohort_audit["ordinary_layers_one_based"] = [
                value + 1 for value in ordinary_layers
            ]
            cohort_audit["gla_layers_one_based"] = [value + 1 for value in gla_layers]
            cohort_audit["scenario_ids"] = list(SCENARIOS)
            audit_path.write_text(json.dumps(cohort_audit, indent=2) + "\n")

    metadata = {
        "experiment": "causal localization of policy-bearing feedback source tokens",
        "questions": len(qids),
        "conditions": list(CONDITIONS),
        "scenarios": list(SCENARIOS),
        "complete_model_forwards_per_canonical_cohort": 2 * len(SCENARIOS),
        "paired_subbatches_per_canonical_cohort": 2,
        "ordinary_layers_one_based": [value + 1 for value in ordinary_layers],
        "gla_layers_one_based": [value + 1 for value in gla_layers],
        "source_scope": (
            "Each feedback token 3-9 separately and jointly; reciprocal downstream "
            "ordinary-attention K/V and complete GLA memory-write crossover at every "
            "applicable layer while preserving source-token outputs."
        ),
        "same_batch_correction": (
            "trusted natural logits + intervention same-batch logits - natural same-batch logits"
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
    print(json.dumps(metadata, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
