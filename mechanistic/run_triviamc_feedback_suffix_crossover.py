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
from .run_feedback_source_localization import SOURCE_TOKEN_INDICES


SCENARIOS = ("natural", "identity_complete_suffix", "reciprocal_complete_suffix")


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
        "identity_error_by_question": np.full((2, n), np.nan, dtype=np.float32),
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
        raise ValueError("Requires token-matched incorrect/lost feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires canonical raw Qwen ChatML serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {str(row["id"]): row for row in manifest["questions"]}
    qids = [str(row["id"]) for row in manifest["questions"]]
    if args.max_cohorts is not None:
        qids = qids[: int(args.max_cohorts) * config.batch_size]
    if len(qids) % config.batch_size:
        raise RuntimeError("Questions must form complete four-question cohorts")
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
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    ]
    gla_layers = [
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if [value + 1 for value in ordinary_layers] != list(range(4, 65, 4)):
        raise RuntimeError("Unexpected ordinary-attention layer inventory")
    if [value + 1 for value in gla_layers] != [
        value for value in range(1, 65) if value % 4 != 0
    ]:
        raise RuntimeError("Unexpected GLA layer inventory")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.npz"
    arrays = _initialize(result_path, qids)
    completed_at_resume = arrays["completed"].astype(bool)
    if completed_at_resume.any():
        # Natural and the verified raw duplicated-row identity are controls, not
        # estimated interventions.  Store their corrected values exactly rather
        # than allowing float32 evaluation order in trusted + raw - raw to add a
        # one-ULP artifact on resume.
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
                    positions = [
                        left_pad
                        + partition[
                            SOURCE_NAMES.index(f"feedback_token_{index}")
                        ][0]
                        for index in SOURCE_TOKEN_INDICES
                    ]
                    if any(
                        len(
                            partition[
                                SOURCE_NAMES.index(f"feedback_token_{index}")
                            ]
                        )
                        != 1
                        for index in SOURCE_TOKEN_INDICES
                    ):
                        raise RuntimeError("Every feedback suffix token must be singular")
                    if positions != list(range(positions[0], positions[0] + 7)):
                        raise RuntimeError("Feedback suffix tokens are not contiguous")
                    row_sources.append(positions)
                    prompt_ids = [
                        int(value)
                        for value in tokenizer(
                            prompts[row], add_special_tokens=False
                        )["input_ids"]
                    ]
                    if input_ids[row, left_pad:].tolist() != prompt_ids:
                        raise RuntimeError("Paired tokenization changed the prompt")
                    digest = _hash(prompts[row])
                    if digest != trusted[condition_index][qid]["prompt_hash"]:
                        raise RuntimeError("Prompt hash differs from trusted Step-1 natural")
                    qi = qid_index[qid]
                    arrays["prompt_hashes"][condition_index, qi] = digest
                    arrays["source_positions"][condition_index, qi] = positions
                    arrays["trusted_natural_logits"][condition_index, qi] = np.asarray(
                        trusted[condition_index][qid]["aggregated_ad_logits"],
                        dtype=np.float32,
                    )
                    pair_audit["rows"].append(
                        {
                            "row": row,
                            "condition": condition,
                            "question_id": qid,
                            "prompt_hash": digest,
                            "feedback_positions": positions,
                            "feedback_tokens": [
                                tokenizer.decode([int(input_ids[row, position])])
                                for position in positions
                            ],
                        }
                    )

            for local in range(2):
                if row_sources[local] != row_sources[local + 2]:
                    raise RuntimeError("Game/Neutral feedback positions are not aligned")
            reciprocal_donors = {
                row: row + 2 if row < 2 else row - 2 for row in range(4)
            }
            scenario_outputs: dict[str, np.ndarray] = {}
            for scenario in ("natural", "reciprocal_complete_suffix"):
                ordinary = None
                gla = None
                try:
                    if scenario != "natural":
                        ordinary = BatchedSDPADownstreamSourceKVPatcher(
                            parts,
                            {
                                row: (
                                    reciprocal_donors[row],
                                    row_sources[row],
                                    ordinary_layers,
                                )
                                for row in range(4)
                            },
                        )
                        gla = BatchedSelectiveGDNSourceWritePatcher(
                            parts,
                            {
                                row: (
                                    reciprocal_donors[row],
                                    row_sources[row],
                                    gla_layers,
                                )
                                for row in range(4)
                            },
                            preserve_source_output=True,
                        )
                    output = _aggregate_logits(
                        _forward(model, parts, input_ids, attention_mask), variant_ids
                    )
                    if ordinary is not None:
                        ordinary.assert_fired()
                    if gla is not None:
                        gla.assert_fired()
                    if not np.all(np.isfinite(output)):
                        raise RuntimeError(f"Non-finite logits in {scenario}")
                    scenario_outputs[scenario] = output
                finally:
                    if gla is not None:
                        gla.close()
                    if ordinary is not None:
                        ordinary.close()

            natural = scenario_outputs["natural"]
            reciprocal = scenario_outputs["reciprocal_complete_suffix"]
            identity_by_condition = np.empty((2, 2, 4), dtype=np.float32)
            identity_error_by_condition = np.empty((2, 2), dtype=np.float32)
            for condition_index in range(2):
                source_rows = [2 * condition_index, 2 * condition_index + 1]
                duplicate_prompts = [
                    prompts[source_rows[0]],
                    prompts[source_rows[0]],
                    prompts[source_rows[1]],
                    prompts[source_rows[1]],
                ]
                duplicate_ids, duplicate_mask, _ = tokenize_batch(
                    tokenizer, duplicate_prompts
                )
                if duplicate_ids.shape[1] > canonical_width:
                    raise RuntimeError("Identity batch exceeds canonical cohort width")
                if duplicate_ids.shape[1] < canonical_width:
                    pad = canonical_width - int(duplicate_ids.shape[1])
                    duplicate_ids = torch.nn.functional.pad(
                        duplicate_ids, (pad, 0), value=int(tokenizer.pad_token_id)
                    )
                    duplicate_mask = torch.nn.functional.pad(
                        duplicate_mask, (pad, 0), value=0
                    )
                duplicate_sources = [
                    row_sources[source_rows[0]],
                    row_sources[source_rows[0]],
                    row_sources[source_rows[1]],
                    row_sources[source_rows[1]],
                ]
                identity_donors = {0: 1, 1: 0, 2: 3, 3: 2}
                ordinary = None
                gla = None
                try:
                    ordinary = BatchedSDPADownstreamSourceKVPatcher(
                        parts,
                        {
                            row: (
                                identity_donors[row],
                                duplicate_sources[row],
                                ordinary_layers,
                            )
                            for row in range(4)
                        },
                    )
                    gla = BatchedSelectiveGDNSourceWritePatcher(
                        parts,
                        {
                            row: (
                                identity_donors[row],
                                duplicate_sources[row],
                                gla_layers,
                            )
                            for row in range(4)
                        },
                        preserve_source_output=True,
                    )
                    identity_output = _aggregate_logits(
                        _forward(
                            model, parts, duplicate_ids, duplicate_mask
                        ),
                        variant_ids,
                    )
                    ordinary.assert_fired()
                    gla.assert_fired()
                finally:
                    if gla is not None:
                        gla.close()
                    if ordinary is not None:
                        ordinary.close()
                selected_identity = identity_output[[0, 2]]
                selected_natural = natural[source_rows]
                identity_error = np.max(
                    np.abs(selected_identity - selected_natural), axis=-1
                )
                if np.max(identity_error) != 0.0:
                    raise RuntimeError(
                        "Duplicated-row same-task suffix identity error is "
                        f"{float(np.max(identity_error))}"
                    )
                identity_by_condition[condition_index] = selected_identity
                identity_error_by_condition[condition_index] = identity_error

            for condition_index in range(2):
                for local, qid in enumerate(pair):
                    row = 2 * condition_index + local
                    qi = qid_index[qid]
                    trusted_logits = arrays["trusted_natural_logits"][condition_index, qi]
                    scenario_logits = np.stack(
                        (
                            natural[row],
                            identity_by_condition[condition_index, local],
                            reciprocal[row],
                        ),
                        axis=0,
                    )
                    arrays["same_batch_natural_logits"][condition_index, qi] = natural[row]
                    arrays["scenario_final_logits_raw"][condition_index, :, qi] = scenario_logits
                    arrays["scenario_final_logits"][condition_index, :, qi] = (
                        trusted_logits[None, :]
                        + scenario_logits
                        - natural[row][None, :]
                    )
                    arrays["scenario_final_logits"][condition_index, 0, qi] = (
                        trusted_logits
                    )
                    arrays["scenario_final_logits"][condition_index, 1, qi] = (
                        trusted_logits
                    )
                    arrays["identity_error_by_question"][condition_index, qi] = (
                        identity_error_by_condition[condition_index, local]
                    )
            cohort_audit["pairs"].append(pair_audit)

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"TriviaMC feedback suffix crossover: "
            f"{int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.2f}; "
            f"identity_error={float(np.nanmax(arrays['identity_error_by_question']))}",
            flush=True,
        )
        if not audit_path.exists():
            cohort_audit["ordinary_layers_one_based"] = [
                value + 1 for value in ordinary_layers
            ]
            cohort_audit["gla_layers_one_based"] = [
                value + 1 for value in gla_layers
            ]
            cohort_audit["scenario_ids"] = list(SCENARIOS)
            audit_path.write_text(json.dumps(cohort_audit, indent=2) + "\n")

    metadata = {
        "experiment": "TriviaMC complete-feedback-suffix policy crossover",
        "questions": len(qids),
        "conditions": list(CONDITIONS),
        "display_conditions": ["Game", "Neutral"],
        "scenarios": list(SCENARIOS),
        "source_token_indices": list(SOURCE_TOKEN_INDICES),
        "complete_model_forwards_per_canonical_cohort": 8,
        "paired_subbatches_per_canonical_cohort": 2,
        "complete_model_forwards_per_paired_subbatch": (
            "two paired-task forwards plus two duplicated-row identity forwards"
        ),
        "ordinary_layers_one_based": [value + 1 for value in ordinary_layers],
        "gla_layers_one_based": [value + 1 for value in gla_layers],
        "source_scope": (
            "Seven contiguous tokens from incorrect/lost through the final period; "
            "downstream ordinary-attention K/V and recurrent GLA k/v/g/beta writes "
            "at every applicable layer; source-token local outputs preserved."
        ),
        "identity_control": (
            "The complete patcher path crosses between distinct duplicated rows "
            "with identical prompts and must be bit-exact to same-batch natural."
        ),
        "same_batch_correction": (
            "Crossover: trusted natural + scenario same-batch - natural same-batch. "
            "Natural and verified raw identity are stored exactly as trusted natural."
        ),
        "elapsed_seconds_after_load": time.monotonic() - started,
        "cohort_seconds": durations,
        "identity_max_abs_error": float(
            np.nanmax(arrays["identity_error_by_question"])
        ),
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
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
