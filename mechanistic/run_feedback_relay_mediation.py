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
    BatchedGLACachedRelayRestorer,
    BatchedGLACachedRelayInterceptor,
    BatchedGLARelayWriteCache,
    BatchedSDPACachedRelayRestorer,
    BatchedSDPACachedRelayInterceptor,
    BatchedSDPARelayWriteCache,
)
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_feedback_source_localization import SOURCE_TOKEN_INDICES, _hash


PRESPECIFIED_RELAY_REGIONS = (
    "second_answer_instruction",
    "second_question_stem",
    "second_option_lines",
    "second_choice_cue_and_query",
    "final_assistant_prefix",
    "other_post_feedback_structure",
    "all_post_feedback_relays",
)
RELAY_REGIONS = (
    "second_answer_instruction",
    "second_question_stem",
    "second_option_lines",
    "second_choice_cue_and_query",
    "final_assistant_prefix",
    "all_post_feedback_relays",
)
SCENARIOS = (
    "natural",
    "cache_restored_no_source_swap",
    "feedback_suffix_swapped",
    *(f"intercept_{name}" for name in RELAY_REGIONS),
)


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise RuntimeError("Existing relay checkpoint uses different questions")
        if arrays["scenario_ids"].astype(str).tolist() != list(SCENARIOS):
            raise RuntimeError("Existing relay checkpoint uses different scenarios")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "scenario_ids": np.asarray(SCENARIOS),
        "completed": np.zeros(n, dtype=bool),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "duplicate_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "scenario_final_logits_raw": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
        "scenario_final_logits": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
        "source_positions": np.full((2, n, 7), -1, dtype=np.int16),
        "region_token_counts": np.full((2, n, len(RELAY_REGIONS)), -1, dtype=np.int16),
    }


def _relay_regions(
    partition: list[list[int]],
    left_pad: int,
    source_positions: list[int],
    final_query: int,
) -> dict[str, list[int]]:
    def positions(name: str) -> list[int]:
        return [left_pad + value for value in partition[SOURCE_NAMES.index(name)]]

    assigned = {
        "second_answer_instruction": positions("second_answer_instruction"),
        "second_question_stem": positions("second_question_stem"),
        "second_option_lines": sorted(
            value
            for rank in range(1, 5)
            for value in positions(f"second_R{rank}_line")
        ),
        "second_choice_cue_and_query": positions("second_choice_cue_and_query"),
        "final_assistant_prefix": [
            value for value in positions("final_assistant_prefix") if value < final_query
        ],
    }
    lower = max(source_positions) + 1
    causal_tail = set(range(lower, final_query))
    used = set(value for rows in assigned.values() for value in rows)
    if not used.issubset(causal_tail):
        raise RuntimeError("A named relay lies outside the post-feedback causal tail")
    otherwise_unassigned = sorted(causal_tail - used)
    if otherwise_unassigned:
        raise RuntimeError(
            "The canonical prompt unexpectedly contains post-feedback tokens outside "
            "the five named relay regions; the frozen scenario inventory must be extended"
        )
    named_union = [
        value
        for name in RELAY_REGIONS[:-1]
        for value in assigned[name]
    ]
    if len(named_union) != len(set(named_union)):
        raise RuntimeError("Relay regions overlap")
    if set(named_union) != causal_tail:
        raise RuntimeError("Relay regions are not exhaustive")
    assigned["all_post_feedback_relays"] = sorted(causal_tail)
    if any(not assigned[name] for name in RELAY_REGIONS):
        raise RuntimeError("Every prespecified relay region must be nonempty")
    return assigned


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
        canonical_batches = [
            _build_batch(config, processor, tokenizer, questions, mappings, cohort, condition)
            for condition in CONDITIONS
        ]
        canonical_width = int(canonical_batches[0]["input_ids"].shape[1])
        cohort_audit: dict[str, Any] = {"question_ids": cohort, "pairs": []}

        # Match Stage A exactly: two questions per forward, ordered as
        # Game(q0), Game(q1), Neutral(q0), Neutral(q1).  A clean forward in
        # that same geometry records the natural relay K/V and GLA writes;
        # later interception forwards restore those cached writes without
        # adding duplicate rows or changing any question's numerical context.
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
                raise RuntimeError("Paired batch exceeds canonical cohort width")
            if input_ids.shape[1] < canonical_width:
                pad = canonical_width - int(input_ids.shape[1])
                input_ids = torch.nn.functional.pad(
                    input_ids, (pad, 0), value=int(tokenizer.pad_token_id)
                )
                attention_mask = torch.nn.functional.pad(attention_mask, (pad, 0), value=0)
            width = int(input_ids.shape[1])
            final_query = width - 1
            row_sources: list[list[int]] = []
            row_regions: list[dict[str, list[int]]] = []
            pair_audit: dict[str, Any] = {"question_ids": pair, "rows": []}
            for condition_index, condition in enumerate(CONDITIONS):
                for local, qid in enumerate(pair):
                    row = 2 * condition_index + local
                    second_question = {
                        **questions[qid],
                        "options": {
                            new: questions[qid]["options"][old]
                            for new, old in mappings[qid]["new_to_original"].items()
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
                        tokenizer, prompts[row], messages[row], questions[qid],
                        second_question, condition, rank_letters,
                        mappings[qid]["original_to_new"],
                    )
                    left_pad = width - int(position_audit["prompt_length"])
                    feedback_positions = [
                        left_pad + partition[SOURCE_NAMES.index(f"feedback_token_{index}")][0]
                        for index in SOURCE_TOKEN_INDICES
                    ]
                    if feedback_positions != list(
                        range(feedback_positions[0], feedback_positions[0] + 7)
                    ):
                        raise RuntimeError("Feedback source positions are not aligned and contiguous")
                    regions = _relay_regions(
                        partition, left_pad, feedback_positions, final_query
                    )
                    row_sources.append(feedback_positions)
                    row_regions.append(regions)
                    prompt_ids = tokenizer(prompts[row], add_special_tokens=False)["input_ids"]
                    if input_ids[row, left_pad:].tolist() != [int(value) for value in prompt_ids]:
                        raise RuntimeError("Paired tokenization changed the prompt")
                    digest = _hash(prompts[row])
                    if digest != trusted[condition_index][qid]["prompt_hash"]:
                        raise RuntimeError("Prompt hash differs from trusted natural run")
                    qi = qid_index[qid]
                    arrays["prompt_hashes"][condition_index, qi] = digest
                    arrays["trusted_natural_logits"][condition_index, qi] = np.asarray(
                        trusted[condition_index][qid]["aggregated_ad_logits"], dtype=np.float32
                    )
                    arrays["source_positions"][condition_index, qi] = feedback_positions
                    arrays["region_token_counts"][condition_index, qi] = [
                        len(regions[name]) for name in RELAY_REGIONS
                    ]
                    pair_audit["rows"].append({
                        "row": row,
                        "condition": condition,
                        "question_id": qid,
                        "prompt_hash": digest,
                        "left_pad": left_pad,
                        "final_query": final_query,
                        "feedback_positions": feedback_positions,
                        "feedback_tokens": [
                            tokenizer.decode([int(input_ids[row, value])])
                            for value in feedback_positions
                        ],
                        "relay_regions": regions,
                        "relay_tokens": {
                            name: [
                                tokenizer.decode([int(input_ids[row, value])]).replace("\n", "\\n")
                                for value in values
                            ]
                            for name, values in regions.items()
                        },
                    })
            for local in range(2):
                if row_sources[local] != row_sources[local + 2]:
                    raise RuntimeError("Game/Neutral source positions are not aligned")
                if row_regions[local] != row_regions[local + 2]:
                    raise RuntimeError("Game/Neutral relay regions are not aligned")

            donors = {row: row + 2 if row < 2 else row - 2 for row in range(4)}
            all_relays = {
                row: row_regions[row]["all_post_feedback_relays"] for row in range(4)
            }
            ordinary_cache = BatchedSDPARelayWriteCache(
                parts, all_relays, ordinary_layers
            )
            gla_cache = BatchedGLARelayWriteCache(parts, all_relays, gla_layers)
            try:
                natural = _aggregate_logits(
                    _forward(model, parts, input_ids, attention_mask), variant_ids
                )
            finally:
                gla_cache.close()
                ordinary_cache.close()
            if set(ordinary_cache.cache) != set(ordinary_layers):
                raise RuntimeError("Clean ordinary cache is incomplete")
            if set(gla_cache.cache) != set(gla_layers):
                raise RuntimeError("Clean GLA cache is incomplete")

            scenario_outputs: list[np.ndarray] = [natural]
            for scenario in SCENARIOS[1:]:
                ordinary = None
                gla = None
                try:
                    if scenario == "cache_restored_no_source_swap":
                        ordinary = BatchedSDPACachedRelayRestorer(
                            parts, all_relays, ordinary_layers, ordinary_cache.cache
                        )
                        gla = BatchedGLACachedRelayRestorer(
                            parts, all_relays, gla_layers, gla_cache.cache
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
                    elif scenario.startswith("intercept_"):
                        region = scenario.removeprefix("intercept_")
                        ordinary = BatchedSDPACachedRelayInterceptor(
                            parts,
                            {
                                row: (
                                    donors[row], row_sources[row],
                                    row_regions[row][region], ordinary_layers,
                                )
                                for row in range(4)
                            },
                            ordinary_cache.cache,
                        )
                        gla = BatchedGLACachedRelayInterceptor(
                            parts,
                            {
                                row: (
                                    donors[row], row_sources[row],
                                    row_regions[row][region], gla_layers,
                                )
                                for row in range(4)
                            },
                            gla_cache.cache,
                        )
                    scenario_output = _aggregate_logits(
                        _forward(model, parts, input_ids, attention_mask), variant_ids
                    )
                    if hasattr(ordinary, "assert_fired"):
                        ordinary.assert_fired()
                    if hasattr(gla, "assert_fired"):
                        gla.assert_fired()
                    scenario_outputs.append(scenario_output)
                finally:
                    if gla is not None:
                        gla.close()
                    if ordinary is not None:
                        ordinary.close()
            scenario_logits = np.stack(scenario_outputs, axis=0)
            for condition_index in range(2):
                for local, qid in enumerate(pair):
                    row = 2 * condition_index + local
                    qi = qid_index[qid]
                    trusted_logits = arrays["trusted_natural_logits"][condition_index, qi]
                    arrays["same_batch_natural_logits"][condition_index, qi] = natural[row]
                    cache_control_index = SCENARIOS.index(
                        "cache_restored_no_source_swap"
                    )
                    arrays["duplicate_natural_logits"][condition_index, qi] = (
                        scenario_logits[cache_control_index, row]
                    )
                    arrays["scenario_final_logits_raw"][condition_index, :, qi] = scenario_logits[:, row]
                    arrays["scenario_final_logits"][condition_index, :, qi] = (
                        trusted_logits[None, :] + scenario_logits[:, row] - natural[row][None, :]
                    )
            cohort_audit["pairs"].append(pair_audit)

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"feedback relay mediation: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.2f}", flush=True,
        )
        if not audit_path.exists():
            cohort_audit["ordinary_layers_one_based"] = [value + 1 for value in ordinary_layers]
            cohort_audit["gla_layers_one_based"] = [value + 1 for value in gla_layers]
            cohort_audit["scenarios"] = list(SCENARIOS)
            audit_path.write_text(json.dumps(cohort_audit, indent=2) + "\n")

    metadata = {
        "experiment": "path-specific cached interception of evaluation-feedback crossover at exhaustive downstream relay regions",
        "questions": len(qids),
        "conditions": list(CONDITIONS),
        "scenarios": list(SCENARIOS),
        "cache_identity_control": (
            "In an otherwise natural forward, restore every post-feedback relay's "
            "ordinary-attention K/V and GLA k/v/g/beta from the same row's clean "
            "cache without applying any feedback-source swap."
        ),
        "unintercepted_cross_position_channel": (
            "Qwen3.6 GLA blocks apply a short causal depthwise convolution to "
            "q/k/v before the delta-rule update intercepted here. This runner "
            "does not patch or restore those convolution states. Consequently, "
            "the outgoing-channel inventory is ordinary-attention K/V plus GLA "
            "recurrent k/v/g/beta, not every possible cross-position route."
        ),
        "relay_regions": list(RELAY_REGIONS),
        "prespecified_vacuous_region": (
            "other_post_feedback_structure contains zero tokens on every audited "
            "canonical prompt because the five named regions already exhaust the tail"
        ),
        "paired_subbatches_per_canonical_cohort": 2,
        "complete_model_forwards_per_canonical_cohort": 2 * len(SCENARIOS),
        "ordinary_layers_one_based": [value + 1 for value in ordinary_layers],
        "gla_layers_one_based": [value + 1 for value in gla_layers],
        "elapsed_seconds_after_load": time.monotonic() - started,
        "cohort_seconds": durations,
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "platform": platform.platform(),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
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
