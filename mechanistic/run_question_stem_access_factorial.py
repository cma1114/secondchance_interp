from __future__ import annotations

import argparse
import contextlib
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
from .downstream_source_intervention import BatchedSDPAQuerySourceAttentionAblator
from .io import atomic_save_npz
from .modeling import get_tokenizer, load_model_and_processor, resolve_answer_tokens
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_fresh_history_double_dissociation import (
    CONTENT_SUMMARY,
    FRESH_TARGET,
    GROUPS,
    NEWLINE_SUMMARY,
    OLD_TARGET,
    FreshOptionLineScrubber,
    _direction_geometry,
)
from .run_fixed_a_final_query_edge_ablation import _option_line_positions


SCENARIOS = (
    "trusted_natural",
    "identity_monitor",
    "block_first_stem",
    "block_second_stem",
    "block_both_stems",
)
TRUSTED_NATURAL, IDENTITY, FIRST, SECOND, BOTH = range(len(SCENARIOS))
ORDINARY_LAYERS = tuple(range(3, 64, 4))


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _queries_with_sources(
    queries: list[int], sources: list[int]
) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for query in sorted(set(int(value) for value in queries)):
        causal = [int(source) for source in sources if int(source) < query]
        if causal:
            result[query] = causal
    if not result:
        raise RuntimeError("No causal query-to-source edges were constructed")
    return result


def _merge_query_specs(
    *specs: dict[int, list[int]],
) -> dict[int, list[int]]:
    queries: dict[int, set[int]] = {}
    for spec in specs:
        for query, sources in spec.items():
            queries.setdefault(int(query), set()).update(int(value) for value in sources)
    result = {query: sorted(sources) for query, sources in sorted(queries.items())}
    if any(source >= query for query, sources in result.items() for source in sources):
        raise RuntimeError("Merged stem blockade contains a noncausal edge")
    return result


def _layer_specs(
    row_specs: list[dict[int, list[int]]],
) -> dict[int, dict[int, dict[int, list[int]]]]:
    rows = {row: spec for row, spec in enumerate(row_specs)}
    if any(not spec for spec in rows.values()):
        raise RuntimeError("Every batch row needs stem-access edges")
    return {layer: rows for layer in ORDINARY_LAYERS}


def _initialize(path: Path, qids: list[str], split: np.ndarray) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses another question order")
        if arrays["scenario_ids"].astype(str).tolist() != list(SCENARIOS):
            raise ValueError("Existing checkpoint uses another scenario inventory")
        return arrays
    n = len(qids)
    coordinate_shape = (2, len(SCENARIOS), n, 64, len(GROUPS), 4)
    return {
        "question_ids": np.asarray(qids),
        "split": split,
        "scenario_ids": np.asarray(SCENARIOS),
        "completed": np.zeros(n, dtype=bool),
        "baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "rank_letters": np.full((n, 4), "", dtype="<U1"),
        "logits": np.full((2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32),
        "fresh_coordinates": np.full(coordinate_shape, np.nan, dtype=np.float32),
        "old_coordinates": np.full(coordinate_shape, np.nan, dtype=np.float32),
        "trusted_max_abs_error": np.full((2, n), np.nan, dtype=np.float32),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "first_source_count": np.zeros((2, n), dtype=np.int16),
        "second_source_count": np.zeros((2, n), dtype=np.int16),
        "first_query_count": np.zeros((2, n), dtype=np.int16),
        "second_query_count": np.zeros((2, n), dtype=np.int16),
    }


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(args.config)
    if (
        config.prompt_mode != "baseline_matched_empty_history"
        or config.feedback_variant != "token_matched_test"
        or config.chat_serialization != "raw_qwen_chatml"
        or config.attn_implementation != "sdpa"
        or int(config.batch_size) != 4
    ):
        raise ValueError("Requires the exact canonical empty-history batch-four SDPA regime")

    manifest = json.loads(Path(config.manifest_path).read_text())["questions"]
    qids = [str(row["id"]) for row in manifest]
    if len(qids) != 500:
        raise ValueError(f"Expected 500 questions, got {len(qids)}")
    questions = {str(row["id"]): row for row in manifest}
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    if len(discovery_ids) != 251 or len(set(qids) - discovery_ids) != 249:
        raise ValueError("Expected frozen 251/249 split")
    split = np.asarray(
        ["discovery" if qid in discovery_ids else "confirmation" for qid in qids]
    )
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    if set(mappings) != set(qids):
        raise ValueError("Remapping plan does not cover the frozen manifest exactly")
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
    loaded = torch.load(args.score_directions, map_location="cpu", weights_only=True)
    if hasattr(loaded, "numpy"):
        direction_array = loaded.float().numpy()
    elif isinstance(loaded, dict) and "directions" in loaded:
        direction_array = loaded["directions"].float().numpy()
    else:
        raise ValueError("Unrecognized score-direction artifact")
    unique_fresh, random_directions, geometry = _direction_geometry(direction_array)
    old_directions = np.stack(
        [
            direction_array[:, CONTENT_SUMMARY, OLD_TARGET],
            direction_array[:, NEWLINE_SUMMARY, OLD_TARGET],
        ],
        axis=1,
    ).astype(np.float32)
    old_directions /= np.maximum(
        np.linalg.norm(old_directions, axis=-1, keepdims=True), 1e-12
    )
    inventory = tuple(
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if inventory != ORDINARY_LAYERS:
        raise RuntimeError(f"Unexpected ordinary-attention inventory: {inventory}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.npz"
    arrays = _initialize(result_path, qids, split)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = args.output_dir / "prompt_audit.json"
    started = time.monotonic()
    durations: list[float] = []
    completed_cohorts = 0

    for start in range(0, len(qids), 4):
        cohort = qids[start : start + 4]
        indices = [qid_index[qid] for qid in cohort]
        if arrays["completed"][indices].all():
            continue
        cohort_started = time.monotonic()
        for qid, qi in zip(cohort, indices):
            old_logits = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=np.float32)
            arrays["baseline_logits"][qi] = old_logits
            arrays["rank_letters"][qi] = np.asarray(
                [LETTERS[int(index)] for index in np.argsort(-old_logits, kind="stable")]
            )

        for condition_index, condition in enumerate(CONDITIONS):
            batch = _build_batch(
                config, processor, tokenizer, questions, mappings, cohort, condition
            )
            width = int(batch["input_ids"].shape[1])
            first_specs: list[dict[int, list[int]]] = []
            second_specs: list[dict[int, list[int]]] = []
            both_specs: list[dict[int, list[int]]] = []
            content_positions: list[list[list[int]]] = []
            newline_positions: list[list[list[int]]] = []
            row_audits: list[dict[str, Any]] = []

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
                partition, partition_audit = _cue_source_partition(
                    tokenizer,
                    batch["prompts"][row],
                    batch["messages"][row],
                    questions[qid],
                    remapped,
                    condition,
                    arrays["rank_letters"][qi].astype(str).tolist(),
                    mappings[qid]["original_to_new"],
                )
                physical = [
                    [left_pad + int(value) for value in positions]
                    for positions in partition
                ]
                first_stem = physical[SOURCE_NAMES.index("first_question_stem")]
                second_stem = physical[SOURCE_NAMES.index("second_question_stem")]
                feedback_start = min(
                    min(physical[SOURCE_NAMES.index(f"feedback_token_{index}")])
                    for index in range(10)
                )
                first_queries = list(range(feedback_start, width))
                second_queries = sorted(
                    value
                    for name in (
                        "second_R1_line",
                        "second_R2_line",
                        "second_R3_line",
                        "second_R4_line",
                        "second_choice_cue_and_query",
                        "final_assistant_prefix",
                    )
                    for value in physical[SOURCE_NAMES.index(name)]
                )
                first_spec = _queries_with_sources(first_queries, first_stem)
                second_spec = _queries_with_sources(second_queries, second_stem)
                first_specs.append(first_spec)
                second_specs.append(second_spec)
                both_specs.append(_merge_query_specs(first_spec, second_spec))

                second_lines, second_audit = _option_line_positions(
                    tokenizer, batch["prompts"][row], remapped
                )
                contents: list[list[int]] = []
                newlines: list[list[int]] = []
                for original in arrays["rank_letters"][qi].astype(str).tolist():
                    displayed = mappings[qid]["original_to_new"][original]
                    positions = [left_pad + int(value) for value in second_lines[displayed]]
                    if len(positions) < 5 or int(batch["input_ids"][row, positions[-1]]) != 198:
                        raise RuntimeError(f"{qid}: malformed repeated option line")
                    contents.append(positions[3:-1])
                    newlines.append([positions[-1]])
                content_positions.append(contents)
                newline_positions.append(newlines)
                arrays["first_source_count"][condition_index, qi] = len(first_stem)
                arrays["second_source_count"][condition_index, qi] = len(second_stem)
                arrays["first_query_count"][condition_index, qi] = len(first_spec)
                arrays["second_query_count"][condition_index, qi] = len(second_spec)
                arrays["prompt_hashes"][condition_index, qi] = _hash_prompt(
                    batch["prompts"][row]
                )
                if arrays["prompt_hashes"][condition_index, qi] != trusted[condition_index][qid]["prompt_hash"]:
                    raise RuntimeError(f"{qid}: {condition} trusted prompt hash mismatch")
                row_audits.append(
                    {
                        "question_id": qid,
                        "left_pad": left_pad,
                        "partition": partition_audit,
                        "first_stem_positions_padded": first_stem,
                        "second_stem_positions_padded": second_stem,
                        "first_query_count": len(first_spec),
                        "second_query_count": len(second_spec),
                        "second_option_lines": second_audit,
                    }
                )

            scenario_specs = {
                "block_first_stem": _layer_specs(first_specs),
                "block_second_stem": _layer_specs(second_specs),
                "block_both_stems": _layer_specs(both_specs),
            }
            for scenario_index, scenario in enumerate(SCENARIOS):
                monitor = None
                ablator = None
                with contextlib.ExitStack() as stack:
                    if scenario != "trusted_natural":
                        monitor = stack.enter_context(
                            FreshOptionLineScrubber(
                                parts,
                                content_positions,
                                newline_positions,
                                old_directions,
                                unique_fresh,
                                random_directions,
                                mode="identity",
                            )
                        )
                    if scenario in scenario_specs:
                        ablator = stack.enter_context(
                            BatchedSDPAQuerySourceAttentionAblator(
                                parts, scenario_specs[scenario]
                            )
                        )
                    output = _aggregate_logits(
                        _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                        variant_ids,
                    )
                    if ablator is not None:
                        ablator.assert_fired()
                if not np.all(np.isfinite(output)):
                    raise RuntimeError(f"Non-finite logits in {condition}/{scenario}")
                arrays["logits"][condition_index, scenario_index, indices] = output
                if monitor is not None:
                    local = monitor.arrays()
                    for row, qi in enumerate(indices):
                        arrays["fresh_coordinates"][condition_index, scenario_index, qi] = local[
                            "pre_fresh"
                        ][:, row]
                        arrays["old_coordinates"][condition_index, scenario_index, qi] = local[
                            "pre_old"
                        ][:, row]

            for row, (qid, qi) in enumerate(zip(cohort, indices)):
                reference = np.asarray(
                    trusted[condition_index][qid]["aggregated_ad_logits"], dtype=np.float32
                )
                arrays["trusted_max_abs_error"][condition_index, qi] = float(
                    np.max(
                        np.abs(
                            arrays["logits"][condition_index, TRUSTED_NATURAL, qi]
                            - reference
                        )
                    )
                )
            if not audit_path.exists():
                audit_path.write_text(
                    json.dumps(
                        {
                            "condition": condition,
                            "rendered_prompt": batch["prompts"][0],
                            "prompt_hash": arrays["prompt_hashes"][condition_index, indices[0]].item(),
                            "rows": row_audits,
                            "ordinary_attention_layers_one_based": [
                                value + 1 for value in ORDINARY_LAYERS
                            ],
                            "scenario_definitions": {
                                "block_first_stem": "All queries from the feedback token onward cannot read 1P question-stem/separator tokens.",
                                "block_second_stem": "All 2P option-line, cue/query, and final-prefix queries cannot read causally prior 2P question-stem/separator tokens.",
                                "block_both_stems": "Exact union of the two preceding edge sets.",
                            },
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        completed_cohorts += 1
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        remaining = 125 - int(arrays["completed"].sum()) // 4
        eta = np.mean(durations) * remaining
        print(
            f"question-stem access: {int(arrays['completed'].sum())}/500; "
            f"cohort={duration:.2f}s eta={eta / 60:.1f}m",
            flush=True,
        )
        if args.max_cohorts is not None and completed_cohorts >= int(args.max_cohorts):
            print(f"Stopped after {args.max_cohorts} benchmark cohorts", flush=True)
            break

    completed = arrays["completed"]
    identity_error = float(
        np.nanmax(
            np.abs(
                arrays["logits"][:, IDENTITY, completed]
                - arrays["logits"][:, TRUSTED_NATURAL, completed]
            )
        )
    )
    metadata = {
        "experiment": "original/repeated question-stem ordinary-attention access factorial",
        "config": config.as_dict(),
        "n_questions": len(qids),
        "conditions": list(CONDITIONS),
        "scenarios": list(SCENARIOS),
        "ordinary_attention_layers_one_based": [value + 1 for value in ORDINARY_LAYERS],
        "complete_model_forwards_per_cohort": 2 * len(SCENARIOS),
        "complete_model_work": "Per task: trusted natural, identity monitor, first-stem block, second-stem block, and joint block.",
        "complete": bool(completed.all()),
        "natural_validation": {
            "max_abs_trusted_logit_error": float(
                np.nanmax(arrays["trusted_max_abs_error"][:, completed])
            )
        },
        "identity_validation": {
            "max_abs_identity_monitor_logit_error": identity_error,
        },
        "direction_geometry": geometry,
        "score_directions_path": str(args.score_directions),
        "score_directions_sha256": hashlib.sha256(
            args.score_directions.read_bytes()
        ).hexdigest(),
        "elapsed_seconds_after_model_load": time.monotonic() - started,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--score-directions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
