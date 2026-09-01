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
    render_chat,
    tokenize_batch,
)
from .prompts import prompt_hash
from .relay_interception import (
    BatchedGLACachedRelayInterceptor,
    BatchedGLARelayWriteCache,
    BatchedSDPACachedRelayInterceptor,
    BatchedSDPARelayWriteCache,
)
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_fixed_bcd_line_transplant import option_line_positions
from .run_semantic_binding_module_factorial import _messages, _remap_question


TASKS = ("Game", "Neutral")
CONDITIONS = ("incorrect_again", "lost_again")
VARIANTS = ("low", "high")
DONOR_ROWS = np.asarray([1, 0, 3, 2], dtype=np.int64)
SCENARIOS = (
    "natural",
    "duplicate_natural",
    "option_lines_swapped",
    "intercept_choice_cue_and_query",
    "intercept_final_assistant_prefix",
    "intercept_both_relays",
)


def _initialize(path: Path, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    pair_ids = [row["pair_id"] for row in rows]
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["pair_ids"].astype(str).tolist() != pair_ids:
            raise RuntimeError("Existing checkpoint uses a different frozen pair plan")
        if arrays["scenario_ids"].astype(str).tolist() != list(SCENARIOS):
            raise RuntimeError("Existing checkpoint uses different scenarios")
        return arrays
    n = len(rows)
    return {
        "pair_ids": np.asarray(pair_ids),
        "question_ids": np.asarray([row["question_id"] for row in rows]),
        "split": np.asarray([row["split"] for row in rows]),
        "target_original_letter": np.asarray([row["target_original_letter"] for row in rows]),
        "target_displayed_letter": np.asarray([row["target_displayed_letter"] for row in rows]),
        "mapping_indices": np.asarray([
            [row["low_mapping_index"], row["high_mapping_index"]] for row in rows
        ], dtype=np.int16),
        "screen_fresh_semantic_logits": np.asarray([
            [row["low_fresh_semantic_centered_logits"], row["high_fresh_semantic_centered_logits"]]
            for row in rows
        ], dtype=np.float32),
        "scenario_ids": np.asarray(SCENARIOS),
        "completed": np.zeros(n, dtype=bool),
        "prompt_hashes": np.full((2, 2, n), "", dtype="<U64"),
        "scenario_final_logits": np.full((2, 2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32),
        "source_token_counts": np.full((2, 2, n), -1, dtype=np.int16),
        "relay_token_counts": np.full((2, 2, n, 2), -1, dtype=np.int16),
    }


def _semantic_mapping(mapping: dict[str, str]) -> dict[str, str]:
    return {original: displayed for displayed, original in mapping.items()}


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

    plan = json.loads(args.pair_plan.read_text())
    rows = list(plan["rows"])
    if args.max_questions is not None:
        rows = rows[: int(args.max_questions)]
    if not rows:
        raise ValueError("No fresh-score pairs selected")
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {str(row["id"]): row for row in manifest["questions"]}
    baseline = json.loads(args.baseline.read_text())["results"]

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
        raise RuntimeError("Unexpected ordinary-attention inventory")
    if len(gla_layers) != 48:
        raise RuntimeError("Unexpected GLA inventory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.npz"
    arrays = _initialize(result_path, rows)
    audit_path = args.output_dir / "prompt_audit.json"
    durations: list[float] = []
    started = time.monotonic()

    for qi, row in enumerate(rows):
        if arrays["completed"][qi]:
            continue
        question_started = time.monotonic()
        qid = row["question_id"]
        first = questions[qid]
        second_low = _remap_question(first, row["low_new_to_original"])
        second_high = _remap_question(first, row["high_new_to_original"])
        seconds = (second_low, second_high, second_low, second_high)
        conditions = (CONDITIONS[0], CONDITIONS[0], CONDITIONS[1], CONDITIONS[1])
        prompts = [
            render_chat(
                processor, _messages(config, first, second, condition),
                config.disable_thinking, config.chat_serialization,
            )
            for second, condition in zip(seconds, conditions)
        ]
        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        width = int(input_ids.shape[1])
        final_query = width - 1
        prompt_token_rows = [
            [int(value) for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
            for prompt in prompts
        ]
        if len({len(values) for values in prompt_token_rows}) != 1:
            raise RuntimeError("Low/high causal prompts are not token-length aligned")

        old_logits = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=np.float64)
        rank_letters = [LETTERS[int(value)] for value in np.argsort(-old_logits, kind="stable")]
        source_positions: dict[int, list[int]] = {}
        choice_positions: dict[int, list[int]] = {}
        final_prefix_positions: dict[int, list[int]] = {}
        row_audits: list[dict[str, Any]] = []
        for physical_row, (prompt, second, condition) in enumerate(zip(prompts, seconds, conditions)):
            mapping = row["low_new_to_original"] if physical_row % 2 == 0 else row["high_new_to_original"]
            partition, position_audit = _cue_source_partition(
                tokenizer, prompt, _messages(config, first, second, condition),
                first, second, condition, rank_letters, _semantic_mapping(mapping),
            )
            left_pad = width - int(position_audit["prompt_length"])
            literal_lines, _literal_audit = option_line_positions(
                tokenizer, prompt, second, "second"
            )
            option_source = sorted(
                left_pad + value
                for displayed in LETTERS
                for value in literal_lines[displayed]
            )
            choice = [
                left_pad + value
                for value in partition[SOURCE_NAMES.index("second_choice_cue_and_query")]
            ]
            final_prefix = [
                left_pad + value
                for value in partition[SOURCE_NAMES.index("final_assistant_prefix")]
                if left_pad + value < final_query
            ]
            if not option_source or not choice or not final_prefix:
                raise RuntimeError("Source and relay regions must all be nonempty")
            if max(option_source) >= min(choice) or max(choice) >= min(final_prefix):
                raise RuntimeError("Fresh source and relay regions are not causally ordered")
            source_positions[physical_row] = option_source
            choice_positions[physical_row] = choice
            final_prefix_positions[physical_row] = final_prefix
            task_index, variant_index = divmod(physical_row, 2)
            arrays["prompt_hashes"][task_index, variant_index, qi] = prompt_hash(prompt)
            arrays["source_token_counts"][task_index, variant_index, qi] = len(option_source)
            arrays["relay_token_counts"][task_index, variant_index, qi] = [
                len(choice), len(final_prefix)
            ]
            row_audits.append({
                "row": physical_row,
                "task": TASKS[task_index],
                "variant": VARIANTS[variant_index],
                "prompt_hash": prompt_hash(prompt),
                "source_positions": option_source,
                "choice_relay_positions": choice,
                "final_prefix_positions": final_prefix,
                "target_original_letter": row["target_original_letter"],
                "target_displayed_letter": row["target_displayed_letter"],
            })
        if len({tuple(values) for values in source_positions.values()}) != 1:
            raise RuntimeError("Complete 2P option blocks do not occupy identical positions")
        if len({tuple(values) for values in choice_positions.values()}) != 1:
            raise RuntimeError("Choice relays do not occupy identical positions")
        if len({tuple(values) for values in final_prefix_positions.values()}) != 1:
            raise RuntimeError("Final-prefix relays do not occupy identical positions")

        donors = {row_index: int(DONOR_ROWS[row_index]) for row_index in range(4)}
        all_relays = {
            row_index: sorted(choice_positions[row_index] + final_prefix_positions[row_index])
            for row_index in range(4)
        }
        ordinary_cache = BatchedSDPARelayWriteCache(parts, all_relays, ordinary_layers)
        gla_cache = BatchedGLARelayWriteCache(parts, all_relays, gla_layers)
        try:
            natural = _aggregate_logits(
                _forward(model, parts, input_ids, attention_mask), variant_ids
            )
        finally:
            gla_cache.close()
            ordinary_cache.close()
        if set(ordinary_cache.cache) != set(ordinary_layers):
            raise RuntimeError("Ordinary relay cache is incomplete")
        if set(gla_cache.cache) != set(gla_layers):
            raise RuntimeError("GLA relay cache is incomplete")

        outputs: list[np.ndarray] = [natural]
        outputs.append(_aggregate_logits(_forward(model, parts, input_ids, attention_mask), variant_ids))
        for scenario in SCENARIOS[2:]:
            ordinary = None
            gla = None
            try:
                if scenario == "option_lines_swapped":
                    ordinary = BatchedSDPADownstreamSourceKVPatcher(
                        parts, {
                            target: (donors[target], source_positions[target], ordinary_layers)
                            for target in range(4)
                        }
                    )
                    gla = BatchedSelectiveGDNSourceWritePatcher(
                        parts, {
                            target: (donors[target], source_positions[target], gla_layers)
                            for target in range(4)
                        }, preserve_source_output=True,
                    )
                else:
                    if scenario == "intercept_choice_cue_and_query":
                        relays = choice_positions
                    elif scenario == "intercept_final_assistant_prefix":
                        relays = final_prefix_positions
                    elif scenario == "intercept_both_relays":
                        relays = all_relays
                    else:
                        raise RuntimeError(f"Unknown scenario {scenario}")
                    ordinary = BatchedSDPACachedRelayInterceptor(
                        parts, {
                            target: (
                                donors[target], source_positions[target], relays[target], ordinary_layers
                            ) for target in range(4)
                        }, ordinary_cache.cache,
                    )
                    gla = BatchedGLACachedRelayInterceptor(
                        parts, {
                            target: (
                                donors[target], source_positions[target], relays[target], gla_layers
                            ) for target in range(4)
                        }, gla_cache.cache,
                    )
                outputs.append(_aggregate_logits(
                    _forward(model, parts, input_ids, attention_mask), variant_ids
                ))
            finally:
                if gla is not None:
                    gla.close()
                if ordinary is not None:
                    ordinary.close()
        stacked = np.stack(outputs)
        for task_index in range(2):
            for variant_index in range(2):
                physical_row = task_index * 2 + variant_index
                arrays["scenario_final_logits"][task_index, variant_index, :, qi] = stacked[:, physical_row]

        if not np.isfinite(stacked).all():
            raise RuntimeError("Non-finite causal logits")
        arrays["completed"][qi] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - question_started
        durations.append(duration)
        print(
            f"fresh-score path crossover: {int(arrays['completed'].sum())}/{len(rows)}; "
            f"seconds={duration:.2f}", flush=True,
        )
        if not audit_path.exists():
            audit_path.write_text(json.dumps({
                "pair": row,
                "rows": row_audits,
                "ordinary_layers_one_based": [value + 1 for value in ordinary_layers],
                "gla_layers_one_based": [value + 1 for value in gla_layers],
            }, indent=2) + "\n")

    metadata = {
        "experiment": "natural fresh-2P evidence crossover and downstream relay mediation",
        "questions": len(rows),
        "tasks": list(TASKS),
        "variants": list(VARIANTS),
        "scenarios": list(SCENARIOS),
        "complete_model_forwards_per_question": 6,
        "complete_model_work": (
            "six full forwards: natural, duplicate natural, complete 2P option-line "
            "source crossover, and three downstream relay interceptions"
        ),
        "ordinary_layers_one_based": [value + 1 for value in ordinary_layers],
        "gla_layers_one_based": [value + 1 for value in gla_layers],
        "elapsed_seconds_after_load": time.monotonic() - started,
        "question_seconds": durations,
        "software": {
            "python": sys.version, "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pair-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
