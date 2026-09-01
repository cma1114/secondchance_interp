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
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_feedback_source_localization import SOURCE_TOKEN_INDICES, _hash


BANDS_ONE_BASED = tuple((start, start + 7) for start in range(1, 65, 8))


def _scenario_specs(
    ordinary_layers: list[int], gla_layers: list[int]
) -> list[tuple[str, list[int], list[int]]]:
    all_layers = set(ordinary_layers) | set(gla_layers)
    if all_layers != set(range(64)):
        raise RuntimeError("Ordinary and GLA layers do not exhaust layers 1--64")

    rows: list[tuple[str, list[int], list[int]]] = [
        ("natural", [], []),
        ("all_layers_swapped", ordinary_layers, gla_layers),
        ("ordinary_all_swapped", ordinary_layers, []),
        ("gla_all_swapped", [], gla_layers),
    ]
    for start, stop in BANDS_ONE_BASED:
        band = set(range(start - 1, stop))
        rows.append((
            f"band_{start:02d}_{stop:02d}_only",
            sorted(band & set(ordinary_layers)),
            sorted(band & set(gla_layers)),
        ))
    for start, stop in BANDS_ONE_BASED:
        band = set(range(start - 1, stop))
        rows.append((
            f"all_except_band_{start:02d}_{stop:02d}",
            sorted(set(ordinary_layers) - band),
            sorted(set(gla_layers) - band),
        ))
    for _start, stop in BANDS_ONE_BASED[:-1]:
        prefix = set(range(stop))
        rows.append((
            f"prefix_through_{stop:02d}",
            sorted(prefix & set(ordinary_layers)),
            sorted(prefix & set(gla_layers)),
        ))
    if len(rows) != 27 or len({row[0] for row in rows}) != len(rows):
        raise RuntimeError("Policy-layer scenario construction changed")
    return rows


def _individual_layer_specs(
    ordinary_layers: list[int], gla_layers: list[int], selected_one_based: list[int]
) -> list[tuple[str, list[int], list[int]]]:
    selected = sorted(set(int(value) for value in selected_one_based))
    if not selected or any(value < 1 or value > 64 for value in selected):
        raise ValueError("Individual-layer plan must contain layers in L1-L64")
    ordinary_set = set(ordinary_layers)
    gla_set = set(gla_layers)
    rows: list[tuple[str, list[int], list[int]]] = [
        ("natural", [], []),
        ("all_layers_swapped", ordinary_layers, gla_layers),
        ("ordinary_all_swapped", ordinary_layers, []),
        ("gla_all_swapped", [], gla_layers),
    ]
    for one_based in selected:
        layer = one_based - 1
        rows.append((
            f"layer_{one_based:02d}_only",
            [layer] if layer in ordinary_set else [],
            [layer] if layer in gla_set else [],
        ))
        rows.append((
            f"all_except_layer_{one_based:02d}",
            sorted(ordinary_set - {layer}),
            sorted(gla_set - {layer}),
        ))
    if len({row[0] for row in rows}) != len(rows):
        raise RuntimeError("Individual-layer scenario names are not unique")
    return rows


def _initialize(path: Path, qids: list[str], scenario_ids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise RuntimeError("Existing checkpoint uses different questions")
        if arrays["scenario_ids"].astype(str).tolist() != scenario_ids:
            raise RuntimeError("Existing checkpoint uses different scenarios")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "scenario_ids": np.asarray(scenario_ids),
        "completed": np.zeros(n, dtype=bool),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "scenario_final_logits_raw": np.full(
            (2, len(scenario_ids), n, 4), np.nan, dtype=np.float32
        ),
        "scenario_final_logits": np.full(
            (2, len(scenario_ids), n, 4), np.nan, dtype=np.float32
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
    qids = all_qids
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
    if args.layer_plan is None:
        scenarios = _scenario_specs(ordinary_layers, gla_layers)
        selected_individual_layers: list[int] = []
    else:
        selected_individual_layers = [
            int(value) for value in json.loads(args.layer_plan.read_text())["layers_one_based"]
        ]
        scenarios = _individual_layer_specs(
            ordinary_layers, gla_layers, selected_individual_layers
        )
    scenario_ids = [row[0] for row in scenarios]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.npz"
    arrays = _initialize(result_path, qids, scenario_ids)
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
            _build_batch(config, processor, tokenizer, questions, mappings, cohort, condition)
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
                attention_mask = torch.nn.functional.pad(attention_mask, (pad, 0), value=0)
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
                    old_logits = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=np.float64)
                    rank_letters = [
                        LETTERS[int(value)] for value in np.argsort(-old_logits, kind="stable")
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
                        left_pad + partition[SOURCE_NAMES.index(f"feedback_token_{index}")][0]
                        for index in SOURCE_TOKEN_INDICES
                    ]
                    if feedback_positions != list(range(feedback_positions[0], feedback_positions[0] + 7)):
                        raise RuntimeError("Policy-bearing feedback tokens are not contiguous")
                    row_sources.append(feedback_positions)
                    prompt_ids = [
                        int(value)
                        for value in tokenizer(prompts[row], add_special_tokens=False)["input_ids"]
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
                        trusted[condition_index][qid]["aggregated_ad_logits"], dtype=np.float32
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
            for scenario_id, selected_ordinary, selected_gla in scenarios:
                ordinary = None
                gla = None
                try:
                    if selected_ordinary:
                        ordinary = BatchedSDPADownstreamSourceKVPatcher(
                            parts,
                            {
                                row: (donors[row], list(row_sources[row]), selected_ordinary)
                                for row in range(4)
                            },
                        )
                    if selected_gla:
                        gla = BatchedSelectiveGDNSourceWritePatcher(
                            parts,
                            {
                                row: (donors[row], list(row_sources[row]), selected_gla)
                                for row in range(4)
                            },
                            preserve_source_output=True,
                        )
                    scenario_outputs.append(
                        _aggregate_logits(_forward(model, parts, input_ids, attention_mask), variant_ids)
                    )
                    if ordinary is not None:
                        ordinary.assert_fired()
                    if gla is not None:
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
            f"feedback policy layer decomposition: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.2f}",
            flush=True,
        )
        if not audit_path.exists():
            cohort_audit["ordinary_layers_one_based"] = [value + 1 for value in ordinary_layers]
            cohort_audit["gla_layers_one_based"] = [value + 1 for value in gla_layers]
            cohort_audit["scenario_specs"] = [
                {
                    "scenario": scenario_id,
                    "ordinary_layers_one_based": [value + 1 for value in ordinary],
                    "gla_layers_one_based": [value + 1 for value in gla],
                }
                for scenario_id, ordinary, gla in scenarios
            ]
            audit_path.write_text(json.dumps(cohort_audit, indent=2) + "\n")

    metadata = {
        "experiment": "full-coverage policy-source layer decomposition",
        "questions": len(qids),
        "conditions": list(CONDITIONS),
        "scenario_count": len(scenarios),
        "scenario_ids": scenario_ids,
        "complete_model_forwards_per_canonical_cohort": 2 * len(scenarios),
        "paired_subbatches_per_canonical_cohort": 2,
        "bands_one_based": [list(value) for value in BANDS_ONE_BASED],
        "selected_individual_layers_one_based": selected_individual_layers,
        "source_scope": (
            "Complete incorrect/lost-through-final-period suffix; source-token output "
            "preserved; downstream ordinary K/V and GLA memory writes crossed only in "
            "the scenario's exhaustive layer set."
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
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-plan", type=Path)
    parser.add_argument("--max-cohorts", type=int)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
