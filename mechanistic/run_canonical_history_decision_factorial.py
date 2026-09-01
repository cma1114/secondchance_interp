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
from .collect_cross_model_behavioral_gate import _assert_prompt_pair, _scenario_messages
from .config import ExperimentConfig
from .downstream_source_intervention import (
    BatchedGDNSourceWriteAblator,
    BatchedSDPAQuerySourceAttentionAblator,
)
from .io import atomic_save_npz
from .jlens_collect import _token_offsets
from .modeling import (
    forward_runtime_kwargs,
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import present_question


MODEL_ID = "ByteDance-Seed/Seed-OSS-36B-Instruct"
MODEL_REVISION = "497f1dca95ebdec98e41d517b9f060ee753c902f"
EXPERIMENT_MODEL_NAME = "Seed-OSS 36B"
CANONICAL_BATCH_SIZE = 4
ATTENTION_LAYERS_ONE_BASED = tuple(range(1, 65))
EXPECTED_GLA_LAYERS = 0
FIRST_DECISION_OPENER = "<seed:bos>assistant"
ALLOWED_SERIALIZATIONS = ("hf_template",)

TASKS = ("Game", "Neutral")
TRUSTED_SCENARIOS = (
    "incorrect_again_nonremapped",
    "lost_again_nonremapped",
)
CELLS = (
    "natural",
    "matching",
    "cyclic_wrong",
    "first_decision",
    "matching_plus_first_decision",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _assert_binding(config: ExperimentConfig) -> None:
    if config.model_id != MODEL_ID or config.model_revision != MODEL_REVISION:
        raise ValueError("Configured model does not match the pinned binding")
    if config.batch_size != CANONICAL_BATCH_SIZE:
        raise ValueError(
            f"Requires canonical batch_size={CANONICAL_BATCH_SIZE}, found {config.batch_size}"
        )
    if config.chat_serialization not in ALLOWED_SERIALIZATIONS:
        raise ValueError(
            f"Requires serialization in {ALLOWED_SERIALIZATIONS}, "
            f"found {config.chat_serialization!r}"
        )
    if config.attn_implementation != "sdpa":
        raise ValueError("Requires the validated SDPA execution path")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires the canonical empty-history prompt")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token-matched incorrect/lost feedback")


def _question_occurrences(prompt: str, text: str) -> list[int]:
    starts: list[int] = []
    cursor = 0
    while True:
        found = prompt.find(text, cursor)
        if found < 0:
            return starts
        starts.append(found)
        cursor = found + len(text)


def _option_lines(
    tokenizer: Any,
    prompt: str,
    question: dict[str, Any],
    occurrence: int,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    question_text = present_question(question)
    starts = _question_occurrences(prompt, question_text)
    if len(starts) != 2:
        raise RuntimeError(
            f"Canonical non-remapped prompt must contain exactly two presentations; "
            f"found {len(starts)}"
        )
    question_start = starts[occurrence]
    question_end = question_start + len(question_text)
    positions: dict[str, list[int]] = {}
    audit: dict[str, Any] = {}
    for letter in LETTERS:
        line = f"  {letter}: {question['options'][letter]}\n"
        start = prompt.find(line, question_start, question_end)
        if start < 0:
            raise RuntimeError(f"Could not locate presentation {occurrence + 1} line {letter}")
        row = [
            index
            for index, (left, right) in enumerate(offsets)
            if right > left and left < start + len(line) and right > start
        ]
        if not row:
            raise RuntimeError(f"Presentation {occurrence + 1} line {letter} is empty")
        positions[letter] = row
        audit[letter] = {
            "text": line.rstrip("\n"),
            "positions": row,
            "tokens": tokenizer.convert_ids_to_tokens([ids[index] for index in row]),
        }
    return positions, audit


def _first_decision_position(tokenizer: Any, prompt: str) -> tuple[int, dict[str, Any]]:
    start = prompt.find(FIRST_DECISION_OPENER)
    if start < 0:
        raise RuntimeError(f"Could not locate first-decision opener {FIRST_DECISION_OPENER!r}")
    if prompt.find(FIRST_DECISION_OPENER, start + 1) < 0:
        raise RuntimeError("Could not locate the final assistant opener")
    end = start + len(FIRST_DECISION_OPENER)
    offsets = _token_offsets(tokenizer, prompt)
    candidates = [
        index for index, (left, right) in enumerate(offsets)
        if right > left and right <= end
    ]
    if not candidates:
        raise RuntimeError("The first-decision opener has no token boundary")
    position = int(candidates[-1])
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return position, {
        "opener": FIRST_DECISION_OPENER,
        "position": position,
        "token_id": int(ids[position]),
        "token": tokenizer.convert_ids_to_tokens([int(ids[position])])[0],
    }


def _merge_specs(
    layer_indices: tuple[int, ...],
    sources: list[list[list[int]]],
    queries: list[list[list[int]]],
    decision_positions: list[int],
    width: int,
    *,
    include_matching: bool,
    cyclic_wrong: bool,
    include_decision: bool,
) -> dict[int, dict[int, dict[int, list[int]]]]:
    specs: dict[int, dict[int, dict[int, list[int]]]] = {}
    for layer in layer_indices:
        layer_rows: dict[int, dict[int, list[int]]] = {}
        for row in range(len(sources)):
            row_specs: dict[int, set[int]] = {}
            if include_matching:
                for target_rank in range(4):
                    source_rank = (target_rank + 1) % 4 if cyclic_wrong else target_rank
                    for query in queries[row][target_rank]:
                        row_specs.setdefault(int(query), set()).update(
                            int(value) for value in sources[row][source_rank]
                        )
            if include_decision:
                decision = int(decision_positions[row])
                for query in range(decision + 1, width):
                    row_specs.setdefault(query, set()).add(decision)
            if not row_specs:
                raise RuntimeError("Constructed an empty intervention row")
            layer_rows[row] = {
                query: sorted(values) for query, values in row_specs.items()
            }
        specs[layer] = layer_rows
    return specs


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses a different question order")
        if arrays["cells"].astype(str).tolist() != list(CELLS):
            raise ValueError("Existing checkpoint uses different cells")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "cells": np.asarray(CELLS),
        "completed": np.zeros(n, dtype=bool),
        "rank_contents": np.full((n, 4), "", dtype="<U1"),
        "baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "logits": np.full((2, len(CELLS), n, 4), np.nan, dtype=np.float32),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "first_decision_positions": np.full((2, n), -1, dtype=np.int16),
        "source_position_counts": np.zeros((2, n, 4), dtype=np.int16),
        "query_position_counts": np.zeros((2, n, 4), dtype=np.int16),
    }


def _load_trusted(
    path: Path, qids: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return baseline logits, rank order, and Game/Neutral natural logits."""
    if path.suffix == ".npz":
        if MODEL_ID != "Qwen/Qwen3.6-27B":
            raise ValueError("NPZ trusted trajectories are only valid for the Qwen binding")
        with np.load(path, allow_pickle=False) as loaded:
            trusted_qids = loaded["question_ids"].astype(str).tolist()
            trusted_index = {qid: index for index, qid in enumerate(trusted_qids)}
            if len(trusted_index) != len(trusted_qids):
                raise ValueError("Trusted trajectory contains duplicate question IDs")
            missing = [qid for qid in qids if qid not in trusted_index]
            if missing:
                raise ValueError(
                    f"Trusted trajectory is missing {len(missing)} requested questions"
                )
            indices = [trusted_index[qid] for qid in qids]
            if loaded["conditions"].astype(str).tolist() != ["game", "neutral"]:
                raise ValueError("Trusted trajectory condition order is not Game/Neutral")
            rank_order = np.asarray(loaded["rank_order"], dtype=int)[indices]
            natural = np.asarray(loaded["direct_logits"], dtype=np.float32)[:, indices]
        baseline = np.full((len(qids), 4), np.nan, dtype=np.float32)
        return baseline, rank_order, natural

    trusted = json.loads(path.read_text())
    if (
        trusted.get("model_id") != MODEL_ID
        or trusted.get("model_revision") != MODEL_REVISION
        or not trusted.get("complete")
    ):
        raise ValueError("Trusted behavior is incomplete or from another model revision")
    baseline_rows = trusted["scenarios"]["baseline"]
    task_rows = [trusted["scenarios"][name] for name in TRUSTED_SCENARIOS]
    missing = [
        qid for qid in qids
        if qid not in baseline_rows or any(qid not in rows for rows in task_rows)
    ]
    if missing:
        raise ValueError(f"Trusted behavior is missing {len(missing)} questions")
    baseline = np.asarray(
        [baseline_rows[qid]["aggregated_ad_logits"] for qid in qids], dtype=np.float32
    )
    rank_order = np.argsort(-baseline, axis=1, kind="stable")
    natural = np.asarray(
        [
            [task_rows[task][qid]["aggregated_ad_logits"] for qid in qids]
            for task in range(2)
        ],
        dtype=np.float32,
    )
    return baseline, rank_order, natural


def _aggregate_final_logits(
    final_logits: Any, variant_ids: dict[str, list[int]]
) -> np.ndarray:
    import torch

    final_logits = final_logits.detach().float().cpu()
    return torch.stack(
        [
            torch.logsumexp(final_logits[:, variant_ids[letter]], dim=-1)
            for letter in LETTERS
        ],
        dim=-1,
    ).numpy()


def _forward(model: Any, parts: Any, input_ids: Any, attention_mask: Any) -> Any:
    import torch

    device = model_input_device(parts)
    kwargs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "return_dict": True,
    }
    kwargs.update(forward_runtime_kwargs(model, input_ids, device))
    with torch.inference_mode():
        try:
            output = model(**kwargs, logits_to_keep=1)
        except TypeError:
            output = model(**kwargs)
    return output.logits[:, 0] if output.logits.shape[1] == 1 else output.logits[:, -1]


def _intervened(
    model: Any,
    parts: Any,
    input_ids: Any,
    attention_mask: Any,
    variant_ids: dict[str, list[int]],
    specs: dict[int, dict[int, dict[int, list[int]]]],
    decision_positions: list[int] | None,
) -> tuple[np.ndarray, dict[str, int]]:
    attention_ablator = BatchedSDPAQuerySourceAttentionAblator(parts, specs)
    gla_ablator = None
    try:
        if decision_positions is not None and EXPECTED_GLA_LAYERS:
            gla_ablator = BatchedGDNSourceWriteAblator(
                parts,
                {row: [position] for row, position in enumerate(decision_positions)},
            )
        logits = _aggregate_final_logits(
            _forward(model, parts, input_ids, attention_mask), variant_ids
        )
        attention_ablator.assert_fired()
        if gla_ablator is not None:
            gla_ablator.assert_fired()
        stats = {
            "sdpa_calls": int(attention_ablator.sdpa_calls),
            "edited_attention_edges": int(attention_ablator.edited_edge_count),
            "gla_rule_calls": 0 if gla_ablator is None else int(gla_ablator.rule_calls),
            "edited_gla_positions": (
                0 if gla_ablator is None else int(gla_ablator.edited_position_count)
            ),
        }
    finally:
        if gla_ablator is not None:
            gla_ablator.close()
        attention_ablator.close()
    if not np.all(np.isfinite(logits)):
        raise RuntimeError("Intervention produced non-finite logits")
    return logits, stats


def run(
    config_path: Path,
    trusted_behavior_path: Path,
    output_dir: Path,
    max_cohorts: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    _assert_binding(config)
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    if config.question_ids is not None:
        wanted = set(config.question_ids)
        qids = [qid for qid in qids if qid in wanted]
        if set(qids) != wanted:
            raise ValueError("Configured question IDs are absent from the manifest")
    if config.max_questions is not None:
        qids = qids[: config.max_questions]
    if max_cohorts is not None:
        qids = qids[: int(max_cohorts) * config.batch_size]
    dataset = "TriviaMC" if qids and qids[0].startswith("triviamc_") else "SimpleMC"

    baseline, rank_order, trusted_natural = _load_trusted(trusted_behavior_path, qids)
    if not np.all(np.isfinite(trusted_natural)):
        raise RuntimeError("Trusted natural logits are non-finite")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids)
    arrays["baseline_logits"][:] = baseline
    arrays["trusted_natural_logits"][:] = trusted_natural
    arrays["rank_contents"][:] = np.asarray(
        [[LETTERS[int(index)] for index in row] for row in rank_order]
    )
    qid_index = {qid: index for index, qid in enumerate(qids)}

    load_started = time.monotonic()
    model, processor, parts = load_model_and_processor(config)
    model_load_seconds = time.monotonic() - load_started
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _text, token_id in resolved[letter]})
        for letter in LETTERS
    }
    attention_layers = tuple(
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if tuple(index + 1 for index in attention_layers) != ATTENTION_LAYERS_ONE_BASED:
        raise RuntimeError("The loaded ordinary-attention inventory differs from the binding")
    gla_layers = tuple(
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    )
    if len(gla_layers) != EXPECTED_GLA_LAYERS:
        raise RuntimeError(
            f"Expected {EXPECTED_GLA_LAYERS} GLA layers, found {len(gla_layers)}"
        )
    print(f"MODEL_LOADED seconds={model_load_seconds:.3f}", flush=True)

    durations: list[float] = []
    stats: list[dict[str, Any]] = []
    audit_path = output_dir / "prompt_audit.json"
    started = time.monotonic()
    for start in range(0, len(qids), config.batch_size):
        cohort = qids[start : start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if np.all(arrays["completed"][indices]):
            continue
        cohort_started = time.monotonic()
        audit: dict[str, Any] = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "question_ids": cohort,
            "tasks": {},
        }
        representative_prompts: list[str] = []

        for task, trusted_scenario in enumerate(TRUSTED_SCENARIOS):
            prompts: list[str] = []
            messages_rows: list[list[dict[str, str]]] = []
            token_rows: list[list[int]] = []
            for qid in cohort:
                messages, remapped = _scenario_messages(
                    trusted_scenario, questions[qid], {letter: letter for letter in LETTERS}
                )
                if remapped is not None:
                    raise RuntimeError("Canonical non-remapped scenario unexpectedly remapped")
                prompt = render_chat(
                    processor,
                    messages,
                    config.disable_thinking,
                    config.chat_serialization,
                    config.chat_template_kwargs,
                )
                prompts.append(prompt)
                messages_rows.append(messages)
                token_rows.append(
                    [int(value) for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
                )

            input_ids, attention_mask, _last = tokenize_batch(tokenizer, prompts)
            width = int(input_ids.shape[1])
            source_rows: list[list[list[int]]] = []
            query_rows: list[list[list[int]]] = []
            decisions: list[int] = []
            span_audits: list[dict[str, Any]] = []
            for row, qid in enumerate(cohort):
                qi = qid_index[qid]
                left_pad = width - len(token_rows[row])
                if input_ids[row, left_pad:].tolist() != token_rows[row]:
                    raise RuntimeError("Batched tokenization differs from audited tokenization")
                first, first_audit = _option_lines(
                    tokenizer, prompts[row], questions[qid], occurrence=0
                )
                second, second_audit = _option_lines(
                    tokenizer, prompts[row], questions[qid], occurrence=1
                )
                decision, decision_audit = _first_decision_position(tokenizer, prompts[row])
                decision += left_pad
                ranks = arrays["rank_contents"][qi].astype(str).tolist()
                sources = [[left_pad + value for value in first[content]] for content in ranks]
                queries = [[left_pad + value for value in second[content]] for content in ranks]
                if any(not values for values in sources + queries):
                    raise RuntimeError("A matching source or receiver span is empty")
                if any(max(sources[rank]) >= min(queries[rank]) for rank in range(4)):
                    raise RuntimeError("A matching source does not precede its receiver")
                if not (max(max(values) for values in sources) < decision < min(min(values) for values in queries)):
                    raise RuntimeError("First-decision token is not between the two option lists")
                source_rows.append(sources)
                query_rows.append(queries)
                decisions.append(decision)
                arrays["first_decision_positions"][task, qi] = decision
                arrays["prompt_hashes"][task, qi] = _hash_prompt(prompts[row])
                for rank in range(4):
                    arrays["source_position_counts"][task, qi, rank] = len(sources[rank])
                    arrays["query_position_counts"][task, qi, rank] = len(queries[rank])
                span_audits.append(
                    {
                        "rank_contents": ranks,
                        "first_option_lines": first_audit,
                        "second_option_lines": second_audit,
                        "first_decision": decision_audit,
                        "first_decision_physical": decision,
                    }
                )

            natural = _aggregate_final_logits(
                _forward(model, parts, input_ids, attention_mask), variant_ids
            )
            if not np.all(np.isfinite(natural)):
                raise RuntimeError("Natural logits are non-finite")
            arrays["logits"][task, CELLS.index("natural"), indices] = natural

            definitions = (
                ("matching", True, False, False),
                ("cyclic_wrong", True, True, False),
                ("first_decision", False, False, True),
                ("matching_plus_first_decision", True, False, True),
            )
            for name, include_matching, cyclic, include_decision in definitions:
                specs = _merge_specs(
                    attention_layers,
                    source_rows,
                    query_rows,
                    decisions,
                    width,
                    include_matching=include_matching,
                    cyclic_wrong=cyclic,
                    include_decision=include_decision,
                )
                values, cell_stats = _intervened(
                    model,
                    parts,
                    input_ids,
                    attention_mask,
                    variant_ids,
                    specs,
                    decisions if include_decision else None,
                )
                arrays["logits"][task, CELLS.index(name), indices] = values
                stats.append({"task": TASKS[task], "cell": name, **cell_stats})

            audit["tasks"][TASKS[task]] = {
                "rendered_prompt": prompts[0],
                "messages": messages_rows[0],
                "prompt_hash": arrays["prompt_hashes"][task, indices[0]].item(),
                **span_audits[0],
            }
            representative_prompts.append(prompts[0])

        _assert_prompt_pair(representative_prompts[0], representative_prompts[1])
        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        if not audit_path.exists():
            _write_json(audit_path, audit)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        max_error = float(
            np.max(
                np.abs(
                    arrays["logits"][:, CELLS.index("natural"), indices]
                    - arrays["trusted_natural_logits"][:, indices]
                )
            )
        )
        print(
            f"PROGRESS model={MODEL_ID} dataset={dataset} "
            f"questions={int(arrays['completed'].sum())}/{len(qids)} "
            f"cohort_seconds={duration:.3f} natural_error={max_error:.9g}",
            flush=True,
        )

    if not np.all(np.isfinite(arrays["logits"])):
        raise RuntimeError("Completed output contains non-finite logits")
    natural_error = float(
        np.max(
            np.abs(
                arrays["logits"][:, CELLS.index("natural")]
                - arrays["trusted_natural_logits"]
            )
        )
    )
    if natural_error != 0.0:
        raise RuntimeError(f"Trusted natural reproduction failed: {natural_error}")
    metadata = {
        "experiment": f"{EXPERIMENT_MODEL_NAME} {dataset} canonical history/decision factorial",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset": dataset,
        "config": config.as_dict(),
        "n_questions": len(qids),
        "tasks": list(TASKS),
        "cells": list(CELLS),
        "complete_model_forwards_per_cohort": 10,
        "ordinary_attention_layers_one_based": list(ATTENTION_LAYERS_ONE_BASED),
        "gla_layers": EXPECTED_GLA_LAYERS,
        "first_decision_source": (
            "The final token of the first assistant answer-generation opener/scaffold. "
            "Every strictly later ordinary-attention query is denied this source; "
            "on Qwen its recurrent GLA write is also removed."
        ),
        "matching_source": "Every token of each complete 1P option line",
        "matching_receiver": "Every token of the identical complete 2P option line",
        "cyclic_wrong_control": "W1<-W2, W2<-W3, W3<-W4, W4<-W1",
        "natural_reproduction_max_absolute_error": natural_error,
        "model_load_seconds": model_load_seconds,
        "elapsed_seconds_after_load": time.monotonic() - started,
        "cohort_seconds": durations,
        "intervention_stats": stats,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trusted-behavior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(args.config, args.trusted_behavior, args.output_dir, args.max_cohorts)


if __name__ == "__main__":
    main()
