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
from .downstream_source_intervention import BatchedSDPAQuerySourceAttentionAblator
from .io import atomic_save_npz
from .modeling import (
    forward_runtime_kwargs,
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .run_all_candidate_matched_relay import _specs
from .run_fixed_a_final_query_edge_ablation import _option_line_positions


MODEL_ID = "ByteDance-Seed/Seed-OSS-36B-Instruct"
MODEL_REVISION = "497f1dca95ebdec98e41d517b9f060ee753c902f"
EXPERIMENT_MODEL_NAME = "Seed-OSS 36B"
CANONICAL_BATCH_SIZE = 4
ATTENTION_LAYERS_ONE_BASED = tuple(range(1, 65))
TASKS = ("Game", "Neutral")
TRUSTED_SCENARIOS = ("incorrect_again_remapped", "lost_again_remapped")
SCENARIOS = ("natural", "joint_matching", "joint_cyclic_wrong")
RANKS = ("W1", "W2", "W3", "W4")


def _assert_binding_config(config: ExperimentConfig) -> None:
    if config.model_id != MODEL_ID or config.model_revision != MODEL_REVISION:
        raise ValueError("Requires the binding's pinned configured model revision")
    if config.batch_size != CANONICAL_BATCH_SIZE:
        raise ValueError(
            f"Requires canonical batch_size={CANONICAL_BATCH_SIZE}, "
            f"found {config.batch_size}"
        )


def _experiment_name(dataset: str) -> str:
    return (
        f"{EXPERIMENT_MODEL_NAME} {dataset} "
        "all-candidate matching-history blockade"
    )


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses a different question order")
        if arrays["scenarios"].astype(str).tolist() != list(SCENARIOS):
            raise ValueError("Existing checkpoint uses different scenarios")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "scenarios": np.asarray(SCENARIOS),
        "completed": np.zeros(n, dtype=bool),
        "rank_contents": np.full((n, 4), "", dtype="<U1"),
        "baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "joint_matching_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "joint_cyclic_wrong_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "source_position_counts": np.zeros((n, 4), dtype=np.int16),
        "query_position_counts": np.zeros((n, 4), dtype=np.int16),
        "cyclic_source_position_counts": np.zeros((n, 4), dtype=np.int16),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
    }


def _aggregate_final_logits(
    final_logits: Any, variant_ids: dict[str, list[int]]
) -> np.ndarray:
    import torch

    # Match the trusted behavioral collector exactly: transfer the final full-
    # vocabulary logits to CPU float32 before aggregating bare and space-prefixed
    # A-D token variants.  Performing logsumexp on the GPU can differ by one
    # float32 ULP even when the underlying model forward is identical.
    final_logits = final_logits.detach().float().cpu()
    values = torch.stack(
        [
            torch.logsumexp(final_logits[:, variant_ids[letter]], dim=-1)
            for letter in LETTERS
        ],
        dim=-1,
    )
    return values.numpy()


def _forward_final_logits(model: Any, parts: Any, input_ids: Any, attention_mask: Any):
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
    logits = output.logits
    if logits.shape[1] == 1:
        return logits[:, 0]
    return logits[:, -1]


def _intervened_forward(
    model: Any,
    parts: Any,
    input_ids: Any,
    attention_mask: Any,
    variant_ids: dict[str, list[int]],
    layer_indices: tuple[int, ...],
    sources: list[list[list[int]]],
    queries: list[list[list[int]]],
    cyclic_wrong: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    specs = _specs(
        layer_indices,
        sources,
        queries,
        tuple(range(4)),
        controls=cyclic_wrong,
    )
    with BatchedSDPAQuerySourceAttentionAblator(parts, specs) as ablator:
        logits = _aggregate_final_logits(
            _forward_final_logits(model, parts, input_ids, attention_mask),
            variant_ids,
        )
        stats = {
            "sdpa_calls": int(ablator.sdpa_calls),
            "edited_edge_count": int(ablator.edited_edge_count),
            "unique_layers_seen": len(set(ablator.layers_seen)),
        }
        if set(ablator.layers_seen) != set(layer_indices):
            raise RuntimeError(
                "The intervention did not execute at every selected attention layer"
            )
    if not np.all(np.isfinite(logits)):
        raise RuntimeError("Non-finite intervention logits")
    return logits, stats


def run(
    config_path: Path,
    remapping_plan_path: Path,
    trusted_behavior_path: Path,
    output_dir: Path,
    max_cohorts: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    _assert_binding_config(config)
    if (
        config.model_loader not in {"causal_lm", "multimodal"}
        or config.chat_serialization != "hf_template"
        or config.attn_implementation != "sdpa"
    ):
        raise ValueError("Requires a supported native HF text path with SDPA")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires the clean empty-history prompt")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token-matched incorrect/lost feedback")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    if config.question_ids is not None:
        wanted = set(config.question_ids)
        qids = [qid for qid in qids if qid in wanted]
        if set(qids) != wanted:
            raise ValueError("Configured question IDs are not all in the manifest")
    if config.max_questions is not None:
        qids = qids[: config.max_questions]
    if max_cohorts is not None:
        qids = qids[: int(max_cohorts) * config.batch_size]
    dataset = "TriviaMC" if qids and qids[0].startswith("triviamc_") else "SimpleMC"

    mappings = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    trusted = json.loads(trusted_behavior_path.read_text())
    if (
        trusted.get("model_id") != MODEL_ID
        or trusted.get("model_revision") != MODEL_REVISION
        or not trusted.get("complete")
    ):
        raise ValueError("Trusted behavioral artifact is incomplete or from another model")
    baseline = trusted["scenarios"]["baseline"]
    trusted_tasks = [trusted["scenarios"][name] for name in TRUSTED_SCENARIOS]
    missing = [
        qid
        for qid in qids
        if qid not in mappings
        or qid not in baseline
        or any(qid not in rows for rows in trusted_tasks)
    ]
    if missing:
        raise ValueError(f"Frozen inputs are missing {len(missing)} questions")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids)
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
    layer_indices = tuple(
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if tuple(index + 1 for index in layer_indices) != ATTENTION_LAYERS_ONE_BASED:
        raise RuntimeError(
            "Unexpected attention inventory: "
            f"{tuple(index + 1 for index in layer_indices)}"
        )
    text_config = getattr(model.config, "text_config", model.config)
    query_heads = int(text_config.num_attention_heads)
    key_value_heads = int(text_config.num_key_value_heads)
    if MODEL_ID == "ByteDance-Seed/Seed-OSS-36B-Instruct" and (
        query_heads != 80 or key_value_heads != 8
    ):
        raise RuntimeError("Unexpected Seed grouped-query-attention configuration")
    print(f"MODEL_LOADED seconds={model_load_seconds:.3f}", flush=True)

    started = time.monotonic()
    durations: list[float] = []
    intervention_stats: list[dict[str, Any]] = []
    audit_path = output_dir / "prompt_audit.json"
    for start in range(0, len(qids), config.batch_size):
        cohort = qids[start : start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if np.all(arrays["completed"][indices]):
            continue
        cohort_started = time.monotonic()

        for qid in cohort:
            qi = qid_index[qid]
            logits = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=float)
            order = np.argsort(-logits, kind="stable")
            arrays["baseline_logits"][qi] = logits
            arrays["rank_contents"][qi] = np.asarray(
                [LETTERS[int(value)] for value in order]
            )

        cohort_audit: dict[str, Any] = {
            "question_ids": cohort,
            "tasks": {},
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
        rendered_prompts: list[str] = []
        for task_index, trusted_scenario in enumerate(TRUSTED_SCENARIOS):
            prompts: list[str] = []
            messages: list[list[dict[str, str]]] = []
            token_rows: list[list[int]] = []
            remapped_questions: list[dict[str, Any]] = []
            for qid in cohort:
                row_messages, remapped = _scenario_messages(
                    trusted_scenario,
                    questions[qid],
                    mappings[qid]["new_to_original"],
                )
                if remapped is None:
                    raise RuntimeError("The matching-history test requires remapped 2P options")
                prompt = render_chat(
                    processor,
                    row_messages,
                    config.disable_thinking,
                    config.chat_serialization,
                    config.chat_template_kwargs,
                )
                prompts.append(prompt)
                messages.append(row_messages)
                remapped_questions.append(remapped)
                token_rows.append(
                    [
                        int(value)
                        for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    ]
                )
            input_ids, attention_mask, _last_indices = tokenize_batch(tokenizer, prompts)
            width = int(input_ids.shape[1])
            source_physical: list[list[list[int]]] = []
            query_physical: list[list[list[int]]] = []
            span_audits: list[dict[str, Any]] = []
            for row, (qid, remapped) in enumerate(zip(cohort, remapped_questions)):
                qi = qid_index[qid]
                left_pad = width - len(token_rows[row])
                if input_ids[row, left_pad:].tolist() != token_rows[row]:
                    raise RuntimeError("Batched tokenization differs from audited tokenization")
                first_positions, first_audit = _option_line_positions(
                    tokenizer, prompts[row], questions[qid]
                )
                second_positions, second_audit = _option_line_positions(
                    tokenizer, prompts[row], remapped
                )
                ranks = arrays["rank_contents"][qi].astype(str).tolist()
                row_sources: list[list[int]] = []
                row_queries: list[list[int]] = []
                for rank, content in enumerate(ranks):
                    second_letter = mappings[qid]["original_to_new"][content]
                    sources = [left_pad + value for value in first_positions[content]]
                    queries = [left_pad + value for value in second_positions[second_letter]]
                    if not sources or not queries or max(sources) >= min(queries):
                        raise RuntimeError("Invalid matching source/query option lines")
                    row_sources.append(sources)
                    row_queries.append(queries)
                    arrays["source_position_counts"][qi, rank] = len(sources)
                    arrays["query_position_counts"][qi, rank] = len(queries)
                    arrays["cyclic_source_position_counts"][qi, rank] = len(
                        first_positions[ranks[(rank + 1) % 4]]
                    )
                source_physical.append(row_sources)
                query_physical.append(row_queries)
                arrays["prompt_hashes"][task_index, qi] = _hash_prompt(prompts[row])
                span_audits.append(
                    {"first": first_audit, "second": second_audit, "ranks": ranks}
                )

            natural = _aggregate_final_logits(
                _forward_final_logits(model, parts, input_ids, attention_mask),
                variant_ids,
            )
            if not np.all(np.isfinite(natural)):
                raise RuntimeError("Non-finite natural logits")
            arrays["natural_logits"][task_index, indices] = natural
            for row, qid in enumerate(cohort):
                arrays["trusted_natural_logits"][task_index, qid_index[qid]] = np.asarray(
                    trusted_tasks[task_index][qid]["aggregated_ad_logits"],
                    dtype=np.float32,
                )

            matching, matching_stats = _intervened_forward(
                model,
                parts,
                input_ids,
                attention_mask,
                variant_ids,
                layer_indices,
                source_physical,
                query_physical,
                cyclic_wrong=False,
            )
            cyclic, cyclic_stats = _intervened_forward(
                model,
                parts,
                input_ids,
                attention_mask,
                variant_ids,
                layer_indices,
                source_physical,
                query_physical,
                cyclic_wrong=True,
            )
            arrays["joint_matching_logits"][task_index, indices] = matching
            arrays["joint_cyclic_wrong_logits"][task_index, indices] = cyclic
            intervention_stats.extend(
                [
                    {"task": TASKS[task_index], "scenario": "matching", **matching_stats},
                    {"task": TASKS[task_index], "scenario": "cyclic_wrong", **cyclic_stats},
                ]
            )
            cohort_audit["tasks"][TASKS[task_index]] = {
                "rendered_prompt": prompts[0],
                "messages": messages[0],
                "prompt_hash": arrays["prompt_hashes"][task_index, indices[0]].item(),
                "rank_contents": span_audits[0]["ranks"],
                "first_option_lines": span_audits[0]["first"],
                "second_option_lines": span_audits[0]["second"],
                "attention_layers_one_based": list(ATTENTION_LAYERS_ONE_BASED),
                "matching_blockade": (
                    "Every complete 2P semantic option line is denied attention reads "
                    "from its complete semantically matching 1P option line."
                ),
                "cyclic_wrong_control": (
                    "For target rank Wr, deny reads from source W(r+1), wrapping W4 to W1."
                ),
            }
            rendered_prompts.append(prompts[0])

        _assert_prompt_pair(rendered_prompts[0], rendered_prompts[1])
        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"{MODEL_ID} {dataset} matching history: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.3f}",
            flush=True,
        )
        if not audit_path.exists():
            _atomic_write_json(audit_path, cohort_audit)

    metadata = {
        "experiment": _experiment_name(dataset),
        "dataset": dataset,
        "config": config.as_dict(),
        "n_questions": len(qids),
        "tasks": list(TASKS),
        "scenarios": list(SCENARIOS),
        "ranks": list(RANKS),
        "attention_layers_one_based": list(ATTENTION_LAYERS_ONE_BASED),
        "attention_architecture": {
            "type": "grouped-query causal self-attention",
            "query_heads": query_heads,
            "key_value_heads": key_value_heads,
            "gla_or_recurrent_layers": 0,
        },
        "source_span": "every token of each complete 1P option line",
        "receiver_span": "every token of the semantically corresponding complete 2P option line",
        "cyclic_control": "W1<-W2, W2<-W3, W3<-W4, W4<-W1 source blockade",
        "complete_model_forwards_per_cohort": 6,
        "complete_model_work": (
            "Per task: one natural forward, one joint all-four matching-line blockade, "
            "and one joint all-four cyclic-wrong-line blockade."
        ),
        "model_load_seconds": model_load_seconds,
        "elapsed_seconds_after_load": time.monotonic() - started,
        "cohort_seconds": durations,
        "intervention_stats": intervention_stats,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    _atomic_write_json(output_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--trusted-behavior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.remapping_plan,
        args.trusted_behavior,
        args.output_dir,
        args.max_cohorts,
    )


if __name__ == "__main__":
    main()
