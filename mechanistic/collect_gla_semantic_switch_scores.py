from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .collect_evaluation_gla_residual_writes import _aggregate_logits, _chunks
from .collect_gla_boundary_workspace_lenses import (
    BOUNDARIES,
    CONDITIONS,
    CONDITION_NAMES,
    GLABoundaryCollector,
    LENS_NAMES,
    _load_lens,
    _trusted_logits,
)
from .collect_remapped_feedback_factorial import _messages, _remap_question
from .config import ExperimentConfig
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import prompt_hash


# Human block numbers. These were frozen from the aggregate unrestricted-vocabulary
# result before any question-level association was calculated.
BLOCKS = (42, 43, 47)
LAYER_INDICES = tuple(block - 1 for block in BLOCKS)
GROUPS = {
    "evaluation": ("incorrect", "wrong", "failure", "failed", "reject", "rejected"),
    "replacement": (
        "replace",
        "replaced",
        "replacement",
        "instead",
        "alternative",
        "alternatives",
        "alternate",
    ),
    "retry": ("again", "retry", "try", "trying", "another", "second", "override"),
}


def _token_groups(tokenizer: Any) -> tuple[list[str], list[list[int]], dict[str, Any]]:
    names, ids, audit = [], [], {}
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    decoded = [tokenizer.decode([i]).strip().lower() for i in range(vocab_size)]
    for name, forms in GROUPS.items():
        group_ids = [i for i, token in enumerate(decoded) if token in set(forms)]
        if not group_ids:
            raise RuntimeError(f"No tokenizer entries found for semantic group {name}")
        names.append(name)
        ids.append(group_ids)
        audit[name] = [
            {"token_id": i, "decoded": tokenizer.decode([i]), "normalized": decoded[i]}
            for i in group_ids
        ]
    return names, ids, audit


def _initialize(path: Path, qids: list[str], width: int) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Checkpoint question IDs differ")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "blocks_one_based": np.asarray(BLOCKS, dtype=np.int16),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "semantic_scores": np.full(
            (2, 2, n, len(BLOCKS), len(BOUNDARIES), len(GROUPS)),
            np.nan,
            dtype=np.float32,
        ),
    }


def collect(
    config_path: Path,
    remapping_plan_path: Path,
    output_dir: Path,
    trusted_evaluation_path: Path,
    trusted_neutral_path: Path,
    lens_repo: str,
    j_filename: str,
    r_filename: str,
    max_cohorts: int | None,
    checkpoint_every_cohorts: int,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4:
        raise ValueError("Exact historical execution requires batch_size=4")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml" or config.attn_implementation != "sdpa":
        raise ValueError("Requires exact raw ChatML + SDPA regime")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    plan = json.loads(remapping_plan_path.read_text())
    plan_rows = {row["question_id"]: row for row in plan["rows"]}

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in ("A", "B", "C", "D")
    }
    group_names, group_ids, token_audit = _token_groups(tokenizer)
    width = int(parts.embedding.weight.shape[-1])
    device = model_input_device(parts)

    j_path, j_checkpoint = _load_lens(lens_repo, j_filename)
    r_path, r_checkpoint = _load_lens(lens_repo, r_filename)
    checkpoints = [j_checkpoint, r_checkpoint]
    gpu_transports = []
    for name, checkpoint in zip(LENS_NAMES, checkpoints):
        if int(checkpoint["d_model"]) != width:
            raise ValueError(f"{name} width mismatch")
        gpu_transports.append(
            {
                layer: checkpoint["J"][layer].to(device=device, dtype=torch.bfloat16)
                for layer in LAYER_INDICES
            }
        )

    head_rows = [parts.output_head.weight.detach()[ids].float() for ids in group_ids]
    head_bias = getattr(parts.output_head, "bias", None)
    if head_bias is not None:
        bias_rows = [head_bias.detach()[ids].float() for ids in group_ids]
    else:
        bias_rows = None

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids, width)
    qid_to_index = {qid: i for i, qid in enumerate(qids)}
    trusted = [
        _trusted_logits(trusted_evaluation_path),
        _trusted_logits(trusted_neutral_path),
    ]
    cohorts = list(_chunks(qids, config.batch_size))
    processed_this_run = 0
    started = time.perf_counter()
    audit = None

    for cohort in cohorts:
        indices = [qid_to_index[qid] for qid in cohort]
        if all(arrays["completed"][i] for i in indices):
            continue
        if any(arrays["completed"][i] for i in indices):
            raise RuntimeError("Checkpoint contains a partially completed cohort")
        if max_cohorts is not None and processed_this_run >= max_cohorts:
            break
        remapped = [
            _remap_question(questions[qid], plan_rows[qid]["new_to_original"])
            for qid in cohort
        ]
        batch_indices = np.asarray(indices)
        for condition_index, condition in enumerate(CONDITIONS):
            messages = [
                _messages(config, questions[qid], remapped_question, condition)
                for qid, remapped_question in zip(cohort, remapped)
            ]
            prompts = [
                render_chat(processor, row, config.disable_thinking, config.chat_serialization)
                for row in messages
            ]
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            collector = GLABoundaryCollector(parts, list(LAYER_INDICES), last_indices)
            try:
                with torch.inference_mode():
                    kwargs = {
                        "input_ids": input_ids.to(device),
                        "attention_mask": attention_mask.to(device),
                        "use_cache": False,
                        "return_dict": True,
                    }
                    try:
                        result = model(**kwargs, logits_to_keep=1)
                    except TypeError:
                        result = model(**kwargs)
                captured = collector.stacked(list(LAYER_INDICES))
            finally:
                collector.close()

            arrays["natural_logits"][condition_index, batch_indices] = _aggregate_logits(
                result, variant_ids
            )
            with torch.inference_mode():
                for lens_index, transports in enumerate(gpu_transports):
                    for block_slot, layer in enumerate(LAYER_INDICES):
                        values = captured[:, block_slot].to(device=device, dtype=torch.bfloat16)
                        flat = values.reshape(-1, width)
                        transported = flat @ transports[layer].T
                        normed = parts.final_norm(
                            transported.to(parts.final_norm.weight.dtype)
                        ).float()
                        scores = []
                        for group_slot, rows in enumerate(head_rows):
                            token_scores = normed @ rows.T
                            if bias_rows is not None:
                                token_scores = token_scores + bias_rows[group_slot]
                            scores.append(torch.logsumexp(token_scores, dim=-1))
                        grouped = torch.stack(scores, dim=-1).reshape(
                            len(cohort), len(BOUNDARIES), len(GROUPS)
                        )
                        arrays["semantic_scores"][
                            lens_index, condition_index, batch_indices, block_slot
                        ] = grouped.cpu().numpy()
            if audit is None:
                audit = {
                    "question_id": cohort[0],
                    "condition": condition,
                    "prompt_hash": prompt_hash(prompts[0]),
                    "prompt": prompts[0],
                }

        arrays["completed"][indices] = True
        processed_this_run += 1
        if (
            processed_this_run == 1
            or processed_this_run % checkpoint_every_cohorts == 0
            or int(arrays["completed"].sum()) == len(qids)
        ):
            atomic_save_npz(result_path, **arrays)
            print(
                f"Semantic scores: {int(arrays['completed'].sum())}/{len(qids)}; "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )

    atomic_save_npz(result_path, **arrays)
    complete = bool(np.all(arrays["completed"]))
    max_error = None
    if complete:
        max_error = max(
            float(
                np.max(
                    np.abs(
                        arrays["natural_logits"][condition]
                        - np.stack([trusted[condition][qid] for qid in qids])
                    )
                )
            )
            for condition in range(2)
        )
        if max_error != 0.0:
            raise RuntimeError(f"Natural logits failed exact reproduction: {max_error}")

    metadata = {
        "complete": complete,
        "completed_questions": int(arrays["completed"].sum()),
        "blocks_one_based": list(BLOCKS),
        "conditions": list(CONDITION_NAMES),
        "boundaries": list(BOUNDARIES),
        "lenses": list(LENS_NAMES),
        "semantic_groups": GROUPS,
        "resolved_group_tokens": token_audit,
        "score_definition": "LogSumExp of final-RMS-normalized workspace-lens scores over exact whole-token lexical variants in each frozen semantic family.",
        "complete_model_forward_passes": int(arrays["completed"].sum() // 4) * 2,
        "lens_transports": int(arrays["completed"].sum() // 4) * 2 * 2 * len(BLOCKS),
        "max_abs_natural_ad_logit_error_vs_trusted": max_error,
        "config": config.as_dict(),
        "remapping_plan": str(remapping_plan_path),
        "trusted_evaluation": str(trusted_evaluation_path),
        "trusted_neutral": str(trusted_neutral_path),
        "lens_files": {"J": j_path, "R": r_path},
        "prompt_audit": audit,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trusted-evaluation", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--lens-repo", default="camilablank/workspace-lenses")
    parser.add_argument("--j-filename", default="qwen3.6-27b/j-lens/lens.pt")
    parser.add_argument("--r-filename", default="qwen3.6-27b/r-lens/lens.pt")
    parser.add_argument("--max-cohorts", type=int)
    parser.add_argument("--checkpoint-every-cohorts", type=int, default=5)
    args = parser.parse_args()
    collect(
        args.config,
        args.remapping_plan,
        args.output_dir,
        args.trusted_evaluation,
        args.trusted_neutral,
        args.lens_repo,
        args.j_filename,
        args.r_filename,
        args.max_cohorts,
        args.checkpoint_every_cohorts,
    )


if __name__ == "__main__":
    main()
