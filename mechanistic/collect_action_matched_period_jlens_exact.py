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

from . import LETTERS
from .collect_action_matched_period_jlens import (
    ANCHORS,
    BatchPositionCollector,
    CONDITIONS,
    DISPLAY_CONDITIONS,
    _build_readouts,
    _period_positions,
    _trusted_logits,
    _write_json,
)
from .collect_evaluation_gla_residual_writes import _aggregate_logits, _chunks
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


def _open_or_create(path: Path, shape: tuple[int, ...], dtype: Any):
    if path.exists():
        array = np.lib.format.open_memmap(path, mode="r+")
        if tuple(array.shape) != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"Incompatible memmap {path}: {array.shape}, {array.dtype}")
        return array
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def collect(
    config_path: Path,
    remapping_plan_path: Path,
    output_dir: Path,
    trusted_evaluation_path: Path,
    trusted_neutral_path: Path,
    baseline_rank_results: Path,
    lens_repo: str,
    lens_filename: str,
    top_k: int,
    max_cohorts: int | None,
    checkpoint_every_cohorts: int,
    transform_batch_size: int,
    build_partial: bool,
) -> None:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4:
        raise ValueError("Exact historical execution requires batch_size=4")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml" or config.attn_implementation != "sdpa":
        raise ValueError("Requires exact raw ChatML + SDPA")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    plan = json.loads(remapping_plan_path.read_text())
    plan_rows = {row["question_id"]: row for row in plan["rows"]}
    if not set(qids) <= set(plan_rows):
        raise ValueError("Remapping plan is incomplete")

    # Crucially, neither the JLens checkpoint nor any Jacobian matrix is loaded
    # before natural residual collection and exact numerical validation.
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    device = model_input_device(parts)
    width = int(parts.embedding.weight.shape[-1])
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]}) for letter in LETTERS
    }
    bare_ids = [resolved[letter][0][1] for letter in LETTERS]

    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = _open_or_create(
        output_dir / "position_residuals.npy",
        (2, len(qids), 64, 2, width),
        np.float16,
    )
    natural_logits = _open_or_create(
        output_dir / "natural_logits.npy", (2, len(qids), 4), np.float32
    )
    completed_path = output_dir / "completed.npy"
    completed = (
        np.load(completed_path)
        if completed_path.exists()
        else np.zeros(len(qids), dtype=bool)
    )
    if completed.shape != (len(qids),):
        raise ValueError("Incompatible completion checkpoint")
    qid_to_index = {qid: index for index, qid in enumerate(qids)}
    trusted = [_trusted_logits(trusted_evaluation_path), _trusted_logits(trusted_neutral_path)]
    audit: dict[str, Any] = {"anchors": list(ANCHORS), "conditions": ["incorrect", "neutral"], "trials": {}}
    audit_path = output_dir / "position_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())

    processed = 0
    started = time.perf_counter()
    for cohort in _chunks(qids, config.batch_size):
        indices = [qid_to_index[qid] for qid in cohort]
        if all(completed[index] for index in indices):
            continue
        if any(completed[index] for index in indices):
            raise RuntimeError("Partially completed cohort")
        if max_cohorts is not None and processed >= max_cohorts:
            break
        remapped = [
            _remap_question(questions[qid], plan_rows[qid]["new_to_original"])
            for qid in cohort
        ]
        for condition_index, condition in enumerate(CONDITIONS):
            messages = [
                _messages(config, questions[qid], remapped_question, condition)
                for qid, remapped_question in zip(cohort, remapped)
            ]
            prompts = [
                render_chat(processor, message, config.disable_thinking, config.chat_serialization)
                for message in messages
            ]
            located, audits = zip(*[
                _period_positions(tokenizer, prompt, condition) for prompt in prompts
            ])
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
            lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
            positions = [
                [position + input_ids.shape[1] - length for position in row]
                for row, length in zip(located, lengths)
            ]
            collector = BatchPositionCollector(parts.layers, positions)
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
                captured = collector.stacked()
            finally:
                collector.close()
            residuals[condition_index, indices] = captured.numpy()
            natural_logits[condition_index, indices] = _aggregate_logits(result, variant_ids)
            if not audit["trials"]:
                for qid, prompt, message, row_audit in zip(cohort, prompts, messages, audits):
                    audit["trials"][f"{'incorrect' if condition_index == 0 else 'neutral'}/{qid}"] = {
                        **row_audit,
                        "prompt_hash": prompt_hash(prompt),
                        "rendered_prompt": prompt,
                        "messages": message,
                        "tokens": row_audit["decoded_tokens"],
                    }
            elif processed == 0 and condition_index == 1:
                for qid, prompt, message, row_audit in zip(cohort, prompts, messages, audits):
                    audit["trials"][f"neutral/{qid}"] = {
                        **row_audit,
                        "prompt_hash": prompt_hash(prompt),
                        "rendered_prompt": prompt,
                        "messages": message,
                        "tokens": row_audit["decoded_tokens"],
                    }
        completed[indices] = True
        processed += 1
        if processed == 1 or processed % checkpoint_every_cohorts == 0 or np.all(completed):
            residuals.flush(); natural_logits.flush(); np.save(completed_path, completed)
            _write_json(audit_path, audit)
            print(
                f"exact residuals: {int(completed.sum())}/{len(qids)} questions; "
                f"{time.perf_counter() - started:.1f}s this command",
                flush=True,
            )

    residuals.flush(); natural_logits.flush(); np.save(completed_path, completed)
    complete = bool(np.all(completed))
    count = int(completed.sum())
    errors = []
    for condition_index in range(2):
        expected = np.stack([trusted[condition_index][qid] for qid in qids[:count]])
        errors.append(float(np.max(np.abs(np.asarray(natural_logits[condition_index, :count]) - expected))))
    maximum_error = max(errors) if errors else None
    if maximum_error != 0.0:
        raise RuntimeError(f"Natural A-D logits failed exact reproduction: {maximum_error}")
    print(f"natural validation exact for {count} questions", flush=True)
    if not complete and not build_partial:
        return

    lens_path = hf_hub_download(repo_id=lens_repo, filename=lens_filename)
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    if sorted(int(layer) for layer in checkpoint["J"]) != list(range(63)):
        raise ValueError("Unexpected JLens layers")
    if int(checkpoint["d_model"]) != width:
        raise ValueError("JLens width mismatch")
    normalized_sum = np.zeros((2, 2, 64, width), dtype=np.float32)
    bare_scores = np.full((2, count, 2, 64, 4), np.nan, dtype=np.float16)
    bare_rows = parts.output_head.weight.detach()[bare_ids].float()
    bare_bias = getattr(parts.output_head, "bias", None)
    if bare_bias is not None:
        bare_bias = bare_bias.detach()[bare_ids].float()
    transform_started = time.perf_counter()
    with torch.inference_mode():
        for layer in range(64):
            J = checkpoint["J"][layer].to(device=device, dtype=torch.bfloat16) if layer < 63 else None
            for condition_index in range(2):
                for start in range(0, count, transform_batch_size):
                    stop = min(start + transform_batch_size, count)
                    values = np.asarray(residuals[condition_index, start:stop, layer]).copy()
                    tensor = torch.from_numpy(values.reshape(-1, width)).to(device=device, dtype=torch.bfloat16)
                    transported = tensor if J is None else tensor @ J.T
                    normed = parts.final_norm(
                        transported.to(parts.final_norm.weight.dtype)
                    ).float().reshape(stop - start, 2, width)
                    normalized_sum[condition_index, :, layer] += normed.sum(dim=0).cpu().numpy()
                    ad = normed @ bare_rows.T
                    if bare_bias is not None:
                        ad = ad + bare_bias
                    bare_scores[condition_index, start:stop, :, layer] = ad.cpu().to(torch.float16).numpy()
            if J is not None:
                del J
            if layer == 0 or (layer + 1) % 8 == 0 or layer == 63:
                print(f"JLens transform: {layer + 1}/64 readouts", flush=True)

    arrays = {
        "question_ids": np.asarray(qids[:count]),
        "completed": np.ones(count, dtype=bool),
        "natural_logits": np.asarray(natural_logits[:, :count]).copy(),
        "normalized_transport_sum": normalized_sum,
        "bare_ad_scores": bare_scores,
    }
    atomic_save_npz(output_dir / "results.npz", **arrays)
    _build_readouts(
        arrays, tokenizer, parts, output_dir / "top_tokens_with_baseline_ranks.json",
        baseline_rank_results, remapping_plan_path, top_k,
    )
    metadata = {
        "config": config.as_dict(),
        "remapping_plan": str(remapping_plan_path),
        "trusted_evaluation": str(trusted_evaluation_path),
        "trusted_neutral": str(trusted_neutral_path),
        "baseline_rank_results": str(baseline_rank_results),
        "n_questions": len(qids),
        "completed_questions": count,
        "complete": complete,
        "conditions": list(CONDITIONS),
        "condition_names": list(DISPLAY_CONDITIONS),
        "anchors": list(ANCHORS),
        "complete_model_forward_passes": (count // 4) * 2,
        "lens_transports": 2 * 64 * ((count + transform_batch_size - 1) // transform_batch_size),
        "batch_rows_per_forward": config.batch_size,
        "max_abs_natural_ad_logit_error_vs_trusted": maximum_error,
        "lens": {"repo": lens_repo, "filename": lens_filename, "local_path": lens_path},
        "layer_alignment": "JLens maps post-block residuals 1--63; readout 64 is natural.",
        "numerical_isolation": (
            "All natural forwards and residual captures finished and reproduced trusted A-D logits "
            "bit-exactly before the JLens checkpoint or any Jacobian matrix was loaded."
        ),
        "residual_collection_seconds": time.perf_counter() - started,
        "lens_transform_seconds": time.perf_counter() - transform_started,
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "platform": platform.platform(),
    }
    _write_json(output_dir / "run_metadata.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trusted-evaluation", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--baseline-rank-results", type=Path, required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument("--lens-filename", default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--max-cohorts", type=int)
    parser.add_argument("--checkpoint-every-cohorts", type=int, default=5)
    parser.add_argument("--transform-batch-size", type=int, default=32)
    parser.add_argument("--build-partial", action="store_true")
    args = parser.parse_args()
    collect(
        args.config, args.remapping_plan, args.output_dir,
        args.trusted_evaluation, args.trusted_neutral, args.baseline_rank_results,
        args.lens_repo, args.lens_filename, args.top_k, args.max_cohorts,
        args.checkpoint_every_cohorts, args.transform_batch_size, args.build_partial,
    )


if __name__ == "__main__":
    main()
