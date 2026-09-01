from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .collect_evaluation_gla_residual_writes import (
    CONDITIONS,
    GLAOutputCollector,
    _aggregate_logits,
    _chunks,
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
from .run_evaluation_update_transplant import _locate_evaluation


def _substantive_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    keep = []
    for token_id in ids:
        decoded = tokenizer.decode([int(token_id)]).strip()
        if len(decoded) >= 2 and any(character.isalnum() for character in decoded):
            keep.append(int(token_id))
    return sorted(set(keep or [int(value) for value in ids]))


def _display(text: str) -> str:
    return text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def _initialize(path: Path, qids: list[str], layers: list[int]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint has different questions")
        if arrays["gla_layers_zero_based"].tolist() != layers:
            raise ValueError("Existing checkpoint has different GLA blocks")
        return arrays
    n, m = len(qids), len(layers)
    return {
        "question_ids": np.asarray(qids),
        "gla_layers_zero_based": np.asarray(layers, dtype=np.int16),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "delta_norm": np.full((n, m), np.nan, dtype=np.float32),
        "transported_norm": np.full((n, m), np.nan, dtype=np.float32),
        "jlens_option_max": np.full((n, m, 4), np.nan, dtype=np.float32),
        "jlens_option_mean": np.full((n, m, 4), np.nan, dtype=np.float32),
        "linear_option_max": np.full((n, m, 4), np.nan, dtype=np.float32),
        "linear_option_mean": np.full((n, m, 4), np.nan, dtype=np.float32),
        "cumulative_transported_norm": np.full((n, m), np.nan, dtype=np.float32),
        "jlens_cumulative_option_max": np.full((n, m, 4), np.nan, dtype=np.float32),
        "jlens_cumulative_option_mean": np.full((n, m, 4), np.nan, dtype=np.float32),
        "linear_cumulative_option_max": np.full((n, m, 4), np.nan, dtype=np.float32),
        "linear_cumulative_option_mean": np.full((n, m, 4), np.nan, dtype=np.float32),
    }


def collect(
    config_path: Path,
    remapping_plan_path: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
    output: Path,
    top_tokens_output: Path,
    lens_repo: str,
    lens_filename: str,
    max_questions: int | None,
    checkpoint_every_cohorts: int,
    audit_per_letter: int,
) -> None:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml" or config.attn_implementation != "sdpa":
        raise ValueError("Requires exact raw ChatML + SDPA regime")
    if config.batch_size != 4:
        raise ValueError("Requires exact historical batch size 4")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    if max_questions is not None:
        qids = qids[: int(max_questions)]
    remapping = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped_baseline = json.loads(remapped_baseline_path.read_text())["results"]

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    device = model_input_device(parts)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if len(layers) != 48 or max(layers) >= 63:
        raise RuntimeError(f"Unexpected GLA blocks: {layers}")

    lens_path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    jacobians = checkpoint["J"]
    if not set(layers) <= {int(value) for value in jacobians}:
        raise ValueError("JLens checkpoint is missing a GLA block")

    option_ids = {
        qid: [_substantive_ids(tokenizer, questions[qid]["options"][letter]) for letter in LETTERS]
        for qid in qids
    }
    union_ids = sorted({token_id for qid in qids for ids in option_ids[qid] for token_id in ids})
    union_index = {token_id: index for index, token_id in enumerate(union_ids)}
    option_union_indices = {
        qid: [[union_index[token_id] for token_id in ids] for ids in option_ids[qid]]
        for qid in qids
    }
    selected_head = parts.output_head.weight.detach()[union_ids].to(device)
    selected_bias = None
    if getattr(parts.output_head, "bias", None) is not None:
        selected_bias = parts.output_head.bias.detach()[union_ids].to(device)

    audit_qids = []
    for letter in LETTERS:
        pool = [
            qid for qid in qids
            if baseline[qid]["answer"] == letter
            and baseline[qid]["answer"] != remapped_baseline[qid]["answer_original_content"]
        ]
        audit_qids.extend(pool[:audit_per_letter])
    audit_set = set(audit_qids)
    top_rows: dict[str, Any] = {
        "definition": "Per-question unrestricted JLens vocabulary readout of the final-position Evaluation-minus-Neutral GLA output vector.",
        "question_ids": audit_qids,
        "questions": {
            qid: {
                "question": questions[qid]["question"],
                "options": questions[qid]["options"],
                "w1": baseline[qid]["answer"],
                "substantive_tokens": {
                    letter: [
                        {"token_id": int(token_id), "token": _display(tokenizer.decode([token_id]))}
                        for token_id in option_ids[qid][li]
                    ]
                    for li, letter in enumerate(LETTERS)
                },
            }
            for qid in audit_qids
        },
        "layers": {},
    }
    if top_tokens_output.exists():
        existing_top_rows = json.loads(top_tokens_output.read_text())
        if existing_top_rows.get("question_ids") != audit_qids:
            raise ValueError("Existing top-token checkpoint uses a different audit sample")
        top_rows = existing_top_rows

    arrays = _initialize(output, qids, layers)
    qid_to_index = {qid: index for index, qid in enumerate(qids)}
    output.parent.mkdir(parents=True, exist_ok=True)
    cohorts = list(_chunks(qids, config.batch_size))
    prompt_audit = None

    for cohort_index, cohort in enumerate(cohorts):
        indices = [qid_to_index[qid] for qid in cohort]
        if all(arrays["completed"][index] for index in indices):
            continue
        if any(arrays["completed"][index] for index in indices):
            raise RuntimeError("Checkpoint contains a partial historical cohort")
        captured_by_condition = []
        for condition_index, condition in enumerate(CONDITIONS):
            prompts, position_rows, prompt_rows = [], [], []
            for qid in cohort:
                question = questions[qid]
                remapped = _remap_question(question, remapping[qid]["new_to_original"])
                messages = _messages(config, question, remapped, condition)
                prompt = render_chat(processor, messages, config.disable_thinking, config.chat_serialization)
                located = _locate_evaluation(tokenizer, prompt, condition)
                prompts.append(prompt)
                prompt_rows.append((messages, prompt, located))
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            for row in range(len(cohort)):
                # GLAOutputCollector accepts an arbitrary list of positions; here only final.
                position_rows.append([int(last_indices[row])])
            collector = GLAOutputCollector(parts, layers, position_rows)
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
                captured = collector.stacked(layers)[:, :, 0].float()
            finally:
                collector.close()
            captured_by_condition.append(captured)
            arrays["natural_logits"][condition_index, indices] = _aggregate_logits(result, variant_ids)
            if prompt_audit is None:
                prompt_audit = {
                    "condition": condition,
                    "question_id": cohort[0],
                    "messages": prompt_rows[0][0],
                    "prompt": prompt_rows[0][1],
                }

        evaluation, neutral = captured_by_condition
        difference = evaluation - neutral
        arrays["delta_norm"][indices] = torch.linalg.vector_norm(difference, dim=-1).cpu().numpy()
        cumulative_transported = torch.zeros(
            (len(cohort), difference.shape[-1]), device=device, dtype=selected_head.dtype
        )
        for layer_slot, layer in enumerate(layers):
            J = jacobians[layer].to(device=device, dtype=selected_head.dtype)
            transported = difference[:, layer_slot].to(device=device, dtype=selected_head.dtype) @ J.T
            del J
            arrays["transported_norm"][indices, layer_slot] = (
                torch.linalg.vector_norm(transported.float(), dim=-1).cpu().numpy()
            )
            with torch.inference_mode():
                linear_scores = transported @ selected_head.T
                if selected_bias is not None:
                    linear_scores = linear_scores + selected_bias
                normed = parts.final_norm(transported.to(parts.final_norm.weight.dtype))
                jlens_scores = normed @ selected_head.T
                if selected_bias is not None:
                    jlens_scores = jlens_scores + selected_bias
                cumulative_transported = cumulative_transported + transported
                arrays["cumulative_transported_norm"][indices, layer_slot] = (
                    torch.linalg.vector_norm(cumulative_transported.float(), dim=-1).cpu().numpy()
                )
                cumulative_linear_scores = cumulative_transported @ selected_head.T
                if selected_bias is not None:
                    cumulative_linear_scores = cumulative_linear_scores + selected_bias
                cumulative_normed = parts.final_norm(
                    cumulative_transported.to(parts.final_norm.weight.dtype)
                )
                cumulative_jlens_scores = cumulative_normed @ selected_head.T
                if selected_bias is not None:
                    cumulative_jlens_scores = cumulative_jlens_scores + selected_bias
            linear_scores = linear_scores.float().cpu().numpy()
            jlens_scores = jlens_scores.float().cpu().numpy()
            cumulative_linear_scores = cumulative_linear_scores.float().cpu().numpy()
            cumulative_jlens_scores = cumulative_jlens_scores.float().cpu().numpy()
            for row, qid in enumerate(cohort):
                for option_index, token_indices in enumerate(option_union_indices[qid]):
                    arrays["linear_option_max"][indices[row], layer_slot, option_index] = float(
                        linear_scores[row, token_indices].max()
                    )
                    arrays["linear_option_mean"][indices[row], layer_slot, option_index] = float(
                        linear_scores[row, token_indices].mean()
                    )
                    arrays["jlens_option_max"][indices[row], layer_slot, option_index] = float(
                        jlens_scores[row, token_indices].max()
                    )
                    arrays["jlens_option_mean"][indices[row], layer_slot, option_index] = float(
                        jlens_scores[row, token_indices].mean()
                    )
                    arrays["linear_cumulative_option_max"][indices[row], layer_slot, option_index] = float(
                        cumulative_linear_scores[row, token_indices].max()
                    )
                    arrays["linear_cumulative_option_mean"][indices[row], layer_slot, option_index] = float(
                        cumulative_linear_scores[row, token_indices].mean()
                    )
                    arrays["jlens_cumulative_option_max"][indices[row], layer_slot, option_index] = float(
                        cumulative_jlens_scores[row, token_indices].max()
                    )
                    arrays["jlens_cumulative_option_mean"][indices[row], layer_slot, option_index] = float(
                        cumulative_jlens_scores[row, token_indices].mean()
                    )
            audit_rows = [row for row, qid in enumerate(cohort) if qid in audit_set]
            if audit_rows:
                with torch.inference_mode():
                    full_scores = parts.output_head(normed[audit_rows]).float()
                    pos_values, pos_ids = torch.topk(full_scores, k=12, dim=-1)
                    neg_values, neg_ids = torch.topk(-full_scores, k=12, dim=-1)
                layer_entry = top_rows["layers"].setdefault(str(layer + 1), {})
                for local, row in enumerate(audit_rows):
                    qid = cohort[row]
                    layer_entry[qid] = {
                        "positive": [
                            {"token_id": int(token_id), "token": _display(tokenizer.decode([int(token_id)])), "score": float(score)}
                            for score, token_id in zip(pos_values[local].cpu(), pos_ids[local].cpu())
                        ],
                        "negative": [
                            {"token_id": int(token_id), "token": _display(tokenizer.decode([int(token_id)])), "score": float(-score)}
                            for score, token_id in zip(neg_values[local].cpu(), neg_ids[local].cpu())
                        ],
                    }
            del transported, normed, cumulative_normed
            del linear_scores, jlens_scores, cumulative_linear_scores, cumulative_jlens_scores

        arrays["completed"][indices] = True
        done = cohort_index + 1
        if done == 1 or done % checkpoint_every_cohorts == 0 or done == len(cohorts):
            atomic_save_npz(output, **arrays)
            top_tokens_output.parent.mkdir(parents=True, exist_ok=True)
            top_tokens_output.write_text(json.dumps(top_rows, indent=2, ensure_ascii=False) + "\n")
            print(f"question-specific GLA JLens: {done}/{len(cohorts)} cohorts", flush=True)

    metadata = {
        "config": str(config_path),
        "remapping_plan": str(remapping_plan_path),
        "baseline": str(baseline_path),
        "remapped_baseline": str(remapped_baseline_path),
        "questions": len(qids),
        "cohorts": len(cohorts),
        "complete_model_forward_passes": len(cohorts) * 2,
        "gla_blocks_human": [layer + 1 for layer in layers],
        "lens_repo": lens_repo,
        "lens_filename": lens_filename,
        "option_token_union_size": len(union_ids),
        "prompt_audit": prompt_audit,
        "software": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-tokens-output", type=Path, required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument("--lens-filename", default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--checkpoint-every-cohorts", type=int, default=5)
    parser.add_argument("--audit-per-letter", type=int, default=2)
    args = parser.parse_args()
    collect(args.config, args.remapping_plan, args.baseline, args.remapped_baseline, args.output, args.top_tokens_output, args.lens_repo, args.lens_filename, args.max_questions, args.checkpoint_every_cohorts, args.audit_per_letter)


if __name__ == "__main__":
    main()
