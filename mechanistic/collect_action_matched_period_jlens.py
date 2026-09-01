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
from .collect_evaluation_gla_residual_writes import _aggregate_logits, _chunks
from .collect_remapped_feedback_factorial import _messages, _remap_question
from .config import ExperimentConfig
from .io import atomic_save_npz
from .jlens_collect import _top_tokens, _unembed_vector
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import FACTORIAL_FEEDBACK, prompt_hash


CONDITIONS = ("incorrect_again", "lost_again")
DISPLAY_CONDITIONS = ("Evaluation", "Matched Neutral")
ANCHORS = ("evaluation_period", "action_period")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _offsets(tokenizer: Any, prompt: str) -> list[tuple[int, int]]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    return [(int(left), int(right)) for left, right in encoded["offset_mapping"]]


def _period_positions(tokenizer: Any, prompt: str, condition: str) -> tuple[list[int], dict[str, Any]]:
    feedback = FACTORIAL_FEEDBACK[condition]
    start = prompt.find(feedback)
    if start < 0 or prompt.find(feedback, start + 1) >= 0:
        raise RuntimeError(f"Expected exactly one feedback clause for {condition}")
    offsets = _offsets(tokenizer, prompt)
    period_chars = [start + index for index, char in enumerate(feedback) if char == "."]
    if len(period_chars) != 2:
        raise RuntimeError(f"Expected two periods in {feedback!r}")
    positions = []
    for character in period_chars:
        hits = [
            index for index, (left, right) in enumerate(offsets)
            if right > left and left <= character < right
        ]
        if len(hits) != 1:
            raise RuntimeError(f"Expected one token overlapping feedback period: {hits}")
        positions.append(hits[0])
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = [tokenizer.decode([ids[position]]) for position in positions]
    if any(token.strip() != "." for token in tokens):
        raise RuntimeError(f"Unexpected period tokens: {tokens!r}")
    return positions, {
        "feedback": feedback,
        "character_offsets": period_chars,
        "unpadded_token_positions_zero_based": positions,
        "decoded_tokens": tokens,
    }


class BatchPositionCollector:
    """Capture two question-specific positions after every transformer block."""

    def __init__(self, layers: Any, positions: list[list[int]]):
        self.positions = positions
        self.values: list[Any] = [None] * len(layers)
        self.handles = [layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(layers)]

    def _hook(self, index: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = output[0] if isinstance(output, (tuple, list)) else output
            rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
            cols = torch.as_tensor(self.positions, device=hidden.device)
            self.values[index] = hidden[rows, cols].detach().to("cpu", dtype=torch.float16)
        return capture

    def stacked(self):
        import torch

        if any(value is None for value in self.values):
            missing = [i for i, value in enumerate(self.values) if value is None]
            raise RuntimeError(f"Failed to capture post-block residuals: {missing}")
        return torch.stack(self.values, dim=1)  # batch, layer, anchor, width

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _initialize(path: Path, qids: list[str], width: int) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Checkpoint question IDs differ")
        return arrays
    return {
        "question_ids": np.asarray(qids),
        "completed": np.zeros(len(qids), dtype=bool),
        "natural_logits": np.full((2, len(qids), 4), np.nan, dtype=np.float32),
        "normalized_transport_sum": np.zeros((2, 2, 64, width), dtype=np.float32),
        "bare_ad_scores": np.full((2, len(qids), 2, 64, 4), np.nan, dtype=np.float16),
    }


def _trusted_logits(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text())["results"]
    return {
        qid: np.asarray(row["aggregated_ad_logits"], dtype=np.float32)
        for qid, row in payload.items()
    }


def _baseline_rank_order(
    path: Path,
    qids: list[str],
    plan_rows: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    results = json.loads(path.read_text())["results"]
    order = np.empty((len(qids), 4), dtype=np.int64)
    for index, qid in enumerate(qids):
        row = results[qid]
        if "aggregated_ad_logits" in row:
            evidence = np.asarray(row["aggregated_ad_logits"], dtype=np.float64)
        else:
            probabilities = row["probs"]
            evidence = np.asarray([
                np.log(max(float(probabilities.get(letter, 0.0)), 1e-30))
                for letter in LETTERS
            ])
        winner = LETTERS.index(row.get("answer", row.get("subject_answer")))
        rest = [candidate for candidate in range(4) if candidate != winner]
        original_order = [winner] + sorted(
            rest, key=lambda candidate: evidence[candidate], reverse=True
        )
        if plan_rows is None:
            order[index] = original_order
        else:
            original_to_new = plan_rows[qid]["original_to_new"]
            order[index] = [
                LETTERS.index(original_to_new[LETTERS[candidate]])
                for candidate in original_order
            ]
    return order


def _insert_rank_rows(row: dict[str, list[dict[str, Any]]], values: np.ndarray) -> None:
    for rank, value in enumerate(values, 1):
        item = {
            "token_id": -rank,
            "token": f"[Baseline rank {rank}]",
            "score": float(value),
            "tracked": True,
        }
        row["top" if value >= 0 else "bottom"].append(item)
    row["top"].sort(key=lambda item: item["score"], reverse=True)
    row["bottom"].sort(key=lambda item: item["score"])


def _build_readouts(
    arrays: dict[str, np.ndarray],
    tokenizer: Any,
    parts: Any,
    output: Path,
    baseline_rank_results: Path,
    remapping_plan: Path,
    top_k: int,
) -> None:
    import torch

    n = int(arrays["completed"].sum())
    means = arrays["normalized_transport_sum"] / n
    qids = arrays["question_ids"].astype(str).tolist()
    plan = json.loads(remapping_plan.read_text())
    plan_rows = {row["question_id"]: row for row in plan["rows"]}
    rank_order = _baseline_rank_order(baseline_rank_results, qids, plan_rows)
    scores = arrays["bare_ad_scores"].astype(np.float32)
    aligned = np.take_along_axis(
        scores,
        np.broadcast_to(rank_order[None, :, None, None, :], scores.shape),
        axis=-1,
    )
    rank_means = aligned.mean(axis=1)
    document: dict[str, Any] = {
        "definition": (
            "Complete post-block residual streams at the evaluation-closing and shared "
            "action-closing periods. Each question is JLens-transported and final-RMS-normalized "
            "before averaging. Evaluation-minus-Neutral contrasts subtract the decoded means."
        ),
        "questions": n,
        "anchors": list(ANCHORS),
        "positions": {},
    }
    with torch.inference_mode():
        for layer in range(64):
            for anchor_index, anchor in enumerate(ANCHORS):
                for condition_index, condition in enumerate(("incorrect", "neutral")):
                    logits = _unembed_vector(
                        parts,
                        torch.from_numpy(means[condition_index, anchor_index, layer]),
                        include_bias=True,
                    )
                    row = _top_tokens(tokenizer, logits, top_k)
                    _insert_rank_rows(row, rank_means[condition_index, anchor_index, layer])
                    document["positions"][f"{condition}/{anchor}/L{layer}"] = row
                logits = _unembed_vector(
                    parts,
                    torch.from_numpy(
                        means[0, anchor_index, layer] - means[1, anchor_index, layer]
                    ),
                    include_bias=False,
                )
                row = _top_tokens(tokenizer, logits, top_k)
                _insert_rank_rows(row, rank_means[0, anchor_index, layer] - rank_means[1, anchor_index, layer])
                document["positions"][f"game_minus_neutral/{anchor}/L{layer}"] = row
    _write_json(output, document)


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
        raise ValueError("Requires exact raw ChatML + SDPA regime")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    plan = json.loads(remapping_plan_path.read_text())
    plan_rows = {row["question_id"]: row for row in plan["rows"]}
    if not set(qids) <= set(plan_rows):
        raise ValueError("Remapping plan is incomplete")

    lens_path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    if sorted(int(layer) for layer in checkpoint["J"]) != list(range(63)):
        raise ValueError("Unexpected JLens layer coverage")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    device = model_input_device(parts)
    width = int(parts.embedding.weight.shape[-1])
    if int(checkpoint["d_model"]) != width:
        raise ValueError("JLens/model width mismatch")
    transports = {
        layer: checkpoint["J"][layer].to(device=device, dtype=torch.bfloat16)
        for layer in range(63)
    }
    del checkpoint

    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]}) for letter in LETTERS
    }
    bare_ids = [resolved[letter][0][1] for letter in LETTERS]
    bare_rows = parts.output_head.weight.detach()[bare_ids].float()
    bare_bias = getattr(parts.output_head, "bias", None)
    if bare_bias is not None:
        bare_bias = bare_bias.detach()[bare_ids].float()

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids, width)
    qid_to_index = {qid: index for index, qid in enumerate(qids)}
    trusted = [_trusted_logits(trusted_evaluation_path), _trusted_logits(trusted_neutral_path)]
    if any(not set(qids) <= set(rows) for rows in trusted):
        raise ValueError("Trusted natural results are incomplete")

    cohorts = list(_chunks(qids, config.batch_size))
    processed = 0
    started = time.perf_counter()
    audit: dict[str, Any] | None = None
    for cohort in cohorts:
        indices = [qid_to_index[qid] for qid in cohort]
        if all(arrays["completed"][index] for index in indices):
            continue
        if any(arrays["completed"][index] for index in indices):
            raise RuntimeError("Checkpoint contains a partially completed cohort")
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
            unpadded_positions, audits = zip(*[
                _period_positions(tokenizer, prompt, condition) for prompt in prompts
            ])
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            prompt_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
            padded_positions = [
                [position + input_ids.shape[1] - length for position in positions]
                for positions, length in zip(unpadded_positions, prompt_lengths)
            ]
            collector = BatchPositionCollector(parts.layers, padded_positions)
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

            batch_indices = np.asarray(indices)
            arrays["natural_logits"][condition_index, batch_indices] = _aggregate_logits(
                result, variant_ids
            )
            with torch.inference_mode():
                for layer in range(64):
                    values = captured[:, layer].to(device=device, dtype=torch.bfloat16)
                    flat = values.reshape(-1, width)
                    transported = flat if layer == 63 else flat @ transports[layer].T
                    normed = parts.final_norm(
                        transported.to(parts.final_norm.weight.dtype)
                    ).float().reshape(len(cohort), 2, width)
                    arrays["normalized_transport_sum"][condition_index, :, layer] += (
                        normed.sum(dim=0).cpu().numpy()
                    )
                    ad = normed @ bare_rows.T
                    if bare_bias is not None:
                        ad = ad + bare_bias
                    arrays["bare_ad_scores"][condition_index, batch_indices, :, layer] = (
                        ad.cpu().to(torch.float16).numpy()
                    )
                    del values, flat, transported, normed, ad

            if audit is None:
                audit = {
                    "anchors": list(ANCHORS),
                    "conditions": ["incorrect", "neutral"],
                    "trials": {},
                }
            if processed == 0:
                for qid, prompt, message, row_audit in zip(cohort, prompts, messages, audits):
                    audit["trials"][f"{'incorrect' if condition_index == 0 else 'neutral'}/{qid}"] = {
                        **row_audit,
                        "prompt_hash": prompt_hash(prompt),
                        "rendered_prompt": prompt,
                        "messages": message,
                        "tokens": row_audit["decoded_tokens"],
                    }

        arrays["completed"][indices] = True
        processed += 1
        if processed == 1 or processed % checkpoint_every_cohorts == 0 or np.all(arrays["completed"]):
            atomic_save_npz(result_path, **arrays)
            print(
                f"period JLens: {int(arrays['completed'].sum())}/{len(qids)} questions; "
                f"{time.perf_counter() - started:.1f}s this command",
                flush=True,
            )

    atomic_save_npz(result_path, **arrays)
    complete = bool(np.all(arrays["completed"]))
    maximum_error = None
    if complete:
        errors = []
        for condition_index in range(2):
            expected = np.stack([trusted[condition_index][qid] for qid in qids])
            errors.append(float(np.max(np.abs(arrays["natural_logits"][condition_index] - expected))))
        maximum_error = max(errors)
        if maximum_error != 0.0:
            raise RuntimeError(f"Natural A-D logits did not reproduce exactly: {maximum_error}")
        _build_readouts(
            arrays, tokenizer, parts, output_dir / "top_tokens_with_baseline_ranks.json",
            baseline_rank_results, remapping_plan_path, top_k,
        )
    if audit is not None:
        _write_json(output_dir / "position_audit.json", audit)
    metadata = {
        "config": config.as_dict(),
        "remapping_plan": str(remapping_plan_path),
        "trusted_evaluation": str(trusted_evaluation_path),
        "trusted_neutral": str(trusted_neutral_path),
        "baseline_rank_results": str(baseline_rank_results),
        "n_questions": len(qids),
        "completed_questions": int(arrays["completed"].sum()),
        "complete": complete,
        "conditions": list(CONDITIONS),
        "condition_names": list(DISPLAY_CONDITIONS),
        "anchors": list(ANCHORS),
        "complete_model_forward_passes": int(arrays["completed"].sum() // 4) * 2,
        "lens_transports": int(arrays["completed"].sum() // 4) * 2 * 63,
        "batch_rows_per_forward": config.batch_size,
        "max_abs_natural_ad_logit_error_vs_trusted": maximum_error,
        "lens": {"repo": lens_repo, "filename": lens_filename, "local_path": lens_path},
        "layer_alignment": "JLens maps post-block residuals 1--63; readout 64 is natural.",
        "averaging": (
            "Each complete residual is JLens-transported and final-RMS-normalized separately, "
            "then averaged across all 500 questions. Contrasts subtract separately decoded means."
        ),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
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
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--max-cohorts", type=int)
    parser.add_argument("--checkpoint-every-cohorts", type=int, default=5)
    args = parser.parse_args()
    collect(
        args.config, args.remapping_plan, args.output_dir,
        args.trusted_evaluation, args.trusted_neutral, args.baseline_rank_results,
        args.lens_repo, args.lens_filename, args.top_k, args.max_cohorts,
        args.checkpoint_every_cohorts,
    )


if __name__ == "__main__":
    main()
