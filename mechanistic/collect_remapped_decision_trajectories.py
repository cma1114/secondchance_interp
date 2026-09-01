from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .collect_remapped_behavior import _messages, _remap_question
from .config import ExperimentConfig
from .modeling import (
    ResidualCollector,
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)


CONDITIONS = ("incorrect", "neutral")
LETTERS = "ABCD"


def _chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def collect(
    config_path: Path,
    plan_path: Path,
    output: Path,
    lens_repo: str,
    lens_filename: str,
) -> None:
    import torch
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    if config.max_questions is not None:
        qids = qids[: int(config.max_questions)]
    plan = json.loads(plan_path.read_text())
    plan_rows = {row["question_id"]: row for row in plan["rows"]}
    if not set(qids) <= set(plan_rows):
        raise ValueError("Remapping plan is missing questions")

    output.mkdir(parents=True, exist_ok=True)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = [[token_id for _, token_id in resolved[letter]] for letter in LETTERS]
    flat_variant_ids = [token_id for group in variant_ids for token_id in group]
    group_slices = []
    cursor = 0
    for group in variant_ids:
        group_slices.append(slice(cursor, cursor + len(group)))
        cursor += len(group)

    lens_path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    jacobians = checkpoint["J"]
    if sorted(int(layer) for layer in jacobians) != list(range(63)):
        raise ValueError("Unexpected JLens layer coverage")

    n_questions = len(qids)
    width = int(parts.embedding.weight.shape[-1])
    residual_path = output / "decision_residuals.tmp.npy"
    completed_path = output / "completed.npy"
    expected_shape = (len(CONDITIONS), n_questions, 64, width)
    if residual_path.exists() and completed_path.exists():
        residuals = np.lib.format.open_memmap(residual_path, mode="r+")
        completed = np.load(completed_path)
        if tuple(residuals.shape) != expected_shape or completed.shape != (2, n_questions):
            raise ValueError("Incompatible checkpoint")
    else:
        residuals = np.lib.format.open_memmap(
            residual_path, mode="w+", dtype=np.float16, shape=expected_shape
        )
        completed = np.zeros((2, n_questions), dtype=bool)

    direct_logits = np.full((2, n_questions, 4), np.nan, dtype=np.float32)
    direct_path = output / "direct_logits.npy"
    if direct_path.exists():
        direct_logits = np.load(direct_path)

    device = model_input_device(parts)
    for ci, condition in enumerate(CONDITIONS):
        pending = [index for index in range(n_questions) if not completed[ci, index]]
        for batch_number, indices in enumerate(_chunks(pending, config.batch_size), 1):
            batch_qids = [qids[index] for index in indices]
            remapped = [
                _remap_question(questions[qid], plan_rows[qid]["new_to_original"])
                for qid in batch_qids
            ]
            prompts = [
                render_chat(
                    processor,
                    _messages(config, questions[qid], remapped_question, condition),
                    config.disable_thinking,
                    config.chat_serialization,
                )
                for qid, remapped_question in zip(batch_qids, remapped)
            ]
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            collector = ResidualCollector(parts, last_indices)
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
                captured = collector.stacked()[:, 1:].numpy()
            finally:
                collector.close()
            residuals[ci, indices] = captured

            logits = result.logits.detach().float().cpu()
            if logits.shape[1] == 1:
                final = logits[:, 0]
            else:
                final = logits[np.arange(len(indices)), last_indices]
            for bi, qi in enumerate(indices):
                direct_logits[ci, qi] = np.asarray([
                    torch.logsumexp(final[bi, group], dim=0).item()
                    for group in variant_ids
                ])
                completed[ci, qi] = True
            residuals.flush()
            np.save(completed_path, completed)
            np.save(direct_path, direct_logits)
            done = int(completed[ci].sum())
            if batch_number == 1 or done % 40 == 0 or done == n_questions:
                print(f"{condition}: {done}/{n_questions}", flush=True)

    token_rows = parts.output_head.weight.detach()[flat_variant_ids].float()
    token_bias = getattr(parts.output_head, "bias", None)
    if token_bias is not None:
        token_bias = token_bias.detach()[flat_variant_ids].float()
    jlens_scores = np.empty((2, n_questions, 64, 4), dtype=np.float32)
    logit_lens_scores = np.empty_like(jlens_scores)
    score_batch = 32

    def decode(values, transform):
        tensor = torch.from_numpy(np.asarray(values).copy()).to(device, dtype=torch.float16)
        if transform is not None:
            tensor = tensor @ transform.T
        normed = parts.final_norm(tensor.to(parts.final_norm.weight.dtype)).float()
        logits = normed @ token_rows.T
        if token_bias is not None:
            logits += token_bias
        return torch.stack([
            torch.logsumexp(logits[:, group_slice], dim=-1)
            for group_slice in group_slices
        ], dim=-1).cpu().numpy()

    with torch.inference_mode():
        for layer in range(64):
            transform = (
                jacobians[layer].to(device, dtype=torch.float16)
                if layer < 63 else None
            )
            for ci in range(2):
                for start in range(0, n_questions, score_batch):
                    stop = min(start + score_batch, n_questions)
                    values = residuals[ci, start:stop, layer]
                    logit_lens_scores[ci, start:stop, layer] = decode(values, None)
                    jlens_scores[ci, start:stop, layer] = decode(values, transform)
            if transform is not None:
                del transform
            if layer == 0 or (layer + 1) % 8 == 0 or layer == 63:
                print(f"decoded readout {layer + 1}/64", flush=True)

    max_error = float(np.max(np.abs(jlens_scores[:, :, -1] - direct_logits)))
    # Residuals are cached as float16, so re-unembedding the natural final
    # readout need not reproduce live float32 logits exactly.  The live logits
    # are saved separately and validated against the trusted behavioral run.
    if max_error > 0.10:
        raise RuntimeError(f"L64 reconstruction error is too large: {max_error}")
    _atomic_npz(
        output / "scores.npz",
        question_ids=np.asarray(qids),
        conditions=np.asarray(CONDITIONS),
        jlens_scores=jlens_scores,
        logit_lens_scores=logit_lens_scores,
        direct_logits=direct_logits,
    )
    metadata = {
        "config": config.as_dict(),
        "plan": str(plan_path),
        "n_questions": n_questions,
        "lens_repo": lens_repo,
        "lens_filename": lens_filename,
        "layer_alignment": "Readouts 1-63 are post-block residuals transported by JLens; readout 64 is the natural final residual for both lenses.",
        "max_l64_reconstruction_error": max_error,
        "python": sys.version,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    residual_path.unlink()
    completed_path.unlink(missing_ok=True)
    direct_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    args = parser.parse_args()
    collect(args.config, args.plan, args.output, args.lens_repo, args.lens_filename)


if __name__ == "__main__":
    main()
