from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
from .prompts import build_factorial_messages, prompt_hash


LETTERS = "ABCD"
CONDITIONS = ("game", "neutral")
FACTORIAL_CONDITIONS = {"game": "incorrect_again", "neutral": "lost_again"}


def _chunks(values: list[int], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_rows(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text())["results"]


def _rank_order(rows: dict[str, Any], qids: list[str]) -> np.ndarray:
    order = np.empty((len(qids), 4), dtype=np.int64)
    for index, qid in enumerate(qids):
        logits = np.asarray(rows[qid]["aggregated_ad_logits"], dtype=np.float64)
        order[index] = np.argsort(-logits, kind="stable")
    return order


def _assert_prompt_pair(tokenizer: Any, game: str, neutral: str) -> dict[str, Any]:
    if game.replace("incorrect", "lost", 1) != neutral:
        raise RuntimeError("Game and Neutral prompts differ beyond incorrect/lost")
    game_ids = tokenizer.encode(game, add_special_tokens=False)
    neutral_ids = tokenizer.encode(neutral, add_special_tokens=False)
    differing = [i for i, (left, right) in enumerate(zip(game_ids, neutral_ids)) if left != right]
    if len(game_ids) != len(neutral_ids) or len(differing) != 1:
        raise RuntimeError(
            f"Expected one model-visible token difference, got lengths "
            f"{len(game_ids)}/{len(neutral_ids)} and positions {differing}"
        )
    position = differing[0]
    return {
        "game_prompt_hash": prompt_hash(game),
        "neutral_prompt_hash": prompt_hash(neutral),
        "differing_token_position": position,
        "game_token_id": int(game_ids[position]),
        "neutral_token_id": int(neutral_ids[position]),
        "game_token": tokenizer.decode([game_ids[position]]),
        "neutral_token": tokenizer.decode([neutral_ids[position]]),
        "prompt_token_count": len(game_ids),
    }


def _decode_scores(
    values: np.ndarray,
    transform: Any,
    parts: Any,
    token_rows: Any,
    token_bias: Any,
    group_slices: list[slice],
    device: Any,
) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(np.asarray(values).copy()).to(device, dtype=torch.float16)
    if transform is not None:
        tensor = tensor @ transform.T
    normed = parts.final_norm(tensor.to(parts.final_norm.weight.dtype)).float()
    logits = normed @ token_rows.T
    if token_bias is not None:
        logits = logits + token_bias
    return torch.stack(
        [torch.logsumexp(logits[:, group], dim=-1) for group in group_slices], dim=-1
    ).cpu().numpy()


def _collect_dataset(
    spec: dict[str, Any],
    model: Any,
    processor: Any,
    parts: Any,
    jacobians: dict[int, Any],
    lens_metadata: dict[str, str],
    max_questions: int | None,
    keep_residuals: bool,
) -> None:
    import torch

    config = ExperimentConfig.load(Path(spec["config"]))
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    baseline = _load_rows(spec["baseline_results"])
    if baseline is None:
        raise ValueError("A same-format first-presentation baseline is required")
    qids = list(baseline)
    if max_questions is not None:
        qids = qids[: int(max_questions)]
    if not set(qids) <= set(questions):
        raise ValueError("Manifest is missing baseline questions")
    trusted = {
        condition: _load_rows(spec.get(f"trusted_{condition}_results"))
        for condition in CONDITIONS
    }
    for condition, rows in trusted.items():
        if rows is not None and not set(qids) <= set(rows):
            raise ValueError(f"Trusted {condition} results are incomplete")

    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = [[token_id for _, token_id in resolved[letter]] for letter in LETTERS]
    flat_variant_ids = [token_id for group in variant_ids for token_id in group]
    group_slices: list[slice] = []
    cursor = 0
    for group in variant_ids:
        group_slices.append(slice(cursor, cursor + len(group)))
        cursor += len(group)

    n = len(qids)
    width = int(parts.embedding.weight.shape[-1])
    residual_path = output / (
        "decision_residuals.npy" if keep_residuals else "decision_residuals.tmp.npy"
    )
    completed_path = output / "completed.npy"
    direct_path = output / "direct_logits.npy"
    expected_shape = (2, n, 64, width)
    if residual_path.exists() and completed_path.exists() and direct_path.exists():
        residuals = np.lib.format.open_memmap(residual_path, mode="r+")
        completed = np.load(completed_path)
        direct_logits = np.load(direct_path)
        if tuple(residuals.shape) != expected_shape or completed.shape != (2, n):
            raise ValueError("Existing checkpoint has incompatible shape")
    else:
        residuals = np.lib.format.open_memmap(
            residual_path, mode="w+", dtype=np.float16, shape=expected_shape
        )
        completed = np.zeros((2, n), dtype=bool)
        direct_logits = np.full((2, n, 4), np.nan, dtype=np.float32)

    prompt_audit: dict[str, Any] = {
        "dataset": spec["name"],
        "conditions": list(CONDITIONS),
        "pair_checks": {},
        "trusted_prompt_checks": {},
    }
    device = model_input_device(parts)
    for ci, condition in enumerate(CONDITIONS):
        pending = [index for index in range(n) if not completed[ci, index]]
        for indices in _chunks(pending, config.batch_size):
            batch_qids = [qids[index] for index in indices]
            prompts = [
                render_chat(
                    processor,
                    build_factorial_messages(
                        questions[qid], FACTORIAL_CONDITIONS[condition], config.prompt_mode
                    ),
                    config.disable_thinking,
                    config.chat_serialization,
                )
                for qid in batch_qids
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

            full_logits = result.logits.detach().float().cpu()
            final = (
                full_logits[:, 0]
                if full_logits.shape[1] == 1
                else full_logits[np.arange(len(indices)), last_indices]
            )
            aggregate = np.asarray(
                [
                    [torch.logsumexp(final[bi, group], dim=0).item() for group in variant_ids]
                    for bi in range(len(indices))
                ],
                dtype=np.float32,
            )
            direct_logits[ci, indices] = aggregate

            trusted_rows = trusted[condition]
            if trusted_rows is not None:
                expected_hashes = [trusted_rows[qid]["prompt_hash"] for qid in batch_qids]
                current_hashes = [prompt_hash(prompt) for prompt in prompts]
                if current_hashes != expected_hashes:
                    raise RuntimeError(f"Trusted prompt mismatch in {spec['name']}/{condition}")
                expected_logits = np.asarray(
                    [trusted_rows[qid]["aggregated_ad_logits"] for qid in batch_qids],
                    dtype=np.float32,
                )
                error = float(np.max(np.abs(aggregate - expected_logits)))
                prompt_audit["trusted_prompt_checks"][condition] = {
                    "checked": True,
                    "max_logit_error": max(
                        error,
                        prompt_audit["trusted_prompt_checks"].get(condition, {}).get(
                            "max_logit_error", 0.0
                        ),
                    ),
                }

            residuals.flush()
            completed[ci, indices] = True
            _atomic_npy(completed_path, completed)
            _atomic_npy(direct_path, direct_logits)
            print(
                f"{spec['name']} {condition}: {int(completed[ci].sum())}/{n}", flush=True
            )

    if not completed.all() or not np.isfinite(direct_logits).all():
        raise RuntimeError("Collection is incomplete or non-finite")

    for audit_index in sorted(set([0, n - 1])):
        qid = qids[audit_index]
        prompts = {
            condition: render_chat(
                processor,
                build_factorial_messages(
                    questions[qid], FACTORIAL_CONDITIONS[condition], config.prompt_mode
                ),
                config.disable_thinking,
                config.chat_serialization,
            )
            for condition in CONDITIONS
        }
        prompt_audit["pair_checks"][qid] = _assert_prompt_pair(
            tokenizer, prompts["game"], prompts["neutral"]
        )
        prompt_audit["pair_checks"][qid]["game_prompt"] = prompts["game"]
        prompt_audit["pair_checks"][qid]["neutral_prompt"] = prompts["neutral"]

    token_rows = parts.output_head.weight.detach()[flat_variant_ids].float()
    token_bias = getattr(parts.output_head, "bias", None)
    if token_bias is not None:
        token_bias = token_bias.detach()[flat_variant_ids].float()
    jlens_scores = np.empty((2, n, 64, 4), dtype=np.float32)
    score_batch = 32
    with torch.inference_mode():
        for layer in range(64):
            transform = (
                jacobians[layer].to(device, dtype=torch.float16) if layer < 63 else None
            )
            for ci in range(2):
                for start in range(0, n, score_batch):
                    stop = min(start + score_batch, n)
                    jlens_scores[ci, start:stop, layer] = _decode_scores(
                        residuals[ci, start:stop, layer],
                        transform,
                        parts,
                        token_rows,
                        token_bias,
                        group_slices,
                        device,
                    )
            if transform is not None:
                del transform
            if layer == 0 or (layer + 1) % 8 == 0 or layer == 63:
                print(f"{spec['name']} JLens: {layer + 1}/64", flush=True)

    max_l64_error = float(np.max(np.abs(jlens_scores[:, :, -1] - direct_logits)))
    if max_l64_error > 0.10:
        raise RuntimeError(f"L64 reconstruction error too large: {max_l64_error}")
    order = _rank_order(baseline, qids)
    _atomic_npz(
        output / "results.npz",
        question_ids=np.asarray(qids),
        conditions=np.asarray(CONDITIONS),
        jlens_scores=jlens_scores,
        direct_logits=direct_logits,
        rank_order=order,
    )
    (output / "prompt_audit.json").write_text(json.dumps(prompt_audit, indent=2) + "\n")
    metadata = {
        "dataset": spec["name"],
        "config": config.as_dict(),
        "baseline_results": spec["baseline_results"],
        "trusted_game_results": spec.get("trusted_game_results"),
        "trusted_neutral_results": spec.get("trusted_neutral_results"),
        "n_questions": n,
        "question_ids": qids,
        "lens": lens_metadata,
        "layer_alignment": (
            "JLens readouts L1-L63 are post-block residuals 1-63 transported to the "
            "final output space; L64 is the natural post-block-64 residual."
        ),
        "score_definition": "logsumexp over bare and leading-space token variants per A-D letter",
        "max_l64_reconstruction_error": max_l64_error,
        "decision_residuals": str(residual_path) if keep_residuals else None,
        "python": sys.version,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not keep_residuals:
        residual_path.unlink()
    completed_path.unlink(missing_ok=True)
    direct_path.unlink(missing_ok=True)


def collect(
    specs_path: Path,
    lens_repo: str,
    lens_filename: str,
    max_questions: int | None,
    keep_residuals: bool,
) -> None:
    import torch
    from huggingface_hub import hf_hub_download

    specs = json.loads(specs_path.read_text())["datasets"]
    if not specs:
        raise ValueError("No dataset specifications")
    configs = [ExperimentConfig.load(Path(spec["config"])) for spec in specs]
    first = configs[0]
    for config in configs[1:]:
        if (config.model_id, config.model_revision) != (first.model_id, first.model_revision):
            raise ValueError("All datasets must use the same model and revision")

    lens_path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    jacobians = checkpoint["J"]
    if sorted(int(layer) for layer in jacobians) != list(range(63)):
        raise ValueError("Unexpected JLens layer coverage")
    model, processor, parts = load_model_and_processor(first)
    if int(checkpoint["d_model"]) != int(parts.embedding.weight.shape[-1]):
        raise ValueError("JLens/model width mismatch")
    lens_metadata = {
        "repo": lens_repo,
        "filename": lens_filename,
        "local_path": lens_path,
    }
    for spec in specs:
        _collect_dataset(
            spec,
            model,
            processor,
            parts,
            jacobians,
            lens_metadata,
            max_questions,
            keep_residuals,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    parser.add_argument("--max-questions", type=int)
    parser.add_argument(
        "--keep-residuals",
        action="store_true",
        help="Retain the final-position residual memmap after compact JLens output is written.",
    )
    args = parser.parse_args()
    collect(
        args.specs,
        args.lens_repo,
        args.lens_filename,
        args.max_questions,
        args.keep_residuals,
    )


if __name__ == "__main__":
    main()
