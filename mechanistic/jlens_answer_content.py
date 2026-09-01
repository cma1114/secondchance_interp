from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .data import load_activation_dataset
from .jlens_collect import (
    CONDITIONS,
    POSITION_CONDITIONS,
    _load_cached_residuals,
)
from .modeling import get_tokenizer, load_model_and_processor, model_input_device
from .prompts import load_trials


def _normalized_piece(text: str) -> str:
    """Normalize a decoded token for within-question distinctiveness checks."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in text if character.isalnum())


def option_token_lists(tokenizer: Any, options: dict[str, str]) -> tuple[list[list[int]], dict[str, Any]]:
    """Return content-bearing, option-distinctive token IDs for A--D.

    Each option is tokenized with the leading space it has in ordinary prose.
    Punctuation-only pieces and pieces shared by more than one answer option are
    removed. Repeated token IDs within one option are counted once. If this
    leaves an option empty, all of its alphanumeric pieces are retained.
    """
    if list(options) != list("ABCD"):
        raise ValueError("Expected ordered A-D options")
    raw: list[list[tuple[int, str]]] = []
    for letter in "ABCD":
        ids = [int(value) for value in tokenizer.encode(" " + options[letter].strip(), add_special_tokens=False)]
        pieces = [(token_id, _normalized_piece(tokenizer.decode([token_id]))) for token_id in ids]
        raw.append([(token_id, piece) for token_id, piece in pieces if piece])

    document_frequency: dict[str, int] = {}
    for rows in raw:
        for piece in {piece for _, piece in rows}:
            document_frequency[piece] = document_frequency.get(piece, 0) + 1

    selected: list[list[int]] = []
    audit: dict[str, Any] = {"options": {}, "fallback_letters": []}
    for letter, rows in zip("ABCD", raw):
        distinctive = [token_id for token_id, piece in rows if document_frequency[piece] == 1]
        used_fallback = not distinctive
        values = distinctive if distinctive else [token_id for token_id, _ in rows]
        values = list(dict.fromkeys(values))
        if not values:
            raise ValueError(f"Option {letter} has no alphanumeric tokenizer pieces")
        selected.append(values)
        if used_fallback:
            audit["fallback_letters"].append(letter)
        audit["options"][letter] = {
            "text": options[letter],
            "token_ids": values,
            "tokens": [tokenizer.decode([token_id]) for token_id in values],
            "used_fallback": used_fallback,
        }
    return selected, audit


def pad_option_tokens(specs: list[list[list[int]]]) -> tuple[np.ndarray, np.ndarray]:
    maximum = max(len(tokens) for question in specs for tokens in question)
    ids = np.zeros((len(specs), 4, maximum), dtype=np.int64)
    mask = np.zeros((len(specs), 4, maximum), dtype=np.float32)
    for qi, question in enumerate(specs):
        if len(question) != 4:
            raise ValueError("Each question must have four option token lists")
        for oi, tokens in enumerate(question):
            ids[qi, oi, : len(tokens)] = tokens
            mask[qi, oi, : len(tokens)] = 1.0
    return ids, mask


def _option_scores(normed, output_head, token_ids, token_mask):
    """Mean vocabulary score for each question-specific option token centroid."""
    import torch

    rows = output_head.weight[token_ids]
    values = torch.einsum("bd,botd->bot", normed.to(rows.dtype), rows).float()
    bias = getattr(output_head, "bias", None)
    if bias is not None:
        values = values + bias[token_ids].float()
    return (values * token_mask).sum(dim=-1) / token_mask.sum(dim=-1)


def collect(
    config_path: str | Path,
    residual_root: str | Path,
    jlens_root: str | Path,
    output_root: str | Path,
    lens_repo: str,
    lens_filename: str,
    batch_size: int,
    preflight_only: bool = False,
) -> None:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    residual_root = Path(residual_root)
    jlens_root = Path(jlens_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    data = load_activation_dataset(residual_root, list(CONDITIONS))
    final_residuals = _load_cached_residuals(residual_root, data)
    with np.load(jlens_root / "jlens_scores.npz", allow_pickle=False) as cached:
        position_qids = cached["position_question_ids"].astype(str).tolist()
        anchors = cached["anchors"].astype(str).tolist()
        position_conditions = cached["position_conditions"].astype(str).tolist()
        position_availability = cached["position_availability"].astype(bool)
    if position_conditions != list(POSITION_CONDITIONS):
        raise ValueError("Unexpected position-condition order")
    position_residuals = np.load(jlens_root / "position_residuals.npy", mmap_mode="r")

    lens_path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    jacobians = checkpoint["J"]
    source_layers = sorted(int(layer) for layer in jacobians)
    if source_layers != list(range(63)):
        raise ValueError(f"Unexpected JLens source layers: {source_layers}")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    trials = load_trials(config.manifest_path, config.baseline_results_path)
    trials_by_id = {trial.question_id: trial for trial in trials}

    all_specs, audit = [], {}
    for qid in data.question_ids:
        tokens, row = option_token_lists(tokenizer, trials_by_id[qid].question["options"])
        all_specs.append(tokens)
        audit[qid] = row
    token_ids, token_mask = pad_option_tokens(all_specs)
    qid_to_index = {qid: index for index, qid in enumerate(data.question_ids)}
    position_indices = np.asarray([qid_to_index[qid] for qid in position_qids], dtype=np.int64)

    device = model_input_device(parts)
    final_scores = np.empty((len(CONDITIONS), len(data.question_ids), 64, 4), dtype=np.float16)
    position_scores = np.empty(
        (len(POSITION_CONDITIONS), len(position_qids), len(anchors), 64, 4), dtype=np.float16
    )

    @torch.inference_mode()
    def transform_final(values: np.ndarray, J, destination: np.ndarray, layer: int) -> None:
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            residual = torch.from_numpy(values[start:stop]).to(device, dtype=torch.float16)
            transported = residual if J is None else residual @ J.T
            normed = parts.final_norm(transported.to(parts.final_norm.weight.dtype))
            ids = torch.from_numpy(token_ids[start:stop]).to(device)
            mask = torch.from_numpy(token_mask[start:stop]).to(device)
            destination[start:stop, layer] = _option_scores(normed, parts.output_head, ids, mask).cpu().numpy()

    @torch.inference_mode()
    def transform_positions(values: np.ndarray, J, destination: np.ndarray, layer: int) -> None:
        # Batch by questions; every prompt anchor for a question uses the same
        # four option-content centroids.
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            residual = torch.from_numpy(np.asarray(values[start:stop])).to(device, dtype=torch.float16)
            count, n_anchors, width = residual.shape
            flat = residual.reshape(count * n_anchors, width)
            transported = flat if J is None else flat @ J.T
            normed = parts.final_norm(transported.to(parts.final_norm.weight.dtype))
            source_indices = position_indices[start:stop]
            ids = torch.from_numpy(token_ids[source_indices]).to(device)
            mask = torch.from_numpy(token_mask[source_indices]).to(device)
            ids = ids[:, None].expand(-1, n_anchors, -1, -1).reshape(count * n_anchors, 4, -1)
            mask = mask[:, None].expand(-1, n_anchors, -1, -1).reshape(count * n_anchors, 4, -1)
            scores = _option_scores(normed, parts.output_head, ids, mask)
            destination[start:stop, :, layer] = scores.reshape(count, n_anchors, 4).cpu().numpy()

    preflight = {
        "n_questions": len(data.question_ids),
        "n_position_questions": len(position_qids),
        "anchors": anchors,
        "maximum_option_tokens": int(token_ids.shape[-1]),
        "fallback_option_count": int(sum(len(row["fallback_letters"]) for row in audit.values())),
        "d_model": int(checkpoint["d_model"]),
        "lens_layers": source_layers,
    }
    if preflight_only:
        J = jacobians[0].to(device, dtype=torch.float16)
        smoke = np.empty((1, 64, 4), dtype=np.float16)
        transform_final(final_residuals[0, :1, 0], J, smoke, 0)
        preflight["smoke_scores"] = smoke[0, 0].astype(float).tolist()
        (output / "preflight.json").write_text(json.dumps(preflight, indent=2))
        print(json.dumps(preflight, indent=2), flush=True)
        return

    for layer in source_layers:
        J = jacobians[layer].to(device, dtype=torch.float16)
        for ci in range(len(CONDITIONS)):
            transform_final(final_residuals[ci, :, layer], J, final_scores[ci], layer)
        for ci in range(len(POSITION_CONDITIONS)):
            transform_positions(position_residuals[ci, :, layer], J, position_scores[ci], layer)
        del J
        if layer == 0 or (layer + 1) % 8 == 0 or layer == source_layers[-1]:
            print(f"option-content JLens {layer + 1}/63", flush=True)

    for ci in range(len(CONDITIONS)):
        transform_final(final_residuals[ci, :, 63], None, final_scores[ci], 63)
    for ci in range(len(POSITION_CONDITIONS)):
        transform_positions(position_residuals[ci, :, 63], None, position_scores[ci], 63)
    print("option-content JLens natural readout 64/64", flush=True)

    np.savez_compressed(
        output / "option_content_scores.npz",
        final_scores=final_scores,
        position_scores=position_scores,
        question_ids=np.asarray(data.question_ids),
        position_question_ids=np.asarray(position_qids),
        conditions=np.asarray(CONDITIONS),
        position_conditions=np.asarray(POSITION_CONDITIONS),
        anchors=np.asarray(anchors),
        position_availability=position_availability,
    )
    (output / "option_token_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    metadata = {
        "config": config.as_dict(),
        "residual_root": str(residual_root),
        "jlens_root": str(jlens_root),
        "lens_repo": lens_repo,
        "lens_filename": lens_filename,
        "lens_path": lens_path,
        "definition": (
            "Mean JLens vocabulary score over unique alphanumeric tokenizer pieces that occur in only one "
            "of the question's four option texts; fallback to all alphanumeric pieces when needed."
        ),
        "preflight": preflight,
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(preflight, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read question-specific option content with a pretrained JLens")
    parser.add_argument("--config", required=True)
    parser.add_argument("--residual-root", required=True)
    parser.add_argument("--jlens-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    collect(
        args.config,
        args.residual_root,
        args.jlens_root,
        args.output,
        args.lens_repo,
        args.lens_filename,
        args.batch_size,
        args.preflight_only,
    )


if __name__ == "__main__":
    main()
