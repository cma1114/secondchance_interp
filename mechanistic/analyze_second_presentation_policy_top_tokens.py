from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_second_presentation_policy_token_cross import (
    CONDITIONS,
    _max_content_tokens,
    _role_names,
    _roles_for_row,
    _source_positions,
)
from .analyze_second_presentation_policy_transport import _load_lens, _readable_top
from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor


LENSES = ("J-lens", "R-lens")
DESTINATIONS = tuple(
    f"R{rank}_{kind}"
    for rank in range(1, 5)
    for kind in ("letter", "semantic", "newline")
)
POSITIONS = ("evaluation_word",) + DESTINATIONS


def _position_columns(roles: dict[str, list[int]], rank: int, kind: str, max_content: int) -> list[int]:
    prefix = f"R{rank}_"
    if kind != "semantic":
        return roles[prefix + kind]
    return [
        roles[prefix + f"content_{index}"][0]
        for index in range(max_content)
        if roles[prefix + f"content_{index}"]
    ]


def _centered_cosine(left: Any, right: Any) -> float:
    import torch

    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float((left @ right / denom.clamp_min(1e-12)).item())


def analyze(args: argparse.Namespace) -> None:
    import torch

    config = ExperimentConfig.load(args.config)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    discovery = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    shards = sorted((args.workspace / "shards").glob("cohort_*.pt"))
    if len(shards) != 125 or not np.all(np.load(args.workspace / "completed.npy")):
        raise RuntimeError("The complete 125-shard workspace is required")
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    max_content = args.max_content if args.max_content is not None else _max_content_tokens(shards)
    role_names = _role_names(max_content)

    # Shards were written in manifest order in fixed groups of config.batch_size.
    # Recovering that order from the manifest avoids loading the 291 GiB cache
    # once merely to read 500 tiny question identifiers.
    manifest = json.loads(Path(config.manifest_path).read_text())
    qids = [str(row["id"]) for row in manifest["questions"]]
    qids = qids[: len(shards) * int(config.batch_size)]
    qid_to_index = {qid: index for index, qid in enumerate(qids)}
    split = np.asarray([0 if qid in discovery else 1 for qid in qids], dtype=np.int8)
    if len(qids) != len(qid_to_index) or len(qids) != 4 * len(shards):
        raise RuntimeError("Question inventory changed")
    split_counts = np.bincount(split, minlength=2).tolist()
    if args.max_shards is None and split_counts != [251, 249]:
        raise RuntimeError("Frozen question inventory changed")
    if min(split_counts) == 0:
        raise RuntimeError("Both frozen splits are required")

    width = int(parts.embedding.weight.shape[-1])
    # condition x question x post-block layer x position x model width
    states = torch.empty((2, len(qids), 64, len(POSITIONS), width), dtype=torch.bfloat16)
    with torch.inference_mode():
        for shard_index, path in enumerate(shards):
            shard = torch.load(path, map_location="cpu", weights_only=False)
            local_qids = [str(value) for value in shard["question_ids"]]
            expected_qids = qids[shard_index * int(config.batch_size) : (shard_index + 1) * int(config.batch_size)]
            if local_qids != expected_qids:
                raise RuntimeError("Shard question order differs from the frozen manifest order")
            for condition_index, condition in enumerate(CONDITIONS):
                payload = shard["payloads"][condition]
                source_rows = _source_positions(payload, condition, tokenizer)
                roles = [
                    _roles_for_row(
                        payload,
                        row,
                        shard["rank_letters"][row],
                        mappings[qid]["original_to_new"],
                        max_content,
                    )
                    for row, qid in enumerate(local_qids)
                ]
                for row, qid in enumerate(local_qids):
                    target = qid_to_index[qid]
                    residual_mask = payload["residual_mask"][row].bool()
                    residual_positions = payload["residual_positions"][row][residual_mask].long().tolist()
                    residual_lookup = {position: index for index, position in enumerate(residual_positions)}
                    source_column = residual_lookup[source_rows[row][3]]
                    states[condition_index, target, :, 0] = payload["residuals"][row, 1:65, source_column]
                    output_index = 1
                    for rank in range(1, 5):
                        for kind in ("letter", "semantic", "newline"):
                            receiver_columns = _position_columns(roles[row], rank, kind, max_content)
                            residual_columns = payload["receiver_in_residual"][row, receiver_columns].long()
                            states[condition_index, target, :, output_index] = (
                                payload["residuals"][row, 1:65]
                                .index_select(1, residual_columns)
                                .mean(1)
                            )
                            output_index += 1
            if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(shards):
                print(f"Top-token residual condensation: {shard_index + 1}/{len(shards)} shards", flush=True)

    lens_paths, checkpoints = {}, {}
    for name, filename in zip(LENSES, (args.j_filename, args.r_filename)):
        path, checkpoint = _load_lens(args.lens_repo, filename)
        lens_paths[name] = path
        checkpoints[name] = checkpoint
    head = parts.output_head.weight.detach()
    device = parts.final_norm.weight.device
    masks = [torch.from_numpy(split == index) for index in range(2)]
    result: dict[str, Any] = {
        "definition": (
            "At every layer and for each lens, select the most Game-favoring and Neutral-favoring "
            "readable vocabulary tokens from the discovery-split incorrect-minus-lost evaluation-word "
            "profile. Freeze those token IDs, then measure raw Game and Neutral evaluation-word and 2P "
            "receiver scores and source-to-receiver cosine only on the 249-question confirmation split."
        ),
        "selection": {
            "source": "literal incorrect/lost token",
            "split": "discovery (251 questions)",
            "positive_tokens_per_layer": args.top_k,
            "negative_tokens_per_layer": args.top_k,
            "filter": "printable ASCII tokens containing at least one alphabetic character",
        },
        "discovery_questions": int(split_counts[0]),
        "confirmation_questions": int(split_counts[1]),
        "layers": list(range(1, 65)),
        "positions": list(POSITIONS),
        "lenses": lens_paths,
        "readouts": {},
    }

    for lens_name in LENSES:
        checkpoint = checkpoints[lens_name]
        lens_rows: dict[str, Any] = {}
        for layer_slot in range(64):
            transport = (
                checkpoint["J"][layer_slot].to(device=device, dtype=torch.bfloat16)
                if layer_slot < 63
                else None
            )
            normalized = torch.empty_like(states[:, :, layer_slot])
            flat = states[:, :, layer_slot].reshape(-1, width)
            for start in range(0, len(flat), 256):
                stop = min(start + 256, len(flat))
                values = flat[start:stop].to(device=device)
                transported = values if transport is None else values @ transport.T
                normalized.reshape(-1, width)[start:stop] = parts.final_norm(
                    transported.to(parts.final_norm.weight.dtype)
                ).detach().cpu()

            split_scores = []
            for mask in masks:
                means = normalized[:, mask].float().mean(1).to(device=head.device, dtype=head.dtype)
                split_scores.append(torch.einsum("cpd,vd->cpv", means, head).float().cpu())
            discovery_scores, confirmation_scores = split_scores
            discovery_contrast = discovery_scores[0, 0] - discovery_scores[1, 0]
            positive = _readable_top(discovery_contrast, tokenizer, True, args.top_k)
            negative = _readable_top(discovery_contrast, tokenizer, False, args.top_k)
            token_ids = [row["token_id"] for row in positive + negative]
            if len(token_ids) != 2 * args.top_k or len(set(token_ids)) != len(token_ids):
                raise RuntimeError(f"Could not freeze {2 * args.top_k} distinct readable tokens")
            token_index = torch.as_tensor(token_ids, dtype=torch.long)
            discovery_direction = discovery_contrast.index_select(0, token_index)
            discovery_direction = discovery_direction - discovery_direction.mean()
            discovery_direction = discovery_direction / torch.linalg.vector_norm(discovery_direction).clamp_min(1e-12)

            positions: dict[str, Any] = {}
            for position_index, position in enumerate(POSITIONS):
                condition_rows = {}
                for condition_index, condition in enumerate(CONDITIONS):
                    raw = confirmation_scores[condition_index, position_index].index_select(0, token_index)
                    source = confirmation_scores[condition_index, 0].index_select(0, token_index)
                    centered = raw - raw.mean()
                    condition_rows[condition] = {
                        "restricted_cosine_to_own_evaluation_word": _centered_cosine(raw, source),
                        "signed_policy_coordinate": float((centered @ discovery_direction).item()),
                        "selected_token_scores": [float(value) for value in raw.tolist()],
                    }
                positions[position] = condition_rows
            lens_rows[str(layer_slot + 1)] = {
                "positive_tokens": positive,
                "negative_tokens": negative,
                "positions": positions,
            }
            if (layer_slot + 1) % 4 == 0 or layer_slot == 0:
                print(f"{lens_name} top-token readout: layer {layer_slot + 1}/64", flush=True)
            del normalized, split_scores, discovery_scores, confirmation_scores, transport
        result["readouts"][lens_name] = lens_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "policy_top_tokens.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"complete": True, "layers": 64, "lenses": list(LENSES)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lens-repo", default="camilablank/workspace-lenses")
    parser.add_argument("--j-filename", default="qwen3.6-27b/j-lens/lens.pt")
    parser.add_argument("--r-filename", default="qwen3.6-27b/r-lens/lens.pt")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-content", type=int)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
