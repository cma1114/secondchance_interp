from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_second_presentation_policy_token_cross import (
    CONDITIONS,
    _projected_writes,
    _roles_for_row,
    _source_labels,
    _source_positions,
)
from .analyze_second_presentation_policy_transport import _load_lens
from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor


LENSES = ("J-lens", "R-lens")
KINDS = ("indent", "letter", "colon", "semantic", "newline")
ROLE_NAMES = tuple(f"R{rank}_{kind}" for rank in range(1, 5) for kind in KINDS)


def _grouped_roles(
    payload: dict[str, Any],
    row: int,
    rank_letters: list[str],
    original_to_new: dict[str, str],
    max_content: int,
) -> dict[str, list[int]]:
    detailed = _roles_for_row(
        payload, row, rank_letters, original_to_new, max_content
    )
    grouped: dict[str, list[int]] = {}
    for rank in range(1, 5):
        prefix = f"R{rank}_"
        for kind in ("indent", "letter", "colon", "newline"):
            grouped[prefix + kind] = detailed[prefix + kind]
        grouped[prefix + "semantic"] = [
            detailed[prefix + f"content_{index}"][0]
            for index in range(max_content)
            if detailed[prefix + f"content_{index}"]
        ]
        if not grouped[prefix + "semantic"]:
            raise RuntimeError(f"No semantic wordpieces for row {row}, rank {rank}")
    return grouped


def _readable_token_ids(tokenizer: Any) -> list[int]:
    result = []
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id])
        stripped = text.strip()
        if not stripped or not any(character.isalpha() for character in stripped):
            continue
        if not all(character.isprintable() and ord(character) < 128 for character in stripped):
            continue
        result.append(token_id)
    if len(result) < 1000:
        raise RuntimeError(f"Readable-vocabulary filter unexpectedly small: {len(result)}")
    return result


def _top_from_subset(
    scores: Any,
    token_ids: list[int],
    tokenizer: Any,
    largest: bool,
    k: int,
) -> list[dict[str, Any]]:
    import torch

    signed = scores if largest else -scores
    # We need only k unique case-folded readable tokens. Pulling hundreds is
    # ample for duplicates; sorting thousands in every source/destination cell
    # dominates runtime without changing the reported list.
    candidate_count = min(max(16 * k, 128), len(token_ids))
    _values, local_ids = torch.topk(signed, k=candidate_count)
    rows, seen = [], set()
    for local_id in local_ids.detach().cpu().tolist():
        token_id = int(token_ids[local_id])
        text = tokenizer.decode([token_id])
        key = text.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {"token_id": token_id, "token": text, "score": float(scores[local_id])}
        )
        if len(rows) == k:
            break
    return rows


def _centered_cosine(left: Any, right: Any) -> float:
    import torch

    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float((left @ right / denominator.clamp_min(1e-12)).item())


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
    max_content = args.max_content
    source_labels = _source_labels(tokenizer)
    width = int(parts.embedding.weight.shape[-1])

    # split x task x ordinary layer x feedback source x exact 2P destination x width
    sums = torch.zeros(
        (2, 2, 16, len(source_labels), len(ROLE_NAMES), width), dtype=torch.float32
    )
    rms_sums = np.zeros(sums.shape[:-1], dtype=np.float64)
    counts = np.zeros((2, 2, len(ROLE_NAMES)), dtype=np.int64)
    layer_indices: list[int] | None = None
    started = time.perf_counter()

    with torch.inference_mode():
        for shard_index, shard_path in enumerate(shards):
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            qids = [str(value) for value in shard["question_ids"]]
            split_by_row = [0 if qid in discovery else 1 for qid in qids]
            for condition_index, condition in enumerate(CONDITIONS):
                payload = shard["payloads"][condition]
                current_layers = payload["ordinary_layer_indices"].long().tolist()
                if layer_indices is None:
                    layer_indices = current_layers
                elif current_layers != layer_indices:
                    raise RuntimeError("Ordinary-attention layer inventory changed")
                sources = _source_positions(payload, condition, tokenizer)
                roles = [
                    _grouped_roles(
                        payload,
                        row,
                        shard["rank_letters"][row],
                        mappings[qids[row]]["original_to_new"],
                        max_content,
                    )
                    for row in range(len(qids))
                ]
                for row, split_index in enumerate(split_by_row):
                    counts[split_index, condition_index] += 1
                for stored_layer, layer_index in enumerate(current_layers):
                    projected = _projected_writes(
                        payload,
                        stored_layer,
                        sources,
                        roles,
                        ROLE_NAMES,
                        parts.layers[layer_index].self_attn.o_proj,
                    )
                    for row, split_index in enumerate(split_by_row):
                        source_role = projected[row].permute(1, 0, 2)
                        sums[split_index, condition_index, stored_layer] += source_role
                        rms_sums[split_index, condition_index, stored_layer] += (
                            torch.sqrt(torch.mean(source_role.square(), dim=-1)).numpy()
                        )
                    del projected
            if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(shards):
                print(
                    f"Source-write semantic cross: {shard_index + 1}/{len(shards)} shards",
                    flush=True,
                )

    if layer_indices is None:
        raise RuntimeError("No layers processed")
    if not np.array_equal(counts[:, 0], counts[:, 1]):
        raise RuntimeError("Task counts differ")
    denominator = counts[:, :, None, None, :, None].clip(min=1)
    means = sums.numpy() / denominator
    mean_rms = rms_sums / counts[:, :, None, None, :].clip(min=1)

    readable_ids = _readable_token_ids(tokenizer)
    readable_index = torch.as_tensor(readable_ids, dtype=torch.long)
    head = parts.output_head.weight.detach()
    readable_head = head.index_select(0, readable_index.to(head.device))
    lens_device = parts.final_norm.weight.device
    lens_paths, checkpoints = {}, {}
    for name, filename in zip(LENSES, (args.j_filename, args.r_filename)):
        path, checkpoint = _load_lens(args.lens_repo, filename)
        lens_paths[name] = path
        checkpoints[name] = checkpoint

    result: dict[str, Any] = {
        "definition": (
            "J- and R-lens readout of the exact ordinary-attention residual write from "
            "each feedback token into each 2P option-line token type. Semantic readouts "
            "average R1-R4 only after exact rank-specific writes are reconstructed; "
            "rank-specific magnitudes and vector agreement are retained. The complete "
            "2P residual is never used as the semantic object."
        ),
        "evidence_label": (
            "Activation-path attribution. The write vector is exact, but an English-token "
            "lens readout is descriptive rather than a causal semantic intervention."
        ),
        "questions": int(counts[:, 0, 0].sum()),
        "split_counts": {
            "discovery": int(counts[0, 0, 0]),
            "confirmation": int(counts[1, 0, 0]),
        },
        "conditions": list(CONDITIONS),
        "layers": [int(value + 1) for value in layer_indices],
        "source_tokens": source_labels,
        "destination_roles": list(ROLE_NAMES),
        "destination_definition": {
            "indent": "line-leading whitespace token",
            "letter": "literal 2P option letter",
            "colon": "colon after the option letter",
            "semantic": "mean across all semantic wordpieces in that option",
            "newline": "literal option-closing newline",
        },
        "lens_paths": lens_paths,
        "readable_vocabulary_size": len(readable_ids),
        "elapsed_seconds_before_lens_readout": time.perf_counter() - started,
        "cells": [],
    }

    kind_role_indices = {
        kind: [rank_index * len(KINDS) + KINDS.index(kind) for rank_index in range(4)]
        for kind in KINDS
    }
    grouped_means = np.stack(
        [means[..., indices, :].mean(axis=-2) for indices in kind_role_indices.values()],
        axis=-2,
    )
    grouped_mean_rms = np.stack(
        [mean_rms[..., indices].mean(axis=-1) for indices in kind_role_indices.values()],
        axis=-1,
    )

    for layer_slot, layer_index in enumerate(layer_indices):
        for lens_name in LENSES:
            checkpoint = checkpoints[lens_name]
            transport = (
                checkpoint["J"][layer_index].to(device=lens_device, dtype=torch.bfloat16)
                if layer_index < 63
                else None
            )
            # split x task x source x destination-kind x width. Semantic
            # readout is rank-averaged; exact rank-specific RMS and vector
            # agreement remain attached to every result cell below.
            values = torch.from_numpy(grouped_means[:, :, layer_slot]).to(
                device=lens_device, dtype=torch.bfloat16
            )
            flat = values.reshape(-1, width)
            if transport is not None:
                flat = flat @ transport.T
            flat = parts.final_norm(flat.to(parts.final_norm.weight.dtype))
            scores = (flat.to(readable_head.dtype) @ readable_head.T).float().cpu()
            scores = scores.reshape(2, 2, len(source_labels), len(KINDS), -1)
            for source_index, source in enumerate(source_labels):
                for kind_index, kind in enumerate(KINDS):
                    role_indices = kind_role_indices[kind]
                    for condition_index, condition in enumerate(CONDITIONS):
                        discovery_scores = scores[0, condition_index, source_index, kind_index]
                        confirmation_scores = scores[1, condition_index, source_index, kind_index]
                        confirmation_group = grouped_means[
                            1, condition_index, layer_slot, source_index, kind_index
                        ]
                        rank_cosines = []
                        for rank_index, role_index in enumerate(role_indices):
                            rank_vector = means[
                                1, condition_index, layer_slot, source_index, role_index
                            ]
                            denominator = np.linalg.norm(rank_vector) * np.linalg.norm(
                                confirmation_group
                            )
                            rank_cosines.append(
                                float(
                                    rank_vector @ confirmation_group
                                    / max(float(denominator), 1e-12)
                                )
                            )
                        result["cells"].append(
                            {
                                "layer": int(layer_index + 1),
                                "lens": lens_name,
                                "task": condition,
                                "source": source,
                                "destination_kind": kind,
                                "discovery_mean_per_question_write_rms": float(
                                    grouped_mean_rms[
                                        0, condition_index, layer_slot, source_index, kind_index
                                    ]
                                ),
                                "confirmation_mean_per_question_write_rms": float(
                                    grouped_mean_rms[
                                        1, condition_index, layer_slot, source_index, kind_index
                                    ]
                                ),
                                "confirmation_rank_specific_write_rms": [
                                    float(
                                        mean_rms[
                                            1,
                                            condition_index,
                                            layer_slot,
                                            source_index,
                                            role_index,
                                        ]
                                    )
                                    for role_index in role_indices
                                ],
                                "confirmation_rank_to_rank_mean_write_cosine": rank_cosines,
                                "discovery_confirmation_readable_vocab_cosine": _centered_cosine(
                                    discovery_scores, confirmation_scores
                                ),
                                "confirmation_top": _top_from_subset(
                                    confirmation_scores,
                                    readable_ids,
                                    tokenizer,
                                    True,
                                    args.top_k,
                                ),
                                "confirmation_bottom": _top_from_subset(
                                    confirmation_scores,
                                    readable_ids,
                                    tokenizer,
                                    False,
                                    args.top_k,
                                ),
                            }
                        )
            print(f"Semantic write lens: {lens_name} L{layer_index + 1}", flush=True)
            del values, flat, scores, transport

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["elapsed_seconds"] = time.perf_counter() - started
    filename = "benchmark.json" if args.max_shards is not None else "policy_write_semantics.json"
    (args.output_dir / filename).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "complete": True,
                "output": filename,
                "cells": len(result["cells"]),
                "elapsed_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


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
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--max-content",
        type=int,
        default=32,
        help="Frozen maximum from the previously validated complete token cross.",
    )
    parser.add_argument("--max-shards", type=int)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
