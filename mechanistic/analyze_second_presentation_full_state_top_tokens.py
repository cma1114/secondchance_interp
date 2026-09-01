from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_second_presentation_policy_token_cross import (
    CONDITIONS,
    _roles_for_row,
    _source_labels,
    _source_positions,
)
from .analyze_second_presentation_policy_transport import (
    _load_lens,
    _receiver_roles,
    _valid_positions,
)
from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor


LENSES = ("J-lens", "R-lens")
KINDS = ("letter", "semantic", "newline")


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


def _top(scores: Any, readable_ids: list[int], tokenizer: Any, k: int) -> list[dict[str, Any]]:
    import torch

    candidate_count = min(max(16 * k, 128), len(readable_ids))
    _values, indices = torch.topk(scores, candidate_count)
    rows, seen = [], set()
    for local_index in indices.detach().cpu().tolist():
        token_id = int(readable_ids[local_index])
        token = tokenizer.decode([token_id])
        key = token.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"token_id": token_id, "token": token, "score": float(scores[local_index])})
        if len(rows) == k:
            break
    return rows


def _max_content_tokens(shards: list[Path]) -> int:
    import torch

    maximum = 0
    for shard_path in shards:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        for condition in CONDITIONS:
            payload = shard["payloads"][condition]
            for row in range(payload["input_ids"].shape[0]):
                mask = payload["receiver_mask"][row].bool()
                positions = payload["receiver_positions"][row][mask].long().tolist()
                ids = payload["input_ids"][row].long().tolist()
                current: list[int] = []
                groups: list[list[int]] = []
                for column, position in enumerate(positions):
                    if len(groups) == 4:
                        break
                    current.append(column)
                    if int(ids[position]) == 198:
                        groups.append(current)
                        current = []
                maximum = max(maximum, *(len(group[3:-1]) for group in groups))
        del shard
    return maximum


def analyze(args: argparse.Namespace) -> None:
    import torch

    started = time.perf_counter()
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

    qids: list[str] = []
    for shard_path in shards:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        qids.extend(
            str(value)
            for value in shard["question_ids"]
            if str(value) not in discovery
        )
        del shard
    if not qids:
        raise RuntimeError("No held-out questions were selected")
    qid_to_index = {qid: index for index, qid in enumerate(qids)}
    if len(qid_to_index) != len(qids):
        raise RuntimeError("Held-out question IDs are not unique")

    max_content = _max_content_tokens(shards)
    source_labels = _source_labels(tokenizer)
    source_names = tuple(f"source_{index}" for index in range(len(source_labels)))
    destination_names = tuple(
        f"R{rank}_{kind}" for rank in range(1, 5) for kind in KINDS
    ) + ("choice_cue_space", "final_decision")
    position_names = source_names + destination_names
    width = int(parts.embedding.weight.shape[-1])
    # condition x heldout question x post-block layer x position x width
    states = torch.empty(
        (2, len(qids), 64, len(position_names), width), dtype=torch.bfloat16
    )

    with torch.inference_mode():
        for shard_index, shard_path in enumerate(shards):
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            local_qids = [str(value) for value in shard["question_ids"]]
            for condition_index, condition in enumerate(CONDITIONS):
                payload = shard["payloads"][condition]
                source_rows = _source_positions(payload, condition, tokenizer)
                detailed_roles = [
                    _roles_for_row(
                        payload,
                        row,
                        shard["rank_letters"][row],
                        mappings[qid]["original_to_new"],
                        max_content,
                    )
                    for row, qid in enumerate(local_qids)
                ]
                tail_roles = [
                    _receiver_roles(
                        payload,
                        row,
                        shard["rank_letters"][row],
                        mappings[qid]["original_to_new"],
                        tokenizer,
                    )
                    for row, qid in enumerate(local_qids)
                ]
                for row, qid in enumerate(local_qids):
                    if qid in discovery:
                        continue
                    target = qid_to_index[qid]
                    residual_positions = _valid_positions(payload, row, "residual")
                    residual_lookup = {
                        position: index for index, position in enumerate(residual_positions)
                    }
                    for source_index, token_position in enumerate(source_rows[row]):
                        column = residual_lookup[token_position]
                        states[condition_index, target, :, source_index] = payload[
                            "residuals"
                        ][row, 1:65, column]

                    output_index = len(source_names)
                    for rank in range(1, 5):
                        for kind in KINDS:
                            prefix = f"R{rank}_"
                            if kind == "semantic":
                                receiver_columns = [
                                    detailed_roles[row][prefix + f"content_{index}"][0]
                                    for index in range(max_content)
                                    if detailed_roles[row][prefix + f"content_{index}"]
                                ]
                            else:
                                receiver_columns = detailed_roles[row][prefix + kind]
                            residual_columns = payload["receiver_in_residual"][
                                row, receiver_columns
                            ].long()
                            states[condition_index, target, :, output_index] = payload[
                                "residuals"
                            ][row, 1:65].index_select(1, residual_columns).mean(1)
                            output_index += 1
                    for tail_name in ("choice_cue_space", "final_decision"):
                        receiver_columns = tail_roles[row][tail_name]
                        residual_columns = payload["receiver_in_residual"][
                            row, receiver_columns
                        ].long()
                        states[condition_index, target, :, output_index] = payload[
                            "residuals"
                        ][row, 1:65].index_select(1, residual_columns).mean(1)
                        output_index += 1
                    if output_index != len(position_names):
                        raise RuntimeError("Position inventory was not filled exactly")
            del shard
            if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(shards):
                print(
                    f"Full-state extraction: {shard_index + 1}/{len(shards)} shards",
                    flush=True,
                )

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
            "Held-out within-task J/R-lens top readable tokens for complete post-block residual "
            "states at all ten feedback tokens, exact 2P option letters, mean semantic "
            "wordpieces, closing newlines, the post-list answer cue, and final decision. "
            "No layer, position, or condition is selected by an effect size."
        ),
        "evidence_label": "Descriptive lens readout of complete residual states; not a causal intervention.",
        "conditions": list(CONDITIONS),
        "heldout_questions": len(qids),
        "layers": list(range(1, 65)),
        "source_tokens": source_labels,
        "source_position_names": list(source_names),
        "destination_position_names": list(destination_names),
        "position_names": list(position_names),
        "max_semantic_wordpieces": max_content,
        "semantic_definition": "Mean complete residual across all semantic wordpieces in that option.",
        "vocabulary_filter": "Decoded token is printable ASCII, nonempty after stripping, and contains at least one alphabetic character.",
        "lens_paths": lens_paths,
        "readouts": {},
    }

    with torch.inference_mode():
        for lens_name in LENSES:
            checkpoint = checkpoints[lens_name]
            lens_rows: dict[str, Any] = {}
            for layer_slot in range(64):
                transport = (
                    checkpoint["J"][layer_slot].to(
                        device=lens_device, dtype=torch.bfloat16
                    )
                    if layer_slot < 63
                    else None
                )
                values = states[:, :, layer_slot]
                flat = values.reshape(-1, width)
                sums = torch.zeros(
                    (2, len(position_names), width), dtype=torch.float32
                )
                for condition_index in range(2):
                    condition_flat = flat[
                        condition_index * len(qids) * len(position_names) :
                        (condition_index + 1) * len(qids) * len(position_names)
                    ]
                    normalized_rows = []
                    for start in range(0, len(condition_flat), 256):
                        stop = min(start + 256, len(condition_flat))
                        batch = condition_flat[start:stop].to(device=lens_device)
                        transported = batch if transport is None else batch @ transport.T
                        normalized_rows.append(
                            parts.final_norm(
                                transported.to(parts.final_norm.weight.dtype)
                            ).float().cpu()
                        )
                    normalized = torch.cat(normalized_rows).reshape(
                        len(qids), len(position_names), width
                    )
                    sums[condition_index] = normalized.mean(0)
                scores = torch.einsum(
                    "cpd,vd->cpv",
                    sums.to(device=readable_head.device, dtype=readable_head.dtype),
                    readable_head,
                ).float().cpu()
                layer_result: dict[str, Any] = {}
                for position_index, position_name in enumerate(position_names):
                    layer_result[position_name] = {
                        condition: _top(
                            scores[condition_index, position_index],
                            readable_ids,
                            tokenizer,
                            args.top_k,
                        )
                        for condition_index, condition in enumerate(CONDITIONS)
                    }
                lens_rows[str(layer_slot + 1)] = layer_result
                print(
                    f"{lens_name} readout: layer {layer_slot + 1}/64",
                    flush=True,
                )
            result["readouts"][lens_name] = lens_rows

    result["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens-repo", default="camilablank/workspace-lenses")
    parser.add_argument("--j-filename", default="qwen3.6-27b/j-lens/lens.pt")
    parser.add_argument("--r-filename", default="qwen3.6-27b/r-lens/lens.pt")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
