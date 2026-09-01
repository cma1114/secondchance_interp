from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor
from .prompts import FACTORIAL_FEEDBACK


CONDITIONS = ("incorrect_again", "lost_again")
NEWLINE_TOKEN_ID = 198


def _find_unique_subsequence(row: list[int], needle: list[int]) -> list[int]:
    hits = [
        start
        for start in range(len(row) - len(needle) + 1)
        if row[start : start + len(needle)] == needle
    ]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one feedback-token match, found {hits}")
    return list(range(hits[0], hits[0] + len(needle)))


def _source_positions(payload: dict[str, Any], condition: str, tokenizer: Any) -> list[list[int]]:
    sequence = tokenizer(FACTORIAL_FEEDBACK[condition], add_special_tokens=False)["input_ids"]
    return [
        _find_unique_subsequence(
            payload["input_ids"][row].long().tolist(), [int(value) for value in sequence]
        )
        for row in range(payload["input_ids"].shape[0])
    ]


def _source_labels(tokenizer: Any) -> list[str]:
    game = tokenizer(FACTORIAL_FEEDBACK[CONDITIONS[0]], add_special_tokens=False)["input_ids"]
    neutral = tokenizer(FACTORIAL_FEEDBACK[CONDITIONS[1]], add_special_tokens=False)["input_ids"]
    if len(game) != len(neutral):
        raise RuntimeError("Feedback conditions do not have aligned token counts")
    labels = []
    for index, (left, right) in enumerate(zip(game, neutral)):
        left_text = tokenizer.decode([int(left)]).replace("\n", "\\n")
        right_text = tokenizer.decode([int(right)]).replace("\n", "\\n")
        labels.append(f"{index}:{left_text}" if left == right else f"{index}:{left_text}|{right_text}")
    return labels


def _receiver_columns(payload: dict[str, Any], row: int) -> tuple[list[int], list[list[int]]]:
    mask = payload["receiver_mask"][row].bool()
    positions = payload["receiver_positions"][row][mask].long().tolist()
    ids = payload["input_ids"][row].long().tolist()
    groups: list[list[int]] = []
    current: list[int] = []
    for column, position in enumerate(positions):
        if len(groups) == 4:
            break
        current.append(column)
        if int(ids[position]) == NEWLINE_TOKEN_ID:
            groups.append(current)
            current = []
    if len(groups) != 4:
        raise RuntimeError(f"Could not segment four 2P option lines: {groups}")
    for group in groups:
        if len(group) < 5:
            raise RuntimeError(f"Unexpectedly short 2P line: {group}")
    return positions, groups


def _max_content_tokens(shard_paths: list[Path]) -> int:
    import torch

    maximum = 0
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        payload = shard["payloads"][CONDITIONS[0]]
        for row in range(payload["input_ids"].shape[0]):
            _, groups = _receiver_columns(payload, row)
            maximum = max(maximum, *(len(group[3:-1]) for group in groups))
    return maximum


def _role_names(max_content: int) -> tuple[str, ...]:
    kinds = ("indent", "letter", "colon") + tuple(
        f"content_{index}" for index in range(max_content)
    ) + ("newline",)
    return tuple(f"R{rank}_{kind}" for rank in range(1, 5) for kind in kinds)


def _roles_for_row(
    payload: dict[str, Any],
    row: int,
    rank_letters: list[str],
    original_to_new: dict[str, str],
    max_content: int,
) -> dict[str, list[int]]:
    _, physical_groups = _receiver_columns(payload, row)
    result: dict[str, list[int]] = {}
    for rank_index, first_letter in enumerate(rank_letters):
        second_letter = original_to_new[first_letter]
        group = physical_groups[ord(second_letter) - ord("A")]
        prefix = f"R{rank_index + 1}_"
        result[prefix + "indent"] = [group[0]]
        result[prefix + "letter"] = [group[1]]
        result[prefix + "colon"] = [group[2]]
        content = group[3:-1]
        for content_index in range(max_content):
            result[prefix + f"content_{content_index}"] = (
                [content[content_index]] if content_index < len(content) else []
            )
        result[prefix + "newline"] = [group[-1]]
    return result


def _projected_writes(
    payload: dict[str, Any],
    stored_layer: int,
    source_positions: list[list[int]],
    roles: list[dict[str, list[int]]],
    role_names: tuple[str, ...],
    output_projection: Any,
) -> Any:
    import torch

    device = output_projection.weight.device
    weights = payload["attention_weights"][:, stored_layer].to(device=device)
    values = payload["attention_values"][:, stored_layer].to(device=device)
    gates = payload["attention_gates"][:, stored_layer].to(device=device)
    if values.shape[1] != weights.shape[1]:
        values = values.repeat_interleave(weights.shape[1] // values.shape[1], dim=1)
    output = torch.zeros(
        (weights.shape[0], len(role_names), len(source_positions[0]), output_projection.out_features),
        dtype=torch.float32,
    )
    for row in range(weights.shape[0]):
        source = torch.as_tensor(source_positions[row], device=device, dtype=torch.long)
        local_values = values[row].index_select(1, source)
        valid_roles: list[int] = []
        head_rows = []
        for role_index, role_name in enumerate(role_names):
            columns = roles[row][role_name]
            if not columns:
                continue
            query = torch.as_tensor(columns, device=device, dtype=torch.long)
            local_weights = weights[row].index_select(1, query).index_select(2, source)
            local_gates = gates[row].index_select(0, query)
            head_rows.append(
                torch.einsum("hqs,hsd,qhd->shd", local_weights, local_values, local_gates)
                / float(len(columns))
            )
            valid_roles.append(role_index)
        head_space = torch.stack(head_rows)
        projected = output_projection(
            head_space.reshape(len(valid_roles) * len(source_positions[row]), -1)
        ).reshape(len(valid_roles), len(source_positions[row]), -1).float().cpu()
        output[row, valid_roles] = projected
    return output


def analyze(args: argparse.Namespace) -> None:
    import torch

    shard_paths = sorted((args.workspace / "shards").glob("cohort_*.pt"))
    if len(shard_paths) != 125:
        raise RuntimeError(f"Expected 125 shards, found {len(shard_paths)}")
    completed = np.load(args.workspace / "completed.npy")
    if not np.all(completed):
        raise RuntimeError("Workspace is incomplete")
    if args.max_shards is not None:
        shard_paths = shard_paths[: args.max_shards]

    max_content = _max_content_tokens(shard_paths)
    role_names = _role_names(max_content)
    config = ExperimentConfig.load(args.config)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    source_labels = _source_labels(tokenizer)
    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    discovery = set(json.loads(args.discovery_plan.read_text())["question_ids"])

    shape = (2, 2, 16, len(source_labels), len(role_names), args.model_width)
    sums = torch.zeros(shape, dtype=torch.float32)
    # Mean per-question write magnitude is the direct analogue of mean raw
    # attention: it keeps Game and Neutral separate and does not allow writes
    # with different directions to cancel before their magnitudes are taken.
    write_rms_sums = np.zeros(shape[:-1], dtype=np.float64)
    attention_sums = np.zeros(shape[:-1], dtype=np.float64)
    counts = np.zeros((2, 2, len(role_names)), dtype=np.int64)
    layer_indices: list[int] | None = None
    started = time.perf_counter()

    with torch.inference_mode():
        for shard_index, shard_path in enumerate(shard_paths):
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            qids = [str(value) for value in shard["question_ids"]]
            split_by_row = [0 if qid in discovery else 1 for qid in qids]
            for condition_index, condition in enumerate(CONDITIONS):
                payload = shard["payloads"][condition]
                current_layers = payload["ordinary_layer_indices"].long().tolist()
                if layer_indices is None:
                    layer_indices = current_layers
                elif current_layers != layer_indices:
                    raise RuntimeError("Ordinary-layer inventory changed")
                sources = _source_positions(payload, condition, tokenizer)
                roles = [
                    _roles_for_row(
                        payload,
                        row,
                        shard["rank_letters"][row],
                        mappings[qids[row]]["original_to_new"],
                        max_content,
                    )
                    for row in range(len(qids))
                ]
                for row, split_index in enumerate(split_by_row):
                    for role_index, role_name in enumerate(role_names):
                        if roles[row][role_name]:
                            counts[split_index, condition_index, role_index] += 1
                for stored_layer, layer_index in enumerate(current_layers):
                    projected = _projected_writes(
                        payload,
                        stored_layer,
                        sources,
                        roles,
                        role_names,
                        parts.layers[layer_index].self_attn.o_proj,
                    )
                    weights = payload["attention_weights"][:, stored_layer].float()
                    for row, split_index in enumerate(split_by_row):
                        source_role_writes = projected[row].permute(1, 0, 2)
                        sums[split_index, condition_index, stored_layer] += source_role_writes
                        write_rms_sums[
                            split_index, condition_index, stored_layer
                        ] += torch.sqrt(
                            torch.mean(source_role_writes.square(), dim=-1)
                        ).numpy()
                        for source_index, source_position in enumerate(sources[row]):
                            for role_index, role_name in enumerate(role_names):
                                columns = roles[row][role_name]
                                if columns:
                                    attention_sums[
                                        split_index,
                                        condition_index,
                                        stored_layer,
                                        source_index,
                                        role_index,
                                    ] += float(weights[row, :, columns, source_position].mean())
                    del projected
            print(f"Token cross: {shard_index + 1}/{len(shard_paths)} shards", flush=True)

    if layer_indices is None:
        raise RuntimeError("No layers processed")
    if not np.array_equal(counts[:, 0], counts[:, 1]):
        raise RuntimeError("Condition role counts differ")
    denominator = counts[:, :, None, None, :, None].clip(min=1)
    means = sums.numpy() / denominator
    mean_per_question_write_rms = write_rms_sums / counts[
        :, :, None, None, :
    ].clip(min=1)
    condition_mean_vector_rms = np.sqrt(np.mean(means * means, axis=-1))
    attention_means = attention_sums / counts[:, :, None, None, :].clip(min=1)
    discovery_delta = means[0, 0] - means[0, 1]
    confirmation_delta = means[1, 0] - means[1, 1]
    discovery_rms = np.sqrt(np.mean(discovery_delta * discovery_delta, axis=-1))
    confirmation_rms = np.sqrt(np.mean(confirmation_delta * confirmation_delta, axis=-1))
    cosine = np.sum(discovery_delta * confirmation_delta, axis=-1) / np.maximum(
        np.linalg.norm(discovery_delta, axis=-1) * np.linalg.norm(confirmation_delta, axis=-1),
        1e-12,
    )

    cells = []
    for layer_slot, layer_index in enumerate(layer_indices):
        for source_index, source_label in enumerate(source_labels):
            for role_index, role_name in enumerate(role_names):
                cells.append(
                    {
                        "layer": int(layer_index + 1),
                        "source": source_label,
                        "destination": role_name,
                        "discovery_write_contrast_rms": float(discovery_rms[layer_slot, source_index, role_index]),
                        "confirmation_write_contrast_rms": float(confirmation_rms[layer_slot, source_index, role_index]),
                        "discovery_mean_per_question_write_rms_game": float(mean_per_question_write_rms[0, 0, layer_slot, source_index, role_index]),
                        "discovery_mean_per_question_write_rms_neutral": float(mean_per_question_write_rms[0, 1, layer_slot, source_index, role_index]),
                        "confirmation_mean_per_question_write_rms_game": float(mean_per_question_write_rms[1, 0, layer_slot, source_index, role_index]),
                        "confirmation_mean_per_question_write_rms_neutral": float(mean_per_question_write_rms[1, 1, layer_slot, source_index, role_index]),
                        "discovery_condition_mean_vector_rms_game": float(condition_mean_vector_rms[0, 0, layer_slot, source_index, role_index]),
                        "discovery_condition_mean_vector_rms_neutral": float(condition_mean_vector_rms[0, 1, layer_slot, source_index, role_index]),
                        "confirmation_condition_mean_vector_rms_game": float(condition_mean_vector_rms[1, 0, layer_slot, source_index, role_index]),
                        "confirmation_condition_mean_vector_rms_neutral": float(condition_mean_vector_rms[1, 1, layer_slot, source_index, role_index]),
                        "discovery_confirmation_cosine": float(cosine[layer_slot, source_index, role_index]),
                        "confirmation_attention_game": float(attention_means[1, 0, layer_slot, source_index, role_index]),
                        "confirmation_attention_neutral": float(attention_means[1, 1, layer_slot, source_index, role_index]),
                        "confirmation_count": int(counts[1, 0, role_index]),
                    }
                )
    minimum_selection_count = max(20, int(0.05 * min(counts[0, 0, 0], counts[1, 0, 0])))
    for cell in cells:
        role_index = role_names.index(cell["destination"])
        cell["eligible_for_selection"] = bool(
            counts[0, 0, role_index] >= minimum_selection_count
            and counts[1, 0, role_index] >= minimum_selection_count
        )
    cells.sort(key=lambda row: row["discovery_write_contrast_rms"], reverse=True)
    eligible_cells = [cell for cell in cells if cell["eligible_for_selection"]]
    result = {
        "definition": "Exact ordinary-attention write from each feedback token to each relative token in every 2P option line; no whole-line averaging.",
        "source_interpretation": {
            "tokens_0_2": "identical pre-difference negative controls",
            "token_3": "literal incorrect/lost policy source",
            "token_4": "evaluation-closing period",
            "tokens_5_9": "post-evaluation contextualized relay candidates",
        },
        "questions": int(sum(counts[:, 0, 0])),
        "discovery_questions": int(counts[0, 0, 0]),
        "confirmation_questions": int(counts[1, 0, 0]),
        "layers": [int(value + 1) for value in layer_indices],
        "source_tokens": source_labels,
        "max_semantic_wordpieces": int(max_content),
        "receiver_roles": list(role_names),
        "elapsed_seconds": time.perf_counter() - started,
        "minimum_questions_per_split_for_selection": int(minimum_selection_count),
        "top_discovery_cells": eligible_cells[:200],
        "all_cells": cells,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = "benchmark.json" if args.max_shards is not None else "policy_token_cross.json"
    (args.output_dir / filename).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": True, "output": filename, "elapsed_seconds": result["elapsed_seconds"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--model-width", type=int, default=5120)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
