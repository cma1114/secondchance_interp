from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_second_presentation_attention_distribution import (
    SOURCE_BINS,
    _source_bins_and_queries,
)


CONDITION_LABELS = ("Game", "Neutral")
SUMMARY_NAMES = ("content_mean", "last_content")
TARGET_NAMES = ("old_unique", "fresh_unique")
DERIVED_SOURCE_NAMES = (
    "system_and_header",
    "first_task_instruction",
    "first_question_stem",
    "first_matching_line",
    "first_other_lines",
    "first_answer_boundary",
    "feedback_sentence",
    "second_answer_instruction",
    "second_question_stem",
    "second_matching_line",
    "second_other_lines",
    "chat_separators_other",
)
COMPONENT_NAMES = ("mixer", "mlp")
NEWLINE_TOKEN_ID = 198


def _valid(payload: dict[str, Any], row: int, kind: str) -> Any:
    mask = payload[f"{kind}_mask"][row].bool()
    return payload[f"{kind}_positions"][row][mask].long()


def _receiver_groups(payload: dict[str, Any], row: int) -> list[list[int]]:
    positions = _valid(payload, row, "receiver")
    ids = payload["input_ids"][row].index_select(0, positions).long().tolist()
    groups: list[list[int]] = []
    current: list[int] = []
    for column, token_id in enumerate(ids):
        if len(groups) == 4:
            break
        current.append(column)
        if int(token_id) == NEWLINE_TOKEN_ID:
            groups.append(current)
            current = []
    if len(groups) != 4:
        raise RuntimeError(f"Could not segment four 2P option lines: {groups}")
    return groups


def _rank_receiver_columns(
    payload: dict[str, Any],
    row: int,
    rank_letters: list[str],
    original_to_new: dict[str, str],
) -> list[dict[str, list[int]]]:
    groups = _receiver_groups(payload, row)
    result = []
    for first_letter in rank_letters:
        physical = LETTERS.index(original_to_new[first_letter])
        group = groups[physical]
        content = group[3:-1]
        if not content:
            raise RuntimeError("2P option line has no semantic content token")
        result.append({"content_mean": content, "last_content": [content[-1]]})
    return result


def _derive_sources(raw: np.ndarray) -> np.ndarray:
    # raw: question x rank x SOURCE_BINS
    output = np.zeros(raw.shape[:2] + (len(DERIVED_SOURCE_NAMES),), dtype=raw.dtype)
    output[..., 0] = raw[..., 0]
    output[..., 1] = raw[..., 1]
    output[..., 2] = raw[..., 2]
    output[..., 5] = raw[..., 7]
    output[..., 6] = raw[..., 8]
    output[..., 7] = raw[..., 9]
    output[..., 8] = raw[..., 10]
    output[..., 11] = raw[..., 15]
    for rank in range(4):
        output[:, rank, 3] = raw[:, rank, 3 + rank]
        output[:, rank, 4] = raw[:, rank, 3:7].sum(-1) - raw[:, rank, 3 + rank]
        output[:, rank, 9] = raw[:, rank, 11 + rank]
        output[:, rank, 10] = raw[:, rank, 11:15].sum(-1) - raw[:, rank, 11 + rank]
    return output


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=1, keepdims=True)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else float("nan")


def _covariance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    left -= left.mean()
    right -= right.mean()
    return float(np.mean(left * right))


def analyze(args: argparse.Namespace) -> None:
    import torch

    config = ExperimentConfig.load(args.config)
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    all_qids = [row["id"] for row in manifest["questions"]]
    qid_to_index = {qid: index for index, qid in enumerate(all_qids)}
    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    discovery_qids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_qids for qid in all_qids])
    confirmation = ~discovery

    score_arrays = np.load(args.score_projections, allow_pickle=False)
    if score_arrays["question_ids"].astype(str).tolist() != all_qids:
        raise RuntimeError("Score projection question order differs")
    rank_indices = score_arrays["rank_indices"].astype(np.int64)
    targets_semantic = np.stack(
        [score_arrays["old_unique"], score_arrays["fresh_unique"]], axis=-1
    ).astype(np.float32)
    targets_ranked = np.stack(
        [targets_semantic[q, rank_indices[q]] for q in range(len(all_qids))]
    )
    total_semantic = score_arrays["projections"].astype(np.float32)
    # Stored summary order: line_mean, content_mean, last_content, newline.
    total_ranked = np.empty((2, len(all_qids), 64, 4, 2, 2), dtype=np.float32)
    for condition_index in range(2):
        for q in range(len(all_qids)):
            total_ranked[condition_index, q] = np.take(
                total_semantic[condition_index, q], rank_indices[q], axis=1
            )[:, :, [1, 2], :]

    shard_paths = sorted((args.workspace / "shards").glob("cohort_*.pt"))
    if len(shard_paths) != 125:
        raise RuntimeError(f"Expected 125 shards, found {len(shard_paths)}")
    if args.max_shards is not None:
        shard_paths = shard_paths[: args.max_shards]
    selected_qids = [qid for path in shard_paths for qid in torch.load(path, map_location="cpu", weights_only=False)["question_ids"]]
    selected_indices = np.asarray([qid_to_index[str(qid)] for qid in selected_qids], dtype=np.int64)

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    directions = torch.load(args.score_directions, map_location="cpu", weights_only=False).float()
    # layer x selected-summary x target x width
    directions = directions[:, [1, 2]]

    source_values = np.zeros(
        (2, len(selected_qids), 16, 4, len(SUMMARY_NAMES), len(TARGET_NAMES), len(SOURCE_BINS)),
        dtype=np.float32,
    )
    component_values = np.zeros(
        (2, len(selected_qids), 64, 4, len(SUMMARY_NAMES), len(TARGET_NAMES), len(COMPONENT_NAMES)),
        dtype=np.float32,
    )
    ordinary_layer_indices: list[int] | None = None
    maximum_partition_error = 0.0
    maximum_projection_reconstruction_error = 0.0
    projection_reconstruction_error_sum_squares = 0.0
    mixer_projection_sum_squares = 0.0
    projection_reconstruction_count = 0
    maximum_mixer_projection_absolute = 0.0
    started = time.perf_counter()

    with torch.inference_mode():
        for shard_number, shard_path in enumerate(shard_paths):
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            qids = [str(value) for value in shard["question_ids"]]
            local_targets = [selected_qids.index(qid) for qid in qids]
            for condition_index, condition in enumerate(CONDITIONS):
                payload = shard["payloads"][condition]
                current_layers = payload["ordinary_layer_indices"].long().tolist()
                if ordinary_layer_indices is None:
                    ordinary_layer_indices = current_layers
                elif current_layers != ordinary_layer_indices:
                    raise RuntimeError("Ordinary layer inventory changed")
                batch = _build_batch(config, processor, tokenizer, questions, mappings, qids, condition)
                if not torch.equal(batch["input_ids"].cpu(), payload["input_ids"]):
                    raise RuntimeError("Rebuilt prompt tokens differ from cached prompt")
                source_bins_by_row = []
                query_columns_by_row = []
                rank_columns_by_row = []
                for row, qid in enumerate(qids):
                    second_question = {
                        **questions[qid],
                        "options": {
                            new: questions[qid]["options"][old]
                            for new, old in mappings[qid]["new_to_original"].items()
                        },
                    }
                    unpadded_bins, _queries, _audit = _source_bins_and_queries(
                        tokenizer,
                        batch["prompts"][row],
                        questions[qid],
                        second_question,
                        condition,
                        shard["rank_letters"][row],
                        mappings[qid]["original_to_new"],
                    )
                    unpadded_length = int(payload["attention_mask"][row].sum())
                    left_pad = int(payload["input_ids"].shape[1] - unpadded_length)
                    padded_bins = [[left_pad + value for value in group] for group in unpadded_bins]
                    assigned = sorted(value for group in padded_bins for value in group)
                    expected = list(range(left_pad, payload["input_ids"].shape[1]))
                    if assigned != expected:
                        raise RuntimeError("Source bins do not exhaust nonpadding prompt tokens")
                    source_bins_by_row.append(padded_bins)
                    receiver_positions = _valid(payload, row, "receiver").tolist()
                    query_columns_by_row.append({value: column for column, value in enumerate(receiver_positions)})
                    rank_columns_by_row.append(
                        _rank_receiver_columns(
                            payload,
                            row,
                            shard["rank_letters"][row],
                            mappings[qid]["original_to_new"],
                        )
                    )

                # Component writes at every layer.
                for layer in range(64):
                    for row, target_row in enumerate(local_targets):
                        receiver_lookup = payload["receiver_in_residual"][row].long()
                        for rank in range(4):
                            for summary_index, summary_name in enumerate(SUMMARY_NAMES):
                                columns = rank_columns_by_row[row][rank][summary_name]
                                residual_columns = receiver_lookup[columns]
                                post = payload["residuals"][row, layer + 1].index_select(0, residual_columns).float()
                                rms = torch.sqrt(post.square().mean(-1).clamp_min(1e-12))
                                direction = directions[layer, summary_index]
                                for component_index, component_name in enumerate(COMPONENT_NAMES):
                                    source = payload[f"{component_name}_outputs"][row, layer].index_select(0, torch.as_tensor(columns)).float()
                                    normalized = source / rms[:, None]
                                    component_values[
                                        condition_index, target_row, layer, rank, summary_index, :, component_index
                                    ] = torch.einsum("qd,td->t", normalized, direction).numpy() / len(columns)

                # Exact source-bin decomposition at ordinary-attention layers.
                for stored_layer, layer in enumerate(current_layers):
                    output_projection = parts.layers[layer].self_attn.o_proj
                    device = output_projection.weight.device
                    weight = output_projection.weight.float()
                    weights = payload["attention_weights"][:, stored_layer].to(device=device).float()
                    values = payload["attention_values"][:, stored_layer].to(device=device).float()
                    gates = payload["attention_gates"][:, stored_layer].to(device=device).float()
                    if values.shape[1] != weights.shape[1]:
                        values = values.repeat_interleave(weights.shape[1] // values.shape[1], dim=1)
                    for row, target_row in enumerate(local_targets):
                        receiver_lookup = payload["receiver_in_residual"][row].long()
                        bin_matrix = torch.zeros(
                            (weights.shape[-1], len(SOURCE_BINS)),
                            device=device,
                            dtype=torch.float32,
                        )
                        for bin_index, positions in enumerate(source_bins_by_row[row]):
                            bin_matrix[
                                torch.as_tensor(positions, device=device, dtype=torch.long),
                                bin_index,
                            ] = 1.0
                        for rank in range(4):
                            for summary_index, summary_name in enumerate(SUMMARY_NAMES):
                                columns = rank_columns_by_row[row][rank][summary_name]
                                residual_columns = receiver_lookup[columns]
                                post = payload["residuals"][row, layer + 1].index_select(0, residual_columns).float().to(device)
                                rms = torch.sqrt(post.square().mean(-1).clamp_min(1e-12))
                                direction = directions[layer, summary_index].to(device)
                                effective = torch.matmul(weight.T, direction.T).T
                                query = torch.as_tensor(columns, device=device, dtype=torch.long)
                                local_weights = weights[row].index_select(1, query)
                                local_gates = gates[row].index_select(0, query)
                                head_count = int(local_weights.shape[0])
                                head_dim = int(values.shape[-1])
                                effective_heads = effective.reshape(
                                    len(TARGET_NAMES), head_count, head_dim
                                )
                                gated_effective = (
                                    local_gates[:, None]
                                    * effective_heads[None]
                                )
                                value_scores = torch.einsum(
                                    "hsd,qthd->qths", values[row], gated_effective
                                )
                                per_source = (
                                    value_scores
                                    * local_weights.permute(1, 0, 2)[:, None]
                                ).sum(2)
                                binned = torch.einsum(
                                    "qts,sb->qtb", per_source, bin_matrix
                                )
                                mean_projection = (
                                    binned / rms[:, None, None]
                                ).mean(0)
                                source_values[
                                    condition_index, target_row, stored_layer, rank, summary_index
                                ] = mean_projection.cpu().numpy()
                                bin_sum = mean_projection.sum(-1)
                                mixer = payload["mixer_outputs"][row, layer].index_select(0, torch.as_tensor(columns)).float().to(device)
                                mixer_projection = (torch.matmul(mixer, direction.T) / rms[:, None]).mean(0)
                                reconstruction_error = bin_sum - mixer_projection
                                maximum_projection_reconstruction_error = max(
                                    maximum_projection_reconstruction_error,
                                    float(reconstruction_error.abs().max()),
                                )
                                projection_reconstruction_error_sum_squares += float(
                                    reconstruction_error.square().sum()
                                )
                                mixer_projection_sum_squares += float(
                                    mixer_projection.square().sum()
                                )
                                projection_reconstruction_count += int(
                                    mixer_projection.numel()
                                )
                                maximum_mixer_projection_absolute = max(
                                    maximum_mixer_projection_absolute,
                                    float(mixer_projection.abs().max()),
                                )
                                maximum_partition_error = max(
                                    maximum_partition_error,
                                    float((local_weights.sum(-1) - 1.0).abs().max()),
                                )
            print(f"Score source attribution: {shard_number + 1}/{len(shard_paths)} shards", flush=True)

    if ordinary_layer_indices is None:
        raise RuntimeError("No ordinary layers processed")
    if not np.isfinite(source_values).all() or not np.isfinite(component_values).all():
        raise RuntimeError("Attribution arrays contain nonfinite values")

    derived = np.zeros(source_values.shape[:-1] + (len(DERIVED_SOURCE_NAMES),), dtype=np.float32)
    for condition_index in range(2):
        for stored_layer in range(16):
            for summary_index in range(2):
                for target_index in range(2):
                    derived[condition_index, :, stored_layer, :, summary_index, target_index] = _derive_sources(
                        source_values[condition_index, :, stored_layer, :, summary_index, target_index]
                    )

    local_confirmation = confirmation[selected_indices]
    local_targets_ranked = targets_ranked[selected_indices]
    result: dict[str, Any] = {
        "definition": "Exact source-bin and complete mixer/MLP contributions projected onto frozen unique-old and unique-fresh 2P score directions after per-token natural-residual RMS normalization.",
        "validation": {
            "questions": len(selected_qids),
            "ordinary_layers": [value + 1 for value in ordinary_layer_indices],
            "all_component_layers": 64,
            "maximum_attention_partition_error": maximum_partition_error,
            "maximum_mixer_projection_reconstruction_error": maximum_projection_reconstruction_error,
            "maximum_mixer_projection_absolute": maximum_mixer_projection_absolute,
            "mixer_projection_reconstruction_relative_rmse": float(
                np.sqrt(
                    projection_reconstruction_error_sum_squares
                    / max(mixer_projection_sum_squares, 1e-30)
                )
            ),
            "projection_reconstruction_cells": projection_reconstruction_count,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "source_groups": list(DERIVED_SOURCE_NAMES),
        "summaries": {},
    }
    for summary_index, summary_name in enumerate(SUMMARY_NAMES):
        summary_rows: dict[str, Any] = {}
        for target_index, target_name in enumerate(TARGET_NAMES):
            target = local_targets_ranked[:, :, target_index]
            target_rows: dict[str, Any] = {"ordinary_attention_sources": {}, "components": {}}
            for stored_layer, layer in enumerate(ordinary_layer_indices):
                layer_rows = {}
                for source_index, source_name in enumerate(DERIVED_SOURCE_NAMES):
                    condition_rows = {}
                    for condition_index, condition_name in enumerate(CONDITION_LABELS):
                        contribution = _center(
                            derived[condition_index, :, stored_layer, :, summary_index, target_index, source_index]
                        )
                        total = _center(
                            total_ranked[condition_index, selected_indices, layer, :, summary_index, target_index]
                        )
                        denom = _covariance(total[local_confirmation], target[local_confirmation])
                        condition_rows[condition_name] = {
                            "confirmation_correlation": _correlation(
                                contribution[local_confirmation], target[local_confirmation]
                            ),
                            "confirmation_target_covariance_fraction": float(
                                _covariance(contribution[local_confirmation], target[local_confirmation])
                                / denom
                            ) if denom else float("nan"),
                        }
                    layer_rows[source_name] = condition_rows
                target_rows["ordinary_attention_sources"][str(layer + 1)] = layer_rows
            for layer in range(64):
                layer_rows = {}
                for component_index, component_name in enumerate(COMPONENT_NAMES):
                    condition_rows = {}
                    for condition_index, condition_name in enumerate(CONDITION_LABELS):
                        contribution = _center(
                            component_values[condition_index, :, layer, :, summary_index, target_index, component_index]
                        )
                        total = _center(
                            total_ranked[condition_index, selected_indices, layer, :, summary_index, target_index]
                        )
                        denom = _covariance(total[local_confirmation], target[local_confirmation])
                        condition_rows[condition_name] = {
                            "confirmation_correlation": _correlation(
                                contribution[local_confirmation], target[local_confirmation]
                            ),
                            "confirmation_target_covariance_fraction": float(
                                _covariance(contribution[local_confirmation], target[local_confirmation])
                                / denom
                            ) if denom else float("nan"),
                        }
                    layer_rows[component_name] = condition_rows
                target_rows["components"][str(layer + 1)] = layer_rows
            summary_rows[target_name] = target_rows
        result["summaries"][summary_name] = summary_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = "benchmark.json" if args.max_shards is not None else "score_source_attribution.json"
    (args.output_dir / filename).write_text(json.dumps(result, indent=2) + "\n")
    if args.max_shards is None:
        np.savez_compressed(
            args.output_dir / "score_source_attribution_arrays.npz",
            question_ids=np.asarray(selected_qids),
            selected_indices=selected_indices,
            source_values=source_values.astype(np.float16),
            derived_source_values=derived.astype(np.float16),
            component_values=component_values.astype(np.float16),
        )
    print(json.dumps({"complete": True, "output": filename, **result["validation"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--score-directions", type=Path, required=True)
    parser.add_argument("--score-projections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shards", type=int)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
