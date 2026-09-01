from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor, model_input_device
from .prompts import FACTORIAL_FEEDBACK


CONDITIONS = ("incorrect_again", "lost_again")
ROLE_NAMES = tuple(
    [f"R{rank}_{kind}" for rank in range(1, 5) for kind in ("line", "newline")]
    + ["choice_cue_space", "final_decision"]
)
STATE_NAMES = (
    "feedback_source_pre",
    "source_specific_write",
    "receiver_pre",
    "receiver_post_attention",
    "receiver_post_block",
)
LENS_NAMES = ("J-lens", "R-lens")


def _find_unique_subsequence(row: list[int], needle: list[int]) -> list[int]:
    hits = [
        start
        for start in range(len(row) - len(needle) + 1)
        if row[start : start + len(needle)] == needle
    ]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one feedback-token match, found {hits}")
    return list(range(hits[0], hits[0] + len(needle)))


def _valid_positions(payload: dict[str, Any], row: int, kind: str) -> list[int]:
    mask_name = "receiver_mask" if kind == "receiver" else "residual_mask"
    position_name = "receiver_positions" if kind == "receiver" else "residual_positions"
    mask = payload[mask_name][row].bool()
    return payload[position_name][row][mask].long().tolist()


def _receiver_roles(
    payload: dict[str, Any],
    row: int,
    rank_letters: list[str],
    original_to_new: dict[str, str],
    tokenizer: Any,
) -> dict[str, list[int]]:
    positions = _valid_positions(payload, row, "receiver")
    ids = payload["input_ids"][row].long().tolist()
    groups: list[list[int]] = []
    current: list[int] = []
    cursor = 0
    for column, position in enumerate(positions):
        if len(groups) >= 4:
            break
        current.append(column)
        token = tokenizer.convert_ids_to_tokens(int(ids[position]))
        if "Ċ" in token:
            groups.append(current)
            current = []
            cursor = column + 1
    if len(groups) != 4:
        raise RuntimeError(f"Could not segment four second-presentation lines: {groups}")

    result: dict[str, list[int]] = {}
    for rank_index, first_letter in enumerate(rank_letters):
        second_letter = original_to_new[first_letter]
        physical = ord(second_letter) - ord("A")
        result[f"R{rank_index + 1}_line"] = groups[physical]
        result[f"R{rank_index + 1}_newline"] = [groups[physical][-1]]

    tail_columns = list(range(cursor, len(positions)))
    if not tail_columns:
        raise RuntimeError("Post-list receiver tail is empty")
    decoded = [tokenizer.convert_ids_to_tokens(int(ids[positions[c]])) for c in tail_columns]
    cue_candidates = [
        tail_columns[index]
        for index in range(1, len(tail_columns) - 1)
        if decoded[index - 1] == "):"
        and decoded[index + 1] == "<|im_end|>"
    ]
    if len(cue_candidates) != 1:
        raise RuntimeError(f"Could not uniquely locate choice-cue space: {decoded}")
    result["choice_cue_space"] = cue_candidates
    result["final_decision"] = [tail_columns[-1]]
    if set(result) != set(ROLE_NAMES):
        raise RuntimeError("Receiver-role inventory is incomplete")
    return result


def _source_positions(payload: dict[str, Any], condition: str, tokenizer: Any) -> list[list[int]]:
    sequence = tokenizer(FACTORIAL_FEEDBACK[condition], add_special_tokens=False)["input_ids"]
    positions = []
    for row in range(payload["input_ids"].shape[0]):
        ids = payload["input_ids"][row].long().tolist()
        positions.append(_find_unique_subsequence(ids, [int(value) for value in sequence]))
    return positions


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


def _contributions(
    payload: dict[str, Any],
    stored_layer: int,
    source_positions: list[list[int]],
    roles: list[dict[str, list[int]]],
    output_projection: Any,
    device: Any,
) -> Any:
    """Return batch x source-token x receiver-role x model-width writes."""
    import torch

    # With automatic model sharding, each ordinary-attention output projection
    # may live on a different GPU.  Reconstruct on that layer's actual device.
    device = output_projection.weight.device
    weights = payload["attention_weights"][:, stored_layer].to(device=device)
    values = payload["attention_values"][:, stored_layer].to(device=device)
    gates = payload["attention_gates"][:, stored_layer].to(device=device)
    if values.shape[1] != weights.shape[1]:
        values = values.repeat_interleave(weights.shape[1] // values.shape[1], dim=1)
    rows = []
    for row in range(weights.shape[0]):
        source = torch.as_tensor(source_positions[row], device=device, dtype=torch.long)
        local_values = values[row].index_select(1, source)
        role_rows = []
        for role in ROLE_NAMES:
            query = torch.as_tensor(roles[row][role], device=device, dtype=torch.long)
            local_weights = weights[row].index_select(1, query).index_select(2, source)
            local_gates = gates[row].index_select(0, query)
            head_space = torch.einsum(
                "hqs,hsd,qhd->shd", local_weights, local_values, local_gates
            ) / float(len(roles[row][role]))
            role_rows.append(head_space)
        # role x source x head x head-dimension
        rows.append(torch.stack(role_rows, dim=0))
    head_space = torch.stack(rows, dim=0)
    batch, roles_count, sources_count, heads, head_dim = head_space.shape
    projected = output_projection(
        head_space.reshape(batch * roles_count * sources_count, heads * head_dim)
    )
    # Return batch x source x role x model-width.
    return projected.reshape(batch, roles_count, sources_count, -1).permute(0, 2, 1, 3).float()


def _load_lens(repo: str, filename: str) -> tuple[str, dict[str, Any]]:
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    return path, torch.load(path, map_location="cpu", weights_only=False)


def _bootstrap_mean(values: np.ndarray, seed: int = 9401, draws: int = 5000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 250):
        stop = min(start + 250, draws)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": len(values)}


def _readable_top(scores: Any, tokenizer: Any, largest: bool, k: int = 24) -> list[dict[str, Any]]:
    import torch

    signed = scores if largest else -scores
    _values, ids = torch.topk(signed, k=min(8192, int(scores.shape[-1])))
    rows, seen = [], set()
    for token_id in ids.detach().cpu().tolist():
        text = tokenizer.decode([int(token_id)])
        stripped = text.strip()
        if not stripped or not any(c.isalpha() for c in stripped):
            continue
        if not all(c.isprintable() and ord(c) < 128 for c in stripped):
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"token_id": int(token_id), "token": text, "score": float(scores[token_id])})
        if len(rows) == k:
            break
    return rows


def analyze(
    config_path: Path,
    workspace: Path,
    remapping_plan_path: Path,
    discovery_plan_path: Path,
    output_dir: Path,
    lens_repo: str,
    j_filename: str,
    r_filename: str,
    candidate_count: int,
) -> None:
    import torch

    config = ExperimentConfig.load(config_path)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    device = model_input_device(parts)
    lens_device = parts.final_norm.weight.device
    mappings = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    discovery = set(json.loads(discovery_plan_path.read_text())["question_ids"])
    shard_paths = sorted((workspace / "shards").glob("cohort_*.pt"))
    if len(shard_paths) != 125:
        raise RuntimeError(f"Expected 125 workspace shards, found {len(shard_paths)}")
    completed = np.load(workspace / "completed.npy")
    if not np.all(completed):
        raise RuntimeError("Workspace is incomplete")

    source_labels = _source_labels(tokenizer)
    source_count = len(source_labels)
    layer_count = 16
    role_count = len(ROLE_NAMES)
    sums = torch.zeros((2, 2, layer_count, source_count, role_count, 5120), dtype=torch.float32)
    attention_sums = np.zeros((2, 2, layer_count, source_count, role_count), dtype=np.float64)
    counts = np.zeros((2, 2), dtype=np.int64)
    layer_indices = None

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
                    raise RuntimeError("Ordinary-layer inventory changed across shards")
                sources = _source_positions(payload, condition, tokenizer)
                roles = [
                    _receiver_roles(
                        payload,
                        row,
                        shard["rank_letters"][row],
                        mappings[qids[row]]["original_to_new"],
                        tokenizer,
                    )
                    for row in range(len(qids))
                ]
                for stored_layer, layer_index in enumerate(current_layers):
                    projected = _contributions(
                        payload,
                        stored_layer,
                        sources,
                        roles,
                        parts.layers[layer_index].self_attn.o_proj,
                        device,
                    ).cpu()
                    weights = payload["attention_weights"][:, stored_layer].float()
                    for row, split_index in enumerate(split_by_row):
                        sums[split_index, condition_index, stored_layer] += projected[row]
                        counts[split_index, condition_index] += int(stored_layer == 0)
                        for source_index, source_position in enumerate(sources[row]):
                            for role_index, role in enumerate(ROLE_NAMES):
                                columns = roles[row][role]
                                attention_sums[
                                    split_index, condition_index, stored_layer, source_index, role_index
                                ] += float(weights[row, :, columns, source_position].mean())
                    del projected
            if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(shard_paths):
                print(f"Policy source screen: {shard_index + 1}/{len(shard_paths)} shards", flush=True)

    if not np.all(counts[:, 0] == counts[:, 1]):
        raise RuntimeError(f"Condition counts differ: {counts}")
    means = sums.numpy() / counts[:, :, None, None, None, None]
    attention_means = attention_sums / counts[:, :, None, None, None]
    discovery_delta = means[0, 0] - means[0, 1]
    confirmation_delta = means[1, 0] - means[1, 1]
    discovery_score = np.linalg.norm(discovery_delta, axis=-1) / np.sqrt(discovery_delta.shape[-1])
    confirmation_score = np.linalg.norm(confirmation_delta, axis=-1) / np.sqrt(confirmation_delta.shape[-1])
    flat_order = np.argsort(-discovery_score.reshape(-1), kind="stable")

    def candidate_row(layer_slot: int, source_index: int, role_index: int) -> dict[str, Any]:
        candidate = {
            "layer_slot": int(layer_slot),
            "layer_zero_based": int(layer_indices[layer_slot]),
            "layer": int(layer_indices[layer_slot] + 1),
            "source_index": int(source_index),
            "source_token": source_labels[source_index],
            "receiver_role": ROLE_NAMES[role_index],
            "receiver_role_index": int(role_index),
            "discovery_contrast_rms": float(discovery_score[layer_slot, source_index, role_index]),
            "confirmation_contrast_rms": float(confirmation_score[layer_slot, source_index, role_index]),
            "discovery_attention_game": float(attention_means[0, 0, layer_slot, source_index, role_index]),
            "discovery_attention_neutral": float(attention_means[0, 1, layer_slot, source_index, role_index]),
            "confirmation_attention_game": float(attention_means[1, 0, layer_slot, source_index, role_index]),
            "confirmation_attention_neutral": float(attention_means[1, 1, layer_slot, source_index, role_index]),
        }
        discovery_vector = discovery_delta[layer_slot, source_index, role_index]
        confirmation_vector = confirmation_delta[layer_slot, source_index, role_index]
        denominator = np.linalg.norm(discovery_vector) * np.linalg.norm(confirmation_vector)
        candidate["discovery_confirmation_write_cosine"] = float(
            discovery_vector @ confirmation_vector / max(denominator, 1e-12)
        )
        return candidate

    if candidate_count < len(ROLE_NAMES):
        raise ValueError(
            f"candidate_count must be at least {len(ROLE_NAMES)} so every receiver role is represented"
        )
    candidates: list[dict[str, Any]] = []
    selected: set[tuple[int, int, int]] = set()
    # Freeze one discovery-best source/layer combination for every receiver
    # role before adding global runners-up.  This prevents a strong final-state
    # family from crowding all four 2P option lines out of the analysis.
    for role_index in range(len(ROLE_NAMES)):
        local = discovery_score[:, :, role_index]
        layer_slot, source_index = np.unravel_index(
            int(np.argmax(local)), local.shape
        )
        key = (int(layer_slot), int(source_index), int(role_index))
        selected.add(key)
        candidates.append(candidate_row(*key))
    for flat in flat_order:
        key = tuple(int(value) for value in np.unravel_index(flat, discovery_score.shape))
        if key in selected:
            continue
        selected.add(key)
        candidates.append(candidate_row(*key))
        if len(candidates) == candidate_count:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "policy_source_screen.npz",
        layer_indices=np.asarray(layer_indices, dtype=np.int16),
        source_labels=np.asarray(source_labels),
        receiver_roles=np.asarray(ROLE_NAMES),
        counts=counts,
        mean_writes=means.astype(np.float16),
        discovery_write_contrast=discovery_delta.astype(np.float32),
        confirmation_write_contrast=confirmation_delta.astype(np.float32),
        mean_attention=attention_means.astype(np.float32),
    )
    (output_dir / "frozen_policy_candidates.json").write_text(
        json.dumps({"selection_split": "discovery", "candidates": candidates}, indent=2) + "\n"
    )

    lens_paths = {}
    checkpoints = {}
    for name, filename in zip(LENS_NAMES, (j_filename, r_filename)):
        path, checkpoint = _load_lens(lens_repo, filename)
        lens_paths[name] = path
        checkpoints[name] = checkpoint
    unique_layers = sorted({candidate["layer_zero_based"] for candidate in candidates})
    transports: dict[str, dict[int, Any | None]] = {name: {} for name in LENS_NAMES}
    for name, checkpoint in checkpoints.items():
        for layer_index in unique_layers:
            for key in (layer_index - 1, layer_index):
                if key < 0 or key in transports[name]:
                    continue
                transports[name][key] = (
                    checkpoint["J"][key].to(device=lens_device, dtype=torch.bfloat16)
                    if key < 63 else None
                )

    normalized_sums = torch.zeros(
        (2, 2, len(candidates), len(STATE_NAMES), len(LENS_NAMES), 5120), dtype=torch.float32
    )
    split_counts = np.zeros(2, dtype=np.int64)
    confirmation_projection = [[] for _ in candidates]
    discovery_units = []
    for candidate in candidates:
        vector = discovery_delta[
            candidate["layer_slot"], candidate["source_index"], candidate["receiver_role_index"]
        ]
        norm = np.linalg.norm(vector)
        discovery_units.append(torch.from_numpy(vector / max(norm, 1e-12)).float())

    with torch.inference_mode():
        for shard_index, shard_path in enumerate(shard_paths):
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            qids = [str(value) for value in shard["question_ids"]]
            split_by_row = [0 if qid in discovery else 1 for qid in qids]
            condition_cache = []
            for condition_index, condition in enumerate(CONDITIONS):
                payload = shard["payloads"][condition]
                sources = _source_positions(payload, condition, tokenizer)
                roles = [
                    _receiver_roles(payload, row, shard["rank_letters"][row], mappings[qids[row]]["original_to_new"], tokenizer)
                    for row in range(len(qids))
                ]
                contribution_by_layer = {}
                for candidate in candidates:
                    slot = candidate["layer_slot"]
                    if slot not in contribution_by_layer:
                        contribution_by_layer[slot] = _contributions(
                            payload,
                            slot,
                            sources,
                            roles,
                            parts.layers[candidate["layer_zero_based"]].self_attn.o_proj,
                            device,
                        ).cpu()
                local_vectors = []
                for candidate_index, candidate in enumerate(candidates):
                    layer_index = candidate["layer_zero_based"]
                    role_index = candidate["receiver_role_index"]
                    source_index = candidate["source_index"]
                    writes = contribution_by_layer[candidate["layer_slot"]][:, source_index, role_index]
                    state_rows = []
                    for row in range(len(qids)):
                        residual_positions = _valid_positions(payload, row, "residual")
                        lookup = {position: index for index, position in enumerate(residual_positions)}
                        source_column = lookup[sources[row][source_index]]
                        receiver_columns = roles[row][candidate["receiver_role"]]
                        residual_columns = payload["receiver_in_residual"][row, receiver_columns].long()
                        source_pre = payload["residuals"][row, layer_index, source_column]
                        receiver_pre = payload["residuals"][row, layer_index].index_select(0, residual_columns).mean(0)
                        mixer = payload["mixer_outputs"][row, layer_index, receiver_columns].mean(0)
                        receiver_post_attention = receiver_pre + mixer
                        receiver_post_block = payload["residuals"][row, layer_index + 1].index_select(0, residual_columns).mean(0)
                        state_rows.append(torch.stack((source_pre, writes[row], receiver_pre, receiver_post_attention, receiver_post_block)))
                    local_vectors.append(torch.stack(state_rows))
                condition_cache.append(torch.stack(local_vectors, dim=1))

            for candidate_index, candidate in enumerate(candidates):
                if any(split_by_row[row] == 1 for row in range(len(qids))):
                    delta = condition_cache[0][:, candidate_index, 1].float() - condition_cache[1][:, candidate_index, 1].float()
                    projections = delta @ discovery_units[candidate_index]
                    confirmation_projection[candidate_index].extend(
                        float(projections[row]) for row in range(len(qids)) if split_by_row[row] == 1
                    )
                layer_index = candidate["layer_zero_based"]
                for condition_index in range(2):
                    states = condition_cache[condition_index][:, candidate_index]
                    for state_index in range(len(STATE_NAMES)):
                        key = layer_index - 1 if state_index in (0, 2) else layer_index
                        for lens_index, lens_name in enumerate(LENS_NAMES):
                            values = states[:, state_index].to(device=lens_device, dtype=torch.bfloat16)
                            transport = transports[lens_name][key]
                            transported = values if transport is None else values @ transport.T
                            normed = parts.final_norm(transported.to(parts.final_norm.weight.dtype)).float().cpu()
                            for row, split_index in enumerate(split_by_row):
                                normalized_sums[split_index, condition_index, candidate_index, state_index, lens_index] += normed[row]
            for row, split_index in enumerate(split_by_row):
                split_counts[split_index] += 1
            if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(shard_paths):
                print(f"Policy representation transport: {shard_index + 1}/{len(shard_paths)} shards", flush=True)

    head = parts.output_head.weight.detach()
    profiles = np.empty(
        (2, 2, len(candidates), len(STATE_NAMES), len(LENS_NAMES), head.shape[0]), dtype=np.float16
    )
    result_candidates = []
    for candidate_index, candidate in enumerate(candidates):
        row = dict(candidate)
        row["confirmation_projection_on_frozen_discovery_direction"] = _bootstrap_mean(
            np.asarray(confirmation_projection[candidate_index]), seed=9401 + candidate_index
        )
        row["lenses"] = {}
        for lens_index, lens_name in enumerate(LENS_NAMES):
            lens_rows = {}
            for state_index, state_name in enumerate(STATE_NAMES):
                scores_by_split = []
                for split_index in range(2):
                    condition_scores = []
                    for condition_index in range(2):
                        mean_norm = normalized_sums[
                            split_index, condition_index, candidate_index, state_index, lens_index
                        ].to(device=head.device, dtype=head.dtype) / float(split_counts[split_index])
                        scores = mean_norm @ head.T
                        profiles[split_index, condition_index, candidate_index, state_index, lens_index] = (
                            scores.detach().float().cpu().numpy().astype(np.float16)
                        )
                        condition_scores.append(scores)
                    scores_by_split.append(condition_scores)
                confirmation_contrast = scores_by_split[1][0] - scores_by_split[1][1]
                lens_rows[state_name] = {
                    "confirmation_game_top": _readable_top(scores_by_split[1][0], tokenizer, True),
                    "confirmation_neutral_top": _readable_top(scores_by_split[1][1], tokenizer, True),
                    "confirmation_game_minus_neutral_top": _readable_top(confirmation_contrast, tokenizer, True),
                    "confirmation_game_minus_neutral_bottom": _readable_top(confirmation_contrast, tokenizer, False),
                }
            source_profile = profiles[1, 0, candidate_index, 0, lens_index].astype(np.float32) - profiles[1, 1, candidate_index, 0, lens_index].astype(np.float32)
            write_profile = profiles[1, 0, candidate_index, 1, lens_index].astype(np.float32) - profiles[1, 1, candidate_index, 1, lens_index].astype(np.float32)
            source_profile -= source_profile.mean()
            write_profile -= write_profile.mean()
            denominator = np.linalg.norm(source_profile) * np.linalg.norm(write_profile)
            lens_rows["source_to_write_full_vocabulary_cosine"] = float(source_profile @ write_profile / max(denominator, 1e-12))
            row["lenses"][lens_name] = lens_rows
        result_candidates.append(row)

    np.savez_compressed(
        output_dir / "policy_transport_full_vocab_profiles.npz",
        profiles=profiles,
        candidates=np.asarray([json.dumps(row, sort_keys=True) for row in candidates]),
        state_names=np.asarray(STATE_NAMES),
        lens_names=np.asarray(LENS_NAMES),
        conditions=np.asarray(CONDITIONS),
        split_names=np.asarray(("discovery", "confirmation")),
    )
    summary = {
        "definition": "Feedback-token residuals, exact token-specific ordinary-attention writes into second-presentation receiver states, and the receiver states before/after those writes, selected on discovery and evaluated on confirmation.",
        "workspace": str(workspace),
        "questions": int(split_counts.sum()),
        "split_counts": {"discovery": int(split_counts[0]), "confirmation": int(split_counts[1])},
        "conditions": list(CONDITIONS),
        "source_tokens": source_labels,
        "receiver_roles": list(ROLE_NAMES),
        "selection_rule": f"For each of the {len(ROLE_NAMES)} receiver roles, freeze the discovery-best layer/source-token RMS norm of the paired Game-minus-Neutral exact source-specific residual-write contrast, then add the globally strongest nonduplicate candidates until {candidate_count}; confirmation data were not used for selection.",
        "lens_paths": lens_paths,
        "lens_boundary_alignment": "A source/receiver state before an ordinary-attention layer uses the preceding post-block lens map. The source-specific write and states after that layer's attention use the current layer's post-block map, matching prior within-layer workspace-lens diagnostics; layer 64 uses the natural output coordinates.",
        "evidence_label": "Activation-path observation. The exact source-specific attention write is reconstructed, but this analysis does not by itself show that its decoded English direction is causal.",
        "candidates": result_candidates,
    }
    (output_dir / "policy_transport_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"complete": True, "candidates": len(candidates), "split_counts": split_counts.tolist()}, indent=2))


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
    parser.add_argument("--candidate-count", type=int, default=12)
    args = parser.parse_args()
    analyze(
        args.config,
        args.workspace,
        args.remapping_plan,
        args.discovery_plan,
        args.output_dir,
        args.lens_repo,
        args.j_filename,
        args.r_filename,
        args.candidate_count,
    )


if __name__ == "__main__":
    main()
