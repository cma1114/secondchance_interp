from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_second_presentation_policy_transport import (
    CONDITIONS,
    ROLE_NAMES,
    _load_lens,
    _readable_top,
    _receiver_roles,
    _source_positions,
    _valid_positions,
)
from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor


SOURCE_SLOTS = (3, 4, 5)
SOURCE_NAMES = ("evaluation_word", "evaluation_period", "action_choose")
POSITION_NAMES = SOURCE_NAMES + ROLE_NAMES
LENS_NAMES = ("J-lens", "R-lens")


def analyze(
    config_path: Path,
    workspace: Path,
    remapping_plan_path: Path,
    discovery_plan_path: Path,
    output_dir: Path,
    lens_repo: str,
    j_filename: str,
    r_filename: str,
    top_k: int,
) -> None:
    import torch

    config = ExperimentConfig.load(config_path)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    mappings = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    discovery = set(json.loads(discovery_plan_path.read_text())["question_ids"])
    shards = sorted((workspace / "shards").glob("cohort_*.pt"))
    if len(shards) != 125 or not np.all(np.load(workspace / "completed.npy")):
        raise RuntimeError("The complete 125-shard workspace is required")

    qids = []
    for path in shards:
        header = torch.load(path, map_location="cpu", weights_only=False)
        qids.extend(str(value) for value in header["question_ids"])
        del header
    if len(qids) != 500 or len(set(qids)) != 500:
        raise RuntimeError("Workspace question inventory is invalid")
    qid_to_index = {qid: index for index, qid in enumerate(qids)}
    split = np.asarray([0 if qid in discovery else 1 for qid in qids], dtype=np.int8)
    if np.bincount(split, minlength=2).tolist() != [251, 249]:
        raise RuntimeError("Frozen split sizes changed")

    width = int(parts.embedding.weight.shape[-1])
    # condition x question x post-block layer x position x width
    states = torch.empty(
        (2, len(qids), 64, len(POSITION_NAMES), width), dtype=torch.bfloat16
    )
    with torch.inference_mode():
        for shard_index, path in enumerate(shards):
            shard = torch.load(path, map_location="cpu", weights_only=False)
            local_qids = [str(value) for value in shard["question_ids"]]
            for condition_index, condition in enumerate(CONDITIONS):
                payload = shard["payloads"][condition]
                source_rows = _source_positions(payload, condition, tokenizer)
                roles = [
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
                    target = qid_to_index[qid]
                    residual_positions = _valid_positions(payload, row, "residual")
                    lookup = {position: index for index, position in enumerate(residual_positions)}
                    for output_slot, source_slot in enumerate(SOURCE_SLOTS):
                        column = lookup[source_rows[row][source_slot]]
                        states[condition_index, target, :, output_slot] = payload[
                            "residuals"
                        ][row, 1:65, column]
                    for role_slot, role in enumerate(ROLE_NAMES, start=len(SOURCE_NAMES)):
                        receiver_columns = roles[row][role]
                        residual_columns = payload["receiver_in_residual"][
                            row, receiver_columns
                        ].long()
                        states[condition_index, target, :, role_slot] = payload[
                            "residuals"
                        ][row, 1:65].index_select(1, residual_columns).mean(1)
            if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(shards):
                print(f"Condensed policy residuals: {shard_index + 1}/{len(shards)} shards", flush=True)

    lens_paths, checkpoints = {}, {}
    for name, filename in zip(LENS_NAMES, (j_filename, r_filename)):
        path, checkpoint = _load_lens(lens_repo, filename)
        lens_paths[name] = path
        checkpoints[name] = checkpoint
    lens_device = parts.final_norm.weight.device
    head = parts.output_head.weight.detach()
    split_masks = [torch.from_numpy(split == index) for index in range(2)]
    result: dict[str, Any] = {
        "definition": (
            "Matched J- and R-lens trajectories of complete post-block residual states at "
            "three feedback-source tokens, every 2P option-line summary/newline, the choice cue, "
            "and the final decision. Each question is lens-transported and normalized before averaging."
        ),
        "positions": list(POSITION_NAMES),
        "layers": list(range(1, 65)),
        "conditions": list(CONDITIONS),
        "split_counts": {"discovery": 251, "confirmation": 249},
        "lenses": lens_paths,
        "readouts": {},
    }
    for lens_name in LENS_NAMES:
        checkpoint = checkpoints[lens_name]
        lens_rows: dict[str, Any] = {}
        for layer_slot in range(64):
            transport = (
                checkpoint["J"][layer_slot].to(device=lens_device, dtype=torch.bfloat16)
                if layer_slot < 63
                else None
            )
            normalized = torch.empty_like(states[:, :, layer_slot])
            flat = states[:, :, layer_slot].reshape(-1, width)
            for start in range(0, len(flat), 256):
                stop = min(start + 256, len(flat))
                values = flat[start:stop].to(device=lens_device)
                transported = values if transport is None else values @ transport.T
                normalized.reshape(-1, width)[start:stop] = parts.final_norm(
                    transported.to(parts.final_norm.weight.dtype)
                ).detach().cpu()
            layer_result: dict[str, Any] = {}
            for split_index, split_name in enumerate(("discovery", "confirmation")):
                mask = split_masks[split_index]
                means = normalized[:, mask].float().mean(1).to(
                    device=head.device, dtype=head.dtype
                )
                # condition x position x vocabulary
                scores = torch.einsum("cpd,vd->cpv", means, head)
                contrast = (scores[0] - scores[1]).float()
                source_profiles = contrast[: len(SOURCE_NAMES)]
                position_rows: dict[str, Any] = {}
                for position_index, position_name in enumerate(POSITION_NAMES):
                    profile = contrast[position_index]
                    centered = profile - profile.mean()
                    cosines = {}
                    for source_index, source_name in enumerate(SOURCE_NAMES):
                        source_profile = source_profiles[source_index]
                        source_centered = source_profile - source_profile.mean()
                        denom = torch.linalg.vector_norm(centered) * torch.linalg.vector_norm(
                            source_centered
                        )
                        cosines[source_name] = float(
                            (centered @ source_centered / denom.clamp_min(1e-12)).item()
                        )
                    condition_rows: dict[str, Any] = {}
                    for condition_index, condition_name in enumerate(CONDITIONS):
                        condition_profile = scores[condition_index, position_index].float()
                        condition_centered = condition_profile - condition_profile.mean()
                        condition_cosines = {}
                        for source_index, source_name in enumerate(SOURCE_NAMES):
                            condition_source = scores[condition_index, source_index].float()
                            condition_source_centered = condition_source - condition_source.mean()
                            condition_denom = (
                                torch.linalg.vector_norm(condition_centered)
                                * torch.linalg.vector_norm(condition_source_centered)
                            )
                            condition_cosines[source_name] = float(
                                (
                                    condition_centered @ condition_source_centered
                                    / condition_denom.clamp_min(1e-12)
                                ).item()
                            )
                        condition_rows[condition_name] = {
                            "vocab_rms": float(
                                torch.sqrt(torch.mean(condition_centered.square())).item()
                            ),
                            "full_vocab_cosine_to_own_sources": condition_cosines,
                            "top": _readable_top(condition_profile, tokenizer, True, top_k),
                            "bottom": _readable_top(condition_profile, tokenizer, False, top_k),
                        }
                    position_rows[position_name] = {
                        "game_minus_neutral_vocab_rms": float(
                            torch.sqrt(torch.mean(centered.square())).item()
                        ),
                        "full_vocab_cosine_to_sources": cosines,
                        "top": _readable_top(profile, tokenizer, True, top_k),
                        "bottom": _readable_top(profile, tokenizer, False, top_k),
                        "conditions": condition_rows,
                    }
                layer_result[split_name] = position_rows
                del scores, contrast
            lens_rows[str(layer_slot + 1)] = layer_result
            del normalized, transport
            if (layer_slot + 1) % 4 == 0 or layer_slot == 0:
                print(f"{lens_name} policy residuals: layer {layer_slot + 1}/64", flush=True)
        result["readouts"][lens_name] = lens_rows

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "policy_residual_trajectory.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"complete": True, "layers": 64, "positions": len(POSITION_NAMES)}))


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
        args.top_k,
    )


if __name__ == "__main__":
    main()
