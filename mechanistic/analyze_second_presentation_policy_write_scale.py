from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_second_presentation_policy_token_cross import (
    CONDITIONS,
    _max_content_tokens,
    _receiver_columns,
    _role_names,
    _roles_for_row,
)


def _rms(values: Any) -> float:
    import torch

    return float(torch.sqrt(torch.mean(values.float().square())).item())


def analyze(args: argparse.Namespace) -> None:
    import torch

    shards = sorted((args.workspace / "shards").glob("cohort_*.pt"))
    if len(shards) != 125 or not np.all(np.load(args.workspace / "completed.npy")):
        raise RuntimeError("The complete 125-shard workspace is required")
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    max_content = args.max_content if args.max_content is not None else _max_content_tokens(shards)
    role_names = _role_names(max_content)
    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    discovery = set(json.loads(args.discovery_plan.read_text())["question_ids"])

    # split x condition x ordinary layer x role
    mixer_sums = np.zeros((2, 2, 16, len(role_names)), dtype=np.float64)
    residual_sums = np.zeros_like(mixer_sums)
    counts = np.zeros((2, 2, len(role_names)), dtype=np.int64)
    layer_indices: list[int] | None = None

    for shard_index, path in enumerate(shards):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        qids = [str(value) for value in shard["question_ids"]]
        for condition_index, condition in enumerate(CONDITIONS):
            payload = shard["payloads"][condition]
            current_layers = payload["ordinary_layer_indices"].long().tolist()
            if layer_indices is None:
                layer_indices = current_layers
            elif current_layers != layer_indices:
                raise RuntimeError("Ordinary-attention layer inventory changed")
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
            for row, qid in enumerate(qids):
                split_index = 0 if qid in discovery else 1
                for role_index, role_name in enumerate(role_names):
                    columns = roles[row][role_name]
                    if not columns:
                        continue
                    counts[split_index, condition_index, role_index] += 1
                    residual_columns = payload["receiver_in_residual"][row, columns].long()
                    for stored_layer, layer_index in enumerate(current_layers):
                        # Average token magnitudes, matching the source-write statistic.
                        mixer_sums[split_index, condition_index, stored_layer, role_index] += float(
                            torch.sqrt(
                                torch.mean(
                                    payload["mixer_outputs"][row, layer_index, columns].float().square(),
                                    dim=-1,
                                )
                            ).mean().item()
                        )
                        residual_sums[split_index, condition_index, stored_layer, role_index] += float(
                            torch.sqrt(
                                torch.mean(
                                    payload["residuals"][row, layer_index]
                                    .index_select(0, residual_columns)
                                    .float()
                                    .square(),
                                    dim=-1,
                                )
                            ).mean().item()
                        )
        if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(shards):
            print(f"Write-scale denominators: {shard_index + 1}/{len(shards)} shards", flush=True)

    if layer_indices is None:
        raise RuntimeError("No ordinary-attention layers found")
    denom = counts[:, :, None, :].clip(min=1)
    mixer_means = mixer_sums / denom
    residual_means = residual_sums / denom
    rows = []
    for split_index, split_name in enumerate(("discovery", "confirmation")):
        for condition_index, condition in enumerate(CONDITIONS):
            for stored_layer, layer_index in enumerate(layer_indices):
                for role_index, role_name in enumerate(role_names):
                    rows.append(
                        {
                            "split": split_name,
                            "condition": condition,
                            "layer": int(layer_index + 1),
                            "destination": role_name,
                            "mean_complete_attention_write_rms": float(
                                mixer_means[split_index, condition_index, stored_layer, role_index]
                            ),
                            "mean_receiver_pre_layer_residual_rms": float(
                                residual_means[split_index, condition_index, stored_layer, role_index]
                            ),
                            "questions": int(counts[split_index, condition_index, role_index]),
                        }
                    )
    result = {
        "definition": (
            "Denominators for interpreting exact feedback-token source writes: the complete "
            "ordinary-attention output and the receiver residual immediately before that layer, "
            "using the same per-question RMS and destination-token averaging as the source-write analysis."
        ),
        "layers": [int(value + 1) for value in layer_indices],
        "receiver_roles": list(role_names),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "policy_write_scale.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": True, "rows": len(rows)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-content", type=int)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
