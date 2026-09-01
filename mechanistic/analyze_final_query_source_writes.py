from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_cue_attention_distribution import (
    CONDITION_LABELS,
    SOURCE_NAMES,
    _cue_source_partition,
    _display_labels,
)
from .analyze_second_presentation_policy_transport import _receiver_roles
from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_second_presentation_attention_distribution import ORDINARY_LAYERS


def _rms_normalize_torch(values: Any) -> Any:
    import torch

    return values.float() / torch.sqrt(
        torch.clamp(values.float().square().mean(dim=-1, keepdim=True), min=1e-12)
    )


def _bootstrap_means(
    values: np.ndarray, indices: np.ndarray, seed: int, draws: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap condition x layer x question x source arrays by question."""
    selected = np.asarray(values[:, :, indices], dtype=np.float32)
    point = selected.mean(axis=2)
    rng = np.random.default_rng(seed)
    samples = np.empty((draws,) + point.shape, dtype=np.float32)
    for start in range(0, draws, 100):
        stop = min(start + 100, draws)
        rows = rng.integers(0, selected.shape[2], size=(stop - start, selected.shape[2]))
        samples[start:stop] = selected[:, :, rows].mean(axis=3).transpose(2, 0, 1, 3)
    low, high = np.quantile(samples, (0.025, 0.975), axis=0)
    return point, low, high


def _decoder_delta(
    residual: Any,
    write: Any,
    coefficients: Any,
) -> Any:
    """Change in the frozen decoded score caused by retaining `write` in residual."""
    after = _rms_normalize_torch(residual)
    without = _rms_normalize_torch(residual - write)
    return (after - without) @ coefficients.float()


def analyze(args: argparse.Namespace) -> None:
    import torch

    started = time.perf_counter()
    config = ExperimentConfig.load(args.config)
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {str(row["id"]): row for row in manifest["questions"]}
    qids = [str(row["id"]) for row in manifest["questions"]]
    if len(qids) != 500:
        raise RuntimeError("Expected the canonical 500-question manifest")
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    if [int(discovery.sum()), int((~discovery).sum())] != [251, 249]:
        raise RuntimeError("Frozen discovery/confirmation split changed")

    shard_paths = sorted((args.workspace / "shards").glob("cohort_*.pt"))
    if len(shard_paths) != 125 or not np.load(args.workspace / "completed.npy").all():
        raise RuntimeError("Complete 125-shard residual workspace is required")
    selected_shards = shard_paths[: args.max_shards] if args.max_shards else shard_paths
    question_count = 4 * len(selected_shards)

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    decoder = torch.load(args.score_decoders, map_location="cpu", weights_only=False)
    if decoder.get("receiver_role") != "final_decision":
        raise RuntimeError("Score decoder is not the final-decision decoder")
    coefficients = decoder["coefficients"].float()
    if tuple(coefficients.shape) != (64, 2, args.model_width, 4):
        raise RuntimeError(f"Unexpected decoder coefficient shape: {tuple(coefficients.shape)}")

    shape = (2, len(ORDINARY_LAYERS), question_count, len(SOURCE_NAMES))
    attention_mass = np.full(shape, np.nan, dtype=np.float32)
    write_rms = np.full(shape, np.nan, dtype=np.float32)
    decoded_delta = np.full(shape + (2, 4), np.nan, dtype=np.float16)
    reconstruction_relative_rmse: list[float] = []
    feedback_token_pairs: list[str] | None = None

    with torch.inference_mode():
        for shard_index, shard_path in enumerate(selected_shards):
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            shard_qids = [str(value) for value in shard["question_ids"]]
            expected = qids[4 * shard_index : 4 * shard_index + len(shard_qids)]
            if shard_qids != expected:
                raise RuntimeError("Workspace question order changed")
            rank_letters = [[str(value) for value in row] for row in shard["rank_letters"]]
            for condition_index, condition in enumerate(CONDITIONS):
                payload = shard["payloads"][condition]
                batch = _build_batch(
                    config, tokenizer, tokenizer, questions, mappings, shard_qids, condition
                )
                if not torch.equal(batch["input_ids"].cpu(), payload["input_ids"].cpu()):
                    raise RuntimeError("Reconstructed prompt tokens differ from workspace")
                current_layers = tuple(
                    int(value) + 1 for value in payload["ordinary_layer_indices"].tolist()
                )
                if current_layers != ORDINARY_LAYERS:
                    raise RuntimeError(f"Ordinary-attention layers changed: {current_layers}")
                roles = [
                    _receiver_roles(
                        payload,
                        row,
                        rank_letters[row],
                        mappings[shard_qids[row]]["original_to_new"],
                        tokenizer,
                    )
                    for row in range(len(shard_qids))
                ]
                partitions: list[list[list[int]]] = []
                local_feedback_tokens: list[list[str]] = []
                for row, qid in enumerate(shard_qids):
                    second_question = {
                        **questions[qid],
                        "options": {
                            new: questions[qid]["options"][old]
                            for new, old in mappings[qid]["new_to_original"].items()
                        },
                    }
                    rows, audit = _cue_source_partition(
                        tokenizer,
                        batch["prompts"][row],
                        batch["messages"][row],
                        questions[qid],
                        second_question,
                        condition,
                        rank_letters[row],
                        mappings[qid]["original_to_new"],
                    )
                    left_pad = int(
                        payload["attention_mask"][row].numel() - audit["prompt_length"]
                    )
                    partitions.append(
                        [[left_pad + value for value in source] for source in rows]
                    )
                    local_feedback_tokens.append(audit["feedback_tokens"])
                if any(row != local_feedback_tokens[0] for row in local_feedback_tokens):
                    raise RuntimeError("Feedback tokenization changed across questions")
                if condition_index == 0:
                    game_feedback = local_feedback_tokens[0]
                else:
                    pairs = [
                        left if left == right else f"{left} | {right}"
                        for left, right in zip(game_feedback, local_feedback_tokens[0])
                    ]
                    if feedback_token_pairs is None:
                        feedback_token_pairs = pairs
                    elif feedback_token_pairs != pairs:
                        raise RuntimeError("Feedback display labels changed")

                for row in range(len(shard_qids)):
                    query_columns = roles[row]["final_decision"]
                    if len(query_columns) != 1:
                        raise RuntimeError("Final decision is not a unique cached receiver")
                    query = int(query_columns[0])
                    target_question = 4 * shard_index + row
                    residual_column = int(payload["receiver_in_residual"][row, query])
                    for stored_layer, layer_number in enumerate(ORDINARY_LAYERS):
                        layer_index = layer_number - 1
                        module = parts.layers[layer_index].self_attn
                        device = module.o_proj.weight.device
                        weights = payload["attention_weights"][
                            row, stored_layer, :, query
                        ].to(device=device, dtype=torch.float32)
                        values = payload["attention_values"][row, stored_layer].to(
                            device=device, dtype=torch.float32
                        )
                        if values.shape[0] != weights.shape[0]:
                            values = values.repeat_interleave(
                                weights.shape[0] // values.shape[0], dim=0
                            )
                        gate = payload["attention_gates"][row, stored_layer, query].to(
                            device=device, dtype=torch.float32
                        )
                        residual = payload["residuals"][
                            row, layer_number, residual_column
                        ].to(device=device, dtype=torch.float32)
                        natural_sum = torch.zeros_like(residual)
                        coefficient = coefficients[layer_number - 1].to(device=device)
                        for source_index, source_positions in enumerate(partitions[row]):
                            positions = torch.as_tensor(
                                source_positions, device=device, dtype=torch.long
                            )
                            local_weights = weights.index_select(1, positions)
                            local_values = values.index_select(1, positions)
                            context = torch.einsum("hs,hsd->hd", local_weights, local_values)
                            write = module.o_proj(
                                (context * gate).reshape(1, -1).to(module.o_proj.weight.dtype)
                            )[0].float()
                            natural_sum += write
                            attention_mass[
                                condition_index, stored_layer, target_question, source_index
                            ] = float(local_weights.sum(dim=1).mean().item())
                            write_rms[
                                condition_index, stored_layer, target_question, source_index
                            ] = float(torch.sqrt(write.square().mean()).item())
                            for target_index in range(2):
                                delta = _decoder_delta(
                                    residual, write, coefficient[target_index]
                                )
                                decoded_delta[
                                    condition_index,
                                    stored_layer,
                                    target_question,
                                    source_index,
                                    target_index,
                                ] = delta.cpu().to(torch.float16).numpy()
                        natural = payload["mixer_outputs"][
                            row, layer_index, query
                        ].float()
                        error = torch.sqrt((natural_sum.cpu() - natural).square().mean())
                        scale = torch.sqrt(natural.square().mean()).clamp_min(1e-12)
                        reconstruction_relative_rmse.append(float((error / scale).item()))
            del shard
            if (shard_index + 1) % 5 == 0 or shard_index + 1 == len(selected_shards):
                print(
                    f"Final-query source writes: {shard_index + 1}/{len(selected_shards)} shards",
                    flush=True,
                )

    arrays = (attention_mass, write_rms, decoded_delta)
    if not all(np.isfinite(value).all() for value in arrays):
        raise RuntimeError("Final-query source arrays contain non-finite values")
    if max(reconstruction_relative_rmse) > 0.02:
        raise RuntimeError(
            "Source writes do not reconstruct natural attention output: "
            f"{max(reconstruction_relative_rmse)}"
        )
    if feedback_token_pairs is None:
        raise RuntimeError("Feedback display labels were not resolved")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_shards:
        benchmark = {
            "benchmark_only": True,
            "shards": len(selected_shards),
            "questions": question_count,
            "complete_model_forwards": 0,
            "max_write_reconstruction_relative_rmse": max(reconstruction_relative_rmse),
            "elapsed_seconds_including_model_load": time.perf_counter() - started,
        }
        (args.output_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
        print(json.dumps(benchmark, indent=2))
        return

    confirmation = ~discovery
    attention_point, attention_low, attention_high = _bootstrap_means(
        attention_mass, confirmation, args.seed, args.bootstrap_draws
    )
    write_point, write_low, write_high = _bootstrap_means(
        write_rms, confirmation, args.seed + 1, args.bootstrap_draws
    )
    display_labels = _display_labels(feedback_token_pairs)
    display_labels[SOURCE_NAMES.index("final_assistant_prefix")] = (
        "Final assistant prefix + final query itself"
    )
    result = {
        "question": (
            "At the exact final decision query, which complete causal-prefix regions "
            "supply ordinary-attention mass and actual residual writes?"
        ),
        "evidence_label": "Cached activation-path decomposition; not a causal intervention.",
        "coverage": {
            "questions": 500,
            "discovery": 251,
            "confirmation": 249,
            "conditions": list(CONDITION_LABELS),
            "ordinary_layers": list(ORDINARY_LAYERS),
            "source_regions": list(SOURCE_NAMES),
            "complete_model_forwards": 0,
        },
        "measurement": {
            "attention_mass": "Mean over ordinary-attention heads, summed within source region.",
            "write_rms": (
                "Per-question RMS of the exact source-specific residual write after "
                "values, gates, head combination, and output projection."
            ),
            "decoded_delta": (
                "Change in the frozen same-layer final-position old/fresh score decoder "
                "when that source write is retained versus subtracted from the natural "
                "post-layer residual."
            ),
        },
        "validation": {
            "all_values_finite": True,
            "max_write_reconstruction_relative_rmse": max(reconstruction_relative_rmse),
            "mean_write_reconstruction_relative_rmse": float(
                np.mean(reconstruction_relative_rmse)
            ),
            "elapsed_seconds_including_model_load": time.perf_counter() - started,
        },
        "display_labels": display_labels,
        "confirmation": {
            "attention_mean": attention_point.tolist(),
            "attention_ci_low": attention_low.tolist(),
            "attention_ci_high": attention_high.tolist(),
            "write_rms_mean": write_point.tolist(),
            "write_rms_ci_low": write_low.tolist(),
            "write_rms_ci_high": write_high.tolist(),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "final_query_source_writes.npz",
        question_ids=np.asarray(qids),
        discovery=discovery,
        ordinary_layers=np.asarray(ORDINARY_LAYERS, dtype=np.int16),
        source_names=np.asarray(SOURCE_NAMES),
        display_labels=np.asarray(display_labels),
        attention_mass=attention_mass,
        write_rms=write_rms,
        decoded_delta=decoded_delta,
    )
    print(json.dumps(result["validation"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--score-decoders", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-width", type=int, default=5120)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=48334072)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
