from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_cue_score_integration import _rms_normalize
from .analyze_second_presentation_policy_transport import CONDITIONS, _receiver_roles
from .config import ExperimentConfig


CONDITION_LABELS = ("Game", "Neutral")
COMPONENT_LABELS = ("mixer", "mlp")
TARGET_LABELS = ("old_unique", "fresh_unique")


def _decoded_delta(
    residual: np.ndarray, write: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    return (_rms_normalize(residual) - _rms_normalize(residual - write)) @ coefficients


def _bootstrap_rank_means(
    values: np.ndarray,
    rank_order: np.ndarray,
    indices: np.ndarray,
    seed: int,
    draws: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap condition x layer x component x target x rank means."""
    aligned = np.empty_like(values, dtype=np.float32)
    for question in range(values.shape[1]):
        aligned[:, question] = np.take(
            values[:, question], rank_order[question], axis=-1
        )
    selected = aligned[:, indices]
    point = selected.mean(axis=1)
    rng = np.random.default_rng(seed)
    samples = np.empty((draws,) + point.shape, dtype=np.float32)
    for start in range(0, draws, 100):
        stop = min(start + 100, draws)
        rows = rng.integers(0, len(indices), size=(stop - start, len(indices)))
        # task x draw x question x layer x component x target x rank
        sample = selected[:, rows].mean(axis=2)
        samples[start:stop] = sample.transpose(1, 0, 2, 3, 4, 5)
    low, high = np.quantile(samples, (0.025, 0.975), axis=0)
    return point, low, high


def analyze(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    started = time.perf_counter()
    config = ExperimentConfig.load(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
    )
    score_arrays = np.load(args.score_projections)
    qids = [str(value) for value in score_arrays["question_ids"]]
    discovery = score_arrays["discovery"].astype(bool)
    rank_order = score_arrays["old_rank_order"].astype(np.int64)
    if len(qids) != 500 or [int(discovery.sum()), int((~discovery).sum())] != [251, 249]:
        raise RuntimeError("Final score projections do not use the frozen canonical split")
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    decoder = torch.load(args.score_decoders, map_location="cpu", weights_only=False)
    if decoder.get("receiver_role") != "final_decision":
        raise RuntimeError("Decoder is not for the final decision position")
    coefficients = decoder["coefficients"].float().numpy()
    if tuple(coefficients.shape) != (64, 2, args.model_width, 4):
        raise RuntimeError(f"Unexpected decoder shape: {coefficients.shape}")

    shard_paths = sorted((args.workspace / "shards").glob("cohort_*.pt"))
    if len(shard_paths) != 125 or not np.load(args.workspace / "completed.npy").all():
        raise RuntimeError("Complete 125-shard workspace is required")
    selected_shards = shard_paths[: args.max_shards] if args.max_shards else shard_paths
    question_count = 4 * len(selected_shards)
    decoded = np.full((2, question_count, 64, 2, 2, 4), np.nan, dtype=np.float16)
    write_rms = np.full((2, question_count, 64, 2), np.nan, dtype=np.float32)

    for shard_index, shard_path in enumerate(selected_shards):
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_qids = [str(value) for value in shard["question_ids"]]
        expected = qids[4 * shard_index : 4 * shard_index + len(shard_qids)]
        if shard_qids != expected:
            raise RuntimeError("Workspace and score-decoder question order differ")
        for condition_index, condition in enumerate(CONDITIONS):
            payload = shard["payloads"][condition]
            for row, qid in enumerate(shard_qids):
                roles = _receiver_roles(
                    payload,
                    row,
                    shard["rank_letters"][row],
                    mappings[qid]["original_to_new"],
                    tokenizer,
                )
                columns = roles["final_decision"]
                if len(columns) != 1:
                    raise RuntimeError("Final decision is not a unique receiver")
                receiver_column = int(columns[0])
                residual_column = int(
                    payload["receiver_in_residual"][row, receiver_column]
                )
                target_question = 4 * shard_index + row
                for layer in range(64):
                    residual = payload["residuals"][
                        row, layer + 1, residual_column
                    ].float().numpy()
                    components = (
                        payload["mixer_outputs"][row, layer, receiver_column]
                        .float()
                        .numpy(),
                        payload["mlp_outputs"][row, layer, receiver_column]
                        .float()
                        .numpy(),
                    )
                    for component_index, write in enumerate(components):
                        write_rms[
                            condition_index, target_question, layer, component_index
                        ] = float(np.sqrt(np.mean(write * write)))
                        for target_index in range(2):
                            decoded[
                                condition_index,
                                target_question,
                                layer,
                                component_index,
                                target_index,
                            ] = _decoded_delta(
                                residual,
                                write,
                                coefficients[layer, target_index],
                            ).astype(np.float16)
        del shard
        if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(selected_shards):
            print(
                f"Final component trajectory: {shard_index + 1}/{len(selected_shards)} shards",
                flush=True,
            )

    if not np.isfinite(decoded).all() or not np.isfinite(write_rms).all():
        raise RuntimeError("Final component arrays contain non-finite values")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_shards:
        benchmark = {
            "benchmark_only": True,
            "shards": len(selected_shards),
            "questions": question_count,
            "layers": 64,
            "components": list(COMPONENT_LABELS),
            "targets": list(TARGET_LABELS),
            "complete_model_forwards": 0,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (args.output_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
        print(json.dumps(benchmark, indent=2))
        return

    point, low, high = _bootstrap_rank_means(
        decoded.astype(np.float32),
        rank_order,
        np.flatnonzero(~discovery),
        args.seed,
        args.bootstrap_draws,
    )
    result = {
        "question": (
            "At the final decision position, which mixer and MLP writes add or remove "
            "decodable old and fresh candidate evidence at each model layer?"
        ),
        "evidence_label": "Held-out component-write decoding; not a causal intervention.",
        "coverage": {
            "questions": 500,
            "discovery": 251,
            "confirmation": 249,
            "conditions": list(CONDITION_LABELS),
            "layers": list(range(1, 65)),
            "components": list(COMPONENT_LABELS),
            "targets": list(TARGET_LABELS),
            "complete_model_forwards": 0,
        },
        "measurement": (
            "For each natural post-layer residual, subtract only that layer's cached "
            "mixer or MLP write and measure the change under the frozen same-layer "
            "four-candidate score decoder. Candidate outputs are then ordered by the "
            "first-presentation rank R1-R4."
        ),
        "confirmation_rank_mean": point.tolist(),
        "confirmation_rank_ci_low": low.tolist(),
        "confirmation_rank_ci_high": high.tolist(),
        "validation": {
            "all_values_finite": True,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "final_component_trajectory.npz",
        question_ids=np.asarray(qids),
        discovery=discovery,
        old_rank_order=rank_order.astype(np.int8),
        component_names=np.asarray(COMPONENT_LABELS),
        target_names=np.asarray(TARGET_LABELS),
        decoded_delta=decoded,
        write_rms=write_rms,
    )
    print(json.dumps(result["validation"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--score-decoders", type=Path, required=True)
    parser.add_argument("--score-projections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-width", type=int, default=5120)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=48334073)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
