from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .analyze_second_presentation_policy_transport import CONDITIONS, _receiver_roles
from .config import ExperimentConfig


CONDITION_LABELS = ("Game", "Neutral")
TARGET_NAMES = ("old_unique", "fresh_unique")
RIDGE_MULTIPLIERS = np.asarray(
    [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0],
    dtype=np.float64,
)


def _center_candidates(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _rms_normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    scale = np.sqrt(np.maximum(np.mean(values * values, axis=-1, keepdims=True), 1e-12))
    return values / scale


def _align_semantic_to_current(
    values: np.ndarray,
    qids: list[str],
    mappings: dict[str, dict[str, Any]],
) -> np.ndarray:
    """Move semantic/original A-D values into the remapped 2P display slots."""
    aligned = np.empty_like(values, dtype=np.float64)
    for qi, qid in enumerate(qids):
        for original_index, original in enumerate(LETTERS):
            current = mappings[qid]["original_to_new"][original]
            aligned[qi, LETTERS.index(current)] = values[qi, original_index]
    return aligned


def _position_controls_current(
    qids: list[str], mappings: dict[str, dict[str, Any]]
) -> np.ndarray:
    """Original and current display-position controls, indexed by current slot."""
    controls = np.zeros((len(qids), 4, 6), dtype=np.float64)
    for qi, qid in enumerate(qids):
        for original_index, original in enumerate(LETTERS):
            current_index = LETTERS.index(mappings[qid]["original_to_new"][original])
            if original_index < 3:
                controls[qi, current_index, original_index] = 1.0
            if current_index < 3:
                controls[qi, current_index, 3 + current_index] = 1.0
    return controls - controls.mean(axis=1, keepdims=True)


def _residualize_target(
    target: np.ndarray,
    other: np.ndarray,
    controls: np.ndarray,
    discovery: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.concatenate([other[..., None], controls], axis=-1)
    coefficients = np.linalg.lstsq(
        design[discovery].reshape(-1, design.shape[-1]),
        target[discovery].reshape(-1),
        rcond=None,
    )[0]
    residual = target - np.einsum("qcf,f->qc", design, coefficients)
    residual = _center_candidates(residual)
    residual -= residual[discovery].mean()
    scale = residual[discovery].std()
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Target residualization produced an invalid scale")
    return residual / scale, coefficients


def _pooled_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = _center_candidates(np.asarray(left, dtype=np.float64)).reshape(-1)
    right = _center_candidates(np.asarray(right, dtype=np.float64)).reshape(-1)
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else float("nan")


def _r2(left: np.ndarray, right: np.ndarray) -> float:
    left = _center_candidates(np.asarray(left, dtype=np.float64))
    right = _center_candidates(np.asarray(right, dtype=np.float64))
    denominator = np.sum(right * right)
    return float(1.0 - np.sum((left - right) ** 2) / denominator) if denominator else float("nan")


def _folds(indices: np.ndarray, winners: np.ndarray, seed: int, count: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(count)]
    for winner in range(4):
        rows = indices[winners[indices] == winner].copy()
        rng.shuffle(rows)
        for offset, row in enumerate(rows):
            buckets[offset % count].append(int(row))
    return [np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets]


def _ridge_fit(
    x: np.ndarray, y: np.ndarray, multiplier: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    centered = x - x_mean
    kernel = centered @ centered.T
    scale = float(np.trace(kernel) / max(len(x), 1))
    regularization = max(multiplier * scale, 1e-8)
    dual = np.linalg.solve(
        kernel + regularization * np.eye(len(x), dtype=np.float32),
        y - y_mean,
    )
    coefficients = centered.T @ dual
    return x_mean, y_mean, coefficients, regularization


def _ridge_predict(
    x: np.ndarray, fit: tuple[np.ndarray, np.ndarray, np.ndarray, float]
) -> np.ndarray:
    x_mean, y_mean, coefficients, _regularization = fit
    return (x - x_mean) @ coefficients + y_mean


def _select_ridge(
    x: np.ndarray,
    y: np.ndarray,
    discovery: np.ndarray,
    winners: np.ndarray,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    discovery_indices = np.flatnonzero(discovery)
    folds = _folds(discovery_indices, winners, seed)
    records: list[dict[str, float]] = []
    for multiplier in RIDGE_MULTIPLIERS:
        predictions = np.full_like(y, np.nan, dtype=np.float32)
        for heldout in folds:
            train = discovery_indices[~np.isin(discovery_indices, heldout)]
            fit = _ridge_fit(x[train], y[train], float(multiplier))
            predictions[heldout] = _ridge_predict(x[heldout], fit)
        score = _pooled_correlation(predictions[discovery], y[discovery])
        records.append(
            {
                "multiplier": float(multiplier),
                "cross_validated_correlation": score,
                "cross_validated_r2": _r2(predictions[discovery], y[discovery]),
            }
        )
    scores = np.asarray([row["cross_validated_correlation"] for row in records])
    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite ridge cross-validation score")
    return float(RIDGE_MULTIPLIERS[int(np.argmax(scores))]), records


def _load_tokenizer(config: ExperimentConfig) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
    )


def _extract_receiver_states(
    shard_paths: list[Path],
    qids: list[str],
    mappings: dict[str, dict[str, Any]],
    tokenizer: Any,
    output_dir: Path,
    model_width: int,
    max_shards: int | None,
    receiver_role: str,
    cache_prefix: str,
) -> tuple[np.ndarray, list[str], float]:
    import torch

    selected = shard_paths if max_shards is None else shard_paths[:max_shards]
    count = len(selected) * 4 if max_shards is not None else len(qids)
    shape = (2, count, 64, model_width)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    local_qids: list[str] = []

    if max_shards is None:
        state_path = output_dir / f"{cache_prefix}_states_bf16.uint16.mmap"
        completed_path = output_dir / f"{cache_prefix}_state_completed.npy"
        expected_bytes = int(np.prod(shape, dtype=np.int64) * 2)
        if state_path.exists() and state_path.stat().st_size != expected_bytes:
            raise RuntimeError("Existing cue-state cache has the wrong size")
        states = np.memmap(
            state_path,
            dtype=np.uint16,
            mode="r+" if state_path.exists() else "w+",
            shape=shape,
        )
        completed = (
            np.load(completed_path).astype(bool)
            if completed_path.exists()
            else np.zeros(len(selected), dtype=bool)
        )
    else:
        states = np.empty(shape, dtype=np.uint16)
        completed = np.zeros(len(selected), dtype=bool)

    if max_shards is None and completed.all():
        return states, list(qids), time.perf_counter() - started

    for shard_index, shard_path in enumerate(selected):
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_qids = [str(value) for value in shard["question_ids"]]
        expected = qids[shard_index * 4 : (shard_index + 1) * 4]
        if shard_qids != expected:
            raise RuntimeError("Workspace shard order differs from Baseline order")
        local_qids.extend(shard_qids)
        if not completed[shard_index]:
            for condition_index, condition in enumerate(CONDITIONS):
                payload = shard["payloads"][condition]
                if int(payload["residuals"].shape[-1]) != model_width:
                    raise RuntimeError("Model width changed")
                for row, qid in enumerate(shard_qids):
                    roles = _receiver_roles(
                        payload,
                        row,
                        shard["rank_letters"][row],
                        mappings[qid]["original_to_new"],
                        tokenizer,
                    )
                    columns = roles[receiver_role]
                    if len(columns) != 1:
                        raise RuntimeError(
                            f"Receiver role {receiver_role!r} is not a unique token"
                        )
                    residual_column = int(
                        payload["receiver_in_residual"][row, columns[0]]
                    )
                    values = payload["residuals"][row, 1:65, residual_column]
                    target = shard_index * 4 + row
                    states[condition_index, target] = values.view(torch.uint16).numpy()
            completed[shard_index] = True
            if isinstance(states, np.memmap):
                states.flush()
                temporary = completed_path.with_suffix(".tmp.npy")
                np.save(temporary, completed)
                temporary.replace(completed_path)
        del shard
        if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(selected):
            print(
                f"{cache_prefix} score extraction: "
                f"{shard_index + 1}/{len(selected)} shards",
                flush=True,
            )

    if not completed.all():
        raise RuntimeError(f"{cache_prefix} state extraction did not complete")
    if local_qids != qids[: len(local_qids)]:
        raise RuntimeError("Extracted question order changed")
    return states, local_qids, time.perf_counter() - started


def _state_layer(states: np.ndarray, condition: int, layer: int) -> np.ndarray:
    import torch

    raw = np.asarray(states[condition, :, layer]).copy()
    return torch.from_numpy(raw).view(torch.bfloat16).float().numpy()


def _bootstrap_selected(
    predictions: np.ndarray,
    target: np.ndarray,
    rank_order: np.ndarray,
    confirmation: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    rows = np.flatnonzero(confirmation)
    ranked = np.take_along_axis(
        _center_candidates(predictions), rank_order[None, :, :], axis=2
    )
    delta = ranked[0] - ranked[1]
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, 4), dtype=np.float64)
    bivalent = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 250):
        stop = min(start + 250, draws)
        picked = rng.choice(rows, size=(stop - start, len(rows)), replace=True)
        samples[start:stop] = delta[picked].mean(axis=1)
        bivalent[start:stop] = (
            samples[start:stop, 3]
            - samples[start:stop, :2].mean(axis=1)
        )
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    bivalent_interval = np.quantile(bivalent, [0.025, 0.975])
    point = delta[rows].mean(axis=0)
    return {
        "confirmation_correlations": {
            CONDITION_LABELS[condition]: _pooled_correlation(
                predictions[condition, rows], target[rows]
            )
            for condition in range(2)
        },
        "Game_minus_Neutral_by_old_rank": {
            f"R{rank + 1}": {
                "mean": float(point[rank]),
                "ci_low": float(intervals[0, rank]),
                "ci_high": float(intervals[1, rank]),
            }
            for rank in range(4)
        },
        "Game_minus_Neutral_bivalent": {
            "definition": "R4 minus mean(R1,R2)",
            "mean": float(point[3] - point[:2].mean()),
            "ci_low": float(bivalent_interval[0]),
            "ci_high": float(bivalent_interval[1]),
        },
    }


def analyze(args: argparse.Namespace) -> None:
    import torch

    started = time.perf_counter()
    config = ExperimentConfig.load(args.config)
    tokenizer = _load_tokenizer(config)
    baseline = json.loads(args.baseline.read_text())["results"]
    qids = [str(value) for value in baseline]
    if len(qids) != 500 or len(set(qids)) != 500:
        raise RuntimeError("Expected exactly 500 unique questions")
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    confirmation = ~discovery
    if [int(discovery.sum()), int(confirmation.sum())] != [251, 249]:
        raise RuntimeError("Frozen split changed")
    shard_paths = sorted((args.workspace / "shards").glob("cohort_*.pt"))
    if len(shard_paths) != 125 or not np.load(args.workspace / "completed.npy").all():
        raise RuntimeError("Complete 125-shard workspace is required")

    states, extracted_qids, extraction_seconds = _extract_receiver_states(
        shard_paths,
        qids,
        mappings,
        tokenizer,
        args.output_dir,
        args.model_width,
        args.max_shards,
        args.receiver_role,
        args.output_prefix,
    )
    if args.max_shards is not None:
        result = {
            "benchmark_only": True,
            "shards": int(args.max_shards),
            "questions": len(extracted_qids),
            "elapsed_seconds": extraction_seconds,
            "projected_extraction_seconds": extraction_seconds * 125 / args.max_shards,
            "complete_model_forwards": 0,
            "state_bytes_full": int(2 * 500 * 64 * args.model_width * 2),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return

    remapped_baseline = json.loads(args.remapped_baseline.read_text())["results"]
    old_semantic = np.asarray([baseline[qid]["aggregated_ad_logits"] for qid in qids])
    fresh_current = np.asarray(
        [remapped_baseline[qid]["aggregated_ad_logits"] for qid in qids]
    )
    old_current = _align_semantic_to_current(old_semantic, qids, mappings)
    old_current = _center_candidates(old_current)
    fresh_current = _center_candidates(fresh_current)
    controls = _position_controls_current(qids, mappings)
    old_unique, old_coefficients = _residualize_target(
        old_current, fresh_current, controls, discovery
    )
    fresh_unique, fresh_coefficients = _residualize_target(
        fresh_current, old_current, controls, discovery
    )
    targets = (old_unique, fresh_unique)
    rank_order = np.argsort(-old_current, axis=1, kind="stable")
    winners = rank_order[:, 0]

    layer_count = 64 if args.max_layers is None else int(args.max_layers)
    if not 1 <= layer_count <= 64:
        raise ValueError("--max-layers must be between 1 and 64")
    predictions = np.empty((2, 500, layer_count, 4, 2), dtype=np.float32)
    coefficients = np.empty((layer_count, 2, args.model_width, 4), dtype=np.float16)
    input_means = np.empty((layer_count, 2, args.model_width), dtype=np.float16)
    output_means = np.empty((layer_count, 2, 4), dtype=np.float32)
    regularizations = np.empty((layer_count, 2), dtype=np.float32)
    trajectory: dict[str, Any] = {}
    cv_scores = np.empty((layer_count, 2), dtype=np.float64)
    decoder_started = time.perf_counter()
    for layer in range(layer_count):
        task_states = np.stack(
            [_rms_normalize(_state_layer(states, condition, layer)) for condition in range(2)]
        )
        shared = task_states.mean(axis=0)
        layer_rows: dict[str, Any] = {}
        for target_index, (target_name, target) in enumerate(zip(TARGET_NAMES, targets)):
            multiplier, cv_records = _select_ridge(
                shared,
                target,
                discovery,
                winners,
                args.seed + 1000 * layer + target_index,
            )
            fit = _ridge_fit(shared[discovery], target[discovery], multiplier)
            coefficients[layer, target_index] = fit[2].astype(np.float16)
            input_means[layer, target_index] = fit[0].astype(np.float16)
            output_means[layer, target_index] = fit[1].astype(np.float32)
            regularizations[layer, target_index] = float(fit[3])
            shared_prediction = _ridge_predict(shared, fit)
            cv_score = next(
                row["cross_validated_correlation"]
                for row in cv_records
                if row["multiplier"] == multiplier
            )
            cv_scores[layer, target_index] = cv_score
            task_metrics: dict[str, Any] = {}
            for condition_index, condition_label in enumerate(CONDITION_LABELS):
                task_prediction = _ridge_predict(task_states[condition_index], fit)
                predictions[condition_index, :, layer, :, target_index] = task_prediction
                task_metrics[condition_label] = {
                    "discovery_correlation": _pooled_correlation(
                        task_prediction[discovery], target[discovery]
                    ),
                    "confirmation_correlation": _pooled_correlation(
                        task_prediction[confirmation], target[confirmation]
                    ),
                    "confirmation_r2": _r2(
                        task_prediction[confirmation], target[confirmation]
                    ),
                }
            layer_rows[target_name] = {
                "selected_ridge_multiplier": multiplier,
                "discovery_cross_validated_correlation": cv_score,
                "discovery_cross_validated_r2": next(
                    row["cross_validated_r2"]
                    for row in cv_records
                    if row["multiplier"] == multiplier
                ),
                "shared_confirmation_correlation": _pooled_correlation(
                    shared_prediction[confirmation], target[confirmation]
                ),
                "shared_confirmation_r2": _r2(
                    shared_prediction[confirmation], target[confirmation]
                ),
                "tasks": task_metrics,
            }
        trajectory[str(layer + 1)] = layer_rows
        if (layer + 1) % 4 == 0 or layer == 0:
            print(
                f"{args.output_prefix} score decoders: layer {layer + 1}/64",
                flush=True,
            )

    decoder_seconds = time.perf_counter() - decoder_started
    if args.max_layers is not None:
        benchmark = {
            "benchmark_only": True,
            "questions": 500,
            "layers_completed": layer_count,
            "cue_state_extraction_seconds": extraction_seconds,
            "decoder_seconds": decoder_seconds,
            "projected_decoder_seconds_64_layers": decoder_seconds * 64 / layer_count,
            "projected_total_seconds": extraction_seconds + decoder_seconds * 64 / layer_count,
            "complete_model_forwards": 0,
            "all_predictions_finite": bool(np.isfinite(predictions).all()),
            "layer_metrics": trajectory,
        }
        (args.output_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
        print(json.dumps(benchmark))
        return

    selected: dict[str, Any] = {}
    for target_index, (target_name, target) in enumerate(zip(TARGET_NAMES, targets)):
        layer = int(np.argmax(cv_scores[:, target_index]))
        selected[target_name] = {
            "selection_rule": "maximum five-fold discovery cross-validated pooled correlation",
            "layer": layer + 1,
            "discovery_cross_validated_correlation": float(cv_scores[layer, target_index]),
            "shared_confirmation_correlation": trajectory[str(layer + 1)][target_name][
                "shared_confirmation_correlation"
            ],
            **_bootstrap_selected(
                predictions[:, :, layer, :, target_index],
                target,
                rank_order,
                confirmation,
                args.seed + 90000 + target_index,
                args.bootstrap_draws,
            ),
        }

    result = {
        "question": (
            f"Does the exact {args.position_label} contain separately decodable "
            "old first-presentation evidence, fresh second-presentation evidence, "
            "and a task-dependent adjustment organized by old rank?"
        ),
        "evidence_label": "Held-out activation decoding; not a causal intervention.",
        "position": args.position_label,
        "coverage": {
            "questions": 500,
            "discovery": 251,
            "confirmation": 249,
            "conditions": list(CONDITION_LABELS),
            "layers": 64,
            "candidate_binding": "four outputs in remapped 2P A-D display order",
            "complete_model_forwards": 0,
        },
        "method": {
            "input": (
                f"RMS-normalized {args.position_label} residual; "
                "Game/Neutral mean used for fitting"
            ),
            "targets": (
                "candidate-centered old and fresh A-D logits, each residualized on "
                "the other score plus 1P and 2P display positions using discovery only"
            ),
            "decoder": (
                "four-output ridge regression; regularization selected independently "
                "at every layer and target by five-fold discovery-only CV; frozen "
                "decoder applied separately to Game and Neutral confirmation states"
            ),
            "layer_selection": "discovery CV only; confirmation untouched",
            "ridge_multiplier_grid": RIDGE_MULTIPLIERS.tolist(),
        },
        "target_control_coefficients": {
            "old_unique": old_coefficients.tolist(),
            "fresh_unique": fresh_coefficients.tolist(),
        },
        "selected": selected,
        "trajectory": trajectory,
        "validation": {
            "question_order_matches_baseline": extracted_qids == qids,
            "all_predictions_finite": bool(np.isfinite(predictions).all()),
            "all_targets_finite": bool(np.isfinite(np.stack(targets)).all()),
            "cue_state_extraction_seconds": extraction_seconds,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.output_prefix}_score_integration.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    np.savez_compressed(
        args.output_dir / f"{args.output_prefix}_score_projections.npz",
        question_ids=np.asarray(qids),
        discovery=discovery,
        old_score_current=old_current.astype(np.float32),
        fresh_score_current=fresh_current.astype(np.float32),
        old_unique=old_unique.astype(np.float32),
        fresh_unique=fresh_unique.astype(np.float32),
        old_rank_order=rank_order.astype(np.int8),
        predictions=predictions.astype(np.float16),
    )
    torch.save(
        {
            "coefficients": torch.from_numpy(coefficients),
            "input_means": torch.from_numpy(input_means),
            "output_means": torch.from_numpy(output_means),
            "regularizations": torch.from_numpy(regularizations),
            "target_names": TARGET_NAMES,
            "receiver_role": args.receiver_role,
            "position_label": args.position_label,
            "note": (
                "Layerwise four-output receiver decoders; fit metadata is in "
                f"{args.output_prefix}_score_integration.json"
            ),
        },
        args.output_dir / f"{args.output_prefix}_score_decoders.pt",
    )
    print(json.dumps({"complete": True, "selected": selected, "validation": result["validation"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receiver-role", default="choice_cue_space")
    parser.add_argument("--output-prefix", default="cue")
    parser.add_argument(
        "--position-label",
        default="single trailing space after 'Your choice (A, B, C, or D):'",
    )
    parser.add_argument("--model-width", type=int, default=5120)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=48333965)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
