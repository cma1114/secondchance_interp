from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


LETTERS = "ABCD"
CONDITIONS = ("incorrect_again", "lost_again")
CONDITION_LABELS = ("Game", "Neutral")
SUMMARY_NAMES = ("line_mean", "content_mean", "last_content", "newline")
NEWLINE_TOKEN_ID = 198


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _align_remapped(
    values: np.ndarray,
    qids: list[str],
    mappings: dict[str, dict[str, Any]],
) -> np.ndarray:
    aligned = np.empty_like(values, dtype=np.float64)
    for qi, qid in enumerate(qids):
        for original_index, original in enumerate(LETTERS):
            new_letter = mappings[qid]["original_to_new"][original]
            aligned[qi, original_index] = values[qi, LETTERS.index(new_letter)]
    return aligned


def _valid(payload: dict[str, Any], row: int, kind: str) -> Any:
    mask = payload[f"{kind}_mask"][row].bool()
    return payload[f"{kind}_positions"][row][mask].long()


def _physical_line_groups(payload: dict[str, Any], row: int) -> list[list[int]]:
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
        raise RuntimeError(f"Could not segment four 2P lines: {groups}")
    for group in groups:
        if len(group) < 5:
            raise RuntimeError(f"Unexpectedly short 2P option line: {group}")
    return groups


def _semantic_groups(
    physical_groups: list[list[int]],
    mapping: dict[str, Any],
) -> list[list[int]]:
    groups: list[list[int]] = []
    for original in LETTERS:
        physical = LETTERS.index(mapping["original_to_new"][original])
        groups.append(physical_groups[physical])
    return groups


def _rms_normalize(values: Any) -> Any:
    import torch

    values = values.float()
    return values / torch.sqrt(values.square().mean(-1, keepdim=True).clamp_min(1e-12))


def _position_controls(
    qids: list[str], mappings: dict[str, dict[str, Any]]
) -> np.ndarray:
    controls = np.zeros((len(qids), 4, 6), dtype=np.float64)
    for qi, qid in enumerate(qids):
        for original_index, original in enumerate(LETTERS):
            if original_index < 3:
                controls[qi, original_index, original_index] = 1.0
            second = LETTERS.index(mappings[qid]["original_to_new"][original])
            if second < 3:
                controls[qi, original_index, 3 + second] = 1.0
    return controls - controls.mean(axis=1, keepdims=True)


def _residualize_target(
    target: np.ndarray,
    other: np.ndarray,
    controls: np.ndarray,
    discovery: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.concatenate([other[..., None], controls], axis=2)
    coefficients = np.linalg.lstsq(
        design[discovery].reshape(-1, design.shape[-1]),
        target[discovery].reshape(-1),
        rcond=None,
    )[0]
    fitted = np.einsum("qcf,f->qc", design, coefficients)
    residual = target - fitted
    residual -= residual[discovery].mean()
    scale = residual[discovery].std()
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Target residualization produced invalid scale")
    return residual / scale, coefficients


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1).astype(np.float64)
    right = right.reshape(-1).astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else float("nan")


def _direction_and_projection(
    states: Any,
    target: np.ndarray,
    discovery: np.ndarray,
) -> tuple[Any, np.ndarray]:
    import torch

    # states: question x candidate x width, already condition-averaged.
    normalized = _rms_normalize(states)
    normalized -= normalized.mean(1, keepdim=True)
    train = normalized[torch.from_numpy(discovery)]
    y = torch.from_numpy(target[discovery].reshape(-1)).float()
    direction = torch.einsum("nd,n->d", train.reshape(-1, train.shape[-1]), y)
    norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(norm) or float(norm) == 0.0:
        raise RuntimeError("Degenerate score direction")
    direction /= norm
    projection = torch.einsum("qcd,d->qc", normalized, direction).numpy()
    if _correlation(projection[discovery], target[discovery]) < 0:
        direction = -direction
        projection = -projection
    return direction, projection


def _bootstrap_rank_means(
    values: np.ndarray,
    rank_indices: np.ndarray,
    mask: np.ndarray,
    seed: int,
    draws: int = 2000,
) -> dict[str, Any]:
    rows = np.flatnonzero(mask)
    ranked = np.stack(
        [values[np.arange(len(values)), rank_indices[:, rank]] for rank in range(4)],
        axis=1,
    )
    point = ranked[rows].mean(0)
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, 4), dtype=np.float64)
    for start in range(0, draws, 250):
        stop = min(start + 250, draws)
        picked = rng.choice(rows, size=(stop - start, len(rows)), replace=True)
        samples[start:stop] = ranked[picked].mean(1)
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        f"R{rank + 1}": {
            "mean": float(point[rank]),
            "ci": [float(intervals[0, rank]), float(intervals[1, rank])],
        }
        for rank in range(4)
    }


def _load_scores(
    qids: list[str],
    baseline_path: Path,
    remapped_baseline_path: Path,
    mappings: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped = json.loads(remapped_baseline_path.read_text())["results"]
    old = np.asarray([baseline[qid]["aggregated_ad_logits"] for qid in qids])
    current_raw = np.asarray([remapped[qid]["aggregated_ad_logits"] for qid in qids])
    return _center(old), _center(_align_remapped(current_raw, qids, mappings))


def _extract_summary_states(
    shard_paths: list[Path],
    qid_to_index: dict[str, int],
    mappings: dict[str, dict[str, Any]],
    max_shards: int | None,
    model_width: int,
    output_dir: Path,
) -> tuple[Any, np.ndarray, int, float]:
    import torch

    selected = shard_paths if max_shards is None else shard_paths[:max_shards]
    storage_questions = len(qid_to_index) if max_shards is None else len(selected) * 4
    state_shape = (2, storage_questions, 64, 4, len(SUMMARY_NAMES), model_width)
    output_dir.mkdir(parents=True, exist_ok=True)
    if max_shards is None:
        cache_path = output_dir / "summary_states_bf16.uint16.mmap"
        completed_path = output_dir / "summary_state_completed.npy"
        ranks_path = output_dir / "summary_state_rank_indices.npy"
        expected_bytes = int(np.prod(state_shape, dtype=np.int64) * 2)
        if cache_path.exists() and cache_path.stat().st_size != expected_bytes:
            raise RuntimeError("Existing summary-state cache has the wrong size")
        states = np.memmap(
            cache_path,
            dtype=np.uint16,
            mode="r+" if cache_path.exists() else "w+",
            shape=state_shape,
        )
        completed = (
            np.load(completed_path).astype(bool)
            if completed_path.exists()
            else np.zeros(len(selected), dtype=bool)
        )
        rank_indices = (
            np.load(ranks_path).astype(np.int64)
            if ranks_path.exists()
            else np.full((storage_questions, 4), -1, dtype=np.int64)
        )
        seen = np.zeros(storage_questions, dtype=bool)
        for shard_index, done in enumerate(completed):
            if done:
                seen[shard_index * 4 : (shard_index + 1) * 4] = True
    else:
        states = torch.empty(state_shape, dtype=torch.bfloat16)
        completed = np.zeros(len(selected), dtype=bool)
        rank_indices = np.full((storage_questions, 4), -1, dtype=np.int64)
        seen = np.zeros(storage_questions, dtype=bool)
    max_line_length = 0
    started = time.perf_counter()
    for shard_index, path in enumerate(selected):
        if completed[shard_index]:
            continue
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if int(shard["model_width"]) != model_width:
            raise RuntimeError(
                f"Model width changed: {shard['model_width']} != {model_width}"
            )
        local_qids = [str(value) for value in shard["question_ids"]]
        expected = list(qid_to_index)[shard_index * 4 : (shard_index + 1) * 4]
        if local_qids != expected:
            raise RuntimeError("Workspace shard order differs from Baseline order")
        shard_states = torch.empty(
            (2, len(local_qids), 64, 4, len(SUMMARY_NAMES), model_width),
            dtype=torch.bfloat16,
        )
        for condition_index, condition in enumerate(CONDITIONS):
            payload = shard["payloads"][condition]
            for row, qid in enumerate(local_qids):
                target = (
                    qid_to_index[qid]
                    if max_shards is None
                    else shard_index * 4 + row
                )
                if condition_index == 0:
                    if seen[target]:
                        raise RuntimeError(f"Duplicate question in workspace: {qid}")
                    seen[target] = True
                    rank_indices[target] = [
                        LETTERS.index(str(value))
                        for value in shard["rank_letters"][row]
                    ]
                groups = _semantic_groups(
                    _physical_line_groups(payload, row), mappings[qid]
                )
                max_line_length = max(max_line_length, max(map(len, groups)))
                for candidate, group in enumerate(groups):
                    residual_columns = payload["receiver_in_residual"][row, group].long()
                    line = payload["residuals"][row, 1:65].index_select(
                        1, residual_columns
                    )
                    content = line[:, 3:-1]
                    shard_states[condition_index, row, :, candidate, 0] = line.mean(1)
                    shard_states[condition_index, row, :, candidate, 1] = content.mean(1)
                    shard_states[condition_index, row, :, candidate, 2] = line[:, -2]
                    shard_states[condition_index, row, :, candidate, 3] = line[:, -1]
        start = shard_index * 4
        stop = start + len(local_qids)
        if max_shards is None:
            states[:, start:stop] = shard_states.view(torch.uint16).numpy()
            states.flush()
        else:
            states[:, start:stop] = shard_states
        completed[shard_index] = True
        if max_shards is None:
            temporary = output_dir / "summary_state_completed.npy.tmp"
            with temporary.open("wb") as handle:
                np.save(handle, completed)
            temporary.replace(completed_path)
            temporary = output_dir / "summary_state_rank_indices.npy.tmp"
            with temporary.open("wb") as handle:
                np.save(handle, rank_indices)
            temporary.replace(ranks_path)
        del shard
        if (shard_index + 1) % 5 == 0 or shard_index + 1 == len(selected):
            print(
                f"Score-state extraction: {shard_index + 1}/{len(selected)} shards",
                flush=True,
            )
    if not seen.all() or np.any(rank_indices < 0):
        raise RuntimeError("Workspace extraction did not cover each expected question once")
    return states, rank_indices, max_line_length, time.perf_counter() - started


def _state_slice(states: Any, condition: int, layer: int, summary: int) -> Any:
    import torch

    if isinstance(states, np.memmap):
        # Copy the small layer slice so PyTorch can safely reinterpret the raw
        # native-bfloat16 bit pattern stored in the resumable memmap.
        raw = np.asarray(states[condition, :, layer, :, summary]).copy()
        return torch.from_numpy(raw).view(torch.bfloat16)
    return states[condition, :, layer, :, summary]


def analyze(args: argparse.Namespace) -> None:
    import torch

    shard_paths = sorted((args.workspace / "shards").glob("cohort_*.pt"))
    if len(shard_paths) != 125:
        raise RuntimeError(f"Expected 125 shards, found {len(shard_paths)}")
    if not np.load(args.workspace / "completed.npy").astype(bool).all():
        raise RuntimeError("Workspace is incomplete")

    baseline_payload = json.loads(args.baseline.read_text())
    qids = [str(value) for value in baseline_payload["results"]]
    if len(qids) != 500 or len(set(qids)) != 500:
        raise RuntimeError("Question inventory is not exactly 500 unique questions")
    qid_to_index = {qid: index for index, qid in enumerate(qids)}
    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    confirmation = ~discovery
    if [int(discovery.sum()), int(confirmation.sum())] != [251, 249]:
        raise RuntimeError("Frozen discovery/confirmation split changed")

    states, rank_indices, max_line_length, extraction_seconds = _extract_summary_states(
        shard_paths,
        qid_to_index,
        mappings,
        args.max_shards,
        args.model_width,
        args.output_dir,
    )
    if args.max_shards is not None:
        result = {
            "benchmark_only": True,
            "shards": args.max_shards,
            "seconds": extraction_seconds,
            "projected_seconds_125_shards": extraction_seconds * 125 / args.max_shards,
            "state_bytes_full": int(states.numel() * states.element_size()),
            "width": args.model_width,
            "max_line_length_seen": max_line_length,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return

    old_score, current_score = _load_scores(
        qids, args.baseline, args.remapped_baseline, mappings
    )
    controls = _position_controls(qids, mappings)
    old_unique, old_control_coefficients = _residualize_target(
        old_score, current_score, controls, discovery
    )
    current_unique, current_control_coefficients = _residualize_target(
        current_score, old_score, controls, discovery
    )
    targets = (old_unique, current_unique)
    target_names = ("old_unique", "fresh_unique")
    directions = torch.empty(
        (64, len(SUMMARY_NAMES), 2, args.model_width), dtype=torch.float32
    )
    projections = np.empty(
        (2, len(qids), 64, 4, len(SUMMARY_NAMES), 2), dtype=np.float32
    )
    result: dict[str, Any] = {
        "definition": {
            "old_score": "candidate-centered original-presentation Baseline A-D logit",
            "fresh_score": "candidate-centered remapped-presentation Baseline A-D logit without Second Chance history, aligned to semantic identity",
            "direction_fit": "condition-mean, RMS-normalized, question-centered 2P residual; score residualized on the other score and both displayed positions; discovery only",
            "task_application": "shared frozen direction applied separately to Game and Neutral residuals",
        },
        "validation": {
            "questions": 500,
            "discovery": 251,
            "confirmation": 249,
            "layers": 64,
            "summaries": list(SUMMARY_NAMES),
            "model_width": args.model_width,
            "extraction_seconds": extraction_seconds,
            "max_line_length": max_line_length,
        },
        "target_control_coefficients": {
            "old_unique": old_control_coefficients.tolist(),
            "fresh_unique": current_control_coefficients.tolist(),
        },
        "trajectory": {},
    }

    for layer in range(64):
        layer_result: dict[str, Any] = {}
        for summary_index, summary_name in enumerate(SUMMARY_NAMES):
            summary_result: dict[str, Any] = {}
            game_state = _state_slice(states, 0, layer, summary_index)
            neutral_state = _state_slice(states, 1, layer, summary_index)
            source = (game_state.float() + neutral_state.float()) / 2.0
            for target_index, (target_name, target) in enumerate(
                zip(target_names, targets)
            ):
                direction, shared_projection = _direction_and_projection(
                    source, target, discovery
                )
                directions[layer, summary_index, target_index] = direction
                task_rows: dict[str, Any] = {
                    "shared_discovery_correlation": _correlation(
                        shared_projection[discovery], target[discovery]
                    ),
                    "shared_confirmation_correlation": _correlation(
                        shared_projection[confirmation], target[confirmation]
                    ),
                }
                for condition_index, condition_label in enumerate(CONDITION_LABELS):
                    task_state = game_state if condition_index == 0 else neutral_state
                    normalized = _rms_normalize(task_state)
                    normalized -= normalized.mean(1, keepdim=True)
                    task_projection = torch.einsum(
                        "qcd,d->qc", normalized, direction
                    ).numpy()
                    projections[
                        condition_index, :, layer, :, summary_index, target_index
                    ] = task_projection
                    task_rows[condition_label] = {
                        "discovery_correlation": _correlation(
                            task_projection[discovery], target[discovery]
                        ),
                        "confirmation_correlation": _correlation(
                            task_projection[confirmation], target[confirmation]
                        ),
                    }
                delta = (
                    projections[0, :, layer, :, summary_index, target_index]
                    - projections[1, :, layer, :, summary_index, target_index]
                )
                task_rows["Game_minus_Neutral_confirmation_by_first_rank"] = (
                    _bootstrap_rank_means(
                        delta,
                        rank_indices,
                        confirmation,
                        args.seed + layer * 100 + summary_index * 10 + target_index,
                        args.bootstrap_draws,
                    )
                )
                summary_result[target_name] = task_rows
            layer_result[summary_name] = summary_result
        result["trajectory"][str(layer + 1)] = layer_result
        if (layer + 1) % 4 == 0 or layer == 0:
            print(f"Score directions: layer {layer + 1}/64", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "score_residual_trajectory.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    torch.save(directions.to(torch.float16), args.output_dir / "score_directions.pt")
    np.savez_compressed(
        args.output_dir / "score_projections.npz",
        question_ids=np.asarray(qids),
        discovery=discovery,
        rank_indices=rank_indices,
        old_score=old_score.astype(np.float32),
        fresh_score=current_score.astype(np.float32),
        old_unique=old_unique.astype(np.float32),
        fresh_unique=current_unique.astype(np.float32),
        projections=projections.astype(np.float16),
    )
    print(
        json.dumps(
            {
                "complete": True,
                "questions": len(qids),
                "layers": 64,
                "summaries": list(SUMMARY_NAMES),
                "extraction_seconds": extraction_seconds,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--model-width", type=int, default=5120)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=48333964)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
