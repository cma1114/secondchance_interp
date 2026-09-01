from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .analyze_second_presentation_policy_transport import CONDITIONS, _receiver_roles
from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor, resolve_answer_tokens


POSITIONS = ("choice_cue_space", "final_decision")


def _letter_scores(states: Any, parts: Any, variant_ids: dict[str, list[int]]) -> Any:
    import torch

    device = parts.final_norm.weight.device
    width = int(states.shape[-1])
    flat = states.reshape(-1, width).to(device=device)
    normalized = parts.final_norm(flat.to(parts.final_norm.weight.dtype))
    unique_ids = sorted({token_id for ids in variant_ids.values() for token_id in ids})
    token_index = {token_id: index for index, token_id in enumerate(unique_ids)}
    ids = torch.as_tensor(unique_ids, device=parts.output_head.weight.device, dtype=torch.long)
    head = parts.output_head.weight.detach().index_select(0, ids)
    token_scores = normalized.to(head.device) @ head.T
    bias = getattr(parts.output_head, "bias", None)
    if bias is not None:
        token_scores = token_scores + bias.detach().index_select(0, ids)
    letter_scores = torch.stack(
        [
            torch.logsumexp(
                token_scores[:, [token_index[token_id] for token_id in variant_ids[letter]]],
                dim=-1,
            )
            for letter in LETTERS
        ],
        dim=-1,
    )
    return (
        letter_scores.reshape(states.shape[:-1] + (4,))
        .detach()
        .float()
        .cpu()
        .numpy()
    )


def _bare_letter_scores(states: Any, parts: Any, bare_ids: dict[str, int]) -> Any:
    """Reconstruct the cached natural A-D logits using the bare answer tokens."""
    import torch

    device = parts.final_norm.weight.device
    width = int(states.shape[-1])
    flat = states.reshape(-1, width).to(device=device)
    normalized = parts.final_norm(flat.to(parts.final_norm.weight.dtype))
    ids = torch.as_tensor(
        [bare_ids[letter] for letter in LETTERS],
        device=parts.output_head.weight.device,
        dtype=torch.long,
    )
    head = parts.output_head.weight.detach().index_select(0, ids)
    scores = normalized.to(head.device) @ head.T
    bias = getattr(parts.output_head, "bias", None)
    if bias is not None:
        scores = scores + bias.detach().index_select(0, ids)
    return scores.reshape(states.shape[:-1] + (4,)).detach().float().cpu().numpy()


def _rank_rows(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, axis=-1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(len(values))[:, None]
    ranks[rows, order] = np.arange(4)[None, :]
    return ranks


def _row_metrics(cue: np.ndarray, final: np.ndarray) -> dict[str, np.ndarray]:
    cue_ranks = _rank_rows(cue)
    final_ranks = _rank_rows(final)
    cue_centered = cue - cue.mean(axis=-1, keepdims=True)
    final_centered = final - final.mean(axis=-1, keepdims=True)
    cue_norm = np.linalg.norm(cue_centered, axis=-1)
    final_norm = np.linalg.norm(final_centered, axis=-1)
    cosine = np.sum(cue_centered * final_centered, axis=-1) / np.maximum(
        cue_norm * final_norm, 1e-12
    )
    spearman = 1.0 - 6.0 * np.sum((cue_ranks - final_ranks) ** 2, axis=-1) / 60.0
    pairwise = []
    for left in range(4):
        for right in range(left + 1, 4):
            pairwise.append(
                np.sign(cue[:, left] - cue[:, right])
                == np.sign(final[:, left] - final[:, right])
            )
    return {
        "winner_agreement": (cue.argmax(axis=-1) == final.argmax(axis=-1)).astype(float),
        "pairwise_order_agreement": np.stack(pairwise, axis=-1).mean(axis=-1),
        "spearman": spearman,
        "centered_cosine": cosine,
    }


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, float]:
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 250):
        stop = min(start + 250, draws)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(len(values)),
    }


def _summarize(cue: np.ndarray, final: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    metrics = _row_metrics(cue, final)
    rng = np.random.default_rng(seed)
    summary = {
        name: _bootstrap_mean(values, rng, draws) for name, values in metrics.items()
    }
    cue_centered = cue - cue.mean(axis=-1, keepdims=True)
    final_centered = final - final.mean(axis=-1, keepdims=True)
    summary["pooled_centered_pearson"] = float(
        np.corrcoef(cue_centered.reshape(-1), final_centered.reshape(-1))[0, 1]
    )
    transitions = np.zeros((4, 4), dtype=int)
    np.add.at(transitions, (cue.argmax(axis=-1), final.argmax(axis=-1)), 1)
    summary["winner_transition_counts"] = transitions.tolist()
    summary["cue_winner_counts"] = np.bincount(cue.argmax(axis=-1), minlength=4).tolist()
    summary["final_winner_counts"] = np.bincount(final.argmax(axis=-1), minlength=4).tolist()
    return summary


def analyze(args: argparse.Namespace) -> None:
    import torch

    started = time.perf_counter()
    config = ExperimentConfig.load(args.config)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _text, token_id in resolved[letter]})
        for letter in LETTERS
    }
    bare_ids = {
        letter: next(
            token_id
            for token_id in variant_ids[letter]
            if tokenizer.decode([token_id]) == letter
        )
        for letter in LETTERS
    }
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    discovery = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    shards = sorted((args.workspace / "shards").glob("cohort_*.pt"))
    if len(shards) != 125 or not np.all(np.load(args.workspace / "completed.npy")):
        raise RuntimeError("The complete 125-shard workspace is required")
    if args.max_shards is not None:
        shards = shards[: args.max_shards]

    question_ids: list[str] = []
    condition_scores: list[list[np.ndarray]] = [[], []]
    bare_vs_aggregated_differences = []
    for shard_index, shard_path in enumerate(shards):
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        local_qids = [str(value) for value in shard["question_ids"]]
        question_ids.extend(local_qids)
        for condition_index, condition in enumerate(CONDITIONS):
            payload = shard["payloads"][condition]
            local_states = torch.empty(
                (len(local_qids), len(POSITIONS), int(payload["residuals"].shape[-1])),
                dtype=torch.bfloat16,
            )
            for row, qid in enumerate(local_qids):
                roles = _receiver_roles(
                    payload,
                    row,
                    shard["rank_letters"][row],
                    mappings[qid]["original_to_new"],
                    tokenizer,
                )
                for position_index, position in enumerate(POSITIONS):
                    receiver_column = roles[position]
                    if len(receiver_column) != 1:
                        raise RuntimeError(f"Expected one column for {position}")
                    residual_column = int(
                        payload["receiver_in_residual"][row, receiver_column[0]]
                    )
                    local_states[row, position_index] = payload["residuals"][
                        row, 64, residual_column
                    ]
            scores = _letter_scores(local_states, parts, variant_ids)
            bare_scores = _bare_letter_scores(local_states, parts, bare_ids)
            expected = payload["natural_logits"].float().numpy()
            bare_vs_aggregated_differences.append(
                float(np.max(np.abs(bare_scores[:, 1] - expected)))
            )
            condition_scores[condition_index].append(scores)
        del shard
        if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(shards):
            print(f"Cue/final correspondence: {shard_index + 1}/{len(shards)} shards", flush=True)

    scores = np.stack([np.concatenate(rows, axis=0) for rows in condition_scores], axis=0)
    split_names = np.asarray(
        ["discovery" if qid in discovery else "confirmation" for qid in question_ids]
    )
    summaries: dict[str, Any] = {}
    for split_index, split in enumerate(("discovery", "confirmation")):
        mask = split_names == split
        if not np.any(mask):
            continue
        summaries[split] = {}
        for condition_index, condition in enumerate(CONDITIONS):
            summaries[split][condition] = _summarize(
                scores[condition_index, mask, 0],
                scores[condition_index, mask, 1],
                seed=args.seed + 100 * split_index + condition_index,
                draws=args.bootstrap_draws,
            )

    token_inventory = {
        letter: [
            {"token_id": int(token_id), "token": tokenizer.decode([token_id])}
            for token_id in ids
        ]
        for letter, ids in variant_ids.items()
    }
    result = {
        "definition": (
            "Question-level correspondence between final-layer A-D scores at the "
            "trailing space after 'Your choice (A, B, C, or D):' and at the final "
            "double-newline prediction position. Bare and space-prefixed token "
            "variants are combined by log-sum-exp for each letter at both positions."
        ),
        "evidence_label": "Descriptive final-layer readout; not a causal intervention.",
        "conditions": list(CONDITIONS),
        "positions": list(POSITIONS),
        "letters": list(LETTERS),
        "token_inventory": token_inventory,
        "question_ids": question_ids,
        "split": split_names.tolist(),
        "scores": scores.tolist(),
        "summaries": summaries,
        "max_bare_vs_aggregated_natural_logit_difference": float(
            max(bare_vs_aggregated_differences)
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=81027)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
