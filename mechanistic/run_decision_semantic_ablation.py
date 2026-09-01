from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .io import atomic_save_npz, json_array, shard_path
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import load_trials, prompt_hash
from .run_historical_answer_intervention import _forward, _prompt_position


LETTERS = "ABCD"
SCENARIOS = (
    "natural",
    "erase_winner_semantic",
    "erase_runner_semantic",
    "erase_all_option_semantics",
    "orthogonal_winner_matched",
)


def _load_option_sets(roots: list[Path]) -> list[tuple[np.ndarray, dict[str, Any]]]:
    if len(roots) != 4:
        raise ValueError("Exactly four option-representation roots are required")
    sets = []
    for root in roots:
        values = np.load(root / "position_residuals.npy", mmap_mode="r")
        metadata = json.loads((root / "metadata.json").read_text())
        if values.shape[0] != len(metadata["question_ids"]):
            raise ValueError(f"Residual/metadata length mismatch in {root}")
        sets.append((values, metadata))
    return sets


def _verify_sets(
    sets: list[tuple[np.ndarray, dict[str, Any]]], qids: list[str], anchor: str
) -> tuple[list[dict[str, int]], list[list[int]]]:
    lookups: list[dict[str, int]] = []
    anchor_indices: list[list[int]] = []
    for _values, metadata in sets:
        lookup = {qid: index for index, qid in enumerate(metadata["question_ids"])}
        if not set(qids) <= set(lookup):
            raise ValueError("An option-representation set is missing requested questions")
        lookups.append(lookup)
        anchor_indices.append(
            [metadata["anchors"].index(f"{anchor}_{letter}") for letter in LETTERS]
        )
    for qid in qids:
        for original in LETTERS:
            occupied = []
            for _values, metadata in sets:
                mapping = metadata.get("mappings", {}).get(qid)
                occupied.append(
                    original if mapping is None else mapping["original_to_new"][original]
                )
            if set(occupied) != set(LETTERS):
                raise ValueError(f"{qid}: content {original} does not occupy A-D once")
    return lookups, anchor_indices


def _semantic_geometry(
    sets: list[tuple[np.ndarray, dict[str, Any]]],
    lookups: list[dict[str, int]],
    anchor_indices: list[list[int]],
    qid: str,
    readout: int,
) -> tuple[np.ndarray, np.ndarray]:
    aligned = []
    layer = readout - 1
    for set_index, (values, metadata) in enumerate(sets):
        raw = np.asarray(
            values[lookups[set_index][qid], layer, anchor_indices[set_index]],
            dtype=np.float32,
        )
        mapping = metadata.get("mappings", {}).get(qid)
        if mapping is not None:
            indices = [LETTERS.index(mapping["original_to_new"][letter]) for letter in LETTERS]
            raw = raw[indices]
        aligned.append(raw)
    average = np.mean(aligned, axis=0, dtype=np.float32)
    centered = average - average.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise RuntimeError(f"{qid} L{readout}: degenerate centered option vector")
    units = centered / norms
    _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * np.finfo(np.float32).eps * singular[0]
    rank = int((singular > tolerance).sum())
    rank = min(rank, 3)
    if rank < 1:
        raise RuntimeError(f"{qid} L{readout}: degenerate option-content subspace")
    return units.astype(np.float32), vh[:rank].astype(np.float32)


def _orthogonal_unit(basis: Any, qid: str, readout: int, seed: int) -> Any:
    import torch

    digest = hashlib.sha256(f"{seed}:{qid}:{readout}:semantic-control".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    random = torch.from_numpy(rng.standard_normal(basis.shape[1])).float().to(basis.device)
    random = random - (random @ basis.T) @ basis
    norm = random.norm()
    if norm <= 1e-8:
        raise RuntimeError("Degenerate option-semantic-orthogonal control")
    return random / norm


class ContinuousDecisionSemanticAblator:
    """Remove question-specific option content at one historical decision token."""

    def __init__(
        self,
        parts: Any,
        readouts: list[int],
        position: int,
        geometry: dict[int, tuple[np.ndarray, np.ndarray]],
        winner: int,
        runner: int,
        qid: str,
        seed: int,
    ) -> None:
        import torch

        self.position = int(position)
        self.winner = int(winner)
        self.runner = int(runner)
        self.qid = qid
        self.seed = int(seed)
        self.geometry = {
            readout: (
                torch.from_numpy(units).float().to(model_input_device(parts)),
                torch.from_numpy(basis).float().to(model_input_device(parts)),
            )
            for readout, (units, basis) in geometry.items()
        }
        self.records: dict[int, Any] = {}
        self.handles = [
            parts.layers[readout - 1].register_forward_hook(self._hook(readout))
            for readout in readouts
        ]

    def _hook(self, readout: int):
        def intervene(_module: Any, _inputs: Any, output: Any) -> Any:
            import torch

            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.shape[0] != 5:
                raise ValueError("Semantic ablation expects a five-row physical batch")
            current = hidden[:, self.position].float()
            units, basis = self.geometry[readout]
            winner_unit = units[self.winner]
            runner_unit = units[self.runner]
            winner_delta = -torch.dot(current[0], winner_unit) * winner_unit
            runner_delta = -torch.dot(current[1], runner_unit) * runner_unit
            full_delta = -((current[2] @ basis.T) @ basis)
            control_unit = _orthogonal_unit(
                basis, self.qid, readout, self.seed
            )
            control_delta = control_unit * winner_delta.norm()
            deltas = torch.stack(
                [
                    winner_delta,
                    runner_delta,
                    full_delta,
                    control_delta,
                    torch.zeros_like(winner_delta),
                ]
            )
            changed = hidden.clone()
            changed[:, self.position] = (current + deltas).to(hidden.dtype)
            self.records[readout] = torch.stack(
                [
                    winner_delta.norm(),
                    runner_delta.norm(),
                    full_delta.norm(),
                    control_delta.norm(),
                    torch.dot(current[0] + winner_delta, winner_unit).abs(),
                ]
            ).detach().cpu()
            if isinstance(output, tuple):
                return (changed, *output[1:])
            if isinstance(output, list):
                return [changed, *output[1:]]
            return changed

        return intervene

    def arrays(self) -> dict[str, np.ndarray]:
        import torch

        ordered = [self.records[key] for key in sorted(self.geometry)]
        if len(ordered) != len(self.geometry):
            raise RuntimeError("Not every selected readout executed its hook")
        values = torch.stack(ordered).numpy()
        return {
            "winner_delta_norm": values[:, 0],
            "runner_delta_norm": values[:, 1],
            "full_delta_norm": values[:, 2],
            "control_delta_norm": values[:, 3],
            "winner_projection_abs_after": values[:, 4],
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __enter__(self) -> "ContinuousDecisionSemanticAblator":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def run(
    config_path: Path,
    plan_path: Path,
    baseline_logits_path: Path,
    option_roots: list[Path],
    output: Path,
    first_readout: int,
    last_readout: int,
    anchor: str,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")
    plan = json.loads(plan_path.read_text())
    qids = plan.get("question_ids", plan.get("confirmation_question_ids"))
    if not qids:
        raise ValueError("Plan has no question IDs")
    trials = load_trials(config.manifest_path, config.baseline_results_path, qids, None)
    baseline_rows = json.loads(baseline_logits_path.read_text())["results"]
    sets = _load_option_sets(option_roots)
    lookups, anchor_indices = _verify_sets(sets, qids, anchor)
    readouts = list(range(first_readout, last_readout + 1))

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in LETTERS]

    output.mkdir(parents=True, exist_ok=True)
    scenario_ids = [
        f"{condition}_{scenario}"
        for condition in ("game", "neutral")
        for scenario in SCENARIOS
    ]
    metadata = {
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "baseline_logits_path": str(baseline_logits_path),
        "question_ids": qids,
        "option_roots": [str(path) for path in option_roots],
        "option_anchor": anchor,
        "first_answer_position": "historical_answer_end",
        "selected_post_block_readouts": readouts,
        "scenarios": scenario_ids,
        "vector_definition": (
            "For each semantic option, average its raw option-anchor residual across "
            "four mappings where it occupies A-D once, subtract the within-question "
            "mean of the four options, and normalize."
        ),
        "intervention": (
            "At each selected post-block readout, remove the live projection at the "
            "historical first-answer decision position."
        ),
        "batch_control": (
            "Saved intervention logits equal a single-example natural forward plus "
            "intervention-minus-untouched-control logits from the same physical batch."
        ),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    audit_path = output / "position_audit.json"
    for completed, trial in enumerate(trials, 1):
        qid = trial.question_id
        if all(shard_path(output, scenario, qid).exists() for scenario in scenario_ids):
            continue
        baseline_logits = np.asarray(
            baseline_rows[qid]["aggregated_ad_logits"], dtype=np.float32
        )
        order = np.argsort(-baseline_logits, kind="stable")
        winner, runner = int(order[0]), int(order[1])
        geometry = {
            readout: _semantic_geometry(
                sets, lookups, anchor_indices, qid, readout
            )
            for readout in readouts
        }

        for condition, prefix in (("incorrect", "game"), ("neutral", "neutral")):
            prompt, position, token_ids, input_ids, attention_mask = _prompt_position(
                processor,
                tokenizer,
                config,
                trial,
                condition,
                "historical_answer_end",
            )
            natural_output = _forward(model, parts, input_ids, attention_mask)
            natural_logits = (
                natural_output.logits[0, -1, canonical_ids]
                .detach().float().cpu().numpy()
            )
            batch_ids, batch_mask, _ = tokenize_batch(tokenizer, [prompt] * 5)
            with ContinuousDecisionSemanticAblator(
                parts,
                readouts,
                position,
                geometry,
                winner,
                runner,
                qid,
                config.seed,
            ) as ablator:
                batch_output = _forward(model, parts, batch_ids, batch_mask)
                audit_arrays = ablator.arrays()
            raw = (
                batch_output.logits[:, -1, canonical_ids]
                .detach().float().cpu().numpy()
            )
            corrected = natural_logits[None, :] + raw[:4] - raw[4:5]
            common = {
                "question_id": qid,
                "condition": condition,
                "prompt_hash": prompt_hash(prompt),
                "decision_position": int(position),
                "decision_token_id": int(token_ids[position]),
                "decision_token": tokenizer.decode([token_ids[position]]),
                "baseline_winner": LETTERS[winner],
                "baseline_runner": LETTERS[runner],
                "baseline_rank_order": [LETTERS[index] for index in order],
                "correct_answer": trial.question["correct_answer"],
            }
            atomic_save_npz(
                shard_path(output, f"{prefix}_natural", qid),
                final_canonical_logits=natural_logits.astype(np.float32),
                metadata=json_array({**common, "scenario": "natural"}),
            )
            for row, scenario in enumerate(SCENARIOS[1:]):
                norm_key = {
                    "erase_winner_semantic": "winner_delta_norm",
                    "erase_runner_semantic": "runner_delta_norm",
                    "erase_all_option_semantics": "full_delta_norm",
                    "orthogonal_winner_matched": "control_delta_norm",
                }[scenario]
                atomic_save_npz(
                    shard_path(output, f"{prefix}_{scenario}", qid),
                    final_canonical_logits=corrected[row].astype(np.float32),
                    raw_batch_canonical_logits=raw[row].astype(np.float32),
                    batch_control_canonical_logits=raw[4].astype(np.float32),
                    layer_delta_norm=audit_arrays[norm_key].astype(np.float32),
                    winner_projection_abs_after=audit_arrays[
                        "winner_projection_abs_after"
                    ].astype(np.float32),
                    metadata=json_array({**common, "scenario": scenario}),
                )
            if not audit_path.exists():
                audit_path.write_text(
                    json.dumps(
                        {
                            **common,
                            "rendered_prompt": prompt,
                            "selected_post_block_readouts": readouts,
                            "max_winner_projection_abs_after": float(
                                audit_arrays["winner_projection_abs_after"].max()
                            ),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
        if completed == 1 or completed % 5 == 0 or completed == len(trials):
            print(f"Decision semantic ablation: {completed}/{len(trials)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline-logits", type=Path, required=True)
    parser.add_argument("--option-roots", nargs=4, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-readout", type=int, default=24)
    parser.add_argument("--last-readout", type=int, default=55)
    parser.add_argument("--anchor", choices=("content_end", "line_end"), default="line_end")
    args = parser.parse_args()
    run(
        args.config,
        args.plan,
        args.baseline_logits,
        args.option_roots,
        args.output,
        args.first_readout,
        args.last_readout,
        args.anchor,
    )


if __name__ == "__main__":
    main()
