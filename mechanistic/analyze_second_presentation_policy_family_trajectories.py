from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_second_presentation_policy_token_cross import (
    CONDITIONS,
    _receiver_columns,
)
from .analyze_second_presentation_policy_transport import (
    _load_lens,
    _receiver_roles,
)
from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor


LENSES = ("J-lens", "R-lens")
KINDS = ("letter", "semantic", "newline")
FAMILIES = {
    "incorrect_failed_mistake_wrong": (
        "incorrect",
        "incorrectly",
        "fail",
        "fails",
        "failed",
        "failing",
        "failure",
        "failures",
        "mistake",
        "mistakes",
        "mistaken",
        "mistakenly",
        "wrong",
        "wrongly",
        "wrongness",
    ),
    "lost_again_resend_repeat": (
        "lost",
        "lose",
        "loses",
        "losing",
        "loss",
        "losses",
        "again",
        "resend",
        "resends",
        "resending",
        "repeat",
        "repeats",
        "repeated",
        "repeating",
    ),
}


def _token_inventory(tokenizer: Any) -> tuple[list[int], dict[str, dict[str, list[int]]]]:
    token_ids: list[int] = []
    inventory: dict[str, dict[str, list[int]]] = {}
    for family, lexemes in FAMILIES.items():
        inventory[family] = {}
        for lexeme in lexemes:
            matches = []
            for token_id in range(len(tokenizer)):
                text = tokenizer.decode([token_id])
                # These words occur after whitespace in the prompts and in normal
                # prose. Restricting to whitespace-prefixed vocabulary entries
                # avoids mixing sentence-initial and word-fragment embeddings.
                if text.startswith(" ") and text.strip().lower() == lexeme:
                    matches.append(token_id)
            if matches:
                inventory[family][lexeme] = matches
                token_ids.extend(matches)
    token_ids = sorted(set(token_ids))
    for family in FAMILIES:
        if len(inventory[family]) < 4:
            raise RuntimeError(
                f"Too few single-token lexemes for {family}: {inventory[family]}"
            )
    return token_ids, inventory


def _destination_names() -> tuple[str, ...]:
    return tuple(
        f"R{rank}_{kind}" for rank in range(1, 5) for kind in KINDS
    ) + ("choice_cue_space", "final_decision")


def _destination_states(
    shard: dict[str, Any],
    condition: str,
    mappings: dict[str, dict[str, Any]],
    tokenizer: Any,
) -> tuple[list[str], Any]:
    import torch

    payload = shard["payloads"][condition]
    local_qids = [str(value) for value in shard["question_ids"]]
    names = _destination_names()
    width = int(payload["residuals"].shape[-1])
    states = torch.empty(
        (len(local_qids), 64, len(names), width), dtype=torch.bfloat16
    )
    for row, qid in enumerate(local_qids):
        _, physical_groups = _receiver_columns(payload, row)
        output_index = 0
        for rank_index, first_letter in enumerate(shard["rank_letters"][row]):
            second_letter = mappings[qid]["original_to_new"][first_letter]
            group = physical_groups[ord(second_letter) - ord("A")]
            columns_by_kind = {
                "letter": [group[1]],
                "semantic": group[3:-1],
                "newline": [group[-1]],
            }
            for kind in KINDS:
                residual_columns = payload["receiver_in_residual"][
                    row, columns_by_kind[kind]
                ].long()
                states[row, :, output_index] = payload["residuals"][
                    row, 1:65
                ].index_select(1, residual_columns).mean(1)
                output_index += 1
        tail_roles = _receiver_roles(
            payload,
            row,
            shard["rank_letters"][row],
            mappings[qid]["original_to_new"],
            tokenizer,
        )
        for tail_name in ("choice_cue_space", "final_decision"):
            residual_columns = payload["receiver_in_residual"][
                row, tail_roles[tail_name]
            ].long()
            states[row, :, output_index] = payload["residuals"][
                row, 1:65
            ].index_select(1, residual_columns).mean(1)
            output_index += 1
        if output_index != len(names):
            raise RuntimeError("Destination inventory was not filled exactly")
    return local_qids, states


def _family_scores(
    token_scores: Any,
    token_ids: list[int],
    inventory: dict[str, dict[str, list[int]]],
) -> Any:
    import torch

    index = {token_id: column for column, token_id in enumerate(token_ids)}
    family_values = []
    for family in FAMILIES:
        lexeme_values = []
        for ids in inventory[family].values():
            columns = torch.as_tensor(
                [index[token_id] for token_id in ids],
                device=token_scores.device,
                dtype=torch.long,
            )
            lexeme_values.append(token_scores.index_select(-1, columns).mean(-1))
        family_values.append(torch.stack(lexeme_values, dim=-1).mean(-1))
    return torch.stack(family_values, dim=-1)


def analyze(args: argparse.Namespace) -> None:
    import torch

    started = time.perf_counter()
    config = ExperimentConfig.load(args.config)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    discovery = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    heldout_qids = [qid for qid in mappings if qid not in discovery]
    qid_to_index = {qid: index for index, qid in enumerate(heldout_qids)}
    if len(heldout_qids) != 249:
        raise RuntimeError(f"Expected 249 held-out questions, found {len(heldout_qids)}")

    shards = sorted((args.workspace / "shards").glob("cohort_*.pt"))
    if len(shards) != 125 or not np.all(np.load(args.workspace / "completed.npy")):
        raise RuntimeError("The complete 125-shard workspace is required")
    if args.max_shards is not None:
        shards = shards[: args.max_shards]

    names = _destination_names()
    width = int(parts.embedding.weight.shape[-1])
    # condition x held-out question x post-layer x destination x width
    states = torch.empty(
        (2, len(heldout_qids), 64, len(names), width), dtype=torch.bfloat16
    )
    filled = np.zeros((2, len(heldout_qids)), dtype=bool)
    for shard_index, shard_path in enumerate(shards):
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        for condition_index, condition in enumerate(CONDITIONS):
            qids, local_states = _destination_states(
                shard, condition, mappings, tokenizer
            )
            for local_index, qid in enumerate(qids):
                if qid in discovery:
                    continue
                target = qid_to_index[qid]
                states[condition_index, target] = local_states[local_index]
                filled[condition_index, target] = True
        del shard
        if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(shards):
            print(f"Destination extraction: {shard_index + 1}/{len(shards)} shards", flush=True)

    if args.max_shards is None and not np.all(filled):
        missing = np.argwhere(~filled)
        raise RuntimeError(f"Missing held-out destination states: {missing[:20].tolist()}")
    selected_questions = np.where(filled.all(axis=0))[0]
    if not len(selected_questions):
        raise RuntimeError("No complete held-out questions were extracted")
    states = states[:, selected_questions]

    token_ids, inventory = _token_inventory(tokenizer)
    token_index = torch.as_tensor(token_ids, dtype=torch.long)
    family_head = parts.output_head.weight.detach().index_select(
        0, token_index.to(parts.output_head.weight.device)
    )
    lens_device = parts.final_norm.weight.device
    lens_paths, checkpoints = {}, {}
    for name, filename in zip(LENSES, (args.j_filename, args.r_filename)):
        path, checkpoint = _load_lens(args.lens_repo, filename)
        lens_paths[name] = path
        checkpoints[name] = checkpoint

    # condition x question x lens x layer x destination x family
    scores = np.empty(
        (
            2,
            len(selected_questions),
            len(LENSES),
            64,
            len(names),
            len(FAMILIES),
        ),
        dtype=np.float32,
    )
    with torch.inference_mode():
        for lens_index, lens_name in enumerate(LENSES):
            checkpoint = checkpoints[lens_name]
            for layer_slot in range(64):
                transport = (
                    checkpoint["J"][layer_slot].to(
                        device=lens_device, dtype=torch.bfloat16
                    )
                    if layer_slot < 63
                    else None
                )
                for condition_index in range(2):
                    values = states[condition_index, :, layer_slot].reshape(-1, width)
                    batches = []
                    for start in range(0, len(values), 256):
                        batch = values[start : start + 256].to(device=lens_device)
                        transported = batch if transport is None else batch @ transport.T
                        normalized = parts.final_norm(
                            transported.to(parts.final_norm.weight.dtype)
                        )
                        token_scores = normalized @ family_head.T
                        batches.append(
                            _family_scores(
                                token_scores, token_ids, inventory
                            ).float().cpu()
                        )
                    condition_scores = torch.cat(batches).reshape(
                        len(selected_questions), len(names), len(FAMILIES)
                    )
                    scores[condition_index, :, lens_index, layer_slot] = (
                        condition_scores.numpy()
                    )
                print(
                    f"{lens_name} family readout: layer {layer_slot + 1}/64",
                    flush=True,
                )

    means = scores.mean(axis=1)
    sem = scores.std(axis=1, ddof=1) / math.sqrt(scores.shape[1])
    lower = means - 1.96 * sem
    upper = means + 1.96 * sem
    result = {
        "definition": (
            "Raw within-task J/R-lens activation trajectories for two prespecified "
            "word sets in complete post-layer residual states at exact 2P destinations. "
            "The sets contain only capitalization and ordinary morphological variants "
            "of the eight user-specified anchors: incorrect, failed, mistake, wrong; "
            "lost, again, resend, repeat. Each variant is averaged over matching "
            "whitespace-prefixed single-token vocabulary entries; set scores average "
            "matched variants equally."
        ),
        "evidence_label": "Descriptive complete-residual lens readout; not source attribution or a causal intervention.",
        "conditions": list(CONDITIONS),
        "heldout_questions": int(scores.shape[1]),
        "layers": list(range(1, 65)),
        "position_names": list(names),
        "family_lexemes_requested": {key: list(value) for key, value in FAMILIES.items()},
        "family_token_inventory": {
            family: {
                lexeme: [
                    {"token_id": int(token_id), "token": tokenizer.decode([token_id])}
                    for token_id in ids
                ]
                for lexeme, ids in rows.items()
            }
            for family, rows in inventory.items()
        },
        "metric": "Mean raw family vocabulary score after lens transport, final model normalization, and unembedding.",
        "interval": "Mean plus/minus 1.96 standard errors over held-out questions, separately within each task.",
        "lens_paths": lens_paths,
        "mean": means.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
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
    parser.add_argument("--lens-repo", default="camilablank/workspace-lenses")
    parser.add_argument("--j-filename", default="qwen3.6-27b/j-lens/lens.pt")
    parser.add_argument("--r-filename", default="qwen3.6-27b/r-lens/lens.pt")
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
