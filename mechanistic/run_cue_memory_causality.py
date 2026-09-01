from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .downstream_source_intervention import (
    BatchedSDPADownstreamAttentionAblator,
    BatchedSDPADownstreamSourceKVPatcher,
    BatchedSelectiveGDNSourceWriteAblator,
    BatchedSelectiveGDNSourceWritePatcher,
)
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    resolve_answer_tokens,
    tokenize_batch,
)
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward


SCENARIOS = ("natural", "cue_swapped", "cue_ablated", "colon_ablated")


def _hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _hidden(output: Any) -> Any:
    return output[0] if isinstance(output, (tuple, list)) else output


def _locate_second_choice_cue(tokenizer: Any, prompt: str) -> dict[str, Any]:
    ids = [int(value) for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    candidates = [
        index
        for index in range(1, len(tokens) - 1)
        if tokens[index - 1] == "):" and tokens[index + 1] == "<|im_end|>"
    ]
    if len(candidates) != 2:
        raise RuntimeError(
            f"Expected first- and second-presentation cue spaces, found {candidates}"
        )
    cue = candidates[-1]
    colon = cue - 1
    if tokenizer.decode([ids[cue]]) != " " or tokenizer.decode([ids[colon]]) != "):":
        raise RuntimeError("Choice-cue tokenization changed")
    return {
        "ids": ids,
        "cue": cue,
        "colon": colon,
        "cue_token": tokenizer.decode([ids[cue]]),
        "colon_token": tokenizer.decode([ids[colon]]),
        "all_cue_candidates": candidates,
    }


class FinalNormPositionCollector:
    """Capture the final normalized residual at one row-specific prompt token."""

    def __init__(self, parts: Any, positions: list[int]) -> None:
        self.positions = positions
        self.value = None
        self.handle = parts.final_norm.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        import torch

        hidden = _hidden(output)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        positions = torch.as_tensor(self.positions, device=hidden.device)
        self.value = hidden[rows, positions].detach()

    def close(self) -> None:
        self.handle.remove()

    def logits(self, parts: Any, variant_ids: dict[str, list[int]]) -> np.ndarray:
        import torch

        if self.value is None:
            raise RuntimeError("Final norm collector did not observe a forward")
        values = []
        bias = getattr(parts.output_head, "bias", None)
        for letter in LETTERS:
            ids = torch.as_tensor(
                variant_ids[letter], device=parts.output_head.weight.device
            )
            rows = parts.output_head.weight.index_select(0, ids)
            logits = self.value.to(rows.device).float() @ rows.float().T
            if bias is not None:
                logits = logits + bias.index_select(0, ids).float()
            values.append(torch.logsumexp(logits, dim=-1))
        return (
            torch.stack(values, dim=-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )


def _initialize(
    path: Path,
    qids: list[str],
    ordinary_layers: list[int],
    gla_layers: list[int],
) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise RuntimeError("Existing checkpoint has different questions")
        if arrays["scenario_ids"].astype(str).tolist() != list(SCENARIOS):
            raise RuntimeError("Existing checkpoint has different scenarios")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "scenario_ids": np.asarray(SCENARIOS),
        "ordinary_layers_one_based": np.asarray(
            [value + 1 for value in ordinary_layers], dtype=np.int16
        ),
        "gla_layers_one_based": np.asarray(
            [value + 1 for value in gla_layers], dtype=np.int16
        ),
        "completed": np.zeros(n, dtype=bool),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "cue_positions": np.full((2, n), -1, dtype=np.int16),
        "colon_positions": np.full((2, n), -1, dtype=np.int16),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "scenario_final_logits": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
        "scenario_final_logits_raw": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
        "scenario_cue_logits": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
    }


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(args.config)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires canonical batch-size-4 SDPA execution")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires canonical empty-history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires action-matched incorrect/lost feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires canonical raw Qwen ChatML")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    all_qids = [row["id"] for row in manifest["questions"]]
    if args.split_plan is None:
        qids = list(all_qids)
    else:
        selected = set(json.loads(args.split_plan.read_text())["question_ids"])
        qids = [qid for qid in all_qids if qid in selected]
    if args.max_cohorts is not None:
        qids = qids[: int(args.max_cohorts) * config.batch_size]
    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    trusted = [
        json.loads(args.trusted_game.read_text())["results"],
        json.loads(args.trusted_neutral.read_text())["results"],
    ]

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    variants = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in variants[letter]})
        for letter in LETTERS
    }
    ordinary_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    ]
    gla_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if [value + 1 for value in ordinary_layers] != list(range(4, 65, 4)):
        raise RuntimeError("Unexpected ordinary-attention layer inventory")
    if len(gla_layers) != 48:
        raise RuntimeError("Unexpected GLA layer inventory")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.npz"
    arrays = _initialize(result_path, qids, ordinary_layers, gla_layers)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    durations: list[float] = []
    audit_path = args.output_dir / "prompt_audit.json"
    started = time.monotonic()

    for cohort_start in range(0, len(qids), config.batch_size):
        cohort = qids[cohort_start : cohort_start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if arrays["completed"][indices].all():
            continue
        if len(cohort) != config.batch_size:
            raise RuntimeError("Canonical question count must form complete cohorts")
        cohort_started = time.monotonic()
        condition_batches = [
            _build_batch(
                config, processor, tokenizer, questions, mappings, cohort, condition
            )
            for condition in CONDITIONS
        ]
        canonical_width = int(condition_batches[0]["input_ids"].shape[1])
        audit: dict[str, Any] = {"question_ids": cohort, "pairs": []}

        for pair_start in range(0, len(cohort), 2):
            pair = cohort[pair_start : pair_start + 2]
            prompts = (
                condition_batches[0]["prompts"][pair_start : pair_start + 2]
                + condition_batches[1]["prompts"][pair_start : pair_start + 2]
            )
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
            if input_ids.shape[1] > canonical_width:
                raise RuntimeError("Paired batch exceeds canonical cohort width")
            if input_ids.shape[1] < canonical_width:
                pad = canonical_width - int(input_ids.shape[1])
                input_ids = torch.nn.functional.pad(
                    input_ids, (pad, 0), value=int(tokenizer.pad_token_id)
                )
                attention_mask = torch.nn.functional.pad(
                    attention_mask, (pad, 0), value=0
                )
            width = int(input_ids.shape[1])
            cue_positions: list[int] = []
            colon_positions: list[int] = []
            pair_audit: dict[str, Any] = {"question_ids": pair, "rows": []}
            for condition_index, condition in enumerate(CONDITIONS):
                for local, qid in enumerate(pair):
                    row = condition_index * 2 + local
                    located = _locate_second_choice_cue(tokenizer, prompts[row])
                    left_pad = width - len(located["ids"])
                    if input_ids[row, left_pad:].tolist() != located["ids"]:
                        raise RuntimeError("Pair tokenization changed the prompt")
                    cue = left_pad + int(located["cue"])
                    colon = left_pad + int(located["colon"])
                    cue_positions.append(cue)
                    colon_positions.append(colon)
                    qi = qid_index[qid]
                    prompt_digest = _hash(prompts[row])
                    trusted_digest = trusted[condition_index][qid]["prompt_hash"]
                    if prompt_digest != trusted_digest:
                        raise RuntimeError("Prompt hash differs from trusted natural run")
                    arrays["prompt_hashes"][condition_index, qi] = prompt_digest
                    arrays["cue_positions"][condition_index, qi] = cue
                    arrays["colon_positions"][condition_index, qi] = colon
                    arrays["trusted_natural_logits"][condition_index, qi] = np.asarray(
                        trusted[condition_index][qid]["aggregated_ad_logits"],
                        dtype=np.float32,
                    )
                    pair_audit["rows"].append({
                        "row": row,
                        "condition": condition,
                        "question_id": qid,
                        "prompt_hash": prompt_digest,
                        "left_padding": left_pad,
                        "cue_position_padded": cue,
                        "colon_position_padded": colon,
                        "cue_token": located["cue_token"],
                        "colon_token": located["colon_token"],
                    })
            if cue_positions[0] != cue_positions[2] or cue_positions[1] != cue_positions[3]:
                raise RuntimeError("Game/Neutral cue positions are not physically aligned")
            if colon_positions[0] != colon_positions[2] or colon_positions[1] != colon_positions[3]:
                raise RuntimeError("Game/Neutral colon positions are not physically aligned")

            swap_specs = {
                row: (
                    row + 2 if row < 2 else row - 2,
                    [cue_positions[row]],
                    ordinary_layers,
                )
                for row in range(4)
            }
            gla_swap_specs = {
                row: (source, positions, gla_layers)
                for row, (source, positions, _layers) in swap_specs.items()
            }
            cue_gla_ablation = {
                row: ([cue_positions[row]], gla_layers) for row in range(4)
            }
            colon_gla_ablation = {
                row: ([colon_positions[row]], gla_layers) for row in range(4)
            }

            raw_final: list[np.ndarray] = []
            raw_cue: list[np.ndarray] = []
            for scenario in SCENARIOS:
                ordinary = None
                gla = None
                collector = None
                try:
                    if scenario == "cue_swapped":
                        ordinary = BatchedSDPADownstreamSourceKVPatcher(
                            parts, swap_specs
                        )
                        gla = BatchedSelectiveGDNSourceWritePatcher(
                            parts, gla_swap_specs, preserve_source_output=True
                        )
                    elif scenario == "cue_ablated":
                        ordinary = BatchedSDPADownstreamAttentionAblator(
                            parts, {row: [cue_positions[row]] for row in range(4)}
                        )
                        gla = BatchedSelectiveGDNSourceWriteAblator(
                            parts, cue_gla_ablation, preserve_source_output=True
                        )
                    elif scenario == "colon_ablated":
                        ordinary = BatchedSDPADownstreamAttentionAblator(
                            parts, {row: [colon_positions[row]] for row in range(4)}
                        )
                        gla = BatchedSelectiveGDNSourceWriteAblator(
                            parts, colon_gla_ablation, preserve_source_output=True
                        )
                    collector = FinalNormPositionCollector(parts, cue_positions)
                    output = _forward(model, parts, input_ids, attention_mask)
                    if ordinary is not None:
                        ordinary.assert_fired()
                    if gla is not None:
                        gla.assert_fired()
                    raw_final.append(_aggregate_logits(output, variant_ids))
                    raw_cue.append(collector.logits(parts, variant_ids))
                finally:
                    if collector is not None:
                        collector.close()
                    if gla is not None:
                        gla.close()
                    if ordinary is not None:
                        ordinary.close()

            final = np.stack(raw_final, axis=0)
            cue = np.stack(raw_cue, axis=0)
            natural_final = final[0]
            natural_cue = cue[0]
            cue_invariance = float(
                np.max(np.abs(cue[1:3] - natural_cue[None, :, :]))
            )
            if cue_invariance > args.cue_invariance_tolerance:
                raise RuntimeError(
                    f"Cue swap/ablation changed the cue itself by {cue_invariance}"
                )

            for condition_index in range(2):
                for local, qid in enumerate(pair):
                    row = condition_index * 2 + local
                    qi = qid_index[qid]
                    trusted_logits = arrays["trusted_natural_logits"][condition_index, qi]
                    arrays["same_batch_natural_logits"][condition_index, qi] = natural_final[row]
                    arrays["scenario_final_logits_raw"][condition_index, :, qi] = final[:, row]
                    arrays["scenario_final_logits"][condition_index, :, qi] = (
                        trusted_logits[None, :]
                        + final[:, row]
                        - natural_final[row][None, :]
                    )
                    arrays["scenario_cue_logits"][condition_index, :, qi] = cue[:, row]
            pair_audit["cue_invariance_max_abs_error"] = cue_invariance
            audit["pairs"].append(pair_audit)

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"cue-memory causality: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.2f}",
            flush=True,
        )
        if not audit_path.exists():
            audit["ordinary_layers_one_based"] = [value + 1 for value in ordinary_layers]
            audit["gla_layers_one_based"] = [value + 1 for value in gla_layers]
            audit["interventions"] = {
                "cue_swapped": (
                    "Reciprocally transplant the cue token's downstream ordinary-attention "
                    "K/V and GLA recurrent-memory write between aligned Game/Neutral rows."
                ),
                "cue_ablated": (
                    "Block the cue K/V from every later ordinary-attention query and set "
                    "its GLA write strength to zero while preserving the cue's own output."
                ),
                "colon_ablated": (
                    "Apply the identical downstream-memory ablation to the immediately "
                    "preceding tokenizer token, which is the combined '):' token."
                ),
            }
            audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    metadata = {
        "experiment": "causal role of the post-list answer-cue memory",
        "questions": len(qids),
        "conditions": list(CONDITIONS),
        "scenarios": list(SCENARIOS),
        "complete_model_forwards_per_canonical_cohort": 8,
        "paired_subbatches_per_canonical_cohort": 2,
        "ordinary_layers_one_based": [value + 1 for value in ordinary_layers],
        "gla_layers_one_based": [value + 1 for value in gla_layers],
        "cue_intervention_scope": (
            "All cross-token communication from the exact trailing cue-space token: "
            "ordinary-attention K/V at all 16 ordinary layers and recurrent-memory "
            "writes at all 48 GLA layers. The cue token's own output remains natural."
        ),
        "same_batch_correction": (
            "trusted natural logits + intervention same-batch logits - natural same-batch logits"
        ),
        "elapsed_seconds_after_load": time.monotonic() - started,
        "cohort_seconds": durations,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    parser.add_argument("--cue-invariance-tolerance", type=float, default=1e-5)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
