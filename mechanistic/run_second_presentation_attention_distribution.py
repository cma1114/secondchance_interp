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
from .io import atomic_save_npz
from .modeling import get_tokenizer, load_model_and_processor, resolve_answer_tokens
from .prompts import ANSWER_ONLY_INSTRUCTION, CAPABILITY_SETUP, FACTORIAL_FEEDBACK, present_question
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .run_fixed_a_final_query_edge_ablation import _token_positions_for_interval


ORDINARY_LAYERS = tuple(range(4, 65, 4))
RANKS = ("R1", "R2", "R3", "R4")
SOURCE_BINS = (
    "system_and_header",
    "first_task_instruction",
    "first_question_stem",
    "first_R1_line",
    "first_R2_line",
    "first_R3_line",
    "first_R4_line",
    "first_answer_boundary",
    "feedback_sentence",
    "second_answer_instruction",
    "second_question_stem",
    "second_R1_line",
    "second_R2_line",
    "second_R3_line",
    "second_R4_line",
    "chat_separators_other",
)


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _overlaps(left: int, right: int, start: int, end: int) -> bool:
    return right > left and left < end and right > start


def _option_line_positions_in_range(
    tokenizer: Any,
    prompt: str,
    question: dict[str, Any],
    question_start: int,
    question_end: int,
    ids: list[int],
    offsets: list[tuple[int, int]],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Locate option lines inside one explicit question occurrence."""
    positions: dict[str, list[int]] = {}
    audit: dict[str, Any] = {}
    for letter in LETTERS:
        line = f"  {letter}: {question['options'][letter]}\n"
        start = prompt.find(line, question_start, question_end)
        if start < 0:
            raise RuntimeError(
                f"Could not locate option line {letter} in question range "
                f"[{question_start}, {question_end})"
            )
        row = _token_positions_for_interval(offsets, start, start + len(line))
        if not row:
            raise RuntimeError(f"Option line {letter} has no tokens")
        positions[letter] = row
        audit[letter] = {
            "text": line.rstrip("\n"),
            "positions": row,
            "tokens": tokenizer.convert_ids_to_tokens([ids[index] for index in row]),
        }
    return positions, audit


def _source_bins_and_queries(
    tokenizer: Any,
    prompt: str,
    first_question: dict[str, Any],
    second_question: dict[str, Any],
    condition: str,
    rank_letters: list[str],
    original_to_new: dict[str, str],
) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    """Partition every unpadded prompt token and locate the four 2P query lines."""
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]

    first_text = present_question(first_question)
    second_text = present_question(second_question)
    first_start = prompt.find(first_text)
    second_start = prompt.find(second_text, first_start + len(first_text))
    feedback_text = FACTORIAL_FEEDBACK[condition]
    feedback_start = prompt.find(feedback_text, first_start + len(first_text))
    second_instruction_start = prompt.find(
        ANSWER_ONLY_INSTRUCTION, feedback_start + len(feedback_text)
    )
    capability_start = prompt.find(CAPABILITY_SETUP)
    if min(
        capability_start,
        first_start,
        feedback_start,
        second_instruction_start,
        second_start,
    ) < 0:
        raise RuntimeError("Could not locate every canonical prompt component")
    if not (
        capability_start < first_start < feedback_start
        < second_instruction_start < second_start
    ):
        raise RuntimeError("Canonical prompt components are out of order")

    first_positions, first_audit = _option_line_positions_in_range(
        tokenizer,
        prompt,
        first_question,
        first_start,
        first_start + len(first_text),
        ids,
        offsets,
    )
    second_positions, second_audit = _option_line_positions_in_range(
        tokenizer,
        prompt,
        second_question,
        second_start,
        second_start + len(second_text),
        ids,
        offsets,
    )
    second_letter_for_first = {str(old): str(new) for old, new in original_to_new.items()}
    if set(second_letter_for_first) != set(LETTERS) or set(second_letter_for_first.values()) != set(LETTERS):
        raise RuntimeError("Remapping is not a complete A-D permutation")

    labels = np.full(len(ids), len(SOURCE_BINS) - 1, dtype=np.int16)
    broad_ranges = {
        0: (0, capability_start),
        1: (capability_start, first_start),
        2: (first_start, first_start + len(first_text)),
        7: (first_start + len(first_text), feedback_start),
        8: (feedback_start, feedback_start + len(feedback_text)),
        9: (feedback_start + len(feedback_text), second_start),
        10: (second_start, second_start + len(second_text)),
    }
    for token_index, (left, right) in enumerate(offsets):
        for bin_index, (start, end) in broad_ranges.items():
            if _overlaps(left, right, start, end):
                labels[token_index] = bin_index
                break

    queries: list[list[int]] = []
    for rank, first_letter in enumerate(rank_letters):
        second_letter = second_letter_for_first[first_letter]
        for token_index in first_positions[first_letter]:
            labels[token_index] = 3 + rank
        for token_index in second_positions[second_letter]:
            labels[token_index] = 11 + rank
        queries.append([int(value) for value in second_positions[second_letter]])

    source_bins = [
        np.flatnonzero(labels == bin_index).astype(int).tolist()
        for bin_index in range(len(SOURCE_BINS))
    ]
    assigned = sorted(value for positions in source_bins for value in positions)
    if assigned != list(range(len(ids))):
        raise RuntimeError("Source bins do not form an exhaustive token partition")
    if any(not positions for positions in source_bins[:-1]):
        empty = [SOURCE_BINS[i] for i, values in enumerate(source_bins[:-1]) if not values]
        raise RuntimeError(f"Unexpected empty canonical source bins: {empty}")

    audit = {
        "rank_letters": rank_letters,
        "first_option_lines": first_audit,
        "second_option_lines": second_audit,
        "second_letter_for_first": second_letter_for_first,
        "source_bins": {
            name: {
                "positions": source_bins[index],
                "tokens": tokenizer.convert_ids_to_tokens(
                    [ids[value] for value in source_bins[index]]
                ),
            }
            for index, name in enumerate(SOURCE_BINS)
        },
        "query_positions_by_rank": {
            rank: queries[index] for index, rank in enumerate(RANKS)
        },
    }
    return source_bins, queries, audit


class AttentionDistributionCollector:
    """Recover exact SDPA weights and aggregate them over exhaustive source bins."""

    def __init__(
        self,
        parts: Any,
        layer_indices: tuple[int, ...],
        source_bins: list[list[list[int]]],
        query_positions: list[list[list[int]]],
    ) -> None:
        import torch

        self.layer_indices = layer_indices
        self.source_bins = source_bins
        self.query_positions = query_positions
        self.layers = {index: parts.layers[index].self_attn for index in layer_indices}
        self.active: int | None = None
        self.metrics: dict[int, tuple[Any, Any]] = {}
        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        self.handles: list[Any] = []
        for layer_index, attention in self.layers.items():
            self.handles.extend(
                [
                    attention.register_forward_pre_hook(self._enter(layer_index)),
                    attention.register_forward_hook(self._leave(layer_index)),
                ]
            )
        torch.nn.functional.scaled_dot_product_attention = self._wrapped_sdpa

    def _enter(self, layer_index: int):
        def enter(_module: Any, _inputs: Any) -> None:
            if self.active is not None:
                raise RuntimeError("Nested ordinary-attention calls are unsupported")
            self.active = layer_index

        return enter

    def _leave(self, layer_index: int):
        def leave(_module: Any, _inputs: Any, _output: Any) -> None:
            if self.active != layer_index:
                raise RuntimeError("Ordinary-attention layer stack became inconsistent")
            self.active = None

        return leave

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        import torch

        result = self.original_sdpa(query, key, value, *args, **kwargs)
        layer_index = self.active
        if layer_index not in self.layers:
            return result

        key_length = int(key.shape[-2])
        head_dim = int(value.shape[-1])
        weights = torch.empty(
            query.shape[0], query.shape[1], query.shape[2], key_length,
            device=query.device, dtype=value.dtype,
        )
        for start in range(0, key_length, head_dim):
            end = min(start + head_dim, key_length)
            synthetic = torch.zeros_like(value)
            positions = torch.arange(start, end, device=value.device)
            dimensions = torch.arange(end - start, device=value.device)
            synthetic[:, :, positions, dimensions] = 1
            recovered = self.original_sdpa(query, key, synthetic, *args, **kwargs)
            weights[..., start:end] = recovered[..., : end - start]

        batch = int(query.shape[0])
        mass = torch.zeros(
            batch, 4, len(SOURCE_BINS), device=query.device, dtype=torch.float32
        )
        max_sum_error = torch.zeros(batch, 4, device=query.device, dtype=torch.float32)
        for row in range(batch):
            for rank in range(4):
                query_index = torch.as_tensor(
                    self.query_positions[row][rank], device=weights.device
                )
                local = weights[row].index_select(1, query_index).float()
                per_bin = []
                for positions in self.source_bins[row]:
                    source_index = torch.as_tensor(positions, device=weights.device)
                    per_bin.append(local.index_select(2, source_index).sum(-1))
                stacked = torch.stack(per_bin, dim=-1)
                mass[row, rank] = stacked.mean(dim=(0, 1))
                max_sum_error[row, rank] = (stacked.sum(-1) - 1.0).abs().max()
        self.metrics[layer_index] = (
            mass.detach().cpu(), max_sum_error.detach().cpu()
        )
        return result

    def close(self) -> None:
        import torch

        torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.active = None

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        import torch

        missing = [index for index in self.layer_indices if index not in self.metrics]
        if missing:
            raise RuntimeError(f"Missing attention distributions at layers {missing}")
        mass = torch.stack(
            [self.metrics[index][0] for index in self.layer_indices], dim=0
        ).numpy()
        errors = torch.stack(
            [self.metrics[index][1] for index in self.layer_indices], dim=0
        ).numpy()
        return mass, errors


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        arrays = dict(np.load(path, allow_pickle=False))
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses a different question order")
        if arrays["source_bins"].astype(str).tolist() != list(SOURCE_BINS):
            raise ValueError("Existing checkpoint uses different source bins")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "ordinary_layers_one_based": np.asarray(ORDINARY_LAYERS, dtype=np.int16),
        "ranks": np.asarray(RANKS),
        "source_bins": np.asarray(SOURCE_BINS),
        "completed": np.zeros(n, dtype=bool),
        "rank_letters": np.full((n, 4), "", dtype="<U1"),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "source_position_counts": np.zeros((2, n, len(SOURCE_BINS)), dtype=np.int16),
        "query_position_counts": np.zeros((2, n, 4), dtype=np.int16),
        "attention_mass": np.full(
            (2, len(ORDINARY_LAYERS), n, 4, len(SOURCE_BINS)),
            np.nan,
            dtype=np.float32,
        ),
        "max_partition_error": np.full(
            (2, len(ORDINARY_LAYERS), n, 4), np.nan, dtype=np.float32
        ),
    }


def run(
    config_path: Path,
    remapping_plan_path: Path,
    baseline_path: Path,
    trusted_game_path: Path,
    trusted_neutral_path: Path,
    output_dir: Path,
    max_cohorts: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires exact historical batch-size-4 SDPA execution")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires action-matched incorrect/lost feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw_qwen_chatml serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    if max_cohorts is not None:
        qids = qids[: int(max_cohorts) * config.batch_size]
    mappings = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    identity_mapping = {letter: letter for letter in LETTERS}
    identity_second_presentation = all(
        row["original_to_new"] == identity_mapping
        and row["new_to_original"] == identity_mapping
        for row in mappings.values()
    )
    baseline = json.loads(baseline_path.read_text())["results"]
    trusted = [
        json.loads(trusted_game_path.read_text())["results"],
        json.loads(trusted_neutral_path.read_text())["results"],
    ]

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    layer_indices = tuple(
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    )
    if tuple(index + 1 for index in layer_indices) != ORDINARY_LAYERS:
        raise RuntimeError(f"Unexpected ordinary-attention layers: {layer_indices}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = output_dir / "prompt_audit.json"
    started = time.monotonic()
    durations: list[float] = []

    for start in range(0, len(qids), config.batch_size):
        cohort = qids[start : start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if np.all(arrays["completed"][indices]):
            continue
        cohort_started = time.monotonic()
        batches = [
            _build_batch(config, processor, tokenizer, questions, mappings, cohort, condition)
            for condition in CONDITIONS
        ]
        cohort_audit: dict[str, Any] = {"question_ids": cohort, "conditions": {}}

        for qid in cohort:
            qi = qid_index[qid]
            logits = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=float)
            order = np.argsort(-logits, kind="stable")
            arrays["rank_letters"][qi] = np.asarray([LETTERS[int(i)] for i in order])

        for ci, (condition, batch) in enumerate(zip(CONDITIONS, batches)):
            width = int(batch["input_ids"].shape[1])
            all_source_bins: list[list[list[int]]] = []
            all_queries: list[list[list[int]]] = []
            row_audits: list[dict[str, Any]] = []
            for row, qid in enumerate(cohort):
                qi = qid_index[qid]
                ids = batch["token_rows"][row]
                left_pad = width - len(ids)
                second_question = {
                    **questions[qid],
                    "options": {
                        new: questions[qid]["options"][old]
                        for new, old in mappings[qid]["new_to_original"].items()
                    },
                }
                bins, queries, audit = _source_bins_and_queries(
                    tokenizer,
                    batch["prompts"][row],
                    questions[qid],
                    second_question,
                    condition,
                    arrays["rank_letters"][qi].astype(str).tolist(),
                    mappings[qid]["original_to_new"],
                )
                physical_bins = [[left_pad + value for value in values] for values in bins]
                physical_queries = [
                    [left_pad + value for value in values] for values in queries
                ]
                all_source_bins.append(physical_bins)
                all_queries.append(physical_queries)
                arrays["source_position_counts"][ci, qi] = np.asarray(
                    [len(values) for values in bins], dtype=np.int16
                )
                arrays["query_position_counts"][ci, qi] = np.asarray(
                    [len(values) for values in queries], dtype=np.int16
                )
                arrays["prompt_hashes"][ci, qi] = _hash_prompt(batch["prompts"][row])
                row_audits.append(audit)

            collector = AttentionDistributionCollector(
                parts, layer_indices, all_source_bins, all_queries
            )
            try:
                natural_output = _forward(
                    model, parts, batch["input_ids"], batch["attention_mask"]
                )
                mass, errors = collector.arrays()
            finally:
                collector.close()
            natural = _aggregate_logits(natural_output, variant_ids)
            if not np.all(np.isfinite(natural)) or not np.all(np.isfinite(mass)):
                raise RuntimeError("Non-finite natural outputs")
            if float(np.max(errors)) > 0.02:
                raise RuntimeError(
                    f"Attention source partition failed: max error {float(np.max(errors)):.6f}"
                )
            for row, qid in enumerate(cohort):
                qi = qid_index[qid]
                arrays["natural_logits"][ci, qi] = natural[row]
                arrays["attention_mass"][ci, :, qi] = mass[:, row]
                arrays["max_partition_error"][ci, :, qi] = errors[:, row]
                arrays["trusted_natural_logits"][ci, qi] = np.asarray(
                    trusted[ci][qid]["aggregated_ad_logits"], dtype=np.float32
                )
            cohort_audit["conditions"][condition] = {
                "rendered_prompt": batch["prompts"][0],
                "prompt_hash": arrays["prompt_hashes"][ci, indices[0]].item(),
                "partition": row_audits[0],
            }

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"2P attention distribution: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.3f}; "
            f"max_partition_error={float(np.nanmax(arrays['max_partition_error'][:, :, indices])):.6f}",
            flush=True,
        )
        if not audit_path.exists():
            audit_path.write_text(
                json.dumps(cohort_audit, indent=2, ensure_ascii=False) + "\n"
            )

    metadata = {
        "experiment": (
            "canonical non-remapped exhaustive natural 2P-line attention distribution"
            if identity_second_presentation
            else "canonical remapped exhaustive natural 2P-line attention distribution"
        ),
        "second_presentation_mapping": (
            "identity" if identity_second_presentation else "question-specific permutation"
        ),
        "config": config.as_dict(),
        "n_questions": len(qids),
        "conditions": list(CONDITIONS),
        "ranks": list(RANKS),
        "ordinary_layers_one_based": list(ORDINARY_LAYERS),
        "source_bins": list(SOURCE_BINS),
        "complete_model_forwards_per_cohort": 2,
        "complete_model_work": (
            "One natural Game and one natural Neutral forward per cohort; exact SDPA "
            "weights are recovered at all 16 ordinary-attention layers and partitioned "
            "over every non-padding source token."
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
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.remapping_plan,
        args.baseline,
        args.trusted_game,
        args.trusted_neutral,
        args.output_dir,
        args.max_cohorts,
    )


if __name__ == "__main__":
    main()
