from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .collect_remapped_behavior import _messages, _remap_question
from .config import ExperimentConfig
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import CHOICE_CUE, present_question, prompt_hash
from .sublayer import _hidden, _replace_hidden


LETTERS = "ABCD"
LAYER = 55  # zero based; user-facing Mixer 56
SELECTED_HEADS = (0, 2, 6, 15)
SCENARIOS = (
    "natural",
    "different_query",
    "different_gate",
    "different_query_gate",
    "different_kv",
    "different_query_gate_kv",
    "different_all_heads",
    "same_winner_query_gate_kv",
)


def _chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {row["question_id"]: row for row in json.loads(path.read_text())["rows"]}


def _mapping_plans(paths: list[Path]) -> list[dict[str, dict[str, Any]]]:
    return [_load_rows(path) for path in paths]


def _same_winner_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        value for value in row["all_candidate_winners"]
        if value["winner_content"] == row["recipient_winner_content"]
    ]
    if not candidates:
        return None
    # Prefer a control that changes the literal letter, then the stronger baseline.
    return max(candidates, key=lambda value: (value["letter_decoupled"], value["margin"]))


def _token_interval_positions(
    tokenizer: Any, prompt: str, second_question: dict[str, Any]
) -> tuple[list[int], dict[str, Any]]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    question_text = present_question(second_question)
    question_start = prompt.rfind(question_text)
    if question_start < 0:
        raise RuntimeError("Could not locate the repeated question")
    option_start = prompt.find("  A: ", question_start)
    cue_start = prompt.find(CHOICE_CUE, question_start + len(question_text))
    if option_start < 0 or cue_start < 0:
        raise RuntimeError("Could not locate repeated options or choice cue")
    interval = (option_start, cue_start + len(CHOICE_CUE))
    positions = [
        index for index, (left, right) in enumerate(offsets)
        if right > left and left < interval[1] and right > interval[0]
    ]
    if not positions:
        raise RuntimeError("Repeated option/cue interval has no tokens")
    return positions, {
        "ids": ids,
        "interval": interval,
        "first_position": positions[0],
        "last_position": positions[-1],
        "first_token": tokenizer.decode([ids[positions[0]]]),
        "last_token": tokenizer.decode([ids[positions[-1]]]),
    }


def _forward(model: Any, parts: Any, input_ids: Any, attention_mask: Any):
    import torch

    device = model_input_device(parts)
    with torch.inference_mode():
        kwargs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "use_cache": False,
            "return_dict": True,
        }
        try:
            return model(**kwargs, logits_to_keep=1)
        except TypeError:
            return model(**kwargs)


def _aggregate_logits(output: Any, last: list[int], variant_ids: list[list[int]]) -> np.ndarray:
    import torch

    logits = output.logits.detach().float()
    if logits.shape[1] == 1:
        final = logits[:, 0]
    else:
        rows = torch.arange(logits.shape[0], device=logits.device)
        cols = torch.as_tensor(last, device=logits.device)
        final = logits[rows, cols]
    return torch.stack(
        [torch.logsumexp(final[:, ids], dim=-1) for ids in variant_ids], dim=-1
    ).cpu().numpy()


class Mixer56StateCollector:
    def __init__(self, parts: Any, last: list[int]) -> None:
        import torch

        self.attention = parts.layers[LAYER].self_attn
        self.last = [int(value) for value in last]
        self.q = None
        self.k = None
        self.v = None
        self.write = None
        self.handles = [
            self.attention.q_proj.register_forward_hook(self._projection("q")),
            self.attention.k_proj.register_forward_hook(self._projection("k")),
            self.attention.v_proj.register_forward_hook(self._projection("v")),
            self.attention.register_forward_hook(self._write),
        ]
        self._torch = torch

    def _projection(self, name: str):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            setattr(self, name, output.detach().clone())
        return capture

    def _write(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = _hidden(output)
        rows = self._torch.arange(hidden.shape[0], device=hidden.device)
        cols = self._torch.as_tensor(self.last, device=hidden.device)
        self.write = hidden[rows, cols].detach().float().cpu().numpy()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def validate(self) -> None:
        if any(value is None for value in (self.q, self.k, self.v, self.write)):
            raise RuntimeError("Mixer 56 state collector missed a projection or write")


class Mixer56StatePatcher:
    def __init__(
        self,
        parts: Any,
        last: list[int],
        source_positions: list[list[int]],
        donor: Mixer56StateCollector,
        patch_query: bool,
        patch_gate: bool,
        patch_kv: bool,
        all_heads: bool = False,
    ) -> None:
        import torch

        self.torch = torch
        self.attention = parts.layers[LAYER].self_attn
        self.last = [int(value) for value in last]
        self.positions = source_positions
        self.donor = donor
        self.patch_query = patch_query
        self.patch_gate = patch_gate
        self.patch_kv = patch_kv
        self.head_dim = int(self.attention.head_dim)
        self.n_heads = int(self.attention.q_proj.out_features // (2 * self.head_dim))
        self.n_kv_heads = int(self.attention.k_proj.out_features // self.head_dim)
        if self.n_heads % self.n_kv_heads:
            raise RuntimeError("Query heads are not divisible by KV heads")
        self.groups = self.n_heads // self.n_kv_heads
        self.query_heads = tuple(range(self.n_heads)) if all_heads else SELECTED_HEADS
        self.kv_heads = tuple(range(self.n_kv_heads)) if all_heads else tuple(
            sorted({head // self.groups for head in self.query_heads})
        )
        self.write = None
        self.handles = [
            self.attention.q_proj.register_forward_hook(self._patch_q),
            self.attention.k_proj.register_forward_hook(self._patch_k),
            self.attention.v_proj.register_forward_hook(self._patch_v),
            self.attention.register_forward_hook(self._capture_write),
        ]

    def _patch_q(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if not (self.patch_query or self.patch_gate):
            return output
        updated = output.clone().view(*output.shape[:-1], self.n_heads, 2, self.head_dim)
        source = self.donor.q.to(device=output.device, dtype=output.dtype).view_as(updated)
        for row, position in enumerate(self.last):
            if self.patch_query:
                updated[row, position, self.query_heads, 0] = source[
                    row, position, self.query_heads, 0
                ]
            if self.patch_gate:
                updated[row, position, self.query_heads, 1] = source[
                    row, position, self.query_heads, 1
                ]
        return updated.reshape_as(output)

    def _patch_projection(self, output: Any, source_tensor: Any) -> Any:
        if not self.patch_kv:
            return output
        updated = output.clone().view(*output.shape[:-1], self.n_kv_heads, self.head_dim)
        source = source_tensor.to(device=output.device, dtype=output.dtype).view_as(updated)
        for row, positions in enumerate(self.positions):
            for head in self.kv_heads:
                updated[row, positions, head] = source[row, positions, head]
        return updated.reshape_as(output)

    def _patch_k(self, _module: Any, _inputs: Any, output: Any) -> Any:
        return self._patch_projection(output, self.donor.k)

    def _patch_v(self, _module: Any, _inputs: Any, output: Any) -> Any:
        return self._patch_projection(output, self.donor.v)

    def _capture_write(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = _hidden(output)
        rows = self.torch.arange(hidden.shape[0], device=hidden.device)
        cols = self.torch.as_tensor(self.last, device=hidden.device)
        self.write = hidden[rows, cols].detach().float().cpu().numpy()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _project_write(write: np.ndarray, output_rows: np.ndarray) -> np.ndarray:
    values = write @ output_rows.T
    return values - values.mean(axis=-1, keepdims=True)


def run(
    config_path: Path,
    second_mapping_path: Path,
    mapping_paths: list[Path],
    discovery_donor_path: Path,
    confirmation_donor_path: Path,
    output: Path,
    max_cohorts: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires the exact batch-of-four SDPA regime")
    if (
        config.prompt_mode != "baseline_matched_empty_history"
        or config.feedback_variant != "token_matched_test"
        or config.chat_serialization != "raw_qwen_chatml"
    ):
        raise ValueError("Requires the canonical token-matched empty-history prompt")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    second = _load_rows(second_mapping_path)
    mapping_plans = _mapping_plans(mapping_paths)
    donor_rows = _load_rows(discovery_donor_path)
    donor_rows.update(_load_rows(confirmation_donor_path))
    if not set(qids) <= set(second) or not set(qids) <= set(donor_rows):
        raise ValueError("Mapping or donor plans do not cover all questions")

    output.mkdir(parents=True, exist_ok=True)
    cohort_dir = output / "cohorts"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = [[token_id for _, token_id in resolved[letter]] for letter in LETTERS]
    canonical_ids = [resolved[letter][0][1] for letter in LETTERS]
    output_rows = parts.output_head.weight.detach()[canonical_ids].float().cpu().numpy()

    attention = parts.layers[LAYER].self_attn
    head_dim = int(attention.head_dim)
    n_heads = int(attention.q_proj.out_features // (2 * head_dim))
    n_kv_heads = int(attention.k_proj.out_features // head_dim)
    if max(SELECTED_HEADS) >= n_heads:
        raise RuntimeError(f"Mixer 56 has only {n_heads} query heads")

    start_time = time.perf_counter()
    completed = 0
    audit = None
    for cohort_index, batch_qids in enumerate(_chunks(qids, 4)):
        path = cohort_dir / f"cohort_{cohort_index:03d}.npz"
        if path.exists():
            continue

        recipient_prompts = []
        different_prompts = []
        same_prompts = []
        semantic_maps = []
        donor_semantics = []
        recipient_semantics = []
        same_available = []
        source_unpadded: list[list[int]] = []
        row_audits = []
        for qid in batch_qids:
            row = donor_rows[qid]
            second_row = second[qid]
            second_question = _remap_question(questions[qid], second_row["new_to_original"])
            recipient = render_chat(
                processor,
                _messages(config, questions[qid], second_question, "incorrect"),
                config.disable_thinking,
                config.chat_serialization,
            )
            donor_index = int(row["donor"]["mapping_index"]) - 1
            donor_question = _remap_question(
                questions[qid], mapping_plans[donor_index][qid]["new_to_original"]
            )
            different = render_chat(
                processor,
                _messages(config, donor_question, second_question, "incorrect"),
                config.disable_thinking,
                config.chat_serialization,
            )
            same_candidate = _same_winner_candidate(row)
            if same_candidate is None:
                same = recipient
                same_available.append(False)
            else:
                same_index = int(same_candidate["mapping_index"]) - 1
                same_question = _remap_question(
                    questions[qid], mapping_plans[same_index][qid]["new_to_original"]
                )
                same = render_chat(
                    processor,
                    _messages(config, same_question, second_question, "incorrect"),
                    config.disable_thinking,
                    config.chat_serialization,
                )
                same_available.append(True)

            recipient_positions, rec_audit = _token_interval_positions(
                tokenizer, recipient, second_question
            )
            donor_positions, donor_audit = _token_interval_positions(
                tokenizer, different, second_question
            )
            same_positions, same_audit = _token_interval_positions(tokenizer, same, second_question)
            # The full prompts intentionally differ in the first option order,
            # but their lengths and the complete patched suffix must coincide.
            if not (
                len(rec_audit["ids"]) == len(donor_audit["ids"])
                == len(same_audit["ids"])
                and recipient_positions == donor_positions == same_positions
                and [rec_audit["ids"][p] for p in recipient_positions]
                == [donor_audit["ids"][p] for p in donor_positions]
                == [same_audit["ids"][p] for p in same_positions]
            ):
                raise RuntimeError(f"{qid}: donor/recipient repeated spans are not aligned")
            recipient_prompts.append(recipient)
            different_prompts.append(different)
            same_prompts.append(same)
            source_unpadded.append(recipient_positions)
            semantic_maps.append([LETTERS.index(second_row["original_to_new"][x]) for x in LETTERS])
            recipient_semantics.append(LETTERS.index(row["recipient_winner_content"]))
            donor_semantics.append(LETTERS.index(row["donor"]["winner_content"]))
            row_audits.append({
                "question_id": qid,
                "recipient_prompt_hash": prompt_hash(recipient),
                "different_donor_prompt_hash": prompt_hash(different),
                "same_donor_prompt_hash": prompt_hash(same),
                "different_donor": row["donor"],
                "same_winner_donor": same_candidate,
                "source_interval": {k: v for k, v in rec_audit.items() if k != "ids"},
            })

        tokenized = {
            "recipient": tokenize_batch(tokenizer, recipient_prompts),
            "different": tokenize_batch(tokenizer, different_prompts),
            "same": tokenize_batch(tokenizer, same_prompts),
        }
        widths = {name: values[0].shape[1] for name, values in tokenized.items()}
        if len(set(widths.values())) != 1:
            raise RuntimeError(f"Cohort {cohort_index}: donor batch widths differ: {widths}")
        width = widths["recipient"]
        source_positions = []
        for prompt, positions in zip(recipient_prompts, source_unpadded):
            length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            source_positions.append([width - length + value for value in positions])

        donor_collectors = {}
        for name in ("different", "same"):
            input_ids, mask, last = tokenized[name]
            collector = Mixer56StateCollector(parts, last)
            try:
                _forward(model, parts, input_ids, mask)
                collector.validate()
            finally:
                collector.close()
            donor_collectors[name] = collector

        scenario_logits = np.empty((len(SCENARIOS), 4, 4), dtype=np.float32)
        scenario_writes = np.empty((len(SCENARIOS), 4, 4), dtype=np.float32)
        input_ids, mask, last = tokenized["recipient"]
        natural_collector = Mixer56StateCollector(parts, last)
        try:
            natural_output = _forward(model, parts, input_ids, mask)
            natural_collector.validate()
        finally:
            natural_collector.close()
        scenario_logits[0] = _aggregate_logits(natural_output, last, variant_ids)
        scenario_writes[0] = _project_write(natural_collector.write, output_rows)

        specs = {
            "different_query": ("different", True, False, False, False),
            "different_gate": ("different", False, True, False, False),
            "different_query_gate": ("different", True, True, False, False),
            "different_kv": ("different", False, False, True, False),
            "different_query_gate_kv": ("different", True, True, True, False),
            "different_all_heads": ("different", True, True, True, True),
            "same_winner_query_gate_kv": ("same", True, True, True, False),
        }
        for scenario_index, scenario in enumerate(SCENARIOS[1:], 1):
            source_name, patch_q, patch_g, patch_kv, all_heads = specs[scenario]
            patcher = Mixer56StatePatcher(
                parts,
                last,
                source_positions,
                donor_collectors[source_name],
                patch_q,
                patch_g,
                patch_kv,
                all_heads,
            )
            try:
                result = _forward(model, parts, input_ids, mask)
            finally:
                patcher.close()
            scenario_logits[scenario_index] = _aggregate_logits(result, last, variant_ids)
            scenario_writes[scenario_index] = _project_write(patcher.write, output_rows)

        _atomic_npz(
            path,
            question_ids=np.asarray(batch_qids),
            scenario_logits=scenario_logits,
            scenario_write_ad=scenario_writes,
            semantic_to_displayed=np.asarray(semantic_maps, dtype=np.int8),
            recipient_winner_semantic=np.asarray(recipient_semantics, dtype=np.int8),
            donor_winner_semantic=np.asarray(donor_semantics, dtype=np.int8),
            same_winner_control_available=np.asarray(same_available, dtype=bool),
        )
        if audit is None:
            audit = {
                "cohort_index": cohort_index,
                "width": width,
                "rows": row_audits,
                "selected_query_heads": list(SELECTED_HEADS),
                "selected_kv_heads": sorted({
                    head // (n_heads // n_kv_heads) for head in SELECTED_HEADS
                }),
            }
            (output / "prompt_audit.json").write_text(
                json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
            )
        completed += 1
        elapsed = time.perf_counter() - start_time
        print(
            f"cohort {cohort_index + 1}/125 complete; session={completed}; "
            f"elapsed={elapsed:.1f}s; mean={elapsed/completed:.1f}s/cohort",
            flush=True,
        )
        if max_cohorts is not None and completed >= max_cohorts:
            break

    if len(list(cohort_dir.glob("cohort_*.npz"))) == 125:
        metadata = {
            "status": "complete",
            "config": config.as_dict(),
            "second_mapping_plan": str(second_mapping_path),
            "mapping_plans": [str(path) for path in mapping_paths],
            "discovery_donor_plan": str(discovery_donor_path),
            "confirmation_donor_plan": str(confirmation_donor_path),
            "scenarios": list(SCENARIOS),
            "mixer_block": 56,
            "mixer_layer_zero_based": LAYER,
            "selected_query_heads": list(SELECTED_HEADS),
            "n_query_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "selected_kv_heads": sorted({head // (n_heads // n_kv_heads) for head in SELECTED_HEADS}),
            "resolved_answer_tokens": resolved,
            "resolved_model_commit": getattr(model.config, "_commit_hash", None),
            "software": {
                "python": sys.version,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "platform": platform.platform(),
        }
        (output / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--second-mapping", type=Path, required=True)
    parser.add_argument("--mapping-plans", type=Path, nargs=3, required=True)
    parser.add_argument("--discovery-donor", type=Path, required=True)
    parser.add_argument("--confirmation-donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.second_mapping,
        args.mapping_plans,
        args.discovery_donor,
        args.confirmation_donor,
        args.output,
        args.max_cohorts,
    )


if __name__ == "__main__":
    main()
