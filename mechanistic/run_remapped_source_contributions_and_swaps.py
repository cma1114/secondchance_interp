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
from .prompts import (
    CHOICE_CUE,
    TOKEN_MATCHED_TEST_GAME_FEEDBACK,
    TOKEN_MATCHED_TEST_NEUTRAL_FEEDBACK,
    present_question,
    prompt_hash,
)
from .sublayer import _hidden, _replace_hidden


LETTERS = "ABCD"
CONDITIONS = ("incorrect", "neutral")
FEEDBACKS = {
    "incorrect": TOKEN_MATCHED_TEST_GAME_FEEDBACK,
    "neutral": TOKEN_MATCHED_TEST_NEUTRAL_FEEDBACK,
}
SWAP_NAMES = ("evaluation", "action_target", "feedback_end")
SOURCE_READOUTS = tuple(range(48, 57))
MIXER_LAYERS = (51, 55)  # zero-based: user-facing Mixers 52 and 56
GROUP_NAMES = (
    "other_structure",
    "system",
    "first_instruction",
    "first_question_stem",
    "first_option_w1",
    "first_option_w2",
    "first_option_w1w2",
    "first_option_other",
    "first_choice_cue",
    "historical_assistant",
    *tuple(f"feedback_slot_{index}" for index in range(12)),
    "second_instruction",
    "second_question_stem",
    "second_option_w1",
    "second_option_w2",
    "second_option_w1w2",
    "second_option_other",
    "second_choice_cue",
    "final_assistant_prefix",
)
GROUP_INDEX = {name: index for index, name in enumerate(GROUP_NAMES)}


def _chunks(values: list[int], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _find_after(text: str, needle: str, start: int) -> tuple[int, int]:
    index = text.find(needle, start)
    if index < 0:
        raise RuntimeError(f"Could not locate {needle!r} after character {start}")
    return index, index + len(needle)


def _token_positions_for_interval(
    offsets: list[tuple[int, int]], interval: tuple[int, int]
) -> list[int]:
    left, right = interval
    return [
        index for index, (start, end) in enumerate(offsets)
        if end > start and start < right and end > left
    ]


def _locate_tokens(tokenizer: Any, prompt: str, condition: str) -> dict[str, Any]:
    feedback = FEEDBACKS[condition]
    feedback_range = _find_after(prompt, feedback, 0)
    if prompt.find(feedback, feedback_range[1]) >= 0:
        raise RuntimeError(f"Expected one feedback sentence in {condition}")
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]

    evaluation_text = "incorrect" if condition == "incorrect" else "lost"
    evaluation_range = _find_after(prompt, evaluation_text, feedback_range[0])
    action_clause = (
        "Choose a different answer."
        if condition == "incorrect"
        else "Choose the answer again."
    )
    action_range = _find_after(prompt, action_clause, feedback_range[0])
    action_text = "different" if condition == "incorrect" else "answer"
    action_target_range = _find_after(prompt, action_text, action_range[0])
    if action_target_range[1] > action_range[1]:
        raise RuntimeError("Action target escaped the action clause")
    feedback_end_range = (feedback_range[1] - 1, feedback_range[1])

    positions = {}
    for name, interval in (
        ("evaluation", evaluation_range),
        ("action_target", action_target_range),
        ("feedback_end", feedback_end_range),
    ):
        found = _token_positions_for_interval(offsets, interval)
        if len(found) != 1:
            raise RuntimeError(f"Expected one token for {condition} {name}; got {found}")
        positions[name] = found[0]
    return {
        "ids": ids,
        "offsets": offsets,
        "positions": positions,
        "decoded": {name: tokenizer.decode([ids[position]]) for name, position in positions.items()},
        "feedback_range": feedback_range,
    }


def _assign_groups(
    prompt: str,
    token_data: dict[str, Any],
    messages: list[dict[str, str]],
    original_question: dict[str, Any],
    remapped_question: dict[str, Any],
    condition: str,
    w1: str,
    w2: str,
    plan_row: dict[str, Any],
) -> np.ndarray:
    offsets = token_data["offsets"]
    groups = np.full(len(offsets), GROUP_INDEX["other_structure"], dtype=np.int16)

    cursor = 0
    system_range = _find_after(prompt, messages[0]["content"], cursor)
    cursor = system_range[1]
    first_user_range = _find_after(prompt, messages[1]["content"], cursor)
    cursor = first_user_range[1]
    second_user_range = _find_after(prompt, messages[3]["content"], cursor)
    first_question = _find_after(prompt, present_question(original_question), first_user_range[0])
    second_question = _find_after(prompt, present_question(remapped_question), second_user_range[0])
    first_choice = _find_after(prompt, CHOICE_CUE, first_question[1])
    second_choice = _find_after(prompt, CHOICE_CUE, second_question[1])

    intervals: list[tuple[str, tuple[int, int]]] = [
        ("system", system_range),
        ("first_instruction", (first_user_range[0], first_question[0])),
        ("first_question_stem", first_question),
        ("first_choice_cue", (first_question[1], first_user_range[1])),
        ("historical_assistant", (first_user_range[1], second_user_range[0])),
        ("second_instruction", (token_data["feedback_range"][1], second_question[0])),
        ("second_question_stem", second_question),
        ("second_choice_cue", (second_question[1], second_user_range[1])),
        ("final_assistant_prefix", (second_user_range[1], len(prompt))),
    ]

    # Broad regions first; option and feedback-token intervals below override.
    for name, interval in intervals:
        for position in _token_positions_for_interval(offsets, interval):
            groups[position] = GROUP_INDEX[name]

    for displayed in LETTERS:
        option_range = _find_after(
            prompt, f"  {displayed}: {original_question['options'][displayed]}", first_question[0]
        )
        if displayed == w1 == w2:
            name = "first_option_w1w2"
        elif displayed == w1:
            name = "first_option_w1"
        elif displayed == w2:
            name = "first_option_w2"
        else:
            name = "first_option_other"
        for position in _token_positions_for_interval(offsets, option_range):
            groups[position] = GROUP_INDEX[name]

        semantic = plan_row["new_to_original"][displayed]
        option_range = _find_after(
            prompt, f"  {displayed}: {remapped_question['options'][displayed]}", second_question[0]
        )
        if semantic == w1 == w2:
            name = "second_option_w1w2"
        elif semantic == w1:
            name = "second_option_w1"
        elif semantic == w2:
            name = "second_option_w2"
        else:
            name = "second_option_other"
        for position in _token_positions_for_interval(offsets, option_range):
            groups[position] = GROUP_INDEX[name]

    feedback_positions = _token_positions_for_interval(offsets, token_data["feedback_range"])
    if len(feedback_positions) > 12:
        raise RuntimeError(f"Feedback has {len(feedback_positions)} tokens; increase slot count")
    for slot, position in enumerate(feedback_positions):
        groups[position] = GROUP_INDEX[f"feedback_slot_{slot}"]
    return groups


class PositionReadoutCollector:
    def __init__(self, parts: Any, positions: dict[str, list[int]]) -> None:
        self.positions = positions
        self.values: dict[str, dict[int, Any]] = {
            name: {} for name in SWAP_NAMES
        }
        self.handles = [
            parts.layers[readout - 1].register_forward_hook(self._hook(readout))
            for readout in SOURCE_READOUTS
        ]

    def _hook(self, readout: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = _hidden(output)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            for name in SWAP_NAMES:
                cols = torch.as_tensor(self.positions[name], device=hidden.device)
                self.values[name][readout] = hidden[rows, cols].detach().clone()
        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class PositionReadoutPatcher:
    def __init__(self, parts: Any, readout: int, positions: list[int], source: Any) -> None:
        self.positions = positions
        self.source = source
        self.handle = parts.layers[readout - 1].register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = _hidden(output)
        updated = hidden.clone()
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        cols = torch.as_tensor(self.positions, device=hidden.device)
        replacement = self.source.to(device=hidden.device, dtype=hidden.dtype)
        if replacement.shape != updated[rows, cols].shape:
            raise RuntimeError("Source and target token states have different shapes")
        updated[rows, cols] = replacement
        return _replace_hidden(output, updated)

    def close(self) -> None:
        self.handle.remove()


class SDPAFinalSourceCollector:
    """Observe exact SDPA inputs and decompose final-query attention writes."""

    def __init__(self, parts: Any, last_indices: list[int], canonical_ids: list[int]) -> None:
        import torch

        self.parts = parts
        self.last_indices = [int(value) for value in last_indices]
        self.layers = {index: parts.layers[index].self_attn for index in MIXER_LAYERS}
        self.active: int | None = None
        self.gates: dict[int, Any] = {}
        self.direct: dict[int, Any] = {}
        self.weights: dict[int, Any] = {}
        self.context_error: dict[int, float] = {}
        self.predicted_context: dict[int, Any] = {}
        self.output_rows = parts.output_head.weight.detach()[canonical_ids].float()
        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        self.handles = []
        for layer_index, attention in self.layers.items():
            self.handles.extend([
                attention.register_forward_pre_hook(self._enter(layer_index)),
                attention.register_forward_hook(self._leave(layer_index)),
                attention.q_proj.register_forward_hook(self._gate_hook(layer_index)),
                attention.o_proj.register_forward_pre_hook(self._context_hook(layer_index)),
            ])
        torch.nn.functional.scaled_dot_product_attention = self._wrapped_sdpa

    def _enter(self, layer_index: int):
        def enter(_module: Any, _inputs: Any) -> None:
            if self.active is not None:
                raise RuntimeError("Nested ordinary attention calls are unsupported")
            self.active = layer_index
        return enter

    def _leave(self, layer_index: int):
        def leave(_module: Any, _inputs: Any, _output: Any) -> None:
            if self.active != layer_index:
                raise RuntimeError("Attention collector layer stack became inconsistent")
            self.active = None
        return leave

    def _gate_hook(self, layer_index: int):
        def capture(module: Any, _inputs: Any, output: Any) -> None:
            import torch

            head_dim = int(self.layers[layer_index].head_dim)
            shaped = output.view(*output.shape[:-1], -1, 2 * head_dim)
            _query, gate = shaped.chunk(2, dim=-1)
            rows = torch.arange(output.shape[0], device=output.device)
            cols = torch.as_tensor(self.last_indices, device=output.device)
            self.gates[layer_index] = gate[rows, cols].sigmoid()
        return capture

    def _context_hook(self, layer_index: int):
        def validate(_module: Any, inputs: Any) -> None:
            import torch

            if layer_index not in self.predicted_context:
                raise RuntimeError("SDPA context was not captured before o_proj")
            hidden = inputs[0]
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            cols = torch.as_tensor(self.last_indices, device=hidden.device)
            actual = hidden[rows, cols].reshape_as(self.predicted_context[layer_index])
            error = (actual.float() - self.predicted_context[layer_index].float()).abs().max()
            self.context_error[layer_index] = float(error.item())
        return validate

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        import torch

        result = self.original_sdpa(query, key, value, *args, **kwargs)
        layer_index = self.active
        if layer_index not in self.layers:
            return result
        gate = self.gates.get(layer_index)
        if gate is None:
            raise RuntimeError("Qwen output gate was not captured")

        attn_mask = kwargs.get("attn_mask", args[0] if args else None)
        rows = torch.arange(query.shape[0], device=query.device)
        cols = torch.as_tensor(self.last_indices, device=query.device)
        final_query = query[rows, :, cols, :].unsqueeze(2)
        final_mask = None
        if attn_mask is not None:
            if attn_mask.ndim != 4:
                raise RuntimeError(f"Unexpected SDPA mask shape {attn_mask.shape}")
            final_mask = attn_mask[rows, :, cols, : key.shape[-2]].unsqueeze(2)

        # Recover the weights from the *same SDPA kernel*, rather than from a
        # separate explicit softmax whose reduced-precision numerics can differ
        # materially on this model.  Synthetic one-hot values make each output
        # dimension equal the attention weight of one source position.  At
        # most two cheap final-query calls are needed because head_dim=256 and
        # these prompts have fewer than 512 source tokens.
        key_length = key.shape[-2]
        head_dim = value.shape[-1]
        weights = torch.empty(
            query.shape[0], query.shape[1], key_length,
            device=query.device, dtype=value.dtype,
        )
        auxiliary_kwargs = {
            "attn_mask": final_mask,
            "dropout_p": 0.0,
            "scale": kwargs.get("scale"),
            "is_causal": False,
        }
        if kwargs.get("enable_gqa", False):
            auxiliary_kwargs["enable_gqa"] = True
        for start in range(0, key_length, head_dim):
            end = min(start + head_dim, key_length)
            synthetic = torch.zeros_like(value)
            positions = torch.arange(start, end, device=value.device)
            dimensions = torch.arange(end - start, device=value.device)
            synthetic[:, :, positions, dimensions] = 1
            recovered = self.original_sdpa(
                final_query, key, synthetic, **auxiliary_kwargs
            )
            weights[:, :, start:end] = recovered[:, :, 0, : end - start]

        if value.shape[1] != query.shape[1]:
            if query.shape[1] % value.shape[1]:
                raise RuntimeError("Query heads are not divisible by KV heads")
            repeat = query.shape[1] // value.shape[1]
            value_for_heads = value.repeat_interleave(repeat, dim=1)
        else:
            value_for_heads = value
        context = torch.einsum("bhk,bhkd->bhd", weights, value_for_heads)
        gated_context = context * gate

        attention = self.layers[layer_index]
        n_heads, head_dim = gated_context.shape[1:]
        projection = attention.o_proj.weight.detach().float().reshape(
            attention.o_proj.out_features, n_heads, head_dim
        )
        effective = torch.einsum("co,ohd->chd", self.output_rows, projection)
        gated_values = value_for_heads * gate[:, :, None, :]
        direct = (
            torch.einsum("bhkd,chd->bhkc", gated_values.float(), effective)
            * weights[..., None]
        )
        self.direct[layer_index] = direct.permute(0, 2, 1, 3).detach().to(
            "cpu", dtype=torch.float16
        )
        self.weights[layer_index] = weights.permute(0, 2, 1).detach().to(
            "cpu", dtype=torch.float16
        )
        self.predicted_context[layer_index] = gated_context.detach()
        return result

    def close(self) -> None:
        import torch

        torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in self.handles:
            handle.remove()
        self.active = None

    def arrays(self) -> tuple[Any, Any, np.ndarray]:
        import torch

        missing = [layer for layer in MIXER_LAYERS if layer not in self.direct]
        if missing:
            raise RuntimeError(f"Missing source contributions for layers {missing}")
        direct = torch.stack([self.direct[layer] for layer in MIXER_LAYERS], dim=1)
        weights = torch.stack([self.weights[layer] for layer in MIXER_LAYERS], dim=1)
        errors = np.asarray([self.context_error[layer] for layer in MIXER_LAYERS], dtype=np.float32)
        return direct.numpy(), weights.numpy(), errors


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


def _aggregate_logits(output: Any, last_indices: list[int], variant_ids: list[list[int]]) -> np.ndarray:
    import torch

    logits = output.logits.detach().float()
    if logits.shape[1] == 1:
        final = logits[:, 0]
    else:
        rows = torch.arange(logits.shape[0], device=logits.device)
        cols = torch.as_tensor(last_indices, device=logits.device)
        final = logits[rows, cols]
    return torch.stack(
        [torch.logsumexp(final[:, ids], dim=-1) for ids in variant_ids], dim=-1
    ).cpu().numpy()


def run(
    config_path: Path,
    plan_path: Path,
    original_baseline_path: Path,
    remapped_baseline_path: Path,
    output: Path,
    max_cohorts: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires exact batch-of-four SDPA execution")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw Qwen ChatML")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    plan_rows = {
        row["question_id"]: row for row in json.loads(plan_path.read_text())["rows"]
    }
    original = json.loads(original_baseline_path.read_text())["results"]
    remapped_baseline = json.loads(remapped_baseline_path.read_text())["results"]

    output.mkdir(parents=True, exist_ok=True)
    cohort_dir = output / "cohorts"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = [[token_id for _, token_id in resolved[letter]] for letter in LETTERS]
    canonical_ids = [resolved[letter][0][1] for letter in LETTERS]

    start_time = time.perf_counter()
    session_done = 0
    first_row_audit = None
    for cohort_index, indices in enumerate(_chunks(list(range(len(qids))), 4)):
        path = cohort_dir / f"cohort_{cohort_index:03d}.npz"
        if path.exists():
            continue
        batch_qids = [qids[index] for index in indices]
        remapped_questions = [
            _remap_question(questions[qid], plan_rows[qid]["new_to_original"])
            for qid in batch_qids
        ]
        prompts: dict[str, list[str]] = {}
        messages: dict[str, list[list[dict[str, str]]]] = {}
        token_data: dict[str, list[dict[str, Any]]] = {}
        tokenized = {}
        group_codes = {}
        for condition in CONDITIONS:
            messages[condition] = [
                _messages(config, questions[qid], remapped_question, condition)
                for qid, remapped_question in zip(batch_qids, remapped_questions)
            ]
            prompts[condition] = [
                render_chat(processor, row, config.disable_thinking, config.chat_serialization)
                for row in messages[condition]
            ]
            token_data[condition] = [
                _locate_tokens(tokenizer, prompt, condition) for prompt in prompts[condition]
            ]
            tokenized[condition] = tokenize_batch(tokenizer, prompts[condition])
            group_codes[condition] = [
                _assign_groups(
                    prompt,
                    data,
                    message,
                    questions[qid],
                    remapped_question,
                    condition,
                    original[qid]["answer"],
                    remapped_baseline[qid]["answer_original_content"],
                    plan_rows[qid],
                )
                for prompt, data, message, qid, remapped_question in zip(
                    prompts[condition], token_data[condition], messages[condition],
                    batch_qids, remapped_questions
                )
            ]

        physical_positions: dict[str, dict[str, list[int]]] = {
            condition: {name: [] for name in SWAP_NAMES} for condition in CONDITIONS
        }
        for row in range(4):
            game = token_data["incorrect"][row]
            neutral = token_data["neutral"][row]
            if len(game["ids"]) != len(neutral["ids"]):
                raise RuntimeError("Token-matched paired prompts have unequal lengths")
            width = tokenized["incorrect"][0].shape[1]
            game_pad = width - len(game["ids"])
            neutral_pad = width - len(neutral["ids"])
            for name in SWAP_NAMES:
                gp = game_pad + game["positions"][name]
                np_ = neutral_pad + neutral["positions"][name]
                if gp != np_:
                    raise RuntimeError(f"{name} is not physically position aligned")
                physical_positions["incorrect"][name].append(gp)
                physical_positions["neutral"][name].append(np_)

        if cohort_index == 0:
            first_row_audit = {
                condition: {
                    name: {
                        "decoded": token_data[condition][0]["decoded"][name],
                        "unpadded_position": int(token_data[condition][0]["positions"][name]),
                        "physical_position": int(physical_positions[condition][name][0]),
                    }
                    for name in SWAP_NAMES
                }
                for condition in CONDITIONS
            }

        natural_logits = np.empty((2, 4, 4), dtype=np.float32)
        patched_logits = np.empty(
            (len(SWAP_NAMES), 2, len(SOURCE_READOUTS), 4, 4), dtype=np.float32
        )
        states: dict[str, dict[str, dict[int, Any]]] = {}
        source_direct = {}
        source_attention = {}
        source_errors = {}
        widths = []
        for ci, condition in enumerate(CONDITIONS):
            input_ids, mask, last = tokenized[condition]
            widths.append(input_ids.shape[1])
            state_collector = PositionReadoutCollector(parts, physical_positions[condition])
            source_collector = SDPAFinalSourceCollector(parts, last, canonical_ids)
            try:
                result = _forward(model, parts, input_ids, mask)
                direct, attention, errors = source_collector.arrays()
            finally:
                source_collector.close()
                state_collector.close()
            natural_logits[ci] = _aggregate_logits(result, last, variant_ids)
            states[condition] = state_collector.values
            source_direct[condition] = direct
            source_attention[condition] = attention
            source_errors[condition] = errors

        if widths[0] != widths[1]:
            raise RuntimeError("Game and Neutral cohort widths differ")
        for si, name in enumerate(SWAP_NAMES):
            for ci, target in enumerate(CONDITIONS):
                source = "neutral" if target == "incorrect" else "incorrect"
                input_ids, mask, last = tokenized[target]
                for ri, readout in enumerate(SOURCE_READOUTS):
                    patcher = PositionReadoutPatcher(
                        parts,
                        readout,
                        physical_positions[target][name],
                        states[source][name][readout],
                    )
                    try:
                        result = _forward(model, parts, input_ids, mask)
                    finally:
                        patcher.close()
                    patched_logits[si, ci, ri] = _aggregate_logits(result, last, variant_ids)

        width = widths[0]
        token_ids = np.full((2, 4, width), -1, dtype=np.int32)
        token_groups = np.full((2, 4, width), -1, dtype=np.int16)
        positions_array = np.empty((2, len(SWAP_NAMES), 4), dtype=np.int16)
        for ci, condition in enumerate(CONDITIONS):
            for row in range(4):
                ids = token_data[condition][row]["ids"]
                pad = width - len(ids)
                token_ids[ci, row, pad:] = ids
                token_groups[ci, row, pad:] = group_codes[condition][row]
            for si, name in enumerate(SWAP_NAMES):
                positions_array[ci, si] = physical_positions[condition][name]

        _atomic_npz(
            path,
            question_ids=np.asarray(batch_qids),
            natural_logits=natural_logits,
            patched_logits=patched_logits,
            source_direct_ad=np.stack([source_direct[c] for c in CONDITIONS]),
            source_attention=np.stack([source_attention[c] for c in CONDITIONS]),
            source_context_max_error=np.stack([source_errors[c] for c in CONDITIONS]),
            token_ids=token_ids,
            token_groups=token_groups,
            swap_positions=positions_array,
            prompt_hashes=np.asarray([[prompt_hash(p) for p in prompts[c]] for c in CONDITIONS]),
        )
        session_done += 1
        elapsed = time.perf_counter() - start_time
        print(
            f"cohort {cohort_index + 1}/125 complete; session={session_done}; "
            f"elapsed={elapsed:.1f}s; mean={elapsed/session_done:.1f}s/cohort",
            flush=True,
        )
        if max_cohorts is not None and session_done >= max_cohorts:
            break

    if len(list(cohort_dir.glob("cohort_*.npz"))) == 125:
        if first_row_audit is None:
            with np.load(cohort_dir / "cohort_000.npz", allow_pickle=False) as first:
                first_row_audit = {}
                for ci, condition in enumerate(CONDITIONS):
                    first_row_audit[condition] = {}
                    for si, name in enumerate(SWAP_NAMES):
                        position = int(first["swap_positions"][ci, si, 0])
                        token_id = int(first["token_ids"][ci, 0, position])
                        first_row_audit[condition][name] = {
                            "decoded": tokenizer.decode([token_id]),
                            "physical_position": position,
                        }
        metadata = {
            "status": "complete",
            "config": config.as_dict(),
            "plan": str(plan_path),
            "n_questions": len(qids),
            "conditions": list(CONDITIONS),
            "swap_names": list(SWAP_NAMES),
            "source_readouts": list(SOURCE_READOUTS),
            "mixer_layers_zero_based": list(MIXER_LAYERS),
            "mixer_blocks_one_based": [layer + 1 for layer in MIXER_LAYERS],
            "group_names": list(GROUP_NAMES),
            "token_audit_first_row": first_row_audit,
            "resolved_answer_tokens": resolved,
            "resolved_model_commit": getattr(model.config, "_commit_hash", None),
            "measurement": (
                "Exact additive per-token/per-head residual writes reconstructed from "
                "the unmodified SDPA inputs at the final query, projected onto canonical A-D rows"
            ),
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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--original-baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.plan,
        args.original_baseline,
        args.remapped_baseline,
        args.output,
        args.max_cohorts,
    )


if __name__ == "__main__":
    main()
