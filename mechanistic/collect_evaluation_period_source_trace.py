from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .collect_evaluation_gla_residual_writes import CONDITIONS, _aggregate_logits, _chunks
from .collect_remapped_feedback_factorial import _messages, _remap_question
from .config import ExperimentConfig
from .downstream_source_intervention import BatchedSelectiveGDNSourceWriteAblator
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import ANSWER_ONLY_INSTRUCTION, FACTORIAL_FEEDBACK, CHOICE_CUE
from .run_evaluation_update_transplant import _locate_evaluation


ROLE_NAMES = (
    "evaluation period",
    "action clause",
    "answer-only instruction",
    "repeated question",
    "option A",
    "option B",
    "option C",
    "option D",
    "choice cue",
    "assistant decision scaffold",
)
ANCHOR_NAMES = (
    "evaluation period",
    "action-clause end",
    "answer-only-instruction end",
    "repeated-question end",
    "option A end",
    "option B end",
    "option C end",
    "option D end",
    "choice-cue end",
    "final decision",
)


def _overlap_position(offsets: list[tuple[int, int]], character: int) -> int:
    hits = [i for i, (left, right) in enumerate(offsets) if right > left and left <= character < right]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one token at character {character}; got {hits}")
    return hits[0]


def _last_token_before(offsets: list[tuple[int, int]], character: int) -> int:
    hits = [i for i, (left, right) in enumerate(offsets) if right > left and right <= character]
    if not hits:
        raise RuntimeError(f"No token ends before character {character}")
    return hits[-1]


def _locate_downstream(
    tokenizer: Any,
    prompt: str,
    condition: str,
    remapped: dict[str, Any],
) -> dict[str, Any]:
    located = _locate_evaluation(tokenizer, prompt, condition)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.asarray(encoded["input_ids"], dtype=np.int32)
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    source = int(located["period_position"])
    feedback = FACTORIAL_FEEDBACK[condition]
    feedback_start = prompt.find(feedback)
    if feedback_start < 0:
        raise RuntimeError("Could not locate feedback")
    feedback_end = feedback_start + len(feedback)
    action_end = _overlap_position(offsets, feedback_end - 1)

    instruction_start = prompt.find(ANSWER_ONLY_INSTRUCTION, feedback_end)
    if instruction_start < 0:
        raise RuntimeError("Could not locate repeated answer-only instruction")
    instruction_end_char = instruction_start + len(ANSWER_ONLY_INSTRUCTION)
    instruction_end = _overlap_position(offsets, instruction_end_char - 1)

    question_start = prompt.find(remapped["question"], instruction_end_char)
    if question_start < 0:
        raise RuntimeError("Could not locate remapped question")
    question_end_char = question_start + len(remapped["question"])
    question_end = _overlap_position(offsets, question_end_char - 1)

    option_ranges: list[tuple[int, int]] = []
    option_ends: list[int] = []
    search_from = question_end_char
    for letter in LETTERS:
        line = f"  {letter}: {remapped['options'][letter]}\n"
        start = prompt.find(line, search_from)
        if start < 0:
            raise RuntimeError(f"Could not locate remapped option {letter}")
        end = start + len(line)
        option_ranges.append((start, end))
        option_ends.append(_overlap_position(offsets, end - 1))
        search_from = end

    choice_start = prompt.find(CHOICE_CUE, search_from)
    if choice_start < 0:
        raise RuntimeError("Could not locate final choice cue")
    choice_end_char = choice_start + len(CHOICE_CUE)
    choice_end = _last_token_before(offsets, choice_end_char)

    role = np.full(len(ids), -1, dtype=np.int8)
    # Character ranges are mapped to every non-special token they overlap.
    ranges = [
        (offsets[source][0], offsets[source][1], 0),
        (offsets[source][1], feedback_end, 1),
        (instruction_start, instruction_end_char, 2),
        (question_start, question_end_char, 3),
        *[(start, end, 4 + i) for i, (start, end) in enumerate(option_ranges)],
        (choice_start, choice_end_char, 8),
    ]
    for index, (left, right) in enumerate(offsets):
        for start, end, value in ranges:
            if right > left and left < end and right > start:
                role[index] = value
                break
    # ChatML and the empty-thinking scaffold after the last user turn have no
    # ordinary character offsets in some tokenizer versions.
    role[choice_end + 1 :] = 9
    role[source] = 0

    anchors = np.asarray(
        [source, action_end, instruction_end, question_end, *option_ends, choice_end, len(ids) - 1],
        dtype=np.int32,
    )
    if len(anchors) != len(ANCHOR_NAMES):
        raise AssertionError("Anchor list and labels disagree")
    if not np.all(anchors >= source):
        raise RuntimeError("A downstream anchor precedes the evaluation period")
    return {
        **located,
        "ids_array": ids,
        "roles": role,
        "anchors": anchors,
    }


class EvaluationPeriodSourceTrace:
    """Exact within-GLA deletion effect of the period's recurrent write.

    Each GLA is replayed with beta=0 only at the evaluation-closing period.
    Natural computation is returned to the model.  The saved natural-minus-
    replay output therefore measures what that source-token write contributes
    when each later token queries the recurrent memory, including all later
    decay and delta-rule interactions inside that GLA.
    """

    def __init__(
        self,
        parts: Any,
        layers: list[int],
        source_positions: list[int],
        canonical_rows: Any,
    ) -> None:
        import torch

        self.parts = parts
        self.layers = layers
        self.source_positions = [int(value) for value in source_positions]
        self.canonical_rows = canonical_rows.float()
        self.z: dict[int, Any] = {}
        self.effects: dict[int, tuple[Any, Any]] = {}
        self.originals: list[tuple[Any, Any]] = []
        self.handles: list[Any] = []
        self.modeling_module: Any | None = None
        self.original_global_rule: Any | None = None
        self.active_layer: int | None = None

        modules = {layer: parts.layers[layer].linear_attn for layer in layers}
        for layer, module in modules.items():
            self.handles.append(module.in_proj_z.register_forward_hook(self._capture_z(layer)))

        use_global = all(not hasattr(module, "chunk_gated_delta_rule") for module in modules.values())
        if use_global:
            modeling_modules = {inspect.getmodule(type(module)) for module in modules.values()}
            if len(modeling_modules) != 1 or None in modeling_modules:
                raise RuntimeError("Could not locate a unique Qwen GLA modeling module")
            self.modeling_module = modeling_modules.pop()
            self.original_global_rule = self.modeling_module.torch_chunk_gated_delta_rule
            for layer, module in modules.items():
                self.handles.append(module.register_forward_pre_hook(self._mark(layer)))
                self.handles.append(module.register_forward_hook(self._clear))
            original = self.original_global_rule

            def wrapped(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
                layer = self.active_layer
                if layer is None:
                    return original(query, key, value, *args, **kwargs)
                return self._replay(layer, original, query, key, value, args, kwargs)

            self.modeling_module.torch_chunk_gated_delta_rule = wrapped
        else:
            if not all(hasattr(module, "chunk_gated_delta_rule") for module in modules.values()):
                raise RuntimeError("Mixed Qwen GLA kernel layouts are unsupported")
            for layer, module in modules.items():
                original = module.chunk_gated_delta_rule

                def wrapped(query: Any, key: Any, value: Any, *args: Any,
                            _layer=layer, _original=original, **kwargs: Any):
                    return self._replay(_layer, _original, query, key, value, args, kwargs)

                self.originals.append((module, original))
                module.chunk_gated_delta_rule = wrapped

    def _capture_z(self, layer: int):
        def capture(module: Any, _inputs: Any, output: Any) -> None:
            self.z[layer] = output.reshape(
                output.shape[0], output.shape[1],
                module.out_features // self.parts.layers[layer].linear_attn.head_v_dim,
                self.parts.layers[layer].linear_attn.head_v_dim,
            )
        return capture

    def _mark(self, layer: int):
        def mark(_module: Any, _args: Any) -> None:
            if self.active_layer is not None:
                raise RuntimeError("Nested Qwen GLA calls are unsupported")
            self.active_layer = layer
        return mark

    def _clear(self, _module: Any, _args: Any, _output: Any) -> None:
        self.active_layer = None

    def _replay(
        self,
        layer: int,
        original: Any,
        query: Any,
        key: Any,
        value: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ):
        import torch

        if "beta" not in kwargs:
            raise RuntimeError("Qwen GLA rule did not pass beta by keyword")
        natural, state = original(query, key, value, *args, **kwargs)
        beta = kwargs["beta"].clone()
        for row, source in enumerate(self.source_positions):
            beta[row, source, :] = 0
        replay_kwargs = dict(kwargs)
        replay_kwargs["beta"] = beta
        replay_kwargs["output_final_state"] = False
        counterfactual, _ = original(query, key, value, *args, **replay_kwargs)

        module = self.parts.layers[layer].linear_attn
        z = self.z[layer]
        batch, sequence = natural.shape[:2]
        natural_flat = natural.reshape(-1, module.head_v_dim)
        counterfactual_flat = counterfactual.reshape(-1, module.head_v_dim)
        z_flat = z.reshape(-1, module.head_v_dim)
        natural_normed = module.norm(natural_flat, z_flat).reshape(batch, sequence, -1)
        counterfactual_normed = module.norm(counterfactual_flat, z_flat).reshape(batch, sequence, -1)
        projected = module.out_proj(natural_normed - counterfactual_normed)
        norm = torch.linalg.vector_norm(projected.float(), dim=-1)
        # `device_map="auto"` can place the output head and this GLA block on
        # different GPUs.  The four canonical rows are tiny; move them to the
        # actual write device before projecting instead of assuming the output
        # head and every traced block are co-located.
        canonical_rows = self.canonical_rows.to(
            device=projected.device, dtype=torch.float32, non_blocking=True
        )
        direct = projected.float() @ canonical_rows.T
        self.effects[layer] = (
            norm.detach().to("cpu", dtype=torch.float16),
            direct.detach().to("cpu", dtype=torch.float16),
        )
        return natural, state

    def stacked(self):
        import torch

        missing = [layer for layer in self.layers if layer not in self.effects]
        if missing:
            raise RuntimeError(f"Missing source traces for blocks {missing}")
        norms = torch.stack([self.effects[layer][0] for layer in self.layers], dim=1)
        direct = torch.stack([self.effects[layer][1] for layer in self.layers], dim=1)
        return norms, direct

    def close(self) -> None:
        for module, original in reversed(self.originals):
            module.chunk_gated_delta_rule = original
        self.originals.clear()
        if self.modeling_module is not None and self.original_global_rule is not None:
            self.modeling_module.torch_chunk_gated_delta_rule = self.original_global_rule
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


def _forward(model: Any, parts: Any, input_ids: Any, attention_mask: Any):
    import torch

    with torch.inference_mode():
        kwargs = {
            "input_ids": input_ids.to(model_input_device(parts)),
            "attention_mask": attention_mask.to(model_input_device(parts)),
            "use_cache": False,
            "return_dict": True,
        }
        try:
            return model(**kwargs, logits_to_keep=1)
        except TypeError:
            return model(**kwargs)


def collect(
    config_path: Path,
    remapping_plan_path: Path,
    output_dir: Path,
    max_questions: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if (
        config.prompt_mode != "baseline_matched_empty_history"
        or config.feedback_variant != "token_matched_test"
        or config.chat_serialization != "raw_qwen_chatml"
        or config.attn_implementation != "sdpa"
        or config.batch_size != 4
    ):
        raise ValueError("Requires the exact historical raw-ChatML, SDPA, batch-of-four regime")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    if max_questions is not None:
        qids = qids[: int(max_questions)]
    remapping = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    if not set(qids) <= set(remapping):
        raise ValueError("Remapping plan is missing questions")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    canonical_ids = [resolved[letter][0][1] for letter in LETTERS]
    canonical_rows = parts.output_head.weight.detach()[canonical_ids]
    layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if len(layers) != 48:
        raise RuntimeError(f"Expected 48 GLA blocks, found {len(layers)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "cohorts"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cohorts = list(_chunks(qids, config.batch_size))
    prompt_audit = None
    for cohort_index, cohort in enumerate(cohorts):
        shard = shard_dir / f"cohort_{cohort_index:03d}.npz"
        if shard.exists():
            continue
        saved: dict[str, list[np.ndarray]] = {
            "natural_logits": [], "ablated_logits": [], "trace_norm": [],
            "trace_ad": [], "token_ids": [], "role_ids": [],
            "anchor_relative_positions": [], "relative_lengths": [],
        }
        condition_prompts = []
        for condition in CONDITIONS:
            prompts, located_rows = [], []
            for qid in cohort:
                remapped_question = _remap_question(
                    questions[qid], remapping[qid]["new_to_original"]
                )
                messages = _messages(config, questions[qid], remapped_question, condition)
                prompt = render_chat(
                    processor, messages, config.disable_thinking, config.chat_serialization
                )
                located = _locate_downstream(tokenizer, prompt, condition, remapped_question)
                prompts.append(prompt)
                located_rows.append((messages, prompt, located))
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
            padded_length = int(input_ids.shape[1])
            source_positions = [
                int(row[2]["period_position"] + padded_length - len(row[2]["ids_array"]))
                for row in located_rows
            ]
            tracer = EvaluationPeriodSourceTrace(
                parts, layers, source_positions, canonical_rows
            )
            try:
                natural_output = _forward(model, parts, input_ids, attention_mask)
                trace_norm, trace_ad = tracer.stacked()
            finally:
                tracer.close()
            saved["natural_logits"].append(_aggregate_logits(natural_output, variant_ids))

            specs = {row: ([source_positions[row]], layers) for row in range(len(cohort))}
            ablator = BatchedSelectiveGDNSourceWriteAblator(
                parts, specs, preserve_source_output=True
            )
            try:
                ablated_output = _forward(model, parts, input_ids, attention_mask)
                ablator.assert_fired()
            finally:
                ablator.close()
            saved["ablated_logits"].append(_aggregate_logits(ablated_output, variant_ids))

            max_relative = max(padded_length - source for source in source_positions)
            norm_relative = np.full(
                (len(cohort), len(layers), max_relative), np.nan, dtype=np.float16
            )
            ad_relative = np.full(
                (len(cohort), len(layers), max_relative, 4), np.nan, dtype=np.float16
            )
            token_relative = np.full((len(cohort), max_relative), -1, dtype=np.int32)
            role_relative = np.full((len(cohort), max_relative), -1, dtype=np.int8)
            anchor_relative = np.full(
                (len(cohort), len(ANCHOR_NAMES)), -1, dtype=np.int32
            )
            lengths = np.zeros(len(cohort), dtype=np.int32)
            for row, (_messages_row, _prompt, located) in enumerate(located_rows):
                pad = padded_length - len(located["ids_array"])
                source = source_positions[row]
                length = padded_length - source
                lengths[row] = length
                norm_relative[row, :, :length] = trace_norm[row, :, source:].numpy()
                ad_relative[row, :, :length] = trace_ad[row, :, source:].numpy()
                token_relative[row, :length] = input_ids[row, source:].cpu().numpy()
                full_roles = np.full(padded_length, -1, dtype=np.int8)
                full_roles[pad:] = located["roles"]
                role_relative[row, :length] = full_roles[source:]
                anchor_relative[row] = located["anchors"] + pad - source
            saved["trace_norm"].append(norm_relative)
            saved["trace_ad"].append(ad_relative)
            saved["token_ids"].append(token_relative)
            saved["role_ids"].append(role_relative)
            saved["anchor_relative_positions"].append(anchor_relative)
            saved["relative_lengths"].append(lengths)
            condition_prompts.append(located_rows)

        max_relative = max(array.shape[-1] for array in saved["trace_norm"])
        for key in ("trace_norm", "token_ids", "role_ids"):
            axis = -1
            fill = np.nan if key == "trace_norm" else -1
            padded = []
            for array in saved[key]:
                width = max_relative - array.shape[axis]
                pads = [(0, 0)] * array.ndim
                pads[axis] = (0, width)
                padded.append(np.pad(array, pads, constant_values=fill))
            saved[key] = padded
        padded_ad = []
        for array in saved["trace_ad"]:
            padded_ad.append(np.pad(
                array, ((0, 0), (0, 0), (0, max_relative - array.shape[-2]), (0, 0)),
                constant_values=np.nan,
            ))
        saved["trace_ad"] = padded_ad
        atomic_save_npz(
            shard,
            question_ids=np.asarray(cohort),
            natural_logits=np.stack(saved["natural_logits"]),
            ablated_logits=np.stack(saved["ablated_logits"]),
            trace_norm=np.stack(saved["trace_norm"]),
            trace_ad=np.stack(saved["trace_ad"]),
            token_ids=np.stack(saved["token_ids"]),
            role_ids=np.stack(saved["role_ids"]),
            anchor_relative_positions=np.stack(saved["anchor_relative_positions"]),
            relative_lengths=np.stack(saved["relative_lengths"]),
        )
        if prompt_audit is None:
            prompt_audit = {
                condition: {
                    "question_id": cohort[0],
                    "messages": condition_prompts[i][0][0],
                    "prompt": condition_prompts[i][0][1],
                    "source_token": tokenizer.decode([
                        int(condition_prompts[i][0][2]["ids_array"][
                            condition_prompts[i][0][2]["period_position"]
                        ])
                    ]),
                    "anchor_tokens": {
                        name: tokenizer.decode([
                            int(condition_prompts[i][0][2]["ids_array"][position])
                        ])
                        for name, position in zip(
                            ANCHOR_NAMES, condition_prompts[i][0][2]["anchors"]
                        )
                    },
                }
                for i, condition in enumerate(CONDITIONS)
            }
        print(f"Evaluation-period source trace: {cohort_index + 1}/{len(cohorts)} cohorts", flush=True)

    metadata = {
        "config": config.as_dict(),
        "remapping_plan": str(remapping_plan_path),
        "question_ids": qids,
        "conditions": list(CONDITIONS),
        "gla_layers_zero_based": layers,
        "role_names": list(ROLE_NAMES),
        "anchor_names": list(ANCHOR_NAMES),
        "n_cohorts": len(cohorts),
        "complete_model_forward_passes": len(cohorts) * len(CONDITIONS) * 2,
        "within_gla_replays_per_natural_forward": len(layers),
        "source_trace_definition": (
            "Within each natural GLA forward, replay the complete recurrence with beta=0 "
            "only at the evaluation-closing period. Natural-minus-replay post-norm, "
            "post-output-projection output is recorded at every later token."
        ),
        "causal_ablation": (
            "A separate complete forward sets beta=0 at the evaluation-closing period "
            "in all 48 GLAs simultaneously while preserving the period token's own "
            "output; only causally later recurrent outputs receive the ablation."
        ),
        "preserve_source_output": True,
        "prompt_audit": prompt_audit,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace the exact downstream contribution of the evaluation-period GLA write"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    collect(args.config, args.remapping_plan, args.output_dir, args.max_questions)


if __name__ == "__main__":
    main()
