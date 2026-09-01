from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .collect_cross_model_behavioral_gate import _assert_prompt_pair, _scenario_messages
from .config import ExperimentConfig
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)


# Binding modules replace these values.
MODEL_ID = "ByteDance-Seed/Seed-OSS-36B-Instruct"
MODEL_REVISION = "497f1dca95ebdec98e41d517b9f060ee753c902f"
EXPERIMENT_MODEL_NAME = "Seed-OSS 36B"
CANONICAL_BATCH_SIZE = 4
EXPECTED_LAYER_COUNT = 64
ALLOWED_SERIALIZATIONS = ("hf_template",)
TRUSTED_SCENARIOS = ("incorrect_again_nonremapped", "lost_again_nonremapped")
CONDITIONS = ("game", "neutral")
SCENARIOS = (
    "identity",
    "uncertainty_ablation",
    "uncertainty_steer_negative",
    "uncertainty_steer_positive",
    "random_ablation",
    "random_steer_negative",
    "random_steer_positive",
)
STEERING_MAGNITUDE = 3.0
STEERING_MEAN_RELATIVE_TOLERANCE = 0.03
STEERING_INDIVIDUAL_MIN_FRACTION = 0.5
STEERING_INDIVIDUAL_MAX_FRACTION = 1.5
RANDOM_UNCERTAINTY_SHIFT_MAX_FRACTION = 0.05
LAYERS = 64


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_binding(config: ExperimentConfig) -> None:
    if (config.model_id, config.model_revision) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("Configured model does not match the pinned binding")
    if int(config.batch_size) != CANONICAL_BATCH_SIZE:
        raise ValueError(
            f"Requires canonical batch_size={CANONICAL_BATCH_SIZE}, found {config.batch_size}"
        )
    if config.chat_serialization not in ALLOWED_SERIALIZATIONS:
        raise ValueError("Configured serialization does not match the pinned binding")
    if config.attn_implementation != "sdpa":
        raise ValueError("Requires the validated SDPA path")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires the canonical empty-first-answer prompt")
    if config.feedback_variant != "token_matched_test" or not config.disable_thinking:
        raise ValueError("Requires token-matched feedback with reasoning disabled")


def _split_confirmation(path: Path, qids: list[str]) -> list[str]:
    payload = json.loads(path.read_text())
    if "confirmation_question_ids" in payload:
        confirmation = set(str(value) for value in payload["confirmation_question_ids"])
    elif "question_ids" in payload:
        confirmation = set(qids) - set(str(value) for value in payload["question_ids"])
    else:
        raise ValueError(f"Unrecognized split plan: {path}")
    result = [qid for qid in qids if qid in confirmation]
    if set(result) != confirmation:
        raise RuntimeError("Confirmation split and trajectory questions disagree")
    return result


def _random_direction(direction: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = np.asarray(direction, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    value = rng.standard_normal(direction.shape[0])
    for _ in range(2):
        value -= float(value @ direction) * direction
    value /= max(float(np.linalg.norm(value)), 1e-12)
    if abs(float(value @ direction)) > 1e-6:
        raise RuntimeError("Random control is not orthogonal")
    return value.astype(np.float32)


def _aggregate(logits: Any, variant_ids: list[list[int]]) -> np.ndarray:
    import torch

    return torch.stack(
        [torch.logsumexp(logits[:, group], dim=-1) for group in variant_ids], dim=-1
    ).detach().float().cpu().numpy()


def _repeat_cache(cache: Any, repeats: int) -> Any:
    """Clone and repeat a Hugging Face cache along its batch dimension."""
    import torch

    repeated = copy.deepcopy(cache)
    # Hybrid Qwen caches contain ordinary DynamicLayer objects and recurrent
    # LinearAttentionLayer objects.  Transformers' generic
    # batch_repeat_interleave dispatches a method that the recurrent layer does
    # not implement.  Beam-style reordering is implemented by both layer
    # families and, with repeated monotone indices, is exactly batch repeat.
    reorder = getattr(repeated, "reorder_cache", None)
    layers = getattr(repeated, "layers", None)
    if callable(reorder) and layers is not None:
        batch_size: int | None = None
        for layer in layers:
            keys = getattr(layer, "keys", None)
            if isinstance(keys, torch.Tensor) and keys.numel():
                batch_size = int(keys.shape[0])
                break
            for value in getattr(layer, "conv_states", {}).values():
                if isinstance(value, torch.Tensor):
                    batch_size = int(value.shape[0])
                    break
            if batch_size is not None:
                break
            for value in getattr(layer, "recurrent_states", {}).values():
                if isinstance(value, torch.Tensor):
                    batch_size = int(value.shape[0])
                    break
            if batch_size is not None:
                break
        if batch_size is None:
            raise RuntimeError("Could not infer the hybrid cache batch size")
        beam_indices = torch.arange(batch_size, dtype=torch.long).repeat_interleave(repeats)
        result = reorder(beam_indices)
        return repeated if result is None else result
    method = getattr(repeated, "batch_repeat_interleave", None)
    if callable(method):
        result = method(repeats)
        return repeated if result is None else result
    if isinstance(repeated, tuple):
        return tuple(
            tuple(value.repeat_interleave(repeats, dim=0) for value in layer)
            for layer in repeated
        )
    if isinstance(repeated, list):
        return [
            tuple(value.repeat_interleave(repeats, dim=0) for value in layer)
            for layer in repeated
        ]
    # Fail closed rather than guessing which private cache tensors carry batch.
    tensor_attributes = {
        name: value for name, value in vars(repeated).items()
        if isinstance(value, torch.Tensor)
    }
    raise TypeError(
        f"Cache {type(repeated)} has no batch_repeat_interleave method; "
        f"tensor attributes={list(tensor_attributes)}"
    )


class BatchedDirectionHook:
    """Apply the seven frozen interventions to one cached 2P token step."""

    def __init__(self, layer: Any, direction: np.ndarray, random: np.ndarray):
        import torch

        self.direction = torch.from_numpy(np.asarray(direction, dtype=np.float32))
        self.random = torch.from_numpy(np.asarray(random, dtype=np.float32))
        self.pre: Any | None = None
        self.post: Any | None = None
        self.pre_random: Any | None = None
        self.post_random: Any | None = None
        self.calls = 0
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.shape[1] != 1:
            raise RuntimeError(f"Uncertainty hook requires one token, got {hidden.shape}")
        if hidden.shape[0] % len(SCENARIOS):
            raise RuntimeError("Expanded uncertainty batch does not divide into scenarios")
        self.calls += 1
        direction = self.direction.to(hidden.device, dtype=hidden.dtype)
        random = self.random.to(hidden.device, dtype=hidden.dtype)
        state = hidden[:, 0]
        changed = state.clone()
        pre = (state.float() @ direction.float()).reshape(-1, len(SCENARIOS))
        pre_random = (state.float() @ random.float()).reshape(-1, len(SCENARIOS))
        rows = state.reshape(-1, len(SCENARIOS), state.shape[-1])
        edited = changed.reshape_as(rows)
        d = direction[None]
        r = random[None]
        edited[:, 1] = _quantized_projection_ablation(
            rows[:, 1], direction, self.direction.to(hidden.device)
        )
        edited[:, 2] = rows[:, 2] - STEERING_MAGNITUDE * d
        edited[:, 3] = rows[:, 3] + STEERING_MAGNITUDE * d
        edited[:, 4] = _quantized_two_coordinate_edit(
            rows[:, 4], random, self.random.to(hidden.device), direction,
            self.direction.to(hidden.device),
            -(rows[:, 4].float() @ self.random.to(hidden.device).float()),
        )
        edited[:, 5] = _quantized_two_coordinate_edit(
            rows[:, 5], random, self.random.to(hidden.device), direction,
            self.direction.to(hidden.device), -STEERING_MAGNITUDE,
        )
        edited[:, 6] = _quantized_two_coordinate_edit(
            rows[:, 6], random, self.random.to(hidden.device), direction,
            self.direction.to(hidden.device), STEERING_MAGNITUDE,
        )
        changed = edited.reshape_as(state)
        post = (changed.float() @ direction.float()).reshape(-1, len(SCENARIOS))
        post_random = (changed.float() @ random.float()).reshape(-1, len(SCENARIOS))
        self.pre = pre.detach().cpu()
        self.post = post.detach().cpu()
        self.pre_random = pre_random.detach().cpu()
        self.post_random = post_random.detach().cpu()
        hidden_changed = hidden.clone()
        hidden_changed[:, 0] = changed
        if isinstance(output, tuple):
            return (hidden_changed, *output[1:])
        if isinstance(output, list):
            return [hidden_changed, *output[1:]]
        return hidden_changed

    def close(self) -> None:
        self.handle.remove()


def _quantized_projection_ablation(
    rows: Any, edit_direction: Any, measurement_direction: Any,
    iterations: int = 3,
) -> Any:
    """Remove a coordinate after quantization, retaining only improving steps.

    The stored unit direction is FP32 while the live residual and edit vector
    are BF16.  A single analytic subtraction is therefore not the exact
    projection after the edited state is rounded back to BF16.  A few measured
    correction steps make the operation satisfy its declared live-coordinate
    gate without changing the targeted direction or dose.
    """
    import torch

    measure = measurement_direction.float()
    edit = edit_direction.float()
    denominator = torch.sum(edit * measure)
    if float(torch.abs(denominator)) < 1e-8:
        raise RuntimeError("Quantized ablation direction is degenerate")
    original_residual = rows.float() @ measure
    candidate = (
        rows.float() - original_residual[:, None] * edit[None]
    ).to(rows.dtype)
    residual = candidate.float() @ measure
    for _ in range(iterations):
        proposal = (
            candidate.float()
            - (residual / denominator)[:, None] * edit[None]
        ).to(rows.dtype)
        proposal_residual = proposal.float() @ measure
        improved = torch.abs(proposal_residual) < torch.abs(residual)
        candidate = torch.where(improved[:, None], proposal, candidate)
        residual = torch.where(improved, proposal_residual, residual)
    return candidate


def _quantized_two_coordinate_edit(
    original: Any,
    primary_edit_direction: Any,
    primary_measurement_direction: Any,
    preserved_edit_direction: Any,
    preserved_measurement_direction: Any,
    target_primary_delta: Any,
    iterations: int = 8,
) -> Any:
    """Meet one edit dose while preserving an orthogonal coordinate in BF16."""
    import torch

    primary_measure = primary_measurement_direction.float()
    preserved_measure = preserved_measurement_direction.float()
    edits = torch.stack(
        [primary_edit_direction.float(), preserved_edit_direction.float()]
    )
    measures = torch.stack([primary_measure, preserved_measure])
    response = measures @ edits.T
    if float(torch.abs(torch.linalg.det(response))) < 1e-8:
        raise RuntimeError("Quantized two-coordinate edit is degenerate")
    batch = original.shape[0]
    target_primary = torch.as_tensor(
        target_primary_delta, device=original.device, dtype=torch.float32
    ).expand(batch)
    target = torch.stack([target_primary, torch.zeros_like(target_primary)], dim=-1)
    original_float = original.float()
    result = (
        original_float + target_primary[:, None] * edits[0][None]
    ).to(original.dtype)

    def errors(value: Any) -> Any:
        return (value.float() - original_float) @ measures.T - target

    residual = errors(result)
    scales = torch.stack(
        [torch.clamp(torch.abs(target_primary), min=STEERING_MAGNITUDE),
         torch.full_like(target_primary, STEERING_MAGNITUDE)],
        dim=-1,
    )
    score = torch.max(torch.abs(residual) / scales, dim=-1).values
    for _ in range(iterations):
        coefficients = torch.linalg.solve(response, residual.T).T
        # A mathematically exact small correction may round to no BF16 change
        # at high-scale coordinates.  Try a short deterministic line search
        # so the correction can cross the local quantization threshold while
        # accepting only a better joint dose/preservation error.
        for multiplier in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            proposal = (
                result.float() - multiplier * (coefficients @ edits)
            ).to(original.dtype)
            proposal_residual = errors(proposal)
            proposal_score = torch.max(
                torch.abs(proposal_residual) / scales, dim=-1
            ).values
            improved = proposal_score < score
            result = torch.where(improved[:, None], proposal, result)
            residual = torch.where(improved[:, None], proposal_residual, residual)
            score = torch.where(improved, proposal_score, score)
    return result


def _initialize(path: Path, qids: list[str], ranks: np.ndarray, first_logits: np.ndarray):
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing intervention checkpoint has different questions")
        if arrays["scenarios"].astype(str).tolist() != list(SCENARIOS):
            raise ValueError("Existing intervention checkpoint has different scenarios")
        # Older one-cohort benchmarks predate deterministic padding.  They are
        # still resumable because they never exercised a partial cohort.
        arrays.setdefault(
            "duplicate_padding_max_abs_error", np.asarray(0.0, dtype=np.float32)
        )
        return arrays
    n = len(qids)
    order = np.asarray(ranks, dtype=np.int64)
    top_two = np.take_along_axis(first_logits, order[:, :2], axis=1)
    return {
        "question_ids": np.asarray(qids),
        "conditions": np.asarray(CONDITIONS),
        "scenarios": np.asarray(SCENARIOS),
        "rank_order": order,
        "first_logits": np.asarray(first_logits, dtype=np.float32),
        "first_top_two_margin": (top_two[:, 0] - top_two[:, 1]).astype(np.float32),
        "completed": np.zeros(n, dtype=bool),
        "logits": np.full((2, LAYERS, len(SCENARIOS), n, 4), np.nan, dtype=np.float32),
        "pre_projection": np.full((2, LAYERS, len(SCENARIOS), n), np.nan, dtype=np.float32),
        "post_projection": np.full((2, LAYERS, len(SCENARIOS), n), np.nan, dtype=np.float32),
        "pre_random_projection": np.full(
            (2, LAYERS, len(SCENARIOS), n), np.nan, dtype=np.float32
        ),
        "post_random_projection": np.full(
            (2, LAYERS, len(SCENARIOS), n), np.nan, dtype=np.float32
        ),
        "duplicate_padding_max_abs_error": np.asarray(0.0, dtype=np.float32),
    }


def _direction_lens(
    parts: Any, tokenizer: Any, directions: np.ndarray,
    variant_ids: list[list[int]],
) -> dict[str, Any]:
    import torch

    device = parts.final_norm.weight.device
    head_weight = parts.output_head.weight
    answer_rows = torch.stack(
        [head_weight[group].float().mean(dim=0) for group in variant_ids]
    )
    answer_mean = answer_rows.mean(dim=0)
    answer_mean /= torch.clamp(torch.linalg.vector_norm(answer_mean), min=1e-12)
    centered_answer_rows = answer_rows - answer_rows.mean(dim=0, keepdim=True)
    _u, singular, vh = torch.linalg.svd(centered_answer_rows, full_matrices=False)
    keep = singular > torch.clamp(singular.max() * 1e-6, min=1e-12)
    answer_contrast_basis = vh[keep]
    result: dict[str, Any] = {
        "_metadata": {
            "format_version": 2,
            "answer_subspace": (
                "Euclidean span of the four mean answer-token unembedding rows "
                "after removing their shared mean"
            ),
            "answer_subspace_rank": int(answer_contrast_basis.shape[0]),
        }
    }
    with torch.inference_mode():
        for layer in range(LAYERS):
            value = torch.from_numpy(directions[layer]).to(
                device=device, dtype=parts.final_norm.weight.dtype
            )[None]
            logits = parts.output_head(parts.final_norm(value))[0].float()
            top_count = min(12, int(logits.numel()))
            positive = torch.topk(logits, top_count).indices.tolist()
            negative = torch.topk(-logits, top_count).indices.tolist()
            direction_float = torch.from_numpy(directions[layer]).to(
                device=device, dtype=torch.float32
            )
            contrast_coordinates = answer_contrast_basis @ direction_float
            result[str(layer + 1)] = {
                "positive_tokens": [tokenizer.decode([int(index)]) for index in positive],
                "negative_tokens": [tokenizer.decode([int(index)]) for index in negative],
                "answer_variant_scores": [
                    float(torch.logsumexp(logits[group], dim=0)) for group in variant_ids
                ],
                "centered_answer_subspace_fraction": float(
                    torch.sum(contrast_coordinates ** 2)
                ),
                "shared_answer_mean_cosine": float(answer_mean @ direction_float),
            }
    return result


def _run_dataset(
    spec: dict[str, Any], dataset_index: int, model: Any, processor: Any, parts: Any,
    directions: np.ndarray, max_cohorts: int | None, output_tag: str | None,
) -> None:
    import torch

    config = ExperimentConfig.load(Path(spec["config"]))
    _assert_binding(config)
    state_dir = Path(spec["output"])
    with np.load(state_dir / "results.npz", allow_pickle=False) as states:
        all_qids = states["question_ids"].astype(str).tolist()
        all_ranks = np.asarray(states["rank_order"], dtype=np.int64)
        all_first_logits = np.asarray(states["first_logits"], dtype=np.float32)
    confirmation = _split_confirmation(Path(spec["split_plan"]), all_qids)
    if max_cohorts is not None:
        confirmation = confirmation[: int(max_cohorts) * config.batch_size]
    index = {qid: row for row, qid in enumerate(all_qids)}
    selected = [index[qid] for qid in confirmation]
    ranks = all_ranks[selected]
    first_logits = all_first_logits[selected]

    with np.load(Path(spec["trusted_trajectory"]), allow_pickle=False) as trusted:
        trusted_qids = trusted["question_ids"].astype(str).tolist()
        trusted_index = {qid: row for row, qid in enumerate(trusted_qids)}
        trusted_natural = np.asarray(
            trusted["direct_logits"][:, [trusted_index[qid] for qid in confirmation]],
            dtype=np.float32,
        )
    manifest = json.loads(Path(config.manifest_path).read_text())["questions"]
    questions = {str(row["id"]): row for row in manifest}
    output_name = "intervention" if output_tag is None else f"intervention_{output_tag}"
    output = state_dir.parent / output_name
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.npz"
    arrays = _initialize(result_path, confirmation, ranks, first_logits)
    qid_index = {qid: row for row, qid in enumerate(confirmation)}

    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = [
        sorted({token_id for _text, token_id in resolved[letter]}) for letter in LETTERS
    ]
    random = np.stack(
        [_random_direction(directions[layer], 20260902 + dataset_index * 10007 + layer * 101)
         for layer in range(LAYERS)]
    )
    if float(np.max(np.abs(np.sum(random * directions, axis=-1)))) > 1e-5:
        raise RuntimeError("Random direction geometry failed")
    if not (output / "direction_lens.json").exists():
        _atomic_json(
            output / "direction_lens.json",
            _direction_lens(parts, tokenizer, directions, variant_ids),
        )

    input_device = model_input_device(parts)
    durations: list[float] = []
    audit: dict[str, Any] = {"dataset": spec["name"], "conditions": {}}
    started = time.monotonic()
    for start in range(0, len(confirmation), config.batch_size):
        cohort = confirmation[start : start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if np.all(arrays["completed"][indices]):
            continue
        # Preserve the numerically validated batch-4 execution regime even
        # when a frozen confirmation split is not divisible by four.  Repeated
        # rows are discarded and are also an exact within-batch identity check.
        execution_cohort = list(cohort)
        while len(execution_cohort) < config.batch_size:
            execution_cohort.append(cohort[0])
        real_count = len(cohort)
        cohort_started = time.monotonic()
        representative_prompts: list[str] = []
        for condition_index, scenario_name in enumerate(TRUSTED_SCENARIOS):
            prompts = []
            for qid in execution_cohort:
                messages, remapping = _scenario_messages(
                    scenario_name, questions[qid], {letter: letter for letter in LETTERS}
                )
                if remapping is not None:
                    raise RuntimeError("Canonical intervention prompt unexpectedly remapped")
                prompts.append(
                    render_chat(
                        processor,
                        messages,
                        config.disable_thinking,
                        config.chat_serialization,
                        config.chat_template_kwargs,
                    )
                )
            representative_prompts.append(prompts[0])
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            if any(position != input_ids.shape[1] - 1 for position in last_indices):
                raise RuntimeError("2P decision is not the final physical token")
            prefix_kwargs = {
                "input_ids": input_ids[:, :-1].to(input_device),
                "attention_mask": attention_mask[:, :-1].to(input_device),
                "use_cache": True,
                "return_dict": True,
            }
            with torch.inference_mode():
                try:
                    prefix = model(**prefix_kwargs, logits_to_keep=1)
                except TypeError:
                    prefix = model(**prefix_kwargs)
            source_cache = prefix.past_key_values
            repeats = len(SCENARIOS)
            expanded_ids = input_ids[:, -1:].repeat_interleave(repeats, dim=0).to(input_device)
            expanded_mask = attention_mask.repeat_interleave(repeats, dim=0).to(input_device)

            for layer in range(LAYERS):
                cache = _repeat_cache(source_cache, repeats)
                hook = BatchedDirectionHook(parts.layers[layer], directions[layer], random[layer])
                try:
                    kwargs = {
                        "input_ids": expanded_ids,
                        "attention_mask": expanded_mask,
                        "past_key_values": cache,
                        "use_cache": True,
                        "return_dict": True,
                    }
                    with torch.inference_mode():
                        try:
                            result = model(**kwargs, logits_to_keep=1)
                        except TypeError:
                            result = model(**kwargs)
                finally:
                    hook.close()
                if (
                    hook.calls != 1
                    or hook.pre is None
                    or hook.post is None
                    or hook.pre_random is None
                    or hook.post_random is None
                ):
                    raise RuntimeError(f"Layer {layer + 1}: intervention hook did not fire exactly once")
                live = result.logits[:, -1].detach().float()
                values = _aggregate(live, variant_ids).reshape(
                    len(execution_cohort), len(SCENARIOS), 4
                )
                pre = hook.pre.numpy()
                post = hook.post.numpy()
                pre_random = hook.pre_random.numpy()
                post_random = hook.post_random.numpy()
                arrays["logits"][condition_index, layer][:, indices] = (
                    values[:real_count].transpose(1, 0, 2)
                )
                arrays["pre_projection"][condition_index, layer][:, indices] = (
                    pre[:real_count].T
                )
                arrays["post_projection"][condition_index, layer][:, indices] = (
                    post[:real_count].T
                )
                arrays["pre_random_projection"][condition_index, layer][:, indices] = (
                    pre_random[:real_count].T
                )
                arrays["post_random_projection"][condition_index, layer][:, indices] = (
                    post_random[:real_count].T
                )
                if real_count < config.batch_size:
                    duplicate_error = max(
                        float(np.max(np.abs(values[real_count:] - values[:1]))),
                        float(np.max(np.abs(pre[real_count:] - pre[:1]))),
                        float(np.max(np.abs(post[real_count:] - post[:1]))),
                        float(
                            np.max(
                                np.abs(pre_random[real_count:] - pre_random[:1])
                            )
                        ),
                        float(
                            np.max(
                                np.abs(post_random[real_count:] - post_random[:1])
                            )
                        ),
                    )
                    arrays["duplicate_padding_max_abs_error"] = np.asarray(
                        max(
                            float(arrays["duplicate_padding_max_abs_error"]),
                            duplicate_error,
                        ),
                        dtype=np.float32,
                    )
                # `kwargs` retains the expanded cache even after `cache` and
                # `result` are deleted.  Seed's full-attention cache is large
                # enough that carrying the prior layer's cache into the next
                # `_repeat_cache` allocation eventually fragments/exhausts a
                # 40 GB device during long runs.  Release every GPU-owning
                # reference before constructing the next layer's cache.
                del (
                    cache, result, live, values, pre, post, pre_random,
                    post_random, kwargs, hook,
                )
            del (
                source_cache,
                prefix,
                prefix_kwargs,
                expanded_ids,
                expanded_mask,
                input_ids,
                attention_mask,
                last_indices,
            )
            gc.collect()
            torch.cuda.empty_cache()

            if not audit["conditions"].get(CONDITIONS[condition_index]):
                audit["conditions"][CONDITIONS[condition_index]] = {
                    "question_id": cohort[0],
                    "rendered_prompt": prompts[0],
                    "prompt_hash": hashlib.sha256(prompts[0].encode()).hexdigest(),
                }
        _assert_prompt_pair(representative_prompts[0], representative_prompts[1])
        arrays["completed"][indices] = True
        _atomic_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        elapsed = time.monotonic() - started
        complete = int(arrays["completed"].sum())
        remaining = len(confirmation) - complete
        eta = elapsed / max(complete, 1) * remaining
        print(
            f"{EXPERIMENT_MODEL_NAME} {spec['name']} uncertainty intervention: "
            f"{complete}/{len(confirmation)} cohort_seconds={duration:.3f} eta_minutes={eta / 60:.2f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    completed = arrays["completed"]
    if not completed.all():
        print("Intervention benchmark stopped before completion", flush=True)
        return
    identity = SCENARIOS.index("identity")
    identity_error = float(
        np.max(
            np.abs(
                arrays["logits"][:, :, identity]
                - trusted_natural[:, None]
            )
        )
    )
    identity_reference = arrays["logits"][:, 0, identity]
    identity_centered = identity_reference - identity_reference.mean(
        axis=-1, keepdims=True
    )
    trusted_centered = trusted_natural - trusted_natural.mean(axis=-1, keepdims=True)
    identity_centered_error = float(
        np.max(np.abs(identity_centered - trusted_centered))
    )
    identity_argmax_agreement = float(
        np.mean(np.argmax(identity_reference, axis=-1) == np.argmax(trusted_natural, axis=-1))
    )
    old_winner = ranks[:, 0][None]
    identity_old_winner_choice = np.argmax(identity_reference, axis=-1) == old_winner
    trusted_old_winner_choice = np.argmax(trusted_natural, axis=-1) == old_winner
    identity_old_winner_choice_agreement = float(
        np.mean(identity_old_winner_choice == trusted_old_winner_choice)
    )
    identity_layer_spread = float(
        np.max(
            np.abs(
                arrays["logits"][:, :, identity]
                - arrays["logits"][:, :1, identity]
            )
        )
    )
    ablation_post = float(
        np.max(np.abs(arrays["post_projection"][:, :, SCENARIOS.index("uncertainty_ablation")]))
    )
    ablation_pre_scale = float(
        np.max(np.abs(arrays["pre_projection"][:, :, SCENARIOS.index("uncertainty_ablation")]))
    )
    ablation_relative_residual = ablation_post / max(ablation_pre_scale, 1e-12)
    negative_shift = arrays["post_projection"][:, :, SCENARIOS.index("uncertainty_steer_negative")] - arrays["pre_projection"][:, :, SCENARIOS.index("uncertainty_steer_negative")]
    positive_shift = arrays["post_projection"][:, :, SCENARIOS.index("uncertainty_steer_positive")] - arrays["pre_projection"][:, :, SCENARIOS.index("uncertainty_steer_positive")]
    random_projection_shifts = []
    for scenario in ("random_ablation", "random_steer_negative", "random_steer_positive"):
        scenario_index = SCENARIOS.index(scenario)
        random_projection_shifts.append(
            arrays["post_projection"][:, :, scenario_index]
            - arrays["pre_projection"][:, :, scenario_index]
        )
    random_projection_shift_max_abs = float(
        np.max(np.abs(np.stack(random_projection_shifts)))
    )
    random_ablation_index = SCENARIOS.index("random_ablation")
    random_ablation_post = float(
        np.max(np.abs(arrays["post_random_projection"][:, :, random_ablation_index]))
    )
    random_ablation_pre_scale = float(
        np.max(np.abs(arrays["pre_random_projection"][:, :, random_ablation_index]))
    )
    random_ablation_relative_residual = (
        random_ablation_post / max(random_ablation_pre_scale, 1e-12)
    )
    random_negative_index = SCENARIOS.index("random_steer_negative")
    random_positive_index = SCENARIOS.index("random_steer_positive")
    random_negative_shift = (
        arrays["post_random_projection"][:, :, random_negative_index]
        - arrays["pre_random_projection"][:, :, random_negative_index]
    )
    random_positive_shift = (
        arrays["post_random_projection"][:, :, random_positive_index]
        - arrays["pre_random_projection"][:, :, random_positive_index]
    )
    if not all(
        np.isfinite(value).all()
        for key, value in arrays.items()
        if key in {
            "logits", "pre_projection", "post_projection",
            "pre_random_projection", "post_random_projection",
        }
    ):
        raise RuntimeError("Intervention outputs are non-finite")
    if identity_layer_spread != 0.0:
        raise RuntimeError(
            f"Cached identity varies by intervention layer: {identity_layer_spread}"
        )
    duplicate_padding_error = float(arrays["duplicate_padding_max_abs_error"])
    if duplicate_padding_error != 0.0:
        raise RuntimeError(
            "Duplicated padding rows did not reproduce exactly: "
            f"{duplicate_padding_error}"
        )
    # Qwen/Seed hidden states are BF16.  An exactly orthogonal float32 edit is
    # rounded when re-entering the model, so the live post-hook dot product is
    # required to be less than 0.5% of its pre-edit scale rather than bit-zero.
    if ablation_relative_residual > 0.005:
        raise RuntimeError(
            "Uncertainty ablation left excessive projection: "
            f"absolute={ablation_post}, relative={ablation_relative_residual}"
        )
    if random_ablation_relative_residual > 0.005:
        raise RuntimeError(
            "Random-control ablation left excessive random projection: "
            f"absolute={random_ablation_post}, "
            f"relative={random_ablation_relative_residual}"
        )
    # BF16 can quantize a requested three-unit edit by roughly one unit on an
    # individual high-scale Seed state.  Validate the *measured live edit*:
    # its mean must be within 3% of the requested dose and every row must move
    # in the requested direction by 0.5x--1.5x that dose.  This is stricter
    # and more scientifically relevant than pretending the FP32 target is
    # exactly representable in BF16.
    negative_magnitude = -negative_shift
    positive_magnitude = positive_shift
    mean_tolerance = STEERING_MAGNITUDE * STEERING_MEAN_RELATIVE_TOLERANCE
    if (
        abs(float(np.mean(negative_shift)) + STEERING_MAGNITUDE) > mean_tolerance
        or abs(float(np.mean(positive_shift)) - STEERING_MAGNITUDE) > mean_tolerance
        or float(np.min(negative_magnitude))
        < STEERING_MAGNITUDE * STEERING_INDIVIDUAL_MIN_FRACTION
        or float(np.max(negative_magnitude))
        > STEERING_MAGNITUDE * STEERING_INDIVIDUAL_MAX_FRACTION
        or float(np.min(positive_magnitude))
        < STEERING_MAGNITUDE * STEERING_INDIVIDUAL_MIN_FRACTION
        or float(np.max(positive_magnitude))
        > STEERING_MAGNITUDE * STEERING_INDIVIDUAL_MAX_FRACTION
    ):
        raise RuntimeError("Steering dose validation failed")
    random_negative_magnitude = -random_negative_shift
    random_positive_magnitude = random_positive_shift
    if (
        abs(float(np.mean(random_negative_shift)) + STEERING_MAGNITUDE)
        > mean_tolerance
        or abs(float(np.mean(random_positive_shift)) - STEERING_MAGNITUDE)
        > mean_tolerance
        or float(np.min(random_negative_magnitude))
        < STEERING_MAGNITUDE * STEERING_INDIVIDUAL_MIN_FRACTION
        or float(np.max(random_negative_magnitude))
        > STEERING_MAGNITUDE * STEERING_INDIVIDUAL_MAX_FRACTION
        or float(np.min(random_positive_magnitude))
        < STEERING_MAGNITUDE * STEERING_INDIVIDUAL_MIN_FRACTION
        or float(np.max(random_positive_magnitude))
        > STEERING_MAGNITUDE * STEERING_INDIVIDUAL_MAX_FRACTION
    ):
        raise RuntimeError("Random-control steering dose validation failed")
    if (
        random_projection_shift_max_abs
        > STEERING_MAGNITUDE * RANDOM_UNCERTAINTY_SHIFT_MAX_FRACTION
    ):
        raise RuntimeError(
            "Orthogonal random controls disturbed the uncertainty coordinate: "
            f"{random_projection_shift_max_abs}"
        )
    _atomic_json(output / "prompt_audit.json", audit)
    _atomic_json(
        output / "run_metadata.json",
        {
            "experiment": "2P decision-position MCQ uncertainty ablation and steering",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset": spec["name"],
            "questions": len(confirmation),
            "canonical_batch_size": config.batch_size,
            "padded_duplicate_rows": (
                -len(confirmation)
            ) % config.batch_size,
            "conditions": list(CONDITIONS),
            "scenarios": list(SCENARIOS),
            "layers": list(range(1, LAYERS + 1)),
            "steering_magnitude": STEERING_MAGNITUDE,
            "direction_source": "1P discovery high-minus-low entropy quartile mean difference",
            "complete_model_calls_per_cohort": 2 + 2 * LAYERS,
            "expanded_one_token_trajectories_per_cohort": 2 * LAYERS * len(SCENARIOS) * config.batch_size,
            "identity_trusted_max_abs_error": identity_error,
            "identity_trusted_centered_max_abs_error": identity_centered_error,
            "identity_trusted_argmax_agreement": identity_argmax_agreement,
            "identity_trusted_old_winner_choice_agreement": (
                identity_old_winner_choice_agreement
            ),
            "identity_layer_spread_max_abs_error": identity_layer_spread,
            "duplicate_padding_max_abs_error": duplicate_padding_error,
            "ablation_post_projection_max_abs": ablation_post,
            "ablation_post_projection_relative_to_pre_max": ablation_relative_residual,
            "negative_steering_mean_shift": float(np.mean(negative_shift)),
            "positive_steering_mean_shift": float(np.mean(positive_shift)),
            "negative_steering_min_magnitude": float(np.min(negative_magnitude)),
            "negative_steering_max_magnitude": float(np.max(negative_magnitude)),
            "positive_steering_min_magnitude": float(np.min(positive_magnitude)),
            "positive_steering_max_magnitude": float(np.max(positive_magnitude)),
            "negative_steering_max_abs_dose_error": float(
                np.max(np.abs(negative_shift + STEERING_MAGNITUDE))
            ),
            "positive_steering_max_abs_dose_error": float(
                np.max(np.abs(positive_shift - STEERING_MAGNITUDE))
            ),
            "orthogonal_random_uncertainty_projection_shift_max_abs": random_projection_shift_max_abs,
            "random_ablation_post_projection_relative_to_pre_max": (
                random_ablation_relative_residual
            ),
            "random_negative_steering_mean_shift": float(
                np.mean(random_negative_shift)
            ),
            "random_positive_steering_mean_shift": float(
                np.mean(random_positive_shift)
            ),
            "all_outputs_finite": True,
            "elapsed_seconds_after_model_load": time.monotonic() - started,
            "cohort_seconds": durations,
            "directions_sha256": _hash(Path(spec["directions_path"])),
            "software": {"python": sys.version, "torch": torch.__version__},
            "platform": platform.platform(),
        },
    )


def run(
    specs_path: Path, max_cohorts: int | None, only_dataset: str | None,
    output_tag: str | None, export_lens_only: bool = False,
) -> None:
    payload = json.loads(specs_path.read_text())
    specs = payload["datasets"]
    direction_path = Path(payload["direction_output"]) / "directions.npz"
    with np.load(direction_path, allow_pickle=False) as loaded:
        slugs = loaded["dataset_slugs"].astype(str).tolist()
        directions = np.asarray(loaded["entropy_mean_diff"], dtype=np.float32)
    if slugs != [spec["slug"] for spec in specs] or directions.shape[1] != LAYERS:
        raise ValueError("Direction artifact does not match the frozen specs")
    for spec in specs:
        spec["directions_path"] = str(direction_path)
    selected = [
        (index, spec) for index, spec in enumerate(specs)
        if only_dataset is None or spec["slug"] == only_dataset
    ]
    if not selected:
        raise ValueError(f"Unknown dataset {only_dataset!r}")
    configs = [ExperimentConfig.load(Path(spec["config"])) for _index, spec in selected]
    for config in configs:
        _assert_binding(config)
    load_started = time.monotonic()
    model, processor, parts = load_model_and_processor(configs[0])
    if len(parts.layers) != EXPECTED_LAYER_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_LAYER_COUNT} layers, found {len(parts.layers)}")
    print(f"MODEL_LOADED seconds={time.monotonic() - load_started:.3f}", flush=True)
    if export_lens_only:
        tokenizer = get_tokenizer(processor)
        resolved = resolve_answer_tokens(tokenizer, configs[0].answer_variants)
        variant_ids = [
            sorted({token_id for _text, token_id in resolved[letter]})
            for letter in LETTERS
        ]
        for dataset_index, spec in selected:
            output_name = (
                "intervention" if output_tag is None
                else f"intervention_{output_tag}"
            )
            output = Path(spec["output"]).parent / output_name
            output.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                output / "direction_lens.json",
                _direction_lens(
                    parts, tokenizer, directions[dataset_index], variant_ids
                ),
            )
        print("DIRECTION_LENS_EXPORT_COMPLETE", flush=True)
        return
    for dataset_index, spec in selected:
        _run_dataset(
            spec, dataset_index, model, processor, parts,
            directions[dataset_index], max_cohorts, output_tag,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    parser.add_argument("--only-dataset", choices=("simplemc", "triviamc"))
    parser.add_argument("--output-tag")
    parser.add_argument("--export-lens-only", action="store_true")
    args = parser.parse_args()
    run(
        args.specs, args.max_cohorts, args.only_dataset, args.output_tag,
        args.export_lens_only,
    )


if __name__ == "__main__":
    main()
