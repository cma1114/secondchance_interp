from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .attention_spans import attention_span_indices
from .config import ExperimentConfig
from .data import decision_letter, load_activation_dataset
from .io import read_metadata, shard_path
from .modeling import (
    QWEN_EMPTY_THINKING,
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import (
    ANSWER_ONLY_INSTRUCTION,
    CHOICE_CUE,
    GAME_FEEDBACK,
    NEUTRAL_FEEDBACK,
    build_messages,
    load_trials,
)


CONDITIONS = ("baseline", "incorrect", "neutral")
POSITION_CONDITIONS = ("incorrect", "neutral")
ANCHORS = (
    "first_question_end",
    "first_answer_decision",
    "historical_answer_end",
    "feedback_subject_end",
    "condition_keyword_end",
    "user_different",
    "action_keyword_end",
    "feedback_end",
    "instruction_letter",
    "instruction_choice",
    "instruction_end",
    "repeated_choice",
    "second_user_end",
    "decision",
)
STRATEGY_FAMILIES = {
    "switch": ("switch", "change", "different", "alternative", "other"),
    "repeat": ("repeat", "same", "again", "keep", "continue"),
    "incorrect": ("wrong", "incorrect", "error", "mistake"),
    "lost": ("lost", "transmission", "missing"),
}


class PositionCollector:
    def __init__(self, layers: Any, positions: list[int]):
        self.positions = positions
        self.values: list[Any] = [None] * len(layers)
        self.handles = [layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(layers)]

    def _hook(self, index: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = output[0] if isinstance(output, (tuple, list)) else output
            indices = torch.as_tensor(self.positions, device=hidden.device)
            self.values[index] = hidden[0, indices].detach().to("cpu", dtype=torch.float16)
        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def stacked(self):
        import torch

        if any(value is None for value in self.values):
            raise RuntimeError("Failed to collect one or more residual layers")
        return torch.stack(self.values, dim=0)


def _answer_labels(data, condition: str) -> np.ndarray:
    labels = []
    for qid in data.question_ids:
        answer = decision_letter(data.metadata[(qid, condition)])
        if answer not in "ABCD":
            raise ValueError(f"Non-A-D output for {condition}/{qid}: {answer!r}")
        labels.append("ABCD".index(answer))
    return np.asarray(labels, dtype=np.int64)


def _stratified_sample(data, count: int, seed: int) -> list[str]:
    baseline = _answer_labels(data, "baseline")
    game = _answer_labels(data, "incorrect")
    strata = baseline * 2 + (game != baseline)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    unique = sorted(np.unique(strata).tolist())
    quotas = {value: count // len(unique) for value in unique}
    for value in unique[: count % len(unique)]:
        quotas[value] += 1
    for value in unique:
        ids = np.flatnonzero(strata == value)
        rng.shuffle(ids)
        chosen.extend(ids[: min(quotas[value], len(ids))].tolist())
    if len(chosen) < count:
        remaining = np.setdiff1d(np.arange(len(strata)), np.asarray(chosen, dtype=int))
        rng.shuffle(remaining)
        chosen.extend(remaining[: count - len(chosen)].tolist())
    return [data.question_ids[index] for index in sorted(chosen[:count])]


def _load_cached_residuals(root: str | Path, data) -> np.ndarray:
    first = shard_path(root, CONDITIONS[0], data.question_ids[0])
    with np.load(first, allow_pickle=False) as shard:
        shape = shard["residuals"].shape
    if shape[0] != 65:
        raise ValueError(f"Expected embedding plus 64 post-block residuals, got {shape}")
    values = np.empty((len(CONDITIONS), len(data.question_ids), 64, shape[-1]), dtype=np.float16)
    for ci, condition in enumerate(CONDITIONS):
        for qi, qid in enumerate(data.question_ids):
            with np.load(shard_path(root, condition, qid), allow_pickle=False) as shard:
                values[ci, qi] = shard["residuals"][1:]
    return values


def _resolve_selected_tokens(tokenizer, config: ExperimentConfig):
    answer_tokens = resolve_answer_tokens(tokenizer, config.answer_variants)
    layout: list[dict[str, Any]] = []
    ids: list[int] = []
    seen: set[int] = set()
    for letter in "ABCD":
        for text, token_id in answer_tokens[letter]:
            if token_id not in seen:
                layout.append({"family": f"answer_{letter}", "text": text, "token_id": token_id})
                ids.append(token_id)
                seen.add(token_id)
    for family, words in STRATEGY_FAMILIES.items():
        for word in words:
            for text in (word, " " + word, word.capitalize(), " " + word.capitalize()):
                encoded = tokenizer.encode(text, add_special_tokens=False)
                if len(encoded) == 1 and int(encoded[0]) not in seen:
                    token_id = int(encoded[0])
                    layout.append({"family": family, "text": text, "token_id": token_id})
                    ids.append(token_id)
                    seen.add(token_id)
    return ids, layout, answer_tokens


def _token_offsets(tokenizer, prompt: str) -> list[tuple[int, int]]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    return [(int(start), int(end)) for start, end in encoded["offset_mapping"]]


def _scoped_token(
    prompt: str,
    offsets: list[tuple[int, int]],
    scope: str,
    needle: str,
    *,
    last: bool = False,
    scope_last: bool = False,
) -> int:
    # Chat templates commonly strip trailing newlines from message content.
    rendered_scope = scope.rstrip()
    scope_start = prompt.rfind(rendered_scope) if scope_last else prompt.find(rendered_scope)
    if scope_start < 0:
        raise RuntimeError(f"Could not locate prompt scope {scope!r}")
    relative = (
        rendered_scope.lower().rfind(needle.lower())
        if last else rendered_scope.lower().find(needle.lower())
    )
    if relative < 0:
        raise RuntimeError(f"Could not locate {needle!r} inside prompt scope {scope!r}")
    start = scope_start + relative
    end = start + len(needle)
    matches = [index for index, (left, right) in enumerate(offsets) if right > start and left < end]
    if not matches:
        raise RuntimeError(f"No token overlaps {needle!r} inside prompt scope")
    return matches[-1]


def _scope_end_token(
    prompt: str,
    offsets: list[tuple[int, int]],
    scope: str,
    *,
    scope_last: bool = False,
) -> int:
    rendered_scope = scope.rstrip()
    scope_start = prompt.rfind(rendered_scope) if scope_last else prompt.find(rendered_scope)
    if scope_start < 0:
        raise RuntimeError(f"Could not locate prompt scope {scope!r}")
    end = scope_start + len(rendered_scope)
    matches = [index for index, (left, right) in enumerate(offsets) if right > end - 1 and left < end]
    if not matches:
        raise RuntimeError(f"No token overlaps the end of prompt scope {scope!r}")
    return matches[-1]


def _anchor_positions(
    tokenizer,
    prompt: str,
    condition: str,
    spans: dict[str, list[int]],
    system_content: str,
    second_user_content: str | None = None,
) -> list[int | None]:
    offsets = _token_offsets(tokenizer, prompt)
    feedback = GAME_FEEDBACK if condition == "incorrect" else NEUTRAL_FEEDBACK
    redacted = spans["redacted_answer"]
    if redacted:
        first_answer_decision = redacted[0] - 1
        historical_answer_end = redacted[-1]
    else:
        assistant_header = "<|im_start|>assistant\n"
        assistant_start = prompt.find(assistant_header)
        if assistant_start < 0:
            raise RuntimeError("Could not locate the first assistant header")
        scaffold_end = (
            assistant_start + len(assistant_header) + len(QWEN_EMPTY_THINKING)
        )
        candidates = [
            index for index, (left, right) in enumerate(offsets)
            if right > left and right <= scaffold_end
        ]
        if not candidates:
            raise RuntimeError("Could not locate the empty historical assistant scaffold")
        first_answer_decision = candidates[-1]
        historical_answer_end = candidates[-1]

    mapping = {
        "first_question_end": spans["first_question"][-1],
        "first_answer_decision": first_answer_decision,
        "historical_answer_end": historical_answer_end,
        "feedback_subject_end": _scoped_token(
            prompt,
            offsets,
            feedback,
            "answer" if condition == "incorrect" else "response",
        ),
        "condition_keyword_end": _scoped_token(
            prompt,
            offsets,
            feedback,
            "incorrect" if condition == "incorrect" else "lost",
        ),
        "user_different": None,
        "action_keyword_end": _scoped_token(
            prompt,
            offsets,
            feedback,
            "answer" if condition == "incorrect" else "again",
            last=condition == "incorrect",
        ),
        "feedback_end": _scope_end_token(prompt, offsets, feedback),
        "instruction_letter": _scoped_token(
            prompt,
            offsets,
            ANSWER_ONLY_INSTRUCTION,
            "letter",
            scope_last=True,
        ),
        "instruction_choice": _scoped_token(
            prompt,
            offsets,
            ANSWER_ONLY_INSTRUCTION,
            "choice",
            scope_last=True,
        ),
        "instruction_end": _scope_end_token(
            prompt,
            offsets,
            ANSWER_ONLY_INSTRUCTION,
            scope_last=True,
        ),
        "repeated_choice": _scoped_token(
            prompt,
            offsets,
            CHOICE_CUE,
            "choice",
            scope_last=True,
        ),
        "second_user_end": _scope_end_token(
            prompt,
            offsets,
            second_user_content or feedback,
        ),
        "decision": spans["query_self"][-1],
    }
    if condition == "incorrect":
        mapping["user_different"] = _scoped_token(
            prompt, offsets, GAME_FEEDBACK, "different"
        )
    return [mapping[name] for name in ANCHORS]


def _collect_position_residuals(
    model,
    processor,
    parts,
    config: ExperimentConfig,
    trials_by_id,
    question_ids: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    tokenizer = get_tokenizer(processor)
    values = np.zeros(
        (len(POSITION_CONDITIONS), len(question_ids), len(parts.layers), len(ANCHORS), parts.embedding.weight.shape[-1]),
        dtype=np.float16,
    )
    availability = np.zeros((len(POSITION_CONDITIONS), len(ANCHORS)), dtype=bool)
    audit: dict[str, Any] = {
        "anchors": list(ANCHORS),
        "conditions": list(POSITION_CONDITIONS),
        "availability": {},
        "trials": {},
    }
    for ci, condition in enumerate(POSITION_CONDITIONS):
        for qi, qid in enumerate(question_ids):
            trial = trials_by_id[qid]
            messages = build_messages(trial.question, condition, config.prompt_mode)
            prompt = render_chat(
                processor,
                messages,
                config.disable_thinking,
                config.chat_serialization,
            )
            annotated_ids, spans = attention_span_indices(tokenizer, prompt, condition, trial.question)
            positions = _anchor_positions(
                tokenizer,
                prompt,
                condition,
                spans,
                messages[0]["content"],
                messages[-1]["content"],
            )
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, [prompt])
            if annotated_ids != input_ids[0, : len(annotated_ids)].tolist():
                raise RuntimeError("Offset-aware and model input tokenizations disagree")
            valid_anchor_indices = [index for index, position in enumerate(positions) if position is not None]
            valid_positions = [int(positions[index]) for index in valid_anchor_indices]
            availability[ci, valid_anchor_indices] = True
            collector = PositionCollector(parts.layers, valid_positions)
            try:
                with torch.inference_mode():
                    kwargs = {
                        "input_ids": input_ids.to(model_input_device(parts)),
                        "attention_mask": attention_mask.to(model_input_device(parts)),
                        "use_cache": False,
                        "return_dict": True,
                    }
                    try:
                        model(**kwargs, logits_to_keep=1)
                    except TypeError:
                        model(**kwargs)
                captured = collector.stacked().numpy()
                values[ci, qi, :, valid_anchor_indices] = captured.transpose(1, 0, 2)
            finally:
                collector.close()
            audit["trials"][f"{condition}/{qid}"] = {
                "positions": positions,
                "tokens": [
                    tokenizer.decode([annotated_ids[position]]) if position is not None else None
                    for position in positions
                ],
                "prompt_length": len(annotated_ids),
            }
            completed = qi + 1
            if completed == 1 or completed % 25 == 0 or completed == len(question_ids):
                print(f"position residuals {condition}: {completed}/{len(question_ids)}", flush=True)
    audit["availability"] = {
        condition: {anchor: bool(availability[ci, ai]) for ai, anchor in enumerate(ANCHORS)}
        for ci, condition in enumerate(POSITION_CONDITIONS)
    }
    return values, audit


def _unembed_vector(parts, vector, include_bias: bool):
    import torch

    weight = parts.output_head.weight
    vector = vector.to(weight.device, dtype=weight.dtype)
    logits = vector @ weight.T
    bias = getattr(parts.output_head, "bias", None)
    if include_bias and bias is not None:
        logits = logits + bias
    return logits.float()


def _top_tokens(tokenizer, logits, k: int) -> dict[str, list[dict[str, Any]]]:
    import torch

    high_values, high_ids = torch.topk(logits, k=k)
    low_values, low_ids = torch.topk(-logits, k=k)
    def rows(values, ids, sign=1.0):
        return [
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "score": float(value) * sign,
            }
            for value, token_id in zip(values.cpu(), ids.cpu())
        ]
    return {"top": rows(high_values, high_ids), "bottom": rows(low_values, low_ids, -1.0)}


def collect(
    config_path: str | Path,
    residual_root: str | Path,
    output_root: str | Path,
    lens_repo: str,
    lens_filename: str,
    position_sample: int,
    top_k: int,
    seed: int,
    preflight_only: bool = False,
    reuse_final_root: str | Path | None = None,
    display_question_plan: str | Path | None = None,
) -> None:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    data = load_activation_dataset(residual_root, list(CONDITIONS))
    cached = _load_cached_residuals(residual_root, data)
    selected_qids = _stratified_sample(data, min(position_sample, len(data.question_ids)), seed)
    if display_question_plan is None:
        display_qids = list(selected_qids)
    else:
        display_plan = json.loads(Path(display_question_plan).read_text())
        display_qids = list(display_plan["discovery_question_ids"])
        missing = sorted(set(display_qids) - set(selected_qids))
        if missing:
            raise ValueError(
                "Display/discovery questions must be included in the position collection; "
                f"missing {len(missing)}"
            )
        if reuse_final_root is not None:
            raise ValueError(
                "A discovery-only display aggregate cannot reuse final means from another run"
            )

    lens_path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    jacobians = checkpoint["J"]
    source_layers = sorted(int(layer) for layer in jacobians)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    if checkpoint["d_model"] != cached.shape[-1] or source_layers != list(range(63)):
        raise ValueError(
            f"Incompatible lens: d_model={checkpoint['d_model']}, layers={source_layers[:2]}..{source_layers[-2:]}"
        )
    selected_ids, selected_layout, answer_tokens = _resolve_selected_tokens(tokenizer, config)
    selected_rows = parts.output_head.weight.detach()[selected_ids].float()
    selected_bias = getattr(parts.output_head, "bias", None)
    if selected_bias is not None:
        selected_bias = selected_bias.detach()[selected_ids].float()

    trials = load_trials(config.manifest_path, config.baseline_results_path)
    trials_by_id = {trial.question_id: trial for trial in trials}

    # The released artifact stores learned maps for blocks 1--63; the bottom
    # row of the official viewer is the model's actual block-64 output. Verify
    # that our stored final residual reproduces its cached natural logits.
    with np.load(shard_path(residual_root, "baseline", data.question_ids[0]), allow_pickle=False) as shard:
        natural_variant = shard["variant_logits"][-1].astype(np.float32)
        natural_residual = torch.from_numpy(shard["residuals"][-1].astype(np.float16)).to(model_input_device(parts))
    with torch.inference_mode():
        natural_norm = parts.final_norm(natural_residual.to(parts.final_norm.weight.dtype)).float()
        natural_selected = natural_norm @ selected_rows.T
    preflight = {
        "question_id": data.question_ids[0],
        "n_prompts": int(checkpoint["n_prompts"]),
        "d_model": int(checkpoint["d_model"]),
        "source_layers": source_layers,
        "natural_cached_vs_recomputed_variant_max_abs": float(
            np.max(np.abs(natural_variant - natural_selected[:8].cpu().numpy()))
        ),
        "last_learned_source_layer": source_layers[-1],
        "natural_final_readout": 63,
    }
    if preflight_only:
        smoke_residuals, smoke_audit = _collect_position_residuals(
            model, processor, parts, config, trials_by_id, selected_qids[:1]
        )
        preflight["position_smoke_shape"] = list(smoke_residuals.shape)
        preflight["position_smoke_audit"] = smoke_audit
        (output / "preflight.json").write_text(json.dumps(preflight, indent=2))
        print(json.dumps(preflight, indent=2), flush=True)
        return

    position_cache = output / "position_residuals.npy"
    position_cache_meta = output / "position_residuals_metadata.json"
    if position_cache.exists() and position_cache_meta.exists():
        cached_meta = json.loads(position_cache_meta.read_text())
        if cached_meta.get("question_ids") != selected_qids:
            cached_qids = cached_meta.get("question_ids", [])
            if (
                len(cached_qids) != len(selected_qids)
                or not set(cached_qids).issubset(data.question_ids)
            ):
                raise ValueError("Position-residual cache question IDs are incompatible with this run")
            # A refreshed Baseline can change the labels used only to stratify the
            # prompt-position sample.  The already collected Game/Neutral position
            # residuals remain valid, so preserve their frozen question IDs rather
            # than paying to execute the same prompts again.
            selected_qids = list(cached_qids)
            print("reused frozen position-sample question IDs from cache", flush=True)
        if cached_meta.get("anchors") != list(ANCHORS):
            raise ValueError("Position-residual cache anchors do not match this run")
        position_residuals = np.load(position_cache, mmap_mode="r")
        position_audit = cached_meta["audit"]
        print(f"reused cached position residuals: {position_residuals.shape}", flush=True)
    else:
        position_residuals, position_audit = _collect_position_residuals(
            model, processor, parts, config, trials_by_id, selected_qids
        )
        np.save(position_cache, position_residuals)
        position_cache_meta.write_text(json.dumps({
            "question_ids": selected_qids,
            "conditions": list(POSITION_CONDITIONS),
            "anchors": list(ANCHORS),
            "audit": position_audit,
        }, indent=2))
        print(f"cached position residuals: {position_cache}", flush=True)

    n_questions = len(data.question_ids)
    n_selected = len(selected_ids)
    display_set = set(display_qids)
    final_display_mask = np.asarray(
        [qid in display_set for qid in data.question_ids], dtype=bool
    )
    position_display_mask = np.asarray(
        [qid in display_set for qid in selected_qids], dtype=bool
    )
    if not final_display_mask.any() or not position_display_mask.any():
        raise ValueError("Display/discovery question set is empty")
    if reuse_final_root is not None:
        with np.load(Path(reuse_final_root) / "jlens_scores.npz", allow_pickle=False) as reused:
            reused_qids = reused["question_ids"].astype(str).tolist()
            if reused_qids != data.question_ids:
                raise ValueError("Reused final JLens question order differs from this run")
            final_scores = reused["final_scores"].astype(np.float16)
            reused_final_mean_norm = reused["final_mean_norm"].astype(np.float32)
        if final_scores.shape != (len(CONDITIONS), n_questions, 64, n_selected):
            raise ValueError(f"Unexpected reused final-score shape: {final_scores.shape}")
        print(f"reused final-position JLens outputs from {reuse_final_root}", flush=True)
    else:
        final_scores = np.empty((len(CONDITIONS), n_questions, 64, n_selected), dtype=np.float16)
        reused_final_mean_norm = None
    position_scores = np.empty(
        (len(POSITION_CONDITIONS), len(selected_qids), len(ANCHORS), 64, n_selected), dtype=np.float16
    )
    final_mean_norm = (
        torch.from_numpy(reused_final_mean_norm)
        if reused_final_mean_norm is not None
        else torch.zeros((len(CONDITIONS), 64, cached.shape[-1]), dtype=torch.float32)
    )
    position_mean_norm = torch.zeros(
        (len(POSITION_CONDITIONS), len(ANCHORS), 64, cached.shape[-1]), dtype=torch.float32
    )
    device = model_input_device(parts)
    batch_size = 64

    # Record the actual unrestricted next-token argmax at the shared first
    # assistant position.  This is deliberately separate from selected A--D
    # JLens scores so the two quantities cannot be conflated again.
    first_answer_anchor = ANCHORS.index("first_answer_decision")
    first_answer_full_vocab_top_ids = np.empty(
        (len(POSITION_CONDITIONS), len(selected_qids)), dtype=np.int32
    )
    with torch.inference_mode():
        for ci in range(len(POSITION_CONDITIONS)):
            values = position_residuals[ci, :, 63, first_answer_anchor]
            for start in range(0, len(values), batch_size):
                stop = min(start + batch_size, len(values))
                residual = torch.from_numpy(np.asarray(values[start:stop]).copy()).to(
                    device, dtype=parts.final_norm.weight.dtype
                )
                logits = parts.output_head(parts.final_norm(residual))
                first_answer_full_vocab_top_ids[ci, start:stop] = (
                    logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
                )
    baseline_full_vocab_top_ids = np.asarray(
        [
            int(data.metadata[(qid, "baseline")]["full_vocab_top_token_id"])
            for qid in selected_qids
        ],
        dtype=np.int32,
    )

    @torch.inference_mode()
    def transform(
        values: np.ndarray,
        J,
        score_target: np.ndarray,
        mean_target,
        layer: int,
        mean_mask: np.ndarray,
    ):
        total = values.shape[0]
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            residual = torch.from_numpy(values[start:stop]).to(device, dtype=torch.float16)
            transported = residual if J is None else residual @ J.T
            normed = parts.final_norm(transported.to(parts.final_norm.weight.dtype)).float()
            logits = normed @ selected_rows.T
            if selected_bias is not None:
                logits = logits + selected_bias
            score_target[start:stop, layer] = logits.cpu().to(torch.float16).numpy()
            local_mask = torch.from_numpy(mean_mask[start:stop]).to(normed.device)
            if local_mask.any():
                mean_target[layer] += normed[local_mask].sum(dim=0).cpu()

    @torch.inference_mode()
    def transform_positions(
        values: np.ndarray,
        J,
        score_target: np.ndarray,
        mean_target,
        layer: int,
        mean_mask: np.ndarray,
    ):
        total, anchors, width = values.shape
        flat = values.reshape(total * anchors, width)
        flat_mean_mask = np.repeat(mean_mask, anchors)
        flat_scores = np.empty((total * anchors, n_selected), dtype=np.float16)
        mean_sum = torch.zeros((anchors, width), dtype=torch.float32)
        for start in range(0, len(flat), batch_size):
            stop = min(start + batch_size, len(flat))
            residual = torch.from_numpy(flat[start:stop]).to(device, dtype=torch.float16)
            transported = residual if J is None else residual @ J.T
            normed = parts.final_norm(transported.to(parts.final_norm.weight.dtype)).float()
            logits = normed @ selected_rows.T
            if selected_bias is not None:
                logits = logits + selected_bias
            flat_scores[start:stop] = logits.cpu().to(torch.float16).numpy()
            local_mask = torch.from_numpy(flat_mean_mask[start:stop])
            if local_mask.any():
                indices = (torch.arange(start, stop) % anchors)[local_mask]
                mean_sum.index_add_(0, indices, normed.cpu()[local_mask])
        score_target[:, :, layer] = flat_scores.reshape(total, anchors, n_selected)
        mean_target[:, layer] = mean_sum

    for layer in source_layers:
        J = jacobians[layer].to(device, dtype=torch.float16)
        if reuse_final_root is None:
            for ci in range(len(CONDITIONS)):
                transform(
                    cached[ci, :, layer], J, final_scores[ci],
                    final_mean_norm[ci], layer, final_display_mask,
                )
        for ci in range(len(POSITION_CONDITIONS)):
            transform_positions(
                position_residuals[ci, :, layer], J, position_scores[ci],
                position_mean_norm[ci], layer, position_display_mask,
            )
        del J
        if layer == 0 or (layer + 1) % 8 == 0 or layer == source_layers[-1]:
            print(f"JLens transported learned map {layer + 1}/63", flush=True)

    # Official JLens visualizations use the model's actual output as their
    # bottom row rather than a stored identity transport.
    if reuse_final_root is None:
        for ci in range(len(CONDITIONS)):
            transform(
                cached[ci, :, 63], None, final_scores[ci],
                final_mean_norm[ci], 63, final_display_mask,
            )
    for ci in range(len(POSITION_CONDITIONS)):
        transform_positions(
            position_residuals[ci, :, 63], None, position_scores[ci],
            position_mean_norm[ci], 63, position_display_mask,
        )
    print("JLens added natural final readout 64/64", flush=True)

    if reuse_final_root is None:
        final_mean_norm /= int(final_display_mask.sum())
    position_mean_norm /= int(position_display_mask.sum())
    position_availability = np.asarray([
        [position_audit["availability"][condition][anchor] for anchor in ANCHORS]
        for condition in POSITION_CONDITIONS
    ], dtype=bool)

    top_tokens: dict[str, Any] = {
        "final": {},
        "positions": {},
        "position_availability": {
            condition: {
                anchor: bool(position_availability[ci, ai])
                for ai, anchor in enumerate(ANCHORS)
            }
            for ci, condition in enumerate(POSITION_CONDITIONS)
        },
    }
    with torch.inference_mode():
        for layer in range(64):
            for ci, condition in enumerate(CONDITIONS):
                logits = _unembed_vector(parts, final_mean_norm[ci, layer], include_bias=True)
                top_tokens["final"][f"{condition}/L{layer}"] = _top_tokens(tokenizer, logits, top_k)
            for first, second, name in ((1, 0, "game_minus_baseline"), (2, 0, "neutral_minus_baseline"), (1, 2, "game_minus_neutral")):
                contrast = final_mean_norm[first, layer] - final_mean_norm[second, layer]
                logits = _unembed_vector(parts, contrast, include_bias=False)
                top_tokens["final"][f"{name}/L{layer}"] = _top_tokens(tokenizer, logits, top_k)
            for ai, anchor in enumerate(ANCHORS):
                game = position_mean_norm[0, ai, layer]
                neutral = position_mean_norm[1, ai, layer]
                for ci, condition in enumerate(POSITION_CONDITIONS):
                    if not position_availability[ci, ai]:
                        continue
                    logits = _unembed_vector(parts, position_mean_norm[ci, ai, layer], include_bias=True)
                    top_tokens["positions"][f"{condition}/{anchor}/L{layer}"] = _top_tokens(tokenizer, logits, top_k)
                if position_availability[:, ai].all():
                    logits = _unembed_vector(parts, game - neutral, include_bias=False)
                    top_tokens["positions"][f"game_minus_neutral/{anchor}/L{layer}"] = _top_tokens(tokenizer, logits, top_k)

    np.savez_compressed(
        output / "jlens_scores.npz",
        final_scores=final_scores,
        position_scores=position_scores,
        final_mean_norm=final_mean_norm.numpy().astype(np.float16),
        position_mean_norm=position_mean_norm.numpy().astype(np.float16),
        question_ids=np.asarray(data.question_ids),
        position_question_ids=np.asarray(selected_qids),
        display_question_ids=np.asarray(display_qids),
        conditions=np.asarray(CONDITIONS),
        position_conditions=np.asarray(POSITION_CONDITIONS),
        anchors=np.asarray(ANCHORS),
        position_availability=position_availability,
        selected_token_ids=np.asarray(selected_ids, dtype=np.int32),
        first_answer_full_vocab_top_ids=first_answer_full_vocab_top_ids,
        baseline_full_vocab_top_ids=baseline_full_vocab_top_ids,
    )
    (output / "selected_token_layout.json").write_text(json.dumps(selected_layout, indent=2))
    (output / "top_tokens.json").write_text(json.dumps(top_tokens, indent=2))
    (output / "position_audit.json").write_text(json.dumps(position_audit, indent=2))
    metadata = {
        "config": config.as_dict(),
        "lens_repo": lens_repo,
        "lens_filename": lens_filename,
        "lens_path": lens_path,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "answer_tokens": answer_tokens,
        "preflight": preflight,
        "position_sampling": {
            "n": len(selected_qids),
            "seed": seed,
            "stratified_by": "generated Baseline letter x Game switch status",
        },
        "display_aggregation": {
            "n": len(display_qids),
            "question_ids": display_qids,
            "plan": str(display_question_plan) if display_question_plan else None,
            "note": "Only these discovery questions contribute to top-token means and contrasts.",
        },
        "layer_alignment": (
            "JLens layers 0--62 applied to cached post-block readouts 1--63; "
            "readout 64 is the model's natural final output"
        ),
        "reused_final_root": str(reuse_final_root) if reuse_final_root is not None else None,
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(preflight, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a pretrained Jacobian lens to Second Chance residuals")
    parser.add_argument("--config", required=True)
    parser.add_argument("--residual-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    parser.add_argument("--position-sample", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--reuse-final-root",
        help="Reuse final-position JLens arrays from a compatible prior run and collect only new prompt anchors",
    )
    parser.add_argument(
        "--display-question-plan",
        help="JSON plan containing discovery_question_ids used for all displayed means/contrasts",
    )
    args = parser.parse_args()
    collect(
        args.config, args.residual_root, args.output, args.lens_repo,
        args.lens_filename, args.position_sample, args.top_k, args.seed,
        args.preflight_only, args.reuse_final_root, args.display_question_plan,
    )


if __name__ == "__main__":
    main()
