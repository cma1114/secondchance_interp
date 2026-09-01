from __future__ import annotations

import argparse
import copy
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
    forward_runtime_kwargs,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
CONDITIONS = ("game", "neutral")
SCENARIOS = ("incorrect_again_nonremapped", "lost_again_nonremapped")
MODEL_ID = "ByteDance-Seed/Seed-OSS-36B-Instruct"
MODEL_REVISION = "497f1dca95ebdec98e41d517b9f060ee753c902f"
EXPERIMENT_MODEL_NAME = "Seed-OSS 36B"
CANONICAL_BATCH_SIZE = 4
EXPECTED_LAYER_COUNT = 64


class FinalPositionResidualCollector:
    """Capture every post-block final-position residual without FP16 requantization."""

    def __init__(self, parts: Any, last_indices: list[int]):
        self.last_indices = last_indices
        self.values: list[Any] = [None] * len(parts.layers)
        self.handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(parts.layers)
        ]

    def _hook(self, index: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = output[0] if isinstance(output, (tuple, list)) else output
            positions = torch.as_tensor(self.last_indices, device=hidden.device)
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            # Seed executes in BF16. FP32 storage preserves every BF16 value;
            # FP16 storage measurably perturbs the exact final readout.
            self.values[index] = hidden[batch, positions].detach().to(
                "cpu", dtype=torch.float32
            )

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def stacked(self):
        import torch

        if any(value is None for value in self.values):
            missing = [index + 1 for index, value in enumerate(self.values) if value is None]
            raise RuntimeError(f"Hooks did not capture Seed layers {missing}")
        return torch.stack(self.values, dim=1)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _chunks(values: list[int], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _assert_binding_config(config: ExperimentConfig) -> None:
    if config.model_id != MODEL_ID or config.model_revision != MODEL_REVISION:
        raise ValueError("Requires the binding's pinned configured model revision")
    if config.batch_size != CANONICAL_BATCH_SIZE:
        raise ValueError(
            f"Requires canonical batch_size={CANONICAL_BATCH_SIZE}, "
            f"found {config.batch_size}"
        )


def _assert_layer_count(layer_count: int) -> None:
    if layer_count != EXPECTED_LAYER_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_LAYER_COUNT} text-decoder layers for "
            f"{EXPERIMENT_MODEL_NAME}, found {layer_count}"
        )


def _experiment_name(dataset_name: str) -> str:
    return (
        f"{EXPERIMENT_MODEL_NAME} {dataset_name} "
        "non-remapped final-position trajectories"
    )


def _rank_order(rows: dict[str, Any], qids: list[str]) -> np.ndarray:
    order = np.empty((len(qids), 4), dtype=np.int64)
    for index, qid in enumerate(qids):
        logits = np.asarray(rows[qid]["aggregated_ad_logits"], dtype=np.float64)
        order[index] = np.argsort(-logits, kind="stable")
    return order


def _prepare_readout(parts: Any, variant_ids: list[list[int]], model: Any):
    import torch
    from accelerate.hooks import remove_hook_from_module

    device = parts.final_norm.weight.device
    norm = copy.deepcopy(parts.final_norm)
    if hasattr(norm, "_hf_hook"):
        remove_hook_from_module(norm, recurse=True)
    norm = norm.to(device).eval()
    flat_ids = [token for group in variant_ids for token in group]
    rows = parts.output_head.weight.detach()[flat_ids].float().to(device)
    bias = getattr(parts.output_head, "bias", None)
    if bias is not None:
        bias = bias.detach()[flat_ids].float().to(device)
    slices: list[slice] = []
    cursor = 0
    for group in variant_ids:
        slices.append(slice(cursor, cursor + len(group)))
        cursor += len(group)
    text_config = getattr(model.config, "text_config", model.config)
    softcap = getattr(text_config, "final_logit_softcapping", None)
    return norm, rows, bias, slices, device, softcap


def _standard_logit_lens(
    residuals: Any,
    norm: Any,
    rows: Any,
    bias: Any,
    slices: list[slice],
    device: Any,
    softcap: float | None,
) -> np.ndarray:
    import torch

    batch, layers, width = residuals.shape
    values = residuals.reshape(batch * layers, width).to(
        device=device, dtype=norm.weight.dtype
    )
    with torch.inference_mode():
        logits = norm(values).float() @ rows.T
        if bias is not None:
            logits = logits + bias
        if softcap is not None:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        scores = torch.stack(
            [torch.logsumexp(logits[:, group], dim=-1) for group in slices], dim=-1
        )
    return scores.reshape(batch, layers, 4).cpu().numpy().astype(np.float32)


def _aggregate_direct(logits: Any, variant_ids: list[list[int]]) -> np.ndarray:
    import torch

    return torch.stack(
        [torch.logsumexp(logits[:, group], dim=-1) for group in variant_ids], dim=-1
    ).float().cpu().numpy()


def _exact_model_head_scores(
    residuals: Any,
    parts: Any,
    variant_ids: list[list[int]],
    softcap: float | None,
) -> np.ndarray:
    """Reconstruct L64 through the model's own norm and full output-head arithmetic."""
    import torch

    device = parts.final_norm.weight.device
    values = residuals.to(device=device, dtype=parts.final_norm.weight.dtype)
    with torch.inference_mode():
        logits = parts.output_head(parts.final_norm(values))
        if softcap is not None:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
    return _aggregate_direct(logits, variant_ids)


def _collect_dataset(
    spec: dict[str, Any],
    model: Any,
    processor: Any,
    parts: Any,
    max_cohorts: int | None,
) -> None:
    import torch

    config = ExperimentConfig.load(Path(spec["config"]))
    _assert_binding_config(config)
    if config.model_loader not in {"causal_lm", "multimodal"} or config.chat_serialization != "hf_template":
        raise ValueError("Requires a supported native Hugging Face text path")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires the clean empty-first-assistant paradigm")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token-matched incorrect/lost feedback")
    layer_count = len(parts.layers)
    _assert_layer_count(layer_count)

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {str(row["id"]): row for row in manifest["questions"]}
    qids = [str(row["id"]) for row in manifest["questions"]]
    if config.question_ids is not None:
        wanted = set(config.question_ids)
        qids = [qid for qid in qids if qid in wanted]
        if set(qids) != wanted:
            raise RuntimeError("Configured questions are absent from the manifest")
    if config.max_questions is not None:
        qids = qids[: config.max_questions]
    if max_cohorts is not None:
        qids = qids[: int(max_cohorts) * config.batch_size]
    if len(qids) % config.batch_size:
        raise RuntimeError("Questions must form complete configured batches")

    trusted_payload = json.loads(Path(spec["trusted_behavior"]).read_text())
    if (
        trusted_payload.get("model_id") != config.model_id
        or trusted_payload.get("model_revision") != config.model_revision
        or not trusted_payload.get("complete")
    ):
        raise RuntimeError("Trusted behavior is incomplete or belongs to another model")
    baseline = trusted_payload["scenarios"]["baseline"]
    trusted = {
        condition: trusted_payload["scenarios"][scenario]
        for condition, scenario in zip(CONDITIONS, SCENARIOS)
    }
    if any(qid not in baseline for qid in qids) or any(
        qid not in trusted[condition] for condition in CONDITIONS for qid in qids
    ):
        raise RuntimeError("Trusted behavior is missing requested questions")

    remapping_rows = {
        str(row["question_id"]): row
        for row in json.loads(Path(spec["remapping_plan"]).read_text())["rows"]
    }
    if any(qid not in remapping_rows for qid in qids):
        raise RuntimeError("Frozen remapping plan is missing requested questions")

    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    residual_path = output / "decision_residuals.npy"
    completed_path = output / "completed.npy"
    direct_path = output / "direct_logits.npy"
    readout_path = output / "readout_scores.npy"
    n = len(qids)
    width = int(parts.embedding.weight.shape[-1])
    expected_residual_shape = (2, n, layer_count, width)
    if all(path.exists() for path in (residual_path, completed_path, direct_path, readout_path)):
        residuals = np.lib.format.open_memmap(residual_path, mode="r+")
        completed = np.load(completed_path)
        direct = np.load(direct_path)
        readout = np.load(readout_path)
        if residuals.shape != expected_residual_shape or completed.shape != (2, n):
            raise RuntimeError("Existing checkpoint has an incompatible shape")
    else:
        residuals = np.lib.format.open_memmap(
            residual_path,
            mode="w+",
            dtype=np.float32,
            shape=expected_residual_shape,
        )
        completed = np.zeros((2, n), dtype=bool)
        direct = np.full((2, n, 4), np.nan, dtype=np.float32)
        readout = np.full((2, n, layer_count, 4), np.nan, dtype=np.float32)

    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = [
        sorted({token_id for _text, token_id in resolved[letter]}) for letter in LETTERS
    ]
    norm, rows, bias, group_slices, readout_device, softcap = _prepare_readout(
        parts, variant_ids, model
    )
    input_device = model_input_device(parts)
    prompt_audit: dict[str, Any] = {
        "dataset": spec["name"],
        "conditions": list(CONDITIONS),
        "pair_checks": {},
        "trusted_prompt_max_logit_error": {condition: 0.0 for condition in CONDITIONS},
    }
    cohort_seconds: list[float] = []
    started = time.monotonic()

    for cohort in _chunks(list(range(n)), config.batch_size):
        cohort_started = time.monotonic()
        prompts_by_condition: list[list[str]] = []
        for condition, scenario in zip(CONDITIONS, SCENARIOS):
            prompts = []
            for index in cohort:
                qid = qids[index]
                messages, remapped = _scenario_messages(
                    scenario,
                    questions[qid],
                    remapping_rows[qid]["new_to_original"],
                )
                if remapped is not None:
                    raise RuntimeError("Non-remapped trajectory prompt unexpectedly remapped")
                prompts.append(
                    render_chat(
                        processor,
                        messages,
                        config.disable_thinking,
                        config.chat_serialization,
                        config.chat_template_kwargs,
                    )
                )
            prompts_by_condition.append(prompts)
        for game_prompt, neutral_prompt in zip(*prompts_by_condition):
            _assert_prompt_pair(game_prompt, neutral_prompt)

        for ci, condition in enumerate(CONDITIONS):
            pending = [index for index in cohort if not completed[ci, index]]
            if not pending:
                continue
            prompts = [prompts_by_condition[ci][cohort.index(index)] for index in pending]
            batch_qids = [qids[index] for index in pending]
            for prompt, qid in zip(prompts, batch_qids):
                if _prompt_hash(prompt) != trusted[condition][qid]["prompt_hash"]:
                    raise RuntimeError(f"Trusted prompt mismatch for {condition}/{qid}")
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            collector = FinalPositionResidualCollector(parts, last_indices)
            try:
                with torch.inference_mode():
                    kwargs = {
                        "input_ids": input_ids.to(input_device),
                        "attention_mask": attention_mask.to(input_device),
                        "return_dict": True,
                    }
                    kwargs.update(
                        forward_runtime_kwargs(model, input_ids, input_device)
                    )
                    try:
                        result = model(**kwargs, logits_to_keep=1)
                    except TypeError:
                        result = model(**kwargs)
                captured = collector.stacked()
            finally:
                collector.close()
            if captured.shape[1:] != (layer_count, width):
                raise RuntimeError(f"Unexpected residual shape {tuple(captured.shape)}")
            residuals[ci, pending] = captured.numpy()
            readout[ci, pending] = _standard_logit_lens(
                captured, norm, rows, bias, group_slices, readout_device, softcap
            )
            # The selected-row FP32 lens and the live full BF16 head need not be
            # bit-identical. Validate the stored final-layer residual through
            # the exact model modules, then use that exact reconstruction at
            # the final layer.
            readout[ci, pending, -1] = _exact_model_head_scores(
                captured[:, -1], parts, variant_ids, softcap
            )
            full_logits = result.logits.detach().float()
            final_logits = (
                full_logits[:, 0]
                if full_logits.shape[1] == 1
                else full_logits[
                    torch.arange(len(pending), device=full_logits.device),
                    last_indices.to(full_logits.device),
                ]
            )
            direct[ci, pending] = _aggregate_direct(final_logits, variant_ids)
            expected = np.asarray(
                [trusted[condition][qid]["aggregated_ad_logits"] for qid in batch_qids],
                dtype=np.float32,
            )
            error = float(np.max(np.abs(direct[ci, pending] - expected)))
            prompt_audit["trusted_prompt_max_logit_error"][condition] = max(
                prompt_audit["trusted_prompt_max_logit_error"][condition], error
            )
            if not np.isfinite(residuals[ci, pending]).all() or not np.isfinite(
                readout[ci, pending]
            ).all() or not np.isfinite(direct[ci, pending]).all():
                raise RuntimeError("Non-finite trajectory output")
            residuals.flush()
            completed[ci, pending] = True
            _atomic_npy(completed_path, completed)
            _atomic_npy(direct_path, direct)
            _atomic_npy(readout_path, readout)

        duration = time.monotonic() - cohort_started
        cohort_seconds.append(duration)
        print(
            f"{config.model_id} {spec['name']} trajectories: {int(completed.all(axis=0).sum())}/{n}; "
            f"cohort_seconds={duration:.3f}",
            flush=True,
        )

    if not completed.all():
        raise RuntimeError("Trajectory collection is incomplete")
    max_l64_error = float(np.max(np.abs(readout[:, :, -1] - direct)))
    if max_l64_error > 0.10:
        raise RuntimeError(
            f"Standard-logit-lens final-layer error is too large: {max_l64_error}"
        )

    for audit_index in sorted({0, n - 1}):
        qid = qids[audit_index]
        game_prompt = prompts_by_condition[0][cohort.index(audit_index)] if audit_index in cohort else None
        neutral_prompt = prompts_by_condition[1][cohort.index(audit_index)] if audit_index in cohort else None
        # Re-render endpoint prompts so the audit never depends on final cohort membership.
        rendered = []
        for scenario in SCENARIOS:
            messages, _ = _scenario_messages(
                scenario, questions[qid], remapping_rows[qid]["new_to_original"]
            )
            rendered.append(
                render_chat(
                    processor,
                    messages,
                    config.disable_thinking,
                    config.chat_serialization,
                    config.chat_template_kwargs,
                )
            )
        _assert_prompt_pair(rendered[0], rendered[1])
        game_ids = tokenizer.encode(rendered[0], add_special_tokens=False)
        neutral_ids = tokenizer.encode(rendered[1], add_special_tokens=False)
        differences = [
            index for index, (left, right) in enumerate(zip(game_ids, neutral_ids))
            if left != right
        ]
        if len(game_ids) != len(neutral_ids) or len(differences) != 1:
            raise RuntimeError("Expected exactly one model-visible Game/Neutral token difference")
        prompt_audit["pair_checks"][qid] = {
            "game_prompt_hash": _prompt_hash(rendered[0]),
            "neutral_prompt_hash": _prompt_hash(rendered[1]),
            "differing_token_position": differences[0],
            "game_token": tokenizer.decode([game_ids[differences[0]]]),
            "neutral_token": tokenizer.decode([neutral_ids[differences[0]]]),
            "prompt_token_count": len(game_ids),
        }

    order = _rank_order(baseline, qids)
    _atomic_npz(
        output / "results.npz",
        question_ids=np.asarray(qids),
        conditions=np.asarray(CONDITIONS),
        readout_scores=readout,
        direct_logits=direct,
        rank_order=order,
    )
    _atomic_json(output / "prompt_audit.json", prompt_audit)
    metadata = {
        "experiment": _experiment_name(spec["name"]),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "dataset": spec["name"],
        "questions": n,
        "conditions": list(CONDITIONS),
        "complete_model_forwards": 2 * n // config.batch_size,
        "position": "final prompt token immediately before the second assistant answer",
        "layers": list(range(1, layer_count + 1)),
        "readout": (
            "standard logit lens: each post-block residual is passed through the model's exact "
            "final norm, final-logit softcap when configured, and selected A-D unembedding rows; "
            "the final layer is validated through the full native output head"
        ),
        "answer_score": "logsumexp over bare and leading-space single-token A-D variants",
        "max_l64_reconstruction_error": max_l64_error,
        "decision_residuals": str(residual_path),
        "residual_storage_dtype": "float32 lossless representation of BF16 model states",
        "elapsed_seconds_after_load": time.monotonic() - started,
        "cohort_seconds": cohort_seconds,
        "all_outputs_finite": bool(
            np.isfinite(direct).all() and np.isfinite(readout).all()
        ),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
        },
        "platform": platform.platform(),
    }
    _atomic_json(output / "run_metadata.json", metadata)
    completed_path.unlink(missing_ok=True)
    direct_path.unlink(missing_ok=True)
    readout_path.unlink(missing_ok=True)
    print(json.dumps(metadata, indent=2), flush=True)


def collect(specs_path: Path, max_cohorts: int | None) -> None:
    specs = json.loads(specs_path.read_text())["datasets"]
    if not specs:
        raise ValueError("No dataset specifications")
    configs = [ExperimentConfig.load(Path(spec["config"])) for spec in specs]
    for config in configs:
        _assert_binding_config(config)
    first = configs[0]
    if any(
        (config.model_id, config.model_revision) != (first.model_id, first.model_revision)
        for config in configs[1:]
    ):
        raise ValueError("All datasets must use the same model revision")
    load_started = time.monotonic()
    model, processor, parts = load_model_and_processor(first)
    print(f"MODEL_LOADED seconds={time.monotonic() - load_started:.3f}", flush=True)
    for spec in specs:
        _collect_dataset(spec, model, processor, parts, max_cohorts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    collect(args.specs, args.max_cohorts)


if __name__ == "__main__":
    main()
