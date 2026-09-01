from __future__ import annotations

import argparse
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
from .jlens_collect import _token_offsets
from .modeling import (
    forward_runtime_kwargs,
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)


# Binding modules replace these values before calling main().
MODEL_ID = "ByteDance-Seed/Seed-OSS-36B-Instruct"
MODEL_REVISION = "497f1dca95ebdec98e41d517b9f060ee753c902f"
EXPERIMENT_MODEL_NAME = "Seed-OSS 36B"
CANONICAL_BATCH_SIZE = 4
EXPECTED_LAYER_COUNT = 64
FIRST_DECISION_OPENER = "<seed:bos>assistant"
ALLOWED_SERIALIZATIONS = ("hf_template",)
TRUSTED_SCENARIOS = ("incorrect_again_nonremapped", "lost_again_nonremapped")
CONDITIONS = ("game", "neutral")


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


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _assert_binding(config: ExperimentConfig) -> None:
    if (config.model_id, config.model_revision) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("Configured model does not match the pinned binding")
    if int(config.batch_size) != CANONICAL_BATCH_SIZE:
        raise ValueError(
            f"Requires canonical batch_size={CANONICAL_BATCH_SIZE}, found {config.batch_size}"
        )
    if config.chat_serialization not in ALLOWED_SERIALIZATIONS:
        raise ValueError(
            f"Requires serialization in {ALLOWED_SERIALIZATIONS}, "
            f"found {config.chat_serialization!r}"
        )
    if config.attn_implementation != "sdpa":
        raise ValueError("Requires the validated SDPA path")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires the canonical empty-first-answer prompt")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token-matched incorrect/lost feedback")
    if not config.disable_thinking:
        raise ValueError("Requires reasoning disabled")


def _first_decision_position(tokenizer: Any, prompt: str) -> tuple[int, dict[str, Any]]:
    start = prompt.find(FIRST_DECISION_OPENER)
    if start < 0:
        raise RuntimeError(f"Could not locate first-decision opener {FIRST_DECISION_OPENER!r}")
    if prompt.find(FIRST_DECISION_OPENER, start + 1) < 0:
        raise RuntimeError("Could not locate the final assistant opener")
    end = start + len(FIRST_DECISION_OPENER)
    offsets = _token_offsets(tokenizer, prompt)
    candidates = [
        index for index, (left, right) in enumerate(offsets)
        if right > left and right <= end
    ]
    if not candidates:
        raise RuntimeError("The first-decision opener has no token boundary")
    position = int(candidates[-1])
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return position, {
        "opener": FIRST_DECISION_OPENER,
        "position_unpadded": position,
        "token_id": int(ids[position]),
        "token": tokenizer.convert_ids_to_tokens([int(ids[position])])[0],
    }


def _load_baseline(path: Path, qids: list[str]) -> np.ndarray:
    payload = json.loads(path.read_text())
    if "results" in payload:
        rows = payload["results"]
    elif "scenarios" in payload and "baseline" in payload["scenarios"]:
        rows = payload["scenarios"]["baseline"]
    else:
        raise ValueError(f"Unrecognized baseline payload: {path}")
    missing = [qid for qid in qids if qid not in rows]
    if missing:
        raise ValueError(f"Baseline is missing {len(missing)} questions")
    return np.asarray(
        [rows[qid]["aggregated_ad_logits"] for qid in qids], dtype=np.float32
    )


def _load_trusted(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        qids = loaded["question_ids"].astype(str).tolist()
        if loaded["conditions"].astype(str).tolist() != list(CONDITIONS):
            raise ValueError("Trusted condition order is not Game/Neutral")
        natural = np.asarray(loaded["direct_logits"], dtype=np.float32)
        ranks = np.asarray(loaded["rank_order"], dtype=np.int64)
    if natural.shape != (2, len(qids), 4) or ranks.shape != (len(qids), 4):
        raise ValueError("Trusted trajectory has incompatible shapes")
    return qids, ranks, natural


def _aggregate(logits: Any, variant_ids: list[list[int]]) -> np.ndarray:
    import torch

    return torch.stack(
        [torch.logsumexp(logits[:, group], dim=-1) for group in variant_ids], dim=-1
    ).detach().float().cpu().numpy()


def _head_scores(residuals: Any, parts: Any, variant_ids: list[list[int]], softcap: float | None) -> np.ndarray:
    import torch

    device = parts.final_norm.weight.device
    values = residuals.to(device=device, dtype=parts.final_norm.weight.dtype)
    with torch.inference_mode():
        logits = parts.output_head(parts.final_norm(values))
        if softcap is not None:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
    # The canonical runners aggregate answer-token variants after promoting
    # vocabulary logits to float32.  Keeping logsumexp in BF16 silently rounds
    # the aggregate to 0.125-logit steps and can even change stable tie order.
    return _aggregate(logits.float(), variant_ids)


class DecisionCollector:
    """Capture post-block residuals at one audited decision position per row."""

    def __init__(self, parts: Any, positions: list[int]):
        self.positions = positions
        self.values: list[Any] = [None] * len(parts.layers)
        self.handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(parts.layers)
        ]

    def _hook(self, index: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.ndim != 3:
                raise RuntimeError(f"Layer {index + 1}: unexpected hidden shape {hidden.shape}")
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            pos = torch.as_tensor(self.positions, device=hidden.device)
            self.values[index] = hidden[batch, pos].detach().to(
                "cpu", dtype=torch.float32
            )

        return capture

    def stacked(self):
        import torch

        if any(value is None for value in self.values):
            missing = [index + 1 for index, value in enumerate(self.values) if value is None]
            raise RuntimeError(f"Hooks missed layers {missing}")
        # batch, layer, width
        return torch.stack(self.values, dim=1)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _initialize_memmaps(output: Path, n: int, layers: int, width: int):
    first_path = output / "first_decision_residuals.npy"
    second_path = output / "second_decision_residuals.npy"
    completed_path = output / "completed.npy"
    first_logits_path = output / "first_logits.npy"
    second_logits_path = output / "second_logits.npy"
    first_expected = (n, layers, width)
    second_expected = (2, n, layers, width)
    if all(path.exists() for path in (first_path, second_path, completed_path, first_logits_path, second_logits_path)):
        first = np.lib.format.open_memmap(first_path, mode="r+")
        second = np.lib.format.open_memmap(second_path, mode="r+")
        completed = np.load(completed_path)
        first_logits = np.load(first_logits_path)
        second_logits = np.load(second_logits_path)
        if (
            first.shape != first_expected
            or second.shape != second_expected
            or completed.shape != (3, n)
            or first_logits.shape != (n, 4)
            or second_logits.shape != (2, n, 4)
        ):
            raise ValueError("Existing uncertainty-state checkpoint has incompatible shape")
    else:
        first = np.lib.format.open_memmap(
            first_path, mode="w+", dtype=np.float32, shape=first_expected
        )
        second = np.lib.format.open_memmap(
            second_path, mode="w+", dtype=np.float32, shape=second_expected
        )
        completed = np.zeros((3, n), dtype=bool)
        first_logits = np.full((n, 4), np.nan, dtype=np.float32)
        second_logits = np.full((2, n, 4), np.nan, dtype=np.float32)
    return first, second, completed, first_logits, second_logits


def _collect_dataset(
    spec: dict[str, Any], model: Any, processor: Any, parts: Any,
    max_cohorts: int | None, output_tag: str | None,
) -> None:
    import torch

    config = ExperimentConfig.load(Path(spec["config"]))
    _assert_binding(config)
    if len(parts.layers) != EXPECTED_LAYER_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_LAYER_COUNT} layers, found {len(parts.layers)}"
        )
    qids, trusted_ranks, trusted_natural = _load_trusted(Path(spec["trusted_trajectory"]))
    if max_cohorts is not None:
        qids = qids[: int(max_cohorts) * config.batch_size]
        trusted_ranks = trusted_ranks[: len(qids)]
        trusted_natural = trusted_natural[:, : len(qids)]
    if len(qids) % config.batch_size:
        raise RuntimeError("Question count must form complete canonical cohorts")
    baseline = _load_baseline(Path(spec["baseline_results"]), qids)
    if not np.array_equal(np.argsort(-baseline, axis=1, kind="stable"), trusted_ranks):
        raise RuntimeError("Baseline logits do not reproduce the trusted rank order")

    manifest = json.loads(Path(config.manifest_path).read_text())["questions"]
    questions = {str(row["id"]): row for row in manifest}
    if any(qid not in questions for qid in qids):
        raise RuntimeError("Manifest is missing trusted questions")

    output = Path(spec["output"])
    if output_tag is not None:
        output = output.parent / f"{output.name}_{output_tag}"
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = [
        sorted({token_id for _text, token_id in resolved[letter]}) for letter in LETTERS
    ]
    width = int(parts.embedding.weight.shape[-1])
    first, second, completed, first_logits, second_logits = _initialize_memmaps(
        output, len(qids), EXPECTED_LAYER_COUNT, width
    )
    text_config = getattr(model.config, "text_config", model.config)
    softcap = getattr(text_config, "final_logit_softcapping", None)
    input_device = model_input_device(parts)
    cohort_seconds: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"dataset": spec["name"], "baseline": {}, "conditions": {}}
    head_reconstruction_error = 0.0
    started = time.monotonic()

    # Train the uncertainty direction on the actual standalone 1P decision
    # forward.  Capturing the same prefix from inside the longer 2P sequence is
    # conceptually causal but produces small sequence-shape-dependent BF16
    # differences on Qwen's recurrent kernels.
    for start in range(0, len(qids), config.batch_size):
        indices = list(range(start, start + config.batch_size))
        if np.all(completed[0, indices]):
            continue
        cohort_started = time.monotonic()
        cohort = [qids[index] for index in indices]
        prompts: list[str] = []
        for qid in cohort:
            messages, remapping = _scenario_messages(
                "baseline", questions[qid], {letter: letter for letter in LETTERS}
            )
            if remapping is not None:
                raise RuntimeError("Canonical baseline prompt unexpectedly remapped")
            prompts.append(
                render_chat(
                    processor,
                    messages,
                    config.disable_thinking,
                    config.chat_serialization,
                    config.chat_template_kwargs,
                )
            )
        input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
        collector = DecisionCollector(parts, [int(value) for value in last_indices])
        try:
            kwargs = {
                "input_ids": input_ids.to(input_device),
                "attention_mask": attention_mask.to(input_device),
                "return_dict": True,
            }
            kwargs.update(forward_runtime_kwargs(model, input_ids, input_device))
            with torch.inference_mode():
                try:
                    result = model(**kwargs, logits_to_keep=1)
                except TypeError:
                    result = model(**kwargs)
            captured = collector.stacked()
        finally:
            collector.close()
        if captured.shape != (len(indices), EXPECTED_LAYER_COUNT, width):
            raise RuntimeError(f"Unexpected standalone 1P captured shape {tuple(captured.shape)}")
        live_logits = result.logits.detach().float()
        final = (
            live_logits[:, 0]
            if live_logits.shape[1] == 1
            else live_logits[
                torch.arange(len(indices), device=live_logits.device),
                torch.as_tensor(last_indices, device=live_logits.device),
            ]
        )
        live_scores = _aggregate(final, variant_ids)
        reconstructed = _head_scores(captured[:, -1], parts, variant_ids, softcap)
        head_reconstruction_error = max(
            head_reconstruction_error,
            float(np.max(np.abs(reconstructed - live_scores))),
        )
        first[indices] = captured.numpy()
        first_logits[indices] = live_scores
        if not np.isfinite(first[indices]).all() or not np.isfinite(first_logits[indices]).all():
            raise RuntimeError("Non-finite standalone 1P uncertainty-state output")
        first.flush()
        completed[0, indices] = True
        _atomic_npy(output / "completed.npy", completed)
        _atomic_npy(output / "first_logits.npy", first_logits)
        _atomic_npy(output / "second_logits.npy", second_logits)
        duration = time.monotonic() - cohort_started
        cohort_seconds.append({"phase": "baseline", "seconds": duration})
        print(
            f"{EXPERIMENT_MODEL_NAME} {spec['name']} baseline uncertainty states: "
            f"{int(completed[0].sum())}/{len(qids)}; cohort_seconds={duration:.3f}",
            flush=True,
        )
        if not audit["baseline"]:
            ids = tokenizer(prompts[0], add_special_tokens=False)["input_ids"]
            audit["baseline"] = {
                "question_id": cohort[0],
                "prompt_hash": _hash_prompt(prompts[0]),
                "rendered_prompt": prompts[0],
                "decision_position_unpadded": len(ids) - 1,
                "decision_token_id": int(ids[-1]),
                "decision_token": tokenizer.convert_ids_to_tokens([int(ids[-1])])[0],
            }

    for condition_index, scenario in enumerate(TRUSTED_SCENARIOS):
        for start in range(0, len(qids), config.batch_size):
            indices = list(range(start, start + config.batch_size))
            completed_row = condition_index + 1
            if np.all(completed[completed_row, indices]):
                continue
            cohort_started = time.monotonic()
            cohort = [qids[index] for index in indices]
            prompts: list[str] = []
            token_rows: list[list[int]] = []
            first_audits: list[dict[str, Any]] = []
            for qid in cohort:
                messages, remapping = _scenario_messages(
                    scenario, questions[qid], {letter: letter for letter in LETTERS}
                )
                if remapping is not None:
                    raise RuntimeError("Canonical uncertainty prompt unexpectedly remapped")
                prompt = render_chat(
                    processor,
                    messages,
                    config.disable_thinking,
                    config.chat_serialization,
                    config.chat_template_kwargs,
                )
                baseline_messages, baseline_remapping = _scenario_messages(
                    "baseline", questions[qid], {letter: letter for letter in LETTERS}
                )
                if baseline_remapping is not None:
                    raise RuntimeError("Canonical baseline prompt unexpectedly remapped")
                baseline_prompt = render_chat(
                    processor,
                    baseline_messages,
                    config.disable_thinking,
                    config.chat_serialization,
                    config.chat_template_kwargs,
                )
                baseline_boundary = baseline_prompt.find(FIRST_DECISION_OPENER)
                prompt_boundary = prompt.find(FIRST_DECISION_OPENER)
                if baseline_boundary < 0 or prompt_boundary < 0:
                    raise RuntimeError("Could not locate the shared first assistant opener")
                baseline_boundary += len(FIRST_DECISION_OPENER)
                prompt_boundary += len(FIRST_DECISION_OPENER)
                if baseline_prompt[:baseline_boundary] != prompt[:prompt_boundary]:
                    raise RuntimeError(
                        "Standalone 1P and 2P prompts differ before the first assistant opener"
                    )
                prompts.append(prompt)
                token_rows.append(
                    [int(value) for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
                )
                _position, position_audit = _first_decision_position(tokenizer, prompt)
                first_audits.append(position_audit)

            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            physical_positions: list[int] = []
            first_physical_positions: list[int] = []
            for row, ids in enumerate(token_rows):
                left_pad = int(input_ids.shape[1]) - len(ids)
                if input_ids[row, left_pad:].tolist() != ids:
                    raise RuntimeError("Batched tokenization differs from audited tokenization")
                first_position = int(first_audits[row]["position_unpadded"]) + left_pad
                last_position = int(last_indices[row])
                if not (0 <= first_position < last_position < input_ids.shape[1]):
                    raise RuntimeError("Invalid 1P/2P decision ordering")
                first_physical_positions.append(first_position)
                physical_positions.append(last_position)

            collector = DecisionCollector(parts, physical_positions)
            try:
                kwargs = {
                    "input_ids": input_ids.to(input_device),
                    "attention_mask": attention_mask.to(input_device),
                    "return_dict": True,
                }
                kwargs.update(forward_runtime_kwargs(model, input_ids, input_device))
                with torch.inference_mode():
                    try:
                        result = model(**kwargs, logits_to_keep=1)
                    except TypeError:
                        result = model(**kwargs)
                captured = collector.stacked()
            finally:
                collector.close()
            if captured.shape != (len(indices), EXPECTED_LAYER_COUNT, width):
                raise RuntimeError(f"Unexpected captured shape {tuple(captured.shape)}")
            second[condition_index, indices] = captured.numpy()
            live_logits = result.logits.detach().float()
            final = (
                live_logits[:, 0]
                if live_logits.shape[1] == 1
                else live_logits[
                    torch.arange(len(indices), device=live_logits.device),
                    torch.as_tensor(last_indices, device=live_logits.device),
                ]
            )
            second_logits[condition_index, indices] = _aggregate(final, variant_ids)
            if not (
                np.isfinite(second[condition_index, indices]).all()
                and np.isfinite(second_logits[condition_index, indices]).all()
            ):
                raise RuntimeError("Non-finite uncertainty-state output")
            second.flush()
            completed[completed_row, indices] = True
            _atomic_npy(output / "completed.npy", completed)
            _atomic_npy(output / "first_logits.npy", first_logits)
            _atomic_npy(output / "second_logits.npy", second_logits)
            duration = time.monotonic() - cohort_started
            cohort_seconds.append({"phase": CONDITIONS[condition_index], "seconds": duration})
            print(
                f"{EXPERIMENT_MODEL_NAME} {spec['name']} {CONDITIONS[condition_index]} "
                f"uncertainty states: {int(completed[completed_row].sum())}/{len(qids)}; "
                f"cohort_seconds={duration:.3f}",
                flush=True,
            )
            if not audit["conditions"].get(CONDITIONS[condition_index]):
                audit["conditions"][CONDITIONS[condition_index]] = {
                    "question_id": cohort[0],
                    "prompt_hash": _hash_prompt(prompts[0]),
                    "rendered_prompt": prompts[0],
                    "first_decision": first_audits[0],
                    "first_decision_physical": first_physical_positions[0],
                    "second_decision_physical": physical_positions[0],
                    "standalone_1p_context_through_assistant_opener_exact": True,
                }

    if not completed.all():
        raise RuntimeError("Uncertainty-state collection is incomplete")
    _assert_prompt_pair(
        audit["conditions"]["game"]["rendered_prompt"],
        audit["conditions"]["neutral"]["rendered_prompt"],
    )
    first_baseline_error = float(np.max(np.abs(first_logits - baseline)))
    second_trusted_error = float(np.max(np.abs(second_logits - trusted_natural)))
    if (
        first_baseline_error != 0.0
        or second_trusted_error != 0.0
        or head_reconstruction_error != 0.0
    ):
        raise RuntimeError(
            "Natural reproduction failed: "
            f"1P={first_baseline_error}, 2P={second_trusted_error}, "
            f"head={head_reconstruction_error}"
        )

    _atomic_npz(
        output / "results.npz",
        question_ids=np.asarray(qids),
        conditions=np.asarray(CONDITIONS),
        baseline_logits=baseline,
        first_logits=first_logits,
        second_logits=second_logits,
        rank_order=trusted_ranks,
    )
    _atomic_json(output / "prompt_audit.json", audit)
    _atomic_json(
        output / "run_metadata.json",
        {
            "experiment": "1P-trained MCQ uncertainty direction state collection",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset": spec["name"],
            "questions": len(qids),
            "conditions": list(CONDITIONS),
            "layers": list(range(1, EXPECTED_LAYER_COUNT + 1)),
            "positions": {
                "first": "final token of the first assistant answer-generation prefix",
                "second": "final prompt token before the second assistant answer",
            },
            "complete_model_forwards": 3 * len(qids) // config.batch_size,
            "residual_storage_dtype": "float32 lossless representation of BF16 states",
            "first_baseline_max_abs_error": first_baseline_error,
            "second_trusted_max_abs_error": second_trusted_error,
            "head_reconstruction_max_abs_error": head_reconstruction_error,
            "all_outputs_finite": True,
            "elapsed_seconds_after_model_load": time.monotonic() - started,
            "cohort_seconds": cohort_seconds,
            "software": {"python": sys.version, "torch": torch.__version__},
            "platform": platform.platform(),
        },
    )
    for path in (output / "completed.npy", output / "first_logits.npy", output / "second_logits.npy"):
        path.unlink(missing_ok=True)


def collect(specs_path: Path, max_cohorts: int | None, output_tag: str | None) -> None:
    specs = json.loads(specs_path.read_text())["datasets"]
    if not specs:
        raise ValueError("No dataset specifications")
    configs = [ExperimentConfig.load(Path(spec["config"])) for spec in specs]
    for config in configs:
        _assert_binding(config)
    load_started = time.monotonic()
    model, processor, parts = load_model_and_processor(configs[0])
    print(f"MODEL_LOADED seconds={time.monotonic() - load_started:.3f}", flush=True)
    for spec in specs:
        _collect_dataset(spec, model, processor, parts, max_cohorts, output_tag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    parser.add_argument("--output-tag")
    args = parser.parse_args()
    collect(args.specs, args.max_cohorts, args.output_tag)


if __name__ == "__main__":
    main()
