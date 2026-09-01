from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import load_trials, prompt_hash
from .run_baseline_mixer_function import (
    SelectedBlockOutputCollector,
    _forward,
    _letter_scores,
    _render,
)
from .sublayer import (
    BatchedRowSourcePositionComponentOutputPatcher,
    ComponentOutputCollector,
    ComponentTarget,
    PositionComponentOutputCollector,
    PositionComponentTarget,
)
from .sublayer_config import SublayerExperimentConfig


LAYER = 55  # zero-based Mixer 56
CONDITIONS = ("incorrect", "neutral")


def _batch_forward(
    model: Any,
    processor: Any,
    parts: Any,
    tokenizer: Any,
    config: SublayerExperimentConfig,
    trials: list[Any],
    condition: str,
    canonical_ids: list[int],
    collect: bool,
):
    prompts = [_render(processor, config, trial, condition) for trial in trials]
    input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
    collector = None
    if collect:
        collector = ComponentOutputCollector(
            parts, [ComponentTarget(LAYER, "mixer")], last_indices
        )
    try:
        result = _forward(model, parts, input_ids, attention_mask)
    finally:
        if collector is not None:
            collector.close()
    logits = result.logits.detach().float().cpu()[:, -1, canonical_ids].numpy()
    outputs = None if collector is None else collector.values[f"mixer_l{LAYER}"]
    return logits, outputs


def _discovery_means(
    model: Any,
    processor: Any,
    parts: Any,
    tokenizer: Any,
    config: SublayerExperimentConfig,
    trials: list[Any],
    canonical_ids: list[int],
    output_path: Path,
    batch_size: int,
):
    import torch

    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as cached:
            return {
                condition: {
                    f"decision__mixer_l{LAYER}": torch.from_numpy(
                        cached[f"mean_{condition}"]
                    ).unsqueeze(0)
                }
                for condition in CONDITIONS
            }
    width = int(parts.embedding.weight.shape[-1])
    sums = {
        condition: np.zeros((4, width), dtype=np.float64)
        for condition in CONDITIONS
    }
    counts = np.zeros(4, dtype=np.int64)
    for start in range(0, len(trials), batch_size):
        batch = trials[start : start + batch_size]
        baseline_logits, _ = _batch_forward(
            model, processor, parts, tokenizer, config, batch, "baseline", canonical_ids, False
        )
        baseline_winners = np.argmax(baseline_logits, axis=1)
        for winner in baseline_winners:
            counts[winner] += 1
        for condition in CONDITIONS:
            _, outputs = _batch_forward(
                model, processor, parts, tokenizer, config, batch, condition, canonical_ids, True
            )
            values = outputs.float().numpy()
            for row, winner in enumerate(baseline_winners):
                sums[condition][winner] += values[row]
        done = min(start + len(batch), len(trials))
        if done == len(batch) or done % 40 == 0 or done == len(trials):
            print(f"Mixer-56 condition means: {done}/{len(trials)}", flush=True)
    if np.any(counts == 0):
        raise RuntimeError(f"Missing a Baseline winner letter in discovery: {counts}")
    means = {
        condition: np.mean(sums[condition] / counts[:, None], axis=0).astype(np.float16)
        for condition in CONDITIONS
    }
    atomic_save_npz(
        output_path,
        letter_counts=counts,
        **{f"mean_{condition}": value for condition, value in means.items()},
    )
    return {
        condition: {
            f"decision__mixer_l{LAYER}": torch.from_numpy(means[condition]).unsqueeze(0)
        }
        for condition in CONDITIONS
    }


def _single_natural(
    model: Any,
    processor: Any,
    parts: Any,
    tokenizer: Any,
    config: SublayerExperimentConfig,
    trial: Any,
    condition: str,
    canonical_ids: list[int],
):
    prompt = _render(processor, config, trial, condition)
    input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, [prompt])
    position = int(last_indices[0])
    target = PositionComponentTarget(LAYER, "mixer", "decision")
    component = PositionComponentOutputCollector(
        parts, [target], {"decision": position}
    )
    block = SelectedBlockOutputCollector(parts, (LAYER,), position)
    try:
        result = _forward(model, parts, input_ids, attention_mask)
    finally:
        component.close()
        block.close()
    logits = result.logits.detach().float().cpu()[0, -1, canonical_ids].numpy()
    return (
        logits,
        component.values[f"decision__mixer_l{LAYER}"],
        block.values[LAYER],
        prompt,
    )


def _patched_batch(
    model: Any,
    processor: Any,
    parts: Any,
    tokenizer: Any,
    config: SublayerExperimentConfig,
    trial: Any,
    canonical_ids: list[int],
    means: dict[str, dict[str, Any]],
    natural_logits: dict[str, np.ndarray],
):
    prompts = []
    for condition in CONDITIONS:
        prompt = _render(processor, config, trial, condition)
        prompts.extend([prompt, prompt])
    input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
    target = PositionComponentTarget(LAYER, "mixer", "decision")
    patcher = BatchedRowSourcePositionComponentOutputPatcher(
        parts,
        [[target], [], [target], []],
        [means["incorrect"], {}, means["neutral"], {}],
        {"decision": int(last_indices[0])},
    )
    try:
        result = _forward(model, parts, input_ids, attention_mask)
    finally:
        patcher.close()
    raw = result.logits.detach().float().cpu()[:, -1, canonical_ids].numpy()
    return {
        "incorrect": natural_logits["incorrect"] + (raw[0] - raw[1]),
        "neutral": natural_logits["neutral"] + (raw[2] - raw[3]),
    }


def _initialize(path: Path, question_ids: list[str], width: int):
    if path.exists():
        cached = dict(np.load(path, allow_pickle=False))
        if cached["question_ids"].astype(str).tolist() != question_ids:
            raise ValueError("Confirmation question IDs changed")
        return cached
    n = len(question_ids)
    return {
        "question_ids": np.asarray(question_ids),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "mean_ablation_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "component_outputs": np.full((2, n, width), np.nan, dtype=np.float16),
        "post_block_residuals": np.full((2, n, width), np.nan, dtype=np.float16),
    }


def _jlens_writes(
    arrays: dict[str, np.ndarray],
    processor: Any,
    parts: Any,
    config: SublayerExperimentConfig,
    lens_repo: str,
    lens_filename: str,
):
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    J = checkpoint["J"][LAYER]
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    ids, layout = [], []
    for letter in "ABCD":
        for text, token_id in resolved[letter]:
            ids.append(token_id)
            layout.append({"letter": letter, "text": text, "token_id": token_id})
    device = model_input_device(parts)
    J = J.to(device=device, dtype=torch.float16)
    rows = parts.output_head.weight.detach()[ids].float().to(device)
    bias = getattr(parts.output_head, "bias", None)
    selected_bias = None if bias is None else bias.detach()[ids].float().to(device)
    n = len(arrays["question_ids"])
    writes = np.empty((2, n, 4), dtype=np.float32)
    with torch.inference_mode():
        for condition_index, condition in enumerate(CONDITIONS):
            for start in range(0, n, 64):
                stop = min(start + 64, n)
                post = torch.from_numpy(
                    arrays["post_block_residuals"][condition_index, start:stop]
                ).to(device=device, dtype=torch.float16)
                output = torch.from_numpy(
                    arrays["component_outputs"][condition_index, start:stop]
                ).to(device=device, dtype=torch.float16)
                full = parts.final_norm((post @ J.T).to(parts.final_norm.weight.dtype)).float()
                without = parts.final_norm(
                    ((post - output) @ J.T).to(parts.final_norm.weight.dtype)
                ).float()
                full_logits = full @ rows.T
                without_logits = without @ rows.T
                if selected_bias is not None:
                    full_logits += selected_bias
                    without_logits += selected_bias
                writes[condition_index, start:stop] = (
                    _letter_scores(full_logits, layout)
                    - _letter_scores(without_logits, layout)
                ).cpu().numpy()
            print(f"Mixer-56 immediate JLens write: {condition}", flush=True)
    return writes


def run(
    config_path: Path,
    discovery_plan_path: Path,
    confirmation_plan_path: Path,
    output: Path,
    batch_size: int,
    lens_repo: str,
    lens_filename: str,
):
    import torch
    import transformers

    config = SublayerExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history" or config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Canonical empty-history raw ChatML prompt required")
    discovery_ids = json.loads(discovery_plan_path.read_text())["question_ids"]
    confirmation_ids = json.loads(confirmation_plan_path.read_text())["question_ids"]
    if set(discovery_ids) & set(confirmation_ids):
        raise ValueError("Discovery and confirmation sets overlap")
    output.mkdir(parents=True, exist_ok=True)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    discovery_trials = load_trials(
        config.manifest_path, config.baseline_results_path, discovery_ids, None
    )
    confirmation_trials = load_trials(
        config.manifest_path, config.baseline_results_path, confirmation_ids, None
    )
    means = _discovery_means(
        model, processor, parts, tokenizer, config, discovery_trials,
        canonical_ids, output / "condition_means.npz", batch_size,
    )
    result_path = output / "mixer56_across_conditions_results.npz"
    arrays = _initialize(result_path, confirmation_ids, int(parts.embedding.weight.shape[-1]))
    for question_index, trial in enumerate(confirmation_trials):
        if bool(arrays["completed"][question_index]):
            continue
        natural = {}
        sources = {}
        blocks = {}
        prompts = {}
        for condition_index, condition in enumerate(CONDITIONS):
            logits, source, block, prompt = _single_natural(
                model, processor, parts, tokenizer, config, trial, condition, canonical_ids
            )
            natural[condition] = logits
            sources[condition] = source
            blocks[condition] = block
            prompts[condition] = prompt
            arrays["natural_logits"][condition_index, question_index] = logits
            arrays["component_outputs"][condition_index, question_index] = source[0].numpy()
            arrays["post_block_residuals"][condition_index, question_index] = block.numpy()
        patched = _patched_batch(
            model, processor, parts, tokenizer, config, trial, canonical_ids, means, natural
        )
        for condition_index, condition in enumerate(CONDITIONS):
            arrays["mean_ablation_logits"][condition_index, question_index] = patched[condition]
        arrays["completed"][question_index] = True
        if question_index == 0 and not (output / "prompt_audit.json").exists():
            (output / "prompt_audit.json").write_text(
                json.dumps(
                    {
                        "question_id": trial.question_id,
                        **{
                            condition: {
                                "prompt_hash": prompt_hash(prompts[condition]),
                                "prompt": prompts[condition],
                            }
                            for condition in CONDITIONS
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        done = int(np.sum(arrays["completed"]))
        if done == 1 or done % 5 == 0 or done == len(confirmation_trials):
            atomic_save_npz(result_path, **arrays)
            print(f"Mixer-56 within-condition confirmation: {done}/{len(confirmation_trials)}", flush=True)
    if "immediate_jlens_write" not in arrays:
        arrays["immediate_jlens_write"] = _jlens_writes(
            arrays, processor, parts, config, lens_repo, lens_filename
        )
    atomic_save_npz(result_path, **arrays)
    metadata = {
        "config": config.as_dict(),
        "n_discovery": len(discovery_ids),
        "n_confirmation": len(confirmation_ids),
        "conditions": list(CONDITIONS),
        "component": {"label": "Mixer 56", "zero_based_layer": LAYER, "one_based_block": 56},
        "mean_ablation": "Condition-specific Mixer-56 mean, equal-weighted over natural Baseline winner letters on the disjoint discovery set.",
        "immediate_write": "JLens finite-difference direct attribution at post-block 56, holding downstream computation fixed.",
        "batch_control": "Each condition's mean-ablation row is corrected by a condition-matched control in the same fixed four-row forward and recentered on single-trial natural logits.",
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--confirmation-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    args = parser.parse_args()
    run(
        args.config, args.discovery_plan, args.confirmation_plan, args.output,
        args.batch_size, args.lens_repo, args.lens_filename,
    )


if __name__ == "__main__":
    main()
