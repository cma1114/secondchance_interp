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
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages, load_trials, prompt_hash
from .sublayer import (
    BatchedRowSourcePositionComponentOutputPatcher,
    ComponentOutputCollector,
    ComponentTarget,
    PositionComponentOutputCollector,
    PositionComponentTarget,
    _hidden,
)
from .sublayer_config import SublayerExperimentConfig


LAYERS = (55, 62)  # zero-based: Mixers 56 and 63
LABELS = ("Mixer 56", "Mixer 63")
SCENARIOS = ("mixer56", "mixer63", "both")


class SelectedBlockOutputCollector:
    """Capture selected post-block residuals at the final prompt position."""

    def __init__(self, parts: Any, layers: tuple[int, ...], last_index: int):
        self.last_index = int(last_index)
        self.values: dict[int, Any] = {}
        self.handles = [
            parts.layers[layer].register_forward_hook(self._hook(layer))
            for layer in layers
        ]

    def _hook(self, layer: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = _hidden(output)
            if hidden.shape[0] != 1:
                raise ValueError("SelectedBlockOutputCollector expects batch size one")
            self.values[layer] = (
                hidden[0, self.last_index].detach().to("cpu", dtype=__import__("torch").float16)
            )

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _render(processor: Any, config: SublayerExperimentConfig, trial: Any, condition: str):
    messages = build_messages(trial.question, condition, config.prompt_mode)
    prompt = render_chat(
        processor,
        messages,
        config.disable_thinking,
        config.chat_serialization,
    )
    return prompt


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


def _collect_discovery_means(
    model: Any,
    processor: Any,
    parts: Any,
    tokenizer: Any,
    config: SublayerExperimentConfig,
    trials: list[Any],
    canonical_ids: list[int],
    output_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as cached:
            return {
                "means": {
                    f"decision__mixer_l{layer}": torch.from_numpy(
                        cached[f"mean_mixer_l{layer}"]
                    ).unsqueeze(0)
                    for layer in LAYERS
                },
                "letter_counts": cached["letter_counts"].astype(int),
            }

    targets = [ComponentTarget(layer, "mixer") for layer in LAYERS]
    width = int(parts.embedding.weight.shape[-1])
    sums = {layer: np.zeros((4, width), dtype=np.float64) for layer in LAYERS}
    counts = np.zeros(4, dtype=np.int64)

    for start in range(0, len(trials), batch_size):
        batch = trials[start : start + batch_size]
        prompts = [_render(processor, config, trial, "baseline") for trial in batch]
        input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
        collector = ComponentOutputCollector(parts, targets, last_indices)
        try:
            result = _forward(model, parts, input_ids, attention_mask)
        finally:
            collector.close()
        logits = result.logits.detach().float().cpu()[:, -1, canonical_ids].numpy()
        winners = np.argmax(logits, axis=1)
        for row, winner in enumerate(winners):
            counts[winner] += 1
            for layer in LAYERS:
                sums[layer][winner] += collector.values[f"mixer_l{layer}"][row].float().numpy()
        done = min(start + len(batch), len(trials))
        if done == len(batch) or done % 40 == 0 or done == len(trials):
            print(f"Baseline discovery means: {done}/{len(trials)}", flush=True)

    if np.any(counts == 0):
        raise RuntimeError(f"A Baseline answer letter is absent from discovery: {counts}")
    # Equal-weight the generated Baseline answer letters so the ablation source
    # does not inherit the model's substantial A/B/C/D frequency imbalance.
    means_np = {
        layer: np.mean(sums[layer] / counts[:, None], axis=0).astype(np.float16)
        for layer in LAYERS
    }
    atomic_save_npz(
        output_path,
        **{f"mean_mixer_l{layer}": means_np[layer] for layer in LAYERS},
        letter_counts=counts,
    )
    return {
        "means": {
            f"decision__mixer_l{layer}": torch.from_numpy(means_np[layer]).unsqueeze(0)
            for layer in LAYERS
        },
        "letter_counts": counts,
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
    collect_baseline: bool,
):
    import torch

    prompt = _render(processor, config, trial, condition)
    input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, [prompt])
    sources = None
    blocks = None
    component_collector = None
    block_collector = None
    if collect_baseline:
        position = int(last_indices[0])
        targets = [PositionComponentTarget(layer, "mixer", "decision") for layer in LAYERS]
        component_collector = PositionComponentOutputCollector(
            parts, targets, {"decision": position}
        )
        block_collector = SelectedBlockOutputCollector(parts, LAYERS, position)
    try:
        result = _forward(model, parts, input_ids, attention_mask)
    finally:
        if component_collector is not None:
            component_collector.close()
        if block_collector is not None:
            block_collector.close()
    logits = result.logits.detach().float().cpu()[0, -1, canonical_ids].numpy()
    full_top = int(torch.argmax(result.logits.detach()[0, -1]).cpu())
    if collect_baseline:
        sources = component_collector.values
        blocks = {layer: block_collector.values[layer] for layer in LAYERS}
    return logits, full_top, sources, blocks, prompt


def _patched_batch(
    model: Any,
    processor: Any,
    parts: Any,
    tokenizer: Any,
    config: SublayerExperimentConfig,
    trial: Any,
    canonical_ids: list[int],
    mean_source: dict[str, Any],
    paired_baseline_source: dict[str, Any],
    natural_baseline_logits: np.ndarray,
    natural_game_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_prompt = _render(processor, config, trial, "baseline")
    game_prompt = _render(processor, config, trial, "incorrect")
    prompts = [baseline_prompt] * 4 + [game_prompt] * 4
    input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
    decision = int(last_indices[0])
    t56 = PositionComponentTarget(55, "mixer", "decision")
    t63 = PositionComponentTarget(62, "mixer", "decision")
    targets_by_row = [[t56], [t63], [t56, t63], [], [t56], [t63], [t56, t63], []]
    sources_by_row = [
        mean_source,
        mean_source,
        mean_source,
        {},
        paired_baseline_source,
        paired_baseline_source,
        paired_baseline_source,
        {},
    ]
    patcher = BatchedRowSourcePositionComponentOutputPatcher(
        parts,
        targets_by_row,
        sources_by_row,
        {"decision": decision},
    )
    try:
        result = _forward(model, parts, input_ids, attention_mask)
    finally:
        patcher.close()
    raw = result.logits.detach().float().cpu()[:, -1, canonical_ids].numpy()
    baseline_corrected = natural_baseline_logits[None, :] + (raw[:3] - raw[3])
    game_corrected = natural_game_logits[None, :] + (raw[4:7] - raw[7])
    return baseline_corrected, game_corrected


def _load_or_initialize(path: Path, qids: list[str], width: int):
    n = len(qids)
    if path.exists():
        cached = dict(np.load(path, allow_pickle=False))
        if cached["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Checkpoint question IDs do not match the confirmation plan")
        return cached
    return {
        "question_ids": np.asarray(qids),
        "completed": np.zeros(n, dtype=bool),
        "correct_indices": np.full(n, -1, dtype=np.int8),
        "baseline_full_top_ids": np.full(n, -1, dtype=np.int32),
        "game_full_top_ids": np.full(n, -1, dtype=np.int32),
        "natural_baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "natural_game_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "baseline_mean_ablation_logits": np.full((n, 3, 4), np.nan, dtype=np.float32),
        "game_baseline_insertion_logits": np.full((n, 3, 4), np.nan, dtype=np.float32),
        "baseline_component_outputs": np.full((n, 2, width), np.nan, dtype=np.float16),
        "baseline_post_block_residuals": np.full((n, 2, width), np.nan, dtype=np.float16),
    }


def _letter_scores(logits: Any, layout: list[dict[str, Any]]):
    import torch

    values = []
    for letter in "ABCD":
        indices = [i for i, row in enumerate(layout) if row["letter"] == letter]
        values.append(torch.logsumexp(logits[:, indices], dim=1))
    return torch.stack(values, dim=1)


def _compute_immediate_jlens_writes(
    arrays: dict[str, np.ndarray],
    model: Any,
    processor: Any,
    parts: Any,
    config: SublayerExperimentConfig,
    lens_repo: str,
    lens_filename: str,
) -> np.ndarray:
    import torch
    from huggingface_hub import hf_hub_download

    lens_path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    jacobians = checkpoint["J"]
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids: list[int] = []
    layout: list[dict[str, Any]] = []
    for letter in "ABCD":
        for text, token_id in resolved[letter]:
            variant_ids.append(token_id)
            layout.append({"letter": letter, "text": text, "token_id": token_id})
    rows = parts.output_head.weight.detach()[variant_ids].float()
    bias = getattr(parts.output_head, "bias", None)
    selected_bias = None if bias is None else bias.detach()[variant_ids].float()
    device = model_input_device(parts)
    rows = rows.to(device)
    if selected_bias is not None:
        selected_bias = selected_bias.to(device)
    n = len(arrays["question_ids"])
    writes = np.empty((n, 2, 4), dtype=np.float32)
    batch_size = 64
    with torch.inference_mode():
        for component_index, layer in enumerate(LAYERS):
            J = jacobians[layer].to(device=device, dtype=torch.float16)
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                post = torch.from_numpy(
                    arrays["baseline_post_block_residuals"][start:stop, component_index]
                ).to(device=device, dtype=torch.float16)
                output = torch.from_numpy(
                    arrays["baseline_component_outputs"][start:stop, component_index]
                ).to(device=device, dtype=torch.float16)
                transported_full = post @ J.T
                transported_without = (post - output) @ J.T
                full_norm = parts.final_norm(
                    transported_full.to(parts.final_norm.weight.dtype)
                ).float()
                without_norm = parts.final_norm(
                    transported_without.to(parts.final_norm.weight.dtype)
                ).float()
                full_logits = full_norm @ rows.T
                without_logits = without_norm @ rows.T
                if selected_bias is not None:
                    full_logits = full_logits + selected_bias
                    without_logits = without_logits + selected_bias
                writes[start:stop, component_index] = (
                    _letter_scores(full_logits, layout)
                    - _letter_scores(without_logits, layout)
                ).cpu().numpy()
            del J
            print(f"Immediate JLens write: {LABELS[component_index]}", flush=True)
    return writes


def run(
    config_path: Path,
    discovery_plan_path: Path,
    confirmation_plan_path: Path,
    output: Path,
    discovery_batch_size: int,
    lens_repo: str,
    lens_filename: str,
) -> None:
    import torch
    import transformers

    config = SublayerExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("This experiment requires the canonical empty-history prompt")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("This experiment requires explicit raw Qwen ChatML")
    discovery_plan = json.loads(discovery_plan_path.read_text())
    confirmation_plan = json.loads(confirmation_plan_path.read_text())
    discovery_ids = list(discovery_plan["question_ids"])
    confirmation_ids = list(confirmation_plan["question_ids"])
    if set(discovery_ids) & set(confirmation_ids):
        raise ValueError("Discovery and confirmation question sets overlap")

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
    discovery = _collect_discovery_means(
        model,
        processor,
        parts,
        tokenizer,
        config,
        discovery_trials,
        canonical_ids,
        output / "discovery_means.npz",
        discovery_batch_size,
    )

    result_path = output / "baseline_mixer_function_results.npz"
    width = int(parts.embedding.weight.shape[-1])
    arrays = _load_or_initialize(result_path, confirmation_ids, width)
    arrays["canonical_token_ids"] = np.asarray(canonical_ids, dtype=np.int32)
    first_audit = None
    for index, trial in enumerate(confirmation_trials):
        if bool(arrays["completed"][index]):
            continue
        baseline_logits, baseline_top, baseline_source, block_values, baseline_prompt = _single_natural(
            model,
            processor,
            parts,
            tokenizer,
            config,
            trial,
            "baseline",
            canonical_ids,
            True,
        )
        game_logits, game_top, _, _, game_prompt = _single_natural(
            model,
            processor,
            parts,
            tokenizer,
            config,
            trial,
            "incorrect",
            canonical_ids,
            False,
        )
        baseline_patched, game_patched = _patched_batch(
            model,
            processor,
            parts,
            tokenizer,
            config,
            trial,
            canonical_ids,
            discovery["means"],
            baseline_source,
            baseline_logits,
            game_logits,
        )
        arrays["correct_indices"][index] = "ABCD".index(trial.question["correct_answer"])
        arrays["baseline_full_top_ids"][index] = baseline_top
        arrays["game_full_top_ids"][index] = game_top
        arrays["natural_baseline_logits"][index] = baseline_logits
        arrays["natural_game_logits"][index] = game_logits
        arrays["baseline_mean_ablation_logits"][index] = baseline_patched
        arrays["game_baseline_insertion_logits"][index] = game_patched
        for component_index, layer in enumerate(LAYERS):
            arrays["baseline_component_outputs"][index, component_index] = (
                baseline_source[f"decision__mixer_l{layer}"][0].numpy()
            )
            arrays["baseline_post_block_residuals"][index, component_index] = (
                block_values[layer].numpy()
            )
        arrays["completed"][index] = True
        if first_audit is None:
            first_audit = {
                "question_id": trial.question_id,
                "baseline_prompt_hash": prompt_hash(baseline_prompt),
                "game_prompt_hash": prompt_hash(game_prompt),
                "baseline_prompt": baseline_prompt,
                "game_prompt": game_prompt,
            }
            (output / "prompt_audit.json").write_text(
                json.dumps(first_audit, indent=2, sort_keys=True)
            )
        done = int(np.sum(arrays["completed"]))
        if done == 1 or done % 5 == 0 or done == len(confirmation_trials):
            atomic_save_npz(result_path, **arrays)
            print(f"Baseline mixer confirmation: {done}/{len(confirmation_trials)}", flush=True)

    if not np.all(arrays["completed"]):
        raise RuntimeError("Confirmation run ended with incomplete questions")
    if "baseline_immediate_jlens_write" not in arrays:
        arrays["baseline_immediate_jlens_write"] = _compute_immediate_jlens_writes(
            arrays,
            model,
            processor,
            parts,
            config,
            lens_repo,
            lens_filename,
        )
    atomic_save_npz(result_path, **arrays)
    metadata = {
        "config": config.as_dict(),
        "discovery_plan": str(discovery_plan_path),
        "confirmation_plan": str(confirmation_plan_path),
        "n_discovery": len(discovery_ids),
        "n_confirmation": len(confirmation_ids),
        "discovery_baseline_letter_counts": discovery["letter_counts"].tolist(),
        "components": [
            {"label": label, "zero_based_layer": layer, "one_based_block": layer + 1}
            for label, layer in zip(LABELS, LAYERS)
        ],
        "mean_ablation": (
            "Replace the final-position Baseline mixer output with the equal-Baseline-letter "
            "mean estimated on the disjoint 251-question discovery set."
        ),
        "baseline_into_game": (
            "Replace the final-position Game mixer output with the paired same-question "
            "natural Baseline output."
        ),
        "immediate_write": (
            "JLens finite-difference direct attribution: compare the transported post-block "
            "Baseline residual with the same residual minus that mixer's output, holding "
            "downstream computation fixed."
        ),
        "batch_control": (
            "Each patched eight-row forward includes condition-matched unpatched control "
            "rows and is recentered on single-trial natural logits."
        ),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the ordinary Baseline function of Mixers 56 and 63"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--confirmation-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discovery-batch-size", type=int, default=8)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    args = parser.parse_args()
    run(
        args.config,
        args.discovery_plan,
        args.confirmation_plan,
        args.output,
        args.discovery_batch_size,
        args.lens_repo,
        args.lens_filename,
    )


if __name__ == "__main__":
    main()
