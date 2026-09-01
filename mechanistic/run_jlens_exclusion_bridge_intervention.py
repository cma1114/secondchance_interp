from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .attention_spans import attention_span_indices
from .config import ExperimentConfig
from .io import atomic_save_npz, json_array, shard_path
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages, load_trials, prompt_hash


SOURCE_LAYERS = tuple(range(40, 48))  # displayed post-block readouts 41--48
ALL_SOURCE_LAYERS = tuple(range(64))
SOURCE_WINDOWS = {
    "L1_16": tuple(range(0, 16)),
    "L17_32": tuple(range(16, 32)),
    "L33_40": tuple(range(32, 40)),
    "L41_48": SOURCE_LAYERS,
    "L49_64": tuple(range(48, 64)),
    "all_layers": ALL_SOURCE_LAYERS,
}
TARGET_LAYERS = tuple(range(43, 50))  # displayed post-block readouts 44--50
EXCLUSION_CONCEPTS = (
    "exclude", "excludes", "excluded", "excluding",
    "restrict", "restricted", "restriction", "restrictions",
    "ban", "banned", "banning", "reject", "rejected", "rejection",
    "eliminate", "eliminated",
)
ALTERNATIVE_CONCEPTS = (
    "instead", "other", "another", "alternative",
    "change", "changed", "changing", "retry", "different",
)


def _resolve_concepts(tokenizer, concepts):
    ids: list[int] = []
    groups: list[list[int]] = []
    seen: dict[int, int] = {}
    for concept in concepts:
        group = []
        for text in (concept, " " + concept, concept.capitalize(), " " + concept.capitalize()):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) != 1:
                continue
            token_id = int(encoded[0])
            if token_id not in seen:
                seen[token_id] = len(ids)
                ids.append(token_id)
            group.append(seen[token_id])
        group = sorted(set(group))
        if group:
            groups.append(group)
    return ids, groups


def _family_score(logits, groups):
    import torch

    return torch.stack([torch.logsumexp(logits[..., group], dim=-1) for group in groups], dim=-1).mean(dim=-1)


class SourcePatcher:
    def __init__(self, layers, patch_layers, position: int, mode: str, directions, targets, full_targets):
        self.handles = []
        for layer in patch_layers:
            self.handles.append(layers[layer].register_forward_hook(
                self._hook(layer, position, mode, directions, targets, full_targets)
            ))

    @staticmethod
    def _hook(layer, position, mode, directions, targets, full_targets):
        def patch(_module, _inputs, output):
            import torch

            hidden = output[0] if isinstance(output, (tuple, list)) else output
            updated = hidden.clone()
            if mode == "direction":
                direction = directions[layer].to(updated.device, dtype=updated.dtype)
                value = updated[0, position]
                coordinate = torch.dot(value, direction)
                updated[0, position] = value + (targets[layer] - coordinate).to(value.dtype) * direction
            elif mode == "full":
                updated[0, position] = full_targets[layer].to(updated.device, dtype=updated.dtype)
            else:
                raise ValueError(mode)
            if isinstance(output, tuple):
                return (updated,) + output[1:]
            if isinstance(output, list):
                return [updated] + list(output[1:])
            return updated
        return patch

    def close(self):
        for handle in self.handles:
            handle.remove()


class DecisionCollector:
    def __init__(self, layers):
        self.values = {}
        self.handles = [layers[layer].register_forward_hook(self._hook(layer)) for layer in TARGET_LAYERS]

    def _hook(self, layer):
        def collect(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            self.values[layer] = hidden[0, -1].detach()
        return collect

    def close(self):
        for handle in self.handles:
            handle.remove()


def run(config_path: Path, jlens_root: Path, output: Path, lens_repo: str, lens_filename: str) -> None:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    position_meta = json.loads((jlens_root / "position_residuals_metadata.json").read_text())
    qids = position_meta["question_ids"]
    anchors = position_meta["anchors"]
    feedback_index = anchors.index("feedback_end")
    position_residuals = np.load(jlens_root / "position_residuals.npy", mmap_mode="r")
    trials = load_trials(config.manifest_path, config.baseline_results_path, qids)
    trial_by_qid = {trial.question_id: trial for trial in trials}
    qid_index = {qid: index for index, qid in enumerate(qids)}

    lens_path = hf_hub_download(repo_id=lens_repo, filename=lens_filename)
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    device = model_input_device(parts)
    resolved_answers = resolve_answer_tokens(tokenizer, config.answer_variants)
    answer_ids = [resolved_answers[letter][0][1] for letter in "ABCD"]
    exclusion_ids, exclusion_groups = _resolve_concepts(tokenizer, EXCLUSION_CONCEPTS)
    alternative_ids, alternative_groups = _resolve_concepts(tokenizer, ALTERNATIVE_CONCEPTS)
    exclusion_rows = parts.output_head.weight.detach()[exclusion_ids].float()
    alternative_rows = parts.output_head.weight.detach()[alternative_ids].float()

    directions = {}
    coordinate_targets = {"incorrect": {}, "neutral": {}}
    for layer in SOURCE_LAYERS:
        reference = torch.from_numpy(
            np.asarray(position_residuals[0, :, layer, feedback_index]).mean(axis=0).copy()
        ).to(device, dtype=torch.float32).requires_grad_(True)
        jacobian = checkpoint["J"][layer].to(device, dtype=torch.float32)
        transported = reference @ jacobian.T
        normed = parts.final_norm(transported.to(parts.final_norm.weight.dtype)).float()
        logits = normed @ exclusion_rows.T
        score = _family_score(logits, exclusion_groups)
        gradient = torch.autograd.grad(score, reference)[0]
        directions[layer] = (gradient / gradient.norm()).detach().cpu()
        direction = directions[layer].numpy().astype(np.float32)
        for ci, condition in enumerate(("incorrect", "neutral")):
            values = np.asarray(position_residuals[ci, :, layer, feedback_index], dtype=np.float32)
            coordinate_targets[condition][layer] = values @ direction
        del jacobian, reference, transported, normed, logits, score, gradient

    target_maps = {layer: checkpoint["J"][layer].to(device, dtype=torch.float16) for layer in TARGET_LAYERS}
    scenarios = (
        "natural_game", "natural_neutral",
        "exclude_neutral_into_game_L41_48", "exclude_game_into_neutral_L41_48",
        "full_neutral_into_game_L41_48", "full_game_into_neutral_L41_48",
        "full_neutral_into_game_all_layers", "full_game_into_neutral_all_layers",
        "full_neutral_into_game_L1_16", "full_game_into_neutral_L1_16",
        "full_neutral_into_game_L17_32", "full_game_into_neutral_L17_32",
        "full_neutral_into_game_L33_40", "full_game_into_neutral_L33_40",
        "full_neutral_into_game_L49_64", "full_game_into_neutral_L49_64",
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metadata.json").write_text(json.dumps({
        "config": config.as_dict(), "question_ids": qids,
        "source_position": "period ending the complete user feedback sentence",
        "source_layers": [layer + 1 for layer in SOURCE_LAYERS],
        "target_position": "final decision token", "target_layers": [layer + 1 for layer in TARGET_LAYERS],
        "scenarios": list(scenarios),
        "direction_definition": "gradient of JLens exclusion-family score at the mean Game feedback-end residual",
        "direction_patch": "set the one-dimensional source coordinate to the paired other-condition natural coordinate at each source layer",
        "full_patch": "replace the complete feedback-end post-block residual with the paired other-condition natural residual",
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "platform": platform.platform(), "resolved_model_commit": getattr(model.config, "_commit_hash", None),
    }, indent=2, sort_keys=True))

    def forward(condition, trial, scenario, qi):
        messages = build_messages(trial.question, condition, config.prompt_mode)
        prompt = render_chat(processor, messages, config.disable_thinking)
        annotated_ids, spans = attention_span_indices(tokenizer, prompt, condition, trial.question)
        feedback_position = spans["feedback_sentence"][-1]
        input_ids, attention_mask, _ = tokenize_batch(tokenizer, [prompt])
        if annotated_ids != input_ids[0, :len(annotated_ids)].tolist():
            raise RuntimeError("Tokenization disagreement")
        patcher = None
        if scenario.startswith("exclude_"):
            source_condition = "neutral" if condition == "incorrect" else "incorrect"
            targets = {
                layer: torch.tensor(coordinate_targets[source_condition][layer][qi], device=device, dtype=torch.float32)
                for layer in SOURCE_LAYERS
            }
            patcher = SourcePatcher(parts.layers, SOURCE_LAYERS, feedback_position, "direction", directions, targets, None)
        elif scenario.startswith("full_"):
            source_ci = 1 if condition == "incorrect" else 0
            window = next((name for name in SOURCE_WINDOWS if scenario.endswith(name)), None)
            if window is None:
                raise ValueError(f"Cannot resolve source window from {scenario}")
            patch_layers = SOURCE_WINDOWS[window]
            full_targets = {
                layer: torch.from_numpy(np.asarray(position_residuals[source_ci, qi, layer, feedback_index]).copy())
                for layer in patch_layers
            }
            patcher = SourcePatcher(parts.layers, patch_layers, feedback_position, "full", directions, None, full_targets)
        collector = DecisionCollector(parts.layers)
        try:
            with torch.inference_mode():
                kwargs = {
                    "input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device),
                    "use_cache": False, "return_dict": True,
                }
                try:
                    result = model(**kwargs, logits_to_keep=1)
                except TypeError:
                    result = model(**kwargs)
        finally:
            collector.close()
            if patcher is not None:
                patcher.close()
        final_logits = result.logits.detach().float().cpu()[0, -1, answer_ids].numpy()
        target_scores = []
        with torch.inference_mode():
            for layer in TARGET_LAYERS:
                residual = collector.values[layer].to(device, dtype=torch.float16)
                transported = residual @ target_maps[layer].T
                normed = parts.final_norm(transported.to(parts.final_norm.weight.dtype)).float()
                logits = normed @ alternative_rows.T
                target_scores.append(float(_family_score(logits, alternative_groups).cpu()))
        return final_logits, np.asarray(target_scores, dtype=np.float32), prompt, feedback_position

    for completed, qid in enumerate(qids, 1):
        qi = qid_index[qid]
        trial = trial_by_qid[qid]
        for scenario in scenarios:
            destination = shard_path(output, scenario, qid)
            if destination.exists():
                continue
            condition = "neutral" if "neutral" in scenario.split("_")[-1] or scenario == "natural_neutral" else "incorrect"
            # Patched scenario names encode the target after "into".
            if "into_neutral" in scenario:
                condition = "neutral"
            elif "into_game" in scenario:
                condition = "incorrect"
            logits, target_scores, prompt, feedback_position = forward(condition, trial, scenario, qi)
            atomic_save_npz(
                destination,
                final_canonical_logits=logits.astype(np.float32),
                decision_alternative_scores=target_scores,
                metadata=json_array({
                    "question_id": qid, "scenario": scenario, "condition": condition,
                    "feedback_position": feedback_position, "prompt_hash": prompt_hash(prompt),
                }),
            )
        if completed == 1 or completed % 10 == 0 or completed == len(qids):
            print(f"exclusion bridge intervention: {completed}/{len(qids)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jlens-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument("--lens-filename", default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")
    args = parser.parse_args()
    run(args.config, args.jlens_root, args.output, args.lens_repo, args.lens_filename)


if __name__ == "__main__":
    main()
