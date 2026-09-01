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
from .run_jlens_exclusion_bridge_intervention import (
    EXCLUSION_CONCEPTS,
    SourcePatcher,
    _family_score,
    _resolve_concepts,
)


# Post-block readouts 41--63. Readout 64 is omitted because an intervention
# after the final block cannot affect the already-computed decision position.
SOURCE_LAYERS = tuple(range(40, 63))
WINDOWS = {
    "L41_63": SOURCE_LAYERS,
    "L49_63": tuple(range(48, 63)),
}


def _scenarios() -> tuple[str, ...]:
    layerwise = []
    for layer in SOURCE_LAYERS:
        displayed = layer + 1
        layerwise.extend((
            f"exclude_neutral_into_game_L{displayed}",
            f"exclude_game_into_neutral_L{displayed}",
        ))
    combined = []
    for window in WINDOWS:
        combined.extend((
            f"exclude_neutral_into_game_{window}",
            f"exclude_game_into_neutral_{window}",
        ))
    return ("natural_game", "natural_neutral", *layerwise, *combined)


def _scenario_target(scenario: str) -> str:
    if scenario == "natural_game" or "into_game" in scenario:
        return "incorrect"
    if scenario == "natural_neutral" or "into_neutral" in scenario:
        return "neutral"
    raise ValueError(scenario)


def _scenario_layers(scenario: str) -> tuple[int, ...]:
    for name, layers in WINDOWS.items():
        if scenario.endswith(name):
            return layers
    marker = scenario.rsplit("_L", 1)
    if len(marker) == 2 and marker[1].isdigit():
        layer = int(marker[1]) - 1
        if layer in SOURCE_LAYERS:
            return (layer,)
    return ()


def run(
    config_path: Path,
    jlens_root: Path,
    output: Path,
    lens_repo: str,
    lens_filename: str,
    max_questions: int | None = None,
) -> None:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    position_meta = json.loads((jlens_root / "position_residuals_metadata.json").read_text())
    qids = position_meta["question_ids"]
    if max_questions is not None:
        qids = qids[:max_questions]
    anchors = position_meta["anchors"]
    feedback_index = anchors.index("feedback_end")
    position_residuals = np.load(jlens_root / "position_residuals.npy", mmap_mode="r")
    position_conditions = tuple(position_meta.get("conditions", ("incorrect", "neutral")))
    if position_conditions != ("incorrect", "neutral"):
        raise ValueError(f"Unexpected position conditions: {position_conditions}")

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
    exclusion_rows = parts.output_head.weight.detach()[exclusion_ids].float()

    directions: dict[int, torch.Tensor] = {}
    coordinates = {
        "incorrect": np.empty((len(SOURCE_LAYERS), len(qids)), dtype=np.float32),
        "neutral": np.empty((len(SOURCE_LAYERS), len(qids)), dtype=np.float32),
    }
    direction_diagnostics = []
    for li, layer in enumerate(SOURCE_LAYERS):
        game_values = np.asarray(
            position_residuals[0, :len(qids), layer, feedback_index], dtype=np.float32
        )
        reference = torch.from_numpy(game_values.mean(axis=0).copy()).to(
            device, dtype=torch.float32
        ).requires_grad_(True)
        jacobian = checkpoint["J"][layer].to(device, dtype=torch.float32)
        transported = reference @ jacobian.T
        normed = parts.final_norm(transported.to(parts.final_norm.weight.dtype)).float()
        logits = normed @ exclusion_rows.T
        score = _family_score(logits, exclusion_groups)
        gradient = torch.autograd.grad(score, reference)[0]
        direction = (gradient / gradient.norm()).detach().cpu()
        directions[layer] = direction
        direction_np = direction.numpy().astype(np.float32)
        for ci, condition in enumerate(("incorrect", "neutral")):
            values = np.asarray(
                position_residuals[ci, :len(qids), layer, feedback_index], dtype=np.float32
            )
            coordinates[condition][li] = values @ direction_np
        delta = coordinates["incorrect"][li] - coordinates["neutral"][li]
        direction_diagnostics.append({
            "layer": layer + 1,
            "mean_game_coordinate": float(coordinates["incorrect"][li].mean()),
            "mean_neutral_coordinate": float(coordinates["neutral"][li].mean()),
            "mean_paired_game_minus_neutral": float(delta.mean()),
            "sd_paired_game_minus_neutral": float(delta.std(ddof=1)),
        })
        del jacobian, reference, transported, normed, logits, score, gradient

    scenarios = _scenarios()
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metadata.json").write_text(json.dumps({
        "config": config.as_dict(),
        "question_ids": qids,
        "source_position": "period ending the complete user feedback sentence",
        "source_layers": [layer + 1 for layer in SOURCE_LAYERS],
        "scenarios": list(scenarios),
        "direction_definition": (
            "layer-specific unit gradient of the JLens exclusion-family score "
            "at the mean Game feedback-end residual"
        ),
        "intervention": (
            "for each patched layer, replace the target condition's one-dimensional "
            "coordinate with the paired other-condition natural coordinate"
        ),
        "direction_diagnostics": direction_diagnostics,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
    }, indent=2, sort_keys=True))

    def forward(condition: str, trial, scenario: str, qi: int):
        messages = build_messages(trial.question, condition, config.prompt_mode)
        prompt = render_chat(processor, messages, config.disable_thinking)
        annotated_ids, spans = attention_span_indices(tokenizer, prompt, condition, trial.question)
        feedback_position = spans["feedback_sentence"][-1]
        input_ids, attention_mask, _ = tokenize_batch(tokenizer, [prompt])
        if annotated_ids != input_ids[0, :len(annotated_ids)].tolist():
            raise RuntimeError("Tokenization disagreement")

        patch_layers = _scenario_layers(scenario)
        patcher = None
        if patch_layers:
            source_condition = "neutral" if condition == "incorrect" else "incorrect"
            targets = {
                layer: torch.tensor(
                    coordinates[source_condition][SOURCE_LAYERS.index(layer), qi],
                    device=device,
                    dtype=torch.float32,
                )
                for layer in patch_layers
            }
            patcher = SourcePatcher(
                parts.layers, patch_layers, feedback_position,
                "direction", directions, targets, None,
            )
        try:
            with torch.inference_mode():
                kwargs = {
                    "input_ids": input_ids.to(device),
                    "attention_mask": attention_mask.to(device),
                    "use_cache": False,
                    "return_dict": True,
                }
                try:
                    result = model(**kwargs, logits_to_keep=1)
                except TypeError:
                    result = model(**kwargs)
        finally:
            if patcher is not None:
                patcher.close()
        logits = result.logits.detach().float().cpu()[0, -1, answer_ids].numpy()
        return logits, prompt, feedback_position

    for completed, qid in enumerate(qids, 1):
        qi = qid_index[qid]
        trial = trial_by_qid[qid]
        for scenario in scenarios:
            destination = shard_path(output, scenario, qid)
            if destination.exists():
                continue
            condition = _scenario_target(scenario)
            logits, prompt, feedback_position = forward(condition, trial, scenario, qi)
            atomic_save_npz(
                destination,
                final_canonical_logits=logits.astype(np.float32),
                metadata=json_array({
                    "question_id": qid,
                    "scenario": scenario,
                    "condition": condition,
                    "feedback_position": feedback_position,
                    "prompt_hash": prompt_hash(prompt),
                    "patched_layers": [layer + 1 for layer in _scenario_layers(scenario)],
                }),
            )
        if completed == 1 or completed % 10 == 0 or completed == len(qids):
            print(f"layerwise exclusion intervention: {completed}/{len(qids)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jlens-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    run(
        args.config, args.jlens_root, args.output, args.lens_repo,
        args.lens_filename, args.max_questions,
    )


if __name__ == "__main__":
    main()
