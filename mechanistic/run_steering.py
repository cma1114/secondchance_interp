from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .io import atomic_save_npz, json_array
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
    variant_layout,
)
from .prompts import build_messages, load_trials, prompt_hash
from .steering import ResidualSteerer, build_schedule
from .steering_config import SteeringConfig


def steering_shard(output_dir: str | Path, scenario_id: str, question_id: str) -> Path:
    return Path(output_dir) / "shards" / scenario_id / f"{question_id}.npz"


def _reference_lookup(direction_data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    qids = [str(value) for value in direction_data["reference_question_ids"].tolist()]
    logits = direction_data["reference_canonical_logits"].astype(np.float64)
    return {qid: logits[index] for index, qid in enumerate(qids)}


def run(
    config_path: str | Path,
    max_questions_override: int | None = None,
    smoke_questions: int = 0,
) -> dict:
    import torch
    import transformers

    steering_config = SteeringConfig.load(config_path)
    base_config = ExperimentConfig.load(steering_config.base_config_path)
    if base_config.batch_size != 1:
        raise ValueError("The steering runner currently requires base_config.batch_size=1")
    with np.load(steering_config.directions_path, allow_pickle=False) as direction_data:
        directions = torch.from_numpy(direction_data["directions"].astype(np.float32))
        controls = torch.from_numpy(direction_data["control_directions"].astype(np.float32))
        mean_gap = direction_data["mean_gap"].astype(np.float64)
        question_ids = [str(value) for value in direction_data["intervention_question_ids"].tolist()]
        reference = _reference_lookup(direction_data)
        direction_metadata = json.loads(str(direction_data["metadata"].item()))
    max_questions = max_questions_override
    if max_questions is None:
        max_questions = steering_config.max_questions
    if max_questions is not None:
        question_ids = question_ids[:max_questions]

    trials = load_trials(
        base_config.manifest_path,
        base_config.baseline_results_path,
        question_ids=question_ids,
    )
    model, processor, parts = load_model_and_processor(base_config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, base_config.answer_variants)
    selected_ids, variant_meta = variant_layout(resolved)
    canonical_positions = [
        next(index for index, value in enumerate(variant_meta) if value["letter"] == letter and value["text"] == letter)
        for letter in "ABCD"
    ]
    if directions.shape != (len(parts.layers) + 1, parts.embedding.weight.shape[-1]):
        raise ValueError(
            f"Direction shape {tuple(directions.shape)} does not match model "
            f"{(len(parts.layers) + 1, parts.embedding.weight.shape[-1])}"
        )

    schedule = build_schedule(
        steering_config.conditions,
        steering_config.scan_readouts,
        steering_config.scan_doses,
        steering_config.detailed_readout,
        steering_config.detailed_doses,
        steering_config.control_readout,
        steering_config.control_doses,
    )
    output_dir = Path(steering_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "steering_config": steering_config.as_dict(),
        "base_config": base_config.as_dict(),
        "direction_metadata": direction_metadata,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "resolved_answer_tokens": resolved,
        "variant_layout": variant_meta,
        "n_text_layers": len(parts.layers),
        "n_questions": len(trials),
        "n_scenarios": len(schedule),
        "schedule": [spec.__dict__ | {"scenario_id": spec.scenario_id} for spec in schedule],
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True))

    if smoke_questions < 0:
        raise ValueError("smoke_questions cannot be negative")
    smoke_questions = min(smoke_questions, len(trials))
    phases = []
    if smoke_questions:
        phases.append(("smoke", trials[:smoke_questions]))
    phases.append(("full", trials[smoke_questions:]))

    completed = 0
    for phase_name, phase_trials in phases:
        if not phase_trials:
            continue
        print(f"Starting {phase_name} phase with {len(phase_trials)} questions", flush=True)
        for scenario_index, spec in enumerate(schedule, start=1):
            pending = [
                trial for trial in phase_trials
                if not steering_shard(output_dir, spec.scenario_id, trial.question_id).exists()
            ]
            print(
                f"[{scenario_index}/{len(schedule)}] {spec.scenario_id}: "
                f"{len(pending)} pending / {len(phase_trials)} phase questions",
                flush=True,
            )
            for trial_index, trial in enumerate(pending, start=1):
                messages = build_messages(trial.question, spec.condition, base_config.prompt_mode)
                prompt = render_chat(processor, messages, base_config.disable_thinking)
                input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, [prompt])
                steerer = None
                if spec.direction_kind != "none":
                    assert spec.readout is not None
                    direction = directions[spec.readout] if spec.direction_kind == "feedback" else controls[spec.readout]
                    steerer = ResidualSteerer(
                        parts,
                        spec.readout,
                        last_indices,
                        direction,
                        float(mean_gap[spec.readout]),
                        spec.dose,
                    )
                try:
                    with torch.inference_mode():
                        device = model_input_device(parts)
                        kwargs = {
                            "input_ids": input_ids.to(device),
                            "attention_mask": attention_mask.to(device),
                            "use_cache": False,
                            "return_dict": True,
                        }
                        try:
                            output = model(**kwargs, logits_to_keep=1)
                        except TypeError:
                            output = model(**kwargs)
                finally:
                    if steerer is not None:
                        steerer.close()

                model_logits = output.logits.detach().float().cpu()
                final_logits = model_logits[0, 0] if model_logits.shape[1] == 1 else model_logits[0, last_indices[0]]
                variant_logits = final_logits[selected_ids].numpy()
                canonical_logits = variant_logits[canonical_positions]
                top_id = int(final_logits.argmax())
                top_text = tokenizer.decode([top_id])
                reference_error = None
                reference_relative_error = None
                if spec.direction_kind == "none":
                    condition_index = {"baseline": 0, "incorrect": 1, "neutral": 2}[spec.condition]
                    expected = reference[trial.question_id][condition_index]
                    reference_error = float(np.max(np.abs(canonical_logits - expected)))
                    reference_scale = max(float(np.max(np.abs(expected))), 1.0)
                    reference_relative_error = reference_error / reference_scale
                    if (
                        reference_error > steering_config.unsteered_abs_tolerance
                        and reference_relative_error > steering_config.unsteered_rel_tolerance
                    ):
                        raise RuntimeError(
                            f"Unsteered reference validation failed for {spec.condition}/{trial.question_id}: "
                            f"abs={reference_error}, rel={reference_relative_error}"
                        )

                metadata = {
                "question_id": trial.question_id,
                "condition": spec.condition,
                "scenario_id": spec.scenario_id,
                "direction_kind": spec.direction_kind,
                "readout": spec.readout,
                "dose": spec.dose,
                "activation_scale": None if spec.readout is None else float(mean_gap[spec.readout]),
                "baseline_answer": trial.baseline_answer,
                "baseline_correct": trial.baseline_correct,
                "correct_answer": trial.question["correct_answer"],
                "prompt_hash": prompt_hash(prompt),
                "prompt_length": int(last_indices[0] + 1),
                "canonical_ad_choice": "ABCD"[int(np.argmax(canonical_logits))],
                "full_vocab_top_token_id": top_id,
                "full_vocab_top_token": top_text,
                "valid_ad_top_token": top_text.strip() in set("ABCD"),
                "unsteered_reference_max_abs_error": reference_error,
                "unsteered_reference_max_rel_error": reference_relative_error,
                }
                atomic_save_npz(
                    steering_shard(output_dir, spec.scenario_id, trial.question_id),
                    canonical_logits=canonical_logits.astype(np.float32),
                    variant_logits=variant_logits.astype(np.float32),
                    metadata=json_array(metadata),
                )
                completed += 1
                if trial_index % 10 == 0 or trial_index == len(pending):
                    print(
                        f"  {spec.scenario_id}: saved {trial_index}/{len(pending)} pending trials",
                        flush=True,
                    )

    return {"output_dir": str(output_dir), "new_shards": completed, "n_scenarios": len(schedule), "n_questions": len(trials)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal feedback-direction steering")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument(
        "--smoke-questions",
        type=int,
        default=0,
        help="Run this many questions through the complete schedule before continuing",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.max_questions, args.smoke_questions), indent=2))


if __name__ == "__main__":
    main()
