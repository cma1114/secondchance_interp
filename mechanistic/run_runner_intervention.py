from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

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
from .runner_intervention import CpuAnswerLens, ReadoutAdd, ReadoutCapture
from .runner_intervention_config import RunnerInterventionConfig


def _chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _forward(model, parts, input_ids, attention_mask):
    import torch

    device = model_input_device(parts)
    with torch.inference_mode():
        kwargs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "use_cache": False,
            "return_dict": True,
        }
        try:
            return model(**kwargs, logits_to_keep=1)
        except TypeError:
            return model(**kwargs)


def _final_ad(output, canonical_ids: list[int]) -> np.ndarray:
    return output.logits[:, -1, canonical_ids].detach().float().cpu().numpy()


def _top_ids(output) -> np.ndarray:
    return output.logits[:, -1].argmax(dim=-1).detach().cpu().numpy()


def _scenario_id(condition: str, kind: str, strength: float | None = None) -> str:
    prefix = {"incorrect": "game", "neutral": "neutral", "baseline": "baseline"}[condition]
    if strength is None:
        return f"{prefix}_{kind}"
    return f"{prefix}_{kind}_x{strength:g}".replace(".", "p")


def run(config: RunnerInterventionConfig) -> None:
    import torch
    import transformers

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        config.question_ids,
        config.max_questions,
    )
    with np.load(config.signal_path, allow_pickle=False) as artifact:
        signal_qids = [str(value) for value in artifact["question_ids"]]
        signal_values = artifact["signal"].astype(np.float32)
        order_values = artifact["order"].astype(np.int64)
    signal_index = {qid: index for index, qid in enumerate(signal_qids)}
    if set(signal_index) != {trial.question_id for trial in trials}:
        raise ValueError("Signal artifact and requested trials do not contain identical IDs")

    model, processor, parts = load_model_and_processor(config)
    if config.intervention_readout > len(parts.layers):
        raise ValueError("Intervention readout exceeds model depth")
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    lens = CpuAnswerLens(parts, canonical_ids)
    scenario_ids = []
    for condition in ("incorrect", "neutral"):
        scenario_ids.append(_scenario_id(condition, "natural"))
        action = "remove" if condition == "incorrect" else "add"
        for strength in config.strengths:
            scenario_ids.append(_scenario_id(condition, action, strength))
        for control_strength in (0.5, 1.0):
            scenario_ids.extend(
                [
                    _scenario_id(condition, f"{action}_rank3_control", control_strength),
                    _scenario_id(condition, f"{action}_orthogonal_control", control_strength),
                ]
            )
        scenario_ids.append(_scenario_id(condition, f"{action}_early_control", 1.0))
    scenario_ids.extend(
        [
            _scenario_id("baseline", "natural"),
            _scenario_id("baseline", "add", 1.0),
        ]
    )

    metadata = {
        "config": config.as_dict(),
        "scenarios": scenario_ids,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "resolved_answer_tokens": resolved,
        "n_text_layers": len(parts.layers),
        "intervention": (
            "calibrated final-position residual update changing the baseline-rank "
            "candidate's native A-D lens contrast by the specified signal"
        ),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    for batch_number, batch in enumerate(_chunks(trials, config.batch_size)):
        ids = [trial.question_id for trial in batch]
        positions = np.asarray([signal_index[qid] for qid in ids])
        signal = torch.from_numpy(signal_values[positions])
        order = order_values[positions]
        runner = order[:, 1]
        rank3 = order[:, 2]

        for condition in config.conditions:
            natural_id = _scenario_id(condition, "natural")
            pending_natural = [
                not shard_path(output_dir, natural_id, qid).exists() for qid in ids
            ]
            action = "remove" if condition == "incorrect" else "add"
            relevant = [natural_id]
            if condition in ("incorrect", "neutral"):
                relevant += [
                    *[_scenario_id(condition, action, strength) for strength in config.strengths],
                    *[
                        _scenario_id(condition, f"{action}_{control}", strength)
                        for strength in (0.5, 1.0)
                        for control in ("rank3_control", "orthogonal_control")
                    ],
                    _scenario_id(condition, f"{action}_early_control", 1.0),
                ]
            else:
                relevant += [_scenario_id("baseline", "add", 1.0)]
            if all(
                shard_path(output_dir, scenario, qid).exists()
                for scenario in relevant
                for qid in ids
            ):
                continue

            prompts = [
                render_chat(
                    processor,
                    build_messages(trial.question, condition, config.prompt_mode),
                    config.disable_thinking,
                )
                for trial in batch
            ]
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            if condition in ("incorrect", "neutral"):
                with (
                    ReadoutCapture(parts, config.intervention_readout, last_indices) as capture,
                    ReadoutCapture(parts, config.early_control_readout, last_indices) as early_capture,
                ):
                    natural = _forward(model, parts, input_ids, attention_mask)
                early_residual = early_capture.value
            else:
                with ReadoutCapture(parts, config.intervention_readout, last_indices) as capture:
                    natural = _forward(model, parts, input_ids, attention_mask)
                early_residual = None
            residual = capture.value
            natural_logits = _final_ad(natural, canonical_ids)
            natural_top_ids = _top_ids(natural)

            base_meta = []
            for index, trial in enumerate(batch):
                base_meta.append(
                    {
                        "question_id": trial.question_id,
                        "condition": condition,
                        "baseline_answer": trial.baseline_answer,
                        "baseline_correct": trial.baseline_correct,
                        "correct_answer": trial.question["correct_answer"],
                        "prompt_hash": prompt_hash(prompts[index]),
                        "prompt_length": int(attention_mask[index].sum()),
                        "winner_letter": "ABCD"[order[index, 0]],
                        "runner_letter": "ABCD"[runner[index]],
                        "rank3_letter": "ABCD"[rank3[index]],
                        "rank4_letter": "ABCD"[order[index, 3]],
                        "candidate_signal": float(signal[index]),
                    }
                )
            for index, trial in enumerate(batch):
                if pending_natural[index]:
                    atomic_save_npz(
                        shard_path(output_dir, natural_id, trial.question_id),
                        final_canonical_logits=natural_logits[index].astype(np.float32),
                        metadata=json_array(
                            {
                                **base_meta[index],
                                "scenario_id": natural_id,
                                "full_vocab_top_token_id": int(natural_top_ids[index]),
                                "full_vocab_top_token": tokenizer.decode([int(natural_top_ids[index])]),
                            }
                        ),
                    )

            sign = -1.0 if condition == "incorrect" else 1.0
            if condition == "baseline":
                sign = 1.0
            interventions = []
            if condition in ("incorrect", "neutral"):
                for strength in config.strengths:
                    scenario = _scenario_id(condition, action, strength)
                    if all(shard_path(output_dir, scenario, qid).exists() for qid in ids):
                        continue
                    delta, achieved = lens.calibrated_delta(
                        residual,
                        runner,
                        sign * float(strength) * signal,
                        config.calibration_steps,
                    )
                    interventions.append(
                        (
                            scenario,
                            config.intervention_readout,
                            delta,
                            achieved,
                            sign * float(strength) * signal,
                        )
                    )
                for control_strength in (0.5, 1.0):
                    rank3_scenario = _scenario_id(
                        condition, f"{action}_rank3_control", control_strength
                    )
                    if not all(
                        shard_path(output_dir, rank3_scenario, qid).exists() for qid in ids
                    ):
                        rank3_delta, rank3_achieved = lens.calibrated_delta(
                            residual,
                            rank3,
                            sign * control_strength * signal,
                            config.calibration_steps,
                        )
                        interventions.append(
                            (
                                rank3_scenario,
                                config.intervention_readout,
                                rank3_delta,
                                rank3_achieved,
                                sign * control_strength * signal,
                            )
                        )
                    orthogonal_scenario = _scenario_id(
                        condition, f"{action}_orthogonal_control", control_strength
                    )
                    if not all(
                        shard_path(output_dir, orthogonal_scenario, qid).exists()
                        for qid in ids
                    ):
                        matched_primary, _ = lens.calibrated_delta(
                            residual,
                            runner,
                            sign * control_strength * signal,
                            config.calibration_steps,
                        )
                        orthogonal = lens.answer_orthogonal_control(
                            residual,
                            matched_primary,
                            config.seed
                            + 1000 * batch_number
                            + 10 * int(control_strength * 10)
                            + (0 if sign < 0 else 1),
                        )
                        interventions.append(
                            (
                                orthogonal_scenario,
                                config.intervention_readout,
                                orthogonal,
                                torch.zeros(len(batch)),
                                torch.zeros(len(batch)),
                            )
                        )
                early_scenario = _scenario_id(condition, f"{action}_early_control", 1.0)
                if not all(
                    shard_path(output_dir, early_scenario, qid).exists() for qid in ids
                ):
                    early_delta, early_achieved = lens.calibrated_delta(
                        early_residual,
                        runner,
                        sign * signal,
                        config.calibration_steps,
                    )
                    interventions.append(
                        (
                            early_scenario,
                            config.early_control_readout,
                            early_delta,
                            early_achieved,
                            sign * signal,
                        )
                    )
            else:
                primary_delta, primary_achieved = lens.calibrated_delta(
                    residual,
                    runner,
                    signal,
                    config.calibration_steps,
                )
                interventions.append(
                    (
                        _scenario_id("baseline", "add", 1.0),
                        config.intervention_readout,
                        primary_delta,
                        primary_achieved,
                        signal,
                    )
                )

            for scenario, readout, delta, achieved, requested in interventions:
                if all(shard_path(output_dir, scenario, qid).exists() for qid in ids):
                    continue
                with ReadoutAdd(parts, readout, last_indices, delta) as intervention:
                    output = _forward(model, parts, input_ids, attention_mask)
                final_logits = _final_ad(output, canonical_ids)
                top_ids = _top_ids(output)
                delta_norm = delta.float().norm(dim=-1).numpy()
                for index, trial in enumerate(batch):
                    path = shard_path(output_dir, scenario, trial.question_id)
                    if path.exists():
                        continue
                    row_meta = {
                        **base_meta[index],
                        "scenario_id": scenario,
                        "intervention_readout": readout,
                        "achieved_native_lens_contrast_change": float(achieved[index]),
                        "requested_native_lens_contrast_change": float(requested[index]),
                        "residual_delta_l2": float(delta_norm[index]),
                        "full_vocab_top_token_id": int(top_ids[index]),
                        "full_vocab_top_token": tokenizer.decode([int(top_ids[index])]),
                    }
                    atomic_save_npz(
                        path,
                        final_canonical_logits=final_logits[index].astype(np.float32),
                        metadata=json_array(row_meta),
                    )
                del output
        completed = min((batch_number + 1) * config.batch_size, len(trials))
        print(f"saved all scenarios for {completed}/{len(trials)} questions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Causally remove and add the layerwise runner residual")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(RunnerInterventionConfig.load(args.config))


if __name__ == "__main__":
    main()
