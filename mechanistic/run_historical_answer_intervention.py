from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .attention_spans import attention_span_indices
from .config import ExperimentConfig
from .historical_answer_intervention import (
    BatchedPositionReadoutAdd,
    JLensAnswerSubspace,
    PositionReadoutCapture,
)
from .io import atomic_save_npz, json_array, shard_path
from .jlens_collect import ANCHORS, _anchor_positions
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages, load_trials, prompt_hash
from .runner_intervention import CpuAnswerLens


SCENARIOS = (
    "natural",
    "erase_winner",
    "erase_runner",
    "erase_all_ad",
    "swap_winner_runner",
    "orthogonal_matched",
)


def _forward(model, parts, input_ids, attention_mask):
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


def _prompt_position(processor, tokenizer, config, trial, condition, anchor):
    messages = build_messages(trial.question, condition, config.prompt_mode)
    prompt = render_chat(
        processor, messages, config.disable_thinking, config.chat_serialization
    )
    annotated_ids, spans = attention_span_indices(
        tokenizer, prompt, condition, trial.question
    )
    if condition == "baseline":
        position = len(annotated_ids) - 1
    else:
        positions = _anchor_positions(
            tokenizer,
            prompt,
            condition,
            spans,
            messages[0]["content"],
            messages[-1]["content"],
        )
        position = positions[ANCHORS.index(anchor)]
        if position is None:
            raise ValueError(f"Anchor {anchor!r} is absent from {condition}")
    input_ids, attention_mask, _ = tokenize_batch(tokenizer, [prompt])
    if annotated_ids != input_ids[0].tolist():
        raise RuntimeError("Offset-aware and model tokenizations disagree")
    return prompt, int(position), annotated_ids, input_ids, attention_mask


def _lens_logits(lens, subspace, residual):
    transported = residual.detach().float().cpu() @ subspace.jacobian.T
    return lens.logits(transported[None, :])[0].detach().numpy().astype(np.float32)


def run(
    config_path: Path,
    plan_path: Path,
    output: Path,
    lens_repo: str,
    lens_filename: str,
    source_readout: int,
    anchor: str,
) -> None:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("This experiment requires baseline_matched_empty_history prompts")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("This experiment requires the preserved raw_qwen_chatml serialization")
    plan = json.loads(plan_path.read_text())
    question_ids = plan.get("question_ids", plan.get("confirmation_question_ids"))
    if not question_ids:
        raise ValueError("Plan has no held-out question IDs")
    trials = load_trials(
        config.manifest_path, config.baseline_results_path, question_ids, None
    )

    lens_path = hf_hub_download(repo_id=lens_repo, filename=lens_filename)
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    source_layer = source_readout - 1
    if source_layer not in checkpoint["J"]:
        raise ValueError(f"JLens has no learned map for source readout {source_readout}")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    answer_rows = parts.output_head.weight.detach()[canonical_ids]
    subspace = JLensAnswerSubspace(
        checkpoint["J"][source_layer], parts.final_norm.weight, answer_rows
    )
    lens = CpuAnswerLens(parts, canonical_ids)
    del checkpoint

    output.mkdir(parents=True, exist_ok=True)
    scenario_ids = [
        f"{prefix}_{scenario}"
        for prefix in ("game", "neutral")
        for scenario in SCENARIOS
    ]
    scenario_ids.insert(0, "baseline_natural")
    metadata = {
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "question_ids": question_ids,
        "scenarios": scenario_ids,
        "source_anchor": anchor,
        "source_readout": source_readout,
        "source_block_completed": source_readout,
        "next_component": f"Mixer {source_readout + 1}",
        "jlens_source_layer_key": source_layer,
        "lens_repo": lens_repo,
        "lens_filename": lens_filename,
        "answer_subspace": {
            **subspace.diagnostics(),
            "definition": (
                "bare A-D unembedding rows transported backward through the learned "
                "JLens map, then centered across A-D"
            ),
        },
        "interventions": {
            "erase_winner": "zero live Baseline winner versus mean(other A-D) JLens numerator",
            "erase_runner": "matched semantic control: zero live Baseline runner versus mean(other A-D) JLens numerator",
            "erase_all_ad": "zero the complete centered three-dimensional A-D JLens numerator",
            "swap_winner_runner": "minimum-L2 update swapping live Baseline winner and runner centered JLens numerators",
            "orthogonal_matched": "deterministic per-question A-D-orthogonal update matched to erase_winner L2 norm",
        },
        "batch_control": (
            "Each five-scenario intervention forward contains a sixth unpatched row. "
            "Saved logits equal single-example natural logits plus intervention-minus-control "
            "logits from that matched physical batch."
        ),
        "resolved_answer_tokens": resolved,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )

    audit_path = output / "position_audit.json"
    for completed, trial in enumerate(trials, 1):
        qid = trial.question_id
        if all(shard_path(output, scenario, qid).exists() for scenario in scenario_ids):
            continue

        baseline_prompt, baseline_position, baseline_ids, input_ids, attention_mask = (
            _prompt_position(
                processor, tokenizer, config, trial, "baseline", anchor
            )
        )
        with PositionReadoutCapture(
            parts, source_readout, baseline_position
        ) as capture:
            baseline_output = _forward(model, parts, input_ids, attention_mask)
        baseline_residual = capture.value
        baseline_logits = (
            baseline_output.logits[0, -1, canonical_ids].detach().float().cpu().numpy()
        )
        baseline_top = int(baseline_output.logits[0, -1].argmax().detach().cpu())
        order = np.argsort(-baseline_logits, kind="stable")
        winner, runner = int(order[0]), int(order[1])

        baseline_meta = {
            "question_id": qid,
            "scenario_id": "baseline_natural",
            "condition": "baseline",
            "prompt_hash": prompt_hash(baseline_prompt),
            "prompt_length": len(baseline_ids),
            "source_position": baseline_position,
            "source_token_id": int(baseline_ids[baseline_position]),
            "source_token": tokenizer.decode([int(baseline_ids[baseline_position])]),
            "winner_letter": "ABCD"[winner],
            "runner_letter": "ABCD"[runner],
            "rank_order": ["ABCD"[index] for index in order],
            "correct_answer": trial.question["correct_answer"],
            "provider_baseline_answer": trial.baseline_answer,
            "full_vocab_top_token_id": baseline_top,
            "full_vocab_top_token": tokenizer.decode([baseline_top]),
        }
        atomic_save_npz(
            shard_path(output, "baseline_natural", qid),
            final_canonical_logits=baseline_logits.astype(np.float32),
            source_jlens_ad_scores=_lens_logits(lens, subspace, baseline_residual),
            source_residual_norm=np.asarray(baseline_residual.norm().item(), dtype=np.float32),
            metadata=json_array(baseline_meta),
        )

        audit = {"question_id": qid, "baseline": baseline_meta, "conditions": {}}
        natural_data = {}
        for condition, prefix in (("incorrect", "game"), ("neutral", "neutral")):
            prompt, history_position, token_ids, natural_input, natural_mask = (
                _prompt_position(processor, tokenizer, config, trial, condition, anchor)
            )
            with PositionReadoutCapture(
                parts, source_readout, history_position
            ) as history_capture:
                natural_output = _forward(
                    model, parts, natural_input, natural_mask
                )
            natural_logits = (
                natural_output.logits[0, -1, canonical_ids]
                .detach().float().cpu().numpy()
            )
            natural_top = int(natural_output.logits[0, -1].argmax().detach().cpu())
            prefix_error = float((history_capture.value - baseline_residual).abs().max())
            natural_data[prefix] = {
                "condition": condition,
                "prompt": prompt,
                "position": history_position,
                "token_ids": token_ids,
                "logits": natural_logits,
                "top": natural_top,
                "residual": history_capture.value,
            }
            natural_meta = {
                **baseline_meta,
                "scenario_id": f"{prefix}_natural",
                "condition": condition,
                "prompt_hash": prompt_hash(prompt),
                "prompt_length": len(token_ids),
                "source_position": history_position,
                "source_token_id": int(token_ids[history_position]),
                "source_token": tokenizer.decode([int(token_ids[history_position])]),
                "prefix_residual_max_abs_error_vs_baseline": prefix_error,
                "full_vocab_top_token_id": natural_top,
                "full_vocab_top_token": tokenizer.decode([natural_top]),
            }
            natural_data[prefix]["metadata"] = natural_meta
            audit["conditions"][condition] = {
                "position": history_position,
                "token_id": int(token_ids[history_position]),
                "token": tokenizer.decode([int(token_ids[history_position])]),
                "prefix_residual_max_abs_error_vs_baseline": prefix_error,
            }

        # Game and Neutral diverge only after this causal prefix. Construct one
        # common perturbation from the representation they actually expose to
        # Mixer 56, rather than from the shorter standalone Baseline sequence.
        shared_residual = natural_data["game"]["residual"]
        cross_condition_error = float(
            (shared_residual - natural_data["neutral"]["residual"]).abs().max()
        )
        audit["game_neutral_source_max_abs_error"] = cross_condition_error
        if cross_condition_error > 1e-5:
            raise RuntimeError(
                "Game and Neutral historical-prefix residuals are not identical: "
                f"max error {cross_condition_error}"
            )
        deltas = {
            "erase_winner": subspace.erase_rank(shared_residual, winner),
            "erase_runner": subspace.erase_rank(shared_residual, runner),
            "erase_all_ad": subspace.erase_all_answer_evidence(shared_residual),
            "swap_winner_runner": subspace.swap(shared_residual, winner, runner),
        }
        deltas["orthogonal_matched"] = subspace.orthogonal_matched_control(
            deltas["erase_winner"], qid, config.seed
        )
        before_scores = _lens_logits(lens, subspace, shared_residual)
        intervention_names = list(SCENARIOS[1:])
        after_scores = {
            name: _lens_logits(lens, subspace, shared_residual + deltas[name])
            for name in intervention_names
        }

        for prefix in ("game", "neutral"):
            row_data = natural_data[prefix]
            condition = row_data["condition"]
            prompt = row_data["prompt"]
            history_position = row_data["position"]
            token_ids = row_data["token_ids"]
            natural_logits = row_data["logits"]
            natural_meta = row_data["metadata"]
            atomic_save_npz(
                shard_path(output, f"{prefix}_natural", qid),
                final_canonical_logits=natural_logits.astype(np.float32),
                source_jlens_ad_scores=before_scores,
                metadata=json_array(natural_meta),
            )

            prompts = [prompt] * (len(intervention_names) + 1)
            batch_ids, batch_mask, _ = tokenize_batch(tokenizer, prompts)
            batch_deltas = torch.stack(
                [deltas[name] for name in intervention_names]
                + [torch.zeros_like(baseline_residual)]
            )
            positions = [history_position] * len(prompts)
            with BatchedPositionReadoutAdd(
                parts, source_readout, positions, batch_deltas
            ):
                patched_output = _forward(model, parts, batch_ids, batch_mask)
            raw = (
                patched_output.logits[:, -1, canonical_ids]
                .detach().float().cpu().numpy()
            )
            corrected = natural_logits[None, :] + raw[:-1] - raw[-1:]
            raw_top = (
                patched_output.logits[:-1, -1].argmax(dim=-1).detach().cpu().numpy()
            )
            for index, name in enumerate(intervention_names):
                delta = deltas[name]
                row_meta = {
                    **natural_meta,
                    "scenario_id": f"{prefix}_{name}",
                    "intervention": name,
                    "residual_delta_l2": float(delta.norm()),
                    "residual_delta_fraction": float(
                        delta.norm() / shared_residual.norm()
                    ),
                    "full_vocab_top_token_id_raw_batch": int(raw_top[index]),
                    "full_vocab_top_token_raw_batch": tokenizer.decode(
                        [int(raw_top[index])]
                    ),
                }
                atomic_save_npz(
                    shard_path(output, f"{prefix}_{name}", qid),
                    final_canonical_logits=corrected[index].astype(np.float32),
                    raw_batch_canonical_logits=raw[index].astype(np.float32),
                    batch_control_canonical_logits=raw[-1].astype(np.float32),
                    source_jlens_ad_scores_before=before_scores,
                    source_jlens_ad_scores_after=after_scores[name],
                    source_residual_delta=delta.numpy().astype(np.float16),
                    metadata=json_array(row_meta),
                )
        if completed == 1 and not audit_path.exists():
            audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True))
        if completed == 1 or completed % 5 == 0 or completed == len(trials):
            print(f"historical answer intervention: {completed}/{len(trials)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causally alter latent Baseline-answer evidence at the empty historical assistant turn"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-readout", type=int, default=55)
    parser.add_argument("--anchor", default="historical_answer_end")
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    args = parser.parse_args()
    run(
        args.config,
        args.plan,
        args.output,
        args.lens_repo,
        args.lens_filename,
        args.source_readout,
        args.anchor,
    )


if __name__ == "__main__":
    main()
