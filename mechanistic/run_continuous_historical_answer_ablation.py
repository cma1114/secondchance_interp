from __future__ import annotations

import argparse
import hashlib
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
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import load_trials, prompt_hash
from .run_historical_answer_intervention import _forward, _prompt_position
from .config import ExperimentConfig


SCENARIOS = (
    "natural",
    "erase_winner_continuous",
    "erase_all_ad_continuous",
    "orthogonal_winner_matched",
    "orthogonal_all_ad_matched",
)


def _build_spaces(checkpoint, norm_weight, answer_rows, model_indices):
    import torch

    gamma_rows = answer_rows.detach().float().cpu() * norm_weight.detach().float().cpu()[None, :]
    spaces = {}
    diagnostics = {}
    for model_index in model_indices:
        # Input to model layer i is post-block readout i, represented by J key i-1.
        key = model_index - 1
        if key not in checkpoint["J"]:
            raise KeyError(f"JLens has no map for model layer input {model_index}")
        decoder = gamma_rows @ checkpoint["J"][key].detach().float().cpu()
        centered = decoder - decoder.mean(dim=0, keepdim=True)
        _u, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        # Four rows centered across answers have exactly three independent
        # contrasts.  The fourth singular value is nonzero only because the
        # learned map and the matrix products are stored in finite precision.
        rank = 3
        basis = vh[:rank].contiguous()
        units = []
        for answer in range(4):
            others = [index for index in range(4) if index != answer]
            contrast = decoder[answer] - decoder[others].mean(dim=0)
            units.append(contrast / contrast.norm())
        spaces[model_index] = {
            "basis": basis,
            "winner_units": torch.stack(units),
        }
        diagnostics[str(model_index + 1)] = {
            "jlens_key": key,
            "rank": rank,
            "largest_singular_value": float(singular[0]),
            "smallest_nonzero_singular_value": float(singular[rank - 1]),
            "fourth_to_first_singular_ratio": float(singular[3] / singular[0]),
        }
    return spaces, diagnostics


def _orthogonal_unit(basis, question_id: str, layer: int, label: str, seed: int):
    import torch

    digest = hashlib.sha256(
        f"{seed}:{question_id}:{layer}:{label}".encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    vector = torch.from_numpy(rng.standard_normal(basis.shape[1])).float()
    vector = vector - (vector @ basis.T) @ basis
    if vector.norm() <= 1e-8:
        raise RuntimeError("Degenerate answer-orthogonal control")
    return vector / vector.norm()


class ContinuousHistoricalAnswerAblator:
    """Repeatedly remove answer identity at one token before selected Mixers.

    Batch rows are winner erasure, full A-D erasure, winner-matched orthogonal
    control, all-A-D-matched orthogonal control, and an untouched batch control.
    """

    def __init__(self, parts, model_indices, position, spaces, winner, controls):
        self.position = int(position)
        self.spaces = spaces
        self.winner = int(winner)
        self.controls = controls
        self.records = {}
        self.handles = [
            parts.layers[index].register_forward_pre_hook(self._hook(index))
            for index in model_indices
        ]

    def _hook(self, model_index):
        def intervene(_module, inputs):
            import torch

            hidden = inputs[0]
            if hidden.shape[0] != 5:
                raise ValueError("Continuous ablation expects a five-row batch")
            changed = hidden.clone()
            current = hidden[:, self.position].float()
            space = self.spaces[model_index]
            basis = space["basis"].to(current.device)
            unit = space["winner_units"][self.winner].to(current.device)

            winner_delta = -torch.dot(current[0], unit) * unit
            all_delta = -((current[1] @ basis.T) @ basis)
            winner_control = self.controls[model_index]["winner"].to(current.device)
            all_control = self.controls[model_index]["all_ad"].to(current.device)
            winner_control_delta = winner_control * winner_delta.norm()
            all_control_delta = all_control * all_delta.norm()
            deltas = torch.stack(
                [
                    winner_delta,
                    all_delta,
                    winner_control_delta,
                    all_control_delta,
                    torch.zeros_like(winner_delta),
                ]
            )
            changed[:, self.position] = (
                current + deltas
            ).to(dtype=hidden.dtype)
            after_winner = torch.dot(current[0] + winner_delta, unit).abs()
            after_all = torch.linalg.vector_norm(
                (current[1] + all_delta) @ basis.T
            )
            self.records[model_index] = torch.stack(
                [
                    winner_delta.norm(),
                    all_delta.norm(),
                    current[0].norm(),
                    current[1].norm(),
                    after_winner,
                    after_all,
                ]
            ).detach()
            return (changed, *inputs[1:])

        return intervene

    def arrays(self):
        import torch

        ordered = [self.records[index] for index in sorted(self.records)]
        if len(ordered) != len(self.handles):
            raise RuntimeError("Not every selected layer ran its intervention hook")
        values = torch.stack(ordered).float().cpu().numpy()
        return {
            "winner_delta_norm": values[:, 0],
            "all_ad_delta_norm": values[:, 1],
            "winner_residual_norm": values[:, 2],
            "all_ad_residual_norm": values[:, 3],
            "winner_score_abs_after": values[:, 4],
            "all_ad_score_norm_after": values[:, 5],
        }

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def run(
    config_path: Path,
    plan_path: Path,
    baseline_root: Path,
    output: Path,
    first_layer: int,
    last_layer: int,
    lens_repo: str,
    lens_filename: str,
) -> None:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")
    plan = json.loads(plan_path.read_text())
    qids = plan.get("question_ids", plan.get("confirmation_question_ids"))
    if not qids:
        raise ValueError("Plan has no question IDs")
    trials = load_trials(config.manifest_path, config.baseline_results_path, qids, None)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    model_indices = list(range(first_layer - 1, last_layer))
    if model_indices[0] < 1 or model_indices[-1] >= len(parts.layers):
        raise ValueError("Selected layer range is outside the model")

    lens_path = hf_hub_download(repo_id=lens_repo, filename=lens_filename)
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    spaces_cpu, diagnostics = _build_spaces(
        checkpoint,
        parts.final_norm.weight,
        parts.output_head.weight.detach()[canonical_ids],
        model_indices,
    )
    del checkpoint
    device = model_input_device(parts)
    spaces = {
        index: {
            name: value.to(device=device, dtype=torch.float32)
            for name, value in tensors.items()
        }
        for index, tensors in spaces_cpu.items()
    }

    output.mkdir(parents=True, exist_ok=True)
    scenario_ids = [
        f"{prefix}_{scenario}"
        for prefix in ("game", "neutral")
        for scenario in SCENARIOS
    ]
    (output / "run_metadata.json").write_text(json.dumps({
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "baseline_root": str(baseline_root),
        "question_ids": qids,
        "scenarios": scenario_ids,
        "position": "historical_answer_end",
        "first_user_facing_layer": first_layer,
        "last_user_facing_layer": last_layer,
        "model_indices_zero_based": model_indices,
        "jlens_diagnostics": diagnostics,
        "lens_repo": lens_repo,
        "lens_filename": lens_filename,
        "batch_rows": list(SCENARIOS[1:]) + ["batch_control"],
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }, indent=2, sort_keys=True))

    audit_path = output / "position_audit.json"
    for completed, trial in enumerate(trials, 1):
        qid = trial.question_id
        with np.load(
            shard_path(baseline_root, "baseline_natural", qid), allow_pickle=False
        ) as source:
            baseline_logits = source["final_canonical_logits"].astype(np.float64)
        order = np.argsort(-baseline_logits, kind="stable")
        winner, runner = int(order[0]), int(order[1])
        controls = {
            index: {
                label: _orthogonal_unit(
                    spaces_cpu[index]["basis"], qid, index + 1, label, config.seed
                ).to(device=device)
                for label in ("winner", "all_ad")
            }
            for index in model_indices
        }

        for condition, prefix in (("incorrect", "game"), ("neutral", "neutral")):
            if all(
                shard_path(output, f"{prefix}_{scenario}", qid).exists()
                for scenario in SCENARIOS
            ):
                continue
            prompt, position, token_ids, input_ids, attention_mask = _prompt_position(
                processor,
                tokenizer,
                config,
                trial,
                condition,
                "historical_answer_end",
            )
            natural_output = _forward(model, parts, input_ids, attention_mask)
            natural_logits = (
                natural_output.logits[0, -1, canonical_ids]
                .detach().float().cpu().numpy()
            )
            prompts = [prompt] * 5
            batch_ids, batch_mask, _ = tokenize_batch(tokenizer, prompts)
            with ContinuousHistoricalAnswerAblator(
                parts,
                model_indices,
                position,
                spaces,
                winner,
                controls,
            ) as ablator:
                batch_output = _forward(model, parts, batch_ids, batch_mask)
                audit_arrays = ablator.arrays()
            raw = (
                batch_output.logits[:, -1, canonical_ids]
                .detach().float().cpu().numpy()
            )
            corrected = natural_logits[None, :] + raw[:4] - raw[4:5]
            base_metadata = {
                "question_id": qid,
                "condition": condition,
                "prompt_hash": prompt_hash(prompt),
                "historical_endpoint": position,
                "historical_endpoint_token": tokenizer.decode([token_ids[position]]),
                "winner_letter": "ABCD"[winner],
                "runner_letter": "ABCD"[runner],
                "rank_order": ["ABCD"[value] for value in order],
            }
            atomic_save_npz(
                shard_path(output, f"{prefix}_natural", qid),
                final_canonical_logits=natural_logits.astype(np.float32),
                metadata=json_array({**base_metadata, "scenario": "natural"}),
            )
            for index, scenario in enumerate(SCENARIOS[1:]):
                delta_key = (
                    "winner_delta_norm"
                    if scenario in {"erase_winner_continuous", "orthogonal_winner_matched"}
                    else "all_ad_delta_norm"
                )
                atomic_save_npz(
                    shard_path(output, f"{prefix}_{scenario}", qid),
                    final_canonical_logits=corrected[index].astype(np.float32),
                    raw_batch_canonical_logits=raw[index].astype(np.float32),
                    batch_control_canonical_logits=raw[4].astype(np.float32),
                    layer_delta_norm=audit_arrays[delta_key].astype(np.float32),
                    winner_score_abs_after=audit_arrays["winner_score_abs_after"].astype(np.float32),
                    all_ad_score_norm_after=audit_arrays["all_ad_score_norm_after"].astype(np.float32),
                    metadata=json_array({**base_metadata, "scenario": scenario}),
                )
            if not audit_path.exists():
                audit_path.write_text(json.dumps({
                    **base_metadata,
                    "selected_user_facing_layers": [index + 1 for index in model_indices],
                    "max_winner_score_abs_after": float(
                        audit_arrays["winner_score_abs_after"].max()
                    ),
                    "max_all_ad_score_norm_after": float(
                        audit_arrays["all_ad_score_norm_after"].max()
                    ),
                }, indent=2, sort_keys=True))
        if completed == 1 or completed % 5 == 0 or completed == len(trials):
            print(f"Continuous historical-answer ablation: {completed}/{len(trials)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, default=33)
    parser.add_argument("--last-layer", type=int, default=64)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    args = parser.parse_args()
    run(
        args.config,
        args.plan,
        args.baseline_root,
        args.output,
        args.first_layer,
        args.last_layer,
        args.lens_repo,
        args.lens_filename,
    )


if __name__ == "__main__":
    main()
