from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .collect_evaluation_gla_residual_writes import _aggregate_logits, _chunks
from .collect_remapped_feedback_factorial import _messages, _remap_question
from .config import ExperimentConfig
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import prompt_hash
from .sublayer import _hidden, middle_norm


CONDITIONS = ("incorrect_again", "lost_again")
CONDITION_NAMES = ("Evaluation", "Matched Neutral")
BOUNDARIES = ("Before GLA", "After GLA", "After MLP")
LENS_NAMES = ("J-lens", "R-lens")


class GLABoundaryCollector:
    """Capture final-position residuals before GLA, after GLA, and after MLP."""

    def __init__(self, parts: Any, layers: list[int], last_indices: list[int]):
        self.last_indices = last_indices
        self.values: dict[int, list[Any]] = {layer: [None, None, None] for layer in layers}
        self.handles = []
        for layer_index in layers:
            layer = parts.layers[layer_index]
            self.handles.extend(
                [
                    layer.register_forward_pre_hook(self._pre_hook(layer_index)),
                    middle_norm(layer).register_forward_pre_hook(self._mid_hook(layer_index)),
                    layer.register_forward_hook(self._post_hook(layer_index)),
                ]
            )

    def _select(self, hidden: Any) -> Any:
        import torch

        rows = torch.arange(hidden.shape[0], device=hidden.device)
        cols = torch.as_tensor(self.last_indices, device=hidden.device)
        return hidden[rows, cols].detach().to("cpu", dtype=torch.float16)

    def _pre_hook(self, layer: int):
        def capture(_module: Any, inputs: Any) -> None:
            self.values[layer][0] = self._select(inputs[0])

        return capture

    def _mid_hook(self, layer: int):
        def capture(_module: Any, inputs: Any) -> None:
            self.values[layer][1] = self._select(inputs[0])

        return capture

    def _post_hook(self, layer: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            self.values[layer][2] = self._select(_hidden(output))

        return capture

    def stacked(self, layers: list[int]):
        import torch

        missing = [
            (layer, boundary)
            for layer in layers
            for boundary, value in enumerate(self.values[layer])
            if value is None
        ]
        if missing:
            raise RuntimeError(f"Missing GLA boundary residuals: {missing}")
        return torch.stack(
            [torch.stack(self.values[layer], dim=1) for layer in layers], dim=1
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _readable_english(token: str) -> bool:
    text = token.strip()
    return bool(text) and all(ord(c) < 128 and c.isprintable() for c in text) and any(
        c.isalpha() for c in text
    )


def _top_english(scores: Any, tokenizer: Any, *, largest: bool, k: int) -> list[dict[str, Any]]:
    import torch

    signed = scores if largest else -scores
    values, ids = torch.topk(signed, k=min(4096, int(scores.shape[-1])))
    rows, seen = [], set()
    for signed_value, token_id in zip(values.detach().float().cpu(), ids.detach().cpu()):
        token_id_int = int(token_id)
        token = tokenizer.decode([token_id_int])
        if not _readable_english(token):
            continue
        display = token.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        normalized = display.strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        score = float(signed_value if largest else -signed_value)
        rows.append({"token_id": token_id_int, "token": display, "score": score})
        if len(rows) == k:
            break
    return rows


def _load_lens(repo: str, filename: str):
    import torch
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=repo,
        filename=filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    return local, torch.load(local, map_location="cpu", weights_only=False)


def _initialize(path: Path, qids: list[str], layers: list[int], width: int) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Checkpoint question IDs differ")
        if arrays["gla_layers_zero_based"].tolist() != layers:
            raise ValueError("Checkpoint GLA layers differ")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "gla_layers_zero_based": np.asarray(layers, dtype=np.int16),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "residual_norm": np.full((2, n, len(layers), 3), np.nan, dtype=np.float32),
        "normalized_transport_sum": np.zeros(
            (2, 2, len(layers), 3, width), dtype=np.float32
        ),
        "lens_ad_scores": np.full(
            (2, 2, n, len(layers), 3, 4), np.nan, dtype=np.float32
        ),
    }


def _trusted_logits(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text())["results"]
    return {
        qid: np.asarray(row["aggregated_ad_logits"], dtype=np.float32)
        for qid, row in payload.items()
    }


def collect(
    config_path: Path,
    remapping_plan_path: Path,
    output_dir: Path,
    trusted_evaluation_path: Path,
    trusted_neutral_path: Path,
    lens_repo: str,
    j_filename: str,
    r_filename: str,
    max_cohorts: int | None,
    checkpoint_every_cohorts: int,
    top_k: int,
    build_partial_readouts: bool,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4:
        raise ValueError("Exact historical execution requires batch_size=4")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml" or config.attn_implementation != "sdpa":
        raise ValueError("Requires exact raw ChatML + SDPA regime")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    plan = json.loads(remapping_plan_path.read_text())
    plan_rows = {row["question_id"]: row for row in plan["rows"]}
    if not set(qids) <= set(plan_rows):
        raise ValueError("Remapping plan is incomplete")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]}) for letter in LETTERS
    }
    variant_groups = [variant_ids[letter] for letter in LETTERS]
    gla_layers = [
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if len(gla_layers) != 48:
        raise RuntimeError(f"Expected 48 GLA layers, found {len(gla_layers)}")
    width = int(parts.embedding.weight.shape[-1])
    device = model_input_device(parts)

    j_path, j_checkpoint = _load_lens(lens_repo, j_filename)
    r_path, r_checkpoint = _load_lens(lens_repo, r_filename)
    checkpoints = [j_checkpoint, r_checkpoint]
    gpu_transports = []
    for name, checkpoint in zip(LENS_NAMES, checkpoints):
        if int(checkpoint["d_model"]) != width:
            raise ValueError(f"{name} width mismatch")
        if not set(gla_layers) <= {int(key) for key in checkpoint["J"]}:
            raise ValueError(f"{name} is missing a GLA source layer")
        gpu_transports.append(
            {
                layer: checkpoint["J"][layer].to(device=device, dtype=torch.bfloat16)
                for layer in gla_layers
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids, gla_layers, width)
    qid_to_index = {qid: index for index, qid in enumerate(qids)}
    trusted = [
        _trusted_logits(trusted_evaluation_path),
        _trusted_logits(trusted_neutral_path),
    ]
    if any(not set(qids) <= set(rows) for rows in trusted):
        raise ValueError("Trusted natural results are incomplete")

    ad_rows = {
        letter: parts.output_head.weight.detach()[variant_ids[letter]].float()
        for letter in LETTERS
    }
    bias = getattr(parts.output_head, "bias", None)
    if bias is not None:
        bias = {
            letter: bias.detach()[variant_ids[letter]].float() for letter in LETTERS
        }
    audit = None
    cohorts = list(_chunks(qids, config.batch_size))
    processed_this_run = 0
    started = time.perf_counter()

    for cohort_index, cohort in enumerate(cohorts):
        indices = [qid_to_index[qid] for qid in cohort]
        if all(arrays["completed"][index] for index in indices):
            continue
        if any(arrays["completed"][index] for index in indices):
            raise RuntimeError("Checkpoint contains a partially completed cohort")
        if max_cohorts is not None and processed_this_run >= max_cohorts:
            break

        remapped = [
            _remap_question(questions[qid], plan_rows[qid]["new_to_original"])
            for qid in cohort
        ]
        for condition_index, condition in enumerate(CONDITIONS):
            messages = [
                _messages(config, questions[qid], remapped_question, condition)
                for qid, remapped_question in zip(cohort, remapped)
            ]
            prompts = [
                render_chat(processor, row, config.disable_thinking, config.chat_serialization)
                for row in messages
            ]
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            collector = GLABoundaryCollector(parts, gla_layers, last_indices)
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
                captured = collector.stacked(gla_layers)
            finally:
                collector.close()

            batch_indices = np.asarray(indices)
            arrays["natural_logits"][condition_index, batch_indices] = _aggregate_logits(
                result, variant_ids
            )
            arrays["residual_norm"][condition_index, batch_indices] = (
                captured.float().norm(dim=-1).numpy()
            )

            with torch.inference_mode():
                for lens_index, transports in enumerate(gpu_transports):
                    for layer_slot, layer in enumerate(gla_layers):
                        values = captured[:, layer_slot].to(device=device, dtype=torch.bfloat16)
                        flat = values.reshape(-1, width)
                        transported = flat @ transports[layer].T
                        normed = parts.final_norm(
                            transported.to(parts.final_norm.weight.dtype)
                        ).float()
                        normed_rows = normed.reshape(len(cohort), 3, width)
                        arrays["normalized_transport_sum"][
                            lens_index, condition_index, layer_slot
                        ] += normed_rows.sum(dim=0).cpu().numpy()

                        selected_scores = []
                        for letter, ids in zip(LETTERS, variant_groups):
                            token_scores = normed @ ad_rows[letter].T
                            if bias is not None:
                                token_scores = token_scores + bias[letter]
                            selected_scores.append(torch.logsumexp(token_scores, dim=-1))
                        ad_scores = torch.stack(selected_scores, dim=-1).reshape(
                            len(cohort), 3, 4
                        )
                        arrays["lens_ad_scores"][
                            lens_index,
                            condition_index,
                            batch_indices,
                            layer_slot,
                        ] = ad_scores.cpu().numpy()
                        del values, flat, transported, normed, normed_rows, ad_scores

            if audit is None:
                audit = {
                    "question_id": cohort[0],
                    "condition": condition,
                    "prompt_hash": prompt_hash(prompts[0]),
                    "prompt": prompts[0],
                    "messages": messages[0],
                    "unpadded_final_position_zero_based": int(last_indices[0]),
                }

        arrays["completed"][indices] = True
        processed_this_run += 1
        if (
            processed_this_run == 1
            or processed_this_run % checkpoint_every_cohorts == 0
            or int(arrays["completed"].sum()) == len(qids)
        ):
            atomic_save_npz(result_path, **arrays)
            print(
                f"GLA boundary residuals: {int(arrays['completed'].sum())}/{len(qids)} "
                f"questions; {time.perf_counter() - started:.1f}s this command",
                flush=True,
            )

    atomic_save_npz(result_path, **arrays)
    complete = bool(np.all(arrays["completed"]))
    max_natural_error = None
    if complete:
        errors = []
        for condition_index in range(2):
            expected = np.stack([trusted[condition_index][qid] for qid in qids])
            errors.append(
                float(np.max(np.abs(arrays["natural_logits"][condition_index] - expected)))
            )
        max_natural_error = max(errors)
        if max_natural_error != 0.0:
            raise RuntimeError(
                f"Natural A-D logits did not reproduce trusted run exactly: {max_natural_error}"
            )
    if complete or build_partial_readouts:
        _build_readouts(
            arrays,
            tokenizer,
            parts,
            output_dir / "readouts.json",
            lens_repo,
            j_filename,
            r_filename,
            top_k,
        )

    metadata = {
        "config": config.as_dict(),
        "remapping_plan": str(remapping_plan_path),
        "trusted_evaluation": str(trusted_evaluation_path),
        "trusted_neutral": str(trusted_neutral_path),
        "n_questions": len(qids),
        "completed_questions": int(arrays["completed"].sum()),
        "complete": complete,
        "conditions": list(CONDITIONS),
        "condition_names": list(CONDITION_NAMES),
        "boundaries": list(BOUNDARIES),
        "gla_layers_zero_based": gla_layers,
        "lenses": {
            "J-lens": {"repo": lens_repo, "filename": j_filename, "local_path": j_path},
            "R-lens": {"repo": lens_repo, "filename": r_filename, "local_path": r_path},
        },
        "complete_model_forward_passes": int(arrays["completed"].sum() // 4) * 2,
        "lens_transports": int(arrays["completed"].sum() // 4) * 2 * 2 * len(gla_layers),
        "batch_rows_per_forward": config.batch_size,
        "max_abs_natural_ad_logit_error_vs_trusted": max_natural_error,
        "boundary_alignment": (
            "The same source-layer workspace lens is applied to the complete final-position "
            "residual immediately before the GLA, immediately after the GLA residual add, "
            "and after the following MLP. After-MLP is the lens's native post-block boundary; "
            "the two within-block states are diagnostic contextual readouts."
        ),
        "score_aggregation": (
            "Every question/state is lens-transported and final-RMS-normalized separately. "
            "Normalized transported residuals are then averaged, so unembedding gives the "
            "exact mean vocabulary score. Contrasts subtract these separately decoded means; "
            "the isolated GLA difference vector is never lensed."
        ),
        "prompt_audit": audit,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def _build_readouts(
    arrays: dict[str, np.ndarray],
    tokenizer: Any,
    parts: Any,
    output_path: Path,
    lens_repo: str,
    j_filename: str,
    r_filename: str,
    top_k: int,
) -> None:
    import torch

    n = int(arrays["completed"].sum())
    means = arrays["normalized_transport_sum"] / n
    layers = arrays["gla_layers_zero_based"].astype(int).tolist()
    head = parts.output_head.weight.detach()
    bias = getattr(parts.output_head, "bias", None)
    result: dict[str, Any] = {
        "definition": (
            "Mean vocabulary scores obtained by separately workspace-lensing each complete "
            "final-position residual state before GLA, after GLA, and after MLP. Score "
            "contrasts are computed after lensing the complete states."
        ),
        "questions": n,
        "conditions": list(CONDITION_NAMES),
        "boundaries": list(BOUNDARIES),
        "lenses": {
            "J-lens": {"repo": lens_repo, "filename": j_filename},
            "R-lens": {"repo": lens_repo, "filename": r_filename},
        },
        "layers": {},
    }

    view_specs = [
        ("Evaluation before GLA", 0, 0, None),
        ("Evaluation after GLA", 0, 1, None),
        ("Evaluation after MLP", 0, 2, None),
        ("Matched Neutral before GLA", 1, 0, None),
        ("Matched Neutral after GLA", 1, 1, None),
        ("Matched Neutral after MLP", 1, 2, None),
        ("Before GLA: Evaluation minus Neutral", 0, 0, (1, 0)),
        ("After GLA: Evaluation minus Neutral", 0, 1, (1, 1)),
        ("After MLP: Evaluation minus Neutral", 0, 2, (1, 2)),
        ("Evaluation GLA change: after minus before", 0, 1, (0, 0)),
        ("Neutral GLA change: after minus before", 1, 1, (1, 0)),
        (
            "GLA contextual change: Evaluation minus Neutral",
            0,
            1,
            ((0, 0), (1, 1), (1, 0)),
        ),
        ("Evaluation MLP change: after MLP minus after GLA", 0, 2, (0, 1)),
        ("Neutral MLP change: after MLP minus after GLA", 1, 2, (1, 1)),
        (
            "MLP contextual change: Evaluation minus Neutral",
            0,
            2,
            ((0, 1), (1, 2), (1, 1)),
        ),
    ]

    with torch.inference_mode():
        for layer_slot, layer in enumerate(layers):
            layer_rows = {}
            for lens_index, lens_name in enumerate(LENS_NAMES):
                states = torch.from_numpy(means[lens_index, :, layer_slot]).to(
                    device=head.device, dtype=head.dtype
                )
                scores = states.reshape(6, -1) @ head.T
                if bias is not None:
                    scores = scores + bias
                scores = scores.reshape(2, 3, -1)
                views = {}
                for label, condition, boundary, subtract in view_specs:
                    vector = scores[condition, boundary]
                    if subtract is not None:
                        if isinstance(subtract[0], tuple):
                            first, second, third = subtract
                            vector = (
                                vector
                                - scores[first[0], first[1]]
                                - scores[second[0], second[1]]
                                + scores[third[0], third[1]]
                            )
                        else:
                            vector = vector - scores[subtract[0], subtract[1]]
                    views[label] = {
                        "positive": _top_english(vector, tokenizer, largest=True, k=top_k),
                        "negative": _top_english(vector, tokenizer, largest=False, k=top_k),
                    }
                layer_rows[lens_name] = views
                del states, scores
            result["layers"][str(layer + 1)] = layer_rows
            print(f"Vocabulary readout: {layer_slot + 1}/{len(layers)} GLA blocks", flush=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode complete final-position residuals before and after every GLA write"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trusted-evaluation", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--lens-repo", default="camilablank/workspace-lenses")
    parser.add_argument("--j-filename", default="qwen3.6-27b/j-lens/lens.pt")
    parser.add_argument("--r-filename", default="qwen3.6-27b/r-lens/lens.pt")
    parser.add_argument("--max-cohorts", type=int)
    parser.add_argument("--checkpoint-every-cohorts", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--build-partial-readouts", action="store_true")
    args = parser.parse_args()
    collect(
        args.config,
        args.remapping_plan,
        args.output_dir,
        args.trusted_evaluation,
        args.trusted_neutral,
        args.lens_repo,
        args.j_filename,
        args.r_filename,
        args.max_cohorts,
        args.checkpoint_every_cohorts,
        args.top_k,
        args.build_partial_readouts,
    )


if __name__ == "__main__":
    main()
