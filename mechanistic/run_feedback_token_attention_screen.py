from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .collect_attention import _attention_tuple
from .config import ExperimentConfig
from .io import atomic_save_npz
from .modeling import load_model_and_processor, model_input_device
from .prompts import load_trials
from .run_feedback_token_trajectory_swap import _render_and_locate


CONDITIONS = ("incorrect", "neutral")
SOURCE_TOKEN_INDICES = (3, 8)  # incorrect/lost; second answer/again


def _save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    atomic_save_npz(path, **arrays)


def run(config_path: Path, output: Path, max_questions: int | None) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires the isolated token_matched_test prompt variant")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw_qwen_chatml serialization")
    if config.attn_implementation != "eager":
        raise ValueError("Exact token attention requires eager attention")

    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        config.question_ids,
        max_questions if max_questions is not None else config.max_questions,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = getattr(processor, "tokenizer", processor)
    ordinary_layers = [
        index
        for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    ]
    if not ordinary_layers:
        raise RuntimeError("Model exposes no conventional self-attention blocks")

    qids = np.asarray([trial.question_id for trial in trials])
    arrays: dict[str, np.ndarray] | None = None
    if output.exists():
        with np.load(output, allow_pickle=False) as existing:
            arrays = {key: existing[key] for key in existing.files}
        if arrays["question_ids"].astype(str).tolist() != qids.astype(str).tolist():
            raise ValueError("Existing output uses different question IDs")

    audit = None
    completed_since_save = 0
    for qi, trial in enumerate(trials):
        for ci, condition in enumerate(CONDITIONS):
            if arrays is not None and bool(arrays["completed"][ci, qi]):
                continue
            prompt_data = _render_and_locate(
                processor, tokenizer, config, trial, condition
            )
            if audit is None:
                audit = {
                    "question_id": trial.question_id,
                    "condition": condition,
                    "feedback_tokens": list(prompt_data["tokens"]),
                    "feedback_absolute_positions_zero_based": prompt_data["positions"],
                    "source_tokens": [
                        prompt_data["tokens"][index] for index in SOURCE_TOKEN_INDICES
                    ],
                    "source_absolute_positions_zero_based": [
                        prompt_data["positions"][index]
                        for index in SOURCE_TOKEN_INDICES
                    ],
                    "final_query_position_zero_based": len(prompt_data["ids"]) - 1,
                    "final_query_token": tokenizer.decode([prompt_data["ids"][-1]]),
                }
            with torch.inference_mode():
                kwargs = {
                    "input_ids": prompt_data["input_ids"].to(model_input_device(parts)),
                    "attention_mask": prompt_data["attention_mask"].to(
                        model_input_device(parts)
                    ),
                    "use_cache": False,
                    "return_dict": True,
                    "output_attentions": True,
                }
                try:
                    result = model(**kwargs, logits_to_keep=1)
                except TypeError:
                    result = model(**kwargs)
            raw = _attention_tuple(result)
            if len(raw) == len(parts.layers):
                ordinary = [raw[index] for index in ordinary_layers]
            else:
                ordinary = [value for value in raw if value is not None]
            if len(ordinary) != len(ordinary_layers):
                raise RuntimeError(
                    f"Expected {len(ordinary_layers)} attention matrices; got {len(ordinary)}"
                )
            source_positions = [
                prompt_data["positions"][index] for index in SOURCE_TOKEN_INDICES
            ]
            weights = torch.stack(
                [
                    layer_attention[0, :, -1, source_positions]
                    for layer_attention in ordinary
                ]
            ).detach().float().cpu().numpy()
            # weights: conventional layer, query head, source token
            if arrays is None:
                n_layers, n_heads, n_sources = weights.shape
                arrays = {
                    "question_ids": qids,
                    "completed": np.zeros((2, len(trials)), dtype=bool),
                    "attention": np.full(
                        (2, len(trials), n_layers, n_heads, n_sources),
                        np.nan,
                        dtype=np.float16,
                    ),
                    "prompt_hashes": np.full((2, len(trials)), "", dtype="<U64"),
                    "ordinary_layer_indices_zero_based": np.asarray(
                        ordinary_layers, dtype=np.int16
                    ),
                }
            arrays["attention"][ci, qi] = weights.astype(np.float16)
            arrays["prompt_hashes"][ci, qi] = prompt_data["prompt_hash"]
            arrays["completed"][ci, qi] = True
            completed_since_save += 1
            del result, raw, ordinary
            if completed_since_save >= 10:
                _save(output, arrays)
                completed_since_save = 0
        if qi == 0 or (qi + 1) % 10 == 0 or qi + 1 == len(trials):
            done = int(arrays["completed"].sum()) if arrays is not None else 0
            print(f"Feedback-token attention: {done}/{2 * len(trials)} forwards", flush=True)

    if arrays is None:
        raise RuntimeError("No trials were collected")
    _save(output, arrays)
    metadata = {
        "config": config.as_dict(),
        "result_file": str(output),
        "n_questions": len(trials),
        "ordinary_attention_model_indices_zero_based": ordinary_layers,
        "ordinary_attention_user_facing_blocks_one_based": [
            index + 1 for index in ordinary_layers
        ],
        "head_indices_in_result_zero_based": True,
        "source_token_indices_within_feedback_zero_based": list(
            SOURCE_TOKEN_INDICES
        ),
        "source_tokens": {
            "incorrect": ["incorrect", "answer"],
            "neutral": ["lost", "again"],
        },
        "measurement": (
            "ordinary softmax-attention weight from the final decision query "
            "to each exact source-token position"
        ),
        "architectural_limit": (
            "Qwen GLA blocks do not expose ordinary pairwise token-attention weights"
        ),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
        "first_prompt_audit": audit,
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    run(args.config, args.output, args.max_questions)


if __name__ == "__main__":
    main()
