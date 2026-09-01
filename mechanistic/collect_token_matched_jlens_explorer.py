from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .jlens_collect import (
    PositionCollector,
    _scope_end_token,
    _scoped_token,
    _token_offsets,
    _top_tokens,
    _unembed_vector,
)
from .modeling import (
    QWEN_EMPTY_THINKING,
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import (
    ANSWER_ONLY_INSTRUCTION,
    CHOICE_CUE,
    TOKEN_MATCHED_TEST_GAME_FEEDBACK,
    TOKEN_MATCHED_TEST_NEUTRAL_FEEDBACK,
    build_messages,
    load_trials,
    present_question,
)


CONDITIONS = ("incorrect", "neutral")
ANCHORS = (
    "first_question_end",
    "first_answer_decision",
    "historical_answer_end",
    "feedback_subject_end",
    "condition_keyword_end",
    "user_different",
    "action_keyword_end",
    "feedback_end",
    "instruction_letter",
    "instruction_choice",
    "instruction_end",
    "repeated_choice",
    "second_user_end",
    "decision",
)
FEEDBACKS = {
    "incorrect": TOKEN_MATCHED_TEST_GAME_FEEDBACK,
    "neutral": TOKEN_MATCHED_TEST_NEUTRAL_FEEDBACK,
}
FEEDBACK_TOKEN_INDICES = {
    "feedback_subject_end": 1,
    "condition_keyword_end": 3,
    "action_keyword_end": 8,
    "feedback_end": 9,
}
EXPECTED_FEEDBACK_TOKENS = {
    "incorrect": (
        "Your", "answer", "was", "incorrect", ".", "Choose", "a",
        "different", "answer", ".",
    ),
    "neutral": (
        "Your", "answer", "was", "lost", ".", "Choose", "the",
        "answer", "again", ".",
    ),
}


def _feedback_positions(tokenizer: Any, prompt: str, condition: str) -> list[int]:
    feedback = FEEDBACKS[condition]
    start = prompt.find(feedback)
    if start < 0 or prompt.find(feedback, start + 1) >= 0:
        raise RuntimeError(f"Expected exactly one feedback occurrence for {condition}")
    end = start + len(feedback)
    offsets = _token_offsets(tokenizer, prompt)
    positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    ]
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = tuple(tokenizer.decode([ids[position]]).strip() for position in positions)
    if tokens != EXPECTED_FEEDBACK_TOKENS[condition]:
        raise RuntimeError(f"Unexpected feedback tokenization for {condition}: {tokens!r}")
    return positions


def _anchor_positions(
    tokenizer: Any,
    prompt: str,
    condition: str,
    question: dict[str, Any],
    second_user_content: str,
) -> list[int | None]:
    offsets = _token_offsets(tokenizer, prompt)
    feedback_positions = _feedback_positions(tokenizer, prompt, condition)
    question_scope = present_question(question)

    assistant_header = "<|im_start|>assistant\n"
    assistant_start = prompt.find(assistant_header)
    if assistant_start < 0:
        raise RuntimeError("Could not locate the historical assistant header")
    scaffold_end = assistant_start + len(assistant_header) + len(QWEN_EMPTY_THINKING)
    scaffold_tokens = [
        index for index, (left, right) in enumerate(offsets)
        if right > left and right <= scaffold_end
    ]
    if not scaffold_tokens:
        raise RuntimeError("Could not locate the historical empty-thinking scaffold")
    historical_end = scaffold_tokens[-1]

    mapping: dict[str, int | None] = {
        "first_question_end": _scope_end_token(prompt, offsets, question_scope),
        "first_answer_decision": historical_end,
        "historical_answer_end": historical_end,
        "feedback_subject_end": feedback_positions[FEEDBACK_TOKEN_INDICES["feedback_subject_end"]],
        "condition_keyword_end": feedback_positions[FEEDBACK_TOKEN_INDICES["condition_keyword_end"]],
        "user_different": (
            feedback_positions[7] if condition == "incorrect" else None
        ),
        "action_keyword_end": feedback_positions[FEEDBACK_TOKEN_INDICES["action_keyword_end"]],
        "feedback_end": feedback_positions[FEEDBACK_TOKEN_INDICES["feedback_end"]],
        "instruction_letter": _scoped_token(
            prompt, offsets, ANSWER_ONLY_INSTRUCTION, "letter", scope_last=True
        ),
        "instruction_choice": _scoped_token(
            prompt, offsets, ANSWER_ONLY_INSTRUCTION, "choice", scope_last=True
        ),
        "instruction_end": _scope_end_token(
            prompt, offsets, ANSWER_ONLY_INSTRUCTION, scope_last=True
        ),
        "repeated_choice": _scoped_token(
            prompt, offsets, CHOICE_CUE, "choice", scope_last=True
        ),
        "second_user_end": _scope_end_token(
            prompt, offsets, second_user_content, scope_last=True
        ),
        "decision": len(offsets) - 1,
    }
    return [mapping[anchor] for anchor in ANCHORS]


def _baseline_rank_order(results_path: str | Path, question_ids: list[str]) -> np.ndarray:
    results = json.loads(Path(results_path).read_text())["results"]
    order = np.empty((len(question_ids), 4), dtype=np.int64)
    for qi, qid in enumerate(question_ids):
        row = results[qid]
        if "aggregated_ad_logits" in row:
            logits = np.asarray(row["aggregated_ad_logits"], dtype=np.float64)
        else:
            probs = row["probs"]
            logits = np.asarray(
                [
                    np.log(max(float(probs.get(letter, 0.0)), 1e-30))
                    for letter in "ABCD"
                ],
                dtype=np.float64,
            )
        winner = "ABCD".index(row.get("answer", row.get("subject_answer")))
        other = [index for index in range(4) if index != winner]
        order[qi] = [winner] + sorted(other, key=lambda index: logits[index], reverse=True)
    return order


def _insert_rank_rows(row: dict[str, list[dict[str, Any]]], values: np.ndarray) -> None:
    for rank, value in enumerate(values, start=1):
        item = {
            "token_id": -rank,
            "token": f"[Baseline rank {rank}]",
            "score": float(value),
            "tracked": True,
        }
        row["top" if value >= 0 else "bottom"].append(item)
    row["top"].sort(key=lambda item: item["score"], reverse=True)
    row["bottom"].sort(key=lambda item: item["score"])


def collect(
    config_path: Path,
    question_plan_path: Path,
    output: Path,
    lens_repo: str,
    lens_filename: str,
    top_k: int,
    baseline_rank_results: Path | None,
) -> None:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires explicit raw_qwen_chatml serialization")

    plan = json.loads(question_plan_path.read_text())
    question_ids = list(
        plan.get("discovery_question_ids", plan.get("question_ids", []))
    )
    if not question_ids:
        raise ValueError("Question plan contains no discovery/question IDs")
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        question_ids=question_ids,
    )
    trial_by_id = {trial.question_id: trial for trial in trials}
    trials = [trial_by_id[qid] for qid in question_ids]
    output.mkdir(parents=True, exist_ok=True)

    lens_path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    jacobians = checkpoint["J"]
    source_layers = sorted(int(layer) for layer in jacobians)
    if source_layers != list(range(63)):
        raise ValueError(f"Unexpected JLens source layers: {source_layers[:3]}...{source_layers[-3:]}")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    if int(checkpoint["d_model"]) != int(parts.embedding.weight.shape[-1]):
        raise ValueError("JLens/model width mismatch")
    answer_tokens = resolve_answer_tokens(tokenizer, config.answer_variants)
    bare_answer_ids = [answer_tokens[letter][0][1] for letter in "ABCD"]

    shape = (
        len(CONDITIONS), len(trials), len(parts.layers), len(ANCHORS),
        int(parts.embedding.weight.shape[-1]),
    )
    residual_path = output / "position_residuals.npy"
    completed_path = output / "position_completed.npy"
    if residual_path.exists() and completed_path.exists():
        residuals = np.lib.format.open_memmap(residual_path, mode="r+")
        completed = np.load(completed_path)
        if tuple(residuals.shape) != shape or completed.shape != shape[:2]:
            raise ValueError("Existing position cache has incompatible shape")
    else:
        residuals = np.lib.format.open_memmap(
            residual_path, mode="w+", dtype=np.float16, shape=shape
        )
        completed = np.zeros(shape[:2], dtype=bool)

    availability = np.ones((len(CONDITIONS), len(ANCHORS)), dtype=bool)
    availability[1, ANCHORS.index("user_different")] = False
    audit: dict[str, Any] = {
        "anchors": list(ANCHORS),
        "conditions": list(CONDITIONS),
        "availability": {
            condition: {
                anchor: bool(availability[ci, ai])
                for ai, anchor in enumerate(ANCHORS)
            }
            for ci, condition in enumerate(CONDITIONS)
        },
        "trials": {},
    }

    for qi, trial in enumerate(trials):
        for ci, condition in enumerate(CONDITIONS):
            if completed[ci, qi]:
                continue
            messages = build_messages(
                trial.question, condition, config.prompt_mode, config.feedback_variant
            )
            prompt = render_chat(
                processor, messages, config.disable_thinking, config.chat_serialization
            )
            positions = _anchor_positions(
                tokenizer, prompt, condition, trial.question, messages[-1]["content"]
            )
            valid_anchor_indices = [
                index for index, position in enumerate(positions) if position is not None
            ]
            valid_positions = [int(positions[index]) for index in valid_anchor_indices]
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, [prompt])
            ids = input_ids[0].tolist()
            collector = PositionCollector(parts.layers, valid_positions)
            try:
                with torch.inference_mode():
                    kwargs = {
                        "input_ids": input_ids.to(model_input_device(parts)),
                        "attention_mask": attention_mask.to(model_input_device(parts)),
                        "use_cache": False,
                        "return_dict": True,
                    }
                    try:
                        model(**kwargs, logits_to_keep=1)
                    except TypeError:
                        model(**kwargs)
                captured = collector.stacked().numpy()
            finally:
                collector.close()
            residuals[ci, qi, :, valid_anchor_indices] = captured.transpose(1, 0, 2)
            completed[ci, qi] = True
            np.save(completed_path, completed)
            if qi == 0:
                audit["trials"][f"{condition}/{trial.question_id}"] = {
                    "positions": positions,
                    "tokens": [
                        tokenizer.decode([ids[position]]) if position is not None else None
                        for position in positions
                    ],
                    "prompt_length": len(ids),
                    "rendered_prompt": prompt,
                }
        if qi == 0 or (qi + 1) % 10 == 0 or qi + 1 == len(trials):
            print(f"position residuals: {int(completed.sum())}/{completed.size}", flush=True)
    residuals.flush()

    device = model_input_device(parts)
    n_questions = len(trials)
    n_anchors = len(ANCHORS)
    width = shape[-1]
    mean_norm = torch.zeros(
        (len(CONDITIONS), n_anchors, 64, width), dtype=torch.float32
    )
    answer_scores = np.full(
        (len(CONDITIONS), n_questions, n_anchors, 64, 4),
        np.nan,
        dtype=np.float16,
    )
    answer_rows = parts.output_head.weight.detach()[bare_answer_ids].float()
    answer_bias = getattr(parts.output_head, "bias", None)
    if answer_bias is not None:
        answer_bias = answer_bias.detach()[bare_answer_ids].float()
    batch_size = 32

    with torch.inference_mode():
        for layer in range(64):
            J = (
                jacobians[layer].to(device, dtype=torch.float16)
                if layer < 63 else None
            )
            for ci in range(len(CONDITIONS)):
                for start in range(0, n_questions, batch_size):
                    stop = min(start + batch_size, n_questions)
                    values = np.asarray(residuals[ci, start:stop, layer]).copy()
                    tensor = torch.from_numpy(values.reshape(-1, width)).to(
                        device, dtype=torch.float16
                    )
                    transported = tensor if J is None else tensor @ J.T
                    normed = parts.final_norm(
                        transported.to(parts.final_norm.weight.dtype)
                    ).float().reshape(stop - start, n_anchors, width)
                    mean_norm[ci, :, layer] += normed.sum(dim=0).cpu()
                    logits = normed @ answer_rows.T
                    if answer_bias is not None:
                        logits = logits + answer_bias
                    answer_scores[ci, start:stop, :, layer] = (
                        logits.cpu().to(torch.float16).numpy()
                    )
            if J is not None:
                del J
            if layer == 0 or (layer + 1) % 8 == 0 or layer == 63:
                print(f"JLens transform: {layer + 1}/64 readouts", flush=True)
    mean_norm /= n_questions

    rank_order = _baseline_rank_order(
        baseline_rank_results or config.baseline_results_path,
        question_ids,
    )
    aligned_order = np.broadcast_to(
        rank_order[:, None, None, :], (n_questions, n_anchors, 64, 4)
    )
    aligned_scores = np.take_along_axis(
        answer_scores.astype(np.float32), aligned_order[None], axis=-1
    )
    rank_means = aligned_scores.mean(axis=1)

    top_tokens: dict[str, Any] = {
        "final": {},
        "positions": {},
        "position_availability": audit["availability"],
        "rank_pseudotokens": {
            "definition": (
                "Question-dependent bare A-D token selected by fixed Baseline answer rank, "
                "then averaged over the frozen 251-question discovery set."
            )
        },
    }
    with torch.inference_mode():
        for layer in range(64):
            for ai, anchor in enumerate(ANCHORS):
                for ci, condition in enumerate(CONDITIONS):
                    if not availability[ci, ai]:
                        continue
                    logits = _unembed_vector(
                        parts, mean_norm[ci, ai, layer], include_bias=True
                    )
                    row = _top_tokens(tokenizer, logits, top_k)
                    _insert_rank_rows(row, rank_means[ci, ai, layer])
                    top_tokens["positions"][f"{condition}/{anchor}/L{layer}"] = row
                if availability[:, ai].all():
                    logits = _unembed_vector(
                        parts,
                        mean_norm[0, ai, layer] - mean_norm[1, ai, layer],
                        include_bias=False,
                    )
                    row = _top_tokens(tokenizer, logits, top_k)
                    _insert_rank_rows(
                        row, rank_means[0, ai, layer] - rank_means[1, ai, layer]
                    )
                    top_tokens["positions"][
                        f"game_minus_neutral/{anchor}/L{layer}"
                    ] = row

    (output / "top_tokens_with_baseline_ranks.json").write_text(
        json.dumps(top_tokens, indent=2, ensure_ascii=False)
    )
    (output / "position_audit.json").write_text(json.dumps(audit, indent=2))
    np.savez_compressed(
        output / "answer_scores.npz",
        scores=answer_scores,
        question_ids=np.asarray(question_ids),
        conditions=np.asarray(CONDITIONS),
        anchors=np.asarray(ANCHORS),
        rank_order=rank_order,
    )
    metadata = {
        "config": config.as_dict(),
        "question_plan": str(question_plan_path),
        "n_questions": n_questions,
        "question_ids": question_ids,
        "lens_repo": lens_repo,
        "lens_filename": lens_filename,
        "lens_path": lens_path,
        "layer_alignment": (
            "JLens maps 0--62 read post-block residuals 1--63; readout 64 is natural."
        ),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect compact JLens token summaries for finalized token-matched prompts"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--question-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default=(
            "qwen3.6-27b/jlens/Salesforce-wikitext/"
            "Qwen3.6-27B_jacobian_lens_n1000.pt"
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--baseline-rank-results", type=Path)
    args = parser.parse_args()
    collect(
        args.config,
        args.question_plan,
        args.output,
        args.lens_repo,
        args.lens_filename,
        args.top_k,
        args.baseline_rank_results,
    )


if __name__ == "__main__":
    main()
