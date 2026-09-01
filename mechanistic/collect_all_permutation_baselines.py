from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages, prompt_hash


def _remap(question: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    result = copy.deepcopy(question)
    result["options"] = {new: question["options"][mapping[new]] for new in LETTERS}
    result["correct_answer"] = {
        original: new for new, original in mapping.items()
    }[question["correct_answer"]]
    return result


def _checkpoint(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _line_spans(tokenizer: Any, prompt: str, question: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    starts = np.zeros(4, dtype=np.int16)
    ends = np.zeros(4, dtype=np.int16)
    for li, letter in enumerate(LETTERS):
        line = f"  {letter}: {question['options'][letter]}\n"
        char_start = prompt.find(line)
        if char_start < 0 or prompt.find(line, char_start + 1) >= 0:
            raise RuntimeError(f"Expected one option line {line!r}")
        char_end = char_start + len(line)
        positions = [
            index for index, (left, right) in enumerate(offsets)
            if right > left and left < char_end and right > char_start
        ]
        if not positions or positions != list(range(positions[0], positions[-1] + 1)):
            raise RuntimeError("Option-line token positions are absent or noncontiguous")
        starts[li] = positions[0]
        ends[li] = positions[-1] + 1
    return starts, ends


def collect(
    config_path: Path,
    plan_path: Path,
    output_dir: Path,
    max_cohorts: int | None,
) -> None:
    import torch

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml")
    if int(config.batch_size) != 4:
        raise ValueError("Requires physical batches of four")

    plan = json.loads(plan_path.read_text())
    qids = list(plan["question_ids"])
    n_mappings = int(plan["n_mappings_per_question"])
    global_mappings = plan.get("mappings")
    row_mappings = {
        row["question_id"]: list(row["mappings"])
        for row in plan.get("rows", [])
    }
    if global_mappings is not None:
        if n_mappings != len(global_mappings):
            raise ValueError("Global mapping count differs from the plan")
    elif set(row_mappings) != set(qids):
        raise ValueError("Question-specific mapping rows are incomplete")
    if n_mappings < 2:
        raise ValueError("At least two mappings are required")
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    if qids != [row["id"] for row in manifest["questions"]]:
        raise ValueError("Plan order differs from the manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    n_questions = len(qids)
    if result_path.exists():
        existing = np.load(result_path, allow_pickle=False)
        logits = np.asarray(existing["aggregated_ad_logits"], dtype=np.float32)
        ad_mass = np.asarray(existing["ad_probability_mass"], dtype=np.float32)
        starts = np.asarray(existing["option_line_starts"], dtype=np.int16)
        ends = np.asarray(existing["option_line_ends"], dtype=np.int16)
        lengths = np.asarray(existing["prompt_lengths"], dtype=np.int16)
        completed = np.asarray(existing["completed"], dtype=bool)
        if logits.shape != (n_mappings, n_questions, 4):
            raise ValueError("Existing screen shape differs")
    else:
        logits = np.full((n_mappings, n_questions, 4), np.nan, dtype=np.float32)
        ad_mass = np.full((n_mappings, n_questions), np.nan, dtype=np.float32)
        starts = np.full((n_mappings, n_questions, 4), -1, dtype=np.int16)
        ends = np.full((n_mappings, n_questions, 4), -1, dtype=np.int16)
        lengths = np.full((n_mappings, n_questions), -1, dtype=np.int16)
        completed = np.zeros((n_mappings, n_questions), dtype=bool)

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    all_ad_ids = sorted({token_id for ids in variant_ids.values() for token_id in ids})
    device = model_input_device(parts)
    audit_path = output_dir / "prompt_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    model_ready = time.monotonic()
    durations: list[float] = []

    cohorts = [qids[start:start + 4] for start in range(0, n_questions, 4)]
    if max_cohorts is not None:
        cohorts = cohorts[: int(max_cohorts)]
    qid_index = {qid: index for index, qid in enumerate(qids)}
    for cohort_index, cohort in enumerate(cohorts):
        cohort_started = time.monotonic()
        indices = [qid_index[qid] for qid in cohort]
        forwards = 0
        for mapping_index in range(n_mappings):
            if completed[mapping_index, indices].all():
                continue
            mappings = [
                (global_mappings if global_mappings is not None else row_mappings[qid])[
                    mapping_index
                ]["new_to_original"]
                for qid in cohort
            ]
            remapped = [
                _remap(questions[qid], mapping)
                for qid, mapping in zip(cohort, mappings)
            ]
            prompts = [
                render_chat(
                    processor,
                    build_messages(question, "baseline", config.prompt_mode, config.feedback_variant),
                    config.disable_thinking,
                    config.chat_serialization,
                )
                for question in remapped
            ]
            for row_index, question_index in enumerate(indices):
                row_starts, row_ends = _line_spans(tokenizer, prompts[row_index], remapped[row_index])
                starts[mapping_index, question_index] = row_starts
                ends[mapping_index, question_index] = row_ends
                lengths[mapping_index, question_index] = len(
                    tokenizer(prompts[row_index], add_special_tokens=False)["input_ids"]
                )
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            with torch.inference_mode():
                # Request the complete sequence explicitly. On this pinned
                # Transformers build, the optimized `logits_to_keep=1` path
                # became numerically invalid after the retained host restarted
                # (while reporting the same package version). Selecting the
                # true final token from full logits is slower but unambiguous.
                output = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    use_cache=False,
                    return_dict=True,
                )
            forwards += 1
            raw = output.logits.detach().float().cpu()
            if raw.ndim != 3 or raw.shape[1] <= int(np.max(last_indices)):
                raise RuntimeError(f"Unexpected full-logit shape {tuple(raw.shape)}")
            final = raw[np.arange(len(cohort)), last_indices]
            if not torch.isfinite(final).all() or float(final.abs().max()) > 1.0e4:
                raise RuntimeError("Non-finite or implausibly large final-token logits")
            probabilities = torch.softmax(final, dim=-1)
            for row_index, question_index in enumerate(indices):
                logits[mapping_index, question_index] = torch.stack([
                    torch.logsumexp(final[row_index, variant_ids[letter]], dim=0)
                    for letter in LETTERS
                ]).numpy()
                ad_mass[mapping_index, question_index] = float(probabilities[row_index, all_ad_ids].sum())
                completed[mapping_index, question_index] = True
            if cohort_index == 0 and mapping_index in (0, 1, n_mappings - 1):
                audit[str(mapping_index)] = {
                    "question_ids": cohort,
                    "mappings": mappings,
                    "prompt_hashes": [prompt_hash(prompt) for prompt in prompts],
                    "option_line_starts": starts[mapping_index, indices].tolist(),
                    "option_line_ends": ends[mapping_index, indices].tolist(),
                    "rendered_prompt_first_question": prompts[0],
                }
            _checkpoint(
                result_path,
                question_ids=np.asarray(qids),
                aggregated_ad_logits=logits,
                ad_probability_mass=ad_mass,
                option_line_starts=starts,
                option_line_ends=ends,
                prompt_lengths=lengths,
                completed=completed,
            )
        if forwards:
            durations.append(time.monotonic() - cohort_started)
        print(
            f"all-permutation screen: {int(completed.sum())}/{completed.size}; "
            f"cohort {cohort_index + 1}/{len(cohorts)}",
            flush=True,
        )

    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    metadata = {
        "config": config.as_dict(),
        "plan": str(plan_path),
        "n_questions": n_questions,
        "n_mappings": n_mappings,
        "complete": bool(completed.all()) if max_cohorts is None else False,
        "complete_model_forwards_per_cohort": n_mappings,
        "physical_prompts_per_complete_cohort": n_mappings * 4,
        "elapsed_seconds_after_model_load": time.monotonic() - model_ready,
        "completed_cohort_durations_seconds": durations,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    collect(args.config, args.plan, args.output_dir, args.max_cohorts)


if __name__ == "__main__":
    main()
