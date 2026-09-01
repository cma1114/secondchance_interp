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
    result["options"] = {
        new: question["options"][mapping[new]] for new in LETTERS
    }
    result["correct_answer"] = {
        original: new for new, original in mapping.items()
    }[question["correct_answer"]]
    return result


def _checkpoint(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


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
        raise ValueError("Requires the historical physical batch size of four")

    plan = json.loads(plan_path.read_text())
    qids = list(plan["question_ids"])
    plan_rows = {row["question_id"]: row for row in plan["rows"]}
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    if qids != [row["id"] for row in manifest["questions"]]:
        raise ValueError("Plan question order must match the historical manifest order")
    n_questions = len(qids)
    n_mappings = 6

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    if result_path.exists():
        existing = np.load(result_path)
        logits = np.asarray(existing["aggregated_ad_logits"], dtype=np.float32)
        ad_mass = np.asarray(existing["ad_probability_mass"], dtype=np.float32)
        completed = np.asarray(existing["completed"], dtype=bool)
    else:
        logits = np.full((n_mappings, n_questions, 4), np.nan, dtype=np.float32)
        ad_mass = np.full((n_mappings, n_questions), np.nan, dtype=np.float32)
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
    audit: dict[str, Any] = (
        json.loads(audit_path.read_text()) if audit_path.exists() else {}
    )
    model_ready_time = time.monotonic()
    cohort_durations: list[float] = []

    cohorts = [qids[start : start + 4] for start in range(0, n_questions, 4)]
    if max_cohorts is not None:
        cohorts = cohorts[: int(max_cohorts)]
    for cohort_index, cohort in enumerate(cohorts):
        cohort_started = time.monotonic()
        forwards_this_cohort = 0
        indices = [qids.index(qid) for qid in cohort]
        for mapping_index in range(n_mappings):
            if completed[mapping_index, indices].all():
                continue
            mappings = [
                plan_rows[qid]["mappings"][mapping_index]["new_to_original"]
                for qid in cohort
            ]
            remapped = [
                _remap(questions[qid], mapping)
                for qid, mapping in zip(cohort, mappings)
            ]
            messages = [
                build_messages(question, "baseline", config.prompt_mode, config.feedback_variant)
                for question in remapped
            ]
            prompts = [
                render_chat(processor, row, config.disable_thinking, config.chat_serialization)
                for row in messages
            ]
            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            with torch.inference_mode():
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
            forwards_this_cohort += 1
            raw = output.logits.detach().float().cpu()
            final = raw[:, 0] if raw.shape[1] == 1 else raw[np.arange(len(cohort)), last_indices]
            probabilities = torch.softmax(final, dim=-1)
            for row_index, question_index in enumerate(indices):
                logits[mapping_index, question_index] = torch.stack([
                    torch.logsumexp(final[row_index, variant_ids[letter]], dim=0)
                    for letter in LETTERS
                ]).numpy()
                ad_mass[mapping_index, question_index] = float(
                    probabilities[row_index, all_ad_ids].sum()
                )
                completed[mapping_index, question_index] = True
            if cohort_index == 0:
                audit[str(mapping_index)] = {
                    "question_ids": cohort,
                    "mappings": mappings,
                    "prompt_hashes": [prompt_hash(prompt) for prompt in prompts],
                    "rendered_prompt_first_question": prompts[0],
                }
            _checkpoint(
                result_path,
                question_ids=np.asarray(qids),
                aggregated_ad_logits=logits,
                ad_probability_mass=ad_mass,
                completed=completed,
            )
        if forwards_this_cohort:
            cohort_durations.append(time.monotonic() - cohort_started)
        print(
            f"W1-fixed permutation screen: {int(completed.sum())}/{completed.size}; "
            f"cohort {cohort_index + 1}/{len(cohorts)}",
            flush=True,
        )

    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    metadata = {
        "config": config.as_dict(),
        "plan": str(plan_path),
        "n_questions": n_questions,
        "n_mappings": n_mappings,
        "complete": bool(completed.all()) if max_cohorts is None else False,
        "complete_model_forwards_per_cohort": n_mappings,
        "physical_prompts_per_complete_cohort": n_mappings * 4,
        "elapsed_seconds_after_model_load": time.monotonic() - model_ready_time,
        "completed_cohort_durations_seconds": cohort_durations,
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
