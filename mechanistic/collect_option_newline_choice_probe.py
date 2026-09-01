from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .collect_contextual_option_representations import (
    BatchedPositionCollector,
    _positions,
)
from .config import ExperimentConfig
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages


LETTERS = "ABCD"


def _remap_question_ordered(
    question: dict[str, Any], new_to_original: dict[str, str]
) -> dict[str, Any]:
    """Remap while preserving the prompt builder's required A-D key order."""
    remapped = copy.deepcopy(question)
    remapped["options"] = {
        letter: question["options"][new_to_original[letter]] for letter in LETTERS
    }
    original_to_new = {
        original: new for new, original in new_to_original.items()
    }
    remapped["correct_answer"] = original_to_new[question["correct_answer"]]
    return remapped


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
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml")
    if config.attn_implementation != "sdpa" or int(config.batch_size) != 4:
        raise ValueError("Requires the historical batch-four SDPA regime")

    plan = json.loads(plan_path.read_text())
    qids = list(plan["question_ids"])
    plan_rows = {row["question_id"]: row for row in plan["rows"]}
    manifest = json.loads(Path(config.manifest_path).read_text())
    source = {row["id"]: row for row in manifest["questions"]}
    if qids != [row["id"] for row in manifest["questions"]]:
        raise ValueError("Plan order must match the historical manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    width = int(parts.embedding.weight.shape[-1])
    n_layers = len(parts.layers)
    n_questions = len(qids)
    n_mappings = 6
    residual_shape = (n_mappings, n_questions, n_layers, 4, width)
    residual_path = output_dir / "option_newline_residuals.npy"
    completed_path = output_dir / "completed.npy"
    result_path = output_dir / "results.npz"

    if residual_path.exists() and completed_path.exists() and result_path.exists():
        residuals = np.lib.format.open_memmap(residual_path, mode="r+")
        completed = np.load(completed_path)
        with np.load(result_path, allow_pickle=False) as loaded:
            logits = np.asarray(loaded["aggregated_ad_logits"], dtype=np.float32)
        if tuple(residuals.shape) != residual_shape:
            raise ValueError("Existing residual cache has the wrong shape")
        if completed.shape != (n_mappings, n_questions):
            raise ValueError("Existing completion checkpoint has the wrong shape")
    else:
        residuals = np.lib.format.open_memmap(
            residual_path, mode="w+", dtype=np.float16, shape=residual_shape
        )
        completed = np.zeros((n_mappings, n_questions), dtype=bool)
        logits = np.full((n_mappings, n_questions, 4), np.nan, dtype=np.float32)

    started_after_load = time.monotonic()
    cohort_durations: list[float] = []
    audit_path = output_dir / "prompt_audit.json"
    cohorts = [qids[start : start + 4] for start in range(0, n_questions, 4)]
    if max_cohorts is not None:
        cohorts = cohorts[: int(max_cohorts)]

    for cohort_index, cohort in enumerate(cohorts):
        cohort_started = time.monotonic()
        question_indices = [qids.index(qid) for qid in cohort]
        forwards = 0
        for mapping_index in range(n_mappings):
            if completed[mapping_index, question_indices].all():
                continue
            questions = []
            prompts = []
            position_rows = []
            lengths = []
            audits = []
            for qid in cohort:
                mapping = plan_rows[qid]["mappings"][mapping_index]["new_to_original"]
                question = _remap_question_ordered(source[qid], mapping)
                messages = build_messages(
                    question, "baseline", config.prompt_mode, config.feedback_variant
                )
                prompt = render_chat(
                    processor,
                    messages,
                    config.disable_thinking,
                    config.chat_serialization,
                )
                all_positions, audit = _positions(tokenizer, prompt, question)
                # _positions returns content-end A-D, then option-closing newline A-D,
                # then the first-decision boundary.  This experiment deliberately
                # collects only the four option-closing newline positions.
                positions = all_positions[4:8]
                ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                questions.append(question)
                prompts.append(prompt)
                position_rows.append(positions)
                lengths.append(len(ids))
                audits.append(audit)

            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            token_width = int(input_ids.shape[1])
            padded_positions = [
                [token_position + token_width - length for token_position in row_positions]
                for row_positions, length in zip(position_rows, lengths)
            ]
            if os.environ.get("OPTION_PROBE_DEBUG") == "1":
                print(
                    "OPTION_PROBE_DEBUG",
                    json.dumps(
                        {
                            "cohort_index": cohort_index,
                            "mapping_index": mapping_index,
                            "question_ids": cohort,
                            "lengths": lengths,
                            "token_width": token_width,
                            "position_rows": position_rows,
                            "padded_positions": padded_positions,
                            "attention_sums": attention_mask.sum(dim=1).tolist(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            collector = BatchedPositionCollector(parts.layers, padded_positions)
            try:
                with torch.inference_mode():
                    kwargs = {
                        "input_ids": input_ids.to(model_input_device(parts)),
                        "attention_mask": attention_mask.to(model_input_device(parts)),
                        "use_cache": False,
                        "return_dict": True,
                    }
                    try:
                        output = model(**kwargs, logits_to_keep=1)
                    except TypeError:
                        output = model(**kwargs)
                forwards += 1
                batch_values = collector.stacked().numpy().transpose(1, 0, 2, 3)
                residuals[mapping_index, question_indices] = batch_values
                raw = output.logits.detach().float().cpu()
                final = (
                    raw[:, 0]
                    if raw.shape[1] == 1
                    else raw[np.arange(len(cohort)), last_indices]
                )
                logits[mapping_index, question_indices] = np.stack(
                    [
                        torch.stack(
                            [
                                torch.logsumexp(final[row, variant_ids[letter]], dim=0)
                                for letter in LETTERS
                            ]
                        ).numpy()
                        for row in range(len(cohort))
                    ]
                )
            finally:
                collector.close()

            completed[mapping_index, question_indices] = True
            residuals.flush()
            np.save(completed_path, completed)
            _checkpoint(
                result_path,
                question_ids=np.asarray(qids),
                aggregated_ad_logits=logits,
                completed=completed,
            )

            if not audit_path.exists() and mapping_index == 0:
                first_ids = tokenizer(prompts[0], add_special_tokens=False)["input_ids"]
                line_tokens = [
                    tokenizer.decode([first_ids[position]]) for position in position_rows[0]
                ]
                audit_path.write_text(
                    json.dumps(
                        {
                            "question_ids": cohort,
                            "mapping_index": mapping_index,
                            "rendered_prompt": prompts[0],
                            "option_closing_newline_positions": position_rows[0],
                            "option_closing_newline_tokens": line_tokens,
                            "full_position_audit": audits[0],
                            "definition": (
                                "Post-block residual at the token overlapping the "
                                "option-closing newline; no other token position collected."
                            ),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if forwards:
            cohort_durations.append(time.monotonic() - cohort_started)
        print(
            f"option-newline choice probe collection: {int(completed.sum())}/"
            f"{completed.size}; cohort {cohort_index + 1}/{len(cohorts)}",
            flush=True,
        )

    metadata = {
        "experiment": "Six-permutation option-newline selected-answer probe cache",
        "config": config.as_dict(),
        "config_path": str(config_path),
        "plan_path": str(plan_path),
        "n_questions": n_questions,
        "n_mappings": n_mappings,
        "n_layers": n_layers,
        "width": width,
        "residual_shape": list(residual_shape),
        "residual_dtype": "float16",
        "anchor": "option-closing newline only",
        "physical_batch_size": 4,
        "complete_model_forwards_per_cohort": 6,
        "complete": bool(completed.all()),
        "elapsed_seconds_after_model_load": time.monotonic() - started_after_load,
        "completed_cohort_durations_seconds": cohort_durations,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


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
