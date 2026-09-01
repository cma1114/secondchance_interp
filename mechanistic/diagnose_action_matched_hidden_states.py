from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import LETTERS
from .collect_evaluation_gla_residual_writes import _aggregate_logits
from .collect_remapped_feedback_factorial import _messages, _remap_question
from .config import ExperimentConfig
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trusted-evaluation", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"][:4]]
    plan = json.loads(args.plan.read_text())
    plan_rows = {row["question_id"]: row for row in plan["rows"]}
    trusted = [
        json.loads(args.trusted_evaluation.read_text())["results"],
        json.loads(args.trusted_neutral.read_text())["results"],
    ]
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    device = model_input_device(parts)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]}) for letter in LETTERS
    }
    remapped = [
        _remap_question(questions[qid], plan_rows[qid]["new_to_original"])
        for qid in qids
    ]
    for condition_index, condition in enumerate(("incorrect_again", "lost_again")):
        messages = [
            _messages(config, questions[qid], remapped_question, condition)
            for qid, remapped_question in zip(qids, remapped)
        ]
        prompts = [
            render_chat(processor, row, config.disable_thinking, config.chat_serialization)
            for row in messages
        ]
        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
                logits_to_keep=1,
            )
        observed = _aggregate_logits(output, variant_ids)
        expected = np.asarray(
            [trusted[condition_index][qid]["aggregated_ad_logits"] for qid in qids],
            dtype=np.float32,
        )
        print(
            condition,
            "max_abs_error=", float(np.max(np.abs(observed - expected))),
            "argmax_match=", float(np.mean(observed.argmax(1) == expected.argmax(1))),
            "hidden_states=", len(output.hidden_states),
            flush=True,
        )


if __name__ == "__main__":
    main()
