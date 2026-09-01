from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .downstream_source_intervention import (
    BatchedDownstreamAttentionAblator,
    BatchedGDNSourceWriteAblator,
)
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import load_trials
from .run_feedback_token_trajectory_swap import _render_and_locate


CONDITIONS = ("incorrect", "neutral")
SCENARIOS = (
    ("attention_evaluation", "attention", "evaluation"),
    ("attention_action", "attention", "action"),
    ("attention_joint", "attention", "joint"),
    ("gdn_evaluation", "gdn", "evaluation"),
    ("gdn_action", "gdn", "action"),
    ("gdn_joint", "gdn", "joint"),
    ("both_evaluation", "both", "evaluation"),
    ("both_action", "both", "action"),
    ("both_joint", "both", "joint"),
)


def _selected_positions(feedback_positions: list[int], source: str) -> list[int]:
    # evaluation is the only semantically differing token in the first clause:
    # Game `incorrect` versus Neutral `lost`.  action is the complete aligned
    # five-token imperative, including its terminal period.
    if source == "evaluation":
        indices = (3,)
    elif source == "action":
        indices = tuple(range(5, 10))
    elif source == "joint":
        indices = (3, *range(5, 10))
    else:
        raise ValueError(f"Unknown source group: {source}")
    return [feedback_positions[index] for index in indices]


def _final_logits(output, answer_ids: list[int]) -> np.ndarray:
    logits = output.logits
    final = logits[:, 0] if logits.shape[1] == 1 else logits[:, -1]
    return final[:, answer_ids].detach().float().cpu().numpy()


def _forward(model, parts, input_ids, attention_mask, answer_ids):
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
            output = model(**kwargs, logits_to_keep=1)
        except TypeError:
            output = model(**kwargs)
    return _final_logits(output, answer_ids)


def run(config_path: Path, output: Path, question_ids_path: Path | None, max_questions: int | None) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")
    if config.attn_implementation != "eager":
        raise ValueError("Exact ordinary-attention ablation requires eager attention")

    question_ids = None
    if question_ids_path is not None:
        plan = json.loads(question_ids_path.read_text())
        question_ids = plan.get("question_ids", plan.get("confirmation_question_ids"))
        if question_ids is None:
            raise ValueError("Question-ID file has no question_ids field")
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        question_ids,
        max_questions,
    )
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    answer_ids = [resolved[letter][0][1] for letter in "ABCD"]

    output.parent.mkdir(parents=True, exist_ok=True)
    qids = np.asarray([trial.question_id for trial in trials])
    scenario_ids = np.asarray([scenario[0] for scenario in SCENARIOS])
    if output.exists():
        with np.load(output, allow_pickle=False) as existing:
            arrays = {key: existing[key] for key in existing.files}
        if arrays["question_ids"].astype(str).tolist() != qids.astype(str).tolist():
            raise ValueError("Existing output uses different question IDs")
        if arrays["scenario_ids"].astype(str).tolist() != scenario_ids.astype(str).tolist():
            raise ValueError("Existing output uses different scenarios")
    else:
        arrays = {
            "question_ids": qids,
            "scenario_ids": scenario_ids,
            "completed": np.zeros((2, len(trials)), dtype=bool),
            "natural_logits": np.full((2, len(trials), 4), np.nan, dtype=np.float32),
            "intervened_logits": np.full(
                (2, len(SCENARIOS), len(trials), 4), np.nan, dtype=np.float32
            ),
            "batch_control_minus_natural": np.full(
                (2, len(trials), 4), np.nan, dtype=np.float32
            ),
        }

    audit = None
    for qi, trial in enumerate(trials):
        for ci, condition in enumerate(CONDITIONS):
            if arrays["completed"][ci, qi]:
                continue
            prompt_data = _render_and_locate(processor, tokenizer, config, trial, condition)
            natural = _forward(
                model, parts, prompt_data["input_ids"], prompt_data["attention_mask"], answer_ids
            )[0]
            arrays["natural_logits"][ci, qi] = natural

            prompts = [prompt_data["prompt"]] * (len(SCENARIOS) + 1)
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
            attention_specs: dict[int, list[int]] = {}
            gdn_specs: dict[int, list[int]] = {}
            for row, (_scenario_id, route, source) in enumerate(SCENARIOS):
                selected = _selected_positions(prompt_data["positions"], source)
                if route in {"attention", "both"}:
                    attention_specs[row] = selected
                if route in {"gdn", "both"}:
                    gdn_specs[row] = selected
            attention_ablator = BatchedDownstreamAttentionAblator(parts, attention_specs)
            gdn_ablator = BatchedGDNSourceWriteAblator(parts, gdn_specs)
            try:
                raw = _forward(model, parts, input_ids, attention_mask, answer_ids)
                attention_ablator.assert_fired()
                gdn_ablator.assert_fired()
            finally:
                gdn_ablator.close()
                attention_ablator.close()
            control = raw[-1]
            corrected = natural[None, :] + raw[:-1] - control[None, :]
            arrays["intervened_logits"][ci, :, qi] = corrected
            arrays["batch_control_minus_natural"][ci, qi] = control - natural
            arrays["completed"][ci, qi] = True
            atomic_save_npz(output, **arrays)

            if audit is None:
                audit = {
                    "question_id": trial.question_id,
                    "condition": condition,
                    "feedback_tokens": list(prompt_data["tokens"]),
                    "feedback_positions_zero_based": prompt_data["positions"],
                    "source_groups": {
                        source: _selected_positions(prompt_data["positions"], source)
                        for source in ("evaluation", "action", "joint")
                    },
                    "final_query_position_zero_based": len(prompt_data["ids"]) - 1,
                }
        if qi == 0 or (qi + 1) % 10 == 0 or qi + 1 == len(trials):
            print(
                f"Downstream source ablation: {int(arrays['completed'].sum())}/"
                f"{2 * len(trials)} condition-trials",
                flush=True,
            )

    ordinary_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    ]
    gdn_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    metadata = {
        "config": config.as_dict(),
        "n_questions": len(trials),
        "scenarios": [
            {"id": scenario_id, "route": route, "source": source}
            for scenario_id, route, source in SCENARIOS
        ],
        "ordinary_attention_layers_zero_based": ordinary_layers,
        "gdn_layers_zero_based": gdn_layers,
        "ordinary_attention_intervention": (
            "in every ordinary-attention block and every head, set each selected "
            "source key's logits to negative infinity for every causally later query"
        ),
        "gdn_intervention": (
            "in every Gated DeltaNet block and every value head, set beta=0 at "
            "the selected source positions, removing their direct recurrent writes"
        ),
        "batch_drift_correction": (
            "corrected intervention = natural single-row logits + intervention-row "
            "logits minus unmodified same-batch control logits"
        ),
        "first_prompt_audit": audit,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate all downstream source routes")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-ids", type=Path)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    run(args.config, args.output, args.question_ids, args.max_questions)


if __name__ == "__main__":
    main()
