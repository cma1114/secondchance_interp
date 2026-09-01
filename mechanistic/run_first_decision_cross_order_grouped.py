from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .collect_remapped_behavior import _messages, _remap_question
from .config import ExperimentConfig
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages, load_trials, prompt_hash
from .run_first_decision_cross_order_patching import (
    CONDITIONS,
    LETTERS,
    _aggregate_logits,
    _decision_position,
    _hidden,
    _initialize,
    _load_mapping_plans,
    _question_ids,
    _select_live_donor,
)
from .run_historical_answer_intervention import _forward


class BatchDecisionCollector:
    def __init__(self, parts: Any, positions: list[int], readouts: list[int]) -> None:
        self.positions = positions
        self.values: dict[int, np.ndarray] = {}
        self.handles = [
            parts.layers[readout - 1].register_forward_hook(self._hook(readout))
            for readout in readouts
        ]

    def _hook(self, readout: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = _hidden(output)
            self.values[readout] = np.stack(
                [
                    hidden[row, position].detach().float().cpu().numpy()
                    for row, position in enumerate(self.positions)
                ]
            )

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class SingleTargetPatcher:
    def __init__(
        self,
        parts: Any,
        target_row: int,
        position: int,
        readout: int,
        donor: np.ndarray,
    ) -> None:
        self.target_row = int(target_row)
        self.position = int(position)
        self.donor = donor
        self.handle = parts.layers[readout - 1].register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any):
        import torch

        hidden = _hidden(output)
        donor = torch.from_numpy(self.donor).to(
            device=hidden.device, dtype=hidden.dtype
        )
        hidden[self.target_row, self.position] = donor
        return output

    def close(self) -> None:
        self.handle.remove()


def _batch_inputs(tokenizer: Any, prompts: list[str]) -> tuple[Any, Any, list[int], list[list[int]]]:
    positions = []
    token_rows = []
    for prompt in prompts:
        position, ids = _decision_position(tokenizer, prompt)
        positions.append(position)
        token_rows.append(ids)
    input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
    width = int(input_ids.shape[1])
    physical = []
    for row, (position, ids) in enumerate(zip(positions, token_rows)):
        left_pad = width - len(ids)
        if input_ids[row, left_pad:].tolist() != ids:
            raise RuntimeError("Batch tokenization changed a prompt")
        physical.append(left_pad + position)
    return input_ids, attention_mask, physical, token_rows


def run(
    config_path: Path,
    plan_path: Path,
    baseline_path: Path,
    second_mapping_plan_path: Path,
    mapping_plan_paths: list[Path],
    output: Path,
    readouts: list[int],
    min_margin: float,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires the historical batch-size-4 SDPA regime")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")
    if not readouts or min(readouts) < 1 or max(readouts) > 63:
        raise ValueError("Readouts must lie in 1--63")

    qids = _question_ids(plan_path)
    qid_set = set(qids)
    trials = load_trials(config.manifest_path, config.baseline_results_path, qids, None)
    trial_by_qid = {trial.question_id: trial for trial in trials}
    baseline_rows = json.loads(baseline_path.read_text())["results"]
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    all_qids = [row["id"] for row in manifest["questions"]]
    second_plan = {
        row["question_id"]: row
        for row in json.loads(second_mapping_plan_path.read_text())["rows"]
    }
    mapping_plans = _load_mapping_plans(mapping_plan_paths, all_qids)

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }

    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.npz"
    arrays = _initialize(result_path, qids, readouts)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    donor_plan_path = output / "donor_plan.json"
    donor_lookup: dict[str, dict[str, Any]] = {}
    if donor_plan_path.exists():
        donor_lookup = {
            row["question_id"]: row
            for row in json.loads(donor_plan_path.read_text()).get("rows", [])
        }
    audit_path = output / "prompt_audit.json"

    for group_start in range(0, len(all_qids), config.batch_size):
        group_qids = all_qids[group_start : group_start + config.batch_size]
        target_qids = [
            qid for qid in group_qids
            if qid in qid_set and not bool(arrays["completed"][qid_index[qid]])
        ]
        if not target_qids:
            continue

        candidate_sets = []
        for mapping_index, mapping_plan in enumerate(mapping_plans, 1):
            prompts = []
            for qid in group_qids:
                remapped = _remap_question(
                    questions[qid], mapping_plan[qid]["new_to_original"]
                )
                prompts.append(
                    render_chat(
                        processor,
                        build_messages(
                            remapped, "baseline", config.prompt_mode,
                            config.feedback_variant,
                        ),
                        config.disable_thinking,
                        config.chat_serialization,
                    )
                )
            input_ids, attention_mask, positions, _ = _batch_inputs(tokenizer, prompts)
            collector = BatchDecisionCollector(parts, positions, readouts)
            try:
                model_output = _forward(model, parts, input_ids, attention_mask)
            finally:
                collector.close()
            logits = _aggregate_logits(model_output.logits[:, -1].float(), variant_ids)
            candidate_sets.append(
                {
                    "mapping_index": mapping_index,
                    "mapping_plan": mapping_plan,
                    "logits": logits.detach().cpu().numpy(),
                    "residuals": collector.values,
                    "prompts": prompts,
                    "positions": positions,
                }
            )

        natural_batches: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            prompts = []
            for qid in group_qids:
                remapped = _remap_question(
                    questions[qid], second_plan[qid]["new_to_original"]
                )
                prompts.append(
                    render_chat(
                        processor,
                        _messages(config, questions[qid], remapped, condition),
                        config.disable_thinking,
                        config.chat_serialization,
                    )
                )
            input_ids, attention_mask, positions, token_rows = _batch_inputs(
                tokenizer, prompts
            )
            collector = BatchDecisionCollector(parts, positions, readouts)
            try:
                model_output = _forward(model, parts, input_ids, attention_mask)
            finally:
                collector.close()
            logits = _aggregate_logits(model_output.logits[:, -1].float(), variant_ids)
            natural_batches[condition] = {
                "prompts": prompts,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "positions": positions,
                "token_rows": token_rows,
                "logits": logits.detach().cpu().numpy(),
                "residuals": collector.values,
            }

        for target_qid in target_qids:
            target_index = qid_index[target_qid]
            row_index = group_qids.index(target_qid)
            baseline_row = baseline_rows[target_qid]
            w1 = baseline_row.get("answer", baseline_row.get("subject_answer"))
            if w1 not in LETTERS:
                w1 = LETTERS[int(np.argmax(baseline_row["aggregated_ad_logits"]))]
            second_mapping = {
                "new_to_original": second_plan[target_qid]["new_to_original"],
                "original_to_new": second_plan[target_qid]["original_to_new"],
            }
            candidates = []
            for candidate_set in candidate_sets:
                mapping = candidate_set["mapping_plan"][target_qid]
                candidates.append(
                    {
                        "mapping_index": candidate_set["mapping_index"],
                        "mapping": mapping,
                        "logits": candidate_set["logits"][row_index],
                        "residuals": {
                            readout: candidate_set["residuals"][readout][row_index]
                            for readout in readouts
                        },
                    }
                )
            donor_row = _select_live_donor(
                target_qid, w1, second_mapping, candidates, min_margin
            )
            donor_candidate = candidates[donor_row["donor"]["mapping_index"] - 1]
            donor_lookup[target_qid] = donor_row

            for condition_index, condition in enumerate(CONDITIONS):
                batch = natural_batches[condition]
                natural = batch["logits"][row_index]
                arrays["natural_logits"][condition_index, target_index] = natural
                for layer_index, readout in enumerate(readouts):
                    patcher = SingleTargetPatcher(
                        parts,
                        row_index,
                        batch["positions"][row_index],
                        readout,
                        donor_candidate["residuals"][readout],
                    )
                    try:
                        patched_output = _forward(
                            model, parts, batch["input_ids"], batch["attention_mask"]
                        )
                    finally:
                        patcher.close()
                    patched = _aggregate_logits(
                        patched_output.logits[row_index, -1].float(), variant_ids
                    ).detach().cpu().numpy()
                    arrays["donor_patched_logits"][
                        condition_index, layer_index, target_index
                    ] = patched
                    arrays["identity_patched_logits"][
                        condition_index, layer_index, target_index
                    ] = natural
                    arrays["identity_source_error_norm"][
                        condition_index, layer_index, target_index
                    ] = 0.0
                    arrays["donor_identity_delta_norm"][
                        condition_index, layer_index, target_index
                    ] = float(
                        np.linalg.norm(
                            donor_candidate["residuals"][readout]
                            - batch["residuals"][readout][row_index]
                        )
                    )

                if not audit_path.exists():
                    prompt = batch["prompts"][row_index]
                    audit_path.write_text(
                        json.dumps(
                            {
                                "question_id": target_qid,
                                "condition": condition,
                                "historical_group_qids": group_qids,
                                "target_row": row_index,
                                "prompt_hash": prompt_hash(prompt),
                                "decision_position_physical": batch["positions"][row_index],
                                "donor": donor_row,
                                "rendered_prompt": prompt,
                            },
                            indent=2,
                            sort_keys=True,
                        ) + "\n"
                    )

            arrays["completed"][target_index] = True
            atomic_save_npz(result_path, **arrays)
            ordered = [donor_lookup[qid] for qid in qids if qid in donor_lookup]
            donor_plan_path.write_text(
                json.dumps(
                    {
                        "question_ids": qids,
                        "min_margin": min_margin,
                        "n_primary": sum(
                            row["primary_letter_decoupled_changed_winner"]
                            for row in ordered
                        ),
                        "rows": ordered,
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n"
            )
            done = int(arrays["completed"].sum())
            if done == 1 or done % 5 == 0 or done == len(qids):
                print(f"grouped cross-order patching: {done}/{len(qids)}", flush=True)

    donor_rows = [donor_lookup[qid] for qid in qids]
    metadata = {
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "baseline_path": str(baseline_path),
        "second_mapping_plan_path": str(second_mapping_plan_path),
        "mapping_plan_paths": [str(path) for path in mapping_plan_paths],
        "readouts": readouts,
        "min_donor_margin": min_margin,
        "n_questions": len(qids),
        "n_primary_letter_decoupled_changed_winner": sum(
            row["primary_letter_decoupled_changed_winner"] for row in donor_rows
        ),
        "intervention": (
            "Preserve each target's exact historical four-question cohort and patch "
            "only that row's complete first-decision residual from another option order."
        ),
        "numerical_control": (
            "Natural and patched executions use the identical historical cohort, row, "
            "batch size, padding, SDPA implementation, and software environment."
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
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--second-mapping-plan", type=Path, required=True)
    parser.add_argument("--mapping-plans", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readouts", nargs="+", type=int, required=True)
    parser.add_argument("--min-margin", type=float, default=0.5)
    args = parser.parse_args()
    run(
        args.config,
        args.plan,
        args.baseline,
        args.second_mapping_plan,
        args.mapping_plans,
        args.output,
        args.readouts,
        args.min_margin,
    )


if __name__ == "__main__":
    main()
