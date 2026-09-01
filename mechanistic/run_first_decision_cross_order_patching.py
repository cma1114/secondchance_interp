from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .io import atomic_save_npz
from .jlens_collect import _token_offsets
from .modeling import (
    QWEN_EMPTY_THINKING,
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages, load_trials, prompt_hash
from .collect_remapped_behavior import _messages, _remap_question
from .run_historical_answer_intervention import _forward


LETTERS = "ABCD"
CONDITIONS = ("incorrect", "neutral")


def _hidden(output: Any):
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_hidden(output: Any, hidden: Any):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + list(output[1:])
    return hidden


def _load_mapping_plans(
    paths: list[Path], qids: list[str]
) -> list[dict[str, dict[str, str]]]:
    if len(paths) != 3:
        raise ValueError("Expected three alternative first-presentation mapping plans")
    plans = []
    for path in paths:
        lookup = {
            row["question_id"]: {
                "new_to_original": row["new_to_original"],
                "original_to_new": row["original_to_new"],
            }
            for row in json.loads(path.read_text())["rows"]
        }
        if not set(qids) <= set(lookup):
            raise ValueError(f"Mapping plan is incomplete: {path}")
        plans.append(lookup)
    identity = {letter: letter for letter in LETTERS}
    for qid in qids:
        occupied = [identity] + [plan[qid]["original_to_new"] for plan in plans]
        for content in LETTERS:
            if {mapping[content] for mapping in occupied} != set(LETTERS):
                raise ValueError(f"{qid}: content {content} does not occupy A-D once")
    return plans


def _question_ids(plan_path: Path) -> list[str]:
    payload = json.loads(plan_path.read_text())
    values = payload.get("question_ids", payload.get("confirmation_question_ids"))
    if not values:
        raise ValueError("Question plan has no question IDs")
    return list(values)


def _aggregate_logits(full_logits: Any, variant_ids: dict[str, list[int]]):
    import torch

    return torch.stack(
        [torch.logsumexp(full_logits[..., variant_ids[letter]], dim=-1) for letter in LETTERS],
        dim=-1,
    )


class DecisionResidualCollector:
    def __init__(self, parts: Any, position: int, readouts: list[int]) -> None:
        self.position = int(position)
        self.values: dict[int, np.ndarray] = {}
        self.handles = [
            parts.layers[readout - 1].register_forward_hook(self._hook(readout))
            for readout in readouts
        ]

    def _hook(self, readout: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = _hidden(output)
            self.values[readout] = hidden[0, self.position].detach().float().cpu().numpy()

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _select_live_donor(
    qid: str,
    w1: str,
    second_mapping: dict[str, dict[str, str]],
    candidates: list[dict[str, Any]],
    min_margin: float,
) -> dict[str, Any]:
    rows = []
    for candidate in candidates:
        logits = candidate["logits"]
        order = np.argsort(-logits, kind="stable")
        winner_letter = LETTERS[int(order[0])]
        mapping = candidate["mapping"]
        winner_content = mapping["new_to_original"][winner_letter]
        current_letter = second_mapping["original_to_new"][winner_content]
        rows.append(
            {
                "mapping_index": candidate["mapping_index"],
                "winner_first_letter": winner_letter,
                "winner_content": winner_content,
                "winner_current_letter": current_letter,
                "literal_letter_content_in_second": second_mapping["new_to_original"][winner_letter],
                "margin": float(logits[order[0]] - logits[order[1]]),
                "changed_winner": winner_content != w1,
                "letter_decoupled": winner_letter != current_letter,
            }
        )
    priority_groups = [
        [row for row in rows if row["changed_winner"] and row["letter_decoupled"] and row["margin"] >= min_margin],
        [row for row in rows if row["changed_winner"] and row["letter_decoupled"]],
        [row for row in rows if row["changed_winner"]],
        [row for row in rows if not row["changed_winner"] and row["letter_decoupled"]],
        rows,
    ]
    selected_group = next(group for group in priority_groups if group)
    donor = max(selected_group, key=lambda row: row["margin"])
    return {
        "question_id": qid,
        "recipient_winner_content": w1,
        "primary_letter_decoupled_changed_winner": bool(
            donor["changed_winner"] and donor["letter_decoupled"] and donor["margin"] >= min_margin
        ),
        "selection_min_margin": float(min_margin),
        "donor": donor,
        "all_candidate_winners": rows,
        "second_mapping": second_mapping,
    }


def _decision_position(tokenizer: Any, prompt: str) -> tuple[int, list[int]]:
    offsets = _token_offsets(tokenizer, prompt)
    assistant_header = "<|im_start|>assistant\n"
    start = prompt.find(assistant_header)
    if start < 0:
        raise RuntimeError("First assistant header is absent")
    scaffold_end = start + len(assistant_header) + len(QWEN_EMPTY_THINKING)
    candidates = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and right <= scaffold_end
    ]
    if not candidates:
        raise RuntimeError("Could not locate first-decision scaffold token")
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return int(candidates[-1]), [int(value) for value in ids]


def _kernel_batch(
    tokenizer: Any, target_prompt: str, filler_prompt: str, batch_size: int
) -> tuple[Any, Any, int, list[int]]:
    """Recreate the historical batch size and left-padding width for one target."""
    target_ids = [
        int(value)
        for value in tokenizer(target_prompt, add_special_tokens=False)["input_ids"]
    ]
    input_ids, attention_mask, _ = tokenize_batch(
        tokenizer, [target_prompt] * batch_size
    )
    desired_width = len(
        tokenizer(filler_prompt, add_special_tokens=False)["input_ids"]
    )
    if desired_width < input_ids.shape[1]:
        raise RuntimeError("Historical kernel width is shorter than the target")
    if desired_width > input_ids.shape[1]:
        import torch

        extra = desired_width - input_ids.shape[1]
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        input_ids = torch.cat(
            [
                torch.full((batch_size, extra), pad_id, dtype=input_ids.dtype),
                input_ids,
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                torch.zeros((batch_size, extra), dtype=attention_mask.dtype),
                attention_mask,
            ],
            dim=1,
        )
    left_pad = int(input_ids.shape[1] - len(target_ids))
    if input_ids[0, left_pad:].tolist() != target_ids:
        raise RuntimeError("Kernel batch changed the target token sequence")
    return input_ids, attention_mask, left_pad, target_ids


class CrossOrderDecisionPatcher:
    """Patch donor and identity residuals in matched rows for selected readouts."""

    def __init__(
        self,
        parts: Any,
        position: int,
        readouts: list[int],
        donor: dict[int, np.ndarray],
        identity: dict[int, np.ndarray],
        kernel_batch_size: int,
    ) -> None:
        self.position = int(position)
        self.readouts = list(readouts)
        self.donor = donor
        self.identity = identity
        self.kernel_batch_size = int(kernel_batch_size)
        self.records: dict[int, tuple[float, float]] = {}
        self.handles = [
            parts.layers[readout - 1].register_forward_hook(self._hook(readout))
            for readout in readouts
        ]

    def _hook(self, readout: int):
        def patch(_module: Any, _inputs: Any, output: Any):
            import torch

            hidden = _hidden(output)
            minimum = 2 * len(self.readouts)
            if hidden.shape[0] != self.kernel_batch_size or minimum > hidden.shape[0]:
                raise RuntimeError(
                    f"Expected kernel batch {self.kernel_batch_size} with at least "
                    f"{minimum} rows, got {hidden.shape[0]}"
                )
            row = self.readouts.index(readout)
            identity_row = len(self.readouts) + row
            donor = torch.from_numpy(self.donor[readout]).to(
                device=hidden.device, dtype=hidden.dtype
            )
            identity = torch.from_numpy(self.identity[readout]).to(
                device=hidden.device, dtype=hidden.dtype
            )
            current = hidden[identity_row, self.position].float()
            self.records[readout] = (
                float((current - identity.float()).norm().detach().cpu()),
                float((donor.float() - identity.float()).norm().detach().cpu()),
            )
            # Inference-mode in-place replacement avoids a full hidden-state
            # clone, allowing the intervention to retain the original batch-10
            # numerical regime on an 80 GB A100.
            hidden[row, self.position] = donor
            hidden[identity_row, self.position] = identity
            return output

        return patch

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _initialize(path: Path, qids: list[str], readouts: list[int]) -> dict[str, np.ndarray]:
    if path.exists():
        arrays = dict(np.load(path, allow_pickle=False))
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Question IDs changed")
        if arrays["readouts"].astype(int).tolist() != readouts:
            raise ValueError("Readouts changed")
        return arrays
    n, layers = len(qids), len(readouts)
    return {
        "question_ids": np.asarray(qids),
        "readouts": np.asarray(readouts, dtype=np.int16),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "donor_patched_logits": np.full((2, layers, n, 4), np.nan, dtype=np.float32),
        "identity_patched_logits": np.full((2, layers, n, 4), np.nan, dtype=np.float32),
        "identity_source_error_norm": np.full((2, layers, n), np.nan, dtype=np.float32),
        "donor_identity_delta_norm": np.full((2, layers, n), np.nan, dtype=np.float32),
    }


def run(
    config_path: Path,
    plan_path: Path,
    baseline_path: Path,
    second_mapping_plan_path: Path,
    mapping_plan_paths: list[Path],
    output: Path,
    readouts: list[int],
    layer_batch_size: int,
    kernel_batch_size: int,
    min_margin: float,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")
    if not readouts or min(readouts) < 1 or max(readouts) > 63:
        raise ValueError("Readouts must lie in 1--63")
    if kernel_batch_size < 2:
        raise ValueError("kernel_batch_size must be at least 2")
    if layer_batch_size > kernel_batch_size // 2:
        raise ValueError("layer_batch_size does not fit donor and identity rows")
    qids = _question_ids(plan_path)
    trials = load_trials(config.manifest_path, config.baseline_results_path, qids, None)
    baseline_rows = json.loads(baseline_path.read_text())["results"]
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    second_plan = {
        row["question_id"]: row
        for row in json.loads(second_mapping_plan_path.read_text())["rows"]
    }
    if not set(qids) <= set(second_plan):
        raise ValueError("Second-presentation mapping plan is incomplete")

    mapping_plans = _load_mapping_plans(mapping_plan_paths, qids)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }

    # Reconstruct the exact physical sequence width used by the established
    # full behavioral collection: manifest-order groups of config.batch_size.
    # GLA numerics depend on both batch size and left-padding width.
    if kernel_batch_size != config.batch_size:
        raise ValueError("kernel_batch_size must equal the established config.batch_size")
    all_qids = [row["id"] for row in manifest["questions"]]
    filler_by_condition: dict[str, dict[str, str]] = {
        condition: {} for condition in CONDITIONS
    }
    width_by_condition: dict[str, dict[str, int]] = {
        condition: {} for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        for start in range(0, len(all_qids), kernel_batch_size):
            group = all_qids[start : start + kernel_batch_size]
            prompts = []
            lengths = []
            for group_qid in group:
                remapped = _remap_question(
                    questions[group_qid], second_plan[group_qid]["new_to_original"]
                )
                rendered = render_chat(
                    processor,
                    _messages(config, questions[group_qid], remapped, condition),
                    config.disable_thinking,
                    config.chat_serialization,
                )
                prompts.append(rendered)
                lengths.append(
                    len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
                )
            filler = prompts[int(np.argmax(lengths))]
            width = int(max(lengths))
            for group_qid in group:
                filler_by_condition[condition][group_qid] = filler
                width_by_condition[condition][group_qid] = width
    for qid in qids:
        if width_by_condition["incorrect"][qid] != width_by_condition["neutral"][qid]:
            raise ValueError(f"Game/Neutral historical kernel widths differ for {qid}")
    output.mkdir(parents=True, exist_ok=True)
    donor_plan_path = output / "donor_plan.json"
    donor_lookup = {}
    if donor_plan_path.exists():
        donor_lookup = {
            row["question_id"]: row
            for row in json.loads(donor_plan_path.read_text()).get("rows", [])
        }
    result_path = output / "results.npz"
    arrays = _initialize(result_path, qids, readouts)

    audit_path = output / "prompt_audit.json"
    for qi, trial in enumerate(trials):
        if bool(arrays["completed"][qi]):
            continue
        qid = trial.question_id
        baseline_row = baseline_rows[qid]
        w1 = baseline_row.get("answer", baseline_row.get("subject_answer"))
        if w1 not in LETTERS:
            w1 = LETTERS[int(np.argmax(baseline_row["aggregated_ad_logits"]))]
        second_mapping = {
            "new_to_original": second_plan[qid]["new_to_original"],
            "original_to_new": second_plan[qid]["original_to_new"],
        }

        # Collect candidate donor states live and singly. This avoids importing
        # batch-kernel differences from the earlier residual caches.
        candidates = []
        for mapping_index, mapping_plan in enumerate(mapping_plans, 1):
            mapping = mapping_plan[qid]
            donor_question = _remap_question(
                questions[qid], mapping["new_to_original"]
            )
            donor_prompt = render_chat(
                processor,
                build_messages(
                    donor_question,
                    "baseline",
                    config.prompt_mode,
                    config.feedback_variant,
                ),
                config.disable_thinking,
                config.chat_serialization,
            )
            donor_position_unpadded, donor_ids = _decision_position(tokenizer, donor_prompt)
            donor_input, donor_mask, donor_left_pad, donor_batch_ids = _kernel_batch(
                tokenizer,
                donor_prompt,
                filler_by_condition["incorrect"][qid],
                kernel_batch_size,
            )
            if donor_ids != donor_batch_ids:
                raise RuntimeError("Donor offset-aware and model tokenizations disagree")
            donor_position = donor_left_pad + donor_position_unpadded
            collector = DecisionResidualCollector(parts, donor_position, readouts)
            try:
                donor_output = _forward(model, parts, donor_input, donor_mask)
            finally:
                collector.close()
            donor_logits = _aggregate_logits(
                donor_output.logits[0, -1].float(), variant_ids
            ).detach().cpu().numpy()
            candidates.append(
                {
                    "mapping_index": mapping_index,
                    "mapping": mapping,
                    "logits": donor_logits,
                    "residuals": collector.values,
                    "prompt": donor_prompt,
                    "position": donor_position,
                }
            )
        donor_row = _select_live_donor(
            qid, w1, second_mapping, candidates, min_margin
        )
        donor_candidate = candidates[int(donor_row["donor"]["mapping_index"]) - 1]
        donor_residuals = donor_candidate["residuals"]
        donor_lookup[qid] = donor_row
        ordered_donors = [donor_lookup[value] for value in qids if value in donor_lookup]
        donor_plan_path.write_text(
            json.dumps(
                {
                    "question_ids": qids,
                    "min_margin": min_margin,
                    "n_primary": sum(
                        row["primary_letter_decoupled_changed_winner"]
                        for row in ordered_donors
                    ),
                    "rows": ordered_donors,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        second_question = _remap_question(
            questions[qid], second_plan[qid]["new_to_original"]
        )
        for condition_index, condition in enumerate(CONDITIONS):
            messages = _messages(
                config, questions[qid], second_question, condition
            )
            prompt = render_chat(
                processor, messages, config.disable_thinking, config.chat_serialization
            )
            position, token_ids = _decision_position(tokenizer, prompt)
            input_ids, attention_mask, left_pad, batch_token_ids = _kernel_batch(
                tokenizer,
                prompt,
                filler_by_condition[condition][qid],
                kernel_batch_size,
            )
            if token_ids != batch_token_ids:
                raise RuntimeError("Offset-aware and model tokenizations disagree")
            position = left_pad + position
            identity_collector = DecisionResidualCollector(parts, position, readouts)
            try:
                natural_output = _forward(model, parts, input_ids, attention_mask)
            finally:
                identity_collector.close()
            identity_residuals = identity_collector.values
            natural = _aggregate_logits(
                natural_output.logits[0, -1].float(), variant_ids
            ).detach().cpu().numpy()
            arrays["natural_logits"][condition_index, qi] = natural

            for start in range(0, len(readouts), layer_batch_size):
                chunk = readouts[start : start + layer_batch_size]
                batch_ids, batch_mask, _patch_left_pad, _ = _kernel_batch(
                    tokenizer,
                    prompt,
                    filler_by_condition[condition][qid],
                    kernel_batch_size,
                )
                if _patch_left_pad != left_pad:
                    raise RuntimeError("Patch and natural batch widths differ")
                patcher = CrossOrderDecisionPatcher(
                    parts,
                    position,
                    chunk,
                    {readout: donor_residuals[readout] for readout in chunk},
                    {readout: identity_residuals[readout] for readout in chunk},
                    kernel_batch_size,
                )
                try:
                    patched_output = _forward(model, parts, batch_ids, batch_mask)
                finally:
                    patcher.close()
                raw = _aggregate_logits(
                    patched_output.logits[:, -1].float(), variant_ids
                ).detach().cpu().numpy()
                count = len(chunk)
                donor_corrected = natural[None, :] + raw[:count] - raw[count : 2 * count]
                identity_corrected = raw[count : 2 * count]
                indices = [readouts.index(readout) for readout in chunk]
                arrays["donor_patched_logits"][condition_index, indices, qi] = donor_corrected
                arrays["identity_patched_logits"][condition_index, indices, qi] = identity_corrected
                for local_index, readout in enumerate(chunk):
                    identity_error, donor_delta = patcher.records[readout]
                    layer_index = indices[local_index]
                    arrays["identity_source_error_norm"][condition_index, layer_index, qi] = identity_error
                    arrays["donor_identity_delta_norm"][condition_index, layer_index, qi] = donor_delta

            if not audit_path.exists():
                baseline_prompt = render_chat(
                    processor,
                    build_messages(
                        questions[qid], "baseline", config.prompt_mode,
                        config.feedback_variant,
                    ),
                    config.disable_thinking,
                    config.chat_serialization,
                )
                baseline_ids = tokenizer(
                    baseline_prompt, add_special_tokens=False
                )["input_ids"]
                audit_path.write_text(
                    json.dumps(
                        {
                            "question_id": qid,
                            "condition": condition,
                            "prompt_hash": prompt_hash(prompt),
                            "decision_position": position,
                            "decision_token_id": token_ids[position - left_pad],
                            "decision_token": tokenizer.decode([token_ids[position - left_pad]]),
                            "recipient_prefix_matches_baseline_through_decision": (
                                token_ids[: position - left_pad + 1]
                                == baseline_ids[: position - left_pad + 1]
                            ),
                            "donor": donor_row,
                            "donor_prompt": donor_candidate["prompt"],
                            "donor_decision_position_physical": donor_candidate["position"],
                            "recipient_left_padding": left_pad,
                            "historical_kernel_width": width_by_condition[condition][qid],
                            "rendered_prompt": prompt,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )

        arrays["completed"][qi] = True
        atomic_save_npz(result_path, **arrays)
        done = int(arrays["completed"].sum())
        if done == 1 or done % 5 == 0 or done == len(qids):
            print(f"cross-order decision patching: {done}/{len(qids)}", flush=True)

    donor_rows = [donor_lookup[qid] for qid in qids]
    metadata = {
        "config": config.as_dict(),
        "plan_path": str(plan_path),
        "baseline_path": str(baseline_path),
        "second_mapping_plan_path": str(second_mapping_plan_path),
        "mapping_plan_paths": [str(path) for path in mapping_plan_paths],
        "readouts": readouts,
        "layer_batch_size": layer_batch_size,
        "kernel_batch_size": kernel_batch_size,
        "min_donor_margin": min_margin,
        "n_questions": len(qids),
        "n_primary_letter_decoupled_changed_winner": sum(
            row["primary_letter_decoupled_changed_winner"] for row in donor_rows
        ),
        "intervention": (
            "At one post-block readout, replace the complete historical first-decision "
            "residual with the same-question residual from another option order."
        ),
        "numerical_control": (
            f"All source, natural, and patched executions retain the established batch-{kernel_batch_size} "
            "kernel regime. Saved donor logits equal natural logits plus donor-patched "
            "minus identity-patched logits from the same physical batch. Unused rows "
            "retain the target prompt."
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
    parser.add_argument("--layer-batch-size", type=int, default=4)
    parser.add_argument("--kernel-batch-size", type=int, default=10)
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
        args.layer_batch_size,
        args.kernel_batch_size,
        args.min_margin,
    )


if __name__ == "__main__":
    main()
