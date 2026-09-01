from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .downstream_source_intervention import (
    BatchedSDPADownstreamAttentionAblator,
    BatchedSelectiveGDNSourceWriteAblator,
)
from .io import atomic_save_npz
from .modeling import get_tokenizer, load_model_and_processor, resolve_answer_tokens
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward


SCENARIOS = ("gla_write", "attention_read", "joint")


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Question IDs changed")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "completed": np.zeros(n, dtype=bool),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "lesioned_logits": np.full((2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32),
    }


def run(
    config_path: Path,
    remapping_plan_path: Path,
    trusted_evaluation_path: Path,
    trusted_neutral_path: Path,
    output_dir: Path,
    max_cohorts: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires exact historical batch-size-4 SDPA execution")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history prompts")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    if max_cohorts is not None:
        qids = qids[: int(max_cohorts) * config.batch_size]
    mappings = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    trusted = [
        json.loads(trusted_evaluation_path.read_text())["results"],
        json.loads(trusted_neutral_path.read_text())["results"],
    ]

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    gla_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    attention_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    ]
    if len(gla_layers) != 48 or len(attention_layers) != 16:
        raise RuntimeError(
            f"Expected 48 GLA and 16 attention layers; got {len(gla_layers)} and {len(attention_layers)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = output_dir / "prompt_audit.json"
    started = time.monotonic()

    for cohort_index, start in enumerate(range(0, len(qids), config.batch_size)):
        cohort = qids[start : start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if np.all(arrays["completed"][indices]):
            continue
        cohort_started = time.monotonic()
        batches = [
            _build_batch(config, processor, tokenizer, questions, mappings, cohort, condition)
            for condition in CONDITIONS
        ]
        if batches[0]["positions"] != batches[1]["positions"]:
            raise RuntimeError("Action-closing periods are not physically aligned")

        for ci, batch in enumerate(batches):
            natural = _aggregate_logits(
                _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                variant_ids,
            )
            specs = {row: [position] for row, position in enumerate(batch["positions"])}
            gla_specs = {
                row: ([position], gla_layers)
                for row, position in enumerate(batch["positions"])
            }

            gla = BatchedSelectiveGDNSourceWriteAblator(
                parts, gla_specs, preserve_source_output=True
            )
            try:
                gla_logits = _aggregate_logits(
                    _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                    variant_ids,
                )
                gla.assert_fired()
            finally:
                gla.close()

            attention = BatchedSDPADownstreamAttentionAblator(parts, specs)
            try:
                attention_logits = _aggregate_logits(
                    _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                    variant_ids,
                )
                attention.assert_fired()
            finally:
                attention.close()

            gla = BatchedSelectiveGDNSourceWriteAblator(
                parts, gla_specs, preserve_source_output=True
            )
            attention = BatchedSDPADownstreamAttentionAblator(parts, specs)
            try:
                joint_logits = _aggregate_logits(
                    _forward(model, parts, batch["input_ids"], batch["attention_mask"]),
                    variant_ids,
                )
                gla.assert_fired()
                attention.assert_fired()
            finally:
                attention.close()
                gla.close()

            scenario_logits = (gla_logits, attention_logits, joint_logits)
            for local, qid in enumerate(cohort):
                qi = qid_index[qid]
                arrays["same_batch_natural_logits"][ci, qi] = natural[local]
                arrays["trusted_natural_logits"][ci, qi] = np.asarray(
                    trusted[ci][qid]["aggregated_ad_logits"], dtype=np.float32
                )
                for si, values in enumerate(scenario_logits):
                    arrays["lesioned_logits"][ci, si, qi] = values[local]

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        elapsed = time.monotonic() - cohort_started
        print(
            f"action-period source lesion: {int(arrays['completed'].sum())}/{len(qids)} questions; "
            f"cohort_seconds={elapsed:.1f}",
            flush=True,
        )
        if not audit_path.exists():
            audit_path.write_text(json.dumps({
                "question_ids": cohort,
                "conditions": {
                    condition: {
                        "feedback": batches[ci]["audits"][0]["feedback"],
                        "action_period_token": batches[ci]["audits"][0]["decoded_tokens"][1],
                        "action_period_position_zero_based_unpadded": batches[ci]["audits"][0]["unpadded_token_positions_zero_based"][1],
                        "rendered_prompt": batches[ci]["prompts"][0],
                    }
                    for ci, condition in enumerate(CONDITIONS)
                },
            }, indent=2, ensure_ascii=False) + "\n")

    metadata: dict[str, Any] = {
        "config": config.as_dict(),
        "remapping_plan": str(remapping_plan_path),
        "trusted_evaluation": str(trusted_evaluation_path),
        "trusted_neutral": str(trusted_neutral_path),
        "n_questions": len(qids),
        "conditions": list(CONDITIONS),
        "scenarios": list(SCENARIOS),
        "historical_batch_size": config.batch_size,
        "complete_model_forwards_per_cohort": 8,
        "gla_layers_zero_based": gla_layers,
        "attention_layers_zero_based": attention_layers,
        "source": "Only the terminal period token of the shared action clause 'Choose the answer again.'",
        "interventions": {
            "gla_write": "Set beta=0 only at the source period in all 48 GLAs, preserving every accumulated pre-period state.",
            "attention_read": "Block only later conventional-attention queries from reading the source period in all 16 ordinary-attention layers.",
            "joint": "Apply both source-specific lesions simultaneously.",
        },
        "preserve_source_output": True,
        "elapsed_seconds_after_load": time.monotonic() - started,
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
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove only the action-ending period's downstream GLA and attention routes"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--trusted-evaluation", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.remapping_plan,
        args.trusted_evaluation,
        args.trusted_neutral,
        args.output_dir,
        args.max_cohorts,
    )


if __name__ == "__main__":
    main()
