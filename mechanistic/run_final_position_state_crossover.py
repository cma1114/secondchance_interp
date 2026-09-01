from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .io import atomic_save_npz
from .modeling import (
    ResidualCollector,
    get_tokenizer,
    load_model_and_processor,
    resolve_answer_tokens,
    tokenize_batch,
)
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward
from .sublayer import (
    ComponentOutputCollector,
    ComponentOutputPatcher,
    ComponentTarget,
    _hidden,
    _replace_hidden,
)


RESIDUAL_SCENARIOS = tuple(f"residual_after_L{layer:02d}_swapped" for layer in range(1, 65))
GLOBAL_COMPONENT_SCENARIOS = (
    "all_mixers_swapped",
    "all_mlps_swapped",
    "all_mixers_and_mlps_swapped",
)
SCENARIOS = ("natural", *RESIDUAL_SCENARIOS, *GLOBAL_COMPONENT_SCENARIOS)


def _hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class FinalPositionResidualReplacer:
    """Replace only the exact final-token post-layer residual by paired donor state."""

    def __init__(
        self,
        parts: Any,
        layer_index: int,
        last_indices: list[int],
        source: Any,
    ) -> None:
        if layer_index < 0 or layer_index >= len(parts.layers):
            raise ValueError(f"Invalid zero-based layer index {layer_index}")
        if len(last_indices) != int(source.shape[0]):
            raise ValueError("One donor residual is required per batch row")
        self.last_indices = tuple(int(value) for value in last_indices)
        self.source = source
        self.handle = parts.layers[layer_index].register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = _hidden(output)
        replacement = self.source.to(device=hidden.device, dtype=hidden.dtype)
        if replacement.shape != (hidden.shape[0], hidden.shape[-1]):
            raise RuntimeError(
                f"Donor residual shape {tuple(replacement.shape)} does not match "
                f"{(hidden.shape[0], hidden.shape[-1])}"
            )
        changed = hidden.clone()
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        cols = torch.as_tensor(self.last_indices, device=hidden.device)
        changed[rows, cols] = replacement
        return _replace_hidden(output, changed)

    def close(self) -> None:
        self.handle.remove()


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise RuntimeError("Existing checkpoint uses different questions")
        if arrays["scenario_ids"].astype(str).tolist() != list(SCENARIOS):
            raise RuntimeError("Existing checkpoint uses different scenarios")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "scenario_ids": np.asarray(SCENARIOS),
        "completed": np.zeros(n, dtype=bool),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "scenario_final_logits_raw": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
        "scenario_final_logits": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
        "residual_capture_max_abs_reconstruction_error": np.full(
            (2, n), np.nan, dtype=np.float32
        ),
    }


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(args.config)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires canonical batch-size-4 SDPA execution")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires canonical empty-history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires action-matched incorrect/lost feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires canonical raw Qwen ChatML serialization")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {str(row["id"]): row for row in manifest["questions"]}
    all_qids = [str(row["id"]) for row in manifest["questions"]]
    qids = (
        all_qids[: int(args.max_cohorts) * config.batch_size]
        if args.max_cohorts is not None
        else all_qids
    )
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    trusted = [
        json.loads(args.trusted_game.read_text())["results"],
        json.loads(args.trusted_neutral.read_text())["results"],
    ]

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _token, token_id in resolved[letter]})
        for letter in LETTERS
    }
    mixer_targets = [ComponentTarget(layer, "mixer") for layer in range(64)]
    mlp_targets = [ComponentTarget(layer, "mlp") for layer in range(64)]
    all_targets = [*mixer_targets, *mlp_targets]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.npz"
    arrays = _initialize(result_path, qids)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = args.output_dir / "prompt_audit.json"
    durations: list[float] = []
    started = time.monotonic()

    for cohort_start in range(0, len(qids), config.batch_size):
        cohort = qids[cohort_start : cohort_start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if arrays["completed"][indices].all():
            continue
        if len(cohort) != config.batch_size:
            raise RuntimeError("Canonical questions must form complete cohorts")
        cohort_started = time.monotonic()
        condition_batches = [
            _build_batch(
                config, processor, tokenizer, questions, mappings, cohort, condition
            )
            for condition in CONDITIONS
        ]
        canonical_width = int(condition_batches[0]["input_ids"].shape[1])
        cohort_audit: dict[str, Any] = {"question_ids": cohort, "pairs": []}

        for pair_start in range(0, len(cohort), 2):
            pair = cohort[pair_start : pair_start + 2]
            prompts = (
                condition_batches[0]["prompts"][pair_start : pair_start + 2]
                + condition_batches[1]["prompts"][pair_start : pair_start + 2]
            )
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
            if input_ids.shape[1] > canonical_width:
                raise RuntimeError("Paired batch exceeds canonical cohort width")
            if input_ids.shape[1] < canonical_width:
                pad = canonical_width - int(input_ids.shape[1])
                input_ids = torch.nn.functional.pad(
                    input_ids, (pad, 0), value=int(tokenizer.pad_token_id)
                )
                attention_mask = torch.nn.functional.pad(attention_mask, (pad, 0), value=0)
            width = int(input_ids.shape[1])
            last_indices = [width - 1] * 4
            pair_audit: dict[str, Any] = {"question_ids": pair, "rows": []}

            for condition_index, condition in enumerate(CONDITIONS):
                for local, qid in enumerate(pair):
                    row = 2 * condition_index + local
                    prompt_ids = [
                        int(value)
                        for value in tokenizer(prompts[row], add_special_tokens=False)[
                            "input_ids"
                        ]
                    ]
                    left_pad = width - len(prompt_ids)
                    if input_ids[row, left_pad:].tolist() != prompt_ids:
                        raise RuntimeError("Paired tokenization changed the prompt")
                    digest = _hash(prompts[row])
                    if digest != trusted[condition_index][qid]["prompt_hash"]:
                        raise RuntimeError("Prompt hash differs from trusted natural run")
                    qi = qid_index[qid]
                    arrays["prompt_hashes"][condition_index, qi] = digest
                    arrays["trusted_natural_logits"][condition_index, qi] = np.asarray(
                        trusted[condition_index][qid]["aggregated_ad_logits"],
                        dtype=np.float32,
                    )
                    pair_audit["rows"].append(
                        {
                            "row": row,
                            "condition": condition,
                            "question_id": qid,
                            "prompt_hash": digest,
                            "left_padding": left_pad,
                            "final_query_position": width - 1,
                            "final_query_token": tokenizer.decode([int(input_ids[row, -1])]),
                        }
                    )

            residual_collector = ResidualCollector(parts, last_indices)
            component_collector = ComponentOutputCollector(parts, all_targets, last_indices)
            try:
                natural_output = _aggregate_logits(
                    _forward(model, parts, input_ids, attention_mask), variant_ids
                )
            finally:
                component_collector.close()
                residual_collector.close()
            residuals = residual_collector.stacked()  # row x boundary(0..64) x width
            components = component_collector.values
            donors = torch.as_tensor([2, 3, 0, 1], dtype=torch.long)
            donor_components = {
                key: value.index_select(0, donors) for key, value in components.items()
            }
            scenario_outputs = [natural_output]

            for layer in range(1, 65):
                patcher = FinalPositionResidualReplacer(
                    parts,
                    layer - 1,
                    last_indices,
                    residuals[:, layer].index_select(0, donors),
                )
                try:
                    output = _aggregate_logits(
                        _forward(model, parts, input_ids, attention_mask), variant_ids
                    )
                finally:
                    patcher.close()
                scenario_outputs.append(output)

            for targets in (mixer_targets, mlp_targets, all_targets):
                patcher = ComponentOutputPatcher(
                    parts, targets, donor_components, last_indices
                )
                try:
                    output = _aggregate_logits(
                        _forward(model, parts, input_ids, attention_mask), variant_ids
                    )
                finally:
                    patcher.close()
                scenario_outputs.append(output)

            scenario_logits = np.stack(scenario_outputs, axis=0)
            if scenario_logits.shape != (len(SCENARIOS), 4, 4):
                raise RuntimeError(f"Unexpected scenario output shape {scenario_logits.shape}")
            for condition_index in range(2):
                for local, qid in enumerate(pair):
                    row = 2 * condition_index + local
                    qi = qid_index[qid]
                    trusted_logits = arrays["trusted_natural_logits"][condition_index, qi]
                    arrays["same_batch_natural_logits"][condition_index, qi] = natural_output[row]
                    arrays["scenario_final_logits_raw"][condition_index, :, qi] = (
                        scenario_logits[:, row]
                    )
                    arrays["scenario_final_logits"][condition_index, :, qi] = (
                        trusted_logits[None, :]
                        + scenario_logits[:, row]
                        - natural_output[row][None, :]
                    )
                    # Post-L64 donor replacement should reproduce the paired donor's
                    # final hidden state up to native bfloat16 capture precision.
                    donor_row = int(donors[row])
                    arrays["residual_capture_max_abs_reconstruction_error"][
                        condition_index, qi
                    ] = float(
                        np.max(
                            np.abs(
                                scenario_logits[64, row]
                                - natural_output[donor_row]
                            )
                        )
                    )
            cohort_audit["pairs"].append(pair_audit)

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"final-position state crossover: {int(arrays['completed'].sum())}/{len(qids)}; "
            f"cohort_seconds={duration:.2f}",
            flush=True,
        )
        if not audit_path.exists():
            cohort_audit["scenario_count"] = len(SCENARIOS)
            cohort_audit["component_target_counts"] = {
                "mixers": len(mixer_targets),
                "mlps": len(mlp_targets),
            }
            audit_path.write_text(json.dumps(cohort_audit, indent=2) + "\n")

    metadata = {
        "experiment": "complete final-position residual-boundary and component crossover",
        "questions": len(qids),
        "conditions": list(CONDITIONS),
        "scenarios": list(SCENARIOS),
        "complete_model_forwards_per_canonical_cohort": 2 * len(SCENARIOS),
        "paired_subbatches_per_canonical_cohort": 2,
        "residual_boundaries": list(range(1, 65)),
        "global_component_factorial": list(GLOBAL_COMPONENT_SCENARIOS),
        "same_batch_correction": (
            "trusted natural logits + intervention same-batch logits - natural same-batch logits"
        ),
        "elapsed_seconds_after_load": time.monotonic() - started,
        "cohort_seconds": durations,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
