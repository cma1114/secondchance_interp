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
from .collect_remapped_feedback_factorial import _remap_question
from .config import ExperimentConfig
from .downstream_source_intervention import (
    BatchedSDPAQuerySourceAttentionAblator,
    BatchedSelectiveGDNSourceWritePatcher,
)
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    resolve_answer_tokens,
    tokenize_batch,
)
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import _aggregate_logits, _forward, _locate_evaluation
from .run_fixed_a_final_query_edge_ablation import _option_line_positions


SCENARIOS = (
    "natural",
    "policy_swapped",
    "matching_blocked",
    "policy_swapped_matching_blocked",
    "cyclic_control_blocked",
    "policy_swapped_cyclic_control_blocked",
    "policy_swapped_mlp49_restored",
)
MLP_LAYER_ONE_BASED = 49
MLP_LAYER_INDEX = MLP_LAYER_ONE_BASED - 1


def _hidden(output: Any) -> Any:
    return output[0] if isinstance(output, (tuple, list)) else output


class MLP49Collector:
    def __init__(self, parts: Any, positions: list[list[int]], direction: Any) -> None:
        self.positions = positions
        self.direction = direction
        self.mlp_values = None
        self.post_values = None
        self.handles = [
            parts.layers[MLP_LAYER_INDEX].mlp.register_forward_hook(self._mlp_hook),
            parts.layers[MLP_LAYER_INDEX].register_forward_hook(self._layer_hook),
        ]

    def _gather(self, hidden: Any) -> Any:
        import torch

        return torch.stack(
            [hidden[row].index_select(0, torch.as_tensor(cols, device=hidden.device))
             for row, cols in enumerate(self.positions)],
            dim=0,
        )

    def _mlp_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.mlp_values = self._gather(_hidden(output)).detach()

    def _layer_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.post_values = self._gather(_hidden(output)).detach()

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []

    def result(self) -> tuple[Any, np.ndarray]:
        import torch

        if self.mlp_values is None or self.post_values is None:
            raise RuntimeError("MLP-49 collector did not observe a complete forward")
        rms = torch.sqrt(self.post_values.float().square().mean(-1).clamp_min(1e-12))
        normalized = self.mlp_values.float() / rms[..., None]
        projection = torch.einsum(
            "brd,d->br", normalized, self.direction.to(normalized.device)
        )
        return self.mlp_values, projection.cpu().numpy()


class MLP49OutputPatcher:
    def __init__(self, parts: Any, positions: list[list[int]], values: Any) -> None:
        self.positions = positions
        self.values = values
        self.handle = parts.layers[MLP_LAYER_INDEX].mlp.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = _hidden(output)
        updated = hidden.clone()
        for row, cols in enumerate(self.positions):
            index = torch.as_tensor(cols, device=hidden.device)
            updated[row, index] = self.values[row].to(hidden.device, hidden.dtype)
        if isinstance(output, tuple):
            return (updated,) + output[1:]
        if isinstance(output, list):
            return [updated] + list(output[1:])
        return updated

    def close(self) -> None:
        self.handle.remove()


def _hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _attention_specs(
    ordinary_layers: list[int],
    sources: list[list[list[int]]],
    queries: list[list[list[int]]],
    cyclic: bool,
) -> dict[int, dict[int, dict[int, list[int]]]]:
    specs: dict[int, dict[int, dict[int, list[int]]]] = {}
    for layer in ordinary_layers:
        rows: dict[int, dict[int, list[int]]] = {}
        for row in range(len(sources)):
            row_specs: dict[int, list[int]] = {}
            for rank in range(4):
                source_rank = (rank + 1) % 4 if cyclic else rank
                for query in queries[row][rank]:
                    row_specs[int(query)] = [int(value) for value in sources[row][source_rank]]
            rows[row] = row_specs
        specs[layer] = rows
    return specs


def _initialize(path: Path, qids: list[str], ordinary_layers: list[int], gla_layers: list[int]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise RuntimeError("Existing checkpoint has different questions")
        if arrays["scenario_ids"].astype(str).tolist() != list(SCENARIOS):
            raise RuntimeError("Existing checkpoint has different scenarios")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "scenario_ids": np.asarray(SCENARIOS),
        "ordinary_layers_one_based": np.asarray([value + 1 for value in ordinary_layers], dtype=np.int16),
        "gla_layers_one_based": np.asarray([value + 1 for value in gla_layers], dtype=np.int16),
        "completed": np.zeros(n, dtype=bool),
        "rank_contents": np.full((n, 4), "", dtype="<U1"),
        "baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "scenario_logits": np.full((2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32),
        "mlp49_old_score_projection": np.full((2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32),
        "mlp49_restore_max_abs_error": np.full((2, n), np.nan, dtype=np.float32),
        "source_position_counts": np.zeros((n, 4), dtype=np.int16),
        "query_position_counts": np.zeros((n, 4), dtype=np.int16),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
    }


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(args.config)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires the canonical batch-size-4 SDPA configuration")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires empty-history action-matched prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires incorrect/lost action-matched feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires canonical raw Qwen ChatML")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    all_qids = [row["id"] for row in manifest["questions"]]
    if args.split_plan is None:
        qids = list(all_qids)
    else:
        split_ids = set(json.loads(args.split_plan.read_text())["question_ids"])
        qids = [qid for qid in all_qids if qid in split_ids]
    if args.max_cohorts is not None:
        qids = qids[: args.max_cohorts * config.batch_size]
    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    baseline = json.loads(args.baseline.read_text())["results"]
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
    ordinary_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "self_attn", None) is not None
    ]
    gla_layers = [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]
    if len(ordinary_layers) != 16 or len(gla_layers) != 48:
        raise RuntimeError("Unexpected ordinary-attention/GLA layer inventory")
    direction_payload = torch.load(args.score_directions, map_location="cpu", weights_only=False).float()
    direction = direction_payload[MLP_LAYER_INDEX, 2, 0].contiguous()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.npz"
    arrays = _initialize(result_path, qids, ordinary_layers, gla_layers)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = args.output_dir / "prompt_audit.json"
    durations: list[float] = []
    run_started = time.monotonic()

    for start in range(0, len(qids), config.batch_size):
        cohort = qids[start : start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if arrays["completed"][indices].all():
            continue
        cohort_started = time.monotonic()
        condition_batches = [
            _build_batch(config, processor, tokenizer, questions, mappings, cohort, condition)
            for condition in CONDITIONS
        ]
        audit: dict[str, Any] = {"question_ids": cohort, "conditions": {}}
        canonical_width = int(condition_batches[0]["input_ids"].shape[1])
        for pair_start in range(0, len(cohort), 2):
            pair = cohort[pair_start : pair_start + 2]
            if len(pair) != 2:
                raise RuntimeError("Canonical cohort must divide into two question pairs")
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
            rows = len(prompts)
            if rows != config.batch_size:
                raise RuntimeError("Paired policy batch must preserve four physical rows")

            sources: list[list[list[int]]] = []
            queries: list[list[list[int]]] = []
            final_semantic: list[list[int]] = []
            periods: list[int] = []
            for condition_index, condition in enumerate(CONDITIONS):
                for local, qid in enumerate(pair):
                    row = condition_index * len(pair) + local
                    prompt = prompts[row]
                    ids = [int(value) for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
                    left_pad = width - len(ids)
                    if input_ids[row, left_pad:].tolist() != ids:
                        raise RuntimeError("Paired tokenization changed a canonical prompt")
                    location = _locate_evaluation(tokenizer, prompt, condition)
                    periods.append(left_pad + int(location["period_position"]))
                    first_positions, first_audit = _option_line_positions(tokenizer, prompt, questions[qid])
                    remapped_question = _remap_question(questions[qid], mappings[qid]["new_to_original"])
                    second_positions, second_audit = _option_line_positions(tokenizer, prompt, remapped_question)
                    scores = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=float)
                    ranks = np.argsort(-scores, kind="stable")
                    rank_letters = [LETTERS[int(value)] for value in ranks]
                    row_sources: list[list[int]] = []
                    row_queries: list[list[int]] = []
                    row_final: list[int] = []
                    for rank, original in enumerate(rank_letters):
                        second_letter = mappings[qid]["original_to_new"][original]
                        source = [left_pad + value for value in first_positions[original]]
                        query = [left_pad + value for value in second_positions[second_letter]]
                        if input_ids[row, query[-1]].item() != 198:
                            raise RuntimeError("2P option line does not close with the expected newline token")
                        if len(query) < 5 or max(source) >= min(query):
                            raise RuntimeError("Invalid source/query option-line alignment")
                        row_sources.append(source)
                        row_queries.append(query)
                        row_final.append(query[-2])
                        qi = qid_index[qid]
                        arrays["source_position_counts"][qi, rank] = len(source)
                        arrays["query_position_counts"][qi, rank] = len(query)
                    sources.append(row_sources)
                    queries.append(row_queries)
                    final_semantic.append(row_final)
                    qi = qid_index[qid]
                    arrays["rank_contents"][qi] = np.asarray(rank_letters)
                    arrays["baseline_logits"][qi] = scores
                    arrays["prompt_hashes"][condition_index, qi] = _hash(prompt)
                    arrays["trusted_natural_logits"][condition_index, qi] = np.asarray(
                        trusted[condition_index][qid]["aggregated_ad_logits"], dtype=np.float32
                    )
                    if pair_start == 0 and local == 0:
                        audit["conditions"][condition] = {
                            "prompt_hash": _hash(prompt),
                            "evaluation_word_token": location["word_token"],
                            "evaluation_period_token": location["period_token"],
                            "period_position_unpadded": int(location["period_position"]),
                            "rank_letters": rank_letters,
                            "first_lines": first_audit,
                            "second_lines": second_audit,
                        }

            policy_specs = {
                row: (
                    row + len(pair) if row < len(pair) else row - len(pair),
                    [periods[row]],
                    gla_layers,
                )
                for row in range(rows)
            }
            matching_specs = _attention_specs(ordinary_layers, sources, queries, cyclic=False)
            control_specs = _attention_specs(ordinary_layers, sources, queries, cyclic=True)

            raw_logits: list[np.ndarray] = []
            projections: list[np.ndarray] = []
            natural_mlp = None
            restored_mlp = None
            for scenario in SCENARIOS:
                policy_patcher = None
                attention_patcher = None
                mlp_patcher = None
                collector = None
                try:
                    if scenario.startswith("policy_swapped"):
                        policy_patcher = BatchedSelectiveGDNSourceWritePatcher(
                            parts, policy_specs, preserve_source_output=False
                        )
                    if "matching_blocked" in scenario:
                        attention_patcher = BatchedSDPAQuerySourceAttentionAblator(parts, matching_specs)
                    elif "cyclic_control_blocked" in scenario:
                        attention_patcher = BatchedSDPAQuerySourceAttentionAblator(parts, control_specs)
                    if scenario == "policy_swapped_mlp49_restored":
                        if natural_mlp is None:
                            raise RuntimeError("Natural MLP-49 values were not collected first")
                        mlp_patcher = MLP49OutputPatcher(parts, final_semantic, natural_mlp)
                    collector = MLP49Collector(parts, final_semantic, direction)
                    output = _forward(model, parts, input_ids, attention_mask)
                    if policy_patcher is not None:
                        policy_patcher.assert_fired()
                    if attention_patcher is not None:
                        attention_patcher.assert_fired()
                    logits = _aggregate_logits(output, variant_ids)
                    mlp_values, projection = collector.result()
                    if scenario == "natural":
                        natural_mlp = mlp_values.detach().clone()
                    if scenario == "policy_swapped_mlp49_restored":
                        restored_mlp = mlp_values.detach().clone()
                    raw_logits.append(logits)
                    projections.append(projection)
                finally:
                    if collector is not None:
                        collector.close()
                    if mlp_patcher is not None:
                        mlp_patcher.close()
                    if attention_patcher is not None:
                        attention_patcher.close()
                    if policy_patcher is not None:
                        policy_patcher.close()

            raw = np.stack(raw_logits, axis=0)
            projected = np.stack(projections, axis=0)
            natural_raw = raw[0]
            if natural_mlp is None or restored_mlp is None:
                raise RuntimeError("Missing MLP-49 intervention controls")
            restore_error = (
                (restored_mlp.float() - natural_mlp.float()).abs()
                .amax(-1).amax(-1).cpu().numpy()
            )
            for condition_index in range(2):
                for local, qid in enumerate(pair):
                    qi = qid_index[qid]
                    row = condition_index * len(pair) + local
                    arrays["same_batch_natural_logits"][condition_index, qi] = natural_raw[row]
                    trusted_logits = arrays["trusted_natural_logits"][condition_index, qi]
                    arrays["scenario_logits"][condition_index, :, qi] = (
                        trusted_logits[None, :] + raw[:, row] - natural_raw[row][None, :]
                    )
                    arrays["mlp49_old_score_projection"][condition_index, :, qi] = projected[:, row]
                    arrays["mlp49_restore_max_abs_error"][condition_index, qi] = restore_error[row]

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"policy-rank factorial: {int(arrays['completed'].sum())}/{len(qids)}; cohort_seconds={duration:.2f}",
            flush=True,
        )
        if not audit_path.exists():
            audit["ordinary_layers_one_based"] = [value + 1 for value in ordinary_layers]
            audit["gla_layers_one_based"] = [value + 1 for value in gla_layers]
            audit["interventions"] = {
                "policy": "Reciprocally transplant incorrect/lost evaluation-period GLA writes at every GLA layer with preserve_source_output=False; the donor-conditioned source-token GLA output is allowed to flow onward.",
                "rank_route": "Block every 2P option-line query from its matching complete 1P option line at every ordinary-attention layer.",
                "cyclic_control": "Block the same receiver from the next-ranked 1P line instead.",
                "mlp49_restore": "Under policy swap, restore the natural recipient MLP-49 output at each final 2P semantic token.",
            }
            audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    metadata = {
        "experiment": "policy-state by retrieved-old-rank causal factorial with MLP-49 restoration",
        "questions": len(qids),
        "conditions": list(CONDITIONS),
        "scenarios": list(SCENARIOS),
        "complete_forwards_per_cohort": 2 * len(SCENARIOS),
        "batch_rows_per_forward": config.batch_size,
        "paired_subbatches_per_canonical_cohort": 2,
        "ordinary_layers_one_based": [value + 1 for value in ordinary_layers],
        "gla_layers_one_based": [value + 1 for value in gla_layers],
        "policy_transplant_preserve_source_output": False,
        "policy_transplant_scope": "complete evaluation-period GLA update, including the donor-conditioned source-token output; not an output-preserved isolation of persistent recurrent memory",
        "elapsed_seconds_after_load": time.monotonic() - run_started,
        "cohort_seconds": durations,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--score-directions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
