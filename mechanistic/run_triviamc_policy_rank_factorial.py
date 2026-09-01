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
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_evaluation_update_transplant import (
    _aggregate_logits,
    _forward,
    _locate_evaluation,
)
from .run_fixed_a_final_query_edge_ablation import _option_line_positions


SCENARIOS = (
    "natural",
    "policy_swapped",
    "matching_blocked",
    "policy_swapped_matching_blocked",
    "cyclic_control_blocked",
    "policy_swapped_cyclic_control_blocked",
)


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
                    row_specs[int(query)] = [
                        int(value) for value in sources[row][source_rank]
                    ]
            rows[row] = row_specs
        specs[layer] = rows
    return specs


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
        "rank_contents": np.full((n, 4), "", dtype="<U1"),
        "baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "fresh_baseline_logits": np.full((n, 4), np.nan, dtype=np.float32),
        "fresh_baseline_answer_original": np.full(n, "", dtype="<U1"),
        "trusted_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "same_batch_natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "scenario_logits_raw": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
        "scenario_logits": np.full(
            (2, len(SCENARIOS), n, 4), np.nan, dtype=np.float32
        ),
        "source_position_counts": np.zeros((n, 4), dtype=np.int16),
        "query_position_counts": np.zeros((n, 4), dtype=np.int16),
        "cyclic_source_position_counts": np.zeros((n, 4), dtype=np.int16),
        "prompt_hashes": np.full((2, n), "", dtype="<U64"),
        "fresh_prompt_hashes": np.full(n, "", dtype="<U64"),
    }


def _fresh_baseline(
    model: Any,
    parts: Any,
    processor: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    questions: dict[str, dict[str, Any]],
    mappings: dict[str, dict[str, Any]],
    cohort: list[str],
    variant_ids: dict[str, list[int]],
) -> tuple[np.ndarray, list[str], list[str]]:
    remapped = [
        _remap_question(questions[qid], mappings[qid]["new_to_original"])
        for qid in cohort
    ]
    prompts = [
        render_chat(
            processor,
            build_messages(
                question, "baseline", config.prompt_mode, config.feedback_variant
            ),
            config.disable_thinking,
            config.chat_serialization,
        )
        for question in remapped
    ]
    input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
    output = _forward(model, parts, input_ids, attention_mask)
    aggregated = _aggregate_logits(output, variant_ids)
    full = output.logits.detach().float()
    final = full[:, 0] if full.shape[1] == 1 else full[:, -1]
    displayed: list[str] = []
    for row, qid in enumerate(cohort):
        decoded = tokenizer.decode([int(final[row].argmax().item())]).strip()
        if decoded not in LETTERS:
            raise RuntimeError(
                f"Fresh remapped baseline top token is not A-D for {qid}: {decoded!r}"
            )
        displayed.append(mappings[qid]["new_to_original"][decoded])
    if not np.all(np.isfinite(aggregated)):
        raise RuntimeError("Non-finite fresh-baseline logits")
    return aggregated, displayed, prompts


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(args.config)
    if config.batch_size != 4 or config.attn_implementation != "sdpa":
        raise ValueError("Requires canonical batch-size-4 SDPA execution")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires canonical empty-history prompts")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token-matched incorrect/lost feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires canonical raw Qwen ChatML")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {str(row["id"]): row for row in manifest["questions"]}
    qids = [str(row["id"]) for row in manifest["questions"]]
    if args.max_cohorts is not None:
        qids = qids[: int(args.max_cohorts) * config.batch_size]
    if len(qids) % config.batch_size:
        raise RuntimeError("Questions must form complete four-question cohorts")
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    baseline = json.loads(args.baseline.read_text())["results"]
    trusted = [
        json.loads(args.trusted_game.read_text())["results"],
        json.loads(args.trusted_neutral.read_text())["results"],
    ]
    required = set(qids)
    for name, rows in (
        ("manifest", questions),
        ("remapping plan", mappings),
        ("baseline", baseline),
        ("trusted Game", trusted[0]),
        ("trusted Neutral", trusted[1]),
    ):
        if not required <= set(rows):
            raise ValueError(f"{name} is missing requested questions")

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
    if [value + 1 for value in ordinary_layers] != list(range(4, 65, 4)):
        raise RuntimeError("Unexpected ordinary-attention inventory")
    if [value + 1 for value in gla_layers] != [
        value for value in range(1, 65) if value % 4 != 0
    ]:
        raise RuntimeError("Unexpected GLA inventory")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.npz"
    arrays = _initialize(result_path, qids)
    completed_at_resume = arrays["completed"].astype(bool)
    if completed_at_resume.any():
        arrays["scenario_logits"][:, 0, completed_at_resume] = arrays[
            "trusted_natural_logits"
        ][:, completed_at_resume]
        atomic_save_npz(result_path, **arrays)
    qid_index = {qid: index for index, qid in enumerate(qids)}
    audit_path = args.output_dir / "prompt_audit.json"
    durations: list[float] = []
    started = time.monotonic()

    for cohort_start in range(0, len(qids), config.batch_size):
        cohort = qids[cohort_start : cohort_start + config.batch_size]
        indices = [qid_index[qid] for qid in cohort]
        if arrays["completed"][indices].all():
            continue
        cohort_started = time.monotonic()
        fresh_logits, fresh_answers, fresh_prompts = _fresh_baseline(
            model, parts, processor, tokenizer, config, questions, mappings,
            cohort, variant_ids,
        )
        for local, qid in enumerate(cohort):
            qi = qid_index[qid]
            arrays["fresh_baseline_logits"][qi] = fresh_logits[local]
            arrays["fresh_baseline_answer_original"][qi] = fresh_answers[local]
            arrays["fresh_prompt_hashes"][qi] = _hash(fresh_prompts[local])
            old = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=float)
            arrays["baseline_logits"][qi] = old
            arrays["rank_contents"][qi] = np.asarray(
                [LETTERS[int(value)] for value in np.argsort(-old, kind="stable")]
            )

        condition_batches = [
            _build_batch(
                config, processor, tokenizer, questions, mappings, cohort, condition
            )
            for condition in CONDITIONS
        ]
        canonical_width = int(condition_batches[0]["input_ids"].shape[1])
        cohort_audit: dict[str, Any] = {
            "question_ids": cohort,
            "fresh_baseline": {
                "prompt_hash": _hash(fresh_prompts[0]),
                "answer_original_content": fresh_answers[0],
            },
            "pairs": [],
        }

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
                attention_mask = torch.nn.functional.pad(
                    attention_mask, (pad, 0), value=0
                )
            width = int(input_ids.shape[1])
            sources: list[list[list[int]]] = []
            queries: list[list[list[int]]] = []
            periods: list[int] = []
            pair_audit: dict[str, Any] = {"question_ids": pair, "rows": []}

            for condition_index, condition in enumerate(CONDITIONS):
                for local, qid in enumerate(pair):
                    row = condition_index * 2 + local
                    qi = qid_index[qid]
                    prompt = prompts[row]
                    ids = [
                        int(value)
                        for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    ]
                    left_pad = width - len(ids)
                    if input_ids[row, left_pad:].tolist() != ids:
                        raise RuntimeError("Paired tokenization changed a prompt")
                    location = _locate_evaluation(tokenizer, prompt, condition)
                    periods.append(left_pad + int(location["period_position"]))
                    first_positions, first_audit = _option_line_positions(
                        tokenizer, prompt, questions[qid]
                    )
                    second_question = _remap_question(
                        questions[qid], mappings[qid]["new_to_original"]
                    )
                    second_positions, second_audit = _option_line_positions(
                        tokenizer, prompt, second_question
                    )
                    ranks = arrays["rank_contents"][qi].astype(str).tolist()
                    row_sources: list[list[int]] = []
                    row_queries: list[list[int]] = []
                    for rank, original in enumerate(ranks):
                        second_letter = mappings[qid]["original_to_new"][original]
                        source = [left_pad + value for value in first_positions[original]]
                        query = [left_pad + value for value in second_positions[second_letter]]
                        if not source or not query or max(source) >= min(query):
                            raise RuntimeError("Invalid source/query option-line alignment")
                        row_sources.append(source)
                        row_queries.append(query)
                        arrays["source_position_counts"][qi, rank] = len(source)
                        arrays["query_position_counts"][qi, rank] = len(query)
                        arrays["cyclic_source_position_counts"][qi, rank] = len(
                            first_positions[ranks[(rank + 1) % 4]]
                        )
                    sources.append(row_sources)
                    queries.append(row_queries)
                    digest = _hash(prompt)
                    if digest != trusted[condition_index][qid]["prompt_hash"]:
                        raise RuntimeError("Prompt hash differs from trusted Step-1 natural")
                    arrays["prompt_hashes"][condition_index, qi] = digest
                    arrays["trusted_natural_logits"][condition_index, qi] = np.asarray(
                        trusted[condition_index][qid]["aggregated_ad_logits"],
                        dtype=np.float32,
                    )
                    pair_audit["rows"].append({
                        "row": row,
                        "condition": condition,
                        "question_id": qid,
                        "prompt_hash": digest,
                        "evaluation_word_token": location["word_token"],
                        "evaluation_period_token": location["period_token"],
                        "evaluation_period_position": periods[-1],
                        "rank_contents": ranks,
                        "first_option_lines": first_audit,
                        "second_option_lines": second_audit,
                    })

            policy_specs = {
                row: (
                    row + 2 if row < 2 else row - 2,
                    [periods[row]],
                    gla_layers,
                )
                for row in range(4)
            }
            matching_specs = _attention_specs(
                ordinary_layers, sources, queries, cyclic=False
            )
            control_specs = _attention_specs(
                ordinary_layers, sources, queries, cyclic=True
            )

            raw_outputs: list[np.ndarray] = []
            for scenario in SCENARIOS:
                policy = None
                attention = None
                try:
                    if scenario.startswith("policy_swapped"):
                        policy = BatchedSelectiveGDNSourceWritePatcher(
                            parts, policy_specs, preserve_source_output=False
                        )
                    if "matching_blocked" in scenario:
                        attention = BatchedSDPAQuerySourceAttentionAblator(
                            parts, matching_specs
                        )
                    elif "cyclic_control_blocked" in scenario:
                        attention = BatchedSDPAQuerySourceAttentionAblator(
                            parts, control_specs
                        )
                    logits = _aggregate_logits(
                        _forward(model, parts, input_ids, attention_mask), variant_ids
                    )
                    if policy is not None:
                        policy.assert_fired()
                    if attention is not None:
                        attention.assert_fired()
                    if not np.all(np.isfinite(logits)):
                        raise RuntimeError(f"Non-finite logits in {scenario}")
                    raw_outputs.append(logits)
                finally:
                    if attention is not None:
                        attention.close()
                    if policy is not None:
                        policy.close()

            raw = np.stack(raw_outputs, axis=0)
            natural = raw[0]
            for condition_index in range(2):
                for local, qid in enumerate(pair):
                    row = condition_index * 2 + local
                    qi = qid_index[qid]
                    trusted_logits = arrays["trusted_natural_logits"][condition_index, qi]
                    arrays["same_batch_natural_logits"][condition_index, qi] = natural[row]
                    arrays["scenario_logits_raw"][condition_index, :, qi] = raw[:, row]
                    arrays["scenario_logits"][condition_index, :, qi] = (
                        trusted_logits[None, :] + raw[:, row] - natural[row][None, :]
                    )
                    arrays["scenario_logits"][condition_index, 0, qi] = trusted_logits
            cohort_audit["pairs"].append(pair_audit)

        arrays["completed"][indices] = True
        atomic_save_npz(result_path, **arrays)
        duration = time.monotonic() - cohort_started
        durations.append(duration)
        print(
            f"TriviaMC policy-rank factorial: {int(arrays['completed'].sum())}/"
            f"{len(qids)}; cohort_seconds={duration:.2f}",
            flush=True,
        )
        if not audit_path.exists():
            cohort_audit.update({
                "scenario_ids": list(SCENARIOS),
                "ordinary_layers_one_based": [value + 1 for value in ordinary_layers],
                "gla_layers_one_based": [value + 1 for value in gla_layers],
                "policy_transplant": (
                    "Reciprocal transplant of the evaluation-closing period GLA "
                    "write at every GLA layer, preserve_source_output=False."
                ),
                "matching_blockade": (
                    "All complete matching 1P-option-line to 2P-option-line reads "
                    "blocked at every ordinary-attention layer."
                ),
                "cyclic_control": "W1<-W2, W2<-W3, W3<-W4, W4<-W1.",
            })
            audit_path.write_text(json.dumps(cohort_audit, indent=2) + "\n")

    metadata = {
        "experiment": "TriviaMC compact policy by retrieved-rank causal factorial",
        "questions": len(qids),
        "conditions": list(CONDITIONS),
        "display_conditions": ["Game", "Neutral"],
        "scenarios": list(SCENARIOS),
        "complete_model_forwards_per_canonical_cohort": 13,
        "complete_model_work": (
            "One four-row standalone remapped baseline plus six paired-task "
            "factorial scenarios for each of two two-question paired subbatches."
        ),
        "ordinary_layers_one_based": [value + 1 for value in ordinary_layers],
        "gla_layers_one_based": [value + 1 for value in gla_layers],
        "policy_transplant_preserve_source_output": False,
        "policy_transplant_scope": (
            "Complete evaluation-closing-period GLA update, including the "
            "donor-conditioned source-token output; not an output-preserved "
            "isolation of persistent recurrent memory."
        ),
        "rank_route_scope": (
            "All four complete matching option-line relations versus all four "
            "cyclic-wrong relations across all 16 ordinary-attention layers."
        ),
        "conflict_definition": (
            "First-presentation aggregated-logit W1 differs from the unrestricted "
            "A-D answer to the standalone remapped second-presentation baseline."
        ),
        "same_batch_correction": (
            "trusted Step-1 natural + intervention same-batch - natural same-batch; "
            "natural stored exactly as trusted."
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
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
