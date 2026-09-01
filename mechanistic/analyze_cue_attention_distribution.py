from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .config import ExperimentConfig
from .prompts import (
    ANSWER_ONLY_INSTRUCTION,
    CAPABILITY_SETUP,
    CHOICE_CUE,
    FACTORIAL_FEEDBACK,
    present_question,
)
from .run_action_period_mediation import CONDITIONS, _build_batch
from .run_second_presentation_attention_distribution import ORDINARY_LAYERS
from .analyze_second_presentation_policy_transport import _receiver_roles


CONDITION_LABELS = ("Game", "Neutral")
BASE_SOURCE_NAMES = (
    "system_and_header",
    "first_task_instruction",
    "first_question_stem",
    "first_R1_line",
    "first_R2_line",
    "first_R3_line",
    "first_R4_line",
    "first_answer_boundary",
)
FEEDBACK_SOURCE_NAMES = tuple(f"feedback_token_{index}" for index in range(10))
TAIL_SOURCE_NAMES = (
    "second_answer_instruction",
    "second_question_stem",
    "second_R1_line",
    "second_R2_line",
    "second_R3_line",
    "second_R4_line",
    "second_choice_cue_and_query",
    "final_assistant_prefix",
    "other_structure",
)
SOURCE_NAMES = BASE_SOURCE_NAMES + FEEDBACK_SOURCE_NAMES + TAIL_SOURCE_NAMES


def _find_after(text: str, needle: str, start: int) -> tuple[int, int]:
    position = text.find(needle, start)
    if position < 0:
        raise RuntimeError(f"Could not locate {needle!r} after character {start}")
    return position, position + len(needle)


def _overlaps(offset: tuple[int, int], interval: tuple[int, int]) -> bool:
    left, right = offset
    start, stop = interval
    return right > left and left < stop and right > start


def _find_unique_subsequence(row: list[int], needle: list[int]) -> list[int]:
    hits = [
        start
        for start in range(len(row) - len(needle) + 1)
        if row[start : start + len(needle)] == needle
    ]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one token-subsequence match, found {hits}")
    return list(range(hits[0], hits[0] + len(needle)))


def _option_line_positions(
    tokenizer: Any, prompt: str, question: dict[str, Any], occurrence_start: int
) -> dict[str, list[int]]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    positions: dict[str, list[int]] = {}
    cursor = occurrence_start
    for letter in LETTERS:
        text = f"  {letter}: {question['options'][letter]}"
        interval = _find_after(prompt, text, cursor)
        rows = [index for index, offset in enumerate(offsets) if _overlaps(offset, interval)]
        if not rows:
            raise RuntimeError(f"Option {letter} did not cover any tokens")
        positions[letter] = rows
        cursor = interval[1]
    return positions


def _cue_source_partition(
    tokenizer: Any,
    prompt: str,
    messages: list[dict[str, str]],
    first_question: dict[str, Any],
    second_question: dict[str, Any],
    condition: str,
    rank_letters: list[str],
    original_to_new: dict[str, str],
) -> tuple[list[list[int]], dict[str, Any]]:
    """Partition every unpadded prompt token for a cue-position source map."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]

    first_text = present_question(first_question)
    second_text = present_question(second_question)
    first_question_range = _find_after(prompt, first_text, 0)
    second_question_range = _find_after(prompt, second_text, first_question_range[1])
    feedback_text = FACTORIAL_FEEDBACK[condition]
    feedback_range = _find_after(prompt, feedback_text, first_question_range[1])
    capability_range = _find_after(prompt, CAPABILITY_SETUP, 0)
    second_instruction_range = _find_after(
        prompt, ANSWER_ONLY_INSTRUCTION, feedback_range[1]
    )
    second_choice_range = _find_after(prompt, CHOICE_CUE, second_question_range[1])
    if not (
        capability_range[0]
        < first_question_range[0]
        < feedback_range[0]
        < second_instruction_range[0]
        < second_question_range[0]
        < second_choice_range[0]
    ):
        raise RuntimeError("Canonical prompt components are out of order")

    first_lines = _option_line_positions(
        tokenizer, prompt, first_question, first_question_range[0]
    )
    second_lines = _option_line_positions(
        tokenizer, prompt, second_question, second_question_range[0]
    )
    feedback_ids = [
        int(value)
        for value in tokenizer(feedback_text, add_special_tokens=False)["input_ids"]
    ]
    feedback_positions = _find_unique_subsequence(ids, feedback_ids)
    if len(feedback_positions) != len(FEEDBACK_SOURCE_NAMES):
        raise RuntimeError(
            f"Expected {len(FEEDBACK_SOURCE_NAMES)} aligned feedback tokens, "
            f"found {len(feedback_positions)}"
        )

    intervals = {
        "system_and_header": (0, capability_range[0]),
        "first_task_instruction": (capability_range[0], first_question_range[0]),
        "first_question_stem": first_question_range,
        "first_answer_boundary": (first_question_range[1], feedback_range[0]),
        "second_answer_instruction": (feedback_range[1], second_question_range[0]),
        "second_question_stem": second_question_range,
        "second_choice_cue_and_query": (second_question_range[1], second_choice_range[1]),
        "final_assistant_prefix": (second_choice_range[1], len(prompt)),
    }
    labels = np.full(len(ids), SOURCE_NAMES.index("other_structure"), dtype=np.int16)
    for name, interval in intervals.items():
        source_index = SOURCE_NAMES.index(name)
        for token_index, offset in enumerate(offsets):
            if _overlaps(offset, interval):
                labels[token_index] = source_index

    for rank_index, first_letter in enumerate(rank_letters):
        second_letter = original_to_new[first_letter]
        for token_index in first_lines[first_letter]:
            labels[token_index] = SOURCE_NAMES.index(f"first_R{rank_index + 1}_line")
        for token_index in second_lines[second_letter]:
            labels[token_index] = SOURCE_NAMES.index(f"second_R{rank_index + 1}_line")
    for feedback_index, token_index in enumerate(feedback_positions):
        labels[token_index] = SOURCE_NAMES.index(f"feedback_token_{feedback_index}")

    positions = [
        np.flatnonzero(labels == source_index).astype(int).tolist()
        for source_index in range(len(SOURCE_NAMES))
    ]
    covered = sorted(position for rows in positions for position in rows)
    if covered != list(range(len(ids))):
        raise RuntimeError("Cue source partition is not exhaustive and disjoint")
    required = [name for name in SOURCE_NAMES if name != "other_structure"]
    empty = [name for name in required if not positions[SOURCE_NAMES.index(name)]]
    if empty:
        raise RuntimeError(f"Required source regions are empty: {empty}")

    return positions, {
        "prompt_length": len(ids),
        "feedback_tokens": [
            tokenizer.decode([ids[position]]).replace("\n", "\\n")
            for position in feedback_positions
        ],
        "source_tokens": {
            name: tokenizer.convert_ids_to_tokens([ids[position] for position in positions[index]])
            for index, name in enumerate(SOURCE_NAMES)
        },
        "messages": [message["role"] for message in messages],
    }


def _bootstrap_intervals(
    values: np.ndarray, seed: int, draws: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Question-bootstrap every condition/layer/source cell together.

    values has shape condition x layer x question x source.
    """
    values = np.asarray(values, dtype=np.float32)
    rng = np.random.default_rng(seed)
    samples = np.empty(
        (draws, values.shape[0], values.shape[1], values.shape[3]), dtype=np.float32
    )
    for start in range(0, draws, 100):
        stop = min(start + 100, draws)
        rows = rng.integers(
            0, values.shape[2], size=(stop - start, values.shape[2])
        )
        # condition x layer x draw x question x source -> draw x condition x layer x source
        samples[start:stop] = values[:, :, rows, :].mean(axis=3).transpose(2, 0, 1, 3)
    low, high = np.quantile(samples, (0.025, 0.975), axis=0)
    return values.mean(axis=2), low, high


def _display_labels(feedback_labels: list[str]) -> list[str]:
    fixed = {
        "system_and_header": "System prompt + ChatML header",
        "first_task_instruction": "1P task instruction",
        "first_question_stem": "1P question stem/separators",
        "first_R1_line": "1P R1 option line",
        "first_R2_line": "1P R2 option line",
        "first_R3_line": "1P R3 option line",
        "first_R4_line": "1P R4 option line",
        "first_answer_boundary": "1P answer cue + decision boundary",
        "second_answer_instruction": "2P answer-only instruction",
        "second_question_stem": "2P question stem/separators",
        "second_R1_line": "2P R1 option line",
        "second_R2_line": "2P R2 option line",
        "second_R3_line": "2P R3 option line",
        "second_R4_line": "2P R4 option line",
        "second_choice_cue_and_query": "2P choice cue + cue query itself",
        "final_assistant_prefix": "After cue query (causally masked)",
        "other_structure": "Other structure",
    }
    labels = []
    for name in SOURCE_NAMES:
        if name.startswith("feedback_token_"):
            labels.append(feedback_labels[int(name.rsplit("_", 1)[1])])
        else:
            labels.append(fixed[name])
    return labels


def analyze(args: argparse.Namespace) -> None:
    import matplotlib
    import torch
    from transformers import AutoTokenizer

    started = time.perf_counter()
    config = ExperimentConfig.load(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
    )
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {str(row["id"]): row for row in manifest["questions"]}
    qids = [str(row["id"]) for row in manifest["questions"]]
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    if [int(discovery.sum()), int((~discovery).sum())] != [251, 249]:
        raise RuntimeError("Frozen discovery/confirmation split changed")

    shard_paths = sorted((args.workspace / "shards").glob("cohort_*.pt"))
    if len(shard_paths) != 125:
        raise RuntimeError(f"Expected 125 workspace shards, found {len(shard_paths)}")
    completed = np.load(args.workspace / "completed.npy").astype(bool)
    if not completed.all():
        raise RuntimeError("Residual workspace is incomplete")
    selected_shards = shard_paths[: args.max_shards] if args.max_shards else shard_paths
    selected_questions = len(selected_shards) * 4
    attention = np.full(
        (2, len(ORDINARY_LAYERS), selected_questions, len(SOURCE_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    partition_error = np.full(
        (2, len(ORDINARY_LAYERS), selected_questions), np.nan, dtype=np.float32
    )
    feedback_token_pairs: list[str] | None = None
    prompt_tokens_verified = 0

    for shard_index, shard_path in enumerate(selected_shards):
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_qids = [str(value) for value in shard["question_ids"]]
        expected_qids = qids[4 * shard_index : 4 * shard_index + len(shard_qids)]
        if shard_qids != expected_qids:
            raise RuntimeError("Workspace question order changed")
        rank_letters = [[str(value) for value in row] for row in shard["rank_letters"]]
        for condition_index, condition in enumerate(CONDITIONS):
            payload = shard["payloads"][condition]
            batch = _build_batch(
                config,
                tokenizer,
                tokenizer,
                questions,
                mappings,
                shard_qids,
                condition,
            )
            if not torch.equal(batch["input_ids"].cpu(), payload["input_ids"].cpu()):
                raise RuntimeError("Reconstructed prompt tokens differ from cached workspace")
            current_layers = tuple(
                int(value) + 1 for value in payload["ordinary_layer_indices"].tolist()
            )
            if current_layers != ORDINARY_LAYERS:
                raise RuntimeError(f"Ordinary-attention layers changed: {current_layers}")
            roles = [
                _receiver_roles(
                    payload,
                    row,
                    rank_letters[row],
                    mappings[shard_qids[row]]["original_to_new"],
                    tokenizer,
                )
                for row in range(len(shard_qids))
            ]
            local_feedback_tokens: list[list[str]] = []
            partitions: list[list[list[int]]] = []
            for row, qid in enumerate(shard_qids):
                second_question = {
                    **questions[qid],
                    "options": {
                        new: questions[qid]["options"][old]
                        for new, old in mappings[qid]["new_to_original"].items()
                    },
                }
                rows, audit = _cue_source_partition(
                    tokenizer,
                    batch["prompts"][row],
                    batch["messages"][row],
                    questions[qid],
                    second_question,
                    condition,
                    rank_letters[row],
                    mappings[qid]["original_to_new"],
                )
                left_pad = int(payload["attention_mask"][row].numel() - audit["prompt_length"])
                partitions.append([[left_pad + value for value in source] for source in rows])
                local_feedback_tokens.append(audit["feedback_tokens"])
            if any(tokens != local_feedback_tokens[0] for tokens in local_feedback_tokens):
                raise RuntimeError("Feedback tokenization changed across questions")
            if condition_index == 0:
                game_feedback = local_feedback_tokens[0]
            else:
                neutral_feedback = local_feedback_tokens[0]
                pairs = [
                    left if left == right else f"{left} | {right}"
                    for left, right in zip(game_feedback, neutral_feedback)
                ]
                if feedback_token_pairs is None:
                    feedback_token_pairs = pairs
                elif feedback_token_pairs != pairs:
                    raise RuntimeError("Feedback-token display labels changed")

            weights = payload["attention_weights"].float()
            query_mask = payload["attention_query_mask"].bool()
            for row in range(len(shard_qids)):
                query_columns = roles[row][args.query_role]
                if len(query_columns) != 1:
                    raise RuntimeError(
                        f"Query role {args.query_role!r} is not a unique cached receiver"
                    )
                query_column = int(query_columns[0])
                if not bool(query_mask[row, :, query_column].all()):
                    raise RuntimeError(
                        f"Query role {args.query_role!r} is masked in an "
                        "ordinary-attention layer"
                    )
                query_weights = weights[row, :, :, query_column, :].mean(dim=1)
                target = 4 * shard_index + row
                for source_index, source_positions in enumerate(partitions[row]):
                    positions = torch.as_tensor(source_positions, dtype=torch.long)
                    attention[condition_index, :, target, source_index] = (
                        query_weights.index_select(1, positions).sum(dim=1).numpy()
                    )
                partition_error[condition_index, :, target] = np.abs(
                    attention[condition_index, :, target].sum(axis=-1) - 1.0
                )
            prompt_tokens_verified += len(shard_qids)
        del shard
        if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(selected_shards):
            print(
                f"{args.output_prefix} attention source map: "
                f"{shard_index + 1}/{len(selected_shards)} shards",
                flush=True,
            )

    if not np.isfinite(attention).all() or not np.isfinite(partition_error).all():
        raise RuntimeError("Cue attention source arrays contain non-finite values")
    if float(partition_error.max()) > 0.01:
        raise RuntimeError(
            f"Cue source partition does not sum to one: {float(partition_error.max())}"
        )
    if feedback_token_pairs is None:
        raise RuntimeError("Feedback-token labels were not resolved")
    display_labels = _display_labels(feedback_token_pairs)
    if args.query_role == "final_decision":
        display_labels[SOURCE_NAMES.index("final_assistant_prefix")] = (
            "Final assistant prefix + final query itself"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_shards:
        result = {
            "benchmark_only": True,
            "shards": len(selected_shards),
            "questions": selected_questions,
            "layers": list(ORDINARY_LAYERS),
            "sources": list(SOURCE_NAMES),
            "max_partition_error": float(partition_error.max()),
            "prompt_rows_verified": prompt_tokens_verified,
            "elapsed_seconds": time.perf_counter() - started,
            "complete_model_forwards": 0,
        }
        (args.output_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    confirmation = ~discovery
    discovery_mean = attention[:, :, discovery].mean(axis=2)
    confirmation_mean = attention[:, :, confirmation].mean(axis=2)
    confirmation_mean_boot, confirmation_low, confirmation_high = _bootstrap_intervals(
        attention[:, :, confirmation], args.seed, args.bootstrap_draws
    )
    if not np.allclose(confirmation_mean, confirmation_mean_boot, atol=1e-7):
        raise RuntimeError("Bootstrap point means differ from direct confirmation means")
    cell_rows: list[list[Any]] = []
    intervals: dict[str, Any] = {}
    for condition_index, condition_label in enumerate(CONDITION_LABELS):
        condition_rows: dict[str, Any] = {}
        for layer_index, layer in enumerate(ORDINARY_LAYERS):
            layer_rows: dict[str, Any] = {}
            for source_index, source_name in enumerate(SOURCE_NAMES):
                record = {
                    "mean": float(confirmation_mean[condition_index, layer_index, source_index]),
                    "ci_low": float(confirmation_low[condition_index, layer_index, source_index]),
                    "ci_high": float(confirmation_high[condition_index, layer_index, source_index]),
                }
                layer_rows[source_name] = record
                cell_rows.append(
                    [
                        condition_label,
                        layer,
                        source_name,
                        display_labels[source_index],
                        int(confirmation.sum()),
                        record["mean"],
                        record["ci_low"],
                        record["ci_high"],
                    ]
                )
            condition_rows[str(layer)] = layer_rows
        intervals[condition_label] = condition_rows
    with (args.output_dir / "attention_distribution.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "ordinary_attention_layer",
                "source_region",
                "display_label",
                "n_confirmation_questions",
                "mean_attention_mass",
                "ci_low",
                "ci_high",
            ]
        )
        writer.writerows(cell_rows)

    family_indices = {
        "all_1P_options": [SOURCE_NAMES.index(f"first_R{rank}_line") for rank in range(1, 5)],
        "first_answer_boundary": [SOURCE_NAMES.index("first_answer_boundary")],
        "all_feedback_tokens": [SOURCE_NAMES.index(name) for name in FEEDBACK_SOURCE_NAMES],
        "all_2P_options": [SOURCE_NAMES.index(f"second_R{rank}_line") for rank in range(1, 5)],
        "second_choice_cue_and_query": [SOURCE_NAMES.index("second_choice_cue_and_query")],
    }
    family_means = {
        condition: {
            family: confirmation_mean[condition_index, :, indices].sum(axis=-1).tolist()
            for family, indices in family_indices.items()
        }
        for condition_index, condition in enumerate(CONDITION_LABELS)
    }
    flat_discovery = discovery_mean.reshape(2, -1)
    flat_confirmation = confirmation_mean.reshape(2, -1)
    split_cosines = []
    for condition_index in range(2):
        left = flat_discovery[condition_index]
        right = flat_confirmation[condition_index]
        split_cosines.append(float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right))))
    summary = {
        "question": f"Where does the exact {args.query_label} attend across its complete causal prefix?",
        "measurement": {
            "query": args.query_label,
            "layers": "Every ordinary-attention layer, 4 through 64 in increments of 4; GLA layers have no ordinary-attention distribution.",
            "heads": "Mean attention mass across all ordinary-attention heads.",
            "sources": "Every non-padding prompt token assigned exactly once; feedback resolved token by token and option lines aligned by 1P rank.",
            "conditions": ["Game (incorrect + again)", "Neutral (lost + again)"],
        },
        "validation": {
            "questions": 500,
            "discovery": int(discovery.sum()),
            "confirmation": int(confirmation.sum()),
            "prompt_rows_verified": prompt_tokens_verified,
            "max_partition_error": float(partition_error.max()),
            "maximum_discovery_confirmation_cell_difference": float(
                np.max(np.abs(discovery_mean - confirmation_mean))
            ),
            "discovery_confirmation_map_cosine": {
                condition: split_cosines[index]
                for index, condition in enumerate(CONDITION_LABELS)
            },
            "all_values_finite": True,
            "complete_model_forwards": 0,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "source_names": list(SOURCE_NAMES),
        "display_labels": display_labels,
        "ordinary_layers": list(ORDINARY_LAYERS),
        "confirmation_family_means": family_means,
        "confirmation_intervals": intervals,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "attention_distribution.npz",
        question_ids=np.asarray(qids),
        discovery=discovery,
        ordinary_layers=np.asarray(ORDINARY_LAYERS, dtype=np.int16),
        source_names=np.asarray(SOURCE_NAMES),
        display_labels=np.asarray(display_labels),
        attention_mass=attention,
        partition_error=partition_error,
    )

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = float(confirmation_mean.max())
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 13.5), sharex=True, constrained_layout=True)
    image = None
    for condition_index, condition in enumerate(CONDITION_LABELS):
        axis = axes[condition_index]
        image = axis.imshow(
            100.0 * confirmation_mean[condition_index].T,
            origin="upper",
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=0.0,
            vmax=100.0 * vmax,
        )
        axis.set_title(f"{condition}: absolute {args.query_label} attention", fontsize=14)
        axis.set_yticks(np.arange(len(display_labels)))
        axis.set_yticklabels(display_labels, fontsize=9)
        axis.set_ylabel("Source region in the causal prefix")
        for boundary in (7.5, 17.5):
            axis.axhline(boundary, color="white", linewidth=0.8, alpha=0.7)
    axes[-1].set_xticks(np.arange(len(ORDINARY_LAYERS)))
    axes[-1].set_xticklabels(ORDINARY_LAYERS)
    axes[-1].set_xlabel("Ordinary-attention layer")
    if image is None:
        raise RuntimeError("No attention heatmap was created")
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.82, pad=0.015)
    colorbar.set_label("Mean attention mass (% of all attention; held-out 249 questions)")
    fig.suptitle(
        f"Where the {args.query_label} attends\n"
        "Every source token is assigned once; Game and Neutral share one color scale",
        fontsize=17,
        fontweight="bold",
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    findings_path = args.output_dir / "HAND_CURATED_FINDINGS.md"
    curated_findings = findings_path.read_text().strip() if findings_path.exists() else ""
    if curated_findings and not curated_findings.startswith("## Findings"):
        raise RuntimeError("HAND_CURATED_FINDINGS.md must begin with '## Findings'")
    report = f"""# Exhaustive source map for the {args.query_label}

## Method

The query is the {args.query_label}.
Every non-padding prompt token is assigned to exactly one source row. Feedback
is resolved token by token; both presentations' option lines are aligned by the
candidate's first-presentation rank. Attention is averaged over heads and
reported separately for Game and Neutral on the frozen 249-question
confirmation split. All 16 applicable ordinary-attention layers, L4--L64, are
included. The remaining 48 layers use GLA, so an ordinary-attention source
distribution is undefined there.

This is cached activation analysis: **zero new model forward passes**.

## Validation

- Prompt tokens exactly matched the cached workspace for all 500 questions in both tasks.
- Maximum exhaustive-partition sum error: `{summary['validation']['max_partition_error']:.6f}`.
- Discovery/confirmation map cosine: Game `{split_cosines[0]:.6f}`, Neutral `{split_cosines[1]:.6f}`.
- Every cell's held-out mean and 95% question-bootstrap interval is in `attention_distribution.csv`.

{curated_findings}

## Artifacts

- Canonical figure: `{args.figure}`
- Compact summary: `{args.output_dir / 'summary.json'}`
- Per-question arrays: `{args.output_dir / 'attention_distribution.npz'}`
- Complete confidence-interval table: `{args.output_dir / 'attention_distribution.csv'}`
- Interpretive prose source: `{findings_path}` (explicitly hand-curated from the generated summary/table; preserved across regeneration).
"""
    (args.output_dir / "REPORT.md").write_text(report)
    print(json.dumps(summary["validation"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--query-role", default="choice_cue_space")
    parser.add_argument(
        "--query-label",
        default="post-list answer-cue space",
    )
    parser.add_argument("--output-prefix", default="cue")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=48333967)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
