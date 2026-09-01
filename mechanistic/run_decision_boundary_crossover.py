from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import prompt_hash
from .run_first_decision_cross_order_patching import (
    _decision_position,
    _hidden,
    _replace_hidden,
)
from .run_fixed_a_full_cache_factorial import (
    _aggregate_logits,
    _cache_inventory,
    _cached_forward,
    _swap_cache_families,
)
from .run_semantic_binding_module_factorial import _forward, _messages, _remap_question


LETTERS = "ABCD"
CELLS = ("game_x", "neutral_x", "game_y", "neutral_y")
CONDITIONS = ("incorrect_again", "lost_again", "incorrect_again", "lost_again")
DONOR_ROWS = np.asarray([2, 3, 0, 1], dtype=np.int64)
EXPECTED_INVENTORY = {"attention_kv": 16, "gla_conv": 48, "gla_recurrent": 48}
N_LAYERS = 64


def _initialize(path: Path, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    qids = [row["question_id"] for row in rows]
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses another question order")
        return arrays
    n = len(rows)
    logits_shape = (4, n, 4)
    return {
        "question_ids": np.asarray(qids),
        "completed": np.zeros(n, dtype=bool),
        "token_aligned": np.zeros(n, dtype=bool),
        "exact_eligible": np.zeros(n, dtype=bool),
        "prefix_logits": np.full(logits_shape, np.nan, dtype=np.float32),
        "natural_logits": np.full(logits_shape, np.nan, dtype=np.float32),
        "identity_logits": np.full(logits_shape, np.nan, dtype=np.float32),
        "cross_logits": np.full(logits_shape, np.nan, dtype=np.float32),
        "full_donor_logits": np.full(logits_shape, np.nan, dtype=np.float32),
        "identity_boundary_logits": np.full(logits_shape, np.nan, dtype=np.float32),
        "cross_boundary_logits": np.full(logits_shape, np.nan, dtype=np.float32),
        "full_donor_boundary_logits": np.full(logits_shape, np.nan, dtype=np.float32),
        "exact_x_first_letter": np.full(n, "", dtype="<U1"),
        "exact_y_first_letter": np.full(n, "", dtype="<U1"),
        "exact_x_original_content": np.full(n, "", dtype="<U1"),
        "exact_y_original_content": np.full(n, "", dtype="<U1"),
        "exact_x_second_letter": np.full(n, "", dtype="<U1"),
        "exact_y_second_letter": np.full(n, "", dtype="<U1"),
        "identity_vs_natural_max_error": np.full(n, np.nan, dtype=np.float32),
        "identity_vs_natural_choice_changes": np.full(n, -1, dtype=np.int16),
        "full_donor_max_error": np.full(n, np.nan, dtype=np.float32),
        "full_donor_boundary_max_error": np.full(n, np.nan, dtype=np.float32),
        "identity_trajectory_dose": np.full((n, N_LAYERS), np.nan, dtype=np.float32),
        "cross_trajectory_dose": np.full((n, N_LAYERS), np.nan, dtype=np.float32),
        "model_calls": np.zeros(n, dtype=np.int16),
        "duration_seconds": np.full(n, np.nan, dtype=np.float32),
    }


class BoundaryTrajectoryCollector:
    """Capture the layer-input trajectory at the final first-decision token."""

    def __init__(self, parts: Any) -> None:
        self.values: dict[int, Any] = {}
        self.final_output: Any | None = None
        self.handles = [
            layer.register_forward_pre_hook(self._hook(index))
            for index, layer in enumerate(parts.layers)
        ]
        self.handles.append(parts.layers[-1].register_forward_hook(self._final_hook))

    def _hook(self, index: int):
        def capture(_module: Any, inputs: Any) -> None:
            hidden = inputs[0]
            if hidden.ndim != 3:
                raise RuntimeError(f"Layer {index}: unexpected hidden shape {hidden.shape}")
            self.values[index] = hidden[:, -1].detach().float().cpu()

        return capture

    def _final_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = _hidden(output)
        if hidden.ndim != 3:
            raise RuntimeError(f"Final layer: unexpected hidden shape {hidden.shape}")
        self.final_output = hidden[:, -1].detach().float().cpu()

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []

    def __enter__(self) -> "BoundaryTrajectoryCollector":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class BoundaryTrajectoryPatcher:
    """Replay a chosen layer-input trajectory for a one-token boundary step."""

    def __init__(
        self,
        parts: Any,
        trajectory: dict[int, Any],
        final_output: Any,
        donor_rows: np.ndarray,
    ):
        import torch

        if set(trajectory) != set(range(N_LAYERS)):
            raise ValueError("Incomplete boundary trajectory")
        self.trajectory = trajectory
        self.final_output = final_output
        self.donor_rows = torch.as_tensor(donor_rows, dtype=torch.long)
        self.dose = torch.full((N_LAYERS, len(donor_rows)), float("nan"))
        self.handles = [
            layer.register_forward_pre_hook(self._hook(index))
            for index, layer in enumerate(parts.layers)
        ]
        self.handles.append(parts.layers[-1].register_forward_hook(self._final_hook))

    def _hook(self, index: int):
        def patch(_module: Any, inputs: Any) -> Any:
            hidden = inputs[0]
            if hidden.ndim != 3 or hidden.shape[1] != 1:
                raise RuntimeError(
                    f"Layer {index}: boundary replay requires one token, got {hidden.shape}"
                )
            source = self.trajectory[index]
            selected = source.index_select(0, self.donor_rows).to(
                device=hidden.device, dtype=hidden.dtype
            )
            self.dose[index] = (
                selected.float() - hidden[:, 0].float()
            ).norm(dim=-1).detach().cpu()
            changed = hidden.clone()
            changed[:, 0] = selected
            return (changed, *inputs[1:])

        return patch

    def _final_hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        hidden = _hidden(output)
        if hidden.ndim != 3 or hidden.shape[1] != 1:
            raise RuntimeError(
                f"Final layer: boundary replay requires one token, got {hidden.shape}"
            )
        selected = self.final_output.index_select(0, self.donor_rows).to(
            device=hidden.device, dtype=hidden.dtype
        )
        changed = hidden.clone()
        changed[:, 0] = selected
        return _replace_hidden(output, changed)

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []

    def __enter__(self) -> "BoundaryTrajectoryPatcher":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _run_boundary(
    model: Any,
    parts: Any,
    boundary_ids: Any,
    boundary_mask: Any,
    cache: Any,
    trajectory: dict[int, Any],
    final_output: Any,
    donor_rows: np.ndarray,
) -> tuple[Any, np.ndarray]:
    with BoundaryTrajectoryPatcher(
        parts, trajectory, final_output, donor_rows
    ) as patcher:
        output = _cached_forward(
            model,
            parts,
            boundary_ids,
            boundary_mask,
            past_key_values=cache,
        )
    return output, patcher.dose.numpy()


def run(
    config_path: Path,
    cohort_path: Path,
    output_dir: Path,
    split: str,
    max_questions: int | None,
    full_cache_tolerance: float,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if (
        config.prompt_mode != "baseline_matched_empty_history"
        or config.feedback_variant != "token_matched_test"
        or config.chat_serialization != "raw_qwen_chatml"
        or config.attn_implementation != "sdpa"
        or int(config.batch_size) != 4
    ):
        raise ValueError("Requires the exact canonical empty-history batch-four SDPA regime")
    cohort = json.loads(cohort_path.read_text())
    rows = [row for row in cohort["rows"] if row["split"] == split]
    if not rows:
        raise ValueError(f"No rows for split {split}")
    manifest = json.loads(Path(config.manifest_path).read_text())["questions"]
    questions = {row["id"]: row for row in manifest}

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, rows)
    model, processor, parts = load_model_and_processor(config)
    if len(parts.layers) != N_LAYERS:
        raise RuntimeError(f"Expected {N_LAYERS} blocks, got {len(parts.layers)}")
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    started = time.monotonic()
    metadata: dict[str, Any] = {
        "experiment": "first-decision complete local-update crossover",
        "config": config.as_dict(),
        "config_path": str(config_path),
        "cohort_path": str(cohort_path),
        "split": split,
        "n_frozen_questions": len(rows),
        "cells": list(CELLS),
        "conditions": list(CONDITIONS),
        "donor_rows": DONOR_ROWS.tolist(),
        "intervention": (
            "Preserve the recipient cache through the token immediately before the "
            "empty first-answer decision position, then replay the donor's complete "
            "64-block layer-input trajectory for that one boundary token. The model "
            "itself consequently writes donor-driven ordinary-attention K/V and GLA "
            "updates into recipient accumulated state."
        ),
        "complete_model_calls_per_exact_question": 9,
        "complete_model_work": (
            "one inclusive prefix capture/screen; one full natural forward; one "
            "pre-boundary cached prefix; identity, crossed, and full-donor one-token "
            "boundary steps; identity, crossed, and full-donor suffix forwards"
        ),
        "expected_cache_inventory": EXPECTED_INVENTORY,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
        "full_cache_tolerance": full_cache_tolerance,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    audit_path = output_dir / "prompt_audit.json"

    for qi, row in enumerate(rows):
        if max_questions is not None and qi >= int(max_questions):
            break
        if arrays["completed"][qi]:
            continue
        question_started = time.monotonic()
        question = questions[row["question_id"]]
        first_x = _remap_question(question, row["x_first_new_to_original"])
        first_y = _remap_question(question, row["y_first_new_to_original"])
        second = _remap_question(question, row["second_new_to_original"])

        prompts: list[str] = []
        token_rows: list[list[int]] = []
        boundaries: list[int] = []
        for first, condition in zip((first_x, first_x, first_y, first_y), CONDITIONS):
            prompt = render_chat(
                processor,
                _messages(config, first, second, condition),
                config.disable_thinking,
                config.chat_serialization,
            )
            boundary, ids = _decision_position(tokenizer, prompt)
            prompts.append(prompt)
            token_rows.append(ids)
            boundaries.append(boundary)

        aligned = bool(
            len({len(ids) for ids in token_rows}) == 1
            and len(set(boundaries)) == 1
        )
        arrays["token_aligned"][qi] = aligned
        if not aligned:
            arrays["completed"][qi] = True
            atomic_save_npz(result_path, **arrays)
            print(f"boundary crossover {split}: screened unaligned {row['question_id']}", flush=True)
            continue
        cut = boundaries[0] + 1
        if token_rows[0][cut:] != token_rows[2][cut:]:
            raise RuntimeError("Game X/Y suffixes differ after first-decision boundary")
        if token_rows[1][cut:] != token_rows[3][cut:]:
            raise RuntimeError("Neutral X/Y suffixes differ after first-decision boundary")
        if cut < 2:
            raise RuntimeError("First-decision boundary is unexpectedly early")

        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        if int(input_ids.shape[1]) != len(token_rows[0]):
            raise RuntimeError("Unexpected padding in aligned four-row batch")
        device = model_input_device(parts)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        calls = 0

        with BoundaryTrajectoryCollector(parts) as collector:
            prefix_output = _cached_forward(
                model, parts, input_ids[:, :cut], attention_mask[:, :cut]
            )
        calls += 1
        trajectory = collector.values
        final_boundary_output = collector.final_output
        if final_boundary_output is None:
            raise RuntimeError("The inclusive prefix did not capture the final boundary output")
        prefix_logits = _aggregate_logits(prefix_output, variant_ids)
        arrays["prefix_logits"][:, qi] = prefix_logits
        prefix_answers = prefix_logits.argmax(axis=-1)
        expected = np.asarray(
            [
                LETTERS.index(row["x_screen_winner_first_letter"]),
                LETTERS.index(row["x_screen_winner_first_letter"]),
                LETTERS.index(row["y_screen_winner_first_letter"]),
                LETTERS.index(row["y_screen_winner_first_letter"]),
            ]
        )
        exact = bool(np.array_equal(prefix_answers, expected))
        arrays["exact_eligible"][qi] = exact
        if not exact:
            arrays["model_calls"][qi] = calls
            arrays["duration_seconds"][qi] = time.monotonic() - question_started
            arrays["completed"][qi] = True
            atomic_save_npz(result_path, **arrays)
            answers = "".join(LETTERS[int(value)] for value in prefix_answers)
            print(
                f"boundary crossover {split}: screened {row['question_id']} "
                f"(exact decisions {answers})",
                flush=True,
            )
            del prefix_output
            torch.cuda.empty_cache()
            continue

        x_letter = LETTERS[int(prefix_answers[0])]
        y_letter = LETTERS[int(prefix_answers[2])]
        x_content = row["x_first_new_to_original"][x_letter]
        y_content = row["y_first_new_to_original"][y_letter]
        if x_content == y_content:
            raise RuntimeError("Exact first decisions do not differ semantically")
        arrays["exact_x_first_letter"][qi] = x_letter
        arrays["exact_y_first_letter"][qi] = y_letter
        arrays["exact_x_original_content"][qi] = x_content
        arrays["exact_y_original_content"][qi] = y_content
        arrays["exact_x_second_letter"][qi] = row["second_original_to_new"][x_content]
        arrays["exact_y_second_letter"][qi] = row["second_original_to_new"][y_content]
        del prefix_output

        natural_output = _forward(model, parts, input_ids, attention_mask)
        calls += 1
        natural_logits = _aggregate_logits(natural_output, variant_ids)
        arrays["natural_logits"][:, qi] = natural_logits
        del natural_output

        pre_output = _cached_forward(
            model, parts, input_ids[:, : cut - 1], attention_mask[:, : cut - 1]
        )
        calls += 1
        pre_cache = pre_output.past_key_values
        del pre_output
        inventory = _cache_inventory(pre_cache)
        if inventory != EXPECTED_INVENTORY:
            raise RuntimeError(f"Unexpected pre-boundary cache inventory {inventory}")
        boundary_ids = input_ids[:, cut - 1 : cut]
        boundary_mask = attention_mask[:, :cut]

        identity_output, identity_dose = _run_boundary(
            model,
            parts,
            boundary_ids,
            boundary_mask,
            copy.deepcopy(pre_cache),
            trajectory,
            final_boundary_output,
            np.arange(4, dtype=np.int64),
        )
        calls += 1
        identity_boundary_logits = _aggregate_logits(identity_output, variant_ids)
        arrays["identity_boundary_logits"][:, qi] = identity_boundary_logits
        arrays["identity_trajectory_dose"][qi] = identity_dose.mean(axis=1)

        suffix_ids = input_ids[:, cut:]
        identity_suffix = _cached_forward(
            model,
            parts,
            suffix_ids,
            attention_mask,
            past_key_values=identity_output.past_key_values,
        )
        calls += 1
        identity_logits = _aggregate_logits(identity_suffix, variant_ids)
        arrays["identity_logits"][:, qi] = identity_logits
        del identity_output, identity_suffix

        cross_output, cross_dose = _run_boundary(
            model,
            parts,
            boundary_ids,
            boundary_mask,
            copy.deepcopy(pre_cache),
            trajectory,
            final_boundary_output,
            DONOR_ROWS,
        )
        calls += 1
        cross_boundary_logits = _aggregate_logits(cross_output, variant_ids)
        arrays["cross_boundary_logits"][:, qi] = cross_boundary_logits
        arrays["cross_trajectory_dose"][qi] = cross_dose.mean(axis=1)
        cross_suffix = _cached_forward(
            model,
            parts,
            suffix_ids,
            attention_mask,
            past_key_values=cross_output.past_key_values,
        )
        calls += 1
        cross_logits = _aggregate_logits(cross_suffix, variant_ids)
        arrays["cross_logits"][:, qi] = cross_logits
        del cross_output, cross_suffix

        donor_pre_cache, counts = _swap_cache_families(
            pre_cache, 7, donor_rows=DONOR_ROWS
        )
        if counts != EXPECTED_INVENTORY:
            raise RuntimeError(f"Full-donor pre-cache swapped {counts}")
        full_donor_output, _full_donor_dose = _run_boundary(
            model,
            parts,
            boundary_ids,
            boundary_mask,
            donor_pre_cache,
            trajectory,
            final_boundary_output,
            DONOR_ROWS,
        )
        calls += 1
        full_donor_boundary_logits = _aggregate_logits(full_donor_output, variant_ids)
        arrays["full_donor_boundary_logits"][:, qi] = full_donor_boundary_logits
        full_donor_suffix = _cached_forward(
            model,
            parts,
            suffix_ids,
            attention_mask,
            past_key_values=full_donor_output.past_key_values,
        )
        calls += 1
        full_donor_logits = _aggregate_logits(full_donor_suffix, variant_ids)
        arrays["full_donor_logits"][:, qi] = full_donor_logits
        del full_donor_output, full_donor_suffix, donor_pre_cache

        identity_error = float(np.max(np.abs(identity_logits - natural_logits)))
        identity_choices = int(
            np.sum(identity_logits.argmax(axis=-1) != natural_logits.argmax(axis=-1))
        )
        full_error = float(
            np.max(np.abs(full_donor_logits - identity_logits[DONOR_ROWS]))
        )
        full_boundary_error = float(
            np.max(
                np.abs(
                    full_donor_boundary_logits
                    - identity_boundary_logits[DONOR_ROWS]
                )
            )
        )
        arrays["identity_vs_natural_max_error"][qi] = identity_error
        arrays["identity_vs_natural_choice_changes"][qi] = identity_choices
        arrays["full_donor_max_error"][qi] = full_error
        arrays["full_donor_boundary_max_error"][qi] = full_boundary_error
        if full_error > full_cache_tolerance or full_boundary_error > full_cache_tolerance:
            raise RuntimeError(
                f"{row['question_id']}: full-donor positive control failed "
                f"(suffix={full_error:.6g}, boundary={full_boundary_error:.6g})"
            )

        arrays["model_calls"][qi] = calls
        arrays["duration_seconds"][qi] = time.monotonic() - question_started
        arrays["completed"][qi] = True
        atomic_save_npz(result_path, **arrays)
        done = int(arrays["completed"].sum())
        print(
            f"boundary crossover {split}: {done}/{len(rows)}; "
            f"eligible={int(arrays['exact_eligible'].sum())}; "
            f"identity_err={identity_error:.4g}; donor_err={full_error:.3g}",
            flush=True,
        )

        if not audit_path.exists():
            audit_path.write_text(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "cells": list(CELLS),
                        "prefix_answers": [LETTERS[int(value)] for value in prefix_answers],
                        "boundary_positions": boundaries,
                        "boundary_tokens": tokenizer.convert_ids_to_tokens(
                            [ids[pos] for ids, pos in zip(token_rows, boundaries)]
                        ),
                        "cut": cut,
                        "cache_inventory": inventory,
                        "prompt_hashes": dict(
                            zip(CELLS, [prompt_hash(prompt) for prompt in prompts])
                        ),
                        "semantic_winners": {"x": x_content, "y": y_content},
                        "second_letters": {
                            "x": row["second_original_to_new"][x_content],
                            "y": row["second_original_to_new"][y_content],
                        },
                        "suffix_identity": {
                            "game": token_rows[0][cut:] == token_rows[2][cut:],
                            "neutral": token_rows[1][cut:] == token_rows[3][cut:],
                        },
                        "rendered_prompts": dict(zip(CELLS, prompts)),
                    },
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

        del (
            pre_cache,
        )
        torch.cuda.empty_cache()

    metadata.update(
        {
            "complete": bool(arrays["completed"].all()),
            "n_token_aligned": int(arrays["token_aligned"].sum()),
            "n_exact_eligible": int(arrays["exact_eligible"].sum()),
            "elapsed_seconds_after_model_load": time.monotonic() - started,
            "model_calls_total": int(arrays["model_calls"].sum()),
        }
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--full-cache-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    run(
        args.config,
        args.cohort,
        args.output,
        args.split,
        args.max_questions,
        args.full_cache_tolerance,
    )


if __name__ == "__main__":
    main()
