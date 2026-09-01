from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .collect_remapped_behavior import _remap_question
from .config import ExperimentConfig
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import (
    build_factorial_messages,
    build_messages,
    prompt_hash,
    repeated_question_turn,
)


LETTERS = "ABCD"
CONDITIONS = ("baseline", "game", "neutral")


def _chunks(values: list[int], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _factorial_remapped_messages(
    question: dict, remapped: dict, condition: str, prompt_mode: str
) -> list[dict[str, str]]:
    factorial = "incorrect_again" if condition == "game" else "lost_again"
    messages = build_factorial_messages(question, factorial, prompt_mode)
    original_repeat = repeated_question_turn(question)
    if not messages[-1]["content"].endswith(original_repeat):
        raise RuntimeError("Could not locate repeated question in factorial prompt")
    messages[-1]["content"] = (
        messages[-1]["content"][: -len(original_repeat)]
        + repeated_question_turn(remapped)
    )
    return messages


class FinalNormalizedResidualCollector:
    def __init__(self, final_norm, last_indices: list[int]):
        self.last_indices = last_indices
        self.value = None
        self.handle = final_norm.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output) -> None:
        import torch

        hidden = output[0] if isinstance(output, (tuple, list)) else output
        indices = torch.as_tensor(self.last_indices, device=hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        # Preserve the model's actual final-norm output without imposing an
        # additional fp16 rounding step. Float32 represents every bf16 value
        # exactly and keeps this endpoint faithful to the output-head input.
        self.value = hidden[batch, indices].detach().to("cpu", dtype=torch.float32)

    def close(self) -> None:
        self.handle.remove()


def collect(
    config_path: Path,
    plan_path: Path,
    baseline_path: Path,
    game_path: Path,
    neutral_path: Path,
    output_dir: Path,
    max_questions: int | None,
) -> None:
    import torch

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw_qwen_chatml")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    plan = json.loads(plan_path.read_text())
    plan_rows = {row["question_id"]: row for row in plan["rows"]}
    trusted = {
        "baseline": json.loads(baseline_path.read_text())["results"],
        "game": json.loads(game_path.read_text())["results"],
        "neutral": json.loads(neutral_path.read_text())["results"],
    }
    qids = list(trusted["game"])
    if max_questions is not None:
        qids = qids[: int(max_questions)]
    expected = set(qids)
    if not expected <= set(questions) or not expected <= set(plan_rows):
        raise ValueError("Manifest or remapping plan is missing requested questions")
    for condition in CONDITIONS:
        if not expected <= set(trusted[condition]):
            raise ValueError(f"Trusted {condition} results are incomplete")

    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: [token_id for _, token_id in resolved[letter]] for letter in LETTERS
    }
    flat_ids = [token_id for letter in LETTERS for token_id in variant_ids[letter]]
    mean_rows = torch.stack([
        parts.output_head.weight.detach()[variant_ids[letter]].float().mean(dim=0)
        for letter in LETTERS
    ]).cpu().numpy()

    n = len(qids)
    width = int(parts.embedding.weight.shape[-1])
    residual_path = output_dir / "normalized_residuals.tmp.npy"
    completed_path = output_dir / "completed.npy"
    logits_path = output_dir / "aggregated_logits.npy"
    hashes_path = output_dir / "prompt_hashes.npy"
    expected_shape = (len(CONDITIONS), n, width)
    if residual_path.exists():
        residuals = np.lib.format.open_memmap(residual_path, mode="r+")
        completed = np.load(completed_path)
        logits = np.load(logits_path)
        hashes = np.load(hashes_path)
        if tuple(residuals.shape) != expected_shape or completed.shape != (3, n):
            raise ValueError("Incompatible existing checkpoint")
    else:
        residuals = np.lib.format.open_memmap(
            residual_path, mode="w+", dtype=np.float32, shape=expected_shape
        )
        completed = np.zeros((3, n), dtype=bool)
        logits = np.full((3, n, 4), np.nan, dtype=np.float32)
        hashes = np.full((3, n), "", dtype="<U64")

    device = model_input_device(parts)
    max_logit_error = 0.0
    for ci, condition in enumerate(CONDITIONS):
        pending = [index for index in range(n) if not completed[ci, index]]
        for indices in _chunks(pending, config.batch_size):
            batch_qids = [qids[index] for index in indices]
            messages = []
            for qid in batch_qids:
                question = questions[qid]
                if condition == "baseline":
                    row_messages = build_messages(
                        question, "baseline", config.prompt_mode, config.feedback_variant
                    )
                else:
                    remapped = _remap_question(
                        question, plan_rows[qid]["new_to_original"]
                    )
                    row_messages = _factorial_remapped_messages(
                        question, remapped, condition, config.prompt_mode
                    )
                messages.append(row_messages)
            prompts = [
                render_chat(
                    processor, value, config.disable_thinking, config.chat_serialization
                )
                for value in messages
            ]
            current_hashes = [prompt_hash(value) for value in prompts]
            for qid, current in zip(batch_qids, current_hashes):
                expected_hash = trusted[condition][qid]["prompt_hash"]
                if current != expected_hash:
                    raise RuntimeError(
                        f"Prompt mismatch for {condition}/{qid}: {current} != {expected_hash}"
                    )

            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            collector = FinalNormalizedResidualCollector(parts.final_norm, last_indices)
            try:
                with torch.inference_mode():
                    kwargs = {
                        "input_ids": input_ids.to(device),
                        "attention_mask": attention_mask.to(device),
                        "use_cache": False,
                        "return_dict": True,
                    }
                    try:
                        result = model(**kwargs, logits_to_keep=1)
                    except TypeError:
                        result = model(**kwargs)
                if collector.value is None:
                    raise RuntimeError("Final norm hook did not run")
                captured = collector.value.numpy()
            finally:
                collector.close()

            full_logits = result.logits.detach().float().cpu()
            if full_logits.shape[1] == 1:
                final = full_logits[:, 0]
            else:
                final = full_logits[np.arange(len(indices)), last_indices]
            aggregate = torch.stack([
                torch.stack([
                    torch.logsumexp(final[bi, variant_ids[letter]], dim=0)
                    for letter in LETTERS
                ])
                for bi in range(len(indices))
            ]).numpy()
            trusted_logits = np.asarray([
                trusted[condition][qid]["aggregated_ad_logits"] for qid in batch_qids
            ], dtype=np.float32)
            error = float(np.max(np.abs(aggregate - trusted_logits)))
            max_logit_error = max(max_logit_error, error)

            residuals[ci, indices] = captured
            logits[ci, indices] = aggregate
            hashes[ci, indices] = current_hashes
            residuals.flush()
            completed[ci, indices] = True
            _atomic_npy(completed_path, completed)
            _atomic_npy(logits_path, logits)
            _atomic_npy(hashes_path, hashes)
            print(
                f"{condition}: {int(completed[ci].sum())}/{n}; "
                f"trusted-logit max error {max_logit_error:.6g}",
                flush=True,
            )

    if not completed.all() or not np.isfinite(logits).all():
        raise RuntimeError("Collection is incomplete or non-finite")
    trusted_all = np.asarray([
        [trusted[condition][qid]["aggregated_ad_logits"] for qid in qids]
        for condition in CONDITIONS
    ], dtype=np.float32)
    max_logit_error = float(np.max(np.abs(logits - trusted_all)))
    trusted_choice = np.argmax(trusted_all, axis=-1)
    collected_choice = np.argmax(logits, axis=-1)
    choice_agreement = float(np.mean(trusted_choice == collected_choice))
    if choice_agreement < 0.99:
        raise RuntimeError(
            f"Trusted aggregated choice agreement is only {choice_agreement:.3%}"
        )
    _atomic_npz(
        output_dir / "results.npz",
        question_ids=np.asarray(qids),
        conditions=np.asarray(CONDITIONS),
        normalized_residuals=np.asarray(residuals),
        aggregated_logits=logits,
        prompt_hashes=hashes,
        mean_answer_rows=mean_rows.astype(np.float32),
        original_to_new=np.asarray([
            [LETTERS.index(plan_rows[qid]["original_to_new"][letter]) for letter in LETTERS]
            for qid in qids
        ], dtype=np.int8),
        baseline_answer=np.asarray([trusted["baseline"][qid]["answer"] for qid in qids]),
    )
    metadata = {
        "config": config.as_dict(),
        "plan": str(plan_path),
        "trusted": {
            "baseline": str(baseline_path),
            "game": str(game_path),
            "neutral": str(neutral_path),
        },
        "n_questions": n,
        "conditions": list(CONDITIONS),
        "representation": "actual final RMSNorm output immediately before lm_head",
        "answer_rows": "mean unembedding row over bare and leading-space token variants",
        "max_trusted_aggregated_logit_error": max_logit_error,
        "trusted_aggregated_choice_agreement": choice_agreement,
        "all_prompt_hashes_exact": True,
        "answer_token_ids": {letter: variant_ids[letter] for letter in LETTERS},
        "flat_answer_token_ids": flat_ids,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    residual_path.unlink()
    completed_path.unlink(missing_ok=True)
    logits_path.unlink(missing_ok=True)
    hashes_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--neutral", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    collect(
        args.config, args.plan, args.baseline, args.game, args.neutral,
        args.output, args.max_questions,
    )


if __name__ == "__main__":
    main()
