from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .collect_remapped_behavior import _messages, _remap_question
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
    TOKEN_MATCHED_TEST_GAME_FEEDBACK,
    TOKEN_MATCHED_TEST_NEUTRAL_FEEDBACK,
    prompt_hash,
)
from .sublayer import _hidden, _replace_hidden, middle_norm


LETTERS = "ABCD"
CONDITIONS = ("incorrect", "neutral")
FEEDBACKS = {
    "incorrect": TOKEN_MATCHED_TEST_GAME_FEEDBACK,
    "neutral": TOKEN_MATCHED_TEST_NEUTRAL_FEEDBACK,
}
KEYWORDS = {"incorrect": "incorrect", "neutral": "lost"}
SOURCE_READOUTS = tuple(range(48, 57))
SUBLAYER_BLOCKS = tuple(range(52, 57))


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _chunks(values: list[int], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _locate_keyword(tokenizer: Any, prompt: str, condition: str) -> dict[str, Any]:
    feedback = FEEDBACKS[condition]
    feedback_start = prompt.find(feedback)
    if feedback_start < 0 or prompt.find(feedback, feedback_start + 1) >= 0:
        raise RuntimeError(f"Expected exactly one feedback sentence for {condition}")
    keyword = KEYWORDS[condition]
    start = prompt.find(keyword, feedback_start, feedback_start + len(feedback))
    if start < 0:
        raise RuntimeError(f"Could not locate {keyword!r} inside feedback")
    end = start + len(keyword)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    positions = [
        index for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    ]
    if len(positions) != 1:
        raise RuntimeError(f"Expected one token for {keyword!r}; got {positions}")
    position = positions[0]
    decoded = tokenizer.decode([ids[position]]).strip()
    if decoded != keyword:
        raise RuntimeError(f"Expected token {keyword!r}; decoded {decoded!r}")
    return {"ids": ids, "position": position, "token_id": ids[position]}


class KeywordReadoutCollector:
    """Collect one feedback token's post-block state at selected readouts."""

    def __init__(self, parts: Any, positions: list[int]) -> None:
        self.positions = positions
        self.values: dict[int, Any] = {}
        self.handles = [
            parts.layers[readout - 1].register_forward_hook(self._hook(readout))
            for readout in SOURCE_READOUTS
        ]

    def _hook(self, readout: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = _hidden(output)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            cols = torch.as_tensor(self.positions, device=hidden.device)
            self.values[readout] = hidden[rows, cols].detach().clone()
        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class KeywordReadoutPatcher:
    """Replace one token's complete post-block state for an exact cohort."""

    def __init__(self, parts: Any, readout: int, positions: list[int], source: Any):
        self.positions = positions
        self.source = source
        self.handle = parts.layers[readout - 1].register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = _hidden(output)
        if hidden.shape[0] != len(self.positions) or self.source.shape[0] != hidden.shape[0]:
            raise RuntimeError("Patch source and target cohort shapes differ")
        updated = hidden.clone()
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        cols = torch.as_tensor(self.positions, device=hidden.device)
        updated[rows, cols] = self.source.to(device=hidden.device, dtype=hidden.dtype)
        return _replace_hidden(output, updated)

    def close(self) -> None:
        self.handle.remove()


class LocalSublayerBoundaryCollector:
    """Capture only the three final-position boundaries in blocks 52--56."""

    def __init__(self, parts: Any, last_indices: list[int]) -> None:
        self.last_indices = last_indices
        self.values: dict[int, list[Any]] = {
            block: [None, None, None] for block in SUBLAYER_BLOCKS
        }
        self.handles = []
        for block in SUBLAYER_BLOCKS:
            layer = parts.layers[block - 1]
            self.handles.extend([
                layer.register_forward_pre_hook(self._pre_hook(block)),
                middle_norm(layer).register_forward_pre_hook(self._mid_hook(block)),
                layer.register_forward_hook(self._post_hook(block)),
            ])

    def _select(self, hidden: Any) -> Any:
        import torch

        rows = torch.arange(hidden.shape[0], device=hidden.device)
        cols = torch.as_tensor(self.last_indices, device=hidden.device)
        return hidden[rows, cols].detach().to("cpu", dtype=torch.float16)

    def _pre_hook(self, block: int):
        def capture(_module: Any, inputs: Any) -> None:
            self.values[block][0] = self._select(inputs[0])
        return capture

    def _mid_hook(self, block: int):
        def capture(_module: Any, inputs: Any) -> None:
            self.values[block][1] = self._select(inputs[0])
        return capture

    def _post_hook(self, block: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            self.values[block][2] = self._select(_hidden(output))
        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def stacked(self) -> Any:
        import torch

        missing = [
            (block, boundary)
            for block, row in self.values.items()
            for boundary, value in enumerate(row) if value is None
        ]
        if missing:
            raise RuntimeError(f"Missing local sublayer boundaries: {missing}")
        return torch.stack([
            torch.stack(self.values[block], dim=1) for block in SUBLAYER_BLOCKS
        ], dim=1)


def _forward(model: Any, parts: Any, input_ids: Any, attention_mask: Any):
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
            return model(**kwargs, logits_to_keep=1)
        except TypeError:
            return model(**kwargs)


def _aggregate_logits(output: Any, last_indices: list[int], variant_ids: list[list[int]]):
    import torch

    logits = output.logits.detach().float()
    if logits.shape[1] == 1:
        final = logits[:, 0]
    else:
        rows = torch.arange(logits.shape[0], device=logits.device)
        cols = torch.as_tensor(last_indices, device=logits.device)
        final = logits[rows, cols]
    return torch.stack(
        [torch.logsumexp(final[:, ids], dim=-1) for ids in variant_ids], dim=-1
    ).cpu().numpy()


def _decode_residuals(parts: Any, values: Any, variant_ids: list[list[int]]) -> np.ndarray:
    import torch

    device = model_input_device(parts)
    hidden = values.to(device=device, dtype=parts.final_norm.weight.dtype)
    shape = hidden.shape[:-1]
    flat = hidden.reshape(-1, hidden.shape[-1])
    with torch.inference_mode():
        normed = parts.final_norm(flat).float()
        rows = parts.output_head.weight.detach().float()
        scores = torch.stack(
            [torch.logsumexp(normed @ rows[ids].T, dim=-1) for ids in variant_ids],
            dim=-1,
        )
    return scores.reshape(*shape, 4).cpu().numpy()


def _initialize(path: Path, qids: list[str]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Checkpoint question IDs differ")
        return arrays
    n = len(qids)
    return {
        "question_ids": np.asarray(qids),
        "completed": np.zeros(n, dtype=bool),
        "natural_logits": np.full((2, n, 4), np.nan, dtype=np.float32),
        "patched_logits": np.full(
            (2, len(SOURCE_READOUTS), n, 4), np.nan, dtype=np.float32
        ),
        "sublayer_scores": np.full(
            (2, n, len(SUBLAYER_BLOCKS), 3, 4), np.nan, dtype=np.float32
        ),
        "keyword_state_distance": np.full(
            (n, len(SOURCE_READOUTS)), np.nan, dtype=np.float32
        ),
    }


def run(config_path: Path, plan_path: Path, output: Path,
        max_questions: int | None, max_cohorts: int | None) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.batch_size != 4:
        raise ValueError("Exact historical execution requires batch_size=4")
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml" or config.attn_implementation != "sdpa":
        raise ValueError("Requires exact raw ChatML + SDPA regime")

    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    qids = [row["id"] for row in manifest["questions"]]
    if max_questions is not None:
        qids = qids[:max_questions]
    plan = json.loads(plan_path.read_text())
    plan_rows = {row["question_id"]: row for row in plan["rows"]}
    if not set(qids) <= set(plan_rows):
        raise ValueError("Remapping plan is incomplete")

    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.npz"
    arrays = _initialize(result_path, qids)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = [[token_id for _, token_id in resolved[letter]] for letter in LETTERS]

    cohorts_done = 0
    start_time = time.perf_counter()
    for indices in _chunks(list(range(len(qids))), config.batch_size):
        if len(indices) != config.batch_size:
            raise ValueError("The frozen 500-question set must divide into cohorts of four")
        if all(arrays["completed"][indices]):
            continue
        if any(arrays["completed"][indices]):
            raise ValueError("A checkpoint may complete only whole exact cohorts")
        batch_qids = [qids[index] for index in indices]
        remapped = [
            _remap_question(questions[qid], plan_rows[qid]["new_to_original"])
            for qid in batch_qids
        ]
        prompts: dict[str, list[str]] = {}
        token_data: dict[str, list[dict[str, Any]]] = {}
        tokenized: dict[str, tuple[Any, Any, list[int]]] = {}
        for condition in CONDITIONS:
            prompts[condition] = [
                render_chat(
                    processor,
                    _messages(config, questions[qid], remapped_question, condition),
                    config.disable_thinking,
                    config.chat_serialization,
                )
                for qid, remapped_question in zip(batch_qids, remapped)
            ]
            token_data[condition] = [
                _locate_keyword(tokenizer, prompt, condition)
                for prompt in prompts[condition]
            ]
            tokenized[condition] = tokenize_batch(tokenizer, prompts[condition])

        for row in range(config.batch_size):
            game = token_data["incorrect"][row]
            neutral = token_data["neutral"][row]
            if game["position"] != neutral["position"] or len(game["ids"]) != len(neutral["ids"]):
                raise RuntimeError("Evaluation tokens are not position aligned")
            mismatches = [i for i, (a, b) in enumerate(zip(game["ids"], neutral["ids"])) if a != b]
            feedback_positions = set()
            for condition in CONDITIONS:
                prompt = prompts[condition][row]
                start = prompt.find(FEEDBACKS[condition])
                end = start + len(FEEDBACKS[condition])
                encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
                feedback_positions |= {
                    i for i, (a, b) in enumerate(encoded["offset_mapping"])
                    if b > a and a < end and b > start
                }
            if not set(mismatches) <= feedback_positions:
                raise RuntimeError("Prompts differ outside the feedback sentence")

        natural: dict[str, np.ndarray] = {}
        sources: dict[str, dict[int, Any]] = {}
        for ci, condition in enumerate(CONDITIONS):
            input_ids, mask, last = tokenized[condition]
            positions = [row["position"] for row in token_data[condition]]
            keyword_collector = KeywordReadoutCollector(parts, positions)
            boundary_collector = LocalSublayerBoundaryCollector(parts, last)
            try:
                result = _forward(model, parts, input_ids, mask)
            finally:
                keyword_collector.close()
                boundary_collector.close()
            natural[condition] = _aggregate_logits(result, last, variant_ids)
            arrays["natural_logits"][ci, indices] = natural[condition]
            sources[condition] = keyword_collector.values
            selected = boundary_collector.stacked()
            arrays["sublayer_scores"][ci, indices] = _decode_residuals(
                parts, selected, variant_ids
            )

        arrays["keyword_state_distance"][indices] = np.stack([
            torch.linalg.vector_norm(
                sources["incorrect"][readout].float()
                - sources["neutral"][readout].float(), dim=-1
            ).cpu().numpy()
            for readout in SOURCE_READOUTS
        ], axis=1)

        for ci, target in enumerate(CONDITIONS):
            source = "neutral" if target == "incorrect" else "incorrect"
            input_ids, mask, last = tokenized[target]
            positions = [row["position"] for row in token_data[target]]
            for ri, readout in enumerate(SOURCE_READOUTS):
                patcher = KeywordReadoutPatcher(
                    parts, readout, positions, sources[source][readout]
                )
                try:
                    result = _forward(model, parts, input_ids, mask)
                finally:
                    patcher.close()
                arrays["patched_logits"][ci, ri, indices] = _aggregate_logits(
                    result, last, variant_ids
                )

        arrays["completed"][indices] = True
        _atomic_npz(result_path, **arrays)
        cohorts_done += 1
        elapsed = time.perf_counter() - start_time
        print(
            f"cohort {indices[0] // 4 + 1}/125 complete; "
            f"session cohorts={cohorts_done}; elapsed={elapsed:.1f}s; "
            f"mean={elapsed / cohorts_done:.1f}s/cohort",
            flush=True,
        )
        if max_cohorts is not None and cohorts_done >= max_cohorts:
            break

    if bool(arrays["completed"].all()):
        metadata = {
            "status": "complete",
            "config": config.as_dict(),
            "plan": str(plan_path),
            "n_questions": len(qids),
            "source_readouts": list(SOURCE_READOUTS),
            "sublayer_blocks": list(SUBLAYER_BLOCKS),
            "sublayer_boundaries": ["before mixer", "after mixer / before MLP", "after MLP"],
            "intervention": (
                "For every exact historical four-question cohort, replace the complete "
                "post-block residual of incorrect with paired lost, or vice versa, at "
                "one readout only; allow that replacement to propagate downstream."
            ),
            "prompt_audit": {
                condition: {
                    "feedback": FEEDBACKS[condition],
                    "keyword": KEYWORDS[condition],
                    "prompt_hash": prompt_hash(prompts[condition][0]),
                    "token_id": token_data[condition][0]["token_id"],
                    "absolute_position": token_data[condition][0]["position"],
                }
                for condition in CONDITIONS
            },
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
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(args.config, args.plan, args.output, args.max_questions, args.max_cohorts)


if __name__ == "__main__":
    main()
