from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor, resolve_answer_tokens


LETTERS = "ABCD"


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _decode_dataset(
    spec: dict,
    parts: object,
    processor: object,
    batch_size: int,
    max_questions: int | None,
) -> dict:
    import torch

    source = Path(spec["source_results"])
    residual_path = Path(spec["residuals"])
    output = Path(spec["output"])
    with np.load(source, allow_pickle=False) as cached:
        question_ids = cached["question_ids"]
        conditions = cached["conditions"]
        direct_logits = cached["direct_logits"].astype(np.float32)
        rank_order = cached["rank_order"]
        jlens_scores = cached["jlens_scores"].astype(np.float32)

    if max_questions is not None:
        limit = min(int(max_questions), len(question_ids))
        question_ids = question_ids[:limit]
        direct_logits = direct_logits[:, :limit]
        rank_order = rank_order[:limit]
        jlens_scores = jlens_scores[:, :limit]

    residuals = np.load(residual_path, mmap_mode="r")
    expected_prefix = (len(conditions), 64, int(parts.embedding.weight.shape[-1]))
    if (
        residuals.shape[0] != expected_prefix[0]
        or residuals.shape[1] < len(question_ids)
        or residuals.shape[2:] != expected_prefix[1:]
    ):
        raise ValueError(
            f"Residual shape {residuals.shape} cannot supply "
            f"({len(conditions)}, {len(question_ids)}, 64, {expected_prefix[-1]})"
        )
    if residuals.dtype != np.float16:
        raise ValueError(f"Expected preserved FP16 residuals, got {residuals.dtype}")
    if not np.isfinite(direct_logits).all() or not np.isfinite(jlens_scores).all():
        raise ValueError("Source result arrays are non-finite")

    tokenizer = get_tokenizer(processor)
    config = ExperimentConfig.load(Path(spec["config"]))
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = [[token_id for _, token_id in resolved[letter]] for letter in LETTERS]
    flat_ids = [token_id for group in variant_ids for token_id in group]
    group_slices: list[slice] = []
    cursor = 0
    for group in variant_ids:
        group_slices.append(slice(cursor, cursor + len(group)))
        cursor += len(group)

    device = next(parts.final_norm.parameters()).device
    rows = parts.output_head.weight.detach()[flat_ids].to(device=device, dtype=torch.float32)
    bias = getattr(parts.output_head, "bias", None)
    if bias is not None:
        bias = bias.detach()[flat_ids].to(device=device, dtype=torch.float32)

    scores = np.empty((len(conditions), len(question_ids), 64, 4), dtype=np.float32)
    with torch.inference_mode():
        for layer in range(64):
            for condition in range(len(conditions)):
                for start in range(0, len(question_ids), batch_size):
                    stop = min(start + batch_size, len(question_ids))
                    hidden = torch.from_numpy(
                        np.asarray(residuals[condition, start:stop, layer]).copy()
                    ).to(device=device, dtype=parts.final_norm.weight.dtype)
                    normed = parts.final_norm(hidden).float()
                    token_logits = normed @ rows.T
                    if bias is not None:
                        token_logits = token_logits + bias
                    aggregate = torch.stack(
                        [torch.logsumexp(token_logits[:, group], dim=-1) for group in group_slices],
                        dim=-1,
                    )
                    scores[condition, start:stop, layer] = aggregate.cpu().numpy()
            if layer == 0 or (layer + 1) % 8 == 0 or layer == 63:
                print(f"{spec['name']} standard logit lens: {layer + 1}/64", flush=True)

    max_l64_error = float(np.max(np.abs(scores[:, :, -1] - direct_logits)))
    if max_l64_error > 0.10:
        raise RuntimeError(f"L64 reconstruction error too large: {max_l64_error}")
    scores[:, :, -1] = direct_logits
    if not np.isfinite(scores).all():
        raise RuntimeError("Standard-logit-lens output is non-finite")

    _atomic_npz(
        output,
        question_ids=question_ids,
        conditions=conditions,
        logit_lens_scores=scores,
        jlens_scores=jlens_scores,
        direct_logits=direct_logits,
        rank_order=rank_order,
    )
    metadata = {
        "dataset": spec["name"],
        "source_results": str(source),
        "residuals": str(residual_path),
        "output": str(output),
        "questions": len(question_ids),
        "conditions": conditions.tolist(),
        "layers": list(range(1, 65)),
        "readout": (
            "standard Qwen logit lens: each stored post-block residual is passed through "
            "the pinned model's exact final norm and selected bare-plus-space A-D "
            "unembedding rows; L64 is replaced by the stored exact live logits"
        ),
        "source_residual_dtype": str(residuals.dtype),
        "all_outputs_finite": True,
        "max_l64_reconstruction_error": max_l64_error,
    }
    metadata_path = output.with_name("run_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def run(specs_path: Path, batch_size: int, max_questions: int | None) -> None:
    specs = json.loads(specs_path.read_text())["datasets"]
    if not specs:
        raise ValueError("No dataset specifications")
    configs = [ExperimentConfig.load(Path(spec["config"])) for spec in specs]
    first = configs[0]
    for config in configs[1:]:
        if (config.model_id, config.model_revision) != (first.model_id, first.model_revision):
            raise ValueError("All datasets must use the same pinned model")
    model, processor, parts = load_model_and_processor(first)
    if len(parts.layers) != 64:
        raise ValueError(f"Expected 64 Qwen blocks, got {len(parts.layers)}")
    metadata = [
        _decode_dataset(spec, parts, processor, batch_size, max_questions) for spec in specs
    ]
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    run(args.specs, args.batch_size, args.max_questions)


if __name__ == "__main__":
    main()
