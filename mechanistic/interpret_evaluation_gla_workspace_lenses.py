from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor, model_input_device


CONDITIONS = ("Evaluation", "Matched Neutral")
POSITIONS = ("Evaluation period", "Final decision")


def _readable_english(token: str) -> bool:
    text = token.strip()
    if not text:
        return False
    return all(ord(character) < 128 and character.isprintable() for character in text) and any(
        character.isalpha() for character in text
    )


def _top_english(scores: Any, tokenizer: Any, *, largest: bool, k: int) -> list[dict[str, Any]]:
    import torch

    signed = scores if largest else -scores
    candidate_count = min(4096, int(scores.shape[-1]))
    values, ids = torch.topk(signed, k=candidate_count)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for signed_value, token_id in zip(values.detach().float().cpu(), ids.detach().cpu()):
        token_id_int = int(token_id)
        token = tokenizer.decode([token_id_int])
        if not _readable_english(token):
            continue
        display = token.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        normalized = display.strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        score = float(signed_value if largest else -signed_value)
        rows.append({"token_id": token_id_int, "token": display, "score": score})
        if len(rows) == k:
            break
    return rows


def _load_lens(path: str, filename: str):
    import torch
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=path,
        filename=filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(local, map_location="cpu", weights_only=False)
    return local, checkpoint


def interpret(
    config_path: Path,
    results_path: Path,
    output_path: Path,
    lens_repo: str,
    j_filename: str,
    r_filename: str,
    top_k: int,
    only_blocks: list[int] | None,
) -> None:
    import torch

    config = ExperimentConfig.load(config_path)
    with np.load(results_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"]):
        raise RuntimeError("The frozen GLA residual-write collection is incomplete")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    device = model_input_device(parts)
    j_path, j_checkpoint = _load_lens(lens_repo, j_filename)
    r_path, r_checkpoint = _load_lens(lens_repo, r_filename)
    checkpoints = {"J-lens": j_checkpoint, "R-lens": r_checkpoint}

    layers = arrays["gla_layers_zero_based"].astype(int).tolist()
    if only_blocks:
        selected = {int(block) - 1 for block in only_blocks}
        layer_slots = [slot for slot, layer in enumerate(layers) if layer in selected]
        missing = sorted(set(selected) - set(layers))
        if missing:
            raise ValueError(f"Requested non-GLA blocks: {[value + 1 for value in missing]}")
    else:
        layer_slots = list(range(len(layers)))

    width = int(parts.embedding.weight.shape[-1])
    for name, checkpoint in checkpoints.items():
        if int(checkpoint["d_model"]) != width:
            raise ValueError(f"{name} width does not match model")
        source_layers = sorted(int(value) for value in checkpoint["J"])
        if not set(layers) <= set(source_layers):
            raise ValueError(f"{name} is missing one or more GLA source layers")

    n = int(arrays["completed"].sum())
    means = arrays["mean_output_sum"].astype(np.float32) / n
    counts = arrays["w1_counts"].astype(int)
    w1_means = np.divide(
        arrays["w1_output_sum"].astype(np.float32),
        np.maximum(counts[:, :, None, None, None], 1),
    )
    head = parts.output_head.weight.detach()
    bias = getattr(parts.output_head, "bias", None)

    result: dict[str, Any] = {
        "definition": (
            "Matched J-lens and RelP R-lens vocabulary-direction readouts of the already-saved "
            "mean GLA residual writes. Each vector is lens-transported, final-RMS-normalized, "
            "and unembedded. Scores are direction readouts, not probabilities or causal effects."
        ),
        "input": str(results_path),
        "questions": n,
        "positions": list(POSITIONS),
        "lenses": {
            "J-lens": {"repo": lens_repo, "filename": j_filename, "local_path": j_path},
            "R-lens": {"repo": lens_repo, "filename": r_filename, "local_path": r_path},
        },
        "layers": {},
    }

    with torch.inference_mode():
        for progress, layer_slot in enumerate(layer_slots, 1):
            layer = layers[layer_slot]
            vectors: list[np.ndarray] = []
            labels: list[str] = []
            for position, position_name in enumerate(POSITIONS):
                vectors.extend(
                    [
                        means[0, layer_slot, position],
                        means[1, layer_slot, position],
                        means[0, layer_slot, position] - means[1, layer_slot, position],
                    ]
                )
                labels.extend(
                    [
                        f"{position_name} / Evaluation",
                        f"{position_name} / Matched Neutral",
                        f"{position_name} / Evaluation minus Matched Neutral",
                    ]
                )
                for letter_index, letter in enumerate("ABCD"):
                    vectors.append(
                        w1_means[0, letter_index, layer_slot, position]
                        - w1_means[1, letter_index, layer_slot, position]
                    )
                    labels.append(f"{position_name} / Evaluation minus Matched Neutral / W1={letter}")

            base = torch.from_numpy(np.stack(vectors)).to(device=device, dtype=torch.bfloat16)
            block_rows: dict[str, Any] = {}
            for lens_name, checkpoint in checkpoints.items():
                transport = checkpoint["J"][layer].to(device=device, dtype=torch.bfloat16)
                transported = base @ transport.T
                normed = parts.final_norm(transported.to(parts.final_norm.weight.dtype))
                scores = normed @ head.T
                if bias is not None:
                    # Bias is common across all directional views and is therefore excluded.
                    pass
                lens_rows: dict[str, Any] = {}
                for row_index, label in enumerate(labels):
                    lens_rows[label] = {
                        "positive": _top_english(scores[row_index], tokenizer, largest=True, k=top_k),
                        "negative": _top_english(scores[row_index], tokenizer, largest=False, k=top_k),
                    }
                block_rows[lens_name] = lens_rows
                del transport, transported, normed, scores
            result["layers"][str(layer + 1)] = block_rows
            print(f"Workspace lenses: {progress}/{len(layer_slots)} GLA blocks", flush=True)
            del base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched J- and R-lens GLA-write readouts")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens-repo", default="camilablank/workspace-lenses")
    parser.add_argument("--j-filename", default="qwen3.6-27b/j-lens/lens.pt")
    parser.add_argument("--r-filename", default="qwen3.6-27b/r-lens/lens.pt")
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--only-blocks", type=int, nargs="+")
    args = parser.parse_args()
    interpret(
        args.config,
        args.results,
        args.output,
        args.lens_repo,
        args.j_filename,
        args.r_filename,
        args.top_k,
        args.only_blocks,
    )


if __name__ == "__main__":
    main()

