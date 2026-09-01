from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor, model_input_device


def _readable_ascii(token: str) -> bool:
    stripped = token.strip()
    if not stripped:
        return False
    return all(ord(character) < 128 and character.isprintable() for character in stripped)


def _top_readable(scores: Any, tokenizer: Any, k: int = 12) -> list[dict]:
    import torch

    candidates = min(1024, int(scores.shape[-1]))
    values, ids = torch.topk(scores, k=candidates)
    rows = []
    for value, token_id in zip(values.detach().float().cpu(), ids.detach().cpu()):
        token = tokenizer.decode([int(token_id)])
        if not _readable_ascii(token):
            continue
        rows.append({
            "token_id": int(token_id),
            "token": token,
            "score": float(value),
        })
        if len(rows) == k:
            break
    return rows


def interpret(
    config_path: Path,
    results_path: Path,
    output_path: Path,
    lens_repo: str,
    lens_filename: str,
) -> None:
    import torch
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    with np.load(results_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"]):
        raise RuntimeError("GLA residual-write collection is incomplete")
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    device = model_input_device(parts)
    lens_path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    J = checkpoint["J"]
    layers = arrays["gla_layers_zero_based"].astype(int).tolist()
    n = int(arrays["completed"].sum())
    means = arrays["mean_output_sum"].astype(np.float32) / n
    w1_counts = arrays["w1_counts"].astype(int)
    w1_means = np.divide(
        arrays["w1_output_sum"].astype(np.float32),
        np.maximum(w1_counts[:, :, None, None, None], 1),
    )
    head = parts.output_head.weight.detach()
    result: dict[str, Any] = {
        "definition": (
            "Vocabulary-direction scores of mean GLA residual writes. Raw uses the "
            "shared unembedding directly. JLens first applies that block's learned "
            "Jacobian transport. Scores are direction readouts, not probabilities."
        ),
        "lens_repo": lens_repo,
        "lens_filename": lens_filename,
        "positions": ["evaluation_period", "final_decision"],
        "layers": {},
    }
    with torch.inference_mode():
        for layer_slot, layer in enumerate(layers):
            layer_rows: dict[str, Any] = {}
            vectors = []
            labels = []
            for position in range(2):
                vectors.extend([
                    means[0, layer_slot, position],
                    means[1, layer_slot, position],
                    means[0, layer_slot, position] - means[1, layer_slot, position],
                ])
                labels.extend([
                    f"position_{position}_evaluation",
                    f"position_{position}_neutral",
                    f"position_{position}_evaluation_minus_neutral",
                ])
                for w1 in range(4):
                    vectors.append(
                        w1_means[0, w1, layer_slot, position]
                        - w1_means[1, w1, layer_slot, position]
                    )
                    labels.append(f"position_{position}_w1_{'ABCD'[w1]}_difference")
            base = torch.from_numpy(np.stack(vectors)).to(device=device, dtype=torch.bfloat16)
            transported = base @ J[layer].to(device=device, dtype=torch.bfloat16).T
            for lens_name, decoded in (("raw", base), ("jlens", transported)):
                scores = decoded @ head.T
                lens_rows = {}
                for row_index, label in enumerate(labels):
                    lens_rows[label] = {
                        "positive": _top_readable(scores[row_index], tokenizer),
                        "negative": _top_readable(-scores[row_index], tokenizer),
                    }
                layer_rows[lens_name] = lens_rows
            result["layers"][str(layer + 1)] = layer_rows
            if (layer_slot + 1) % 8 == 0 or layer_slot + 1 == len(layers):
                print(f"Interpreted {layer_slot + 1}/{len(layers)} GLA blocks", flush=True)
            del base, transported
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode mean GLA residual writes")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    args = parser.parse_args()
    interpret(args.config, args.results, args.output, args.lens_repo, args.lens_filename)


if __name__ == "__main__":
    main()
