from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor, model_input_device


LEXICONS = {
    "exclusion": (
        "exclude", "excludes", "excluded", "excluding",
        "restrict", "restricted", "restriction", "restrictions",
        "ban", "banned", "banning",
        "reject", "rejected", "rejection",
        "eliminate", "eliminated",
    ),
    "alternative": (
        "instead", "other", "another", "alternative",
        "change", "changed", "changing", "retry", "different",
    ),
}
POSITIONS = ("feedback_end", "decision")


def _resolve_tokens(tokenizer):
    ids: list[int] = []
    layout: list[dict] = []
    seen: set[int] = set()
    for family, concepts in LEXICONS.items():
        for concept in concepts:
            for text in (concept, " " + concept, concept.capitalize(), " " + concept.capitalize()):
                encoded = tokenizer.encode(text, add_special_tokens=False)
                if len(encoded) != 1:
                    continue
                token_id = int(encoded[0])
                if token_id in seen:
                    continue
                seen.add(token_id)
                ids.append(token_id)
                layout.append({"family": family, "concept": concept, "text": text, "token_id": token_id})
    return ids, layout


def collect(config_path: Path, jlens_root: Path, output: Path, lens_repo: str, lens_filename: str) -> None:
    import torch
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    metadata = json.loads((jlens_root / "position_residuals_metadata.json").read_text())
    anchors = metadata["anchors"]
    position_indices = [anchors.index(name) for name in POSITIONS]
    residuals = np.load(jlens_root / "position_residuals.npy", mmap_mode="r")
    if residuals.shape[:4] != (2, len(metadata["question_ids"]), 64, len(anchors)):
        raise ValueError(f"Unexpected position residual shape: {residuals.shape}")

    lens_path = hf_hub_download(repo_id=lens_repo, filename=lens_filename)
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    jacobians = checkpoint["J"]
    if sorted(int(layer) for layer in jacobians) != list(range(63)):
        raise ValueError("Expected 63 learned JLens maps")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    token_ids, layout = _resolve_tokens(tokenizer)
    rows = parts.output_head.weight.detach()[token_ids].float()
    bias = getattr(parts.output_head, "bias", None)
    if bias is not None:
        bias = bias.detach()[token_ids].float()
    device = model_input_device(parts)
    scores = np.empty((2, residuals.shape[1], len(POSITIONS), 64, len(token_ids)), dtype=np.float16)

    @torch.inference_mode()
    def transform(values: np.ndarray, jacobian, target: np.ndarray) -> None:
        batch_size = 64
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            source = torch.from_numpy(np.asarray(values[start:stop])).to(device, dtype=torch.float16)
            transported = source if jacobian is None else source @ jacobian.T
            normed = parts.final_norm(transported.to(parts.final_norm.weight.dtype)).float()
            logits = normed @ rows.T
            if bias is not None:
                logits = logits + bias
            target[start:stop] = logits.cpu().to(torch.float16).numpy()

    for layer in range(63):
        jacobian = jacobians[layer].to(device, dtype=torch.float16)
        for condition in range(2):
            for pi, anchor_index in enumerate(position_indices):
                transform(residuals[condition, :, layer, anchor_index], jacobian, scores[condition, :, pi, layer])
        del jacobian
        if layer == 0 or (layer + 1) % 8 == 0 or layer == 62:
            print(f"transmission JLens map {layer + 1}/63", flush=True)
    for condition in range(2):
        for pi, anchor_index in enumerate(position_indices):
            transform(residuals[condition, :, 63, anchor_index], None, scores[condition, :, pi, 63])

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "transmission_scores.npz",
        scores=scores,
        conditions=np.asarray(("incorrect", "neutral")),
        positions=np.asarray(POSITIONS),
        question_ids=np.asarray(metadata["question_ids"]),
        token_ids=np.asarray(token_ids, dtype=np.int32),
    )
    (output / "token_layout.json").write_text(json.dumps(layout, indent=2))
    (output / "collection_metadata.json").write_text(json.dumps({
        "config": config.as_dict(),
        "lens_repo": lens_repo,
        "lens_filename": lens_filename,
        "positions": list(POSITIONS),
        "lexicons": {key: list(values) for key, values in LEXICONS.items()},
        "n_questions": len(metadata["question_ids"]),
        "question_ids": metadata["question_ids"],
        "layer_alignment": "learned JLens maps for readouts 1--63; natural unembedding for readout 64",
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jlens-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    args = parser.parse_args()
    collect(args.config, args.jlens_root, args.output, args.lens_repo, args.lens_filename)


if __name__ == "__main__":
    main()
