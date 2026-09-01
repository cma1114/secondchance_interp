from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor, model_input_device
from .run_jlens_exclusion_bridge_intervention import (
    EXCLUSION_CONCEPTS,
    _family_score,
    _resolve_concepts,
)
from .run_jlens_exclusion_layerwise import SOURCE_LAYERS


def run(
    config_path: Path,
    jlens_root: Path,
    output: Path,
    lens_repo: str,
    lens_filename: str,
) -> None:
    import torch
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    metadata = json.loads((jlens_root / "position_residuals_metadata.json").read_text())
    anchors = metadata["anchors"]
    feedback_index = anchors.index("feedback_end")
    residuals = np.load(jlens_root / "position_residuals.npy", mmap_mode="r")

    lens_path = hf_hub_download(repo_id=lens_repo, filename=lens_filename)
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    device = model_input_device(parts)
    exclusion_ids, exclusion_groups = _resolve_concepts(tokenizer, EXCLUSION_CONCEPTS)
    exclusion_rows = parts.output_head.weight.detach()[exclusion_ids].float()

    @torch.inference_mode()
    def scores(values: np.ndarray, jacobian) -> np.ndarray:
        result = []
        for start in range(0, len(values), 64):
            batch = torch.from_numpy(values[start:start + 64]).to(device, dtype=torch.float16)
            transported = batch @ jacobian.T
            normed = parts.final_norm(transported.to(parts.final_norm.weight.dtype)).float()
            logits = normed @ exclusion_rows.T
            result.append(_family_score(logits, exclusion_groups).cpu().numpy())
        return np.concatenate(result)

    rows = []
    for layer in SOURCE_LAYERS:
        game = np.asarray(residuals[0, :, layer, feedback_index], dtype=np.float32)
        neutral = np.asarray(residuals[1, :, layer, feedback_index], dtype=np.float32)
        reference = torch.from_numpy(game.mean(axis=0).copy()).to(
            device, dtype=torch.float32
        ).requires_grad_(True)
        jacobian32 = checkpoint["J"][layer].to(device, dtype=torch.float32)
        transported = reference @ jacobian32.T
        normed = parts.final_norm(transported.to(parts.final_norm.weight.dtype)).float()
        score = _family_score(normed @ exclusion_rows.T, exclusion_groups)
        gradient = torch.autograd.grad(score, reference)[0]
        direction = (gradient / gradient.norm()).detach().cpu().numpy().astype(np.float32)
        game_coordinate = game @ direction
        neutral_coordinate = neutral @ direction
        game_ablated = game + (neutral_coordinate - game_coordinate)[:, None] * direction[None]
        neutral_inserted = neutral + (game_coordinate - neutral_coordinate)[:, None] * direction[None]
        jacobian16 = jacobian32.to(dtype=torch.float16)
        game_score = scores(game, jacobian16)
        game_ablated_score = scores(game_ablated, jacobian16)
        neutral_score = scores(neutral, jacobian16)
        neutral_inserted_score = scores(neutral_inserted, jacobian16)
        rows.append({
            "layer": layer + 1,
            "mean_coordinate_game_minus_neutral": float((game_coordinate - neutral_coordinate).mean()),
            "game_natural_exclusion_score": float(game_score.mean()),
            "game_ablated_exclusion_score": float(game_ablated_score.mean()),
            "game_ablation_score_delta": float((game_ablated_score - game_score).mean()),
            "neutral_natural_exclusion_score": float(neutral_score.mean()),
            "neutral_inserted_exclusion_score": float(neutral_inserted_score.mean()),
            "neutral_insertion_score_delta": float((neutral_inserted_score - neutral_score).mean()),
        })
        del jacobian32, jacobian16, reference, transported, normed, score, gradient
        print(f"validated exclusion score manipulation: L{layer + 1}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "definition": (
            "JLens exclusion-family score before and after paired one-dimensional "
            "coordinate replacement at the feedback-ending period"
        ),
        "layers": rows,
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
    run(args.config, args.jlens_root, args.output, args.lens_repo, args.lens_filename)


if __name__ == "__main__":
    main()
