from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor, model_input_device


LETTERS = "ABCD"
POSITIVE_TOKENS = (
    "正确答案", "正确", " correct", " Correct", "correct", "Correct",
    " right", " Right", "right", "Right",
)
NEGATIVE_TOKENS = (
    "错误", " incorrect", " Incorrect", "incorrect", "Incorrect",
    " wrong", " Wrong", "wrong", "Wrong",
)
RANK_LAYERS = (46, 47)


def _single_token_ids(tokenizer, texts: tuple[str, ...]) -> dict[str, int]:
    result = {}
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            result[text] = int(ids[0])
    return result


def _bootstrap_rate(values: np.ndarray, seed: int, draws: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    sample = values[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    return np.quantile(sample, (0.025, 0.975)).tolist()


def analyze(
    config_path: Path,
    discovery_root: Path,
    confirmation_root: Path,
    baseline_results: Path,
    manifest_path: Path,
    output: Path,
    lens_repo: str,
    lens_filename: str,
) -> dict:
    import torch
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    discovery_meta = json.loads((discovery_root / "metadata.json").read_text())
    confirmation_meta = json.loads((confirmation_root / "metadata.json").read_text())
    discovery = np.load(discovery_root / "position_residuals.npy", mmap_mode="r")
    confirmation = np.load(confirmation_root / "position_residuals.npy", mmap_mode="r")
    qids = list(discovery_meta["question_ids"]) + list(confirmation_meta["question_ids"])
    if len(qids) != 500 or len(set(qids)) != 500:
        raise ValueError("Expected 500 unique discovery+confirmation questions")
    anchor_indices_discovery = [
        discovery_meta["anchors"].index(f"line_end_{letter}") for letter in LETTERS
    ]
    anchor_indices_confirmation = [
        confirmation_meta["anchors"].index(f"line_end_{letter}") for letter in LETTERS
    ]

    baseline = json.loads(baseline_results.read_text())["results"]
    manifest = {row["id"]: row for row in json.loads(manifest_path.read_text())["questions"]}
    winners = np.asarray([LETTERS.index(baseline[qid]["answer"]) for qid in qids])
    objective = np.asarray([LETTERS.index(manifest[qid]["correct_answer"]) for qid in qids])
    wrong = winners != objective

    lens_path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    jacobians = checkpoint["J"]
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    device = model_input_device(parts)

    positive = _single_token_ids(tokenizer, POSITIVE_TOKENS)
    negative = _single_token_ids(tokenizer, NEGATIVE_TOKENS)
    if "正确答案" not in positive:
        raise ValueError("The exact 正确答案 token is not a single vocabulary token")
    positive_ids = torch.tensor(sorted(set(positive.values())), device=device)
    negative_ids = torch.tensor(sorted(set(negative.values())), device=device)
    exact_id = positive["正确答案"]
    output_weight = parts.output_head.weight

    exact_scores = np.empty((500, 64, 4), dtype=np.float32)
    margins = np.empty((500, 64, 4), dtype=np.float32)
    exact_ranks: dict[int, np.ndarray] = {
        layer: np.empty((500, 4), dtype=np.int32) for layer in RANK_LAYERS
    }

    for layer in range(64):
        values = np.concatenate([
            np.asarray(discovery[:, layer, anchor_indices_discovery]),
            np.asarray(confirmation[:, layer, anchor_indices_confirmation]),
        ], axis=0)
        natural = torch.from_numpy(values.reshape(-1, values.shape[-1])).to(
            device, dtype=torch.float16
        )
        if layer < 63:
            transformed = natural @ jacobians[layer].to(device, dtype=torch.float16).T
        else:
            transformed = natural
        with torch.inference_mode():
            normed = parts.final_norm(transformed.to(parts.final_norm.weight.dtype))
            pos = normed @ output_weight[positive_ids].T
            neg = normed @ output_weight[negative_ids].T
            exact = normed @ output_weight[exact_id]
            margin = torch.logsumexp(pos.float(), dim=-1) - torch.logsumexp(
                neg.float(), dim=-1
            )
            exact_scores[:, layer] = exact.float().reshape(500, 4).cpu().numpy()
            margins[:, layer] = margin.reshape(500, 4).cpu().numpy()
            if layer + 1 in RANK_LAYERS:
                ranks = []
                for start in range(0, len(normed), 128):
                    batch = parts.output_head(normed[start:start + 128]).float()
                    target = batch[:, exact_id]
                    ranks.append((batch > target[:, None]).sum(dim=1).add(1).cpu())
                exact_ranks[layer + 1][:] = torch.cat(ranks).reshape(500, 4).numpy()
        del values, natural, transformed, normed, pos, neg, exact, margin
        if layer == 0 or (layer + 1) % 8 == 0 or layer + 1 in RANK_LAYERS:
            print(f"newline correctness JLens: {layer + 1}/64", flush=True)

    per_layer = []
    rows = np.arange(500)
    for layer in range(64):
        exact_prediction = exact_scores[:, layer].argmax(axis=1)
        margin_prediction = margins[:, layer].argmax(axis=1)
        per_layer.append({
            "layer": layer + 1,
            "exact_token_argmax_matches_baseline": float(np.mean(exact_prediction == winners)),
            "exact_token_argmax_matches_objective": float(np.mean(exact_prediction == objective)),
            "correct_minus_incorrect_argmax_matches_baseline": float(np.mean(margin_prediction == winners)),
            "correct_minus_incorrect_argmax_matches_objective": float(np.mean(margin_prediction == objective)),
            "baseline_winner_exact_score_advantage": float(np.mean(
                exact_scores[rows, layer, winners]
                - (exact_scores[:, layer].sum(axis=1) - exact_scores[rows, layer, winners]) / 3
            )),
            "baseline_winner_margin_advantage": float(np.mean(
                margins[rows, layer, winners]
                - (margins[:, layer].sum(axis=1) - margins[rows, layer, winners]) / 3
            )),
            "exact_token_argmax_distribution": {
                letter: float(np.mean(exact_prediction == index))
                for index, letter in enumerate(LETTERS)
            },
            "margin_argmax_distribution": {
                letter: float(np.mean(margin_prediction == index))
                for index, letter in enumerate(LETTERS)
            },
        })

    selected = {}
    for layer in RANK_LAYERS:
        exact_prediction = exact_scores[:, layer - 1].argmax(axis=1)
        margin_prediction = margins[:, layer - 1].argmax(axis=1)
        rank = exact_ranks[layer]
        winner_rank = rank[rows, winners]
        nonwinner_mask = np.ones((500, 4), dtype=bool)
        nonwinner_mask[rows, winners] = False
        nonwinner_ranks = rank[nonwinner_mask]
        selected[str(layer)] = {
            "exact_token_top12_at_baseline_winner": float(np.mean(winner_rank <= 12)),
            "exact_token_top12_at_nonwinner": float(np.mean(nonwinner_ranks <= 12)),
            "exact_token_top12_by_absolute_letter": {
                letter: float(np.mean(rank[:, index] <= 12))
                for index, letter in enumerate(LETTERS)
            },
            "exact_argmax_matches_baseline": float(np.mean(exact_prediction == winners)),
            "exact_argmax_matches_objective": float(np.mean(exact_prediction == objective)),
            "margin_argmax_matches_baseline": float(np.mean(margin_prediction == winners)),
            "margin_argmax_matches_objective": float(np.mean(margin_prediction == objective)),
            "wrong_trials_margin_argmax_matches_baseline": float(np.mean(margin_prediction[wrong] == winners[wrong])),
            "wrong_trials_margin_argmax_matches_objective": float(np.mean(margin_prediction[wrong] == objective[wrong])),
            "wrong_trial_count": int(wrong.sum()),
            "margin_baseline_match_ci": _bootstrap_rate(margin_prediction == winners, 20260810 + layer),
            "margin_objective_match_ci": _bootstrap_rate(margin_prediction == objective, 20260910 + layer),
        }

    summary = {
        "definitions": {
            "exact_token": "The single vocabulary token 正确答案 ('correct answer').",
            "correct_minus_incorrect_margin": (
                "Log-sum-exp score over single-token English/Chinese correct/right variants "
                "minus log-sum-exp score over incorrect/wrong/error variants."
            ),
            "argmax": "Which of the four option-newline positions has the largest score.",
        },
        "n_questions": 500,
        "baseline_wrong_questions": int(wrong.sum()),
        "positive_tokens": positive,
        "negative_tokens": negative,
        "selected_layers": selected,
        "per_layer": per_layer,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    np.savez_compressed(
        output.with_suffix(".npz"),
        question_ids=np.asarray(qids),
        winners=winners,
        objective=objective,
        exact_scores=exact_scores,
        margins=margins,
        exact_ranks_l46=exact_ranks[46],
        exact_ranks_l47=exact_ranks[47],
    )
    print(json.dumps({
        "n_questions": 500,
        "baseline_wrong_questions": int(wrong.sum()),
        "selected_layers": selected,
    }, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    args = parser.parse_args()
    analyze(
        args.config,
        args.discovery_root,
        args.confirmation_root,
        args.baseline_results,
        args.manifest,
        args.output,
        args.lens_repo,
        args.lens_filename,
    )


if __name__ == "__main__":
    main()
