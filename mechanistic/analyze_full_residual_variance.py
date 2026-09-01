from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data import decision_letter, load_activation_dataset
from .io import shard_path
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    resolve_answer_tokens,
)


CONDITIONS = ("baseline", "incorrect", "neutral")
LABELS = {"incorrect": "Game", "neutral": "Neutral"}


def _labels(data, condition: str) -> np.ndarray:
    labels = []
    for qid in data.question_ids:
        answer = decision_letter(data.metadata[(qid, condition)])
        if answer not in "ABCD":
            raise ValueError(f"Non-A-D output for {qid}/{condition}: {answer!r}")
        labels.append("ABCD".index(answer))
    return np.asarray(labels, dtype=np.int64)


def _load_residuals(root: Path, condition: str, qids: list[str]) -> np.ndarray:
    with np.load(shard_path(root, condition, qids[0]), allow_pickle=False) as shard:
        shape = shard["residuals"].shape
    if shape[0] != 65:
        raise ValueError(f"Expected embedding plus 64 residual readouts, got {shape}")
    values = np.empty((len(qids), 64, shape[-1]), dtype=np.float16)
    for index, qid in enumerate(qids):
        with np.load(shard_path(root, condition, qid), allow_pickle=False) as shard:
            values[index] = shard["residuals"][1:].astype(np.float16)
    return values


def _bootstrap_weights(strata: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """Paired cluster bootstrap giving each Baseline answer letter equal weight."""
    rng = np.random.default_rng(seed)
    weights = np.zeros((draws + 1, len(strata)), dtype=np.float32)
    for letter in range(4):
        indices = np.flatnonzero(strata == letter)
        if len(indices) < 2:
            raise ValueError(f"Too few questions in Baseline-answer stratum {letter}")
        weights[0, indices] = 0.25 / len(indices)
        counts = rng.multinomial(
            len(indices), np.full(len(indices), 1.0 / len(indices)), size=draws
        )
        weights[1:, indices] = counts * (0.25 / len(indices))
    return weights


def _variance_stats(values, weights, answer_basis=None):
    import torch

    norms = torch.sum(values * values, dim=1)
    means = weights @ values
    total = weights @ norms - torch.sum(means * means, dim=1)
    total = torch.clamp(total, min=0)
    result = {"means": means, "total": total}
    if answer_basis is not None:
        projected = values @ answer_basis
        projected_norms = torch.sum(projected * projected, dim=1)
        projected_means = weights @ projected
        answer = weights @ projected_norms - torch.sum(projected_means * projected_means, dim=1)
        answer = torch.clamp(answer, min=0)
        result["answer"] = answer
        result["complement"] = torch.clamp(total - answer, min=0)
    return result


def _summarize(draws: np.ndarray) -> dict[str, float | list[float]]:
    return {
        "estimate": float(draws[0]),
        "ci": np.quantile(draws[1:], [0.025, 0.975]).astype(float).tolist(),
    }


def _ratio_summary(numerator, denominator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio = (numerator / denominator).float().cpu().numpy()
    return ratio[0], np.quantile(ratio[1:], 0.025), np.quantile(ratio[1:], 0.975)


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })


def _plot(output: Path, layers: np.ndarray, curves: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib.pyplot as plt

    _style()
    colors = {"incorrect": "#0072B2", "neutral": "#D55E00"}
    panels = (
        ("raw_total", "A  Raw residual stream", "Raw variance / Baseline"),
        ("normed_total", "B  JLens-normalized full stream", "Normalized variance / Baseline"),
        ("answer", "C  A–D contrast subspace", "A–D variance / Baseline"),
        ("complement", "D  Orthogonal complement", "Complement variance / Baseline"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.35), sharex=True)
    for axis, (metric, title, ylabel) in zip(axes.flat, panels):
        axis.axhline(1, color="#555555", lw=0.8, ls=(0, (3, 2)))
        for condition in ("incorrect", "neutral"):
            values = curves[condition][metric]
            color = colors[condition]
            axis.fill_between(
                layers, values["low"], values["high"], color=color, alpha=0.24, linewidth=0
            )
            axis.plot(layers, values["low"], color=color, lw=0.55, alpha=0.65)
            axis.plot(layers, values["high"], color=color, lw=0.55, alpha=0.65)
            axis.plot(layers, values["estimate"], color=color, lw=1.55, label=LABELS[condition])
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xlim(1, 64)
        axis.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.7)
        axis.set_axisbelow(True)
    axes[0, 0].legend(frameon=False)
    for axis in axes[1]:
        axis.set_xlabel("Residual readout")
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    fig.text(
        0.5, -0.005,
        "Condition means are removed across questions; bands are paired, answer-letter-stratified 95% bootstrap CIs.",
        ha="center", va="top", fontsize=7.2, color="#555555",
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def analyze(
    config_path: Path,
    baseline_root: Path,
    second_chance_root: Path,
    lens_repo: str,
    lens_filename: str,
    output_data: Path,
    output_figure: Path,
    bootstrap: int,
    seed: int,
) -> dict:
    import torch
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    baseline_data = load_activation_dataset(baseline_root, ["baseline"])
    second_data = load_activation_dataset(second_chance_root, ["incorrect", "neutral"])
    if baseline_data.question_ids != second_data.question_ids:
        raise ValueError("Baseline and Second Chance question orders differ")
    qids = baseline_data.question_ids
    prior = _labels(baseline_data, "baseline")

    print("loading residual shards", flush=True)
    residuals = {
        "baseline": _load_residuals(baseline_root, "baseline", qids),
        "incorrect": _load_residuals(second_chance_root, "incorrect", qids),
        "neutral": _load_residuals(second_chance_root, "neutral", qids),
    }

    lens_path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    jacobians = checkpoint["J"]
    if sorted(int(layer) for layer in jacobians) != list(range(63)):
        raise ValueError("Expected JLens maps for readouts 1 through 63")

    print("loading model normalization and output directions", flush=True)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    letter_rows = []
    for letter in "ABCD":
        token_ids = [token_id for _, token_id in resolved[letter]]
        letter_rows.append(parts.output_head.weight.detach()[token_ids].float().mean(dim=0))
    letter_rows = torch.stack(letter_rows)
    letter_rows = letter_rows - letter_rows.mean(dim=0, keepdim=True)
    _, singular_values, right = torch.linalg.svd(letter_rows, full_matrices=False)
    rank = int(torch.sum(singular_values > singular_values[0] * 1e-5).item())
    if rank != 3:
        raise ValueError(f"Expected a three-dimensional A-D contrast subspace, got rank {rank}")

    device = model_input_device(parts)
    answer_basis = right[:rank].T.to(device=device, dtype=torch.float32)
    weights = torch.from_numpy(_bootstrap_weights(prior, bootstrap, seed)).to(device)
    layers = np.arange(1, 65)
    metric_names = ("raw_total", "normed_total", "answer", "complement")
    curves = {
        condition: {
            metric: {
                "estimate": np.empty(64), "low": np.empty(64), "high": np.empty(64)
            }
            for metric in metric_names
        }
        for condition in ("incorrect", "neutral")
    }
    gamma = {
        condition: {name: np.empty(64) for name in ("estimate", "low", "high")}
        for condition in ("incorrect", "neutral")
    }
    correlation = {
        condition: {name: np.empty(64) for name in ("estimate", "low", "high")}
        for condition in ("incorrect", "neutral")
    }

    batch_size = 64

    @torch.inference_mode()
    def transport(values: np.ndarray, J):
        batches = []
        for start in range(0, len(values), batch_size):
            residual = torch.from_numpy(values[start : start + batch_size]).to(
                device=device, dtype=torch.float16
            )
            transported = residual if J is None else residual @ J.T
            batches.append(
                parts.final_norm(transported.to(parts.final_norm.weight.dtype)).float()
            )
        return torch.cat(batches, dim=0)

    with torch.inference_mode():
        for layer in range(64):
            J = None if layer == 63 else jacobians[layer].to(device=device, dtype=torch.float16)
            raw_stats = {}
            normed_stats = {}
            normed_values = {}
            for condition in CONDITIONS:
                raw = torch.from_numpy(residuals[condition][:, layer]).to(
                    device=device, dtype=torch.float32
                )
                raw_stats[condition] = _variance_stats(raw, weights)
                normed = transport(residuals[condition][:, layer], J)
                normed_values[condition] = normed
                normed_stats[condition] = _variance_stats(normed, weights, answer_basis)

            for condition in ("incorrect", "neutral"):
                for metric, source, key in (
                    ("raw_total", raw_stats, "total"),
                    ("normed_total", normed_stats, "total"),
                    ("answer", normed_stats, "answer"),
                    ("complement", normed_stats, "complement"),
                ):
                    point, low, high = _ratio_summary(
                        source[condition][key], source["baseline"][key]
                    )
                    curves[condition][metric]["estimate"][layer] = point
                    curves[condition][metric]["low"][layer] = low
                    curves[condition][metric]["high"][layer] = high

                baseline = normed_values["baseline"]
                current = normed_values[condition]
                cross = weights @ torch.sum(baseline * current, dim=1)
                centered_cross = cross - torch.sum(
                    normed_stats["baseline"]["means"] * normed_stats[condition]["means"], dim=1
                )
                gamma_draws = centered_cross / normed_stats["baseline"]["total"]
                corr_draws = centered_cross / torch.sqrt(
                    normed_stats["baseline"]["total"] * normed_stats[condition]["total"]
                )
                for target, values in ((gamma, gamma_draws), (correlation, corr_draws)):
                    array = values.float().cpu().numpy()
                    target[condition]["estimate"][layer] = array[0]
                    target[condition]["low"][layer] = np.quantile(array[1:], 0.025)
                    target[condition]["high"][layer] = np.quantile(array[1:], 0.975)

            del J, raw_stats, normed_stats, normed_values
            if layer == 0 or (layer + 1) % 8 == 0:
                print(f"processed readout {layer + 1}/64", flush=True)

    output_data.mkdir(parents=True, exist_ok=True)
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"layers": layers, "question_ids": np.asarray(qids), "prior": prior}
    for condition in ("incorrect", "neutral"):
        for metric in metric_names:
            for statistic in ("estimate", "low", "high"):
                arrays[f"{condition}_{metric}_{statistic}"] = curves[condition][metric][statistic]
        for name, values in (("full_gain", gamma), ("full_correlation", correlation)):
            for statistic in ("estimate", "low", "high"):
                arrays[f"{condition}_{name}_{statistic}"] = values[condition][statistic]
    np.savez_compressed(output_data / "full_residual_variance.npz", **arrays)

    selected = {}
    for layer in (32, 40, 48, 52, 56, 60, 64):
        index = layer - 1
        selected[str(layer)] = {}
        for condition in ("incorrect", "neutral"):
            selected[str(layer)][LABELS[condition]] = {
                metric: {
                    "estimate": float(curves[condition][metric]["estimate"][index]),
                    "ci": [
                        float(curves[condition][metric]["low"][index]),
                        float(curves[condition][metric]["high"][index]),
                    ],
                }
                for metric in metric_names
            }
            selected[str(layer)][LABELS[condition]]["full_gain"] = {
                "estimate": float(gamma[condition]["estimate"][index]),
                "ci": [
                    float(gamma[condition]["low"][index]),
                    float(gamma[condition]["high"][index]),
                ],
            }
            selected[str(layer)][LABELS[condition]]["full_correlation"] = {
                "estimate": float(correlation[condition]["estimate"][index]),
                "ci": [
                    float(correlation[condition]["low"][index]),
                    float(correlation[condition]["high"][index]),
                ],
            }
    summary = {
        "n_questions": len(qids),
        "bootstrap_draws": bootstrap,
        "answer_subspace_rank": rank,
        "centering": "Condition-specific weighted mean removed across questions at every layer.",
        "balancing": "Generated Baseline answer letters receive equal weight.",
        "normalization": "JLens transport followed by the model final RMSNorm.",
        "selected_layers": selected,
    }
    (output_data / "full_residual_variance_summary.json").write_text(json.dumps(summary, indent=2))
    _plot(output_figure, layers, curves)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--second-chance-root", type=Path, required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = analyze(
        args.config, args.baseline_root, args.second_chance_root,
        args.lens_repo, args.lens_filename, args.output_data, args.output_figure,
        args.bootstrap, args.seed,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
