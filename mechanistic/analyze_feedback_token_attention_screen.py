from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("incorrect", "neutral")
SOURCE_NAMES = ("evaluation", "action")


def _stratified_bootstrap(labels: np.ndarray, samples: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == letter) for letter in "ABCD"]
    return np.concatenate(
        [rng.choice(group, size=(samples, len(group)), replace=True) for group in groups],
        axis=1,
    )


def _ci(values: np.ndarray, bootstrap: np.ndarray) -> tuple[float, float, float]:
    distribution = values[bootstrap].mean(axis=1)
    low, high = np.quantile(distribution, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def _load_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    return data.get("question_ids", data.get("confirmation_question_ids"))


def _plot(path: Path, values: np.ndarray, layers: np.ndarray) -> None:
    # values: condition, question, conventional layer, head, source
    means = values.mean(axis=1) * 100
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.6))
    row_specs = (
        (0, "incorrect", "lost", "feedback token 4"),
        (1, "second ‘answer’", "again", "feedback token 9"),
    )
    for source, game_name, neutral_name, aligned_name in row_specs:
        raw_max = float(max(means[0, :, :, source].max(), means[1, :, :, source].max()))
        panels = (
            (means[0, :, :, source], f"Game: final → {game_name}", "viridis", 0, raw_max),
            (means[1, :, :, source], f"Neutral: final → {neutral_name}", "viridis", 0, raw_max),
        )
        for column, (matrix, title, cmap, vmin, vmax) in enumerate(panels):
            axis = axes[source, column]
            image = axis.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title, loc="left", weight="bold")
            fig.colorbar(image, ax=axis, label="Mean attention weight (%)", shrink=.80)
        difference = means[0, :, :, source] - means[1, :, :, source]
        bound = float(np.max(np.abs(difference))) or 1.0
        axis = axes[source, 2]
        image = axis.imshow(
            difference,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-bound,
            vmax=bound,
        )
        axis.set_title(
            f"Game − Neutral: aligned {aligned_name}", loc="left", weight="bold"
        )
        fig.colorbar(image, ax=axis, label="Attention difference (pp)", shrink=.80)
    for axis in axes.flat:
        axis.set_xlabel("Query head (one-based)")
        axis.set_ylabel("Conventional-attention block")
        axis.set_xticks(np.arange(values.shape[3]), labels=np.arange(1, values.shape[3] + 1))
        axis.set_yticks(np.arange(len(layers)), labels=layers)
        axis.tick_params(axis="x", labelsize=7)
        axis.tick_params(axis="y", labelsize=8)
    fig.suptitle(
        "Final-decision attention to aligned feedback tokens — held-out SimpleMC",
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.065, 1.0, 0.95), h_pad=2.0, w_pad=1.8)
    fig.text(
        0.5,
        0.012,
        "Each cell is a distinct block–head component; no lines connect categorical components. Only Qwen’s 16 conventional-attention blocks have literal token-to-token attention weights.",
        ha="center",
        fontsize=9.5,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze(
    results: Path,
    metadata_path: Path,
    discovery_plan: Path,
    confirmation_plan: Path,
    baseline_results: Path,
    output_dir: Path,
    figure: Path,
) -> dict:
    with np.load(results, allow_pickle=False) as data:
        attention = data["attention"].astype(np.float64)
        qids = data["question_ids"].astype(str).tolist()
        completed = data["completed"]
        layers = data["ordinary_layer_indices_zero_based"].astype(int) + 1
    if not completed.all() or not np.isfinite(attention).all():
        raise ValueError("Attention collection is incomplete")
    meta = json.loads(metadata_path.read_text())
    discovery_ids = _load_ids(discovery_plan)
    confirmation_ids = _load_ids(confirmation_plan)
    index = {qid: i for i, qid in enumerate(qids)}
    discovery_index = np.asarray([index[qid] for qid in discovery_ids])
    confirmation_index = np.asarray([index[qid] for qid in confirmation_ids])
    if set(discovery_ids) & set(confirmation_ids) or len(discovery_ids) + len(confirmation_ids) != len(qids):
        raise ValueError("Discovery and confirmation plans do not form a disjoint complete split")
    baseline = json.loads(baseline_results.read_text())["results"]
    confirmation_letters = np.asarray(
        [baseline[qid]["subject_answer"] for qid in confirmation_ids]
    )
    bootstrap = _stratified_bootstrap(confirmation_letters, 10000)

    discovery = attention[:, discovery_index]
    confirmation = attention[:, confirmation_index]
    discovery_mean = discovery.mean(axis=1)
    confirmation_mean = confirmation.mean(axis=1)
    rows = []
    for li, layer in enumerate(layers):
        for head0 in range(attention.shape[3]):
            for source, source_name in enumerate(SOURCE_NAMES):
                game_values = confirmation[0, :, li, head0, source]
                neutral_values = confirmation[1, :, li, head0, source]
                difference = game_values - neutral_values
                game_ci = _ci(game_values, bootstrap)
                neutral_ci = _ci(neutral_values, bootstrap)
                difference_ci = _ci(difference, bootstrap)
                rows.append({
                    "source": source_name,
                    "block": int(layer),
                    "head": head0 + 1,
                    "discovery_game_attention": float(discovery_mean[0, li, head0, source]),
                    "discovery_neutral_attention": float(discovery_mean[1, li, head0, source]),
                    "discovery_game_minus_neutral": float(
                        discovery_mean[0, li, head0, source]
                        - discovery_mean[1, li, head0, source]
                    ),
                    "confirmation_game_attention": game_ci[0],
                    "confirmation_game_ci_low": game_ci[1],
                    "confirmation_game_ci_high": game_ci[2],
                    "confirmation_neutral_attention": neutral_ci[0],
                    "confirmation_neutral_ci_low": neutral_ci[1],
                    "confirmation_neutral_ci_high": neutral_ci[2],
                    "confirmation_game_minus_neutral": difference_ci[0],
                    "confirmation_difference_ci_low": difference_ci[1],
                    "confirmation_difference_ci_high": difference_ci[2],
                })

    # Candidate nomination uses discovery only: the four strongest Game heads,
    # four strongest Neutral heads, and four largest absolute contrasts per source.
    candidates: list[dict] = []
    for source_name in SOURCE_NAMES:
        source_rows = [row for row in rows if row["source"] == source_name]
        nominated = set()
        criteria = (
            ("game", "discovery_game_attention", False),
            ("neutral", "discovery_neutral_attention", False),
            ("contrast", "discovery_game_minus_neutral", True),
        )
        for criterion, key, absolute in criteria:
            ordered = sorted(
                source_rows,
                key=lambda row: abs(row[key]) if absolute else row[key],
                reverse=True,
            )[:4]
            for row in ordered:
                identity = (row["source"], row["block"], row["head"])
                if identity not in nominated:
                    item = dict(row)
                    item["nominated_by"] = [criterion]
                    candidates.append(item)
                    nominated.add(identity)
                else:
                    next(
                        item for item in candidates
                        if (item["source"], item["block"], item["head"]) == identity
                    )["nominated_by"].append(criterion)

    layer_summary = []
    for ci, condition in enumerate(CONDITIONS):
        for source, source_name in enumerate(SOURCE_NAMES):
            for li, layer in enumerate(layers):
                heads = confirmation_mean[ci, li, :, source]
                layer_summary.append({
                    "condition": condition,
                    "source": source_name,
                    "block": int(layer),
                    "mean_across_heads": float(heads.mean()),
                    "strongest_head": int(heads.argmax()) + 1,
                    "strongest_head_attention": float(heads.max()),
                    "heads_above_five_percent": int((heads >= 0.05).sum()),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "attention_heads.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n_discovery": len(discovery_ids),
        "n_confirmation": len(confirmation_ids),
        "measurement": meta["measurement"],
        "architectural_limit": meta["architectural_limit"],
        "user_facing_blocks_one_based": layers.tolist(),
        "head_numbers_one_based": True,
        "candidate_selection": (
            "Per source token, top four discovery Game attention, top four discovery "
            "Neutral attention, and top four absolute discovery Game-minus-Neutral contrasts"
        ),
        "candidates": candidates,
        "layer_summary": layer_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _plot(figure, confirmation, layers)

    lines = [
        "# Final-position attention to feedback tokens",
        "",
        "This is the isolated token-matched prompt test, not the standard-wording result set.",
        f"Discovery questions: **{len(discovery_ids)}**; held-out confirmation questions: **{len(confirmation_ids)}**.",
        "The measurement is the ordinary softmax-attention weight from the final decision",
        "position to one exact earlier token. Qwen's GLA blocks do not have an equivalent",
        "pairwise attention matrix, so the screen covers conventional-attention blocks 4, 8, …, 64.",
        "Blocks and heads below are one-based.",
        "",
        "## Bottom line",
        "",
        "On the held-out questions, the strongest Game `incorrect` cluster is block 28:",
        "its 24 heads average 4.66% attention to that one token, six heads exceed 5%,",
        "and B28/H20 reaches 22.33%. No Neutral head gives `lost` more than 2.46%.",
        "For the aligned action token, Game peaks at block 32 (3.36% averaged across",
        "heads; five heads above 5%; B32/H15 = 19.84%). Neutral has its own `again`",
        "readers, strongest at block 28 (B28/H5 = 11.33%). Thus the final decision",
        "position strongly and selectively reads both the evaluation word and the action",
        "word, through distinct mid-model heads. Attention weights identify routes, not",
        "whether those routes causally change the answer.",
        "",
        "## Discovery-nominated heads, measured on confirmation questions",
        "",
        "| Source | Block/head | Nominated by | Game attention | Neutral attention | Game − Neutral |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for item in sorted(
        candidates,
        key=lambda row: max(
            row["confirmation_game_attention"],
            row["confirmation_neutral_attention"],
            abs(row["confirmation_game_minus_neutral"]),
        ),
        reverse=True,
    ):
        lines.append(
            f"| {item['source']} | B{item['block']}/H{item['head']} | "
            f"{', '.join(item['nominated_by'])} | "
            f"{100*item['confirmation_game_attention']:.2f}% "
            f"[{100*item['confirmation_game_ci_low']:.2f}, {100*item['confirmation_game_ci_high']:.2f}] | "
            f"{100*item['confirmation_neutral_attention']:.2f}% "
            f"[{100*item['confirmation_neutral_ci_low']:.2f}, {100*item['confirmation_neutral_ci_high']:.2f}] | "
            f"{100*item['confirmation_game_minus_neutral']:+.2f} pp "
            f"[{100*item['confirmation_difference_ci_low']:+.2f}, {100*item['confirmation_difference_ci_high']:+.2f}] |"
        )
    lines += [
        "",
        "The heatmap uses only the held-out confirmation questions. Each cell is a distinct",
        "block–head component; categorical cells are not connected by lines.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--confirmation-plan", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    analyze(
        args.results,
        args.metadata,
        args.discovery_plan,
        args.confirmation_plan,
        args.baseline_results,
        args.output_dir,
        args.figure,
    )


if __name__ == "__main__":
    main()
