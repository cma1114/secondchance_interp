#!/usr/bin/env python3
"""Cluster JLens vocabulary readouts into semantic families.

This is a compact, local analysis of the saved top/bottom vocabulary lists from
the complete-residual GLA boundary-lens experiment.  It does not run the model.

The clustering is deliberately used only to consolidate synonymous and
morphological token variants.  Signed scores remain block-specific, and a
family's score is the single most extreme member token rather than a sum over
variants.  This avoids inflating concepts that happen to have many tokenizer
spellings in the displayed vocabulary tail.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModel, AutoTokenizer


VIEW = "GLA contextual change: Evaluation minus Neutral"
LENS = "J-lens"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K_VALUES = (6, 12, 24)

# Anchors name the human-readable, task-relevant families.  Each family is the
# union of the unsupervised clusters containing its anchors.  The complete
# unsupervised cluster catalog is saved separately, so these labels are auditable.
FAMILY_ANCHORS = {
    "Answer / response": ("answer", "response", "respond", "reply"),
    "Incorrect / wrong": ("incorrect", "wrong"),
    "Failure / error": ("failure", "failed", "error", "unsuccessful"),
    "Replace / substitute": ("replace", "replacement", "substitute"),
    "Alternative / instead": ("alternative", "alternatives", "instead"),
    "Again / retry": ("again", "retry", "retries"),
    "Previous / last": ("previous", "last"),
    "Another": ("another", "second"),
    "New / update": ("new", "updated", "revised"),
    "Same": ("same", "remember"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--distance-threshold", type=float, default=0.25)
    parser.add_argument("--display-first-block", type=int, default=33)
    parser.add_argument("--display-last-block", type=int, default=55)
    return parser.parse_args()


def normalize_token(raw: str) -> str | None:
    """Return a conservative alphabetic token rendering or None."""
    token = raw.strip().strip("._-()[]{}")
    if not re.fullmatch(r"[A-Za-z][A-Za-z'-]{2,}", token):
        return None
    # Interior capitals usually indicate code identifiers rather than words.
    if any(char.isupper() for char in token[1:]):
        return None
    if not re.search(r"[aeiouyAEIOUY]", token):
        return None
    return token.lower()


def load_entries(readouts: Path) -> tuple[dict, list[dict]]:
    data = json.loads(readouts.read_text())
    entries: list[dict] = []
    for block_string, layer in data["layers"].items():
        block = int(block_string)
        for side in ("positive", "negative"):
            for rank, item in enumerate(layer[LENS][VIEW][side], start=1):
                normalized = normalize_token(item["token"])
                if normalized is None:
                    continue
                entries.append(
                    {
                        "block": block,
                        "side": side,
                        "rank": rank,
                        "raw_token": item["token"],
                        "token": normalized,
                        "score": float(item["score"]),
                    }
                )
    return data, entries


def embed_tokens(
    tokens: list[str], model_name: str, cache_dir: Path | None
) -> np.ndarray:
    cache = str(cache_dir) if cache_dir is not None else None
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache).eval()
    encoded = tokenizer(
        [f"word: {token}" for token in tokens],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()


def fit_clusters(
    tokens: list[str], embeddings: np.ndarray, threshold: float
) -> tuple[dict[str, int], dict[int, list[str]]]:
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    ).fit_predict(embeddings)
    token_to_cluster = dict(zip(tokens, labels.tolist(), strict=True))
    clusters: dict[int, list[str]] = defaultdict(list)
    for token, label in token_to_cluster.items():
        clusters[label].append(token)
    return token_to_cluster, dict(clusters)


def build_families(
    token_to_cluster: dict[str, int], clusters: dict[int, list[str]]
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    family_tokens: dict[str, set[str]] = {}
    anchors_found: dict[str, list[str]] = {}
    for family, anchors in FAMILY_ANCHORS.items():
        found = [anchor for anchor in anchors if anchor in token_to_cluster]
        cluster_ids = {token_to_cluster[anchor] for anchor in found}
        family_tokens[family] = {
            token for cluster_id in cluster_ids for token in clusters[cluster_id]
        }
        anchors_found[family] = found
    return family_tokens, anchors_found


def score_families(
    entries: list[dict], family_tokens: dict[str, set[str]]
) -> pd.DataFrame:
    rows: list[dict] = []
    blocks = sorted({entry["block"] for entry in entries})
    for k in K_VALUES:
        for block in blocks:
            block_entries = [
                entry
                for entry in entries
                if entry["block"] == block and entry["rank"] <= k
            ]
            for family, tokens in family_tokens.items():
                matches = [entry for entry in block_entries if entry["token"] in tokens]
                if matches:
                    winner = max(matches, key=lambda entry: abs(entry["score"]))
                    rows.append(
                        {
                            "k": k,
                            "block": block,
                            "family": family,
                            "score": winner["score"],
                            "representative_token": winner["raw_token"],
                            "rank": winner["rank"],
                            "present": True,
                        }
                    )
                else:
                    rows.append(
                        {
                            "k": k,
                            "block": block,
                            "family": family,
                            "score": np.nan,
                            "representative_token": "",
                            "rank": np.nan,
                            "present": False,
                        }
                    )
    return pd.DataFrame(rows)


def save_cluster_catalog(
    entries: list[dict], clusters: dict[int, list[str]], output: Path
) -> pd.DataFrame:
    token_stats: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        token_stats[entry["token"]].append(entry)
    rows = []
    for cluster_id, members in clusters.items():
        cluster_entries = [entry for member in members for entry in token_stats[member]]
        peak = max(cluster_entries, key=lambda entry: abs(entry["score"]))
        rows.append(
            {
                "cluster_id": cluster_id,
                "members": "; ".join(sorted(members)),
                "member_count": len(members),
                "blocks_present": len({entry["block"] for entry in cluster_entries}),
                "peak_score": peak["score"],
                "peak_block": peak["block"],
                "peak_token": peak["raw_token"],
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        "peak_score", key=lambda series: series.abs(), ascending=False
    )
    frame.to_csv(output, index=False)
    return frame


def plot_heatmaps(
    scores: pd.DataFrame,
    output: Path,
    first_block: int,
    last_block: int,
) -> None:
    families = list(FAMILY_ANCHORS)
    blocks = list(range(first_block, last_block + 1))
    matrices = []
    for k in K_VALUES:
        subset = scores[scores["k"] == k]
        pivot = subset.pivot(index="family", columns="block", values="score")
        matrices.append(pivot.reindex(index=families, columns=blocks).to_numpy())
    vmax = max(float(np.nanmax(np.abs(matrix))) for matrix in matrices)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e8e8e8")

    fig, axes = plt.subplots(1, 3, figsize=(17, 7.8), constrained_layout=True)
    for index, (axis, k, matrix) in enumerate(zip(axes, K_VALUES, matrices, strict=True)):
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        axis.set_title(f"Top/bottom {k} tokens", fontsize=15, pad=12)
        axis.set_xticks(range(0, len(blocks), 2), [blocks[i] for i in range(0, len(blocks), 2)])
        axis.set_xlabel("GLA block")
        axis.set_yticks(range(len(families)))
        axis.set_yticklabels(families if index == 0 else [])
        axis.tick_params(axis="both", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xticks(np.arange(-0.5, len(blocks), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(families), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.7)
        axis.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=axes, location="bottom", shrink=0.54, pad=0.08)
    colorbar.set_label(
        "Signed JLens score: red = added more in Evaluation; blue = removed more",
        fontsize=11,
    )
    fig.suptitle(
        "Semantic content of each GLA's Evaluation-minus-Neutral contextual change",
        fontsize=18,
        y=1.08,
    )
    fig.text(
        0.5,
        -0.01,
        "Gray means no member of the semantic family appeared within that top/bottom-k list. "
        "Each cell uses one strongest member token, never a sum over synonyms.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, entries = load_entries(args.readouts)
    tokens = sorted({entry["token"] for entry in entries})
    embeddings = embed_tokens(tokens, args.embedding_model, args.cache_dir)
    token_to_cluster, clusters = fit_clusters(tokens, embeddings, args.distance_threshold)
    family_tokens, anchors_found = build_families(token_to_cluster, clusters)
    scores = score_families(entries, family_tokens)
    scores.to_csv(args.output_dir / "block_family_scores.csv", index=False)
    catalog = save_cluster_catalog(entries, clusters, args.output_dir / "cluster_catalog.csv")

    sensitivity_frames = []
    for threshold in (0.20, 0.25, 0.30):
        threshold_map, threshold_clusters = fit_clusters(tokens, embeddings, threshold)
        threshold_families, _ = build_families(threshold_map, threshold_clusters)
        threshold_scores = score_families(entries, threshold_families)
        threshold_scores.insert(0, "distance_threshold", threshold)
        sensitivity_frames.append(threshold_scores)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    sensitivity.to_csv(args.output_dir / "threshold_sensitivity.csv", index=False)
    plot_heatmaps(
        scores,
        args.figure,
        args.display_first_block,
        args.display_last_block,
    )

    key_blocks = [39, 42, 43, 47, 49, 50, 51, 53, 55]
    key = scores[(scores["k"] == 24) & scores["block"].isin(key_blocks)]
    summary = {
        "definition": (
            "JLens top/bottom vocabulary tokens for each GLA contextual change: "
            "(Evaluation after-before) - (Neutral after-before)."
        ),
        "embedding_model": args.embedding_model,
        "clustering": {
            "method": "agglomerative average-linkage cosine clustering",
            "distance_threshold": args.distance_threshold,
            "tokens_clustered": len(tokens),
            "clusters": len(clusters),
            "candidate_pool": "union of saved top/bottom 24 tokens over all GLA blocks",
        },
        "family_scoring": (
            "At each block and cutoff, retain the signed score of the single "
            "most extreme token belonging to the family. Missing is not zero."
        ),
        "anchors_found": anchors_found,
        "family_members": {key: sorted(value) for key, value in family_tokens.items()},
        "key_block_scores_top24": key.replace({np.nan: None}).to_dict(orient="records"),
        "catalog_rows": len(catalog),
        "threshold_sensitivity": [0.20, 0.25, 0.30],
        "figure": str(args.figure),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
