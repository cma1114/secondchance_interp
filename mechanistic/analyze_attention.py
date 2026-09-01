from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

from .attention_spans import SPAN_NAMES
from .io import read_metadata, shard_path


def _common_questions(attention_dir: Path) -> list[str]:
    groups = []
    for condition in ("incorrect", "neutral"):
        groups.append({path.stem for path in (attention_dir / "shards" / condition).glob("*.npz")})
    common = sorted(set.intersection(*groups))
    if not common:
        raise FileNotFoundError("No paired Game/Neutral attention shards")
    return common


def _load_attention(attention_dir: Path, qids: list[str]):
    first = np.load(shard_path(attention_dir, "incorrect", qids[0]), allow_pickle=False)
    shape = first["attention_mass"].shape
    masses = np.empty((2, len(qids), *shape), dtype=np.float32)
    direct = np.empty((2, len(qids), shape[0], shape[1], 4), dtype=np.float32)
    final_logits = np.empty((2, len(qids), 4), dtype=np.float32)
    counts = np.empty((2, len(qids), shape[-1]), dtype=np.float32)
    metadata: list[list[dict]] = [[], []]
    for ci, condition in enumerate(("incorrect", "neutral")):
        for qi, qid in enumerate(qids):
            with np.load(shard_path(attention_dir, condition, qid), allow_pickle=False) as data:
                masses[ci, qi] = data["attention_mass"]
                direct[ci, qi] = data["head_direct_ad"]
                final_logits[ci, qi] = data["final_canonical_logits"]
                meta = read_metadata(data)
                metadata[ci].append(meta)
                counts[ci, qi] = [len(meta["span_token_positions"][name]) for name in SPAN_NAMES]
    return masses, direct, final_logits, counts, metadata


def _load_logits(mechanistic_dir: Path, qids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    logits = np.empty((3, len(qids), 4), dtype=np.float64)
    winner = np.empty(len(qids), dtype=np.int64)
    for ci, condition in enumerate(("baseline", "incorrect", "neutral")):
        for qi, qid in enumerate(qids):
            with np.load(shard_path(mechanistic_dir, condition, qid), allow_pickle=False) as data:
                logits[ci, qi] = data["canonical_logits"][-1]
                if ci == 0:
                    winner[qi] = int(np.argmax(logits[ci, qi]))
    logits -= logits.mean(axis=-1, keepdims=True)
    return logits, winner


def _compression(target: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    denominator = np.sum(baseline * baseline, axis=-1)
    return -np.sum((target - baseline) * baseline, axis=-1) / np.maximum(denominator, 1e-12)


def _mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _winner_advantage(values: np.ndarray, winners: np.ndarray) -> np.ndarray:
    selected = np.take_along_axis(values, winners[:, None, None, None], axis=-1)[..., 0]
    return selected - (values.sum(axis=-1) - selected) / 3.0


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    flat = np.asarray(p_values).ravel()
    order = np.argsort(flat)
    ranked = flat[order] * len(flat) / np.arange(1, len(flat) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted.reshape(np.asarray(p_values).shape)


def _correlations(values: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rho = np.empty(values.shape[1:], dtype=np.float64)
    p = np.empty_like(rho)
    for layer in range(values.shape[1]):
        for head in range(values.shape[2]):
            rho[layer, head], p[layer, head] = spearmanr(values[:, layer, head], target)
    return rho, p


def _partial_rank_correlations(
    values: np.ndarray, target: np.ndarray, nuisance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(target)), nuisance])
    target_rank = rankdata(target)
    target_residual = target_rank - design @ np.linalg.lstsq(design, target_rank, rcond=None)[0]
    correlation = np.empty(values.shape[1:], dtype=np.float64)
    p_value = np.empty_like(correlation)
    for layer in range(values.shape[1]):
        for head in range(values.shape[2]):
            ranked = rankdata(values[:, layer, head])
            residual = ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]
            correlation[layer, head], p_value[layer, head] = pearsonr(residual, target_residual)
    return correlation, p_value


def _heatmaps(path: Path, layers: list[int], game_feedback, neutral_feedback, game_keyword, direct_diff):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    shared_feedback_max = np.nanmax(
        np.concatenate([game_feedback.ravel(), neutral_feedback.ravel()])
    )
    panels = (
        (game_feedback, "Game: attention mass to feedback sentence", "viridis", "feedback"),
        (neutral_feedback, "Neutral: attention mass to feedback sentence", "viridis", "feedback"),
        (game_keyword, "Game: mean attention per ‘incorrect’ occurrence", "viridis", None),
        (direct_diff, "Game − Neutral: direct write to baseline-winner advantage", "coolwarm", "symmetric"),
    )
    for axis, (values, title, cmap, scaling) in zip(axes.flat, panels):
        kwargs = {}
        if scaling == "feedback":
            kwargs = {"vmin": 0, "vmax": shared_feedback_max}
        if scaling == "symmetric":
            bound = np.nanpercentile(np.abs(values), 98)
            kwargs = {"vmin": -bound, "vmax": bound}
        image = axis.imshow(values, aspect="auto", origin="lower", cmap=cmap, **kwargs)
        axis.set_title(title)
        axis.set_xlabel("Attention head")
        axis.set_ylabel("Transformer block (0-indexed)")
        axis.set_yticks(np.arange(len(layers)), labels=layers)
        fig.colorbar(image, ax=axis, shrink=.82)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def _top_tokens_for_head(
    attention_dir: Path, qids: list[str], condition: str, layer_index: int, head: int
) -> list[dict]:
    totals: dict[str, float] = {}
    top_one: dict[str, int] = {}
    for qid in qids:
        with np.load(shard_path(attention_dir, condition, qid), allow_pickle=False) as data:
            meta = read_metadata(data)
            positions = data["top_attention_position"][layer_index, head].astype(int)
            weights = data["top_attention_weight"][layer_index, head].astype(float)
            tokens = meta["tokens"]
            for position, weight in zip(positions, weights):
                token = tokens[position]
                totals[token] = totals.get(token, 0.0) + float(weight)
            first = tokens[positions[0]]
            top_one[first] = top_one.get(first, 0) + 1
    ordered = sorted(totals, key=totals.get, reverse=True)[:15]
    return [
        {
            "token": token,
            "mean_attention_weight_within_top8": totals[token] / len(qids),
            "fraction_trials_top_attended_token": top_one.get(token, 0) / len(qids),
        }
        for token in ordered
    ]


def _feedback_keyword_top8(
    attention_dir: Path, qids: list[str], condition: str, layer_index: int, head: int
) -> dict:
    values = []
    covered = []
    for qid in qids:
        with np.load(shard_path(attention_dir, condition, qid), allow_pickle=False) as data:
            meta = read_metadata(data)
            targets = set(meta["span_token_positions"]["condition_keyword"]) & set(
                meta["span_token_positions"]["feedback_sentence"]
            )
            positions = data["top_attention_position"][layer_index, head].astype(int)
            weights = data["top_attention_weight"][layer_index, head].astype(float)
            lookup = dict(zip(positions, weights))
            values.append(sum(lookup.get(position, 0.0) for position in targets))
            covered.append(bool(targets & set(positions)))
    mean, low, high = (float(value) for value in _mean_ci(np.asarray(values)))
    return {
        "mean_top8_lower_bound": mean,
        "ci_low": low,
        "ci_high": high,
        "fraction_trials_keyword_in_top8": float(np.mean(covered)),
    }


def analyze(attention_dir: Path, mechanistic_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    qids = _common_questions(attention_dir)
    masses, direct, attention_final_logits, counts, metadata = _load_attention(attention_dir, qids)
    logits, winners = _load_logits(mechanistic_dir, qids)
    run_meta = json.loads((attention_dir / "run_metadata.json").read_text())
    layers = run_meta["full_attention_layers"]
    spans = {name: index for index, name in enumerate(SPAN_NAMES)}

    feedback = masses[..., spans["feedback_sentence"]]
    keyword = masses[..., spans["condition_keyword"]]
    action = masses[..., spans["action_keyword"]]
    keyword_per_token = keyword / np.maximum(counts[..., spans["condition_keyword"]][:, :, None, None], 1)
    action_per_token = action / np.maximum(counts[..., spans["action_keyword"]][:, :, None, None], 1)
    feedback_per_token = feedback / np.maximum(counts[..., spans["feedback_sentence"]][:, :, None, None], 1)
    game_compression = _compression(logits[1], logits[0])
    neutral_compression = _compression(logits[2], logits[0])
    strategic_compression = game_compression - neutral_compression

    baseline_probability = np.exp(logits[0] - logits[0].max(axis=-1, keepdims=True))
    baseline_probability /= baseline_probability.sum(axis=-1, keepdims=True)
    sorted_baseline = np.sort(logits[0], axis=-1)
    prompt_lengths = np.asarray([
        (metadata[0][index]["prompt_length"] + metadata[1][index]["prompt_length"]) / 2
        for index in range(len(qids))
    ])
    nuisance = np.column_stack([
        prompt_lengths,
        sorted_baseline[:, -1] - sorted_baseline[:, -2],
        logits[0].std(axis=-1),
        -np.sum(baseline_probability * np.log(np.maximum(baseline_probability, 1e-12)), axis=-1),
        np.asarray([metadata[0][index]["baseline_correct"] for index in range(len(qids))], dtype=float),
        np.eye(4)[winners, :3],
    ])
    nuisance[:, :4] = (nuisance[:, :4] - nuisance[:, :4].mean(axis=0)) / np.maximum(
        nuisance[:, :4].std(axis=0), 1e-12
    )

    direct_advantage = np.stack([
        _winner_advantage(direct[condition], winners) for condition in range(2)
    ])
    direct_difference = direct_advantage[0] - direct_advantage[1]

    rho, p = _correlations(keyword_per_token[0], strategic_compression)
    partial_r, partial_p = _partial_rank_correlations(
        keyword_per_token[0], strategic_compression, nuisance
    )
    partial_q = _bh_fdr(partial_p)
    game_feedback_mean, _, _ = _mean_ci(feedback[0])
    neutral_feedback_mean, _, _ = _mean_ci(feedback[1])
    game_keyword_mean, _, _ = _mean_ci(keyword_per_token[0])
    direct_diff_mean, _, _ = _mean_ci(direct_difference)

    _heatmaps(
        output_dir / "feedback_attention_heatmaps",
        layers,
        game_feedback_mean,
        neutral_feedback_mean,
        game_keyword_mean,
        direct_diff_mean,
    )

    rows = []
    for li, layer in enumerate(layers):
        for head in range(masses.shape[3]):
            values = keyword_per_token[0, :, li, head]
            mean, low, high = (x.item() for x in _mean_ci(values))
            rows.append({
                "layer": layer,
                "head": head,
                "game_incorrect_attention_per_token": mean,
                "game_incorrect_attention_ci_low": low,
                "game_incorrect_attention_ci_high": high,
                "neutral_lost_transmission_attention_per_token": float(keyword_per_token[1, :, li, head].mean()),
                "game_minus_neutral_condition_keyword_attention_per_token": float(
                    (keyword_per_token[0, :, li, head] - keyword_per_token[1, :, li, head]).mean()
                ),
                "game_different_answer_attention_per_token": float(action_per_token[0, :, li, head].mean()),
                "neutral_again_attention_per_token": float(action_per_token[1, :, li, head].mean()),
                "game_feedback_sentence_attention": float(game_feedback_mean[li, head]),
                "neutral_feedback_sentence_attention": float(neutral_feedback_mean[li, head]),
                "game_minus_neutral_direct_winner_write": float(direct_diff_mean[li, head]),
                "rho_attention_with_strategic_compression": float(rho[li, head]),
                "correlation_p": float(p[li, head]),
                "partial_r_attention_with_strategic_compression": float(partial_r[li, head]),
                "partial_correlation_p": float(partial_p[li, head]),
                "partial_correlation_fdr_q": float(partial_q[li, head]),
            })
    rows.sort(key=lambda row: row["game_incorrect_attention_per_token"], reverse=True)
    import csv
    with (output_dir / "attention_heads.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    strongest_correlation = max(
        rows, key=lambda row: abs(row["partial_r_attention_with_strategic_compression"])
    )
    top_token_profiles = []
    for row in rows[:8]:
        layer_index = layers.index(row["layer"])
        top_token_profiles.append({
            "layer": row["layer"],
            "head": row["head"],
            "game_top_tokens": _top_tokens_for_head(
                attention_dir, qids, "incorrect", layer_index, row["head"]
            ),
            "game_user_incorrect_top8": _feedback_keyword_top8(
                attention_dir, qids, "incorrect", layer_index, row["head"]
            ),
            "neutral_top_tokens": _top_tokens_for_head(
                attention_dir, qids, "neutral", layer_index, row["head"]
            ),
            "neutral_feedback_keywords_top8": _feedback_keyword_top8(
                attention_dir, qids, "neutral", layer_index, row["head"]
            ),
        })
    summary = {
        "n_questions": len(qids),
        "full_attention_layers": layers,
        "n_heads": masses.shape[3],
        "game_compression_mean": float(game_compression.mean()),
        "neutral_compression_mean": float(neutral_compression.mean()),
        "strategic_compression_mean": float(strategic_compression.mean()),
        "final_choice_agreement_with_activation_run": {
            "game": float(np.mean(np.argmax(attention_final_logits[0], axis=-1) == np.argmax(logits[1], axis=-1))),
            "neutral": float(np.mean(np.argmax(attention_final_logits[1], axis=-1) == np.argmax(logits[2], axis=-1))),
        },
        "attention_run_switch_rates_against_existing_baseline": {
            "game": float(np.mean(np.argmax(attention_final_logits[0], axis=-1) != winners)),
            "neutral": float(np.mean(np.argmax(attention_final_logits[1], axis=-1) != winners)),
            "game_minus_neutral": float(
                np.mean(np.argmax(attention_final_logits[0], axis=-1) != winners)
                - np.mean(np.argmax(attention_final_logits[1], axis=-1) != winners)
            ),
        },
        "top_incorrect_attention_heads": rows[:20],
        "top_token_profiles": top_token_profiles,
        "strongest_attention_compression_correlation": strongest_correlation,
        "n_partial_correlations_fdr_below_05": int(np.sum(partial_q < .05)),
        "limitations": [
            "Token-level attention exists only in the model's 16 full-attention blocks; the other 48 blocks use gated linear attention.",
            "Attention weight is not causal evidence.",
            "The direct A-D write is an unnormalized direct-logit attribution of the full head output, not an isolated feedback-token value contribution.",
        ],
    }
    (output_dir / "attention_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze final-position attention to feedback tokens")
    parser.add_argument("--attention-dir", required=True)
    parser.add_argument("--mechanistic-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = analyze(Path(args.attention_dir), Path(args.mechanistic_dir), Path(args.output_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
