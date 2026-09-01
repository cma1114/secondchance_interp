from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .io import read_metadata, shard_path


def _contrast(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    rows = np.arange(len(values))
    selected = values[rows, target]
    return selected - (values.sum(axis=-1) - selected) / 3.0


def _bootstrap(values: np.ndarray, draws: int, seed: int) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(draws)
    for index in range(draws):
        means[index] = values[rng.integers(0, n, n)].mean()
    return float(values.mean()), np.quantile(means, [0.025, 0.975]).tolist()


def _load_shards(root: Path, condition: str, qids: list[str]):
    rows = []
    for qid in qids:
        with np.load(shard_path(root, condition, qid), allow_pickle=False) as source:
            rows.append(
                {
                    "mass": source["historical_attention_mass"].astype(np.float64),
                    "token_attention": source["token_attention"].astype(np.float64),
                    "route": source["historical_route_direct_ad"].astype(np.float64),
                    "gate": source["query_output_gate"].astype(np.float64),
                    "historical_positions": source["historical_positions"].astype(int),
                    "historical_end": int(source["historical_end_position"]),
                    "metadata": read_metadata(source),
                }
            )
    return rows


def _plot(summary: dict, head_rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    token_labels = summary["historical_tokens_english"]
    game = np.asarray(summary["token_attention"]["game_mean_percent"])
    neutral = np.asarray(summary["token_attention"]["neutral_mean_percent"])
    heads = np.arange(len(head_rows))
    span = np.asarray([row["game_span_attention"] for row in head_rows]) * 100
    endpoint = np.asarray([row["game_endpoint_attention"] for row in head_rows]) * 100
    percentile = np.asarray([row["winner_weighted_value_percentile"] for row in head_rows]) * 100
    write = np.asarray([row["game_value_path_winner_contrast"] for row in head_rows])

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    x = np.arange(len(token_labels))
    width = 0.38
    axes[0, 0].bar(x - width / 2, game, width, color="#3595F6", label="Game")
    axes[0, 0].bar(x + width / 2, neutral, width, color="#F07F31", label="Neutral")
    axes[0, 0].set_xticks(x, token_labels, rotation=45, ha="right", fontsize=8)
    axes[0, 0].set_ylabel("Mean attention per head (%)")
    axes[0, 0].set_title("A  Where attention within the historical bucket goes", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].scatter(span, endpoint, c=heads, cmap="viridis", s=45)
    for head in np.argsort(-span)[:6]:
        axes[0, 1].annotate(f"H{head}", (span[head], endpoint[head]), fontsize=8)
    axes[0, 1].set_xlabel("Whole historical-bucket attention (%)")
    axes[0, 1].set_ylabel("Answer-endpoint attention (%)")
    axes[0, 1].set_title("B  High span attention is not endpoint attention", loc="left", fontweight="bold")

    axes[1, 0].scatter(endpoint, percentile, c=heads, cmap="viridis", s=45)
    axes[1, 0].axhline(50, color="#666666", lw=0.8, ls="--")
    for head in np.argsort(-endpoint)[:6]:
        axes[1, 0].annotate(f"H{head}", (endpoint[head], percentile[head]), fontsize=8)
    axes[1, 0].set_xlabel("Answer-endpoint attention (%)")
    axes[1, 0].set_ylabel("JLens direction value-survival percentile")
    axes[1, 0].set_title("C  Do endpoint-attending heads preserve the direction?", loc="left", fontweight="bold")

    colors = np.where(write >= 0, "#009E73", "#CC79A7")
    axes[1, 1].bar(heads, write, color=colors)
    axes[1, 1].axhline(0, color="#666666", lw=0.8)
    axes[1, 1].set_xlabel("Mixer 56 head")
    axes[1, 1].set_ylabel("Immediate winner contrast from erase update")
    axes[1, 1].set_title("D  Value-path answer write is tiny", loc="left", fontweight="bold")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.55)
        axis.set_axisbelow(True)
    figure.tight_layout(h_pad=2.2, w_pad=2.0)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"mixer56_answer_read.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def analyze(
    diagnostic_root: Path,
    intervention_root: Path,
    output: Path,
    bootstrap: int,
    seed: int,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    qids = sorted(
        path.stem for path in (diagnostic_root / "shards" / "incorrect").glob("*.npz")
        if shard_path(diagnostic_root, "neutral", path.stem).exists()
        and shard_path(intervention_root, "baseline_natural", path.stem).exists()
    )
    if not qids:
        raise FileNotFoundError("No complete diagnostic/intervention questions")
    game_rows = _load_shards(diagnostic_root, "incorrect", qids)
    neutral_rows = _load_shards(diagnostic_root, "neutral", qids)
    with np.load(diagnostic_root / "weight_diagnostic.npz", allow_pickle=False) as source:
        weights = {name: source[name].copy() for name in source.files}
    num_heads = int(weights["num_heads"])
    directions = weights["answer_directions"].astype(np.float64)
    value_per_unit = weights["value_per_unit_direction"].astype(np.float64)
    answer_through_output = weights["answer_through_output_projection"].astype(np.float64)

    baseline_logits, winner, deltas, actual = [], [], [], {"game": [], "neutral": []}
    for qid in qids:
        with np.load(shard_path(intervention_root, "baseline_natural", qid), allow_pickle=False) as source:
            values = source["final_canonical_logits"].astype(np.float64)
        baseline_logits.append(values)
        winner.append(int(values.argmax()))
        with np.load(shard_path(intervention_root, "game_erase_winner", qid), allow_pickle=False) as source:
            deltas.append(source["source_residual_delta"].astype(np.float64))
        for condition in ("game", "neutral"):
            with np.load(shard_path(intervention_root, f"{condition}_natural", qid), allow_pickle=False) as natural:
                natural_logits = natural["final_canonical_logits"].astype(np.float64)
            with np.load(shard_path(intervention_root, f"{condition}_erase_winner", qid), allow_pickle=False) as changed:
                changed_logits = changed["final_canonical_logits"].astype(np.float64)
            actual[condition].append(changed_logits - natural_logits)
    winner = np.asarray(winner, dtype=int)
    deltas = np.asarray(deltas)
    actual = {name: np.asarray(values) for name, values in actual.items()}

    historical_positions = game_rows[0]["historical_positions"]
    endpoint_relative = int(
        np.flatnonzero(historical_positions == game_rows[0]["historical_end"])[0]
    )
    decoded_tokens = game_rows[0]["metadata"]["historical_decoded_tokens"]
    english = []
    counts: dict[str, int] = {}
    for index, token in enumerate(decoded_tokens):
        base = {
            "<|im_end|>": "end marker",
            "\n": "newline",
            "<|im_start|>": "start marker",
            "assistant": "assistant",
            "<think>": "think start",
            "\n\n": "double newline",
            "</think>": "think end",
            "user": "user",
        }.get(token, repr(token))
        counts[base] = counts.get(base, 0) + 1
        english.append(f"{base} {counts[base]}" if counts[base] > 1 else base)
    english[endpoint_relative] += " (answer endpoint)"

    def stack(rows, key):
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    condition_data = {}
    for name, rows in (("game", game_rows), ("neutral", neutral_rows)):
        mass = stack(rows, "mass")
        gate = stack(rows, "gate")
        token = np.asarray(
            [row["token_attention"][:, row["historical_positions"]] for row in rows]
        )
        endpoint = token[:, :, endpoint_relative]
        route = stack(rows, "route")
        condition_data[name] = {
            "mass": mass,
            "gate": gate,
            "token": token,
            "endpoint": endpoint,
            "route": route,
        }

    # Every winner-erasure update is parallel to its corresponding unit JLens
    # answer direction. Preserve the head-dimensional value and output gate
    # between W_V and W_O rather than incorrectly treating the gate as scalar.
    coefficients = np.einsum("qx,qx->q", deltas, directions[winner])
    raw_head_value = value_per_unit[winner] * coefficients[:, None, None]
    predicted, predicted_head = {}, {}
    for name, data in condition_data.items():
        gated_value = (
            raw_head_value
            * data["gate"]
            * data["endpoint"][..., None]
        )
        per_head = np.einsum(
            "qhd,hcd->qhc", gated_value, answer_through_output, optimize=True
        )
        predicted_head[name] = per_head
        predicted[name] = per_head.sum(axis=1)

    actual_contrast = {name: _contrast(values, winner) for name, values in actual.items()}
    predicted_contrast = {name: _contrast(values, winner) for name, values in predicted.items()}
    winner_weighted_value = weights["value_random_percentile"][winner]
    winner_weighted_key = weights["key_random_percentile"][winner]

    head_rows = []
    for head in range(num_heads):
        row = {
            "head": head,
            "game_span_attention": float(condition_data["game"]["mass"][:, head].mean()),
            "neutral_span_attention": float(condition_data["neutral"]["mass"][:, head].mean()),
            "game_endpoint_attention": float(condition_data["game"]["endpoint"][:, head].mean()),
            "neutral_endpoint_attention": float(condition_data["neutral"]["endpoint"][:, head].mean()),
            "winner_weighted_value_percentile": float(winner_weighted_value[:, head].mean()),
            "winner_weighted_key_percentile": float(winner_weighted_key[:, head].mean()),
            "game_value_path_winner_contrast": float(
                _contrast(predicted_head["game"][:, head], winner).mean()
            ),
            "neutral_value_path_winner_contrast": float(
                _contrast(predicted_head["neutral"][:, head], winner).mean()
            ),
        }
        head_rows.append(row)

    endpoint_summary = {}
    for name, data in condition_data.items():
        span_mean = data["mass"].mean(axis=0)
        endpoint_mean = data["endpoint"].mean(axis=0)
        endpoint_summary[name] = {
            "mean_span_attention_per_head": float(data["mass"].mean()),
            "mean_endpoint_attention_per_head": float(data["endpoint"].mean()),
            "aggregate_endpoint_fraction_of_span_attention": float(
                data["endpoint"].sum() / data["mass"].sum()
            ),
            "highest_span_attention_heads": np.argsort(-span_mean)[:8].astype(int).tolist(),
            "highest_endpoint_attention_heads": np.argsort(-endpoint_mean)[:8].astype(int).tolist(),
        }

    causal_bridge = {}
    for name in ("game", "neutral"):
        predicted_estimate = _bootstrap(predicted_contrast[name], bootstrap, seed + (0 if name == "game" else 10))
        actual_estimate = _bootstrap(actual_contrast[name], bootstrap, seed + (1 if name == "game" else 11))
        correlation = float(np.corrcoef(predicted_contrast[name], actual_contrast[name])[0, 1])
        causal_bridge[name] = {
            "predicted_immediate_value_path_winner_contrast": {
                "mean": predicted_estimate[0], "ci": predicted_estimate[1]
            },
            "actual_final_winner_contrast": {"mean": actual_estimate[0], "ci": actual_estimate[1]},
            "trial_correlation": correlation,
        }

    summary = {
        "n": len(qids),
        "historical_tokens_raw": decoded_tokens,
        "historical_tokens_english": english,
        "answer_endpoint_relative_index": endpoint_relative,
        "attention_summary": endpoint_summary,
        "token_attention": {
            "game_mean_percent": (condition_data["game"]["token"].mean(axis=(0, 1)) * 100).tolist(),
            "neutral_mean_percent": (condition_data["neutral"]["token"].mean(axis=(0, 1)) * 100).tolist(),
        },
        "all_head_winner_weighted_projection": {
            "mean_value_survival_percentile": float(winner_weighted_value.mean()),
            "mean_key_survival_percentile": float(winner_weighted_key.mean()),
            "heads_above_90th_value_percentile": np.flatnonzero(
                winner_weighted_value.mean(axis=0) >= 0.9
            ).astype(int).tolist(),
            "heads_below_10th_value_percentile": np.flatnonzero(
                winner_weighted_value.mean(axis=0) <= 0.1
            ).astype(int).tolist(),
        },
        "causal_bridge": causal_bridge,
        "head_metrics": head_rows,
    }
    (output / "mixer56_answer_read_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    with (output / "mixer56_head_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(head_rows[0]))
        writer.writeheader()
        writer.writerows(head_rows)
    _plot(summary, head_rows, output)

    game = endpoint_summary["game"]
    neutral = endpoint_summary["neutral"]
    projection = summary["all_head_winner_weighted_projection"]
    report = f"""# What Mixer 56 reads from the historical assistant bucket

This diagnostic uses all **{len(qids)}** frozen held-out SimpleMC questions. The
source boundary is `resid_pre[56]`, the input read by Mixer 56.

## The 25% number was a bucket, not the answer endpoint

The source-partition bucket contains fourteen tokens: the first user closing
marker, the assistant header and empty-thinking scaffold, the assistant closing
marker, and the next user header. Averaged over heads, Mixer 56 assigned the
whole bucket **{100*game['mean_span_attention_per_head']:.1f}%** attention in
Game and **{100*neutral['mean_span_attention_per_head']:.1f}%** in Neutral.

The answer-decodable `\\n\\n` endpoint itself received only
**{100*game['mean_endpoint_attention_per_head']:.2f}%** in Game and
**{100*neutral['mean_endpoint_attention_per_head']:.2f}%** in Neutral. It accounts
for **{100*game['aggregate_endpoint_fraction_of_span_attention']:.1f}%** and
**{100*neutral['aggregate_endpoint_fraction_of_span_attention']:.1f}%** of the
historical bucket's attention, respectively.

## Do the value projections preserve the JLens answer direction?

Across the actual held-out winner-letter distribution, the answer direction's
mean value-projection survival percentile was
**{100*projection['mean_value_survival_percentile']:.1f}** relative to isotropic
unit directions; its mean key-projection percentile was
**{100*projection['mean_key_survival_percentile']:.1f}**. Thus the direction is
not globally annihilated, but survival is head-dependent.

## Does the surviving value path explain the causal result?

Composing each head's value and output projections with the exact endpoint
attention and output gate predicts an immediate winner-contrast change of
**{causal_bridge['game']['predicted_immediate_value_path_winner_contrast']['mean']:+.4f}**
in Game and
**{causal_bridge['neutral']['predicted_immediate_value_path_winner_contrast']['mean']:+.4f}**
in Neutral for the prior winner-erasure intervention. The corresponding actual
final changes were
**{causal_bridge['game']['actual_final_winner_contrast']['mean']:+.4f}** and
**{causal_bridge['neutral']['actual_final_winner_contrast']['mean']:+.4f}**.

The head table and figure show which endpoint-attending heads preserve the
direction and what immediate A-D-aligned write they make.
"""
    (output / "MIXER56_ANSWER_READ_REPORT.md").write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--intervention-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(
        args.diagnostic_root,
        args.intervention_root,
        args.output,
        args.bootstrap,
        args.seed,
    )


if __name__ == "__main__":
    main()
