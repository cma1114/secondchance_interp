from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_jlens_answer_content import (
    RANKS,
    _output_labels,
    answer_letter_scores,
    baseline_rank_order,
)
from .answer_emergence_figures import Z_975, macro_mean_and_se
from .data import load_activation_dataset


METHODS = ("native_logit_lens", "jlens", "pooled_probe")
METHOD_LABELS = {
    "native_logit_lens": "Native logit lens",
    "jlens": "Jacobian lens",
    "pooled_probe": "Cross-fitted pooled probe",
}
METHOD_UNITS = {
    "native_logit_lens": "Centered Game - Neutral score (logit units)",
    "jlens": "Centered Game - Neutral score (logit units)",
    "pooled_probe": "Centered Game - Neutral score (Baseline SD units)",
}


def _center(scores: np.ndarray) -> np.ndarray:
    return scores.astype(np.float64) - scores.astype(np.float64).mean(axis=-1, keepdims=True)


def _balanced_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean([np.mean(prediction[target == letter] == letter) for letter in range(4)]))


def summarize_method(scores: np.ndarray, order: np.ndarray, prior: np.ndarray) -> list[dict]:
    """Summarize paired Game-minus-Neutral scores aligned by fixed Baseline rank.

    ``scores`` is condition x question x layer x option with the condition order
    Baseline, Game, Neutral.
    """
    centered = _center(scores)
    contrast = centered[1] - centered[2]
    aligned = np.take_along_axis(contrast, order[:, None, :], axis=-1)
    rows = []
    for rank, label in enumerate(RANKS):
        mean, se = macro_mean_and_se(aligned[:, :, rank], prior)
        rows.append({
            "rank": label,
            "mean": mean,
            "ci_low": mean - Z_975 * se,
            "ci_high": mean + Z_975 * se,
        })
    return rows


def analyze(
    residual_root: str | Path,
    jlens_root: str | Path,
    pooled_probe_path: str | Path,
    output_root: str | Path,
) -> dict:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    data = load_activation_dataset(residual_root, ["baseline", "incorrect", "neutral"])
    order, prior = baseline_rank_order(data)

    native = np.transpose(data.logits[:, :, 1:].astype(np.float64), (1, 0, 2, 3))
    layout = json.loads((Path(jlens_root) / "selected_token_layout.json").read_text())
    with np.load(Path(jlens_root) / "jlens_scores.npz", allow_pickle=False) as cached:
        jlens_qids = cached["question_ids"].astype(str).tolist()
        if jlens_qids != data.question_ids:
            raise ValueError("JLens and activation question orders differ")
        jlens = answer_letter_scores(cached["final_scores"].astype(np.float64), layout)
    with np.load(pooled_probe_path, allow_pickle=False) as cached:
        probe_qids = cached["simple_question_ids"].astype(str).tolist()
        conditions = cached["conditions"].astype(str).tolist()
        if probe_qids != data.question_ids:
            raise ValueError("Pooled-probe and activation question orders differ")
        if conditions != ["baseline", "incorrect", "neutral"]:
            raise ValueError(f"Unexpected pooled-probe condition order: {conditions}")
        # Readout 0 is the embedding; the JLens artifacts begin after block 1.
        probe = cached["simple_scores"][:, :, 1:].astype(np.float64)
    arrays = {"native_logit_lens": native, "jlens": jlens, "pooled_probe": probe}
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError(f"Readout shapes differ: { {key: value.shape for key, value in arrays.items()} }")

    generated = {
        condition: _output_labels(data, condition)
        for condition in ("baseline", "incorrect", "neutral")
    }
    payload = {
        "layers": list(range(1, 65)),
        "ranks": list(RANKS),
        "methods": {},
        "notes": {
            "contrast": "Paired same-question Game minus Neutral scores.",
            "ranking": (
                "The generated Baseline answer is rank 1; the other options are ordered by final Baseline A-D logits."
            ),
            "centering": "Each condition is centered across its four options within question before subtraction.",
            "ci": "95% normal intervals for an equal-weight macro-average over original-answer letters.",
            "alignment": "All methods use post-block readouts 1-64. Native and JLens are identical at readout 64.",
        },
    }
    csv_rows = []
    summaries = {}
    accuracies = {}
    for method in METHODS:
        rows = summarize_method(arrays[method], order, prior)
        summaries[method] = rows
        accuracy = {}
        centered = _center(arrays[method])
        for ci, condition in ((1, "incorrect"), (2, "neutral")):
            target = generated[condition]
            values = []
            for layer in range(64):
                values.append(_balanced_accuracy(centered[ci, :, layer].argmax(axis=-1), target))
            accuracy[condition] = np.asarray(values)
        accuracies[method] = accuracy
        payload["methods"][method] = {
            "label": METHOD_LABELS[method],
            "units": METHOD_UNITS[method],
            "series": [
                {
                    "rank": row["rank"],
                    "mean": np.round(row["mean"], 4).tolist(),
                    "ci_low": np.round(row["ci_low"], 4).tolist(),
                    "ci_high": np.round(row["ci_high"], 4).tolist(),
                }
                for row in rows
            ],
            "balanced_accuracy": {
                condition: np.round(values, 4).tolist() for condition, values in accuracy.items()
            },
        }
        for row in rows:
            for layer, mean, low, high in zip(
                payload["layers"], row["mean"], row["ci_low"], row["ci_high"]
            ):
                csv_rows.append({
                    "method": method,
                    "rank": row["rank"],
                    "layer": layer,
                    "mean": float(mean),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "units": METHOD_UNITS[method],
                })

    reliable_layers = np.arange(47, 64)  # readouts 48--64, established in the prior probe audit
    correlations = {}
    for first, second in (
        ("native_logit_lens", "jlens"),
        ("native_logit_lens", "pooled_probe"),
        ("jlens", "pooled_probe"),
    ):
        first_values = np.stack([row["mean"][reliable_layers] for row in summaries[first]]).reshape(-1)
        second_values = np.stack([row["mean"][reliable_layers] for row in summaries[second]]).reshape(-1)
        correlations[f"{first}__{second}"] = float(np.corrcoef(first_values, second_values)[0, 1])

    key_layers = (48, 52, 56, 60, 64)
    summary = {"correlations_readouts_48_64_all_ranks": correlations, "key_layers": {}}
    for method in METHODS:
        summary["key_layers"][method] = {}
        for layer in key_layers:
            index = layer - 1
            summary["key_layers"][method][str(layer)] = {
                row["rank"]: {
                    "mean": float(row["mean"][index]),
                    "ci_low": float(row["ci_low"][index]),
                    "ci_high": float(row["ci_high"][index]),
                }
                for row in summaries[method]
            }
            summary["key_layers"][method][str(layer)]["balanced_accuracy"] = {
                condition: float(values[index]) for condition, values in accuracies[method].items()
            }

    (output / "readout_method_comparison.json").write_text(json.dumps(payload, separators=(",", ":")))
    (output / "readout_method_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "readout_method_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    _plot(summaries, output / "readout_method_comparison.png")
    return summary


def _plot(summaries: dict[str, list[dict]], destination: Path) -> None:
    import matplotlib.pyplot as plt

    from .all_trial_figures import _style
    from .answer_emergence_figures import RANK_COLORS

    _style()
    layers = np.arange(1, 65)
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.25), sharex=True)
    logit_limit = max(
        abs(value)
        for method in ("native_logit_lens", "jlens")
        for row in summaries[method]
        for value in np.concatenate((row["ci_low"], row["ci_high"]))
    )
    for axis, method, panel in zip(axes, METHODS, "ABC"):
        for row, color in zip(summaries[method], RANK_COLORS):
            axis.fill_between(
                layers, row["ci_low"], row["ci_high"], color=color, alpha=0.16, linewidth=0
            )
            axis.plot(layers, row["mean"], color=color, linewidth=1.45, label=row["rank"])
        axis.axhline(0, color="#555555", linewidth=0.7)
        axis.axvspan(1, 47.5, color="#BDBDBD", alpha=0.12, linewidth=0)
        axis.axvline(48, color="#777777", linewidth=0.75, linestyle=(0, (3, 2)))
        axis.set_xlim(1, 64)
        axis.set_xticks((1, 8, 16, 24, 32, 40, 48, 56, 64))
        axis.set_title(f"{panel}  {METHOD_LABELS[method]}", loc="left", fontweight="bold")
        axis.set_ylabel(METHOD_UNITS[method].replace("Centered ", ""))
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        if method in ("native_logit_lens", "jlens"):
            axis.set_ylim(-logit_limit * 1.03, logit_limit * 1.03)
    axes[0].text(24, axes[0].get_ylim()[1] * 0.88, "low answer-decoding reliability", ha="center", color="#666666")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.supxlabel("Residual readout (64 = natural final residual)", y=0.01)
    figure.tight_layout(rect=(0, 0.04, 1, 0.92), w_pad=1.25)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed-rank Game-minus-Neutral answer readouts")
    parser.add_argument("--residual-root", required=True)
    parser.add_argument("--jlens-root", required=True)
    parser.add_argument("--pooled-probe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.residual_root, args.jlens_root, args.pooled_probe, args.output), indent=2))


if __name__ == "__main__":
    main()
