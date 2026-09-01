from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .run_fixed_a_kv_layer_localization import ATTENTION_BLOCKS, BANDS, LAYER_CELLS


def _entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -(probabilities * np.log2(np.clip(probabilities, 1e-12, None))).sum(axis=-1)


def _interval(values: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    means = values[indices].mean(axis=1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def _load(path: Path) -> dict[str, np.ndarray]:
    arrays = dict(np.load(path / "results.npz", allow_pickle=False))
    if not np.all(arrays["completed"]):
        raise ValueError(f"Incomplete result: {path}")
    if arrays["layer_cells"].astype(str).tolist() != list(LAYER_CELLS):
        raise ValueError(f"Unexpected layer cells: {path}")
    return arrays


def _transfers(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    identity = arrays["layer_logits"][0]
    patched = arrays["layer_logits"]
    x = np.asarray(["ABCD".index(v) for v in arrays["x_second_letter"].astype(str)])
    y = np.asarray(["ABCD".index(v) for v in arrays["y_second_letter"].astype(str)])
    q = np.arange(len(x))
    identity_answers = identity.argmax(axis=-1)
    identity_entropy = _entropy(identity)
    output: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "game_margin", "neutral_margin", "game_minus_neutral_margin",
            "game_donor_selection", "neutral_donor_selection",
            "game_entropy_bits", "neutral_entropy_bits",
            "game_answer_changed", "neutral_answer_changed",
        )
    }
    for cell_index in range(len(LAYER_CELLS)):
        cell = patched[cell_index]
        game_x = cell[0, q, x] - cell[0, q, y] - identity[0, q, x] + identity[0, q, y]
        game_y = cell[2, q, y] - cell[2, q, x] - identity[2, q, y] + identity[2, q, x]
        neutral_x = cell[1, q, x] - cell[1, q, y] - identity[1, q, x] + identity[1, q, y]
        neutral_y = cell[3, q, y] - cell[3, q, x] - identity[3, q, y] + identity[3, q, x]
        game = 0.5 * (game_x + game_y)
        neutral = 0.5 * (neutral_x + neutral_y)
        output["game_margin"].append(game)
        output["neutral_margin"].append(neutral)
        output["game_minus_neutral_margin"].append(game - neutral)
        answers = cell.argmax(axis=-1)
        output["game_donor_selection"].append(
            0.5 * ((answers[0] == y).astype(float) - (identity_answers[0] == y).astype(float) +
                   (answers[2] == x).astype(float) - (identity_answers[2] == x).astype(float))
        )
        output["neutral_donor_selection"].append(
            0.5 * ((answers[1] == y).astype(float) - (identity_answers[1] == y).astype(float) +
                   (answers[3] == x).astype(float) - (identity_answers[3] == x).astype(float))
        )
        entropy = _entropy(cell) - identity_entropy
        output["game_entropy_bits"].append(0.5 * (entropy[0] + entropy[2]))
        output["neutral_entropy_bits"].append(0.5 * (entropy[1] + entropy[3]))
        output["game_answer_changed"].append(
            0.5 * ((answers[0] != identity_answers[0]).astype(float) +
                   (answers[2] != identity_answers[2]).astype(float))
        )
        output["neutral_answer_changed"].append(
            0.5 * ((answers[1] != identity_answers[1]).astype(float) +
                   (answers[3] != identity_answers[3]).astype(float))
        )
    return {key: np.stack(value) for key, value in output.items()}


def _summarize(
    root: Path, reference_root: Path, draws: int, seed: int
) -> dict[str, Any]:
    complete = _load(root)
    eligible = complete["first_decision_valid"].astype(bool)
    arrays = {
        "question_ids": complete["question_ids"][eligible],
        "x_second_letter": complete["x_second_letter"][eligible],
        "y_second_letter": complete["y_second_letter"][eligible],
        "natural_logits": complete["natural_logits"][:, eligible],
        "first_decision_logits": complete["first_decision_logits"][:, eligible],
        "layer_logits": complete["layer_logits"][:, :, eligible],
    }
    transfers = _transfers(arrays)
    n = len(arrays["question_ids"])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(draws, n))
    cells = []
    for cell_index, cell in enumerate(LAYER_CELLS):
        cells.append({
            "cell": cell,
            "target_blocks": list(_blocks_for_cell(cell)),
            **{metric: _interval(values[cell_index], indices) for metric, values in transfers.items()},
        })

    reference = dict(np.load(reference_root / "results.npz", allow_pickle=False))
    reference_ids = reference["question_ids"].astype(str)
    ref_lookup = {qid: i for i, qid in enumerate(reference_ids)}
    ref_indices = np.asarray([ref_lookup[qid] for qid in arrays["question_ids"].astype(str)])
    reference_valid = reference["first_decision_valid"][ref_indices].astype(bool)
    common = reference_valid
    if not np.any(common):
        raise ValueError("No questions are eligible in both the current and reference runs")
    reference_cells = reference["source_cells"].astype(str).tolist()
    ref_identity = reference["source_logits"][reference_cells.index("identity")][
        :, ref_indices[common]
    ]
    ref_selected = reference["source_logits"][reference_cells.index("selected_option")][
        :, ref_indices[common]
    ]
    current_identity = arrays["layer_logits"][LAYER_CELLS.index("identity")]
    current_selected = arrays["layer_logits"][LAYER_CELLS.index("all_selected_option")]
    identity = current_identity[:, common]
    selected = current_selected[:, common]
    validation = {
        "n_historical": int(len(complete["question_ids"])),
        "n_eligible": int(n),
        "n_reference_common_eligible": int(common.sum()),
        "reference_eligibility_disagreements": int((~common).sum()),
        "identity_vs_reference_max_abs_ad_logit_error": float(np.max(np.abs(identity - ref_identity))),
        "all_selected_vs_reference_max_abs_ad_logit_error": float(np.max(np.abs(selected - ref_selected))),
        "identity_vs_natural_answer_differences": int(
            np.sum(current_identity.argmax(-1) != arrays["natural_logits"].argmax(-1))
        ),
        "patched_layer_counts": dict(zip(LAYER_CELLS, complete["patched_layer_counts"].astype(int).tolist())),
    }
    all_idx = LAYER_CELLS.index("all_selected_option")
    necessities = []
    for band in BANDS:
        without_idx = LAYER_CELLS.index(f"without_{band}")
        necessities.append({
            "band": band,
            "blocks": list(BANDS[band]),
            **{
                metric: _interval(
                    transfers[metric][all_idx] - transfers[metric][without_idx], indices
                )
                for metric in ("game_margin", "neutral_margin", "game_minus_neutral_margin")
            },
        })
    return {"n": n, "validation": validation, "cells": cells, "necessity": necessities}


def _blocks_for_cell(cell: str) -> tuple[int, ...]:
    if cell == "identity": return ()
    if cell.startswith("block_"): return (int(cell.rsplit("_", 1)[1]),)
    if cell in BANDS: return BANDS[cell]
    if cell.startswith("without_"):
        omit = set(BANDS[cell.removeprefix("without_")])
        return tuple(block for block in ATTENTION_BLOCKS if block not in omit)
    if cell == "all_selected_option": return ATTENTION_BLOCKS
    raise KeyError(cell)


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    colors = {"game_margin": "#2f91f3", "neutral_margin": "#f0803c"}
    markers = {"discovery": "o", "confirmation": "s"}
    alpha = {"discovery": 0.55, "confirmation": 1.0}
    single_indices = [LAYER_CELLS.index(f"block_{block:02d}") for block in ATTENTION_BLOCKS]
    for metric, color in colors.items():
        for split in ("discovery", "confirmation"):
            rows = summary[split]["cells"]
            means = np.asarray([rows[i][metric]["mean"] for i in single_indices])
            cis = np.asarray([rows[i][metric]["ci"] for i in single_indices])
            axes[0, 0].errorbar(
                ATTENTION_BLOCKS, means,
                yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
                marker=markers[split], color=color, alpha=alpha[split], capsize=2,
                linewidth=1, label=f"{split.title()} {'Game' if metric.startswith('game') else 'Neutral'}",
            )
    axes[0, 0].axhline(0, color="#999", linestyle="--", linewidth=1)
    axes[0, 0].set_title("A  Individual-layer semantic transfer", loc="left")
    axes[0, 0].set_ylabel("Donor semantic-margin change")
    axes[0, 0].set_xlabel("Ordinary-attention block")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)

    for split, color in (("discovery", "#777777"), ("confirmation", "#2674d9")):
        rows = summary[split]["cells"]
        metric = "game_minus_neutral_margin"
        means = np.asarray([rows[i][metric]["mean"] for i in single_indices])
        cis = np.asarray([rows[i][metric]["ci"] for i in single_indices])
        axes[0, 1].errorbar(
            ATTENTION_BLOCKS, means,
            yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
            marker=markers[split], color=color, capsize=2, linewidth=1,
            label=split.title(),
        )
    axes[0, 1].axhline(0, color="#999", linestyle="--", linewidth=1)
    axes[0, 1].set_title("B  Individual-layer Game minus Neutral", loc="left")
    axes[0, 1].set_ylabel("Differential margin change")
    axes[0, 1].set_xlabel("Ordinary-attention block")
    axes[0, 1].legend(frameon=False)

    band_names = list(BANDS)
    band_indices = [LAYER_CELLS.index(name) for name in band_names]
    x = np.arange(len(band_names))
    width_offsets = {"game_margin": -0.10, "neutral_margin": 0.10}
    rows = summary["confirmation"]["cells"]
    for metric, color in colors.items():
        means = np.asarray([rows[i][metric]["mean"] for i in band_indices])
        cis = np.asarray([rows[i][metric]["ci"] for i in band_indices])
        axes[1, 0].errorbar(
            x + width_offsets[metric], means,
            yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
            fmt="o", capsize=4, color=color,
            label="Game" if metric.startswith("game") else "Neutral",
        )
    axes[1, 0].axhline(0, color="#999", linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(x, ["4–16", "20–32", "36–48", "52–64"])
    axes[1, 0].set_title("C  Held-out four-block band sufficiency", loc="left")
    axes[1, 0].set_ylabel("Donor semantic-margin change")
    axes[1, 0].set_xlabel("Transplanted blocks")
    axes[1, 0].legend(frameon=False)

    rows = summary["confirmation"]["necessity"]
    metric = "game_minus_neutral_margin"
    means = np.asarray([row[metric]["mean"] for row in rows])
    cis = np.asarray([row[metric]["ci"] for row in rows])
    axes[1, 1].errorbar(
        x, means, yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
        fmt="o", capsize=4, color="#8a3ffc",
    )
    axes[1, 1].axhline(0, color="#999", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(x, ["4–16", "20–32", "36–48", "52–64"])
    axes[1, 1].set_title("D  Held-out contribution lost when band is omitted", loc="left")
    axes[1, 1].set_ylabel("All-layer minus leave-band-out Game−Neutral effect")
    axes[1, 1].set_xlabel("Omitted blocks")
    for axis in axes.flat:
        axis.grid(alpha=0.15)
    fig.suptitle("Which ordinary-attention layers use the first answer's semantic K/V")
    fig.text(
        0.5, -0.01,
        "Negative in A/C = movement toward donor answer; positive in B/D = contribution to weaker reinstatement in Game. Bars are 95% question-bootstrap CIs.",
        ha="center", fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fmt(value: dict[str, Any]) -> str:
    lo, hi = value["ci"]
    return f"{value['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _fmt_pp(value: dict[str, Any]) -> str:
    lo, hi = np.asarray(value["ci"]) * 100
    return f"{100 * value['mean']:+.1f} [{lo:+.1f}, {hi:+.1f}] pp"


def analyze(
    discovery: Path, confirmation: Path,
    reference_discovery: Path, reference_confirmation: Path,
    output: Path, figure: Path, draws: int, seed: int,
) -> None:
    summary = {
        "design": {
            "attention_blocks": list(ATTENTION_BLOCKS),
            "bands": {key: list(value) for key, value in BANDS.items()},
            "primary_endpoint": "symmetric donor semantic-margin transfer",
        },
        "discovery": _summarize(discovery, reference_discovery, draws, seed),
        "confirmation": _summarize(confirmation, reference_confirmation, draws, seed + 1),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _plot(summary, figure)
    lines = [
        "# Fixed-A selected-option K/V layer localization", "",
        "## Validation", "",
    ]
    for split in ("discovery", "confirmation"):
        val = summary[split]["validation"]
        lines.extend([
            f"- {split.title()}: {val['n_eligible']}/{val['n_historical']} exact-regime questions.",
            f"- Identity versus prior source run maximum A-D error: {val['identity_vs_reference_max_abs_ad_logit_error']:.6g} logits.",
            f"- All-layer selected-option versus prior source run maximum A-D error: {val['all_selected_vs_reference_max_abs_ad_logit_error']:.6g} logits.",
            f"- Cached identity versus unsplit natural answer differences: {val['identity_vs_natural_answer_differences']}.",
        ])
    for split in ("discovery", "confirmation"):
        lines.extend(["", f"## {split.title()} layer effects", "",
            "| Cell | Blocks | Game margin | Neutral margin | Game − Neutral | Game donor chosen | Neutral donor chosen |",
            "|---|---|---:|---:|---:|---:|---:|"])
        for row in summary[split]["cells"]:
            blocks = ",".join(map(str, row["target_blocks"])) or "none"
            lines.append(
                f"| {row['cell']} | {blocks} | {_fmt(row['game_margin'])} | "
                f"{_fmt(row['neutral_margin'])} | {_fmt(row['game_minus_neutral_margin'])} | "
                f"{_fmt_pp(row['game_donor_selection'])} | {_fmt_pp(row['neutral_donor_selection'])} |"
            )
        lines.extend(["", f"### {split.title()} leave-one-band-out necessity", "",
            "| Omitted band | Game contribution | Neutral contribution | Game − Neutral contribution |",
            "|---|---:|---:|---:|"])
        for row in summary[split]["necessity"]:
            lines.append(
                f"| {row['band']} | {_fmt(row['game_margin'])} | {_fmt(row['neutral_margin'])} | "
                f"{_fmt(row['game_minus_neutral_margin'])} |"
            )
    lines.extend(["", f"Canonical figure: `{figure}`.", ""])
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--reference-discovery", type=Path, required=True)
    parser.add_argument("--reference-confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    analyze(args.discovery, args.confirmation, args.reference_discovery,
            args.reference_confirmation, args.output, args.figure, args.draws, args.seed)


if __name__ == "__main__":
    main()
