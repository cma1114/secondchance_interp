from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .semantic_mapping import displayed_argmax_to_semantic_indices


LETTERS = "ABCD"
DIRECTIONS = ("evaluation_into_neutral", "neutral_into_evaluation")


def _entropy_bits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-300)), axis=-1)


def _interval(values, strata, rng, draws=10_000):
    values = np.asarray(values, dtype=float)
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    samples = np.zeros(draws, dtype=float)
    for group in groups:
        selected = rng.choice(group, size=(draws, len(group)), replace=True)
        samples += values[selected].sum(axis=1)
    samples /= len(values)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def _fmt(value, scale=1.0, digits=3, signed=True):
    sign = "+" if signed else ""
    return (
        f"{value['mean'] * scale:{sign}.{digits}f} "
        f"[{value['ci_low'] * scale:{sign}.{digits}f}, "
        f"{value['ci_high'] * scale:{sign}.{digits}f}]"
    )


def analyze(
    results_path: Path,
    metadata_path: Path,
    baseline_path: Path,
    remapped_baseline_path: Path,
    remapping_plan_path: Path,
    output_dir: Path,
    seed: int,
):
    with np.load(results_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"]):
        raise RuntimeError("Transplant run is incomplete")
    metadata = json.loads(metadata_path.read_text())
    qids = arrays["question_ids"].astype(str).tolist()
    scenarios = arrays["scenario_ids"].astype(str).tolist()
    baseline = json.loads(baseline_path.read_text())["results"]
    remapped_baseline = json.loads(remapped_baseline_path.read_text())["results"]
    plan = {
        row["question_id"]: row
        for row in json.loads(remapping_plan_path.read_text())["rows"]
    }
    w1 = np.asarray([baseline[qid]["answer"] for qid in qids])
    w2 = np.asarray([remapped_baseline[qid]["answer_original_content"] for qid in qids])
    conflict = w1 != w2
    strata = w1.copy()

    def content_aligned(values):
        aligned = np.empty_like(values)
        for qi, qid in enumerate(qids):
            original_to_new = plan[qid]["original_to_new"]
            aligned[..., qi, :] = np.stack([
                values[..., qi, LETTERS.index(original_to_new[original])]
                for original in LETTERS
            ], axis=-1)
        return aligned

    natural = content_aligned(arrays["trusted_natural_logits"].astype(float))
    patched = content_aligned(arrays["patched_logits"].astype(float))
    natural_centered = natural - natural.mean(axis=-1, keepdims=True)
    patched_centered = patched - patched.mean(axis=-1, keepdims=True)
    w1_index = np.asarray([LETTERS.index(value) for value in w1])
    w2_index = np.asarray([LETTERS.index(value) for value in w2])
    qi = np.arange(len(qids))

    mapping_rows = [plan[qid] for qid in qids]
    natural_answer = displayed_argmax_to_semantic_indices(
        arrays["trusted_natural_logits"], mapping_rows
    )
    patched_answer = displayed_argmax_to_semantic_indices(
        arrays["patched_logits"], mapping_rows
    )
    natural_entropy = _entropy_bits(natural)
    patched_entropy = _entropy_bits(patched)
    rng = np.random.default_rng(seed)

    def summarize(mask):
        local_strata = strata[mask]
        output = {"n": int(mask.sum()), "scenarios": {}}
        for si, scenario in enumerate(scenarios):
            # Each direction is signed toward its own donor task: Evaluation
            # for evaluation_into_neutral, Neutral for neutral_into_evaluation.
            n_margin = (
                natural_centered[1, qi, w1_index]
                - natural_centered[1, qi, w2_index]
            )
            e2n_margin = (
                patched_centered[0, si, qi, w1_index]
                - patched_centered[0, si, qi, w2_index]
            )
            e_margin = (
                natural_centered[0, qi, w1_index]
                - natural_centered[0, qi, w2_index]
            )
            n2e_margin = (
                patched_centered[1, si, qi, w1_index]
                - patched_centered[1, si, qi, w2_index]
            )
            e2n_signed_margin = n_margin - e2n_margin
            n2e_signed_margin = n2e_margin - e_margin

            n_w1 = natural_answer[1] == w1_index
            e2n_w1 = patched_answer[0, si] == w1_index
            e_w1 = natural_answer[0] == w1_index
            n2e_w1 = patched_answer[1, si] == w1_index
            e2n_signed_choice = n_w1.astype(float) - e2n_w1.astype(float)
            n2e_signed_choice = n2e_w1.astype(float) - e_w1.astype(float)

            n_w2 = natural_answer[1] == w2_index
            e2n_w2 = patched_answer[0, si] == w2_index
            e_w2 = natural_answer[0] == w2_index
            n2e_w2 = patched_answer[1, si] == w2_index
            e2n_signed_w2 = e2n_w2.astype(float) - n_w2.astype(float)
            n2e_signed_w2 = e_w2.astype(float) - n2e_w2.astype(float)

            e2n_signed_entropy = patched_entropy[0, si] - natural_entropy[1]
            n2e_signed_entropy = natural_entropy[0] - patched_entropy[1, si]
            metrics = {
                "evaluation_into_neutral": {
                    "signed_w1_vs_w2_margin_transfer": e2n_signed_margin,
                    "signed_w1_selection_transfer": e2n_signed_choice,
                    "signed_w2_selection_transfer": e2n_signed_w2,
                    "signed_entropy_transfer_bits": e2n_signed_entropy,
                },
                "neutral_into_evaluation": {
                    "signed_w1_vs_w2_margin_transfer": n2e_signed_margin,
                    "signed_w1_selection_transfer": n2e_signed_choice,
                    "signed_w2_selection_transfer": n2e_signed_w2,
                    "signed_entropy_transfer_bits": n2e_signed_entropy,
                },
            }
            cell = {"directions": {}, "bidirectional_average": {}}
            for direction in DIRECTIONS:
                cell["directions"][direction] = {
                    metric: _interval(values[mask], local_strata, rng)
                    for metric, values in metrics[direction].items()
                }
            for metric in metrics[DIRECTIONS[0]]:
                average = 0.5 * (
                    metrics[DIRECTIONS[0]][metric] + metrics[DIRECTIONS[1]][metric]
                )
                cell["bidirectional_average"][metric] = _interval(
                    average[mask], local_strata, rng
                )
            output["scenarios"][scenario] = cell
        return output

    summary = {
        "design": {
            "stage": metadata["stage"],
            "directions": list(DIRECTIONS),
            "positive_sign": {
                "evaluation_into_neutral": "Movement of Neutral toward the Evaluation donor.",
                "neutral_into_evaluation": "Movement of Evaluation toward the Neutral donor.",
            },
            "intervention": metadata["intervention"],
        },
        "all_questions": summarize(np.ones(len(qids), dtype=bool)),
        "w1_not_equal_w2": summarize(conflict),
        "w1_equal_w2": summarize(~conflict),
        "batch_drift": {
            "max_abs_same_batch_control_minus_trusted": float(
                np.max(np.abs(arrays["same_batch_control_minus_trusted"]))
            ),
            "mean_abs_same_batch_control_minus_trusted": float(
                np.mean(np.abs(arrays["same_batch_control_minus_trusted"]))
            ),
            "correction": metadata["batch_drift_correction"],
        },
    }
    if metadata["stage"] == "gate":
        gate = summary["w1_not_equal_w2"]["scenarios"][scenarios[0]]
        e2n = gate["directions"]["evaluation_into_neutral"][
            "signed_w1_vs_w2_margin_transfer"
        ]
        n2e = gate["directions"]["neutral_into_evaluation"][
            "signed_w1_vs_w2_margin_transfer"
        ]
        pooled = gate["bidirectional_average"]["signed_w1_vs_w2_margin_transfer"]
        summary["gate_decision"] = {
            "criterion": (
                "Proceed to localization only if the bidirectional-average W1-vs-W2 "
                "margin-transfer CI is strictly positive and at least one directional "
                "margin-transfer CI is strictly positive."
            ),
            "passed": bool(
                pooled["ci_low"] > 0 and (e2n["ci_low"] > 0 or n2e["ci_low"] > 0)
            ),
            "strict_bidirectional_replication": bool(
                e2n["ci_low"] > 0 and n2e["ci_low"] > 0
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        f"# Evaluation-period GLA-update transplant: {metadata['stage']}",
        "",
        "Positive values mean movement toward the named donor: `evaluation_into_neutral` "
        "moves Neutral toward Evaluation, while `neutral_into_evaluation` moves "
        "Evaluation toward Neutral. Primary inference uses W1 != W2 questions.",
        "",
        f"Questions: {len(qids)} total; {int(conflict.sum())} W1 != W2.",
        "",
        "| Scenario | Direction | W1-vs-W2 margin transfer | W1-selection transfer | W2-selection transfer | Entropy transfer |",
        "|---|---|---:|---:|---:|---:|",
    ]
    primary = summary["w1_not_equal_w2"]["scenarios"]
    for scenario in scenarios:
        for direction in DIRECTIONS:
            cell = primary[scenario]["directions"][direction]
            lines.append(
                f"| `{scenario}` | `{direction}` | "
                f"{_fmt(cell['signed_w1_vs_w2_margin_transfer'])} | "
                f"{_fmt(cell['signed_w1_selection_transfer'], 100, 1)} pp | "
                f"{_fmt(cell['signed_w2_selection_transfer'], 100, 1)} pp | "
                f"{_fmt(cell['signed_entropy_transfer_bits'])} bits |"
            )
        cell = primary[scenario]["bidirectional_average"]
        lines.append(
            f"| `{scenario}` | **bidirectional average** | "
            f"**{_fmt(cell['signed_w1_vs_w2_margin_transfer'])}** | "
            f"**{_fmt(cell['signed_w1_selection_transfer'], 100, 1)} pp** | "
            f"**{_fmt(cell['signed_w2_selection_transfer'], 100, 1)} pp** | "
            f"**{_fmt(cell['signed_entropy_transfer_bits'])} bits** |"
        )
    if "gate_decision" in summary:
        lines += [
            "",
            "## Gate decision",
            "",
            f"Passed localization gate: **{summary['gate_decision']['passed']}**.",
            f"Strict bidirectional replication: **{summary['gate_decision']['strict_bidirectional_replication']}**.",
            "",
            summary["gate_decision"]["criterion"],
        ]
    lines += [
        "",
        "## Validation",
        "",
        f"Maximum absolute same-batch control drift before correction: "
        f"{summary['batch_drift']['max_abs_same_batch_control_minus_trusted']:.6f} logits.",
        "All intervention effects are anchored to the previously validated exact natural logits.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation-period GLA-update transplants")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    analyze(
        args.results,
        args.metadata,
        args.baseline,
        args.remapped_baseline,
        args.remapping_plan,
        args.output_dir,
        args.seed,
    )


if __name__ == "__main__":
    main()
