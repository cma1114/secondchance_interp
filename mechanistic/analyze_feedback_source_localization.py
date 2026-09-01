from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .run_feedback_source_localization import SCENARIOS, SOURCE_TOKEN_INDICES


TASKS = ("Game", "Neutral")
TOKEN_LABELS = {
    3: "incorrect / lost",
    4: "first period",
    5: "Choose",
    6: "the",
    7: "answer",
    8: "again",
    9: "final period",
}


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _mean_ci(
    values: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, list[float]]:
    selected = np.asarray(values[indices], dtype=np.float64)
    point = float(selected.mean())
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        stop = min(start + 200, draws)
        rows = rng.integers(0, len(selected), size=(stop - start, len(selected)))
        samples[start:stop] = selected[rows].mean(axis=1)
    return point, [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def _ratio_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, list[float]]:
    num = np.asarray(numerator[indices], dtype=np.float64)
    den = np.asarray(denominator[indices], dtype=np.float64)
    point = float(num.sum() / den.sum())
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        stop = min(start + 200, draws)
        rows = rng.integers(0, len(num), size=(stop - start, len(num)))
        samples[start:stop] = num[rows].sum(axis=1) / den[rows].sum(axis=1)
    return point, [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def analyze(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    qids = arrays["question_ids"].astype(str).tolist()
    scenarios = arrays["scenario_ids"].astype(str).tolist()
    if scenarios != list(SCENARIOS):
        raise RuntimeError("Scenario definition changed")
    if len(qids) != 500 or not arrays["completed"].all():
        raise RuntimeError("A complete 500-question result is required")
    logits = arrays["scenario_final_logits"].astype(np.float64)
    if logits.shape != (2, len(SCENARIOS), 500, 4) or not np.isfinite(logits).all():
        raise RuntimeError("Causal logits are incomplete or non-finite")

    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    if [int(discovery.sum()), int((~discovery).sum())] != [251, 249]:
        raise RuntimeError("Frozen split changed")
    split_masks = {"discovery": discovery, "confirmation": ~discovery}

    baseline = json.loads(args.baseline.read_text())["results"]
    fresh = json.loads(args.remapped_baseline.read_text())["results"]
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    rank_order = np.empty((500, 4), dtype=np.int64)
    w1 = np.empty(500, dtype=np.int64)
    w2 = np.empty(500, dtype=np.int64)
    for qi, qid in enumerate(qids):
        old = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=np.float64)
        semantic_rank = np.argsort(-old, kind="stable")
        rank_order[qi] = [
            LETTERS.index(mappings[qid]["original_to_new"][LETTERS[int(index)]])
            for index in semantic_rank
        ]
        w1[qi] = rank_order[qi, 0]
        w2[qi] = int(np.argmax(fresh[qid]["aggregated_ad_logits"]))

    centered = _center(logits)
    natural = centered[:, 0]
    task_vector = natural[::-1] - natural
    denominator = np.sum(task_vector * task_vector, axis=-1)
    delta = centered - natural[:, None]
    numerator = np.sum(delta * task_vector[:, None], axis=-1)
    choices = np.argmax(logits, axis=-1)
    switch = choices != w1[None, None, :]
    choose_w1 = choices == w1[None, None, :]
    ranked = np.empty_like(centered)
    for qi in range(500):
        ranked[:, :, qi] = np.take(centered[:, :, qi], rank_order[qi], axis=-1)
    bivalent = ranked[..., 3] - ranked[..., :2].mean(axis=-1)

    summary: dict[str, Any] = {
        "question": "Which policy-bearing feedback token causally sends task information forward?",
        "coverage": {
            "questions": 500,
            "discovery": 251,
            "confirmation": 249,
            "tasks": list(TASKS),
            "source_tokens": TOKEN_LABELS,
            "ordinary_attention_layers": list(range(4, 65, 4)),
            "gla_layers": [value for value in range(1, 65) if value % 4 != 0],
        },
        "validation": {
            "all_complete": True,
            "all_finite": True,
            "corrected_natural_max_abs_error": float(
                np.max(np.abs(logits[:, 0] - arrays["trusted_natural_logits"]))
            ),
        },
        "definitions": {
            "transfer_fraction": (
                "Projection of intervention-induced centered A-D logit change onto "
                "the paired natural donor-minus-recipient task vector"
            ),
            "switch": "Final answer is not semantic W1",
            "bivalent": "Centered R4 logit minus mean centered R1/R2 logit",
        },
        "splits": {},
    }
    raw = arrays["scenario_final_logits_raw"].astype(np.float64)
    summary["validation"]["same_batch_natural_max_abs_error"] = float(
        np.max(np.abs(raw[:, 0] - arrays["same_batch_natural_logits"]))
    )

    for split_index, (split_name, mask) in enumerate(split_masks.items()):
        indices = np.flatnonzero(mask)
        split: dict[str, Any] = {"questions": int(mask.sum()), "tasks": {}}
        for task_index, task in enumerate(TASKS):
            rows: dict[str, Any] = {}
            for scenario_index, scenario in enumerate(scenarios):
                seed = args.seed + split_index * 100000 + task_index * 10000 + scenario_index
                rng = np.random.default_rng(seed)
                transfer, transfer_ci = _ratio_ci(
                    numerator[task_index, scenario_index], denominator[task_index],
                    indices, rng, args.bootstrap_draws,
                )
                sw, sw_ci = _mean_ci(
                    switch[task_index, scenario_index], indices, rng, args.bootstrap_draws
                )
                w1_rate, w1_ci = _mean_ci(
                    choose_w1[task_index, scenario_index], indices, rng, args.bootstrap_draws
                )
                biv_delta, biv_ci = _mean_ci(
                    bivalent[task_index, scenario_index] - bivalent[task_index, 0],
                    indices, rng, args.bootstrap_draws,
                )
                rows[scenario] = {
                    "transfer_fraction": transfer,
                    "transfer_fraction_ci": transfer_ci,
                    "switch_rate": sw,
                    "switch_rate_ci": sw_ci,
                    "choose_W1_rate": w1_rate,
                    "choose_W1_rate_ci": w1_ci,
                    "bivalent_change": biv_delta,
                    "bivalent_change_ci": biv_ci,
                }
            split["tasks"][task] = rows
        summary["splits"][split_name] = split

    suffix = "feedback_suffix_3_9_swapped"
    checks: dict[str, Any] = {}
    gate_pass = True
    for split_name in ("discovery", "confirmation"):
        for task in TASKS:
            row = summary["splits"][split_name]["tasks"][task][suffix]
            passed = row["transfer_fraction"] > 0 and row["transfer_fraction_ci"][0] > 0
            checks[f"{split_name}_{task}"] = {
                "transfer_fraction": row["transfer_fraction"],
                "ci": row["transfer_fraction_ci"],
                "passed": bool(passed),
            }
            gate_pass = gate_pass and passed
    summary["relay_mediation_gate"] = {
        "passed": bool(gate_pass),
        "criterion": "complete-suffix transfer positive with 95% CI lower bound > 0 in both tasks and both splits",
        "checks": checks,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    labels = [TOKEN_LABELS[index] for index in SOURCE_TOKEN_INDICES] + ["complete suffix"]
    selected_scenarios = [
        *(f"feedback_token_{index}_swapped" for index in SOURCE_TOKEN_INDICES),
        suffix,
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    colors = {"Game": "#d95f02", "Neutral": "#1b7fb8"}
    for task_index, task in enumerate(TASKS):
        rows = summary["splits"]["confirmation"]["tasks"][task]
        values = np.asarray([rows[name]["transfer_fraction"] for name in selected_scenarios])
        cis = np.asarray([rows[name]["transfer_fraction_ci"] for name in selected_scenarios])
        errors = np.stack((values - cis[:, 0], cis[:, 1] - values))
        axes[0].bar(
            x + (task_index - 0.5) * 0.36, values, 0.36,
            yerr=errors, capsize=3, color=colors[task], label=task,
        )
        rates = np.asarray([rows[name]["switch_rate"] for name in selected_scenarios])
        rate_cis = np.asarray([rows[name]["switch_rate_ci"] for name in selected_scenarios])
        rate_errors = np.stack((rates - rate_cis[:, 0], rate_cis[:, 1] - rates))
        axes[1].bar(
            x + (task_index - 0.5) * 0.36, rates, 0.36,
            yerr=rate_errors, capsize=3, color=colors[task], label=task,
        )
        axes[1].axhline(
            rows["natural"]["switch_rate"], color=colors[task], ls=":", lw=1
        )
    for axis in axes:
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.legend()
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(title="A  Donor task-vector transfer (confirmation)", ylabel="Transfer fraction")
    axes[1].set(title="B  Raw switch rates (confirmation)", ylabel="Switch rate")
    fig.suptitle("Which feedback token causally sends policy information forward?")
    fig.savefig(args.figure, dpi=190)
    plt.close(fig)

    lines = [
        "# Feedback-source localization",
        "",
        "This experiment reciprocally swaps only the downstream ordinary-attention and GLA memory writes of each feedback token from `incorrect/lost` through the final period. The source token's own residual remains natural. Game and Neutral are reported separately.",
        "",
        f"The prespecified complete-suffix relay-mediation gate **{'passed' if gate_pass else 'did not pass'}**.",
        "",
        "## Confirmation results",
        "",
        "| Source write crossed over | Game transfer (95% CI) | Game switch | Neutral transfer (95% CI) | Neutral switch |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, scenario in zip(labels, selected_scenarios):
        game = summary["splits"]["confirmation"]["tasks"]["Game"][scenario]
        neutral = summary["splits"]["confirmation"]["tasks"]["Neutral"][scenario]
        lines.append(
            f"| {label} | {game['transfer_fraction']:.3f} "
            f"[{game['transfer_fraction_ci'][0]:.3f}, {game['transfer_fraction_ci'][1]:.3f}] | "
            f"{100*game['switch_rate']:.1f}% | {neutral['transfer_fraction']:.3f} "
            f"[{neutral['transfer_fraction_ci'][0]:.3f}, {neutral['transfer_fraction_ci'][1]:.3f}] | "
            f"{100*neutral['switch_rate']:.1f}% |"
        )
    lines += [
        "",
        "Natural switch rates and every discovery estimate are retained in `summary.json`.",
        "",
        "## Interpretation rule",
        "",
        "A token is a causal downstream source only to the extent that crossing over its complete ordinary-attention and GLA memory writes moves the recipient toward the paired donor task. Individual token effects are not added because later contextualized tokens can redundantly carry information from the earlier keyword.",
        "",
        f"Canonical figure: [{args.figure.name}]({args.figure.resolve()})",
    ]
    args.report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
