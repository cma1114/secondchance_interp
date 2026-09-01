from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .run_action_period_source_lesion import SCENARIOS
from .semantic_mapping import displayed_argmax_to_semantic_indices


LETTERS = "ABCD"
CONDITION_NAMES = ("Evaluation", "Matched Neutral")


def _align(values: np.ndarray, qids: list[str], mappings: dict[str, dict]) -> np.ndarray:
    out = np.empty_like(values)
    for qi, qid in enumerate(qids):
        original_to_new = mappings[qid]["original_to_new"]
        for oi, original in enumerate(LETTERS):
            out[..., qi, oi] = values[..., qi, LETTERS.index(original_to_new[original])]
    return out


def _entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(shifted)
    p /= p.sum(axis=-1, keepdims=True)
    return -(p * np.log2(np.maximum(p, 1e-300))).sum(axis=-1)


def _metrics(
    logits: np.ndarray,
    w1i: np.ndarray,
    w2i: np.ndarray,
    choice: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.arange(len(w1i))
    centered = logits - logits.mean(axis=-1, keepdims=True)
    return {
        "w1_selection": (choice == w1i).astype(float),
        "w2_selection": (choice == w2i).astype(float),
        "switch_away_from_w1": (choice != w1i).astype(float),
        "w1_centered_advantage": 4.0 / 3.0 * centered[rows, w1i],
        "w1_minus_w2_margin": logits[rows, w1i] - logits[rows, w2i],
        "entropy_bits": _entropy(logits),
        "ad_spread_sd": centered.std(axis=-1),
    }


def _interval(values: np.ndarray, labels: np.ndarray, seed: int, draws: int = 5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = np.zeros(draws, dtype=float)
    for label in np.unique(labels):
        group = values[labels == label]
        selected = rng.choice(group, size=(draws, len(group)), replace=True)
        sampled += selected.sum(axis=1)
    sampled /= len(values)
    lo, hi = np.quantile(sampled, (0.025, 0.975))
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def _fmt(row: dict, scale: float = 1.0, digits: int = 2) -> str:
    return (
        f"{row['mean'] * scale:+.{digits}f} "
        f"[{row['ci_low'] * scale:+.{digits}f}, {row['ci_high'] * scale:+.{digits}f}]"
    )


def analyze(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    metadata_path = args.results.parent / "run_metadata.json"
    run_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    preserve_source_output = run_metadata.get("preserve_source_output")
    if not np.all(arrays["completed"]):
        raise RuntimeError("Action-period source-lesion run is incomplete")
    qids = arrays["question_ids"].astype(str).tolist()
    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    baseline = json.loads(args.baseline.read_text())["results"]
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    discovery = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    w1 = np.asarray([
        baseline[qid]["answer"] for qid in qids
    ])
    w2 = np.asarray([remapped[qid]["answer_original_content"] for qid in qids])
    w1i = np.asarray([LETTERS.index(value) for value in w1])
    w2i = np.asarray([LETTERS.index(value) for value in w2])
    conflict = w1 != w2
    discovery_mask = np.asarray([qid in discovery for qid in qids])

    max_error = float(np.max(np.abs(
        arrays["same_batch_natural_logits"] - arrays["trusted_natural_logits"]
    )))
    if max_error != 0.0:
        raise RuntimeError(f"Natural logits did not exactly reproduce trusted run: {max_error}")
    natural = _align(arrays["trusted_natural_logits"].astype(float), qids, mappings)
    lesioned = _align(arrays["lesioned_logits"].astype(float), qids, mappings)
    mapping_rows = [mappings[qid] for qid in qids]
    natural_choice = displayed_argmax_to_semantic_indices(
        arrays["trusted_natural_logits"], mapping_rows
    )
    lesioned_choice = displayed_argmax_to_semantic_indices(
        arrays["lesioned_logits"], mapping_rows
    )
    natural_metrics = [
        _metrics(natural[ci], w1i, w2i, natural_choice[ci]) for ci in range(2)
    ]
    lesioned_metrics = [
        [
            _metrics(lesioned[ci, si], w1i, w2i, lesioned_choice[ci, si])
            for si in range(len(SCENARIOS))
        ]
        for ci in range(2)
    ]

    masks = {
        "all": np.ones(len(qids), dtype=bool),
        "conflict_W1_not_equal_W2": conflict,
        "no_conflict_W1_equal_W2": ~conflict,
        "conflict_W1_A": conflict & (w1 == "A"),
        "conflict_W1_B_to_D": conflict & (w1 != "A"),
        "no_conflict_W1_A": (~conflict) & (w1 == "A"),
        "no_conflict_W1_B_to_D": (~conflict) & (w1 != "A"),
        "discovery_conflict": discovery_mask & conflict,
        "confirmation_conflict": (~discovery_mask) & conflict,
        "discovery_no_conflict": discovery_mask & (~conflict),
        "confirmation_no_conflict": (~discovery_mask) & (~conflict),
    }
    metric_names = tuple(natural_metrics[0])
    summary: dict[str, object] = {
        "definitions": {
            "W1": "Semantic answer selected on the original first-presentation Baseline.",
            "W2": "Semantic answer selected by a fresh Baseline under the remapped second presentation.",
            "effect": "Lesioned minus natural within the same condition and exact four-question cohort.",
            "condition_specific_difference": "Evaluation lesion effect minus Matched-Neutral lesion effect.",
            "gla_write": "Only the action-ending period's GLA write is removed; accumulated pre-period memory is preserved.",
            "attention_read": "Only later ordinary-attention reads from the action-ending period are blocked.",
            "joint": "Both source-specific routes are removed.",
        },
        "validation": {
            "questions": len(qids),
            "conflict": int(conflict.sum()),
            "no_conflict": int((~conflict).sum()),
            "max_abs_same_batch_natural_minus_trusted": max_error,
            "preserve_source_output": preserve_source_output,
        },
        "subsets": {},
    }

    seed = 70000
    for subset_index, (subset_name, mask) in enumerate(masks.items()):
        if not np.any(mask):
            continue
        labels = w1[mask]
        cell: dict[str, object] = {"n": int(mask.sum()), "scenarios": {}}
        for si, scenario in enumerate(SCENARIOS):
            scenario_cell: dict[str, object] = {name: {} for name in CONDITION_NAMES}
            scenario_cell["Evaluation_minus_Matched_Neutral_effect"] = {}
            for mi, metric in enumerate(metric_names):
                effects = []
                for ci, condition in enumerate(CONDITION_NAMES):
                    effect = lesioned_metrics[ci][si][metric] - natural_metrics[ci][metric]
                    effects.append(effect)
                    scenario_cell[condition][metric] = _interval(
                        effect[mask], labels, seed + subset_index * 1000 + si * 100 + mi * 3 + ci
                    )
                scenario_cell["Evaluation_minus_Matched_Neutral_effect"][metric] = _interval(
                    (effects[0] - effects[1])[mask],
                    labels,
                    seed + subset_index * 1000 + si * 100 + mi * 3 + 2,
                )
            cell["scenarios"][scenario] = scenario_cell
        summary["subsets"][subset_name] = cell

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # One presentation figure: within-condition effects first; the interaction
    # remains available in the machine-readable summary and report.
    panels = (
        ("conflict_W1_not_equal_W2", "w1_selection", "A  Conflict: W1 selection", 100.0, "percentage points"),
        ("conflict_W1_not_equal_W2", "w1_minus_w2_margin", "B  Conflict: W1−W2 margin", 1.0, "logits"),
        ("no_conflict_W1_equal_W2", "w1_selection", "C  No conflict: W1 selection", 100.0, "percentage points"),
        ("conflict_W1_not_equal_W2", "entropy_bits", "D  Conflict: A−D entropy", 1.0, "bits"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(SCENARIOS))
    offsets = (-0.12, 0.12)
    colors = ("#2f8df3", "#f07f32")
    labels = ("Evaluation", "Matched Neutral")
    for ax, (subset, metric, title, scale, ylabel) in zip(axes.flat, panels):
        for ci, condition in enumerate(CONDITION_NAMES):
            rows = [summary["subsets"][subset]["scenarios"][scenario][condition][metric] for scenario in SCENARIOS]
            means = np.asarray([row["mean"] for row in rows]) * scale
            lows = means - np.asarray([row["ci_low"] for row in rows]) * scale
            highs = np.asarray([row["ci_high"] for row in rows]) * scale - means
            ax.errorbar(
                x + offsets[ci], means, yerr=np.vstack([lows, highs]),
                fmt="o", markersize=7, capsize=5, linewidth=2,
                color=colors[ci], label=labels[ci],
            )
        ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set_xticks(x, ("GLA write", "Attention read", "Joint"))
        ax.set_ylabel(f"Lesioned − natural ({ylabel})")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("What does the action-ending period itself add?", fontsize=17, fontweight="bold")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    conflict_cell = summary["subsets"]["conflict_W1_not_equal_W2"]
    no_conflict_cell = summary["subsets"]["no_conflict_W1_equal_W2"]
    lines = [
        "# Action-ending-period source lesion",
        "",
        "## Question",
        "",
        "Does the period ending `Choose the answer again.` add causal information beyond the state already present after the evaluation clause? The intervention preserves all pre-period state and removes only this period's GLA write, later ordinary-attention reads, or both.",
        "",
        "An effect is always **lesioned minus natural within the named condition**. Percentage-point values are changes in trial-level answer selection, not changes relative to the other condition.",
        "",
        "## Conflict trials (W1 != W2)",
        "",
        "| Route removed | Condition | W1 selection | W1−W2 margin | Entropy |",
        "|---|---|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for condition in CONDITION_NAMES:
            rows = conflict_cell["scenarios"][scenario][condition]
            lines.append(
                f"| `{scenario}` | {condition} | {_fmt(rows['w1_selection'], 100, 1)} pp | "
                f"{_fmt(rows['w1_minus_w2_margin'], 1, 3)} | {_fmt(rows['entropy_bits'], 1, 3)} bits |"
            )
    lines += [
        "",
        "## No-conflict trials (W1 = W2)",
        "",
        "| Route removed | Condition | W1 selection | W1 centered advantage | Entropy |",
        "|---|---|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for condition in CONDITION_NAMES:
            rows = no_conflict_cell["scenarios"][scenario][condition]
            lines.append(
                f"| `{scenario}` | {condition} | {_fmt(rows['w1_selection'], 100, 1)} pp | "
                f"{_fmt(rows['w1_centered_advantage'], 1, 3)} | {_fmt(rows['entropy_bits'], 1, 3)} bits |"
            )
    disc_conflict = summary["subsets"]["discovery_conflict"]
    conf_conflict = summary["subsets"]["confirmation_conflict"]
    disc_no_conflict = summary["subsets"]["discovery_no_conflict"]
    conf_no_conflict = summary["subsets"]["confirmation_no_conflict"]
    lines += [
        "",
        "## Bottom line",
        "",
        "The corrected ordinary-attention intervention is not a null. With the "
        "source period's own residual output preserved, blocking later reads "
        "changes Neutral conflict-trial W1-minus-W2 margin by "
        f"{_fmt(conflict_cell['scenarios']['attention_read']['Matched Neutral']['w1_minus_w2_margin'], 1, 3)}. "
        "That direction replicates almost exactly on discovery and confirmation: "
        f"{_fmt(disc_conflict['scenarios']['attention_read']['Matched Neutral']['w1_minus_w2_margin'], 1, 3)} and "
        f"{_fmt(conf_conflict['scenarios']['attention_read']['Matched Neutral']['w1_minus_w2_margin'], 1, 3)}.",
        "",
        "In Game no-conflict trials, the same read blockade instead raises W1's "
        "centered advantage by "
        f"{_fmt(no_conflict_cell['scenarios']['attention_read']['Evaluation']['w1_centered_advantage'], 1, 3)}, "
        "again with the same sign on discovery and confirmation: "
        f"{_fmt(disc_no_conflict['scenarios']['attention_read']['Evaluation']['w1_centered_advantage'], 1, 3)} and "
        f"{_fmt(conf_no_conflict['scenarios']['attention_read']['Evaluation']['w1_centered_advantage'], 1, 3)}. "
        "The action-ending period therefore is genuinely read by later ordinary "
        "attention, with task- and conflict-dependent consequences.",
        "",
        "The effects remain modest: no pooled W1-selection change excludes zero, "
        "and the joint lesion does not reproduce the main Game-versus-Neutral "
        "behavioral gap. The supported conclusion is a small causal downstream "
        "source, not a dominant policy bottleneck. The former claim of exactly "
        "zero ordinary-attention impact was an instrument artifact.",
    ]
    lines += [
        "",
        "## Validation",
        "",
        f"- 500 questions: {int(conflict.sum())} conflict and {int((~conflict).sum())} no-conflict.",
        f"- Maximum same-batch natural deviation from trusted logits: {max_error:.6f}.",
        f"- Source-token local output preservation: `{preserve_source_output}` (from `run_metadata.json`).",
        "- Exact historical batches of four and SDPA execution were preserved.",
        f"- Canonical figure: `{args.figure}`.",
        "- All condition-specific differences, letter strata, and frozen discovery/confirmation intervals are in `summary.json`.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
