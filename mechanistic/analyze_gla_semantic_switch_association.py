from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm


LETTERS = ("A", "B", "C", "D")
LENSES = ("J-lens", "R-lens")
SPECS = (
    (42, "evaluation"),
    (43, "replacement"),
    (47, "retry"),
)
METRICS = ("Evaluation GLA change", "Contextual GLA change")


def _load_results(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _result_rows(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text())["results"]


def _bootstrap_diff(x: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    observed = float(x[y].mean() - x[~y].mean())
    draws = np.empty(5000)
    a, b = np.flatnonzero(y), np.flatnonzero(~y)
    for i in range(len(draws)):
        draws[i] = x[rng.choice(a, len(a), replace=True)].mean() - x[
            rng.choice(b, len(b), replace=True)
        ].mean()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return observed, float(lo), float(hi)


def _fit_adjusted(
    y: np.ndarray,
    x: np.ndarray,
    margin: np.ndarray,
    neutral: np.ndarray | None = None,
) -> dict:
    xz = (x - x.mean()) / x.std(ddof=0)
    mz = (margin - margin.mean()) / margin.std(ddof=0)
    columns = [xz, mz]
    if neutral is not None:
        columns.append(neutral.astype(float))
    design = sm.add_constant(np.column_stack(columns))
    try:
        fit = sm.Logit(y.astype(float), design).fit(disp=False, maxiter=500)
        beta = float(fit.params[1])
        lo, hi = fit.conf_int()[1]
        return {
            "odds_ratio_per_sd": float(np.exp(beta)),
            "ci95": [float(np.exp(lo)), float(np.exp(hi))],
            "p_value": float(fit.pvalues[1]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def analyze(
    results_path: Path,
    metadata_path: Path,
    plan_path: Path,
    evaluation_path: Path,
    neutral_path: Path,
    remapped_baseline_path: Path,
    discovery_plan_path: Path,
    confirmation_plan_path: Path,
    output_dir: Path,
    figure_path: Path,
) -> None:
    arrays = _load_results(results_path)
    metadata = json.loads(metadata_path.read_text())
    plan_rows = {row["question_id"]: row for row in json.loads(plan_path.read_text())["rows"]}
    evaluation = _result_rows(evaluation_path)
    neutral = _result_rows(neutral_path)
    baseline = _result_rows(remapped_baseline_path)
    qids = arrays["question_ids"].astype(str).tolist()
    blocks = arrays["blocks_one_based"].astype(int).tolist()
    group_names = list(metadata["semantic_groups"])

    discovery_ids = set(json.loads(discovery_plan_path.read_text())["question_ids"])
    confirmation_ids = set(json.loads(confirmation_plan_path.read_text())["question_ids"])
    if discovery_ids | confirmation_ids != set(qids) or discovery_ids & confirmation_ids:
        raise ValueError("Historical discovery/confirmation split does not partition questions")

    natural_error = float(metadata["max_abs_natural_ad_logit_error_vs_trusted"])
    if natural_error != 0.0:
        raise ValueError("Natural outputs did not reproduce exactly")

    switch = np.zeros(len(qids), dtype=bool)
    neutral_switch = np.zeros(len(qids), dtype=bool)
    baseline_margin = np.zeros(len(qids), dtype=float)
    for i, qid in enumerate(qids):
        w1_new = plan_rows[qid]["baseline_content_new_letter"]
        w1_index = LETTERS.index(w1_new)
        switch[i] = evaluation[qid]["aggregated_ad_answer_new_letter"] != w1_new
        neutral_switch[i] = neutral[qid]["aggregated_ad_answer_new_letter"] != w1_new
        logits = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=float)
        baseline_margin[i] = logits[w1_index] - np.max(np.delete(logits, w1_index))

    split_masks = {
        "Discovery": np.asarray([qid in discovery_ids for qid in qids]),
        "Confirmation": np.asarray([qid in confirmation_ids for qid in qids]),
        "All": np.ones(len(qids), dtype=bool),
    }

    rows = []
    detailed = {}
    scores = arrays["semantic_scores"]
    for lens_index, lens in enumerate(LENSES):
        for block, group in SPECS:
            b = blocks.index(block)
            g = group_names.index(group)
            eval_change = scores[lens_index, 0, :, b, 1, g] - scores[
                lens_index, 0, :, b, 0, g
            ]
            neutral_change = scores[lens_index, 1, :, b, 1, g] - scores[
                lens_index, 1, :, b, 0, g
            ]
            metric_values = {
                "Evaluation GLA change": eval_change,
                "Contextual GLA change": eval_change - neutral_change,
            }
            for metric, values in metric_values.items():
                for split, mask in split_masks.items():
                    x, y = values[mask], switch[mask]
                    diff, lo, hi = _bootstrap_diff(x, y, seed=42 + block + lens_index)
                    auc = float(roc_auc_score(y, x))
                    baseline_adjusted = _fit_adjusted(
                        y,
                        x,
                        baseline_margin[mask],
                    )
                    baseline_neutral_adjusted = _fit_adjusted(
                        y,
                        x,
                        baseline_margin[mask],
                        neutral_switch[mask],
                    )
                    row = {
                        "lens": lens,
                        "block": block,
                        "semantic_family": group,
                        "metric": metric,
                        "split": split,
                        "n": int(mask.sum()),
                        "switch_n": int(y.sum()),
                        "switch_rate": float(y.mean()),
                        "mean_switched": float(x[y].mean()),
                        "mean_repeated": float(x[~y].mean()),
                        "switched_minus_repeated": diff,
                        "difference_ci_low": lo,
                        "difference_ci_high": hi,
                        "auc": auc,
                        "baseline_adjusted_odds_ratio_per_sd": baseline_adjusted.get(
                            "odds_ratio_per_sd", np.nan
                        ),
                        "baseline_adjusted_ci95": baseline_adjusted.get(
                            "ci95", [np.nan, np.nan]
                        ),
                        "baseline_adjusted_p_value": baseline_adjusted.get("p_value", np.nan),
                        "baseline_neutral_adjusted_odds_ratio_per_sd": (
                            baseline_neutral_adjusted.get("odds_ratio_per_sd", np.nan)
                        ),
                        "baseline_neutral_adjusted_ci95": baseline_neutral_adjusted.get(
                            "ci95", [np.nan, np.nan]
                        ),
                        "baseline_neutral_adjusted_p_value": baseline_neutral_adjusted.get(
                            "p_value", np.nan
                        ),
                    }
                    rows.append(row)
                    detailed[f"{lens}|{block}|{group}|{metric}|{split}"] = row

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "associations.csv", index=False)

    # One canonical figure: independent point estimates, never connected as a series.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/secondchance-mpl")
    import matplotlib.pyplot as plt

    primary = frame[
        (frame["metric"] == "Contextual GLA change")
        & (frame["split"].isin(["Discovery", "Confirmation"]))
    ].copy()
    labels = ["L42 incorrect/wrong", "L43 replace/instead", "L47 again/another"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for lens_index, lens in enumerate(LENSES):
        ax = axes[lens_index]
        for split_index, split in enumerate(("Discovery", "Confirmation")):
            part = primary[(primary["lens"] == lens) & (primary["split"] == split)].sort_values(
                "block"
            )
            y = np.arange(3) + (split_index - 0.5) * 0.18
            x = part["switched_minus_repeated"].to_numpy()
            lo = part["difference_ci_low"].to_numpy()
            hi = part["difference_ci_high"].to_numpy()
            ax.errorbar(
                x,
                y,
                xerr=np.vstack([x - lo, hi - x]),
                fmt="o",
                capsize=4,
                label=split,
            )
        ax.axvline(0, color="0.35", lw=1, ls="--")
        ax.set_yticks(np.arange(3), labels)
        ax.set_title(lens)
        ax.set_xlabel("Semantic score: switched minus repeated\n(95% bootstrap CI)")
        ax.grid(axis="x", alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Do stronger revision-semantic GLA changes predict W1 avoidance?")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "definition": (
            "Switching means avoiding W1 semantic content under Evaluation. Positive "
            "switched-minus-repeated differences and odds ratios above one mean stronger "
            "decoded revision semantics predict switching. The primary adjusted model controls "
            "the fresh remapped-Baseline W1 margin; a second robustness model additionally "
            "controls Matched Neutral W1 avoidance."
        ),
        "natural_reproduction_error": natural_error,
        "switch_rate_all": float(switch.mean()),
        "neutral_switch_rate_all": float(neutral_switch.mean()),
        "rows": detailed,
        "figure": str(figure_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    conf = frame[
        (frame["metric"] == "Contextual GLA change") & (frame["split"] == "Confirmation")
    ].sort_values(["lens", "block"])
    lines = [
        "# Revision-semantic strength and switching",
        "",
        "## Bottom line",
        "",
        "Only the block-47 retry/alternative-action family (`again`, `retry`, `another`, "
        "`second`, and related tokens) shows a reproducible unadjusted relationship: questions "
        "with a stronger Evaluation-minus-Neutral GLA-47 retry write are more likely to avoid "
        "W1. The effect appears in both historical splits and both lenses, but is modest and its "
        "confidence interval includes no association after controlling for the fresh remapped-"
        "Baseline W1 margin. Block-42 incorrectness and block-43 replacement strength do not "
        "reliably distinguish switched from repeated trials.",
        "",
        "## Frozen confirmation estimates",
        "",
        "The score is the question-level contextual GLA change: Evaluation's after-minus-before "
        "semantic-family score minus Matched Neutral's corresponding change. Positive values "
        "mean the GLA expresses more of that semantic family in Evaluation.",
        "",
        "| Lens | Readout | Switched minus repeated [95% CI] | AUC | Baseline-adjusted OR per SD [95% CI] |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in conf.iterrows():
        ci = row.get("baseline_adjusted_ci95", [np.nan, np.nan])
        lines.append(
            f"| {row['lens']} | L{int(row['block'])} {row['semantic_family']} | "
            f"{row['switched_minus_repeated']:+.3f} "
            f"[{row['difference_ci_low']:+.3f}, {row['difference_ci_high']:+.3f}] | "
            f"{row['auc']:.3f} | {row.get('baseline_adjusted_odds_ratio_per_sd', np.nan):.2f} "
            f"[{ci[0]:.2f}, {ci[1]:.2f}] |"
        )
    lines += [
        "",
        "![Semantic strength association](" + str(figure_path.resolve()) + ")",
        "",
        "## Interpretation constraints",
        "",
        "The token families and blocks were frozen from the aggregate vocabulary explorer before "
        "this question-level association was inspected. Discovery and confirmation use the "
        "historical 251/249 split. This remains observational: a positive association can show "
        "that the readout tracks revision behavior, but not that the English-token direction itself "
        "causes switching.",
        "",
        "## Artifacts",
        "",
        "- [Tidy association table](associations.csv)",
        "- [Numerical summary](summary.json)",
        "- [Question-level semantic scores](../run/results.npz)",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--remapping-plan", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--neutral", type=Path, required=True)
    p.add_argument("--remapped-baseline", type=Path, required=True)
    p.add_argument("--discovery-plan", type=Path, required=True)
    p.add_argument("--confirmation-plan", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)
    a = p.parse_args()
    analyze(
        a.results,
        a.metadata,
        a.remapping_plan,
        a.evaluation,
        a.neutral,
        a.remapped_baseline,
        a.discovery_plan,
        a.confirmation_plan,
        a.output_dir,
        a.figure,
    )


if __name__ == "__main__":
    main()
