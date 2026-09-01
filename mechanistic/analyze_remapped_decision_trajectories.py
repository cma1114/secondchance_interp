from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LETTERS = "ABCD"
COLORS = {"W1": "#2589f5", "W2": "#ef7d32", "Other two": "#53bd73"}


def _intervals(values: np.ndarray, strata: np.ndarray, seed: int = 20260813):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    boot = np.empty((5000, values.shape[1]), dtype=float)
    for draw in range(len(boot)):
        picked = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        boot[draw] = values[picked].mean(axis=0)
    return values.mean(axis=0), *np.quantile(boot, (0.025, 0.975), axis=0)


def analyze(scores_path: Path, original_path: Path, remapped_path: Path,
            plan_path: Path, output: Path, figure: Path) -> None:
    arrays = np.load(scores_path)
    qids = arrays["question_ids"].tolist()
    original = json.loads(original_path.read_text())["results"]
    remapped = json.loads(remapped_path.read_text())["results"]
    plan = {row["question_id"]: row for row in json.loads(plan_path.read_text())["rows"]}
    w1 = np.asarray([LETTERS.index(original[qid]["answer"]) for qid in qids])
    w2 = np.asarray([LETTERS.index(remapped[qid]["answer_original_content"]) for qid in qids])
    conflict = w1 != w2
    selected = np.flatnonzero(conflict)
    strata = w1[conflict]
    layers = np.arange(1, 65)

    summary = {"definitions": {
        "W1": "semantic option selected by Baseline under the original first-presentation mapping",
        "W2": "semantic option selected by a fresh Baseline solution of the remapped second presentation",
        "conflict": "questions where W1 differs from W2",
        "score": "aggregated A-D token evidence after JLens or ordinary logit-lens decoding, centered across the four displayed answer letters within question, condition, and readout",
    }, "n_total": len(qids), "n_conflict": int(conflict.sum()), "lenses": {}}

    figure.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for row_index, (key, label) in enumerate([
        ("jlens_scores", "JLens"), ("logit_lens_scores", "Logit lens")
    ]):
        raw = arrays[key].astype(float)
        # Use the saved live model logits at the natural final readout. Earlier
        # readouts are decoded from float16 residual caches; substituting the
        # live endpoint makes L64 exactly identical to the behavioral analysis.
        raw[:, :, -1] = arrays["direct_logits"].astype(float)
        # displayed-letter coordinates -> original semantic-content coordinates
        aligned = np.empty_like(raw)
        for qi, qid in enumerate(qids):
            for content in range(4):
                new_letter = LETTERS.index(plan[qid]["original_to_new"][LETTERS[content]])
                aligned[:, qi, :, content] = raw[:, qi, :, new_letter]
        centered = aligned - aligned.mean(axis=-1, keepdims=True)
        q = selected
        game_w1 = centered[0, q, :, w1[q]]
        neutral_w1 = centered[1, q, :, w1[q]]
        w1_diff = game_w1 - neutral_w1
        game_w2 = centered[0, q, :, w2[q]]
        neutral_w2 = centered[1, q, :, w2[q]]
        w2_diff = game_w2 - neutral_w2
        other_diff = np.empty_like(w1_diff)
        for ri, qi in enumerate(q):
            others = [index for index in range(4) if index not in (w1[qi], w2[qi])]
            other_diff[ri] = (
                centered[0, qi, :, others].mean(axis=0)
                - centered[1, qi, :, others].mean(axis=0)
            )

        ax = axes[row_index, 0]
        for name, values, color in [
            ("Game", game_w1, COLORS["W1"]),
            ("Neutral", neutral_w1, COLORS["W2"]),
        ]:
            mean, low, high = _intervals(values, strata, 20260813 + row_index)
            ax.plot(layers, mean, lw=2.2, color=color, label=name)
            ax.fill_between(layers, low, high, color=color, alpha=.18)
        ax.axhline(0, color="#777", ls="--", lw=1)
        ax.set_title(f"{chr(65 + row_index * 2)}  {label}: evidence for W1")
        ax.set_ylabel("Centered decoded evidence (logit units)")
        ax.legend(frameon=False)

        ax = axes[row_index, 1]
        lens_summary = {}
        for name, values in [("W1", w1_diff), ("W2", w2_diff), ("Other two", other_diff)]:
            mean, low, high = _intervals(values, strata, 20260823 + row_index)
            ax.plot(layers, mean, lw=2.2, color=COLORS[name], label=name)
            ax.fill_between(layers, low, high, color=COLORS[name], alpha=.18)
            lens_summary[name] = {
                "l64_mean": float(mean[-1]), "l64_ci_low": float(low[-1]),
                "l64_ci_high": float(high[-1]),
                "first_readout_pointwise_ci_excludes_zero": int(
                    next((layer for layer, lo, hi in zip(layers, low, high)
                          if lo > 0 or hi < 0), 0)
                ),
            }
        ax.axhline(0, color="#777", ls="--", lw=1)
        ax.set_title(f"{chr(66 + row_index * 2)}  {label}: Game minus Neutral")
        ax.set_ylabel("Paired centered-evidence difference")
        ax.legend(frameon=False)
        summary["lenses"][label] = lens_summary

    for ax in axes[-1]:
        ax.set_xlabel("Residual readout")
    for ax in axes.ravel():
        ax.set_xlim(1, 64)
        ax.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(
        "When does Game reduce evidence for the first-pass semantic winner?\n"
        f"SimpleMC remapped second presentation; W1 ≠ W2 trials (n={len(selected)}); pointwise 95% bootstrap CIs",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    j = summary["lenses"]["JLens"]
    report = f"""# Layerwise course of W1 suppression after option remapping

## Definitions

- **W1**: the semantic option selected by Baseline on the original first presentation.
- **W2**: the semantic option selected when Baseline solves the remapped second presentation afresh.
- **Conflict trial**: W1 and W2 are different semantic contents (n={len(selected)}).
- **Centered evidence**: the decoded score for an option's currently displayed A-D token minus the mean decoded score across all four answer tokens for that same question, condition, and layer.
- **Game minus Neutral**: a paired within-question difference. A negative W1 value means Game represents W1 less strongly than Neutral does.

The plot uses both JLens and ordinary logit lens at the final decision position. Pointwise 95% confidence intervals bootstrap questions within original-W1 letter strata.

## Final readout check

At readout 64, JLens and logit lens coincide with the model's natural output readout. The paired Game-minus-Neutral centered-evidence differences are:

- W1: {j['W1']['l64_mean']:+.3f} [{j['W1']['l64_ci_low']:+.3f}, {j['W1']['l64_ci_high']:+.3f}] logits.
- W2: {j['W2']['l64_mean']:+.3f} [{j['W2']['l64_ci_low']:+.3f}, {j['W2']['l64_ci_high']:+.3f}] logits.
- Mean of the other two contents: {j['Other two']['l64_mean']:+.3f} [{j['Other two']['l64_ci_low']:+.3f}, {j['Other two']['l64_ci_high']:+.3f}] logits.

The figure is `{figure}`.
"""
    (output / "REPORT.md").write_text(report)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--original-baseline", type=Path, required=True)
    p.add_argument("--remapped-baseline", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)
    a = p.parse_args()
    analyze(a.scores, a.original_baseline, a.remapped_baseline, a.plan, a.output, a.figure)


if __name__ == "__main__":
    main()
