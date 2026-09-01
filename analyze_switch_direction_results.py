#!/usr/bin/env python3
"""Bootstrap and document the clean switch-specific direction experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


ROOT = Path("outputs/mechanistic/qwen36_27b_simplemc_clean/analysis/switch_direction")


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    n1 = int(labels.sum()); n0 = len(labels) - n1
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [.025, .975])]


def bootstrap(npz_path: Path, repetitions: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    with np.load(npz_path, allow_pickle=False) as data:
        groups = data["groups"].astype(int)
        scores = data["cross_fitted_scores"][2].astype(float)
        compression = data["compression"].astype(float)
    neither = np.flatnonzero(groups == 0)
    game_only = np.flatnonzero(groups == 1)
    primary = np.r_[neither, game_only]
    labels = groups[primary] == 1
    layers = np.arange(47, 53)
    observed = {
        str(layer): auc(labels, scores[primary, layer]) for layer in layers
    }
    observed["47_52_mean"] = float(np.mean(list(observed.values())))
    late = np.nanmean(compression[:, 53:65], axis=1)
    final = compression[:, -1]
    late_difference = float(np.nanmean(late[game_only]) - np.nanmean(late[neither]))
    final_difference = float(np.nanmean(final[game_only]) - np.nanmean(final[neither]))

    auc_draws = {key: np.empty(repetitions) for key in observed}
    late_draws = np.empty(repetitions)
    final_draws = np.empty(repetitions)
    for draw in range(repetitions):
        sampled_neither = rng.choice(neither, len(neither), replace=True)
        sampled_game = rng.choice(game_only, len(game_only), replace=True)
        sampled = np.r_[sampled_neither, sampled_game]
        sampled_labels = groups[sampled] == 1
        layer_aucs = []
        for layer in layers:
            value = auc(sampled_labels, scores[sampled, layer])
            auc_draws[str(layer)][draw] = value
            layer_aucs.append(value)
        auc_draws["47_52_mean"][draw] = np.mean(layer_aucs)
        late_draws[draw] = np.nanmean(late[sampled_game]) - np.nanmean(late[sampled_neither])
        final_draws[draw] = np.nanmean(final[sampled_game]) - np.nanmean(final[sampled_neither])
    return {
        "bootstrap_repetitions": repetitions,
        "auc": {
            key: {"estimate": value, "ci95": interval(auc_draws[key])}
            for key, value in observed.items()
        },
        "compression_game_only_minus_neither": {
            "late_layers_53_64": {"estimate": late_difference, "ci95": interval(late_draws)},
            "final_layer": {"estimate": final_difference, "ci95": interval(final_draws)},
        },
    }


def main() -> None:
    summary = json.loads((ROOT / "switch_direction_summary.json").read_text())
    uncertainty = bootstrap(ROOT / "switch_direction.npz")
    result = {"summary": summary, "uncertainty": uncertainty}
    (ROOT / "switch_direction_inference.json").write_text(json.dumps(result, indent=2) + "\n")

    rows = list(csv.DictReader((ROOT / "switch_direction_layers.csv").open()))
    selected = {
        int(row["layer"]): row
        for row in rows
        if row["variant"] == "answer_orthogonal" and int(row["layer"]) in (36, 44, 47, 48, 49, 50, 51, 52, 56, 60, 64)
    }
    auc50 = uncertainty["auc"]["50"]
    auc_cluster = uncertainty["auc"]["47_52_mean"]
    late = uncertainty["compression_game_only_minus_neither"]["late_layers_53_64"]
    final = uncertainty["compression_game_only_minus_neither"]["final_layer"]
    report = f"""# Clean switch-specific feedback-direction experiment

## Bottom line

There is a reproducible **late residual direction associated with Game-only switching**, but the present analysis does not identify it as a compression-control signal.

After nuisance adjustment and removal of the baseline answer-letter subspace, the direction distinguishes held-out Game-only-switch trials from neither-switch trials most strongly at readout 50: AUC {auc50['estimate']:.3f}, bootstrap 95% CI [{auc50['ci95'][0]:.3f}, {auc50['ci95'][1]:.3f}]. The mean held-out AUC across readouts 47–52 is {auc_cluster['estimate']:.3f} [{auc_cluster['ci95'][0]:.3f}, {auc_cluster['ci95'][1]:.3f}], so the result is a short late-layer band rather than a one-layer spike.

However, the readout-50 projection does not significantly predict the continuous amount of later compression (Spearman rho {float(selected[50]['spearman_with_late_compression']):.3f}, p={float(selected[50]['spearman_p']):.3f}). The largest observed correlation is only rho {float(selected[52]['spearman_with_late_compression']):.3f} at readout 52 (uncorrected p={float(selected[52]['spearman_p']):.3f}). Full-vocabulary unembedding of the direction is predominantly incoherent and does not reveal a clear error, retry, switching, or compression concept.

The evidence therefore supports a **late question-independent correlate of whether the Game changes the answer**, not yet a common signal shown to initiate compression.

## Design

- Model: Qwen3.6-27B, checkpoint `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Dataset: 500 SimpleMC questions.
- Neutral prompt: corrected empty setup text; no literal `None` and no leading newline.
- Existing clean baseline and Game activations were paired with 500 newly collected neutral activations.
- Behavioral groups from the self-hosted final A-D argmax: {summary['groups']['game_only']} Game-only switches, {summary['groups']['neither']} neither switches, {summary['groups']['both']} both switch, and {summary['groups']['neutral_only']} neutral-only switches.
- At each readout, the basic observation was the paired residual difference `Game - neutral` for the same question.
- The switch direction was the letter-balanced Game-only mean difference minus the neither-switch mean difference.
- Baseline margin, spread, entropy, correctness, and original winner letter were removed as nuisance variables. The baseline answer-letter centroid subspace was then projected out.
- Five question-held-out folds supplied all reported classification scores. Fold directions were estimated without the scored questions.

## Layerwise result

| Readout | Held-out AUC | Fold cosine | rho with later compression |
|---:|---:|---:|---:|
"""
    for layer in selected:
        row = selected[layer]
        report += (
            f"| {layer} | {float(row['heldout_auc_game_only_vs_neither']):.3f} | "
            f"{float(row['mean_fold_cosine']):.3f} | "
            f"{float(row['spearman_with_late_compression']):+.3f} |\n"
        )
    report += f"""

The switch association begins becoming appreciable around readouts 47–50, peaks at 50, and weakens again toward the final output. At readout 50, independently estimated fold directions have mean cosine {float(selected[50]['mean_fold_cosine']):.3f}. This is reasonably stable, but the AUC of approximately .60 is a modest effect.

For comparison, the unadjusted direction reaches AUC .674 at readout 47. Its reduction after nuisance and answer-subspace removal shows that much of the easily decoded switch/non-switch difference consists of baseline difficulty and answer-related geometry. The remaining .60-AUC signal is the relevant candidate question-independent component.

## Compression by behavioral group

Game-only trials have more Game-versus-neutral compression in the last twelve readouts than neither-switch trials: mean difference {late['estimate']:+.3f} [{late['ci95'][0]:+.3f}, {late['ci95'][1]:+.3f}] in the normalized projection coefficient. At the final readout the difference is {final['estimate']:+.3f} [{final['ci95'][0]:+.3f}, {final['ci95'][1]:+.3f}].

This group difference is expected under the behavioral hypothesis, but it is outcome-conditioned: the groups were defined using the final choices. More importantly, variation in the discovered residual direction does not explain variation in later compression within the held-out questions. Thus the direction and compression co-occur at the group level without a demonstrated trial-level mediation relationship.

## Logit lens of the direction

The vocabulary lens is mostly multilingual fragments, names, and formatting/code tokens through the network. Around the predictive band (47–52), no coherent semantic family appears. At the final readout, `Choices`, `respond`, and a response-related Chinese token appear among the positive top 15, but alongside unrelated tokens such as `Assignment`, `Providers`, and `_legacy`; the negative side is similarly heterogeneous.

This is substantially cleaner than the previous `None`-dominated feedback direction, but it is not positively interpretable as “compress,” “switch,” “wrong,” or “choose another.”

## Interpretation

The experiment finds something real but narrower than the proposed mechanism:

1. A common residual component associated with behaviorally consequential Game-only switches exists.
2. It appears late, around the same stage at which the answer representation is becoming output-aligned.
3. It is only moderately predictive after removing obvious confounds.
4. It does not convincingly predict subsequent compression and has no clear vocabulary-level interpretation.

The most defensible description is therefore **a late switch-associated state**, which could be part of the decision process or a consequence of it. It is not currently evidence that a common feedback signal is passed early and causes question-specific compression.

## Artifacts

- `switch_direction_layers.csv`: all layerwise estimates and sensitivities.
- `switch_direction.npz`: directions, cross-fitted scores, compression coefficients, and group labels.
- `switch_direction_inference.json`: bootstrap intervals.
- `switch_direction_lens/feedback_direction_top_tokens.csv`: top vocabulary associations at every readout.
- `switch_direction_lens/feedback_direction_vocab_logits.npz`: full 65 x 248,320 lens matrix.
"""
    (ROOT / "SWITCH_DIRECTION_REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
