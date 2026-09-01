# Original winner versus freshly regenerated winner

## Definitions

- The **original winner** is the option content chosen by the standalone
  Baseline under the original mapping.
- The **remapped winner** is the content chosen by a fresh standalone Baseline
  that sees only the remapped question.
- The primary analysis uses questions where those contents differ. Positive
  original-minus-remapped suppression means Game preferentially suppresses the
  first-pass winner; negative means it preferentially suppresses the fresh
  remapped-presentation winner.

## Winner discordance

The independent remapped Baseline selected a different content on
**273/500 (54.6%)**
questions. Its mean A–D probability mass was
99.99%; aggregated
A–D and unrestricted decisions differed on
16
questions.

## Game-specific target suppression on discordant questions

| Required fresh remapped-winner margin | N | Original-winner suppression | Remapped-winner suppression | Original minus remapped |
|---|---:|---:|---:|---:|
| All discordant | 273 | +0.476 [+0.398, +0.554] | +0.065 [-0.009, +0.140] | +0.410 [+0.285, +0.536] |
| ≥0.25 logits | 235 | +0.492 [+0.404, +0.579] | +0.093 [+0.011, +0.178] | +0.399 [+0.254, +0.535] |
| ≥0.50 logits | 197 | +0.517 [+0.422, +0.613] | +0.126 [+0.039, +0.217] | +0.391 [+0.236, +0.547] |

Suppression is `Neutral logit - Game logit` for the named content. The final
column is the decisive comparison: positive supports retrieval of the original
winner; negative supports regeneration and suppression of the current winner.

## Which content is ultimately selected?

On the 273 discordant questions:

| Final choice | Game | Neutral |
|---|---:|---:|
| Original winner | 20.9% | 39.2% |
| Fresh remapped winner | 45.1% | 40.7% |
| Either other option | 34.1% | 20.1% |

Game-minus-Neutral avoidance of the original winner is
+18.3 [+12.8, +23.8] percentage points;
avoidance of the fresh remapped winner is
-4.4 [-10.6, +1.8] percentage points.

## Absolute-letter robustness check

The fresh remapped Baseline is strongly absolute-letter-biased: among discordant
trials, W2 occupied A/B/C/D on
186/20/44/23 trials, whereas remapped W1 occupied those letters on
12/91/87/83 trials.
This is not a novel A bias in the fresh run: the original and remapped Baselines
selected literal A on 240/500 and 260/500 questions, respectively. The
within-discordant imbalance arises because the derangement forces W1 away from
its original letter while a fresh answer is free to express the model's usual
letter preference.
Therefore, a post-specified robustness model uses all four logits per question
and controls both question and absolute answer letter. It estimates:

- original-winner suppression: +0.728 [+0.622, +0.837]
- fresh-remapped-winner suppression: +0.411 [+0.272, +0.555]
- original minus remapped: +0.317 [+0.151, +0.475]

The decisive contrast remains positive after this adjustment. This robustness
check was added after observing the fresh Baseline's letter imbalance and is not
the frozen primary analysis.

Defining the fresh remapped winner by the aggregated A–D logits rather than the
unrestricted top token gives N=276 discordant
trials and an original-minus-remapped suppression contrast of
+0.400 [+0.274, +0.524].

## Layerwise localization

A subsequent exact-run analysis traced the centered evidence for W1 and W2 at
the final decision position through all 64 residual readouts. Both JLens and
ordinary logit lens agree that the Game-specific reduction of W1 is not a
gradual early-layer divergence: it becomes visible around readouts 52--54,
makes its largest step at readout 56, and persists through the output. At the
exact final readout, Game minus Neutral is -0.476 logits for W1, -0.065 for W2,
and +0.270 for the mean of the other two semantic contents.

- [Layerwise report](../layerwise_decision_trajectories/analysis/REPORT.md)
- [Presentation figure](../../../../figures/qwen36_simplemc_remapped_w1_layerwise.png)
