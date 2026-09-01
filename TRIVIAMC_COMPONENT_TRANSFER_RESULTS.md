# Qwen3.6-27B TriviaMC component transfer: results

## Bottom line

The eight components selected on SimpleMC transfer strongly to all 500 frozen
TriviaMC questions. No TriviaMC outcome was used to select or redefine them.

Replacing the eight Game component outputs with their paired same-question
Neutral outputs removed:

- 82.7% of the Game-Neutral A-D entropy gap;
- 72.1% of the centered A-D logit-spread gap;
- 70.5% of the original-winner-advantage gap;
- 78.1% of excess Game switching.

The reciprocal Game-into-Neutral patch recreated 61.1%, 77.9%, 73.6%, and
52.1% of those respective gaps. Every all-eight effect has a paired-bootstrap
95% interval excluding zero in both directions.

This establishes that the SimpleMC localization was not merely a
dataset-specific causal fit. The same late components mediate most of the
model's Game-specific A-D flattening, loss of original-winner advantage, and
answer switching on a second dataset.

## Preregistered design

- Model: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Data: all 500 questions in the frozen Qwen3.6-27B TriviaMC manifest.
- Targets: the exact eight SimpleMC-selected components.
- Groups: the exact SimpleMC flattening, switching, and union groups.
- Intervention: paired same-question final-position component-output
  replacement in both Neutral-into-Game and Game-into-Neutral directions.
- Outcomes: A-D entropy, centered A-D spread, Baseline-winner advantage, and
  switching relative to the exact self-hosted Baseline A-D argmax.
- Aggregations: dataset weighted, equal-letter macro, and separate original
  A/B/C/D results.

The smoke test completed 4/4 questions with the exact commit, 100/100 expected
shards, valid Baseline metadata, and no target-condition prompt-hash
mismatches. The full transfer completed 500/500 questions and 12,500/12,500
expected shards.

## Natural self-hosted behavior

The direct checkpoint run closely reproduces the earlier OpenRouter behavior:

| Measure | Self-hosted | OpenRouter |
|---|---:|---:|
| Baseline accuracy | 63.6% | 63.4% |
| Game switching | 29.6% | 28.0% |
| Neutral switching | 15.0% | 15.6% |
| Game-Neutral normalized A-D entropy gap | 0.317 bits | 0.348 bits |

Self-hosted and OpenRouter Baseline answers agree on 459/500 questions
(91.8%). Thus the transfer is being evaluated on a faithful behavioral
reproduction, not a materially different self-hosted regime.

In the self-hosted run, the natural Game-minus-Neutral gaps were:

- entropy: +0.2199 nats;
- centered A-D spread: -1.1666 logits;
- original-winner advantage: -3.1309 logits;
- switching: +14.60 percentage points.

## Held-out transfer effects

Effects below are patch minus the natural target condition. For
Neutral-into-Game, negative entropy, positive spread, positive winner
advantage, and negative switching remove Game behavior.

| Neutral-into-Game patch | Entropy | Spread | Winner advantage | Switch |
|---|---:|---:|---:|---:|
| MLP 63 | -0.0918 | +0.5617 | +1.0257 | -0.8 pp |
| Mixer 62 | -0.0141 | -0.0700 | +0.1961 | -7.6 pp |
| Flattening group | -0.1673 | +0.7925 | +1.6501 | -3.8 pp |
| Switching group | -0.0736 | +0.1323 | +0.6918 | -11.2 pp |
| All eight | -0.1819 | +0.8416 | +2.2079 | -11.4 pp |

For the reciprocal Game-into-Neutral sufficiency test, the Game-like signs are
positive entropy, negative spread, negative winner advantage, and positive
switching:

| Game-into-Neutral patch | Entropy | Spread | Winner advantage | Switch |
|---|---:|---:|---:|---:|
| MLP 63 | +0.0149 | -0.5353 | -1.1772 | -0.2 pp |
| Mixer 62 | -0.0006 | +0.0583 | -0.0861 | +5.2 pp |
| Flattening group | +0.0892 | -0.8485 | -1.7896 | -2.6 pp |
| Switching group | +0.0469 | -0.2710 | -0.9780 | +7.4 pp |
| All eight | +0.1343 | -0.9088 | -2.3053 | +7.6 pp |

All-eight 95% paired-bootstrap intervals are:

- entropy: [-0.2057, -0.1593];
- spread: [+0.7781, +0.9062];
- winner advantage: [+2.0575, +2.3610];
- switching: [-14.6, -8.2] percentage points.

The reverse all-eight patch changes entropy by +0.1343 [0.1134, 0.1561],
spread by -0.9088 [-0.9786, -0.8398], winner advantage by -2.3053
[-2.4751, -2.1416], and switching by +7.6 [4.8, 10.4] percentage points.

## Cross-dataset comparison

| All-eight fraction mediated | SimpleMC removal | TriviaMC removal | SimpleMC reverse | TriviaMC reverse |
|---|---:|---:|---:|---:|
| Entropy | 89.8% | 82.7% | 65.3% | 61.1% |
| Spread | 96.4% | 72.1% | 90.3% | 77.9% |
| Winner advantage | 78.1% | 70.5% | 74.2% | 73.6% |
| Switching | 68.1% | 78.1% | 70.8% | 52.1% |

The individual-component structure also transfers. Across the eight fixed
components, the descriptive SimpleMC-TriviaMC correlations of
Neutral-into-Game effects are 0.86 for entropy, 0.94 for spread, and 0.85 for
switching. Winner-advantage effects correlate weakly because MLP 63 restores
considerably more winner advantage on TriviaMC than on SimpleMC.

## Functional dissociation

The two group profiles remain differentiated, although not perfectly pure:

- The flattening group removes 76.1% of the entropy gap and 67.9% of the
  spread gap, but only 26.0% of excess switching.
- The switching group removes 76.7% of excess switching, but only 33.5% of the
  entropy gap and 11.3% of the spread gap.
- MLP 63 alone removes 41.7% of the entropy gap and 48.2% of the spread gap
  while changing switching by only 0.8 percentage points.
- Mixer 62 alone removes 52.1% of excess switching while failing to sharpen
  spread, again distinguishing its causal role from literal flattening.

The groups overlap at Mixers 50, 51, and 60, so this is a replicated
functional specialization, not two anatomically disjoint circuits. TriviaMC
also shows that the switching group has some compressive effect; the especially
clean SimpleMC dissociation was stronger than the cross-dataset invariant.

## Equal-letter and per-letter results

Equal weighting of original A, B, C, and D leaves the all-eight transfer
nearly unchanged. The Neutral-into-Game patch removes 81.4% of the entropy
gap, 70.8% of the spread gap, 69.1% of the winner-advantage gap, and 75.9% of
the switch gap. The reverse equal-letter fractions are 58.5%, 75.8%, 72.2%,
and 49.9%.

The flattening signs transfer for every original letter: the all-eight patch
lowers entropy, raises spread, and restores winner advantage for A, B, C, and
D. The switching effect remains letter dependent:

| Original Baseline winner | Natural Game-Neutral switch gap | All-eight removal effect |
|---|---:|---:|
| A | +36.0 pp | -30.5 pp |
| B | +1.0 pp | -1.0 pp |
| C | 0.0 pp | +2.7 pp |
| D | +10.8 pp | -7.5 pp |

TriviaMC therefore replicates the SimpleMC letter signature: most
Game-specific switching occurs when A was the original winner; D shows a
smaller transferable effect; B has little natural gap; and C has no natural
switch gap for the circuit to mediate and responds in the opposite direction
under intervention. This points toward a token/letter-specific implementation
of switching rather than an abstract letter-invariant operation.

## Interpretation

The strongest defensible mechanistic claim is:

> Across SimpleMC and TriviaMC, a small set of late final-position components
> causally mediates most of Qwen3.6-27B's incorrect-feedback-specific
> flattening and answer switching. MLP 63 is consistently specialized toward
> flattening, while Mixer 62 and the four-mixer switching group are
> consistently specialized toward changing the selected answer. The switching
> implementation is strongly answer-letter dependent.

This is now a cross-dataset causal mechanism, not merely a logit-level
description. It still does not explain what semantic or upstream signal drives
MLP 63 and Mixer 62, which is the natural next interpretability question.

## Artifacts and compute

- Full report:
  `outputs/causal/qwen36_27b_triviamc_component_transfer/transfer/analysis/COMPONENT_CAUSAL_REPORT.md`
- Dataset and letter-macro effects:
  `outputs/causal/qwen36_27b_triviamc_component_transfer/transfer/analysis/component_causal_effects.csv`
- Per-letter effects:
  `outputs/causal/qwen36_27b_triviamc_component_transfer/transfer/analysis/component_causal_effects_by_letter.csv`
- Summary:
  `outputs/causal/qwen36_27b_triviamc_component_transfer/transfer/analysis/component_causal_summary.json`
- Frozen transfer plan:
  `outputs/causal/qwen36_27b_triviamc_component_transfer/plans/transfer_plan.json`
- Outcome and geometry figures:
  `outputs/causal/qwen36_27b_triviamc_component_transfer/transfer/analysis/causal_outcome_sweep.svg`
  and `causal_geometry_sweep.svg` in the same directory.

The measured account-credit reduction for this batch was $1.44. Vast instance
46566562 was stopped, not destroyed, after retrieval and validation.
