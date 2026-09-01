# Qwen3.6-27B SimpleMC causal component sweep: results

## Bottom line

The experiment found a strong, held-out causal localization of the model's
Game-versus-Neutral behavior at the final prompt position.

Two partially overlapping late-component groups have different causal
profiles:

1. A **flattening group** controls most of the Game-specific increase in A-D
   entropy and reduction in A-D logit spread, with MLP 63 the dominant single
   component.
2. A **switching group** controls much of the Game-specific reduction in the
   original winner's advantage and increase in answer switching, while having
   almost no net effect on entropy or spread in the Neutral-into-Game removal
   test.

Jointly replacing the eight selected Game component outputs with their
same-question Neutral outputs removed 68% of the observed excess switching,
restored 78% of the original-winner-advantage gap, and removed 90-96% of the
two flattening gaps on 249 held-out questions. The reciprocal
Game-into-Neutral intervention recreated 65-90% of the four gaps. This is much
stronger evidence than the previous direction-steering and token-attention
experiments.

The main qualification is answer-letter heterogeneity. The flattening effect
is present for all four original letters, but the switching effect is driven
especially by original-A trials and reverses on original-C trials. The result
therefore supports a robust general flattening mechanism plus a causal but not
letter-invariant switching implementation.

## Design

- Model: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Data: frozen 500-question SimpleMC manifest and exact corrected prompts.
- Position: final prompt position used to predict the first answer token.
- Intervention: replace the complete mixer or MLP output at that position with
  the same component's output for the same question in the paired condition;
  allow all downstream computation to run normally.
- Discovery: all 128 components on a fixed 251-question split,
  Neutral-into-Game only.
- Confirmation: adaptively selected components on the untouched 249-question
  split, individually and jointly in both patch directions.
- Selection: top eight components overall by the larger eligible score for
  literal flattening or switching, with no family quota.

The complete run produced 32,630 discovery shards, 5,976 confirmation shards,
and 24 smoke-test shards. All expected shards are present locally.

## Selected components

The top-eight no-quota selection was:

| Component | Primary discovery family | Discovery score |
|---|---|---:|
| MLP 63 | flattening | 0.680 |
| Mixer 62 | switching | 0.341 |
| Mixer 63 | flattening | 0.268 |
| Mixer 50 | switching | 0.087 |
| Mixer 60 | flattening | 0.083 |
| Mixer 39 | flattening | 0.074 |
| Mixer 51 | switching | 0.073 |
| Mixer 59 | flattening | 0.065 |

Eligibility groups can overlap. The grouped confirmation patches were:

- Flattening: MLP 63 and Mixers 39, 50, 51, 59, 60, 63.
- Switching: Mixers 50, 51, 60, 62.
- Union: all eight components above.

## Held-out causal effects

On the confirmation split, the natural Game-minus-Neutral gaps were:

- entropy: +0.1586;
- centered A-D logit spread: -0.4365;
- original-winner advantage: -1.5752 logits;
- switch probability: +28.92 percentage points.

The table reports patch-minus-natural-target effects. For Neutral-into-Game,
negative entropy, positive spread, positive winner advantage, and negative
switching remove Game behavior.

| Neutral-into-Game patch | Entropy | Spread | Winner advantage | Switch |
|---|---:|---:|---:|---:|
| MLP 63 | -0.1395 | +0.3461 | -0.0720 | -0.40 pp |
| Mixer 62 | +0.0452 | -0.0971 | +0.3787 | -14.06 pp |
| Flattening group | -0.2096 | +0.4719 | +0.4709 | -0.40 pp |
| Switching group | +0.0010 | +0.0231 | +0.6821 | -19.68 pp |
| All eight | -0.1424 | +0.4206 | +1.2309 | -19.68 pp |

All-eight 95% paired-bootstrap intervals exclude zero for every primary
outcome: entropy [-0.1705, -0.1158], spread [+0.3538, +0.4871], winner
advantage [+1.0919, +1.3742], and switching [-25.30, -14.06] percentage
points.

The reverse all-eight Game-into-Neutral patch produced entropy +0.1035, spread
-0.3943, winner advantage -1.1696 logits, and switching +20.48 percentage
points. Its 95% intervals also exclude zero for all four outcomes. Thus the
joint result is bidirectional rather than merely a destructive ablation.

### Equal-letter analysis

Giving original A, B, C, and D equal weight leaves the all-eight effects large:

- removes 81.5% of the entropy gap;
- removes 87.5% of the spread gap;
- restores 72.3% of the winner-advantage gap;
- removes 10.56 percentage points, or 55.7%, of the letter-macro switch gap.

The reverse equal-letter patch recreates 52.8%, 80.3%, 68.0%, and 58.4% of
those respective gaps.

### Letter heterogeneity

MLP 63's flattening signs are consistent for every original letter: entropy
changes by -0.098 to -0.166 and spread by +0.312 to +0.394.

The all-eight Neutral-into-Game switch effects are not homogeneous:

| Original Baseline winner | Natural Game-Neutral switch gap | Patch effect | Fraction removed |
|---|---:|---:|---:|
| A | +46.9 pp | -36.9 pp | 79% |
| B | +5.7 pp | -5.7 pp | 100% |
| C | +4.3 pp | +8.5 pp | -200% |
| D | +18.9 pp | -8.1 pp | 43% |

The held-out Baseline-winner counts are A=130, B=35, C=47, D=37. This makes
the dataset-weighted switching estimate behaviorally appropriate for this
sample, but it cannot be described as an abstract letter-general switch
circuit.

## Interpretation

The most defensible mechanistic story is now:

- Game-specific flattening is not merely visible in final logits. It is
  causally mediated by a small collection of late final-position components.
- MLP 63 is the dominant single flattening component, although the full
  flattening group is more cleanly bidirectional than MLP 63 alone.
- Switching is not simply the behavioral consequence of generic flattening.
  The switching-group removal changes winner advantage and switching strongly
  while leaving entropy and spread almost unchanged. Conversely, the
  flattening-group removal strongly sharpens A-D evidence while leaving Game
  switching almost unchanged in the primary direction.
- The two groups overlap and interact, so this is a functional dissociation,
  not proof of two anatomically disjoint circuits.
- The patch establishes causal mediation under the paired prompts. It does not
  yet identify what semantic feature each component reads, nor prove that the
  model represents an explicit abstract instruction such as "add noise."

This is credible as a primary causal account of the observed SimpleMC behavior
for this checkpoint, provided the paper states the strong answer-letter
dependence of the switching portion. Replication on another dataset would be
needed before claiming a model-general mechanism.

## Artifacts and compute

- Full confirmation report:
  `outputs/causal/qwen36_27b_simplemc_causal_sweep/confirmation/analysis/COMPONENT_CAUSAL_REPORT.md`
- Dataset and letter-macro effects:
  `outputs/causal/qwen36_27b_simplemc_causal_sweep/confirmation/analysis/component_causal_effects.csv`
- Per-letter effects:
  `outputs/causal/qwen36_27b_simplemc_causal_sweep/confirmation/analysis/component_causal_effects_by_letter.csv`
- Machine-readable summary:
  `outputs/causal/qwen36_27b_simplemc_causal_sweep/confirmation/analysis/component_causal_summary.json`
- Frozen confirmation plan and discovery rankings:
  `outputs/causal/qwen36_27b_simplemc_causal_sweep/plans/confirmation_plan.json`
- Outcome and geometry figures:
  `outputs/causal/qwen36_27b_simplemc_causal_sweep/confirmation/analysis/causal_outcome_sweep.svg`
  and `causal_geometry_sweep.svg` in the same directory.

Vast instance 46566562 ran for approximately 4.2 hours at $1.0389/hour. The
measured account-credit reduction was $4.38. The instance was stopped, not
destroyed, after all outputs were retrieved and validated.
