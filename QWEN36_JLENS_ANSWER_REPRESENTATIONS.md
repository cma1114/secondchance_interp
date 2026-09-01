# Qwen3.6-27B JLens answer-representation analysis

## Question

This analysis separates two possible readouts of the model's answer state:

1. **Letter code:** JLens scores for the valid `A`, `B`, `C`, and `D` token
   variants.
2. **Option content:** a question-specific surface-content readout. Each option
   text is tokenized, punctuation-only and within-question shared pieces are
   removed, and the remaining token logits are averaged. When an option has no
   distinctive pieces, all of its alphanumeric pieces are used.

The four scores are centered within question. They are then aligned using one
fixed Baseline ranking in every condition and layer: the actually generated
Baseline answer is rank 1; the other three options are ordered by their final
Baseline A-D logits. This avoids redefining “winner” after the intervention.

The final decision position uses all 500 SimpleMC questions. The eleven other
prompt positions use the fixed 128-question sample from the expanded JLens run.
Confidence intervals are equal-weight macro estimates over original-answer
letters.

## Letter-code result

The A-D readout is reliable at the natural final residual. Its balanced
accuracy against the condition's generated output is 98.9% in Baseline, 94.9%
in Game, and 97.9% in Neutral.

At the final decision position, the Game-minus-Neutral contrast is:

| Readout | Original winner | Runner-up | Rank 3 | Rank 4 |
|---:|---:|---:|---:|---:|
| L48 | -0.030 | -0.024 | +0.010 | +0.043 |
| L56 | -0.426 | -0.055 | +0.180 | +0.302 |
| L60 | -0.547 | -0.037 | +0.193 | +0.391 |
| L64 | -1.046 | -0.162 | +0.329 | +0.879 |

At L64, the 95% intervals are [-1.155, -0.937] for the original winner,
[-0.242, -0.082] for the runner-up, [+0.253, +0.405] for rank 3, and
[+0.817, +0.941] for rank 4.

Thus the clearest A-D result is not a specific runner-up boost. Game produces a
late redistribution away from the original winner and, to a lesser degree, the
runner-up, toward the lower-ranked options. A small difference is detectable by
L48, but the large effect is concentrated after L52.

## Option-content result

The exact option-text readout is much weaker. At L64 its balanced accuracy
against the generated answer is only 32.6% in Baseline, 31.0% in Game, and
37.3% in Neutral, versus 25% chance.

At the final decision position, the Game-minus-Neutral option-content contrast
is:

| Readout | Original winner | Runner-up | Rank 3 | Rank 4 |
|---:|---:|---:|---:|---:|
| L48 | -0.007 | -0.010 | -0.002 | +0.019 |
| L56 | -0.032 | +0.007 | -0.003 | +0.028 |
| L60 | -0.130 | -0.010 | +0.042 | +0.098 |
| L64 | -0.135 | +0.012 | +0.044 | +0.080 |

At L64, only the original-winner reduction (95% CI [-0.194, -0.076]) and rank-4
increase ([+0.030, +0.130]) are individually clear. The runner-up contrast is
essentially zero ([-0.040, +0.063]).

This surface-content result should not be treated as a robust semantic decoder.
JLens predicts vocabulary logits, while the model is about to emit a letter,
not the option text. Exact option tokens are therefore a noisy proxy for the
underlying answer concept. In addition, at the feedback-end position the option
texts are about to recur verbatim in the repeated question, so high content
scores there need not express an answer belief.

## Bottom line

The A-D analysis works and reinforces a late, Game-specific redistribution
away from the original answer. The simple option-text analysis does not let us
confidently say whether the semantic answer representation is preserved while
only the letter-selection code changes. A letter-invariant semantic matching
probe would be required to answer that stronger question.

## Matched comparison with native logit lens and probes

The four-rank Game-minus-Neutral analysis was subsequently repeated without
changing the questions, fixed Baseline ranks, within-question centering, paired
subtraction, or confidence intervals. Only the readout method changed:

1. native logit lens;
2. JLens;
3. the pooled, five-fold cross-fitted Baseline answer probe trained on all
   1,000 SimpleMC and TriviaMC Baseline questions.

All three methods show the same qualitative redistribution in the reliable
late-layer window. At readout 52:

| Method | Original winner | Runner-up | Rank 3 | Rank 4 |
|---|---:|---:|---:|---:|
| Native logit lens | -0.114 | -0.023 | +0.047 | +0.091 |
| JLens | -0.115 | -0.040 | +0.035 | +0.120 |
| Pooled probe (Baseline SD) | -0.260 | +0.042 | +0.101 | +0.117 |

Native and JLens are in logit units; probe values are in held-out Baseline
standard-deviation units and cannot be compared by magnitude. Across all four
ranks and readouts 48--64, trajectory correlations are 0.963 between native
logit lens and JLens, 0.801 between native lens and the probe, and 0.827 between
JLens and the probe.

Both native logit lens and JLens begin a run of at least four consecutive
readouts with a significantly negative original-winner contrast at readout 48.
The pooled probe shows an earlier contrast, but its Game and Neutral answer
decoding is near chance before readout 48, so that earlier signal is a
condition-wide probe response rather than interpretable answer identity. By
readout 52, all methods have usable answer decoding and agree closely.

Therefore, JLens is not uniquely responsible for the finding. Its main benefit
is a cleaner pre-answer region and a more output-aligned intermediate scale.
The decisive improvement was the matched, paired visualization of all four
fixed Baseline ranks. The substantive onset around readouts 48--52 is robust to
all three readout methods.

### Definition of the paired fixed-rank four-option visualization

For each question, the four candidate identities are fixed once using the
final Baseline result. The actually generated Baseline answer is the original
winner. The other three options are ordered by their final Baseline A-D logits
and labeled original runner-up, rank 3, and rank 4. These labels never change
across layers or conditions.

At each residual readout, the four Game scores and four Neutral scores are
separately centered by subtracting their within-question A-D mean. The centered
Neutral score is then subtracted from the centered Game score for the same
question and same fixed candidate. Only after this paired subtraction are
questions aligned by Baseline rank and averaged. Original-answer letters are
given equal macro weight, and the confidence intervals retain the paired
question structure.

The resulting lines answer: relative to Neutral, how does Game change the
evidence for the original Baseline winner, runner-up, rank 3, and rank 4 as the
computation proceeds? Because each four-option vector is centered, the four
lines sum to zero at every readout. They describe redistribution within the A-D
answer space, not a common increase or decrease shared by all answer tokens.

This presentation differs from the previous analyses in several ways:

- Raw Baseline, Game, and Neutral trajectories were dominated by their shared
  layerwise evolution; pairing exposes the smaller condition difference.
- Original-winner margins collapsed the alternatives into whichever competitor
  was strongest at a given layer, so they could not show where lost winner
  evidence went.
- A-D spread compressed the four candidates into one unsigned dispersion and
  could not distinguish winner suppression from boosting of particular lower
  ranks.
- Dynamic-leader analyses changed candidate identity when the leader changed;
  fixed Baseline ranks track the same question-specific option throughout.
- Switch-stratified figures conditioned on the eventual outcome, whereas this
  visualization uses all 500 questions.

The combination, rather than any individual ingredient, makes it visually
clear that the late Game-specific change is not simply winner-to-runner
transfer: the original winner declines, the runner-up receives little
consistent benefit, and ranks 3--4 gain, with rank 4 eventually showing the
largest positive contrast.

### Complementary Baseline contrasts

Game-minus-Baseline is the most direct contrast for the hypothesis that the
model begins with its ordinary answer computation and then modifies it after
incorrect feedback. It asks how each original Baseline-ranked candidate changes
between the original decision and the Second Chance decision.

It is not by itself a clean estimate of strategic incorrect-feedback effects,
because Game also differs from Baseline in conversation length, answer history,
the redacted prior turn, repeated-question context, and the requirement to
answer again. The corresponding Neutral-minus-Baseline contrast is therefore
needed as the redo/regeneration control. Showing Game-minus-Baseline beside
Neutral-minus-Baseline reveals whether a Game-minus-Neutral result comes from
an active Game transformation, an unusual Neutral transformation, or both.

The three contrasts are algebraically redundant:

`Game - Neutral = (Game - Baseline) - (Neutral - Baseline)`.

Accordingly, a compact presentation should show Game-minus-Baseline and
Neutral-minus-Baseline together, while retaining the existing Game-minus-Neutral
figure as the focused strategic contrast rather than placing all three in one
redundant panel set.

## Artifacts

- Canonical interactive explorer:
  `outputs/mechanistic/qwen36_27b_jlens_answer_content/analysis/preserved_figures/jlens_answer_representation_explorer.html`
- Analysis table:
  `outputs/mechanistic/qwen36_27b_jlens_answer_content/analysis/answer_representation_trajectories.csv`
- Compact analysis data:
  `outputs/mechanistic/qwen36_27b_jlens_answer_content/analysis/answer_representation_trajectories.json`
- Option-content JLens scores:
  `outputs/mechanistic/qwen36_27b_jlens_answer_content/option_content_scores.npz`
- Per-question token audit:
  `outputs/mechanistic/qwen36_27b_jlens_answer_content/option_token_audit.json`
- Collection code: `mechanistic/jlens_answer_content.py`
- Analysis code: `mechanistic/analyze_jlens_answer_content.py`
- Canonical three-method comparison:
  `outputs/mechanistic/qwen36_27b_jlens_answer_content/analysis/method_comparison/readout_method_comparison.png`
- Three-method numerical table:
  `outputs/mechanistic/qwen36_27b_jlens_answer_content/analysis/method_comparison/readout_method_comparison.csv`
- Comparison code: `mechanistic/compare_answer_readouts.py`

The Vast A100 was active for approximately 8.4 minutes at $1.0389/hour, for
approximately $0.15 of GPU time. The instance was stopped and preserved.
