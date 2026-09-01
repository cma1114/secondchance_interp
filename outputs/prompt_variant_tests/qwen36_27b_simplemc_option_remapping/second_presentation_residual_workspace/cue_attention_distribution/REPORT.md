# Exhaustive source map for the post-list answer cue

## Method

The query is the exact trailing space after `Your choice (A, B, C, or D):`.
Every non-padding prompt token is assigned to exactly one source row. Feedback
is resolved token by token; both presentations' option lines are aligned by the
candidate's first-presentation rank. Attention is averaged over heads and
reported separately for Game and Neutral on the frozen 249-question
confirmation split. All 16 applicable ordinary-attention layers, L4--L64, are
included. The remaining 48 layers use GLA, so an ordinary-attention source
distribution is undefined there.

This is cached activation analysis: **zero new model forward passes**.

## Validation

- Prompt tokens exactly matched the cached workspace for all 500 questions in both tasks.
- Maximum exhaustive-partition sum error: `0.000616`.
- Discovery/confirmation map cosine: Game `0.999744`, Neutral `0.999684`.
- Every cell's held-out mean and 95% question-bootstrap interval is in `attention_distribution.csv`.

## Findings

This section is explicitly hand-curated interpretation of the generated
summary/table and is now preserved by the analyzer through
`HAND_CURATED_FINDINGS.md` rather than being silently lost on regeneration.

At the layer-36 peak of held-out old/fresh cue decodability, the cue is not
primarily reading the four raw first-presentation option lines. Its absolute
attention allocation is:

| Source region at L36 | Game | Neutral |
|---|---:|---:|
| 2P question stem and separators | 21.54% | 20.73% |
| all four 2P option lines | 15.68% | 14.24% |
| cue prefix and query itself | 15.68% | 18.40% |
| 1P answer cue and empty decision boundary | 12.57% | 20.25% |
| complete feedback sentence | 18.45% | 3.37% |
| all four raw 1P option lines | 2.67% | 3.98% |

The layerwise trajectory distinguishes the tasks without relying on a
Game-minus-Neutral plot:

- In **Game**, feedback is already a major cue source at L28 (21.19%) and
  remains large at L32--44 (15.77%, 18.45%, 16.19%, and 14.74%). At L36,
  8.19% comes from contextualized `the` and 4.82% from literal `incorrect`.
- In **Neutral**, feedback falls from 16.03% at L28 to 9.58% at L32 and only
  3.37%, 2.05%, and 2.31% at L36--44. Instead, the 1P answer-cue/decision-
  boundary region receives 24.34% at L32, 20.25% at L36, 13.91% at L40, and
  25.11% at L44.
- In both tasks, all four 2P lines become substantial cue sources by L32--48.
  They are rank graded by L36: R1 receives the most attention, followed by
  R2, R3, and R4. At L48 Neutral allocates 10.51% to the 2P R1 line versus
  7.35% in Game; at L52 it allocates 10.88% versus 6.36%. The corresponding
  Game-minus-Neutral paired differences are -3.16 points
  `[-3.92,-2.41]` and -4.52 points `[-5.57,-3.48]`. The pooled R2--R4
  differences include zero at both layers, so this late task divergence is
  specifically concentrated on R1.
- By L60--64, the first-answer boundary again becomes the largest nonsystem
  source (25.77--40.06% in Game and 27.81--41.01% in Neutral), while direct
  feedback attention is small.

The two clearest mid-layer task differences replicate tightly. At L36, Game
exceeds Neutral in total feedback attention by 15.09 percentage points
`[14.43,15.74]`, while Neutral exceeds Game at the 1P answer boundary by 7.68
points `[7.07,8.29]`. These are paired 95% question-bootstrap intervals on the
frozen confirmation split.

## What this adds to the score analysis

The cue's old- and fresh-score decodability rises around L32--36. This map now
shows what the cue reads while that happens. Direct reads of the raw 1P option
lines are small at L36; the much larger candidate-bearing sources are the four
2P option lines. Prior exact source attribution established that matching 1P
lines write graded old score into those 2P semantic positions over L32--48.
Together, the results support an activation path from a 1P option line into
its matching 2P representation and then into the cue. They do not by themselves
prove that this path, rather than the large 1P answer-boundary route, causes the
cue's old-score decoding.

The feedback route is already strongly task dependent at L32--44, before the
cue's rank-specific Game/Neutral transformation becomes statistically clear at
L50. The map therefore corrects the earlier shorthand that L50 begins the
policy computation: L50 is the onset of the reliably decoded rank
transformation, not the onset of policy input to the cue.

This is an exact attention-allocation measurement, not a causal source lesion.
Attention mass also does not specify the direction or magnitude of the
resulting residual write.

## Artifacts

- Canonical figure: `figures/qwen36_cue_attention_distribution.png`
- Compact summary: `outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/cue_attention_distribution/summary.json`
- Per-question arrays: `outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/cue_attention_distribution/attention_distribution.npz`
- Complete confidence-interval table: `outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/cue_attention_distribution/attention_distribution.csv`
