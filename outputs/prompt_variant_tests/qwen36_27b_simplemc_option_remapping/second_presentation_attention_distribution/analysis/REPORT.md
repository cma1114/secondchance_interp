# Exhaustive attention distribution from second-presentation option lines

## What was measured

For every SimpleMC question, every ordinary-attention layer **4--64**, and both
Game (`incorrect`) and Neutral (`lost`), this analysis asks where each complete
second-presentation option line attends.  The four target lines are aligned by
the candidate's first-presentation rank, R1 through R4.  Attention is averaged
over all tokens in the target line and all attention heads.

Every non-padding source token belongs to exactly one row in the figure.  Thus
the rows are an exhaustive distribution, not a selected set of destinations.
The green outline marks the first-presentation line containing the same answer
content as the second-presentation target line.

## What the distribution shows

The matching 1P line is important, but it is not the only substantial historical
source. Averaging across the four 2P target lines on held-out questions:

- At layer 4, all four 1P option lines receive **16.8%** of attention. Their
  combined share is **35.5% at layer 12**,
  **34.5% at layer 36**, **37.1%
  at layer 52**, and **22.0% at layer 64**.
- The single semantically matching 1P line rises from **5.9% at layer 4** to
  **12.8% at layer 12**, and is **16.0%
  at layer 48** and **15.4% at layer 52**. The other
  three 1P option lines jointly receive **22.6% at
  layer 12** and **21.7% at layer 52**. Thus each 2P option line reads the whole first-pass
  candidate set while preferentially reading its semantic match.
- The 1P answer cue and decision boundary receive **7.0%
  at layer 36**, **8.9% at layer 60**, and
  **13.0% at layer 64**. This is a plausible place
  for relational first-pass information, but the attention measurement does
  not establish what information is read or whether it is causally necessary.
- Game and Neutral allocate attention differently to the feedback sentence.
  Game attention is **3.4%, 4.2%,
  4.0%, and 2.2%** at layers
  28, 36, 44, and 48; Neutral is **1.6%,
  1.1%, 1.0%, and
  0.6%**. These are direct reads
  of the policy-bearing sentence, not evidence that the sentence itself
  contains candidate rankings.
- The repeated question stem is also a major source: **26.2%
  at layer 4**, **15.2% at layer 12**, and
  **13.2% at layer 36**. The current 2P option line's own
  causal prefix receives **10.9% at layer 36**,
  **13.0% at layer 52**, and
  **25.5% at layer 64**.

The central new fact is therefore that the 2P line has simultaneous access to
three ingredients: its matching 1P semantic line, the other three 1P candidate
lines, and the first-answer boundary, while Game additionally reads the
`incorrect` feedback much more strongly at layers 28--48. This makes a
distributed rank/policy computation plausible. It does **not** yet show which
of those nonmatching reads carries winner rank or whether any one is causally
required.

## Validation

- All 500 questions completed; the canonical figure uses the frozen 249-question confirmation split.
- Maximum error when summing the exhaustive source rows to one: **0.004067**.
- Natural answer agreement with the trusted canonical outputs: **100.0%**.
- Mean absolute error against the previously measured matching-line trajectory: **0.000768** (99th percentile **0.006566**).

## Artifacts

- Canonical figure: `figures/qwen36_second_presentation_attention_distribution.png`
- Cell-level means and confidence intervals: `outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_attention_distribution/analysis/attention_distribution.csv`
- Machine-readable summary and top source regions: `outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_attention_distribution/analysis/summary.json`
