# Policy information in the second-presentation residual stream

## Question and method

This analysis asks when the minimal `incorrect` versus `lost` difference is
present while Qwen3.6-27B processes the four second-presentation (2P) option
lines, and which feedback positions write it there.

The prompts are paired within each of all 500 questions. They are identical
except for the single evaluation word. At every layer, the analysis compares
the complete residual at a feedback source position with the complete residual
at each 2P option-line token. It uses the full J-lens and R-lens vocabulary
profiles; no hand-picked semantic family determines the result.

Here “J-lens” and “R-lens” mean learned per-layer transports from the external
`camilablank/workspace-lenses` checkpoint followed by the model's real final
norm and unembedding; they are not the plain final-norm logit lens implemented
by `modeling.py:cpu_lens`. Human-readable top-token tables additionally
filter to printable ASCII tokens containing at least one alphabetic character.
The filter affects presentation only, not the underlying vocabulary vectors or
causal-write metrics.

At every ordinary-attention layer, the cached attention weights, values,
gates, and output projection reconstruct the exact residual write from each
individual feedback token. The first completed source screen summarized those
writes at ten receiver roles: the mean over each of the four 2P lines, each
line's closing newline, the post-list choice-cue space, and the final decision
position. It did **not** preserve separate source-write estimates for the
option-letter and semantic-content tokens within each line. `R1` through `R4`
below mean the semantic candidates ranked first through fourth on the first
presentation, not literal display positions. A “line” result averages its
tokens; a “newline” result uses its literal closing newline.

Source/receiver/layer candidates were selected using 251 discovery questions.
Their vector direction and all choices were then frozen before evaluation on
249 held-out questions.

## What is directly written from feedback into 2P

The condition-dependent source write into the four 2P option newlines follows
a reproducible three-stage trajectory:

| Feedback source | Peak ordinary-attention layer | Held-out mean write RMS across R1--R4 newlines | Discovery/held-out vector cosine |
|---|---:|---:|---:|
| literal `incorrect` / `lost` token | 32 | 0.0143 | 0.9981 |
| evaluation-closing period immediately after that word | 44 | 0.0141 | 0.9995 |
| contextualized `Choose` token in the otherwise identical action clause | 60 | 0.0300 | 0.9981 |

This is a transformation across feedback positions, not three independent
surface-word copies. The L60 `Choose` token is textually identical in Game and
Neutral, but its residual has already been contextualized by `incorrect` or
`lost`. At L60, the four 2P newlines attend to that token more in Game
(0.824--0.939%) than Neutral (0.478--0.542%). The resulting exact writes are
highly reproducible, but their full-vocabulary similarity to the source state
is only 0.033--0.051. Thus the late write carries a transformed
condition-dependent action signal; it should not be described as literally
copying the word “Choose.”

The exact source-write curves also show why naming only L60 would be wrong.
The literal evaluation word and its closing period already make substantial,
replicated direct writes into every 2P option newline at L28--L44. L60 is the
largest single direct feedback-token write, not the onset of policy
information in 2P.

## What the complete 2P residual represents

The complete 2P residual contains the condition difference before L60 and
changes its readable form over depth. The clearest held-out J/R-lens summaries
are:

| Layers | Game-minus-Neutral content at all four 2P option lines/newlines |
|---|---|
| 24--28 | The R-lens first becomes plainly evaluative: `wrongly`, `falsely`, `failure`, `unsuccessful`. The J-lens is not yet consistently readable. |
| 36 | Both lenses converge on retry semantics: `again`, `retry`, `failed`, `reconsider`. |
| 40--48 | The contrast shifts toward action/history semantics: `again`, `previously`, `remaining`, then `replaced`, `revised`, `replacement`. |
| 52--60 | It becomes explicitly remapping-like: `newly`, `changed`, `swapped`, then `permutation` and `permutations`. |

The discovery and held-out trajectories are nearly identical. The four ranks
also look strikingly similar at this level. This analysis therefore establishes
that feedback policy is available in every 2P option residual; it does not yet
identify a rank-specific policy binding inside those residuals.

The high full-vocabulary alignment with the feedback-word state at very early
layers (roughly L3--L12) is not, by itself, a clean semantic onset. At those
layers the feedback-word lens profile and the 2P profile are both largely
uninterpretable in English. The first defensible English readout is the
R-lens result at L24--L28; both lenses agree by L36.

## Direct routes to the post-list and final-decision positions

Two earlier, source-preserving routes are especially clean:

- At L36, the post-list choice-cue space attends directly to the literal
  `incorrect`/`lost` token: 4.823% in Game versus 0.229% in Neutral on held-out
  questions. The exact source-specific write has held-out RMS 0.0559,
  discovery/held-out cosine 0.9995, and source-to-write full-vocabulary cosine
  0.320 under J-lens and 0.311 under R-lens. Its positive tokens include
  `incorrect`, `failed`, `failure`, `mistake`, `corrected`, and `revised`;
  its opposite includes `continuity`, `restore`, `normal`, and
  `uninterrupted`.
- At L28, the final decision position attends directly to the
  evaluation-closing period: 5.889% in Game versus 1.280% in Neutral. The exact
  write has held-out RMS 0.0303 and discovery/held-out cosine 0.99994. After
  that write, the complete final residual reads out `incorrect`, `failure`,
  `failures`, and `unsuccessful`.

These findings show that policy is not confined to a single evaluation-period
GLA memory. Ordinary attention also carries it directly into 2P, the post-list
summary, and the final decision at identifiable layers.

- [Raw token-cross source/write artifact](policy_token_cross_raw/policy_token_cross.json)
- [Discovery-frozen readable top-token artifact](policy_top_tokens/policy_top_tokens.json)

## What this establishes, and what it does not

This is exact activation-path evidence: it identifies the source token, receiver
position, layer, attention allocation, and residual write. It is stronger than
a free-floating probe correlation, but it is not yet a causal policy-feature
intervention.

The result establishes that every 2P option residual jointly contains its
current option computation and a feedback-conditioned state by L24--L36, with
continued transformed feedback writes through L60. It does not yet establish
where the old 1P score, fresh 2P score, and rank-specific policy adjustment are
separately represented, nor which decoded policy component causes the known
late R1 attention attenuation or final preferential switching. Those are the
next analyses supported by the retained workspace.

## Artifacts

- [Canonical figure](../../../../figures/qwen36_second_presentation_policy_transport.png)
- [Frozen design and cache specification](PLAN.md)
- [Source-specific held-out summary](policy_transport/policy_transport_summary.json)
- [Complete J/R residual trajectory](policy_transport/policy_residual_trajectory.json)
- [Complete exact source-write trajectory](policy_transport/policy_source_trajectory_summary.json)

## Score information and rank-dependent policy inside 2P residuals

The second analysis fits one-dimensional old-score and fresh-score directions
on the 251 discovery questions and applies them without refitting to the 249
held-out questions. The targets are residualized against one another and
against both 1P and 2P display position.

- Unique old 1P score is most strongly held-out decodable from 2P semantic
  residuals at layer 49: `r=0.349` for the complete-line mean, `r=0.344` for
  semantic-content tokens, and `r=0.289` at the final semantic token. The
  closing newline is substantially weaker.
- Unique fresh 2P score is also decodable, though less strongly: its held-out
  complete-line correlation peaks near layer 47 at `r=0.227`.
- The Game-minus-Neutral rank adjustment is already present inside the 2P
  semantic-token residuals. At the held-out final semantic token at layer 50,
  projection on the frozen old-score direction changes by `-0.172` standard
  units for R1, `-0.031` for R2, `+0.067` for R3, and `+0.135` for R4. The
  bivalent contrast, R4 minus the mean of R1 and R2, is `+0.237`, 95% CI
  `[+0.195,+0.278]`. The same ordered pattern appears along the distinct
  fresh-score direction.
- The rank-dependent task difference begins around layer 34 and strengthens
  sharply across layers 45--50. Thus it is not merely a final-logit effect and
  is not concentrated at the option newline.

These are held-out activation results, not yet causal source attribution. The
next cached-data analysis therefore resolves each plausible feedback source
against the option letter, every semantic wordpiece, and the newline
separately, rather than reusing the earlier whole-line compression.

## Exact feedback-source × 2P-token map

The token-resolved follow-up is complete. It reconstructs the exact ordinary-
attention write from each of the ten feedback tokens into the indentation,
option letter, colon, every relative semantic wordpiece, and newline of all
four 2P options at all 16 ordinary-attention layers. The 251/249 discovery and
confirmation split is preserved. Cells with fewer than 20 questions in either
split remain in the complete map but cannot be selected as peaks.

The direct feedback writes are concentrated at structural anchors rather than
the semantic answer wordpieces:

| Feedback source | Option-letter peak | Semantic-wordpiece peak | Newline peak |
|---|---:|---:|---:|
| literal `incorrect` / `lost` | L32, RMS 0.0137 | L56, RMS 0.00348 | L32, RMS 0.0143 |
| evaluation-closing period | L16, RMS 0.0157 | L36, RMS 0.00219 | L44, RMS 0.0141 |
| contextualized later `Choose` | L60, RMS 0.0115 | L60, RMS 0.00481 | L60, RMS 0.0300 |

At L32, 2P option letters attend to the literal `incorrect` token by 2.27% in
Game, versus 0.079% to `lost` in Neutral; the corresponding newline values are
2.03% versus 0.106%. The exact-write contrast replicates almost perfectly
between discovery and confirmation (mean vector cosine 0.998--0.999 across
the four ranks). Direct writes into semantic wordpieces are real but much
smaller on average.

The R1--R4 trajectories are nearly the same at each of these source-specific
peaks. Therefore these direct feedback-to-2P writes provide a generic policy
state to every option line; they do not themselves exhibit the bivalent
R1/R2-down, R4-up pattern seen in complete semantic-token residuals around
L49--L50. The remaining localization problem is where that generic policy
state is combined with candidate-specific old and fresh evidence.

## Raw attention to feedback, described within each task

The source-write contrasts above are not raw attention maps. The raw held-out
attention map, averaged over R1--R4 but retaining the option letter, semantic
wordpieces, and closing newline separately, shows different within-task
allocation patterns.

In **Game**, the model reads both the literal `incorrect` token and the period
that follows it. Their relative importance changes with layer and receiver:

- At option letters, the period dominates at L16 (2.249% versus 0.595%), the
  word dominates at L32 (2.268% versus 1.614%), and the period is again larger
  at L44 (0.780% versus 0.445%).
- At closing newlines, the two are similar at L20 (1.508% versus 1.428%); the
  word is larger at L32 (2.026% versus 1.740%); and the period is larger at L44
  (1.144% versus 0.749%).
- Semantic wordpieces read both sources more weakly. At L32 they allocate
  0.404% to `incorrect` and 0.285% to the period.

In **Neutral**, direct reads of `lost` are substantially weaker, while the
period and later action-clause tokens receive more of the available feedback
attention:

- At option letters, `lost` versus period attention is 0.157% versus 0.698% at
  L16 and 0.079% versus 0.261% at L32.
- At closing newlines, it is 0.335% versus 1.652% at L20 and 0.106% versus
  0.384% at L32.
- Reads into semantic wordpieces are small throughout.

Thus attention is not simply divided equally between the evaluation word and
its period. Game repeatedly reads both, with alternating layer-specific
emphasis. Neutral largely deemphasizes the literal `lost` token and relies
more on the contextualized period and later clause. These are absolute
within-task attention patterns; the Game-minus-Neutral write is a subsequent
comparison, not the primary cognitive description.

- [Canonical token-cross figure](../../../../figures/qwen36_second_presentation_policy_token_cross.png)
- [Raw feedback-attention heatmap](../../../../figures/qwen36_second_presentation_policy_attention_heatmap.png)
- [Compact token-cross summary](policy_token_cross/SUMMARY.json)
- [Complete token-level map](policy_token_cross/policy_token_cross.json)

## Raw within-task feedback writes and lens profiles

The preceding source-write plots used an unsigned **contrast RMS**: for each
source/receiver/layer cell, they subtracted the mean Neutral write vector from
the mean Game write vector and then took the RMS across the 5,120 residual
dimensions. This measures the distance between two write vectors. It does not
mean that Game writes “that much more,” and it has no sign.

The follow-up therefore reconstructs a raw write magnitude separately in each
task. For each question and source-to-receiver edge, it takes the RMS of the
exact source-specific residual write across the 5,120 dimensions, then averages
that scalar over held-out questions and R1--R4. Discovery/confirmation
replication is high across eligible cells (`r=0.973` in Game and `r=0.962` in
Neutral).

The raw map confirms that the middle-layer `the` route is a real residual
write, not merely a large attention weight:

| Source and layer | Destination | Game raw write RMS | Neutral raw write RMS |
|---|---|---:|---:|
| literal `incorrect` / `lost`, L32 | option letter | 0.01605 | 0.00066 |
| literal `incorrect` / `lost`, L32 | semantic wordpieces | 0.00377 | 0.00020 |
| literal `incorrect` / `lost`, L32 | closing newline | 0.01798 | 0.00166 |
| contextualized `the`, L32 | closing newline | 0.01843 | 0.00243 |
| contextualized `the`, L36 | semantic wordpieces | 0.00957 | 0.00015 |
| contextualized `the`, L44 | option letter | 0.01463 | 0.00107 |
| contextualized `Choose`, L60 | option letter | 0.01925 | 0.01001 |
| contextualized `Choose`, L60 | semantic wordpieces | 0.00887 | 0.00429 |
| contextualized `Choose`, L60 | closing newline | 0.05117 | 0.02309 |

Thus Game has a strong middle-layer route through the literal evaluation word,
its period, and especially the contextualized `the`; the latter writes into
semantic wordpieces most strongly at L36 and into option letters through L44.
Neutral largely omits that middle-layer route. Its literal `lost` contribution
instead peaks much later at L56 (0.01031 at letters, 0.00300 at semantic
wordpieces, and 0.01255 at newlines). Both tasks receive the late contextualized
`Choose` write at L60, but it is about twice as large in Game. Common tokens
such as `Your` and `answer` also make substantial raw late writes, so raw
magnitude alone must not be equated with policy content.

All six raw-write panels now use the same 0--0.0512 RMS color scale. On that
common scale, the semantic-wordpiece writes are visibly and quantitatively
smaller than the letter and newline writes; the earlier independently scaled
panels obscured that comparison.

### Are these RMS writes large?

The raw RMS values have now been normalized by the complete ordinary-attention
output at the same receiver token and layer, using the same held-out questions
and per-question RMS averaging. This is a vector-magnitude comparison, not a
variance partition: source contributions may point in opposite directions and
cancel in the complete attention output.

- The largest single feedback-token contribution is the Game evaluation period
  into 2P option letters at L16: **17.1%** of the complete attention-update RMS.
- At L32, literal `incorrect` contributes **9.2%** of the option-letter update,
  **8.5%** of the newline update, but only **1.8%** of the semantic-wordpiece
  update.
- The contextualized Game `the` contribution is **3.8%** of the complete
  semantic-wordpiece update at L36 and **7.3%** of the option-letter update at
  L44.
- The large absolute `Choose` newline write at L60 is **5.0%** of Game's and
  **2.3%** of Neutral's complete attention update. It looks large in absolute
  RMS partly because the complete late update is itself large.
- Across every cell, a single feedback token is at most **2.38%** of the
  receiver's pre-layer residual RMS. That is not negligible for one token in a
  long context, but it is not a dominant fraction of the whole residual.

Thus the middle-layer Game routes are practically substantial components of
the relevant attention updates, especially at option letters and newlines.
The semantic-wordpiece routes are real but modest.

The old label “alignment with feedback word” was also derivative. At each
layer and position, the procedure (1) lens-transports the residual, (2) applies
the model's final normalization and unembedding to obtain one score for every
vocabulary item, (3) averages those vocabulary-score vectors over questions,
(4) forms Game minus Neutral at the 2P position and at the `incorrect`/`lost`
position, (5) centers each difference vector over vocabulary, and (6) takes
their cosine. It therefore measures whether the **condition-induced vocabulary
change** at 2P resembles the condition-induced change at the feedback word. It
is not a raw within-task semantic similarity.

The raw-lens follow-up instead performs step (6) between Game 2P and Game's own
`incorrect` state, and separately between Neutral 2P and Neutral's own `lost`
state, without cross-condition subtraction. These trajectories replicate
almost exactly across the frozen splits (`r>0.999` for both lenses and tasks),
but the within-task cosines are broadly positive in both conditions and the
highest-scoring raw vocabulary items are mostly shared. The raw similarity is
therefore dominated by common prompt and option-processing state. The
Game-minus-Neutral lens contrast remains useful for isolating policy-specific
content, but it must be presented as a contrast after the two tasks are shown
separately.

The statement that “Neutral is more aligned early” refers only to this raw,
full-vocabulary within-task cosine. For example, at the R1 line under the
R-lens, Neutral exceeds Game at L16 (0.627 versus 0.386). The old contrast plot
does not show either of those numbers: it compares `(Game - Neutral)` at 2P
with `(incorrect - lost)` at the feedback word. The two plots therefore cannot
be read as opposite estimates of one quantity. More importantly, the raw
full-vocabulary cosine should not be described as policy alignment, because it
is dominated by vocabulary shared across both prompts.

### Superseded complete-residual top-token diagnostic

The first top-token follow-up lensed the **complete 2P residual**. Shared
multiple-choice and correctness processing dominated that state, so Game and
Neutral appeared almost identical. That analysis answered the wrong question:
it did not isolate what any evaluation token contributed. Its two figures are
archival diagnostics and are not evidence about policy transport.

An exact-position audit nevertheless resolves an important ambiguity in that
diagnostic. It uses all 249 held-out questions, every post-layer residual from
L1 through L64, both J- and R-lenses, and separates each 2P option's literal
letter, mean semantic wordpieces, and closing newline. The condition labels are
correct: at the literal feedback word over L28--44, Game's top tokens are
consistently `incorrect`/`error`/`correct`, whereas Neutral's are
`lost`/`forgotten`/`missing`/`loss`.

The downstream `incorrect` tokens in Neutral are real but localized. Exact
`incorrect` or `incorrectly` appears repeatedly in the top-ten vocabulary list
at **option letters and closing newlines in both tasks**, chiefly over
L28--L45. Neither word ever appears in the top ten at the **semantic
wordpieces** of R1--R4 in either task. At L36, for example, both tasks' option
letters and newlines simultaneously rank `correct` and `incorrect`-family
words, while their semantic wordpieces have almost identical candidate-content
profiles. Thus the complete-residual lens is exposing shared
answer/correctness-domain geometry at structural option positions, not showing
that Neutral was mislabeled as Game or that Neutral literally received the
`incorrect` feedback state. The later post-list cue and final-decision states
show `incorrect` much more often in Game, but source-specific policy transport
must still be established from the exact-write analysis below rather than from
top-token membership alone.

The machine-readable exact-position audit is
[`full_state_top_tokens.json`](full_state_top_tokens/full_state_top_tokens.json).

### Requested-word trajectories in complete destination residuals

This follow-up replaces top-ten inspection with raw layerwise scores for the
two word sets requested by the user. The corrected lexicon contains **only**
capitalization and ordinary morphological variants of the eight anchors:
`incorrect`, `failed`, `mistake`, `wrong`; and `lost`, `again`, `resend`,
`repeat`. It contains no added semantic neighbors such as `error`, `retry`,
`recover`, `missing`, or `forgotten`. Fourteen variants (23 vocabulary tokens
after capitalization) matched the first set and 12 variants (20 vocabulary
tokens) matched the second. Capitalization variants are averaged within a
word-form before word-forms are averaged equally. Only normal space-prefixed
single-token vocabulary entries are used.

The score is the mean vocabulary score after J- or R-lens transport, final
model normalization, and unembedding. It is measured at the option letter,
semantic wordpieces, and closing newline of R1--R4, plus the post-list answer
cue and final decision, across all 64 layers. Game and Neutral are shown
separately. These are readouts of each destination's **complete residual**, not
source-specific writes or causal effects.

The corrected J-lens result is:

- **Option letters:** the `incorrect/failed/mistake/wrong` set is strongly and
  almost identically active in both tasks from about L20. Averaged over R1--R4,
  both peak at L30 (3.50 in Game, 3.51 in Neutral). A Game excess develops
  later: at L36 the means are 3.03 and 2.47.
- **Semantic wordpieces:** the same set is present but much weaker. Its
  rank-mean peak is at L34 (1.17 in Game, 1.07 in Neutral); at L36 the values
  are 0.98 and 0.65.
- **Closing newlines:** the set is stronger than at the semantic wordpieces and
  becomes more Game-weighted around L34--44. The rank-mean values are 2.99 and
  2.72 at L34, then 2.76 and 2.08 at L36.
- **Post-list answer cue:** the task separation is large. At L36, the first-set
  activation is 5.66 in Game and 2.54 in Neutral.
- **Final decision:** the separation also persists here. At L44, the first-set
  activation is 3.23 in Game and 1.71 in Neutral.

The salient temporal ordering is therefore:

1. Every 2P option-closing newline develops a similar
   `incorrect/failed/mistake/wrong` trajectory. It is already rising by about
   L20, is large by L28, peaks at L34, and remains elevated through roughly
   L48. The rank-mean Game trajectory is 0.50 at L20, 2.36 at L28, 2.99 at
   L34, and 1.77 at L48. At L20 the Neutral mean is 0.46, so the tasks are
   still nearly identical at the option newlines.
2. The post-list answer-cue space contains a much larger version. Its Game
   trajectory is 0.85 at L20, 4.85 at L28, peaks at 5.66 at L36, and remains
   4.49 at L44. Crucially, the Neutral value is only 0.22 at L20: the
   Game signal is already about four times as large here, before a comparable
   separation is visible at either the option newlines or final decision. This
   is the dominant complete-residual concentration of the requested
   incorrect-related vocabulary among the measured 2P positions.
3. The final decision position shows a related but smaller trajectory: 0.27 at
   L20 versus 0.12 in Neutral, 3.22 at L28, a Game peak of 3.89 at L34, and
   3.23 at L44. The two L20 values are both small and visually close; the
   early task separation is therefore specific to the measured answer-cue
   space, not yet clearly present at the final decision. The final-decision
   trajectory is more muted, but its Game peak is not later; the Neutral peak
   occurs at L37.

The answer-cue space is specifically the trailing space after `Your choice
(A, B, C, or D):`. It is not the only token between the last 2P option and the
final decision position. After the final option newline, the exact tail is the
separator line, newline, the tokenized `Your choice (A, B, C, or D):`, the
trailing space, `<|im_end|>`, newline, `<|im_start|>`, `assistant`, newline,
`<think>`, a blank line, `</think>`, and then the final double-newline token.
The residual at that final double-newline is the final decision position used
to predict the answer letter. The answer-cue space was preselected because it
is the uniquely identifiable endpoint of the explicit choice prompt, not
because prior evidence showed that it was the unique policy carrier. The
current trajectory figure omits the other intervening tail tokens even though
their residuals exist in the remote cache. Consequently, the figure establishes
an early policy-sensitive state at this space but does not establish that the
state is absent from, or not relayed through, the intervening scaffold.

The `lost/again/resend/repeat` set does **not** form a clean Neutral-specific
trajectory. At option letters, Game peaks later and slightly higher (1.48 at
L37) than Neutral (1.28 at L22); at semantic wordpieces the middle- and
late-layer scores are negative in both tasks. This exact requested set
therefore does not support the earlier interpretation based on the improperly
expanded `loss/retry` family. R1--R4 remain very similar within each token type,
and the R-lens reproduces the main timing and task separation with smaller
absolute values.

The complete-residual result is compatible with a shared answer-evaluation
state at structural option positions and a stronger later Game state at the
answer cue and final decision. It does not identify which source supplied that
state; the exact evaluation-token write analysis below addresses source
attribution separately.

The canonical figure is
[`qwen36_second_presentation_policy_family_trajectories.png`](../../../../figures/qwen36_second_presentation_policy_family_trajectories.png),
and the exact arrays and matched token inventory are in
[`policy_family_trajectories.json`](policy_family_trajectories/policy_family_trajectories.json).

### Does the answer-cue space already contain the final A–D ranking?

Yes, substantially but not completely. At the final layer, the model's final
normalization and output head were applied to the cached residual at (1) the
trailing space after `Your choice (A, B, C, or D):` and (2) the final
double-newline that predicts the answer. For every letter, the bare and
space-prefixed vocabulary tokens (for example, `A` and ` A`) were combined by
log-sum-exp at both positions. The analysis therefore compares like with like
rather than treating tokenization style as a different answer.

On the **249-question held-out confirmation split**:

The “final” scores in this correspondence calculation are reconstructed from
cached bf16 residuals through the final norm/head, not copied from the trusted
emitted logits. The maximum bare-token reconstruction discrepancy is about
0.18 logits and changes roughly 3--6% of final argmaxes. The separate
provisional-switching comparison below uses trusted emitted answers and is not
affected by this reconstruction error.

- **Game:** the cue and final positions agree on the top letter for 60.6% of
  questions (95% bootstrap CI 54.6--66.7%), agree on 76.7% of all six pairwise
  A--D orderings (74.2--79.1%), and have pooled within-question centered-score
  correlation 0.782. The cue's top letter matches the model's actual answer on
  61.4% of questions (55.4--67.5%).
- **Neutral:** top-letter agreement is 64.3% (58.2--70.3%), pairwise-order
  agreement is 79.1% (76.6--81.5%), and centered-score correlation is 0.856.
  The cue's top letter matches the actual answer on 66.7% of questions
  (60.6--72.3%).

The discovery split independently gives the same result, somewhat more
strongly: cue-versus-final top-letter agreement is 66.1% in Game and 72.1% in
Neutral; the cue predicts the actual answer on 66.5% and 74.1%, respectively.
For context, the confirmation top-letter agreement expected if the cue and
actual answer were independent but retained their observed letter biases is
33.7% in Game and 25.6% in Neutral.

Thus, before the assistant scaffold, the answer-cue space already contains a
decision-like A--D ordering that strongly anticipates the emitted answer. The
remaining scaffold is not a passive copy: the cue and final top letters differ
on roughly 36--39% of held-out questions, even though most pairwise ordering is
preserved. This is a descriptive residual readout, not yet proof that the cue
state causally determines the final answer.

Using the cue's top-ranked letter as a **provisional answer** shows that the
distinctive behavior is already present there. Each provisional remapped
letter was mapped back to its original semantic option, and switching was
defined exactly as in the behavioral analysis: choosing content other than the
first-presentation winner W1. On confirmation, overall provisional switching
is **67.9% in Game** and **49.4% in Neutral**, a Game excess of 18.5 points
(95% stratified-bootstrap CI 11.6--25.3). The emitted answers switch 62.7% and
45.0%, a nearly identical 17.7-point excess (12.0--23.7). Discovery replicates
this: the Game--Neutral gap is 13.5 points at both cue and final output.

The later scaffold changes the absolute rates mainly on **no-conflict** trials.
On held-out conflict trials, cue switching is 80.1% in Game and 59.6% in
Neutral, versus final rates of 82.4% and 60.3%: essentially unchanged. On
no-conflict trials, cue switching is 53.1% and 37.2%, but falls by the final
output to 38.9% and 26.5%. The paired reductions are 14.2 points in Game
(5.3--23.0) and 10.6 points in Neutral (1.8--19.5). Therefore the subsequent
tokens do important question-level processing, but they do **not** create the
preferential Game switching effect. They mainly stabilize W1 when the old and
fresh winners agree.

The canonical figure is
[`qwen36_cue_final_letter_correspondence.png`](../../../../figures/qwen36_cue_final_letter_correspondence.png),
and exact statistics and scores are in
[`summary.json`](cue_final_letter_correspondence/summary.json) and
[`cue_final_letter_correspondence.json`](cue_final_letter_correspondence/cue_final_letter_correspondence.json).
The exact provisional-switching analysis is
[`provisional_switching.json`](cue_final_letter_correspondence/provisional_switching.json).

### Does the answer cue separately contain old evidence, fresh evidence, and the rank policy?

Yes, but the three results are not equally strong. Because the cue is one
shared summary vector rather than four option positions, the analysis used a
four-output decoder in the remapped 2P A--D display order. Old and fresh
candidate scores were centered and each residualized on the other score plus
both display positions. Four-output ridge decoders were fit to the
Game/Neutral-mean cue residual using only the 251 discovery questions;
regularization and the primary layer were frozen before evaluation on the 249
confirmation questions. Every residual layer 1--64 was analyzed. This required
no new model forwards.

- **Fresh 2P evidence is clearly present.** The discovery-selected layer is
  **35**. On confirmation, the shared task-mean correlation is `r=0.239`
  `[0.167,0.309]`; separately it is `0.157` `[0.074,0.233]` in Game and
  `0.102` `[0.029,0.177]` in Neutral.
- **Unique old 1P evidence is present but weaker and task-dependent.** The
  discovery-selected layer is **50**. The shared confirmation correlation is
  `r=0.121` `[0.043,0.194]`. Neutral retains a clear old-score readout,
  `r=0.148` `[0.083,0.213]`, whereas Game does not at that frozen layer,
  `r=0.034` `[-0.041,0.104]`. The full held-out trajectory shows broader
  mid- and late-layer old-score decoding, but layer 50 is the prespecified
  primary test and was not replaced using confirmation results.
- **The task difference is organized by old rank at the cue.** At layer 50,
  Neutral's decoded old-evidence profile is
  `[+0.157,+0.088,-0.125,-0.119]` for R1--R4. Game's profile is much flatter:
  `[+0.036,-0.008,-0.066,+0.037]`. The Game-minus-Neutral differences are
  `[-0.121,-0.095,+0.059,+0.157]`. Their bivalent contrast,
  `R4 - mean(R1,R2)`, is `+0.265` `[+0.070,+0.470]`. In the complete held-out
  trajectory this old-evidence bivalent interval excludes zero continuously
  from **layers 50--64**.
- The analogous fresh-evidence bivalent effect at its frozen layer 35 is
  `+0.082` `[-0.153,+0.324]`, so it is not established there.

Thus the cue contains a fresh candidate ranking in both tasks and a weaker
retrievable old ranking. More importantly, by layer 50 its old-evidence
geometry already expresses the policy distinction: Neutral preserves the old
top-heavy ranking, while Game flattens it and relatively reallocates support
toward the old lowest-ranked candidate. This is held-out activation decoding,
not yet a causal cue intervention.

- [Canonical cue score figure](../../../../figures/qwen36_cue_score_integration.png)
- [Compact statistical summary](cue_score_integration/summary.json)
- [Complete layerwise decoder result](cue_score_integration/cue_score_integration.json)
- [Compact per-question projections](cue_score_integration/cue_score_projections.npz)

### Where the answer cue reads while old and fresh evidence emerge

An exhaustive follow-up partitions every non-padding source token exactly once
and measures the exact cue-space query at every applicable ordinary-attention
layer, L4--64. Game and Neutral are reported separately before comparison.

At L36, where held-out old/fresh cue decodability is strongest in the broad
mid-layer rise, the cue allocates only 2.67% of attention in Game and 3.98% in
Neutral directly to the four raw 1P option lines. The larger candidate-bearing
sources are the four 2P lines (15.68% and 14.24%) and the 2P question stem
(21.54% and 20.73%). This complements the prior exact source result: matching
1P lines write graded old evidence into 2P semantic positions over L32--48,
and the cue then reads those 2P positions. This establishes a plausible
activation path, not causal mediation.

The tasks route history and policy differently at the cue. At L36, Game gives
18.45% of cue attention to the feedback sentence, versus 3.37% in Neutral; the
paired difference is +15.09 percentage points `[+14.43,+15.74]`. Neutral gives
20.25% to the 1P answer-cue/empty-decision-boundary region, versus 12.57% in
Game; the paired Game-minus-Neutral difference is -7.68 points
`[-8.29,-7.07]`. The Game feedback route is large from L28 through L48, so
policy input reaches the cue before the reliably decoded rank transformation
at L50.

The cue also reads the 2P lines in old-rank order. At L48, R1 receives 10.51%
in Neutral and 7.35% in Game; at L52 the values are 10.88% and 6.36%. The
paired Game-minus-Neutral differences are -3.16 points `[-3.92,-2.41]` and
-4.52 points `[-5.57,-3.48]`. The pooled R2--R4 differences include zero at
both layers. Thus the previously observed late attenuation of W1 retrieval is
present at the cue itself and is specifically concentrated on its read of the
2P R1 line.

- [Canonical exhaustive cue-source figure](../../../../figures/qwen36_cue_attention_distribution.png)
- [Full cue-source report and confidence intervals](cue_attention_distribution/REPORT.md)
- [Compact cue-source summary](cue_attention_distribution/summary.json)

### Does the answer cue causally drive the final decision?

A four-condition intervention now tests the cue as a downstream source rather
than merely decoding it. The cue's own residual was held fixed. At every
ordinary-attention layer, all later queries either received the aligned
Game/Neutral donor cue K/V or were blocked from the cue. At every GLA layer,
the corresponding recurrent write was transplanted or removed only for later
tokens. Natural execution and the identical ablation of the immediately
preceding `):` token were controls. All 500 questions and all 64 layers were
included; natural A--D logits and the cue state itself reproduced exactly.

The reciprocal swap gives the clean causal result. On the 249-question
confirmation split, it moves the final ranking 13.2% toward Neutral's natural
task-specific final vector when Neutral cue memory is patched into Game, and
14.8% toward Game in the reverse direction. The discovery values are 10.6%
and 13.1%. When the cue winners disagree, the swap also raises the donor cue
winner's final margin on both splits. Later tokens therefore causally read
condition-specific ranking information stored at the cue.

The corrected cue/colon lesion arms now block downstream ordinary-attention K/V
access as well as GLA writes, while preserving the cue token's own residual.
Complete cue-route ablation changes individual final rankings beyond the
neighboring-colon control: on confirmation, the cue-specific centered-logit
displacement is +0.044 `[+0.035,+0.053]` in Game and +0.033
`[+0.023,+0.043]` in Neutral. It nevertheless does not materially reduce the
main Game--Neutral switching difference. The all-question gap changes by +1.6
points `[-3.6,+6.8]` on discovery and -1.2 `[-5.2,+3.2]` on confirmation;
conflict and no-conflict estimates are likewise small and include zero. By
contrast, reciprocal cue-memory swaps reduce preferential Game switching by
6.0 points on discovery and 7.2 on confirmation. The cue is therefore a
causal, policy-bearing summary channel, but the main behavioral difference does
not require this cue route.

- [Canonical causal cue figure](../../../../figures/qwen36_cue_memory_causality.png)
- [Full causal report](cue_memory_causality/analysis/REPORT.md)
- [Compact causal statistics](cue_memory_causality/analysis/summary.json)

### Exact semantic content carried by evaluation-token writes

The corrected analysis reconstructs the exact ordinary-attention write from
one evaluation token into one 2P destination before applying either lens. The
complete receiver residual is never lensed. Coverage is exhaustive over all ten
tokens in `Your answer was incorrect/lost. Choose the answer again.`, every
ordinary-attention layer (L4, L8, ..., L64), every R1--R4 option line, and the
line-leading space, option letter, colon, semantic wordpieces, and closing
newline separately. Peak layers are selected on 251 discovery questions;
magnitudes and J/R-consensus token lists below use 249 held-out questions.

The task-specific routes are now clear.

**Game (`incorrect`):**

- The literal `incorrect` token writes most strongly into option letters and
  closing newlines at **L32** (RMS 0.0160 and 0.0180). Its readable direction
  evolves from `incorrectly`/`falsely`/`incorrect` at L28, through
  `supposed`/`claimed` at L32, to `failed`/`incorrect` at L36. Error/correction
  language reappears at L44, L52, L56, and L60. The semantic-wordpiece write is
  much smaller (L32 RMS 0.0032), but it shows the same claimed/supposed and
  failed/incorrect progression.
- The period immediately after `incorrect` carries contextualized evaluation
  content rather than merely punctuation. At **L36**, its writes into letters,
  semantic wordpieces, and newlines read out as
  `failed`/`unsuccessful`/`incorrect`/`corrected`. At **L44**, the letter and
  newline writes shift toward `correct`/`correctly`/`correctness`. Its largest
  raw letter and newline writes occur earlier, but those peak directions are
  not consistently interpretable as English policy semantics.
- The later contextualized `the` is a major middle-layer relay. Its write
  changes from `Wrong`/`Incorrect` around **L24--28**, to `incorrect` around
  **L36**, to `answer`/`exactly`/`correct` at **L44**, then
  `pairwise`/`comparisons` at L48 and `option`/`answer` at L52. It peaks at L44
  into both option letters (RMS 0.0146) and semantic wordpieces (0.0088), while
  its newline magnitude peaks at L32 (0.0184).

**Neutral (`lost`):**

- The literal `lost` token supplies little interpretable middle-layer content.
  It becomes a strong, clean late write at **L56--60**: `lost`/`loss`/`lose`
  into letters, semantic wordpieces, and newlines. At L56 the corresponding
  RMS values are 0.0103, 0.0029, and 0.0126.
- The period after `lost` carries an earlier contextualized recovery signal.
  Its L28 writes include `return`; at **L36**, its letter/newline writes read
  out as `restored`/`recovery`/`restore`. This route is substantially smaller
  than Game's middle-layer period and `the` routes, especially into semantic
  wordpieces.
- Neutral's contextualized `the` has no comparably strong or stable
  policy-semantic trajectory. Its writes are small and mostly lexically
  incoherent; this is a genuine absence in the source-specific readout, not a
  consequence of subtracting Game from Neutral.

**Shared instruction relays:** `Choose` writes
`Choose`/`choosing`/`choice` at **L60** into all three main destination types;
`again` writes `again` equivalents primarily at **L56**. These are present in
both tasks, although the `Choose` write is about twice as large in Game. They
are instruction transport, not the minimal policy distinction.

Across the key cells, each R1--R4 rank-specific write has cosine above 0.99
with the rank-mean write. The evaluation semantics are therefore being
broadcast to the same token types in all four 2P options; this analysis does
not reveal rank-specific policy content in the write direction. Structurally,
letters and closing newlines receive the strongest policy-bearing writes,
semantic wordpieces receive smaller but sometimes readable writes, and colons
receive almost none.

This is activation-path attribution: the attention write itself is reconstructed
exactly, but the English J/R-lens interpretation remains descriptive rather
than a causal semantic intervention.

- [Canonical exact-write magnitude and semantic figure](../../../../figures/qwen36_second_presentation_policy_write_semantics.png)
- [Complete exact-write semantic table](policy_write_semantics/policy_write_semantics.json)
- [Write magnitude relative to complete attention updates](../../../../figures/qwen36_second_presentation_policy_write_fraction.png)

## Exact sources of old score, fresh score, and their task-dependent use

The exhaustive source attribution is complete. In both tasks, ordinary
attention to the semantically matching first-presentation line supplies a
graded old-rank signal to the second-presentation semantic residual across
layers 32--48. At layer 32 the exact rank write is nearly identical in Game
`[+0.131,-0.039,-0.224,-0.400]` and Neutral
`[+0.131,-0.040,-0.224,-0.396]`. Thus the task difference is not created by
different initial retrieval of old rank.

Fresh score has no single dominant attention source; MLPs around layers 29--31
make the clearest early component writes. The strongest replicating
task-dependent rank transformation appears at **MLP 49** in the final semantic
token. Its old-score write is `[+1.177,+0.639,+0.410,-0.263]` in Game and
`[+1.386,+0.685,+0.292,-0.432]` in Neutral. The held-out bivalent difference is
+0.296 `[+0.223,+0.371]`.

Accordingly, MLP 49 does not absolutely suppress R1 in Game: it favors R1 in
both tasks, but Neutral reinstates the old leaders more strongly, while Game
shifts relative support toward the lower-ranked candidates. This exact write
localizes the policy divergence but is not yet a causal lesion.

- [Full source-attribution report](score_source_attribution/REPORT.md)
- [Canonical source-attribution figure](../../../../figures/qwen36_second_presentation_score_source_attribution.png)

## Nonlinear audit of the apparent categorical-winner remainder

The earlier all-candidate report used a linear first-pass-score control and
found an additional R1 term. That is not sufficient to distinguish a winner
state from a nonlinear score threshold. The corrected audit therefore modeled
both the candidate's first-pass score and its gap to the best competing
candidate with frozen flexible curves, while retaining display-position
controls.

Under the flexible model, the additional R1 coefficient is -0.161
[-0.434,+0.098] in discovery and +0.183 [-0.115,+0.458] in confirmation. Adding
the R1 indicator worsens held-out prediction by 0.7%. Near-tie R1-minus-R2
contrasts also include zero. The established result is a graded, nonlinear
rank effect; a separate categorical winner state is not established.

- [Full nonlinear audit](categorical_winner_audit/REPORT.md)
- [Canonical audit figure](../../../../figures/qwen36_categorical_winner_nonlinearity_audit.png)

## Causal policy × retrieved-rank factorial

The reciprocal evaluation-period GLA transplant was crossed with joint
blockade of all four matching 1P-to-2P routes, using a joint cyclic blockade as
control. Every GLA layer and every ordinary-attention layer was included. The
factorial shows that the evaluation-period update is causally upstream of the
rank policy, separately in each task:

- **Game:** replacing the `incorrect` period update with Neutral's `lost`
  update changes the matching-route rank contrast by +0.453
  [+0.356,+0.551] logits in discovery and +0.426 [+0.335,+0.519] in
  confirmation. Conflict-trial W1 choice rises from 18.4% to 33.1% on
  confirmation.
- **Neutral:** replacing the `lost` update with Game's `incorrect` update
  changes the same contrast in the opposite direction: -0.773
  [-0.895,-0.644] and -0.727 [-0.848,-0.606]. Conflict-trial W1 choice falls
  from 39.7% to 20.6% on confirmation.

The previously nominated MLP 49 write is not a necessary local mediator.
Restoring the natural recipient MLP-49 output at the four final 2P semantic
tokens removes its local rank-projection difference, but leaves 94% of Game's
and 97% of Neutral's final bivalent effect, as well as about 85% of each
behavioral effect. The policy-conditioned rank state must be present elsewhere
or reconstructed downstream.

- [Full causal report](policy_rank_factorial/analysis/REPORT.md)
- [Canonical causal figure](../../../../figures/qwen36_policy_rank_factorial.png)

## Final-decision position: where the evidence and policy arrive

The cue-space program has now been repeated and extended at the exact final
decision token. Frozen decoders cover every residual layer; attention and exact
source-write maps cover all 16 ordinary-attention layers and every causal-prefix
token; mixer/MLP attribution covers all 64 layers; and the causal follow-up
uses all 500 questions.

Both old 1P and fresh 2P evidence are present at the final token. The
discovery-selected held-out peaks are **L56** for old evidence (`r=.323`) and
**L60** for fresh evidence (`r=.295`). At L56, the decoded old ranking is much
less top-heavy in Game than Neutral: Game minus Neutral for
`R4 - mean(R1,R2)` is `+.263` `[+.144,+.381]`. Ordinary attention, not the MLP,
supplies the replicated task separation at L52 and L56.

Exact source reconstruction and causal swaps distinguish two incoming routes.
The four 2P option lines write old-rank information at L40, but Game/Neutral
swaps there are essentially null; this is mostly shared evidence. At L52/L56,
the contextualized repeated-question and post-list choice-cue states carry the
task-specific old-rank treatment. Swapping these states transfers the final
rank geometry in both directions on both frozen splits. On confirmation,
Neutral scaffold into Game also lowers conflict switching from 82.5% to 78.1%,
though that winner-change result does not replicate on discovery.

Finally, blocking the final query from all four complete 1P option lines across
the full ordinary-attention range L4--64 does not reproducibly remove
preferential Game switching. The final decision therefore receives old rank
primarily through contextualized downstream relays rather than reconstructing
it by directly rereading the raw 1P lines.

- [Full final-position report](final_position_program/REPORT.md)
- [Canonical final-position figure](../../../../figures/qwen36_final_query_attention.png)
