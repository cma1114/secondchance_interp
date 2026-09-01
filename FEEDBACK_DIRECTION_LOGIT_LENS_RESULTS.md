# Logit lens of the Game-minus-neutral feedback direction

> **Contaminated direction.** The neutral activations used to construct this
> direction contain the unintended literal system-prefix token `None`. The
> lens correctly exposes that contamination: `None` variants are its strongest
> final-layer neutral associations. This artifact is useful diagnostically but
> the direction is not interpretable as a clean incorrect-feedback signal.

## Method

For each residual readout (l), the saved direction is

\[
v_l = \operatorname{unit}\left(\mathbb{E}[h_l^{Game}-h_l^{Neutral}]\right).
\]

I passed each (v_l) through Qwen's actual final RMSNorm and full-vocabulary
unembedding head:

\[
z_l = W_U\,\operatorname{RMSNorm}_{final}(v_l).
\]

This was done for all 65 readouts and all 248,320 vocabulary tokens from the
exact checkpoint revision used in the experiment,
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. The absolute logit scale is not
meaningful because the input is a unit direction rather than a naturally
scaled residual state. Token ordering, sign, and changes across layers are the
interpretable quantities.

Positive tokens align with Game relative to neutral; negative tokens align
with neutral relative to Game.

## Result

The direction becomes semantically interpretable, but mostly as a representation
of the **literal prompt contrast**, not as a clean abstract control signal for
switching.

| Readout | Highest or otherwise diagnostic positive tokens | Diagnostic negative tokens | Interpretation |
|---:|---|---|---|
| 8–24 | Mostly multilingual fragments, names, and code tokens | Likewise incoherent | Game and neutral are perfectly linearly separable here, but the mean-difference direction is not aligned with a readable vocabulary concept. |
| 30 | `撤回` (“withdraw/retract”), `撤销` (“revoke/cancel”); `still` is positive rank 342 | Mostly incoherent | Weak reversal/withdrawal flavor, but no clean feedback concept at the main causal readout. |
| 36 | `-errors`, `LastError`, `Restart`, `retry`, `补救` (“remedy”), `认错` (“admit error”), `remaining` | Mostly incoherent | Error/retry semantics first become clearly visible. |
| 48 | `remaining`, `retry`, `.retry`, `改正`/`更正` (“correct/rectify”), “other” tokens | `Preference`, `preference`, `choice` | A retry/correction/remaining-alternatives representation is readable. |
| 56 | `remaining`, `_attempts`, `Retry`, `Wrong`, `correct`, `wrong`, `重试` (“retry”) | `prediction`, `predicted`, `NO` | Explicit negative-feedback and retry language. |
| 64 | `remaining` (rank 3), `still` (6), `previous` (11), `your` (15), `runner` (20), `correct` (45); `incorrect` rank 76 and `wrong` rank 197 | ` None` (negative rank 1), `None` (2), followed by other `None` variants | The final direction is an almost literal encoding of the Game wording versus the neutral condition's `None` system-prefix artifact. |

The final-layer extremes are particularly revealing:

| Positive/Game-aligned token | Rank | Lens logit |
|---|---:|---:|
| `remaining` | 3 | +7.08 |
| `still` | 6 | +6.92 |
| `previous` | 11 | +6.54 |
| `your` | 15 | +6.38 |
| `runner` | 20 | +6.21 |
| `correct` | 45 | +5.77 |
| `incorrect` | 76 | +5.46 |
| `wrong` | 197 | +4.94 |

| Negative/neutral-aligned token | Negative rank | Lens logit |
|---|---:|---:|
| ` None` | 1 | -5.88 |
| `None` | 2 | -5.72 |
| `.None` | 3 | -4.60 |
| tab + `None` | 4 | -4.53 |
| `,None` | 5 | -4.27 |

## Relation to the causal result

This does **not** rescue the single-direction causal hypothesis.

The direction steered at readout 30 is almost orthogonal to the readable final
direction: cosine similarity between (v_{30}) and (v_{64}) is 0.003. Its
top vocabulary tokens are largely incoherent, although a few withdrawal terms
appear. At readout 36, where steering had its largest categorical effect,
retry/error/remedy tokens are present, which is a more interesting alignment.
But the same readout-36 intervention still failed sufficiency in neutral and
changed only a handful of answers.

The layerwise directions evolve substantially rather than representing one
persistent vector carried through the network. Adjacent-layer cosine is around
0.77–0.95 at the inspected layers, while cosine with the final direction is
near zero through readout 36 and only rises late (0.149 at readout 48 and 0.326
at readout 56).

Nor does the direction directly encode a letter-independent instruction to
depress whichever answer currently leads. At readout 30 its canonical letter
scores are A = -1.01, B = -0.27, C = -1.49, and D = +1.00. Those are fixed
letter associations, not the dynamic pattern “negative on this question's
winner.” The winner suppression observed after steering must therefore arise
from downstream, context-dependent processing of the perturbation rather than
from directly unembedding the direction as a winner-suppression logit vector.

The strongest interpretable feature is the literal token `None`. In the
faithful historical neutral prompt, the system prefix begins with the string
`None`; the Game prefix instead says the previous answer was incorrect. The
final-layer direction therefore exposes exactly the concern raised by the
causal results: the probe can perfectly identify prompt condition using lexical
and formatting information that need not be the mechanism producing strategic
switching.

## Conclusion

The lens says that the direction contains a real progression toward
error/retry/correction semantics, especially from readout 36 onward. But by the
final layer it is heavily dominated by surface prompt identity—most starkly
Game-associated wording versus the neutral `None` marker. It is not aligned
with A–D in an answer-invariant way, and the readable final direction is not the
same geometric direction that produced the small readout-30 causal effect.

So the most accurate interpretation is: **a layer-evolving prompt-state
contrast with increasingly explicit retry/error semantics, not a clean
standalone “suppress the leader and switch” feature.**

## Artifacts

- Full 65 × 248,320 lens matrix:
  `outputs/mechanistic/qwen36_27b_simplemc/analysis/feedback_direction_lens/feedback_direction_vocab_logits.npz`
- Top 50 positive and negative tokens at every readout:
  `outputs/mechanistic/qwen36_27b_simplemc/analysis/feedback_direction_lens/feedback_direction_top_tokens.csv`
- Selected-readout JSON:
  `outputs/mechanistic/qwen36_27b_simplemc/analysis/feedback_direction_lens/feedback_direction_logit_lens.json`
