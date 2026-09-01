# Prospective final-answer decoding at the final decision position

This analysis asks whether the model's exact eventual four-answer score pattern is linearly available at the final decision position before it is aligned with the fixed output readout used by JLens. It fits one shared Game+Neutral decoder and separate Game-only and Neutral-only decoders at every post-block residual L1–L64. Every basis is evaluated on both conditions. All fitting, centering, and ridge selection use frozen discovery questions; all reported curves use held-out confirmation questions.

The cross-task curves are the cleanest test of representational basis: the Neutral coefficient basis is applied to Game residuals and the Game basis to Neutral residuals. The shared fit has twice as many training rows as a task-specific fit, so shared-versus-specific performance alone is not interpreted as a basis test. Stable displayed-letter means are estimated on discovery questions and removed before semantic-rank analysis.

## Main findings

The prospective final four-answer pattern is linearly decodable well before it becomes visible through the model's fixed output lens. The effect is especially clear on TriviaMC: held-out shared-decoder cosine is already 0.369 at L32 and 0.676 at L40, while fixed-JLens cosine is approximately zero at both layers. Thus the late JLens rise is primarily a late alignment of an earlier linear representation with the output vocabulary, not the first appearance of all answer-relevant information at the final decision position.

The representation is predominantly condition-general. A decoder trained in Game and evaluated in Neutral, or vice versa, remains close to a decoder trained and evaluated in the same condition, particularly in the late layers. The modest cross-condition penalty shows that the bases are not perfectly identical, but there is no large per-condition-versus-transfer collapse that would support wholly different Game and Neutral answer-code bases.

Chance is reported two ways. A fully shuffled question-to-target assignment is the absolute null and is approximately zero. The stronger W1-matched null shuffles paired Game/Neutral final targets only among confirmation questions with the same displayed first-presentation winner letter. It therefore preserves the easiest old-winner structure while destroying question-specific final geometry. The dotted curve and band in the first panel show this stronger null.

The more specific eventual-switch ordering is later than the broad final-pattern decodability: on held-out switch trials, decoded R2 exceeds R1 reliably only around L44–L48. On the paired questions where Game switches but Neutral stays, the across-condition mean remains R1-favoring; it never develops a sustained positive R2−R1 interval. The Game-minus-Neutral R2−R1 difference, however, is decodable by L34–L35. This rejects the proposed shared-R2-first sequence on that selected subset: an early task-dependent difference exists in a non-output-aligned linear basis and is later rotated and amplified into answer-logit space.

## SimpleMC

Discovery n=251; confirmation n=249; paired Game-switch/Neutral-stay confirmation subset n=39. [Figure](../../../../../figures/prospective_decoding/qwen36_simplemc_prospective_answer_decoding.png)

Selected held-out mean cosine similarities (Shared / Matched task / Cross-task / fixed JLens):

- L32: 0.131 / 0.123 / 0.110 / 0.007
- L40: 0.403 / 0.392 / 0.340 / 0.047
- L44: 0.600 / 0.601 / 0.541 / 0.089
- L48: 0.803 / 0.826 / 0.756 / 0.485
- L52: 0.889 / 0.890 / 0.850 / 0.742
- L56: 0.915 / 0.918 / 0.897 / 0.883
- L60: 0.919 / 0.920 / 0.891 / 0.893
- L64: 0.963 / 0.954 / 0.934 / 1.000

At L32, the absolute shuffle median is -0.004 and the stronger W1-matched shuffle median is 0.049 [-0.005, 0.102]. At L40 they are 0.000 and 0.170 [0.116, 0.225].

The matched-minus-cross-task cosine penalty is 0.070 [0.049, 0.093] at L48 and 0.021 [0.010, 0.032] at L56.

On held-out switch trials, the first sustained layer at which the shared decoder's R2−R1 95% CI is positive is L46 in Game and L44 in Neutral. These are descriptive outcome slices, not causal evidence about why switching occurred.

In the paired Game-switch/Neutral-stay subset, the shared R2−R1 component first has a sustained positive CI at no layer (the interval never stays positive for three layers); the Game−Neutral R2−R1 component does so at L35. Exact trajectories and intervals are in `summary.json`.

## TriviaMC difficulty-filtered

Discovery n=250; confirmation n=250; paired Game-switch/Neutral-stay confirmation subset n=17. [Figure](../../../../../figures/prospective_decoding/qwen36_triviamc_prospective_answer_decoding.png)

Selected held-out mean cosine similarities (Shared / Matched task / Cross-task / fixed JLens):

- L32: 0.369 / 0.362 / 0.318 / -0.007
- L40: 0.676 / 0.665 / 0.589 / 0.016
- L44: 0.769 / 0.773 / 0.738 / 0.028
- L48: 0.859 / 0.878 / 0.838 / 0.554
- L52: 0.919 / 0.927 / 0.916 / 0.789
- L56: 0.953 / 0.954 / 0.943 / 0.924
- L60: 0.957 / 0.956 / 0.946 / 0.939
- L64: 0.989 / 0.984 / 0.980 / 1.000

At L32, the absolute shuffle median is 0.002 and the stronger W1-matched shuffle median is 0.253 [0.216, 0.292]. At L40 they are 0.003 and 0.462 [0.428, 0.497].

The matched-minus-cross-task cosine penalty is 0.040 [0.026, 0.054] at L48 and 0.011 [0.005, 0.017] at L56.

On held-out switch trials, the first sustained layer at which the shared decoder's R2−R1 95% CI is positive is L48 in Game and L48 in Neutral. These are descriptive outcome slices, not causal evidence about why switching occurred.

In the paired Game-switch/Neutral-stay subset, the shared R2−R1 component first has a sustained positive CI at no layer (the interval never stays positive for three layers); the Game−Neutral R2−R1 component does so at L34. Exact trajectories and intervals are in `summary.json`.

## Scope

This is activation/decoding evidence. Earlier held-out decoding than JLens shows that the eventual answer pattern is linearly present before it is output-aligned; it does not prove that the decoded direction is causally used. A late decoder onset is stronger evidence for late linear construction, but it cannot rule out an earlier nonlinear representation. Switch-conditioned panels are selected by the eventual answer and cannot identify the cause of switching.
