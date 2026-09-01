# Qwen3.6-27B TriviaMC complete-feedback-suffix policy crossover

## What was tested

For every frozen TriviaMC question, the Game and Neutral prompts differ only at `incorrect` versus `lost`. We crossed the downstream state emitted by all seven feedback-suffix tokens (`incorrect/lost . Choose the answer again .`) from Game into Neutral and from Neutral into Game. The edit covers every ordinary-attention K/V write and every recurrent GLA write. It preserves each source token's own local output, so this is a causal test of information sent from the feedback suffix to later tokens, not a residual replacement at the source tokens.

The natural, real same-task identity, and reciprocal crossover scenarios were the only conditions. No individual-token, layer-localization, matching-history, or Step-4 factorial condition was included.

## Primary result

The prespecified replication gate **passed**.

| Frozen split | Recipient | Donor state | Task-vector transfer (95% CI) |
|---|---|---|---:|
| Discovery | Game | Neutral | +0.920 [+0.907, +0.932] |
| Discovery | Neutral | Game | +0.933 [+0.921, +0.944] |
| Confirmation | Game | Neutral | +0.905 [+0.892, +0.918] |
| Confirmation | Neutral | Game | +0.920 [+0.905, +0.934] |

A value of 1 would mean that, along the measured natural Game-versus-Neutral A-D scoring difference, the recipient moved all the way to its paired donor. A value of 0 would mean no movement in that donor-policy direction.

## Confirmation secondary readouts

| Recipient | Natural old-W1 choice | Opposite-task suffix old-W1 choice | Change |
|---|---:|---:|---:|
| Game | +68.4 [+62.4, +74.0]% | +74.0 [+68.4, +79.2]% | +5.6 [+1.6, +10.0] pp |
| Neutral | +73.2 [+67.6, +78.4]% | +69.2 [+63.6, +74.8]% | -4.0 [-8.0, +0.0] pp |

Opposite-task suffix minus natural candidate-centered logit changes by old rank:

| Recipient | W1 | W2 | W3 | W4 |
|---|---:|---:|---:|---:|
| Game | +0.621 [+0.533, +0.707] | -0.199 [-0.286, -0.110] | -0.187 [-0.262, -0.109] | -0.235 [-0.310, -0.161] |
| Neutral | -0.656 [-0.743, -0.567] | +0.210 [+0.120, +0.298] | +0.207 [+0.131, +0.283] | +0.239 [+0.164, +0.314] |

## Interpretation

This experiment asks whether the contextualized feedback suffix is sufficient to send the task policy forward on a new dataset. Positive reciprocal transfer on both frozen halves means that replacing only what those seven source tokens make available to later computation moves final answer scoring toward the opposite task's natural pattern. It does not by itself identify which later token receives the state or how the state interacts with retrieved candidate rank; that is the separate Step-4 question.

Discovery transfer estimates were 0.920 into Game and 0.933 into Neutral. Confirmation estimates were 0.905 and 0.920, respectively.

## Validation and scope

- Raw same-task identity maximum absolute error: 0.00000000.
- Corrected natural maximum absolute error to trusted Step 1: 0.00000000.
- Corrected identity maximum absolute error to trusted Step 1: 0.00000000.
- All 500 questions completed; every saved logit was finite; all audited source spans contained seven contiguous token positions.
- GLA recurrent writes are crossed, but the short local GLA convolution is not. The causal claim is therefore about the complete intercepted downstream output state, not about every possible architectural channel.
- Correctness is not an endpoint.

See `figures/qwen36_triviamc_feedback_suffix_step3.png` and `summary.json`.
