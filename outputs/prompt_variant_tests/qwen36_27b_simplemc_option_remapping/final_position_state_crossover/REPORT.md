# Final-position state crossover

## Bottom line

The final answer position does contain a causally effective task state. Swapping its exact residual state between paired Game and Neutral prompts has almost no task-directed effect through the early and middle layers, becomes practically visible around layer 48, transfers roughly one third of the paired task difference through layers 52--60, and then jumps to 82--85% at layer 63. Layer 64 is the exact donor-state positive control, not a meaningful localization result.

The component crossover identifies the writer class: replacing every final-position sequence-mixer write (ordinary attention or GLA) reproduces essentially the complete paired donor task vector and donor switch rate in both directions and both frozen splits. Replacing every final-position MLP write transfers only 10--22% of the continuous task vector and does not transfer behavior. Thus the late final-position task state is written overwhelmingly by sequence mixers, not by the local MLPs.

This closes the final receiver question, but not the upstream pathway. It does not by itself identify which evaluation token or which 2P/scaffold relay supplies the decisive mixer inputs.

## Method

For each question, Game and Neutral prompts were paired with identical question and option content. The intervention replaced only the residual stream at the exact final answer position after each layer 1--64 with the paired task's state, then allowed the remaining layers to run naturally. Separate crossovers replaced all 64 final-position sequence-mixer outputs, all 64 MLP outputs, or both. Every reported output used a same-batch natural control and the frozen discovery/confirmation split.

The transfer fraction projects the intervention-induced centered A--D logit change onto that question's complete paired natural Game-versus-Neutral difference. Zero means no movement toward the paired donor; one means exact reproduction of the donor's natural task-specific A--D ranking.

The prespecified all-mixer localization gate **passed**.

## When the final-position task state becomes causally effective

Confirmation estimates are shown below; discovery follows the same trajectory.

| Swap after layer | Game receives Neutral: transfer | Game switch rate | Neutral receives Game: transfer | Neutral switch rate |
|---:|---:|---:|---:|---:|
| 32 | 0.017 [0.008, 0.025] | 61.0% | 0.009 [0.003, 0.015] | 45.8% |
| 36 | 0.027 [0.017, 0.038] | 61.4% | 0.028 [0.020, 0.036] | 45.8% |
| 40 | 0.043 [0.031, 0.056] | 59.0% | 0.026 [0.019, 0.034] | 45.8% |
| 44 | 0.055 [0.041, 0.070] | 59.0% | 0.044 [0.035, 0.053] | 45.8% |
| 48 | 0.140 [0.122, 0.159] | 57.4% | 0.167 [0.154, 0.180] | 45.8% |
| 52 | 0.272 [0.251, 0.295] | 54.2% | 0.314 [0.291, 0.337] | 49.4% |
| 56 | 0.335 [0.310, 0.360] | 51.8% | 0.352 [0.327, 0.375] | 50.6% |
| 60 | 0.359 [0.328, 0.387] | 51.4% | 0.350 [0.329, 0.369] | 51.0% |
| 61 | 0.436 [0.411, 0.462] | 51.0% | 0.394 [0.366, 0.420] | 51.8% |
| 62 | 0.464 [0.439, 0.488] | 49.4% | 0.401 [0.372, 0.428] | 51.8% |
| 63 | 0.817 [0.798, 0.837] | 45.0% | 0.845 [0.822, 0.869] | 61.8% |
| 64 | 1.000 [1.000, 1.000] | 45.0% | 1.000 [1.000, 1.000] | 62.7% |

Natural confirmation switch rates are 62.7% in Game and 45.0% in Neutral. After a layer-63 state crossover, Game falls to 45.0% and Neutral rises to 61.8%, already nearly reproducing the paired donor behavior before the exact layer-64 donor-state control.

Layer 63 is a GLA layer; layer 64 is ordinary attention. The large layer-63 jump therefore localizes the last nontrivial consolidation step to the layer-63 recurrent sequence mixer, while the exact layer-64 crossover simply installs the donor's finished state.

## Which component writes the task state

| Split | Recipient | All mixers transfer (95% CI) | All MLPs transfer (95% CI) | Joint transfer | Natural switch | Mixer-swap switch | MLP-swap switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| discovery | Game | 1.000 [1.000, 1.001] | 0.137 [0.062, 0.207] | 1.000 | 62.5% | 49.0% | 65.3% |
| discovery | Neutral | 0.999 [0.997, 1.000] | 0.224 [0.151, 0.292] | 1.000 | 49.0% | 62.5% | 49.4% |
| confirmation | Game | 1.000 [0.999, 1.002] | 0.102 [0.023, 0.175] | 1.000 | 62.7% | 45.4% | 67.1% |
| confirmation | Neutral | 0.999 [0.998, 1.000] | 0.196 [0.112, 0.272] | 1.000 | 45.0% | 62.7% | 47.0% |

The all-mixer result is a sufficiency/crossover result over the entire final-position computation, not a claim that every mixer layer is individually necessary. The joint mixer-plus-MLP condition is an algebraic reconstruction control: because those are the two additive writer families at the final position, replacing both necessarily reconstructs the paired donor state. It is not independent evidence beyond the exact reconstruction check.

The MLP crossover changes continuous logits modestly but fails the behavioral-direction test: on confirmation it moves Game from 62.7% to 67.1% switching, away from the Neutral donor's 45.0%; Neutral moves only from 45.0% to 47.0%. In contrast, the all-mixer swap moves Game to 45.4% and Neutral to 62.7%, essentially the donor rates.

## Validation

- 500/500 questions completed with finite outputs.
- 17,000 complete model forwards; no omitted layers.
- Same-batch corrected natural-logit error: exactly 0.
- Layer-64 donor reconstruction maximum absolute A--D logit error: exactly 0.
- Discovery and confirmation reproduce the same qualitative boundary and component results.

## Remaining mechanistic gap

We now know that task-specific information becomes causally sufficient at the final position across layers 48--63 and that sequence mixers write it. We still need a direct mediated path showing which evaluation-token state is read into which 2P/scaffold relay, and which later mixer writes that relayed state into the final position. That evaluation-to-relay-to-final pathway is the next mechanistic experiment; no conditional suffix-localization run was launched automatically.

Canonical figure: [qwen36_final_position_state_crossover.png](/Users/christopherackerman/repos/secondchance_interp/figures/qwen36_final_position_state_crossover.png)

Machine-readable estimates and confidence intervals are in `summary.json`.
