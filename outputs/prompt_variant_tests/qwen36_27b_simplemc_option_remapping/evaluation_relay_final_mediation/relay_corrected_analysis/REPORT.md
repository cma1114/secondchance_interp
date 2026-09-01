# Corrected evaluation-feedback relay mediation

## Bottom line

Crossing the complete feedback suffix transfers 0.925 [0.909, 0.939] of the paired Neutral task vector into a Game recipient and 0.941 [0.927, 0.955] of the paired Game task vector into a Neutral recipient on confirmation.

Restoring every later post-feedback relay's outgoing ordinary-attention and recurrent-GLA writes mediates 51.8% [49.4%, 54.0%] of the Game-recipient transfer and 41.4% [38.6%, 44.4%] of the Neutral-recipient transfer.

The result replicates on discovery: 53.8% [51.1%, 56.4%] for Game and 43.6% [40.5%, 46.7%] for Neutral.

Thus no single later token region is the policy bottleneck. The nominal joint proportions are lower bounds, not estimates of a physiological bypass: restored tokens retain their source-crossed local outputs, and source-crossed assistant-prefix output can reach the adjacent readout through the unintercepted short GLA convolution.

## Intervention

The complete `incorrect/lost . Choose the answer again .` suffix was crossed between Game and Neutral. For each relay condition, that region's downstream ordinary-attention K/V and recurrent GLA k/v/g/β writes were restored from an exact clean duplicate of the recipient task while the relay token's own source-crossed residual was retained. Every applicable layer and every token after feedback and before the final answer position was covered.

## Confirmation results

`Transfer reduction` is the fraction of the paired natural donor-task vector removed by restoring a relay. `Mediated proportion` divides that reduction by the complete source-crossover transfer.

| Relay restored | Game remaining | Game reduction | Game mediated | Neutral remaining | Neutral reduction | Neutral mediated |
|---|---:|---:|---:|---:|---:|---:|
| second answer instruction | 0.762 [0.728, 0.793] | 0.163 [0.139, 0.188] | 17.6% [15.0%, 20.5%] | 0.824 [0.801, 0.847] | 0.117 [0.096, 0.138] | 12.5% [10.2%, 14.7%] |
| second question stem | 0.668 [0.622, 0.714] | 0.257 [0.216, 0.298] | 27.8% [23.3%, 32.4%] | 0.767 [0.727, 0.802] | 0.174 [0.144, 0.209] | 18.5% [15.3%, 22.2%] |
| second option lines | 0.669 [0.625, 0.709] | 0.256 [0.216, 0.299] | 27.7% [23.4%, 32.3%] | 0.735 [0.701, 0.768] | 0.206 [0.177, 0.236] | 21.9% [18.8%, 25.1%] |
| second choice cue and query | 0.668 [0.632, 0.703] | 0.257 [0.224, 0.294] | 27.8% [24.2%, 31.8%] | 0.716 [0.674, 0.756] | 0.226 [0.188, 0.263] | 24.0% [19.9%, 28.0%] |
| final assistant prefix | 0.650 [0.619, 0.685] | 0.275 [0.239, 0.308] | 29.7% [26.0%, 33.2%] | 0.752 [0.717, 0.787] | 0.189 [0.157, 0.219] | 20.1% [16.6%, 23.3%] |
| all post feedback relays | 0.446 [0.425, 0.468] | 0.479 [0.455, 0.501] | 51.8% [49.4%, 54.0%] | 0.551 [0.521, 0.580] | 0.390 [0.364, 0.416] | 41.4% [38.6%, 44.4%] |

## Controls and scope

The actual `cache_restored_no_source_swap` scenario reproduced the same-batch natural run with maximum absolute raw A-D-logit error 0; after the trusted-natural correction its maximum error was 0.

Across the 8 natural/source/interception scenarios shared with the historical run, the corrected raw and corrected-logit arrays are bit-for-bit identical (maximum errors 0 and 0). The audit therefore found a vacuous advertised control, not a change in the causal mediation estimates.

The route inventory is not exhaustive over every GLA cross-position mechanism. Qwen3.6 applies a short causal depthwise convolution to GLA q/k/v before the intercepted delta-rule update. Those convolution states were not restored. Moreover, downstream-only restoration deliberately keeps each restored token's source-crossed local output. At the final assistant prefix, immediately adjacent to the readout, that output can leak through the unintercepted convolution. The joint mediation proportions therefore likely understate relay mediation, and the transfer that survives joint restoration cannot be interpreted as a bypass fraction or assigned among persistent GLA memory and direct ordinary-attention reads. A convolution-safe joint control is required.

Discovery estimates, raw switch rates, conflict/no-conflict outcomes, W1 rates, bivalent rank effects, and all validation controls are retained in `summary.json`.
