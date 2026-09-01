# Where the confidence dose-response travels

Slope of the per-question Game-minus-Neutral old-winner push on z-scored
first-pass confidence, inside each causal cell of the canonical
history/decision factorial. Each natural slope exactly reproduces the
audited dose-response value (asserted), and each baseline ranking exactly
reproduces the factorial's stored rank order (asserted).

| Model | Dataset | Natural | First-decision cut | Matching-line cut | Joint |
|---|---|---:|---:|---:|---:|
| Qwen3.6-27B | SimpleMC | +0.727 [+0.595, +0.851] | +0.786 [+0.640, +0.920] | +0.169 [+0.107, +0.234] | +0.175 [+0.063, +0.286] |
| Qwen3.6-27B | TriviaMC | +1.260 [+1.190, +1.322] | +1.410 [+1.310, +1.504] | +0.240 [+0.179, +0.298] | +0.021 [-0.082, +0.125] |
| Seed-OSS-36B | SimpleMC | +0.947 [+0.799, +1.093] | +0.929 [+0.786, +1.077] | +0.107 [+0.003, +0.207] | +0.110 [+0.004, +0.215] |
| Seed-OSS-36B | TriviaMC | +0.318 [+0.182, +0.457] | +0.332 [+0.198, +0.462] | +0.204 [+0.085, +0.320] | +0.212 [+0.093, +0.328] |
| Gemma-4-31B | SimpleMC | +0.557 [+0.456, +0.660] | +0.565 [+0.461, +0.669] | +0.018 [-0.051, +0.088] | +0.041 [-0.023, +0.107] |
| Gemma-4-31B | TriviaMC | +0.099 [-0.017, +0.214] | +0.078 [-0.034, +0.185] | +0.036 [-0.043, +0.114] | +0.021 [-0.080, +0.123] |

Severing every outgoing signal from the would-be first-answer position
leaves the confidence scaling untouched in all six cells. Blocking the
matching option-line reads collapses it by 77-97% in four of the five
cells that show scaling (Qwen both datasets, Seed SimpleMC, Gemma
SimpleMC) and partially on Seed TriviaMC. In Qwen TriviaMC the remnant
left by the matching cut (+0.240) is removed by additionally cutting the
first-decision position (+0.021): Qwen's backup route through that
position carries graded confidence only once the line route is gone.
The graded prior confidence the Game policy consumes therefore travels
with the retrieved option-line scores themselves, not as a summary
stored at the answer position.
