# Per-condition attribution of the confidence dose-response

For each condition separately: the old winner's centered final logit minus
its centered first-presentation logit, and that change's slope on z-scored
first-pass confidence. The Game-minus-Neutral difference of the two slopes
reproduces the primary dose-response slope exactly (asserted). Per-condition
slopes include shared second-reading re-scoring, which the difference
cancels; they attribute, they do not adjust. Descriptive only.

| Model | Dataset | Game: mean change / slope on confidence | Neutral: mean change / slope on confidence |
|---|---|---:|---:|
| Qwen3.6-27B | SimpleMC | -0.716 / -0.798 [-0.941, -0.642] | -0.145 / -0.071 [-0.161, +0.016] |
| Qwen3.6-27B | TriviaMC | -2.643 / -2.308 [-2.440, -2.164] | -1.010 / -1.048 [-1.163, -0.927] |
| Seed-OSS-36B | SimpleMC | -2.233 / -1.105 [-1.267, -0.947] | -0.715 / -0.158 [-0.252, -0.070] |
| Seed-OSS-36B | TriviaMC | -2.794 / -0.203 [-0.378, -0.025] | -0.535 / +0.115 [-0.021, +0.248] |
| Gemma-4-31B | SimpleMC | -0.621 / +0.041 [-0.138, +0.205] | +0.285 / +0.598 [+0.420, +0.766] |
| Gemma-4-31B | TriviaMC | -0.301 / -0.001 [-0.295, +0.294] | +0.707 / +0.098 [-0.195, +0.396] |

Reading: only Qwen shows Game-side confidence scaling on both datasets
(with Neutral flat on SimpleMC and far shallower on TriviaMC). Seed shows
the same pattern on SimpleMC only. Gemma's Game slopes are flat; its
SimpleMC Neutral slope is positive, so its differential dose-response may
be Neutral-side reinstatement, supported in one cell only and therefore a
hypothesis, not a finding.
