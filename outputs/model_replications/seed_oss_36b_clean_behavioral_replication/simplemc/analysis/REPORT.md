# Seed-OSS-36B-Instruct SimpleMC behavioral gate

## Design and validity

All 500 frozen SimpleMC questions were run with the model's native chat template. The first assistant turn was empty. In each mapping, Game and Neutral differed only at `incorrect` versus `lost` in `Your answer was ... Choose the answer again.` Every remapped option moved to a different letter.

Primary choice readout: **aggregated A-D argmax**. Answer-only compliance ranged from **92.6% to 100.0%**; all outputs were finite.

As a format-compliance sensitivity check, restricting to questions where Baseline, Game, and Neutral all emitted unrestricted A-D tokens left a same-order gap of **+15.4 [+11.4, +19.4] points** (n=448) and a remapped semantic gap of **+9.2 [+5.8, +12.6] points** (n=468).

## Behavioral result

With the same option order, Game switched semantic answers on **31.0%** of questions and Neutral on **16.8%**; the paired Game-minus-Neutral gap was **+14.2 [+10.4, +18.2] percentage points**.

After every option content moved to a new letter, Game switched semantic answers on **63.6%** and Neutral on **55.0%**; the semantic gap was **+8.6 [+5.4, +12.0] percentage points**.

## What Game changes

In the remapped presentation, Game-minus-Neutral centered evidence by the model's own first-pass rank was W1/W2/W3/W4 = **-0.585 [-0.707, -0.461] / +0.217 [+0.092, +0.342] / +0.298 [+0.171, +0.422] / +0.070 [-0.043, +0.183] logits**.

The old winner's semantic content, now at a new letter, was suppressed by **+0.585 [+0.461, +0.713] logits** in Game relative to Neutral. The old literal letter, now attached to different content, changed by **-0.143 [-0.258, -0.026] logits**. Their semantic-minus-letter contrast was **+0.728 [+0.545, +0.914] logits**.

## Mechanistic follow-up decision

Prespecified gate passed: **True**.

The gate requires a positive remapped semantic switch gap, positive suppression of the semantic old winner, and stronger targeting of that content than of its former literal letter on the frozen confirmation split.
