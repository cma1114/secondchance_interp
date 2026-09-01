# Seed-OSS-36B-Instruct TriviaMC behavioral gate

## Design and validity

All 500 frozen TriviaMC questions were run with the model's native chat template. The first assistant turn was empty. In each mapping, Game and Neutral differed only at `incorrect` versus `lost` in `Your answer was ... Choose the answer again.` Every remapped option moved to a different letter.

Primary choice readout: **aggregated A-D argmax**. Answer-only compliance ranged from **98.8% to 100.0%**; all outputs were finite.

As a format-compliance sensitivity check, restricting to questions where Baseline, Game, and Neutral all emitted unrestricted A-D tokens left a same-order gap of **+7.7 [+5.1, +10.3] points** (n=494) and a remapped semantic gap of **+5.0 [+2.0, +8.0] points** (n=498).

## Behavioral result

With the same option order, Game switched semantic answers on **16.0%** of questions and Neutral on **8.4%**; the paired Game-minus-Neutral gap was **+7.6 [+5.2, +10.2] percentage points**.

After every option content moved to a new letter, Game switched semantic answers on **30.6%** and Neutral on **25.2%**; the semantic gap was **+5.4 [+2.4, +8.4] percentage points**.

## What Game changes

In the remapped presentation, Game-minus-Neutral centered evidence by the model's own first-pass rank was W1/W2/W3/W4 = **-1.068 [-1.221, -0.919] / +0.497 [+0.359, +0.636] / +0.421 [+0.296, +0.540] / +0.150 [+0.026, +0.278] logits**.

The old winner's semantic content, now at a new letter, was suppressed by **+1.068 [+0.911, +1.220] logits** in Game relative to Neutral. The old literal letter, now attached to different content, changed by **+0.181 [+0.048, +0.318] logits**. Their semantic-minus-letter contrast was **+0.887 [+0.642, +1.127] logits**.

## Mechanistic follow-up decision

Prespecified gate passed: **True**.

The gate requires a positive remapped semantic switch gap, positive suppression of the semantic old winner, and stronger targeting of that content than of its former literal letter on the frozen confirmation split.
