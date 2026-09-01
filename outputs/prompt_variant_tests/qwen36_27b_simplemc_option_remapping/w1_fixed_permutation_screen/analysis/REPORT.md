# W1-fixed option-permutation feasibility screen

## Bottom line

The identity companion reproduced trusted choices on 98.4% of questions; maximum A-D logit discrepancy was 0.1250.

Keeping W1 at the identical displayed letter while permuting only the other three options produced 222/500 questions where W1 was selected in the identity ordering but lost in at least one alternative ordering.
The frozen discovery/confirmation counts are 115/107.

## By original W1 letter

- A: 77/237 eligible (32.5%).
- B: 44/72 eligible (61.1%).
- C: 64/104 eligible (61.5%).
- D: 37/79 eligible (46.8%).

For W1=A, the W1 option line and every preceding token are byte-identical across all six prompts. Any winner-status difference is therefore necessarily constructed only after the model encounters later competitors.

The eligible-pair artifact freezes one strongest unchosen permutation and, when available, one nonidentity chosen-status control for each question. These pairs are the appropriate cohort for locating and causally transplanting candidate strength.

## Why this is useful

The feasibility gate passed comfortably: 115 discovery and 107 held-out
confirmation questions provide a same-content, same-letter, same-position
contrast. The cleanest subset is W1=A (41 discovery, 36 confirmation). For
those 77 questions, the complete prompt prefix through the A option line is
identical across the chosen and unchosen first presentations. The change in
whether A wins must therefore be computed only after the model encounters the
later competitors; it cannot be a changed representation already written at
the A line.

This sharpens the missing-mechanism hypothesis. The option line can carry A's
semantic content and absolute plausibility, while a separate state at the
first-decision boundary records which displayed option actually won. Previous
fixed-A donor experiments held the first decision equal to A in both histories,
so they could not test that second signal.

## How much ordering sensitivity is this?

The 222-question count is an “at least once across five alternative
permutations” statistic. Among the 492 questions whose same-batch identity
companion selected W1, an individual alternative ordering displaced W1 on
581/2,460 trials (23.6%). Of the 222 sensitive questions, the median was two
losing permutations and the mean was 2.62; 270/492 questions remained stable
under all five alternatives.

Sensitivity was strongly letter-dependent. The per-alternative loss rates
were 14.5% for fixed A, 34.2% for B, 36.0% for C, and 25.1% for D. This is
consistent with the model's previously observed answer-letter asymmetry, but
the clean A subset still shows substantial sensitivity despite keeping the A
line and its entire prefix identical.

Most sensitive cases were comparatively weak Baseline decisions: their median
identity W1 margin was 0.375 logits, versus 1.000 for questions stable under all
five permutations. The result is not only a near-tie artifact, however: 40
sensitive questions had identity margins above 1 logit and seven exceeded 2
logits. Mean A-D probability mass was 99.995%, ruling out malformed or
non-answer outputs as the source.

## Frozen next causal test

Use only the 77 W1=A pairs. Keep the recipient prompt and second presentation
fixed. At the empty first-answer decision boundary, replace the recipient's
ordinary-attention K/V with K/V from the matched donor in which the identical A
content was selected. Compare this with recipient-state reinsertion, separately
in Game and Neutral. The primary endpoints are the centered logit and choice
rate of that semantic A content at its second-presentation letter.

The discriminating prediction is directional: importing the “A was selected”
boundary state should make Game avoid that semantic content but make Neutral
retain it. A movement in the same direction in both conditions would instead
indicate generic answer evidence rather than previous-answer binding. Start
with all ordinary-attention blocks jointly; localize by depth only if this
joint prerequisite succeeds on both frozen splits.
