# Option-newline selected-answer probe

## Design

At every post-block readout, a single linear ranker scores the four residuals at the exact option-closing newline positions previously used to construct content-aligned option directions. It is trained on 251 discovery questions across all six option permutations and tested on 249 entirely held-out questions. Static A-D position means are removed using discovery data. This probe establishes candidate-value decodability at the newline; it does not establish that the newline contains a clean mapping-invariant semantic code. Earlier corrected cross-mapping tests found only modest content geometry around layers 32--43 and strong late displayed-letter/position structure.

## Results

The collector reproduced the prior six-permutation choices on 100.00% of presentations (maximum A-D logit difference 0).

The letter-only Baseline chose A and achieved 51.9% [46.7, 56.9] on held-out questions.
The strongest descriptive held-out probe readout was 53 with 64.9% [60.5, 69.1] top-1 accuracy. Final-readout accuracy was 62.0% [57.8, 66.3].

At readout 53, the paired improvement over the letter-only predictor was 13.0
percentage points [5.3, 20.8]. The selected option also exceeded the
highest-scoring rejected option by 3.84 probe-score units [2.77, 4.96]. Thus
the decoder is using question- and mapping-specific information beyond the
model's large absolute preference for A.

The exact matched selectedness analysis retained 107 held-out same-content/same-letter pairs: A=36, B=24, C=31, D=16.

Panel C asks the particularly strict question: does the score attached to the same W1 option at the same displayed letter change when distractor ordering makes it win versus lose? W1=A is a built-in causal-prefix sanity check: its option-line residual is token-for-token identical before any later distractor is seen, so any A difference should be exactly zero.

It does. At readout 53, the score for the identical W1 content at the identical
letter was 4.36 units higher [3.06, 5.79] when W1 won than when it lost. The
letter-stratified differences were exactly 0 for A (n=36), +4.31 [1.82, 6.94]
for B (n=24), +9.73 [6.82, 13.01] for C (n=31), and +3.81 [2.08, 5.57] for D
(n=16). The effect remained +2.06 [1.42, 2.75] at the final readout.

The exact zero for A is important rather than a failure: nothing after the A
newline can retroactively alter that local residual. For B--D, earlier option
context can change the local state. The positive B--D differences therefore
show that the same option representation carries a context-dependent value or
selectedness signal, not merely stable semantic content or a static letter
signature.

This is a correlational decoding test. Strong held-out accuracy establishes a linearly readable candidate-value signal at the option-closing newline; it does not establish mapping-invariant semantic content or that the model causally uses the fitted direction.

Canonical figure: `figures/qwen36_option_newline_choice_probe.png`.
