# Candidate-history policy-binding crossover

Stage C reciprocally crosses Game and Neutral relay state for the same question, mapping, candidate, and fresh evidence. The final assistant prefix remains free, so the readout-side GLA convolution artifact is not reintroduced.

## Confirmation-conflict headline

### Game

- `feedback_suffix_swapped` donor-task vector transfer: 93.0% [91.3%, 94.7%]
- `relay_task_swapped_all_semantics` donor-task vector transfer: 19.6% [16.6%, 22.6%]
- `relay_task_swapped_all_pre_prefix` donor-task vector transfer: 52.5% [46.9%, 58.0%]
- Feedback-source transfer intercepted by semantic relays: 25.6%
- Feedback-source transfer intercepted by all pre-prefix relays: 58.8%
- One-candidate target-minus-off-target transfer (percentage points):
  - R1: +7.5 [+4.6, +10.5]
  - R2: +9.0 [+3.3, +14.8]
  - R3: +9.6 [+5.2, +14.2]
  - R4: +12.2 [+8.0, +16.8]

### Neutral

- `feedback_suffix_swapped` donor-task vector transfer: 93.0% [90.6%, 95.4%]
- `relay_task_swapped_all_semantics` donor-task vector transfer: 24.3% [19.6%, 29.2%]
- `relay_task_swapped_all_pre_prefix` donor-task vector transfer: 60.5% [55.3%, 65.3%]
- Feedback-source transfer intercepted by semantic relays: 18.9%
- Feedback-source transfer intercepted by all pre-prefix relays: 51.9%
- One-candidate target-minus-off-target transfer (percentage points):
  - R1: +14.1 [+9.5, +18.7]
  - R2: +10.3 [+4.6, +15.4]
  - R3: +14.4 [+6.6, +23.8]
  - R4: +10.9 [+6.3, +15.9]

## Discovery replication

- Game: feedback 93.0%, all semantics 20.7%, all pre-prefix 48.7%.
- Neutral: feedback 93.3%, all semantics 20.7%, all pre-prefix 56.1%.

## Interpretation

The 2P semantic wordpieces are not a policy-blind old-history pipe. Their outgoing state already contains a candidate-specific fraction of the Game/Neutral transformation: joint semantic crossover transfers about one fifth to one quarter of the full task vector, and every one-candidate target-minus-off-target interval is positive on held-out conflicts.

Policy continues to accumulate after those semantic tokens. Crossing the entire pre-prefix tail transfers roughly one half to three fifths of the task vector and adopts the donor answer on about 54--56% of natural task-disagreement questions. The complete feedback source remains the stronger positive control at about 93% vector transfer and 87% donor-choice adoption. Thus policy is already bound candidate-by-candidate at 2P semantics, then is further replicated or transformed across newlines, structural tokens, and cue/query state before the freely recomputed final prefix and late final-position computation.

The complementary mediation cells agree: holding semantic relays recipient-clean intercepts 25.6% of Game-directed and 18.9% of Neutral-directed feedback transfer; holding the full pre-prefix tail recipient-clean intercepts 58.8% and 51.9%. Residual transfer is not evidence that those relays lack policy: the final prefix is deliberately free, direct downstream feedback reads remain available, and the intervention does not exchange the short GLA convolution history.

## Validation

All 500 questions and 2,750 complete forwards finished. Natural reproduction, trusted-natural correction, and the real no-perturbation restoration control are all exactly 0.0-error; every completed output is finite.

## Scope

The frozen manifest has no clean independent high/low old-evidence donor for one fixed semantic candidate. This run answers policy binding with reciprocal task donors and R1--R4 stratification; it does not fabricate the missing axis.
