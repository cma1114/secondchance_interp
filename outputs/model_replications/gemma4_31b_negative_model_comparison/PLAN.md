# Gemma 4 31B negative-model comparison

## Objective

Locate where the successful Qwen3.6-27B/Seed-OSS 36B causal chain fails in
`google/gemma-4-31B-it`. The historical API run is only motivation: it used the
older prompt and found a strong generic redo transformation (22.8% Game versus
25.7% Neutral switching), not the current clean `incorrect`/`lost` contrast.
Every claim in this program therefore comes from a new pinned local run in the
canonical clean empty-history paradigm.

The candidate failure points are:

1. no distinct feedback-policy state is constructed;
2. semantic first-presentation candidate history is not causally reused;
3. both ingredients exist but policy does not change how recalled rank is used;
4. the interaction exists internally but does not reach the final answer rank.

## Frozen shared scope

- **Model:** `google/gemma-4-31B-it`, revision
  `842da3794eaa0b77d5f08bae87a17459d91ff475`, BF16, Transformers 5.5.4,
  official native processor and chat template, text-only inputs, reasoning
  disabled. Every complete Gemma
  prompt forward constructs a fresh KV cache and discards it after the forward;
  this is required because the current Transformers Gemma 4 no-cache attention
  path is numerically corrupt. No cache is shared across conditions.
- Preserve the official processor's text-only `mm_token_type_ids` companion
  (all zeros) on every prompt forward; omitting it selects a different Gemma 4
  mask-construction path and fails exact generation validation.
- Use one canonical question per batch. Ordinary collection therefore uses one
  prompt row. Cross-row causal controls may use duplicated Game/Neutral rows
  from that same question, whose token lengths and physical positions are
  exactly equal; no batch may mix questions or introduce variable-length
  left-padding. This avoids the documented Gemma 4 padding-sensitive logit
  corruption while preserving every question, cell, and real identity test.
- **Architecture:** the complete 60-layer text decoder. All descriptive and
  causal layer ranges are L1--L60. Gemma's 50 sliding-attention and 10
  full-attention layers are all included; no successful-model cutoff is reused.
- **Datasets:** all 500 canonical SimpleMC questions and all 500 frozen
  difficulty-filtered TriviaMC questions. Existing result-independent
  discovery/confirmation splits and balanced full derangements are reused.
- **Prompt:** canonical clean baseline-matched empty first assistant turn.
  Game and Neutral differ only at the single feedback word `incorrect`/`lost`;
  both continue `Choose the answer again.`
- **Candidate identity:** semantic A--D identity is frozen from Gemma's own
  same-format first-presentation aggregated A--D logits. Displayed-order stable
  tie handling is used and every remapped option moves to another letter.
- **Inference:** unrestricted next-token output remains the behavioral outcome;
  aggregated single-token A/space-A through D/space-D scores supply complete
  candidate probabilities and continuous causal endpoints.

## Stage 1 — clean behavior and semantic remapping

Run Baseline, same-order Game/Neutral, and completely remapped Game/Neutral on
both datasets. Report semantic old-W1 switching, former-letter avoidance,
rankwise centered logits/probabilities, entropy, answer-only compliance, and
paired bootstrap intervals on both frozen splits. This stage establishes
whether the historical negative behavior survives the only-difference-clean
prompt; later stages run on both datasets regardless of sign.

Validity requires exact duplicate-Baseline A--D logits, finite outputs, exactly
one Game/Neutral prompt-token difference, a complete derangement for every
question, and single-token A--D variants.

## Stage 2 — complete final-decision trajectories and decoding

For non-remapped Game and Neutral, cache the exact final prompt-position
post-block residual after every layer L1--L60 for all 1,000 questions. Produce:

- standard Gemma logit-lens trajectories in raw, candidate-centered, and
  displayed-letter-controlled coordinates for all/switch/no-switch slices;
- a background scale showing per-question similarity to the exact final A--D
  vector;
- discovery-only shared, Game-only, and Neutral-only ridge decoders predicting
  the exact final centered candidate distribution, evaluated on confirmation;
- paired Game-minus-Neutral policy-vector decoding and cross-condition transfer.

All questions are primary; switch/no-switch panels are descriptive
postselection. Require all 60 hooks, finite FP32 stored residuals, exact prompt
hashes against Stage 1, and exact native final-head reconstruction at L60.

## Stage 3 — matching-history blockade

On completely remapped prompts, compare natural, all-four semantic matching
1P-option-line to 2P-option-line attention blockade, and all-four cyclic
wrong-line blockade. Edit query-to-key attention edges only, across every
Gemma text-attention layer L1--L60. Sources and receivers are the complete
option lines; all four candidates are edited jointly. The cyclic condition
controls receiver lines, number of routes, layers, and structural intervention
while supplying the wrong semantic history.

Primary outcomes are matching-minus-cyclic rankwise centered evidence and the
change in the Game-minus-Neutral semantic old-W1 choice gap. This determines
whether Gemma's generic redo causally reuses matching semantic history and
whether that use differs by task.

## Stage 4 — complete feedback-suffix crossover

Reciprocally transplant the outgoing ordinary-attention K/V state of the full
contiguous suffix `incorrect/lost . Choose the answer again .` as seen by all
later tokens, at every layer L1--L60. Include native natural and real
distinct-row same-task identity. Report transfer of the same-question natural
Game-minus-Neutral centered A--D vector, rankwise evidence, and choice.

This tests whether Gemma constructs a distinct transferable feedback-policy
state. It does not localize that state to one token or prove behavioral use.

## Stage 5 — conditional direct policy × recollection factorial

Run on both datasets if Stages 3 and 4 show both prerequisites are alive:

- Stage 3: a stable matching-versus-cyclic candidate-history effect in at
  least one task or the paired task interaction; and
- Stage 4: nontrivial held-out suffix transfer of the natural task vector.

Cross installed Game/Neutral suffix state with intact, matching-blocked, and
cyclic-wrong history access inside both recipient prompts. The primary
interaction is the matching-minus-cyclic old-W1 effect under installed Game
policy minus the same effect under installed Neutral policy. If a prerequisite
is absent, record the earlier causal failure and do not run a factorial whose
interaction cannot be interpreted.

## Operations and completion

Every long path is resumable and checkpointed after each complete question.
Before GPU work: audit the full Vast fleet, prestart one intended host,
inspect and count every complete forward, benchmark the exact longest path, and
report the measured forecast. The complete requested batch is capped at $15;
coverage is never narrowed to meet the cap. Compact outputs are retrieved and
all GPUs stopped before local analysis. Completion requires inspected canonical
PNGs, an indexed integrated report and README entry, actual-charge
reconciliation, and authenticated fleet finalization with no GPU running.

## Completion record

All five stages completed on all 500 SimpleMC and all 500 TriviaMC questions
using the full L1--L60 scope.  The behavioral choice-rate gate did not pass, but
the prespecified continuous and causal prerequisites did, so the conditional
direct factorial was earned and completed.

- Remapped Game-minus-Neutral semantic switching was +0.6 points
  `[-2.2,+3.4]` on SimpleMC and +0.8 `[-0.6,+2.2]` on TriviaMC.
- The corresponding continuous old-rank vectors were
  `[-0.850,+0.121,+0.366,+0.362]` and
  `[-0.650,+0.176,+0.283,+0.190]` logits.
- Held-out policy-vector decoding became persistently positive at L35 and L37;
  the standard logit lens became informative later.
- Matching semantic-history blockade produced held-out Game-minus-Neutral route
  interactions led by R1 +1.360 on SimpleMC and +0.716 on TriviaMC.
- Complete feedback-suffix K/V crossover transferred 0.9995 and 0.9554 of the
  held-out continuous policy vectors.
- The direct installed-policy × matching-history factorial gave held-out R1
  interactions of +1.398 `[1.087,1.713]` and +0.683 `[0.392,0.973]`, with
  complementary negative changes for R2--R4 and the same vectors in both
  recipient prompts.  Choice interactions remained small and uncertain.

The integrated interpretation and direct artifact index are in [REPORT.md](REPORT.md).
