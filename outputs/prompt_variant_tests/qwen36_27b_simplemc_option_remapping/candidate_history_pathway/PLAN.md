# Candidate-history entry, relay, and policy-binding program

## Status

Stages A and B are complete on all 500 questions. The exhaustive Stage-A factorial identifies
the 2P semantic wordpieces as the dominant receiver route for matching 1P
history in both Game and Neutral: blocking semantic reads nearly reproduces
the complete matching-edge blockade, while semantic reads alone recover most
of the route from the all-closed state. Newlines are a smaller secondary route;
leading spaces, option letters, and colons are individually small. See the
[Stage A report](stage_a/analysis/REPORT.md) and
[canonical figure](../../../../figures/qwen36_candidate_history_entry_factorial.png).

Stage B traces the complete matching-edge lesion through every token after the
2P list begins. Corrected analysis uses the canonical 273 W1!=W2 questions,
not the 234 questions on which Neutral switched. On confirmation conflict
trials, semantic-wordpiece outgoing state is the strongest single relay,
recovering 59.0% of the source-deficit rank vector in Game and 61.9% in
Neutral. Newlines, option structure, and the post-list cue/query are secondary
relays. Restoring the four pre-prefix groups recovers 94.1% in both tasks. The
lower nominal all-five recovery is convolution-confounded because restored
prefix tokens retain lesioned local outputs immediately beside the readout.
The completed convolution-safe control confirms the artifact: leaving only the
final four prefix tokens free recovers 97.7% in Game and 96.5% in Neutral on
confirmation, and 97.9%/96.3% on discovery. The five-region tail therefore
accounts for essentially the whole measured path. Both ordinary-attention K/V
and recurrent GLA writes carry recoverable portions, but the carrier cells are
not an additive mechanism decomposition. See the
[Stage B report](stage_b/analysis/REPORT.md) and
[canonical figure](../../../../figures/qwen36_candidate_history_relay_mediation.png),
plus the [convolution-safe report](convolution_control/analysis/REPORT.md) and
[control figure](../../../../figures/qwen36_candidate_history_convolution_control.png).

## Scientific objective

Establish a serial causal account of how graded first-presentation candidate
history enters the semantically matching second-presentation option line, how
old history is combined with fresh second-presentation evidence and feedback
policy, and how that contextualized state reaches the exact final answer
position.

The program uses the canonical 500-question action-matched remapped SimpleMC
paradigm, the frozen 251-question discovery / 249-question confirmation split,
raw Qwen ChatML, batch-of-four SDPA execution, and the pinned Qwen3.6-27B
revision. Game and Neutral are always analyzed separately before their
contrast. All interventions preserve the visible prompt.

## Stage A: exhaustive destination-token entry factorial

### Question

Which physical token classes in each 2P option line are necessary, sufficient,
redundant, or synergistic receivers of reads from the semantically matching 1P
option line?

### Exhaustive token partition

Every 2P option line is partitioned into five disjoint physical token classes,
with an exact tokenizer audit on every question:

1. line-leading standalone space;
2. space-prefixed option-letter token;
3. colon;
4. all semantic answer wordpieces, including answer-internal punctuation; and
5. closing newline.

No line token is omitted or assigned twice. The classification is defined by
the canonical token sequence (first, second, third, middle, last) and validated
against decoded tokens and character offsets.

### Full availability factorial

For all 32 subsets of these five classes, allow only the selected destination
classes to read the matching 1P line and block matching-line reads for every
other class. The all-open mask is the natural control; the all-closed mask is
the established complete matching-edge blockade. This includes every
block-one and allow-only-one cell while measuring all higher-order
interactions.

Every non-natural mask receives a balanced wrong-line control with the same
destination queries and layer coverage. The wrong source for target rank `r`
is offset by `1 + ((question_index + r) mod 3)`, so each of the three
nonmatching ranks is used evenly across questions and ranks. Wrong lines are
not assumed inert: report their absolute effects and the matched-minus-wrong
specific effect. Audit source-token counts and include them in compact output.

The intervention covers all 16 ordinary-attention layers L4, L8, ..., L64.
This is a receiver-edge experiment; GLA has no direct query-to-1P-line edge and
is therefore not an entry mechanism in Stage A.

### Outcomes and interpretation

- Candidate-centered A-D logit for each target rank R1--R4.
- W1 choice and conflict/no-conflict switching.
- Block-one necessity and allow-only sufficiency, both raw and corrected by the
  balanced wrong-line lesion.
- Full 32-cell Boolean factorial and Shapley decomposition of the complete
  natural-versus-all-closed effect, with interactions retained rather than
  forced additive.
- Game and Neutral separately on discovery and confirmation, followed by the
  task contrast.

Validation requires 500/500 completion, finite logits, exact prompt/token
audits, exact same-batch natural reproduction, and replication of the existing
complete matching-edge effect.

## Stage B: downstream relay mediation

### Source perturbation

Use the replicated Stage-A complete matching-edge blockade as the upstream
candidate-history perturbation. It blocks every matching 1P-line to 2P-line
ordinary-attention edge in all 16 ordinary-attention layers (L4, L8, ..., L64).
Stage A showed that semantic wordpieces carry most of this effect, but the
primary Stage-B source remains the complete line-to-line lesion so relay
mediation traces the established full effect rather than a narrower proxy.

Each question and task also has a natural run and one complete cyclic
balanced-wrong-line blockade. The wrong-line lesion is crossed only with no
restoration and with the complete joint both-mechanism restoration. This keeps
the semantic-match-versus-generic-lesion anchor inside Stage B and tests whether
joint rescue is specific to the correct historical source.

### Exhaustive relay inventory

Stage A fixes the final nonempty inventory at five groups, covering every
token from the beginning of the 2P option list through the token immediately
before the final answer readout:

1. all 2P semantic wordpieces;
2. all four option-closing newlines;
3. all 2P leading-space, option-letter, and colon structure;
4. every post-list divider, choice-cue, and cue-space/query token; and
5. every final assistant-prefix token before the readout.

These groups are disjoint and their union is the exact causal tail. Tokens in
the 2P instruction and question stem are excluded because they precede every
affected 2P receiver and therefore cannot mediate a state first written at
those receivers. Earlier option tokens remain in the global groups, but the
downstream-only restoration operation preserves every selected relay token's
locally source-perturbed output and restores only what later tokens can read
from it. Thus causal impossibility, rather than an inherited cutoff, removes
pre-source positions from each candidate's effective relay set.

### Necessity, sufficiency, and redundancy

While the complete matching source perturbation remains active, restore each
relay region's outgoing state to its clean natural-recipient value. The frozen
targeted inventory contains 26 restoration cells per task:

- five single-region restorations;
- five all-except-one complements;
- the complete joint restoration;
- three named redundancy pairs: newline+cue, newline+assistant-prefix, and
  cue+assistant-prefix;
- the five singles and joint repeated in ordinary-attention-only mode; and
- the five singles and joint repeated in GLA-only mode.

Together with natural, matching-block baseline, balanced-wrong baseline, and
balanced-wrong joint restoration, this gives 30 distinct scenarios per task
and 60 complete model forwards per canonical four-question cohort. This design
preserves the necessity/sufficiency, redundancy, specificity, and mechanism
questions that Stage B was built to answer without prepaying for all unnamed
relay subsets.

If a named pair's mediation differs reliably from the sum of its singles in
either direction, first verify that the interaction is not created by the
unintercepted short GLA convolution. No higher-order follow-up is launched from
a prefix-containing interaction until a minimal convolution-safe joint control
passes.

### Stage-B result

The run completed 7,542 exact forwards with 500/500 questions, finite outputs,
0.0 trusted-natural error, and 0.0 restoration-only error in all three carrier
modes on all 28 sentinels. The balanced wrong-line lesion is much smaller than
the matching lesion, and the matching-specific joint rescue replicates, so the
path is semantic-history specific rather than a generic consequence of lesion
size.

The original analyzer was accidentally given the trusted Neutral-results file
as its remapped baseline. It therefore defined its 234 "conflict" questions as
questions where Neutral switched, rather than the canonical W1!=W2 set. The
analyzer now validates baseline prompt provenance, hashes every analysis input,
and requires the canonical count of 273 conflicts (137 discovery, 136
confirmation).

On corrected confirmation conflict trials, single both-mode restorations
recover 59.0% (semantic), 37.6% (newline), 31.7% (structure), 25.0%
(cue/query), and 16.5% (prefix) of the Game source-deficit rank vector; Neutral
gives 61.9%, 39.5%, 40.1%, 31.9%, and 19.0%. Discovery preserves the same broad
ordering. Nominal all-five ordinary-only recovery is 21.5%/30.1%, GLA-only is
13.7%/17.3%, and both is 36.8%/48.4%.

The all-except-prefix restoration recovers 94.1% `[91.3%,96.5%]` in Game and
94.1% `[92.6%,95.6%]` in Neutral. The nominal all-five restoration falls to 36.8%
and 48.4%, but that cell is not physiologically interpretable: the restorer
deliberately preserves every restored token's lesioned local output and only
changes what later ordinary-attention and recurrent-GLA operations can read.
The final readout is adjacent to the prefix and can therefore receive those
lesioned outputs through the short GLA convolution that the intervention does
not intercept. Exact no-source identity controls cannot reveal this
lesion-dependent leak. The completed 500-question convolution-safe joint
control verifies that account. Freeing the final four prefix tokens recovers
97.7% `[96.3%,99.1%]` in Game and 96.5% `[95.7%,97.4%]` in Neutral on
confirmation, with 97.9%/96.3% on discovery. Thus the five-region inventory
accounts for essentially the whole measured candidate-history path. Every
formally flagged named-pair interaction contains the prefix, so the prior
triple escalation remains withdrawn.

### Mechanism factorial

Every single region and the joint region set is restored in ordinary-only,
GLA-only, and both modes. This identifies which outgoing carrier is sufficient
to expose each selected relay's clean state and tests ordinary/GLA redundancy
at the joint path. It does not by itself name every bypass. Do not force carrier
components to sum to 100%. Effect surviving joint both-mode restoration can
include direct source-to-final ordinary attention, source-written recurrent
state bypassing selected relays, short causal GLA q/k/v convolution, or
downstream reconstruction.

### Mandatory real identity controls

The exhaustive four-question benchmark already executes every relay mask and
mechanism variant on the real no-source-restoration path and requires raw error
exactly zero. The full targeted run repeats the complete joint restoration with
no source perturbation in ordinary-only, GLA-only, and both modes on 28 frozen
questions. These are seven complete canonical cohorts chosen to span semantic
wordpiece counts, source-line lengths, and prompt lengths, with 14 discovery
and 14 confirmation questions. Any nonzero raw error fails the run. The first
sentinel cohort is also the exact benchmark cohort, so the benchmark tests 66
complete forwards rather than only the 60-forward main path.

Primary reporting uses mediated amounts. Mediated fractions are reported only
where the paired source-lesion denominator is stable and nonzero, with paired
bootstrap ratio intervals. Complements measure whether the other four regions
suffice and each excluded region's unique contribution relative to the joint;
they are not interpreted as proving that the excluded region alone carries a
remainder.

## Stage C: policy-binding crossover at replicated relays

Stage C is designed around the Stage-B relays that replicate in both frozen
splits. Its primary targets are the 2P semantic-wordpiece relay and the broad
pre-prefix relay set. The convolution-safe joint control has passed; no
prefix-interaction triples are earned. Stage C asks whether each relay carries raw old
history, a generic policy flag, or candidate-specific policy-conditioned
history.

### Convolution-safe gate before Stage C

Qwen3.6-27B's pinned configuration has `linear_conv_kernel_dim = 4`. The
minimal control therefore repeats the matching lesion with every relay restored
while leaving the final three assistant-prefix tokens free to recompute, which
exactly covers the preceding-token support that can enter the final readout's
causal convolution. A second conservative cell leaves the final four prefix
tokens free. The existing all-five and all-except-prefix cells, the matching
and balanced-wrong lesion baselines, and a balanced-wrong conservative crossing
are repeated inside the same run. The conservative operation receives genuine
no-source restoration-only controls in ordinary-only, GLA-only, and both modes
on all 28 frozen sentinels.

The full control completed eight scenarios per task, 16 complete forwards per
canonical cohort, plus six identity forwards on each of seven sentinel
cohorts: 2,042 complete model forwards over all 500 questions. Natural error,
restoration-only error, and the maximum difference across five shared Stage-B
cells were all exactly 0.0. The kernel-minus-one three-token cell gives partial
confirmation recovery (73.3% Game, 76.6% Neutral), because its pinned boundary
can still contaminate freely recomputed tokens across layers. The conservative
four-token cell restores 97.7%/96.5% on confirmation and 97.9%/96.3% on
discovery. The gate therefore passes: the nominal all-five collapse is
convolution-boundary leakage, not a real prefix-state interaction.

### Prespecified ideal donor factorial (partly unavailable)

Construct paired donor states crossing:

- high versus low old 1P evidence for the same semantic candidate while
  holding its 2P text and display position fixed; and
- Game versus Neutral policy with identical question, mapping, and fresh 2P
  evidence.

At each replicated relay, transplant complete outgoing ordinary-attention and
GLA state across all applicable layers while preserving the relay token's own
local output. Run reciprocal directions and both frozen splits. The policy
axis below was completed. The independent old-evidence axis was not executable
from the frozen manifest and remains a future prompt-manipulation experiment.

### Frozen executable Stage-C design

The remapping manifest contains one deterministic mapping per question. It
does **not** contain two otherwise-identical prompts in which the same semantic
candidate has high versus low old 1P evidence. Creating that donor axis would
therefore require a new prompt mutation or an out-of-distribution activation
construction, neither of which is silently substituted for the prespecified
clean crossover. Stage C records this unavailable axis explicitly instead of
pretending that different questions or different candidates hold semantics
fixed.

The executable clean factorial uses reciprocal same-question Game/Neutral
donors. Question, option text, 1P evidence, remapping, display position, fresh
2P evidence, and every relay token are identical; only the feedback policy
differs. Complete outgoing ordinary-attention K/V and GLA k/v/g/beta are
crossed at all applicable layers while each recipient relay token's local
output is preserved. The cells are natural, a genuine self-restoration
identity, complete seven-token feedback-suffix swap, four separate R1--R4 2P
semantic-wordpiece relay swaps, all four semantic relays jointly, and the
complete pre-prefix causal tail jointly. Two additional mediation cells swap
the feedback suffix while pinning either all semantic relays or the complete
pre-prefix tail to recipient-clean outgoing state.

The final assistant prefix is left free in every scientific cell. This makes
the crossover convolution-safe: the final four prefix tokens can recompute
from the transplanted or restored pre-prefix state, rather than injecting the
known pinned-boundary artifact. Candidate-specific transfer is tested by the
one-rank cells against unswapped ranks; generic task transfer is tested by the
joint cells; source-to-relay policy mediation is tested by the two
feedback-swap interception cells. Frozen R1--R4 strata retain the old-history
ordering needed to ask whether task transfer depends on old rank, but they are
not mislabeled as an independently randomized old-evidence axis.

### Completed Stage-C result

The full 500-question run completed 2,750 exact forwards with finite outputs,
0.0 natural reproduction error, 0.0 trusted-natural correction error, and 0.0
real self-restoration error. On confirmation conflicts, the full feedback
suffix transfers 93.0% of the donor Game/Neutral task vector in both
directions. Joint crossover of the four 2P semantic relays transfers 19.6%
`[16.6%,22.6%]` into Game and 24.3% `[19.6%,29.2%]` into Neutral. The complete
pre-prefix tail transfers 52.5% `[46.9%,58.0%]` and 60.5% `[55.3%,65.3%]`.
Discovery independently preserves this ordering.

Every one-candidate semantic crossover preferentially transfers donor policy
on the swapped candidate: target-minus-off-target effects are +7.5 to +12.2
percentage points in Game and +10.3 to +14.4 points in Neutral, with all eight
held-out paired intervals above zero. This rejects the pure policy-blind-pipe
account. A candidate-specific fraction of policy is already bound at 2P
semantic wordpieces, and additional policy accumulates across later option and
cue/query relays before the freely recomputed prefix and late final-position
computation. Recipient-clean semantic relays intercept 25.6%/18.9% of full
feedback transfer; the complete pre-prefix tail intercepts 58.8%/51.9%.

### Remaining candidate-binding and bypass extensions

- Swap one candidate relay at a time as well as all four jointly.
- Measure transfer of raw old score, generic task state, R1--R4 rank shape,
  candidate-centered logits, and switching.
- Cross old-history donor and policy donor independently to estimate the
  old-history-by-policy interaction at the relay. This was not possible in the
  frozen one-mapping-per-question manifest and requires a new controlled
  prompt manipulation.
- Repeat the decisive relay crossover while restoring or blocking direct
  downstream access to the original feedback suffix. This tests whether the
  relay state is sufficient rather than immediately overwritten by the
  recipient's natural policy source.

A candidate-specific policy effect that changes with donor old rank and
transfers only for the swapped candidate is evidence that policy-conditioned
history is already bound at that relay. A task transfer common to every
candidate is consistent with a generic policy carrier. Old-score transfer
without task transfer supports a raw-history pipe. Null transfer is not alone
proof of absence; donor-state rejection and distributed coding are evaluated
against the full-relay and complete-suffix positive controls.

## Execution gates and operations

Each stage is implemented, locally tested, then benchmarked on the exact full
path before launch. The benchmark records every complete model forward,
runtime, projected compute cost, prompt/token audits, natural identity, and
intervention liveness. No full stage launches if the cumulative evidence-based
forecast exceeds the active spending cap. Runs checkpoint after every
four-question cohort, are monitored at least every ten minutes, and are
retrieved before the GPU is stopped. Analysis and documentation occur only
after the GPU is stopped.

Final outputs are one canonical PNG per completed stage, compact
machine-readable summaries, a human-readable integrated report, README and
mechanistic-synthesis updates, and a reconciled Vast operations ledger.
