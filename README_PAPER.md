# How a language model plays Second Chance

This is the condensed, paper-facing summary of this repository. The complete
research log, with every experiment and correction, remains in
[README.md](README.md). Nothing here is new; every claim below is established
there, with causal evidence unless noted.

## The game

The model answers a multiple-choice question. Its answer is not shown. It then
sees one sentence of feedback and the same question again:

- **Game:** `Your answer was incorrect. Choose the answer again.`
- **Neutral:** `Your answer was lost. Choose the answer again.`

The two prompts differ by one token. The model has no visible record of its
first answer. Yet under `incorrect` it switches answers far more often, its
switches go overwhelmingly to its own former runner-up, and when its first
answer was wrong, about half of its changed answers become correct. One word
turns retry into revision — and the revision is competent.

Models: Qwen3.6-27B (primary), Seed-OSS-36B (cross-architecture replication),
and two contrast models — Gemma-4-31B (computes the policy internally but
rarely expresses it in choices) and OLMo-2-32B (shows the opposite behavioral
response). Datasets:
SimpleMC (primary) and TriviaMC (replication). A remapped variant, in which
the options are re-lettered between the two presentations, is used as a
diagnostic: it shows that everything below tracks answer *content*, not
answer letters.

## The story

**1. The model's first-pass evaluations survive in the prompt's hidden state.**
While reading the options the first time, the model leaves graded scores in
the states of the option lines themselves. Those states — not any record of
the emitted answer — are the memory. The literal first decision carries only
the output letter, and that letter is causally irrelevant to what follows.
The model does not consult what it said; it consults how it graded.

**2. On the second reading, each option looks up its own history.**
Each repeated option line reads the first-presentation line with the same
semantic content, retrieving that candidate's old score. This retrieval is
identical in Game and Neutral: the same information is fetched, at the same
layers, either way. Fresh evidence from the second presentation is computed
alongside it and combined with it. This lookup is the mechanism on both
prompt variants: with the letters kept stable, blocking the matching lines
still removes the effect in all three models (in Gemma, at score level)
while a wrong-line control does nothing — and severing every outgoing signal
from the position where the first answer would have been generated changes
almost nothing on its own. Even when a stable-letter shortcut is available,
the model consults its graded evaluations, not a record of its would-be
answer.

**3. The single feedback token reprograms what the retrieved history does.**
Under `lost`, the retrieved scores amplify the old candidates in rank order —
the old winner most. Under `incorrect`, the same retrieved scores now push the
old winner down, withhold its amplification, and modestly support the
lowest-ranked candidates. Swapping the internal state written at the feedback
sentence reciprocally swaps these policies, in both models: the route is
shared, and one token's worth of state decides how it is used. The toggle is
the invariant; the resting position varies. `incorrect` turns the recollection
route against the old winner in every model and dataset tested, while what
`lost` does with the route differs — Qwen uses it to reinstate the old
ranking, Seed leaves it idle.

**4. There is no self-model and no second-choice targeting.**
No discrete "my answer was X" representation was found despite many direct
attempts: the suppression follows graded old evidence, not a stored winner
tag, and no experiment could locate a portable winner code. The runner-up is
never identified or promoted. In fact, when the old winner's score is
redistributed, the runner-up gains *less* of it than the lower-ranked
candidates do in three of four model-by-dataset comparisons (Qwen SimpleMC
−0.25 logits relative to R3/R4, Seed SimpleMC −0.50, Qwen TriviaMC −0.17; the
fourth is indistinguishable from zero) — never reliably more. The runner-up
wins most switches anyway, by default: suppress the old winner, and the
highest-scoring surviving candidate takes its place. It wins because it starts
closest to the top, despite being the policy's least-favored alternative.

**5. Both models revise from memory; how much fresh rethinking contributes
differs.** The model computes fresh scores on the second reading in both
models, but in Qwen they are nearly bystanders: blocking recollection removes
most of the preferential switching, while deleting the decodable fresh-score
signal moves the policy effect by only +0.04 logits against a natural −0.52.
In Seed, fresh evaluation genuinely participates: the same deletion removes
about a third of the score-level policy effect on SimpleMC (+0.22 of −0.60),
and about half on TriviaMC's *conflict* questions — those where the model's
fresh solve disagrees with its remembered winner. On those conflict questions
Seed also keeps a reliable policy effect even when recollection and the fresh
signal are lesioned together, through a route not yet localized. So
recollection is the shared backbone of the strategy in both models, and the
sharper claim — that fresh computation is dispensable — holds only for Qwen.

**6. The size of the revision tracks how confident the model originally was —
but only Qwen clearly does this inside Game.** Across models, the more
confident the first pass — the model's own score gap between its winner and
runner-up — the larger and the more winner-targeted the Game-minus-Neutral
adjustment at the final decision (five of six model-dataset cells for size,
all six for targeting). A difference, though, can be moved by either side.
Splitting the conditions: only Qwen shows, on both datasets, that the Game
condition itself pushes the old winner further below its first-pass standing
the more confident that first pass was, while Neutral stays near the
first-pass standing (flat in confidence on SimpleMC, far shallower on
TriviaMC). Seed shows the same pattern on one dataset; in Gemma the Game side
is flat and the scaling may sit in Neutral's re-amplification instead — one
dataset, a hypothesis only. All of this is measured on natural runs, not
lesioned. And on high-confidence questions none of it reaches the choice:
even the largest adjustments fail to cross the decision margins.
The confidence signal also travels with the memory itself: blocking the
matching-line reads collapses the confidence-scaling of the push (by
77–97% in four of the five cells that show it, partially in the fifth),
while severing the entire output of the would-be first-answer position —
where a stored "how sure was I" summary would live — leaves it untouched
in every cell. The graded prior confidence the policy consumes is read off
the retrieved scores, not consulted from a summary at the answer position.
(Qwen alone keeps a small backup through that position, visible only once
the line route is cut.)

**7. The revised answer exists internally before it is expressed.**
At the final decision position, the policy-adjusted answer ranking — old
winner down, alternatives up — is linearly decodable in a basis the output
vocabulary cannot yet read: from about layer 33 of 64 in Qwen, and from layer
36–39 in Seed. The late layers then rotate and amplify this plan into output
coordinates. The two models run different expression schedules: Qwen
integrates the adjustment before any ordering becomes readable, so on switch
trials the eventual answer is already on top from the first visible moment;
Seed expresses the recalled old ranking first — the old winner readable in the
lead — and the adjustment overtakes it over the following ~15 layers. Same
feedback-policy and recalled-rank ingredients, different timing; the causal
decomposition above additionally shows that Seed uses fresh-score information
more strongly. Notably, what surfaces first in Seed is the *memory*, even
though fresh evaluation also contributes downstream.

In short: the model plays Second Chance by re-reading its own grading notes,
and the word `incorrect` flips the sign on how those notes are applied to the
candidate it previously preferred. It is a sensible, directed strategy — and a
minimal one in Qwen. Seed implements the same feedback-controlled recollection
logic but also uses fresh candidate evaluation measurably; no portable
self-simulation or discrete prior-winner code has been found in either model.

## What replicates, and what is scoped

- Within Qwen, the behavioral effect, its semantic (content-not-letter)
  nature, the Game-side suppression machinery, the feedback-token policy
  source, and the policy-by-history interaction all replicate on TriviaMC.
- Across models, the core replicates on Seed-OSS-36B — a different
  architecture with no recurrent memory: the behavioral effect on both
  datasets, the causal necessity of matching recollection for the Game
  effect, essentially complete policy transfer from the feedback sentence's
  state, the plan-before-readout timing, and the direct policy-by-recollection
  interaction — installing the `incorrect` state makes the matching route
  suppress the old winner (+1.1 to +1.3 logits, +7 to +10 points of choice,
  both datasets), while the `lost` state leaves the route inert.
- The effect is not universal, and the failures are informative. Gemma-4-31B
  fails the behavioral test — Game and Neutral switch at the same rate — yet
  causally contains the whole machinery in its answer scores, on both
  datasets: the feedback suffix transfers essentially all of its score
  policy, matching recollection is causally used, and the installed policy
  changes what the recollection route does, in the same direction as Qwen and
  Seed. OLMo-2-32B shows the opposite behavioral response — `incorrect`
  makes it keep its old answer *more*. Behavioral success at Second Chance is
  therefore a thresholded readout of an underlying scored policy: models can
  have and express it (Qwen, Seed), have it without expressing it (Gemma), or
  invert it (OLMo).
- What sets the threshold is measured, though not yet intervened on: the
  policy's push on the old winner is similar in absolute size in all three
  models (0.6–2.3 logits), but the models' decision margins — how far the
  leading answer sits above the runner-up — differ by an order of magnitude
  (median 0.7–3.3 logits in Qwen, 2.3–12 in Seed, 5.5–18 in Gemma). The push
  *relative to the margin* falls in exactly the behavioral order on both
  datasets (0.78/0.67/0.16 on SimpleMC, 0.50/0.19/0.05 on TriviaMC), and the
  share of questions close enough to the boundary for the push to flip drops
  from 43% (Qwen) to 2% (Gemma TriviaMC). What separates a "strategic" from a
  "non-strategic" model here is not the strength of the internal policy; it
  is how confidently the model holds its answers.
- Neutral's graded reinstatement route is causally established only on
  Qwen-SimpleMC; it did not replicate on Qwen-TriviaMC or Seed. The
  descriptive fact — Neutral amplifies the old winner far more than Game —
  holds everywhere. The robust cross-model mechanism is the Game side.
- The decoded-fresh-score removal has now been run on Qwen and Seed. The clean
  recollection-versus-fresh-evidence dissociation is Qwen-specific; Seed's
  fresh coordinate carries a material minority of the score-level effect, and
  conflict questions expose a component not removed by either current lesion.
  In both models, "fresh removal" means removing a linearly decoded coordinate,
  not every distributed or nonlinear fresh representation. The direct
  question-rereading and no-self-model interventions remain Qwen-only, and
  sufficiency of recollection has not been established in either model.
- Game is strategic but not surgical: alongside the targeted redistribution it
  adds some general uncertainty across all four answers.

## Where to look

- Full research log and evidence index: [README.md](README.md)
- Integrated mechanistic account:
  [QWEN36_GAME_NEUTRAL_MECHANISTIC_SYNTHESIS.md](QWEN36_GAME_NEUTRAL_MECHANISTIC_SYNTHESIS.md)
- Corrections ledger:
  [outputs/operations/scientific_corrections.json](outputs/operations/scientific_corrections.json)
