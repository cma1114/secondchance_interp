# Final-decision position: evidence, policy, sources, and causality

## Bottom line

By the exact final decision position, both Game and Neutral contain separately
decodable old first-presentation (1P) evidence and fresh second-presentation
(2P) evidence. The task-specific treatment of the old ranking becomes explicit
late: the held-out old-score decoder peaks at **layer 56**, where Game has much
less support for old ranks R1--R2 and more for R4 than Neutral.

The incoming computation is now localized more precisely:

- At **layer 40**, the four 2P option lines write old-rank information into the
  final position. Removing those writes weakens old-top-rank preservation in
  both tasks, while swapping Game and Neutral versions has essentially no
  effect. This is predominantly a shared evidence route, not the task-policy
  route.
- At **layers 52 and 56**, contextualized 2P scaffold positions--the repeated
  question/stem and the post-list choice cue, plus the 2P answer-only
  instruction at layer 52--carry task-specific old-rank treatment. Reciprocal
  Game/Neutral swaps move the final old-rank geometry in opposite directions on
  both frozen splits. On held-out conflict trials, giving Game the Neutral
  scaffold also lowers switching from 82.5% to 78.1%.
- Blocking the final position from directly reading all four raw 1P option
  lines at **every ordinary-attention layer, 4--64**, does not reproducibly
  remove preferential Game switching. The relevant 1P information has
  therefore already been relayed or recomputed elsewhere before the final
  query; the final token does not primarily reconstruct it by rereading the raw
  1P option lines.

This does **not** show that the 2P scaffold originated the policy. The causal
result shows where the final position receives task-conditioned ranking
information. Prior evaluation-token-to-2P and cue analyses make feedback-to-
scaffold relay the leading account, but this experiment does not intervene on
that entire chain at once.

## What was measured

All analyses use the 500-question canonical remapped, action-matched dataset:
the two prompts are identical except for `incorrect` (Game) versus `lost`
(Neutral). The frozen split contains 251 discovery and 249 confirmation
questions. Game and Neutral are always analyzed separately before their
contrast.

The destination is the exact final double-newline token whose residual predicts
the emitted A--D answer. The cached descriptive analyses cover:

- the complete residual stream at all **64 layers**;
- ordinary-attention and MLP writes at all **64 layers**;
- final-query attention at every ordinary-attention layer
  (**4, 8, ..., 64**); and
- an exhaustive, disjoint 27-region partition of every source token available
  to that query, with the exact residual write reconstructed after value
  projection, gating, head combination, and output projection.

The source-write reconstruction error is 0.49% at worst and 0.33% on average.
No source region was omitted. These cached analyses require no new model
forwards.

For causality, only the exact final query was changed. The source positions'
own residuals and all earlier computation remained natural. An ablation
prevented that query from attending to the named source positions. A swap gave
the Game query the paired Neutral source K/V states and vice versa. Natural
same-batch controls reproduced exactly. The complete causal run used 500
questions and 1,500 full forwards.

## 1. Information present at the final position

A four-output held-out decoder was fit to the final residual in the remapped
2P A--D order. Old and fresh scores were each statistically separated from the
other score and from 1P/2P display position before fitting. Layer and ridge
strength were selected on discovery only.

- **Old 1P evidence:** selected **layer 56**; pooled held-out correlation
  `r=0.323`, with `r=0.281` in Game and `r=0.273` in Neutral.
- **Fresh 2P evidence:** selected **layer 60**; pooled held-out correlation
  `r=0.295`, with `r=0.263` in Game and `r=0.257` in Neutral.

Thus, both tasks retain both evidence sources. The task difference is not that
Game loses the old score entirely. At layer 56, Game minus Neutral in decoded
old evidence is R1 `-0.205` `[-0.277,-0.133]`, R2 `-0.024`
`[-0.103,+0.062]`, R3 `+0.080` `[-0.003,+0.163]`, and R4 `+0.149`
`[+0.059,+0.234]`. The bivalent contrast
`R4 - mean(R1,R2)` is `+0.263` `[+0.144,+0.381]`. The analogous fresh-score
contrast is not established (`+0.042` `[-0.067,+0.151]`).

This is activation decoding, not a causal result.

## 2. Which final-position computation creates that geometry

Projecting each layer's ordinary-attention and MLP write onto the frozen old-
score decoder shows a replicated separation at **layers 52 and 56**. The
quantity below is each component's decoded contribution to
`R4 - mean(R1,R2)`:

| Layer | Split | Game mixer | Neutral mixer | Game - Neutral |
|---:|---|---:|---:|---:|
| 52 | discovery | -0.074 | -0.232 | +0.159 |
| 52 | confirmation | -0.052 | -0.187 | +0.135 |
| 56 | discovery | -0.159 | -0.314 | +0.155 |
| 56 | confirmation | -0.121 | -0.244 | +0.122 |

The MLP writes do not show a comparable task separation there. Thus the late
task-specific old-rank transformation enters the final residual primarily
through ordinary attention at layers 52 and 56.

## 3. Where those attention writes come from

Raw attention mass is spread broadly and looks similar in the two tasks; it is
not itself a content attribution. The exact source-specific writes identify
the information carried by each attended region.

At **layer 40**, each 2P option line contributes to old-rank geometry. The most
replicated task differences are the R2 line (Game minus Neutral `-0.050` on
discovery, `-0.054` on confirmation) and R4 line (`+0.072`, `+0.072`). This is
a decoded write, not yet evidence that those differences control behavior.

At the decisive late layers, the strongest replicated contextual sources are:

| Layer/source | Game contribution | Neutral contribution | Game - Neutral |
|---|---:|---:|---:|
| 52, 2P question/stem | -0.021 | -0.069 | +0.048 |
| 52, 2P choice cue/query | -0.014 | -0.034 | +0.020 |
| 56, 2P question/stem | -0.067 | -0.142 | +0.075 |
| 56, 2P choice cue/query | -0.030 | -0.080 | +0.051 |

Values are confirmation means for the old-rank bivalent decoded contribution.
Discovery gives the same signs and similar magnitudes. Neutral's contextual 2P
scaffold therefore writes a substantially stronger preserve-the-old-top-ranks
signal into the final position than Game's scaffold.

## 4. Causal test

The descriptive map froze three interventions before behavior was inspected:

1. swap or ablate all four complete 2P option lines at layer 40;
2. swap or ablate the 2P answer-only instruction, question/stem, and choice cue
   at layer 52, and the question/stem and choice cue at layer 56; and
3. ablate all four complete 1P option lines at every ordinary-attention layer
   4--64, completing the earlier truncated direct-read test.

### Held-out switch rates

Switching means choosing anything other than semantic W1, the 1P winner.
Percentages below are the 249-question confirmation split.

| Intervention | Game all | Neutral all | Game conflict | Neutral conflict | Game no conflict | Neutral no conflict |
|---|---:|---:|---:|---:|---:|---:|
| Natural | 62.7 | 45.0 | 82.5 | 61.3 | 38.4 | 25.0 |
| L40 2P lines swapped | 62.7 | 44.6 | 82.5 | 60.6 | 38.4 | 25.0 |
| L40 2P lines ablated | 62.2 | 45.4 | 82.5 | 62.0 | 37.5 | 25.0 |
| L52/56 scaffold swapped | 59.4 | 45.8 | 78.1 | 62.0 | 36.6 | 25.9 |
| L52/56 scaffold ablated | 61.8 | 47.0 | 81.0 | 62.0 | 38.4 | 28.6 |
| All-layer direct 1P lines ablated | 61.0 | 44.6 | 81.0 | 61.3 | 36.6 | 24.1 |

The cleanest behavioral result is the **scaffold swap into Game**: overall
switching changes by `-3.21` points `[-6.02,-0.80]`, driven by conflict trials
(`-4.38` `[-8.76,-0.73]`). The same behavioral effect is zero on the discovery
split, however, so it is held-out evidence but not a split-replicated winner-
change result. The other winner-change effects are small or inconsistent.

### Continuous final-rank effects

The final-logit geometry is more stable than thresholded winner changes and
replicates across splits:

- **L40 2P-line swap:** approximately zero in both tasks. L40 ablation weakens
  old-top-rank preservation in both tasks: bivalent change `+0.016/+0.010` in
  Game and `+0.020/+0.024` in Neutral (discovery/confirmation).
- **L52/56 scaffold swap:** Neutral scaffold into Game makes the Game ranking
  more Neutral-like (`-0.020/-0.028`); Game scaffold into Neutral makes Neutral
  less top-preserving (`+0.036/+0.029`). All four 95% intervals exclude zero.
- **L52/56 scaffold ablation:** bivalent change `+0.028/+0.025` in Game and
  `+0.117/+0.120` in Neutral. Removing the scaffold erases much more old-top-
  rank preservation in Neutral, exactly as predicted by the source-write map.
- **All-layer direct 1P-line ablation:** Game is null (`-0.003/-0.003`). Neutral
  changes modestly in the opposite direction (`-0.021/-0.025`), but the final
  winner barely changes. Direct raw-1P reads are not the main final-decision
  route.

## Mechanistic interpretation

The best-supported final-position story is:

1. By middle layers, the final residual already contains both old and fresh
   evidence. Layer-40 reads from the four 2P option lines contribute a mostly
   task-shared old-ranking signal.
2. By layers 52 and 56, the repeated question and choice-cue states contain a
   contextualized task signal. Neutral uses this route to strongly preserve
   the old top ranks. Game supplies a weaker version, allowing more movement
   away from W1 and relatively toward lower old ranks.
3. The final position combines these streams through ordinary attention; the
   task-dependent old-rank geometry is explicit by layer 56, while independent
   fresh evidence remains decodable and peaks later at layer 60.
4. The final token does not need to reread the raw 1P option lines directly.
   Earlier safe cue swaps show that the post-list cue is a real causal channel.
   Whether it is necessary remains open because the historical complete-cue
   lesion's ordinary-attention component was a Boolean-mask no-op.

What remains open is the complete upstream mediation chain: which exact
evaluation-token writes create the task-specific state later stored in the 2P
question/cue scaffold, and which intermediate positions are jointly necessary.
The present result identifies the final receiver layers and causal source
regions; it does not yet prove that entire source-to-relay-to-final chain in one
intervention.

## Exact final-position state crossover

A subsequent reciprocal Game/Neutral crossover tested the final receiver itself
at every post-layer boundary 1--64. The paired donor state has little
task-directed causal effect through layer 44. Transfer becomes practically
visible at layer 48, reaches 27--36% over layers 52--60, rises to 39--46% at
layers 61--62, and jumps to 82--85% at layer 63. Layer 64 reconstructs the exact
finished donor state and is a positive control rather than mechanistic
localization. On confirmation, the layer-63 swap changes Game switching from
62.7% to 45.0% and Neutral switching from 45.0% to 61.8%, nearly the paired
donor rates.

Replacing every final-position sequence-mixer write transfers approximately
100% of the paired donor task vector and donor switch behavior in both
directions and on both frozen splits. Replacing every MLP write transfers only
10--22% of the continuous task vector and does not transfer behavior. Thus the
late final-position task state is written overwhelmingly by sequence mixers.
The remaining problem is upstream mediation: identifying which evaluation and
2P/scaffold states feed those late mixer writes.

- [State-crossover report](../../final_position_state_crossover/REPORT.md)
- [Canonical state-crossover figure](../../../../../figures/qwen36_final_position_state_crossover.png)

## Validation and artifacts

- All 500 questions completed; every result is finite.
- Same-batch corrected natural-logit error: exactly `0`.
- Source positions and counts match between paired Game/Neutral prompts.
- Causal benchmark and full run used the exact same 12-forward cohort path.
- Full inference after model load: 1,034.5 seconds.

- [Canonical figure](../../../../../figures/qwen36_final_query_attention.png)
- [Compact causal summary](analysis/summary.json)
- [Compact causal arrays](causality/run/results.npz)
- [Final-position score-decoder summary](score/final_score_integration.json)
- [Exhaustive attention arrays](attention/attention_distribution.npz)
- [Exact source-write arrays](source_writes/final_query_source_writes.npz)
- [Component-write arrays](components/final_component_trajectory.npz)
