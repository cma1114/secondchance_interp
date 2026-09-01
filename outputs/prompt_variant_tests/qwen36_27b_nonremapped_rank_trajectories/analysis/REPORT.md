# Non-remapped final-decision trajectories by first-presentation rank

This report shows Qwen3.6-27B's complete L1–L64 JLens trajectory at the final decision position in the current clean prompt. Game and Neutral differ only at the single `incorrect`/`lost` token, and the second question and options retain their original displayed order.

R1–R4 are frozen separately for each question from the same-format first-presentation aggregated A-D logits. Each layer's four scores are centered within question. The all-trial figures are the primary activation description. Switch/no-switch figures condition on the model's eventual aggregated-A-D choice and are therefore descriptive, not causal evidence about why it switched. The background tint is the mean per-question cosine similarity between that layer's centered four-candidate score vector and the exact final vector: red is inverse, white is unaligned, and blue is final-aligned. The positive half of the color mapping is cubic so that variation among late similarities of roughly 0.5–0.95 remains visible; colorbar ticks report the untransformed similarity values.

## SimpleMC

- **all:** Game n=500, Neutral n=500; [figure](../../../../figures/qwen36_simplemc_nonremapped_rank_trajectories_all.png)
- **switch:** Game n=174, Neutral n=132; [figure](../../../../figures/qwen36_simplemc_nonremapped_rank_trajectories_switch.png)
- **stay:** Game n=326, Neutral n=368; [figure](../../../../figures/qwen36_simplemc_nonremapped_rank_trajectories_stay.png)
- **Companions:** [non-centered A-D scores](../../../../figures/qwen36_simplemc_nonremapped_rank_trajectories_raw.png) · [displayed-letter-controlled scores](../../../../figures/qwen36_simplemc_nonremapped_rank_trajectories_letter_controlled.png)

On all trials, the L64 R1 score is 0.953 in Game and 1.524 in Neutral; the paired Game-minus-Neutral difference is -0.571 `[-0.645, -0.499]`. 

On switch trials, R2 is the largest mean L64 score in both tasks: Game R1/R2 = 0.388/0.679; Neutral R1/R2 = 0.518/0.893. On no-switch trials, R1 dominates: Game R1 = 1.254; Neutral R1 = 1.884.

**Numerical-host audit (Game):** the current retained host agrees with the older trusted run on 95.4% of aggregated A-D choices (maximum logit difference 0.875). The current run has 174 switch trials versus 176 in the old-host run. Grouping the current activations by the trusted old-host choices leaves the L64 switch R1/R2 means at 0.377/0.651, so the late R2-takeover conclusion is unchanged. The canonical figures use the current run's own choices, keeping activations and outcomes from the same execution.

## TriviaMC difficulty-filtered

- **all:** Game n=500, Neutral n=500; [figure](../../../../figures/qwen36_triviamc_nonremapped_rank_trajectories_all.png)
- **switch:** Game n=91, Neutral n=54; [figure](../../../../figures/qwen36_triviamc_nonremapped_rank_trajectories_switch.png)
- **stay:** Game n=409, Neutral n=446; [figure](../../../../figures/qwen36_triviamc_nonremapped_rank_trajectories_stay.png)
- **Companions:** [non-centered A-D scores](../../../../figures/qwen36_triviamc_nonremapped_rank_trajectories_raw.png) · [displayed-letter-controlled scores](../../../../figures/qwen36_triviamc_nonremapped_rank_trajectories_letter_controlled.png)

On all trials, the L64 R1 score is 2.432 in Game and 4.065 in Neutral; the paired Game-minus-Neutral difference is -1.633 `[-1.762, -1.510]`. 

On switch trials, R2 is the largest mean L64 score in both tasks: Game R1/R2 = 0.518/0.905; Neutral R1/R2 = 0.670/1.104. On no-switch trials, R1 dominates: Game R1 = 2.858; Neutral R1 = 4.476.

## Cross-dataset reading

The rank-separated state in the **fixed JLens output readout** is predominantly late in both datasets. The visible ordering begins to grow around L48–L52 and changes steeply around L54–L56. Neutral develops a much larger late R1 advantage than Game. Game is therefore best described here as weaker late amplification of the recalled first-pass winner—not as undirected noise added uniformly to all four answers.

Conditioning on the eventual outcome gives the expected but useful decomposition. On no-switch trials, R1 becomes dominant late. On switch trials, R2 overtakes R1 late in both Game and Neutral and in both datasets. Thus the R2 takeover is not a Game-only computation. These selected panels cannot establish why a question switched, because they are defined using the final choice itself.

**Paper-figure interpretation.** The displayed-letter-controlled companions are the preferred paper figures for this result. In their switch panels, once a meaningful semantic-rank separation becomes visible at the final decision position, R2 is already above R1; there is no output-readable phase in which R1 first becomes the leading candidate and is only later overtaken. This weighs against a serial final-position story in which the model first reconstructs W1 as its prospective answer and then suppresses it. It does not show that remembered information first arrives only near L50: separate causal experiments place matching 1P-to-2P semantic-history transmission across ordinary-attention layers 4–48 and policy-dependent use of that route before the late readout. Nor does this activation plot alone prove a direct suppressive operation. Combined with the matching-history lesions, the supported account is that semantic recollection identifies the old winner, Game uses that rank information to reduce W1 relative to alternatives, and the resulting answer ordering becomes output-readable at the final decision position around L50–L56. Late sublayer decomposition further shows that much of the final Game-versus-Neutral difference is weaker Game-side amplification of W1 rather than one large negative W1 write.

The apparent inverse R1–R4 ordering around L8–L15 is displayed-letter geometry, not mapping-independent rank information. At SimpleMC L10, for example, the centered displayed-letter means are A/B/C/D = −1.325/+0.057/+0.090/+1.178, while the apparent Game rank means are −0.417/−0.088/+0.154/+0.351. Removing the displayed-letter means leaves +0.001/0.000/+0.002/−0.003. TriviaMC shows the same collapse. The displayed-letter-controlled companion therefore provides the cleaner sensitivity view of semantic-rank organization.

The background informativeness scale answers a different question from the rank-line means: whether an individual question's complete four-candidate geometry at a layer resembles the exact final geometry. It remains approximately unaligned through the early layers and rises sharply around L50–L56, with real non-monotonic variation through L63. L64 equals the exact final readout and therefore has similarity 1 by construction.

## Prospective-decoder refinement

The [held-out prospective-answer decoder](../prospective_decoding/analysis/REPORT.md) shows that “output-readable late” must not be paraphrased as “all prospective answer information is first built late.” A learned linear decoder predicts the exact eventual four-answer score pattern substantially before the fixed JLens does. On TriviaMC, shared-decoder cosine is 0.369 at L32 and 0.676 at L40, versus fixed-JLens cosine -0.007 and 0.016. Game-trained and Neutral-trained bases also transfer well across conditions, establishing a predominantly condition-general prospective code with a modest task-specific component.

The switch-specific ordering is still comparatively late: on held-out eventual-switch trials, the shared decoder's R2−R1 interval first remains positive at L44–L48. In paired Game-switch/Neutral-stay questions, the across-condition mean remains R1-favoring, while the Game-minus-Neutral R2−R1 difference becomes decodable at L34–L35. The supported timing account is therefore an earlier, non-output-aligned task-dependent precursor followed by late alignment and amplification into answer-token space—not first creation of the entire decision state at L50.

## Measurement scope

Readouts L1–L63 are post-block residuals transported to final-output space by the fixed Qwen3.6-27B Jacobian lens. L64 is replaced with the exact live aggregated A-D logits from the natural forward. The score for a letter is log-sum-exp over its bare and leading-space token variants. This is activation/decoding evidence at the final decision token; it is not a layerwise causal intervention.

The non-centered companion retains each layer's common A-D offset. Across L1–L63 that offset mixes generic answer-token readiness with layer-dependent JLens scale, so it should not be read as a calibrated layerwise confidence trajectory. The displayed-letter-controlled companion subtracts the task/layer-specific across-question mean for each displayed letter before aligning candidates by first-presentation rank. It is a sensitivity analysis that removes the stable A/B/C/D geometry; the centered behaviorally complete plot remains canonical.

## Matched standard-logit-lens check

The same cached Qwen final-position residuals have now also been read with the
conventional logit lens used for Seed-OSS: Qwen's pinned final RMS norm and its
own bare-plus-space A--D output rows. The displayed-letter-controlled
switch-trial result is unchanged. The standard-lens R2-minus-R1 interval is
sustained positive from L50/L51 on SimpleMC Game/Neutral and L50/L52 on
TriviaMC Game/Neutral. Thus R2 is already ahead when a reliable output-readable
switch ordering emerges under either readout. The earlier Qwen-versus-Seed
timing difference is not an artifact of comparing a Qwen Jacobian lens with a
Seed standard logit lens. See the [direct comparison report](../standard_logit_lens/comparison/REPORT.md)
and [canonical figure](../../../../figures/model_replications/qwen36_jlens_vs_standard_logit_lens_switch_r2_r1.png).
