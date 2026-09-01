# Non-remapped final-decision trajectories by first-presentation rank

This report shows Qwen3.6-27B's complete L1–L64 Standard logit lens trajectory at the final decision position in the current clean prompt. Game and Neutral differ only at the single `incorrect`/`lost` token, and the second question and options retain their original displayed order.

R1–R4 are frozen separately for each question from the same-format first-presentation aggregated A-D logits. Each layer's four scores are centered within question. The all-trial figures are the primary activation description. Switch/no-switch figures condition on the model's eventual aggregated-A-D choice and are therefore descriptive, not causal evidence about why it switched. The background tint is the mean per-question cosine similarity between that layer's centered four-candidate score vector and the exact final vector: red is inverse, white is unaligned, and blue is final-aligned. The positive half of the color mapping is cubic so that variation among late similarities of roughly 0.5–0.95 remains visible; colorbar ticks report the untransformed similarity values.

## SimpleMC

- **all:** Game n=500, Neutral n=500; [figure](../../../../figures/model_replications/qwen36_standard_logit_lens_simplemc_nonremapped_rank_trajectories_all.png)
- **switch:** Game n=174, Neutral n=132; [figure](../../../../figures/model_replications/qwen36_standard_logit_lens_simplemc_nonremapped_rank_trajectories_switch.png)
- **stay:** Game n=326, Neutral n=368; [figure](../../../../figures/model_replications/qwen36_standard_logit_lens_simplemc_nonremapped_rank_trajectories_stay.png)
- **Companions:** [non-centered A-D scores](../../../../figures/model_replications/qwen36_standard_logit_lens_simplemc_nonremapped_rank_trajectories_raw.png) · [displayed-letter-controlled scores](../../../../figures/model_replications/qwen36_standard_logit_lens_simplemc_nonremapped_rank_trajectories_letter_controlled.png)

On all trials, the L64 R1 score is 0.953 in Game and 1.524 in Neutral; the paired Game-minus-Neutral difference is -0.571 `[-0.645, -0.499]`. 

On switch trials, R2 is the largest mean L64 score in both tasks: Game R1/R2 = 0.388/0.679; Neutral R1/R2 = 0.518/0.893. On no-switch trials, R1 dominates: Game R1 = 1.254; Neutral R1 = 1.884.

## TriviaMC difficulty-filtered

- **all:** Game n=500, Neutral n=500; [figure](../../../../figures/model_replications/qwen36_standard_logit_lens_triviamc_nonremapped_rank_trajectories_all.png)
- **switch:** Game n=91, Neutral n=54; [figure](../../../../figures/model_replications/qwen36_standard_logit_lens_triviamc_nonremapped_rank_trajectories_switch.png)
- **stay:** Game n=409, Neutral n=446; [figure](../../../../figures/model_replications/qwen36_standard_logit_lens_triviamc_nonremapped_rank_trajectories_stay.png)
- **Companions:** [non-centered A-D scores](../../../../figures/model_replications/qwen36_standard_logit_lens_triviamc_nonremapped_rank_trajectories_raw.png) · [displayed-letter-controlled scores](../../../../figures/model_replications/qwen36_standard_logit_lens_triviamc_nonremapped_rank_trajectories_letter_controlled.png)

On all trials, the L64 R1 score is 2.432 in Game and 4.065 in Neutral; the paired Game-minus-Neutral difference is -1.633 `[-1.762, -1.510]`. 

On switch trials, R2 is the largest mean L64 score in both tasks: Game R1/R2 = 0.518/0.905; Neutral R1/R2 = 0.670/1.104. On no-switch trials, R1 dominates: Game R1 = 2.858; Neutral R1 = 4.476.

## Cross-dataset reading

The rank-separated decision state is predominantly late in both datasets. After displayed-letter geometry is removed, meaningful rank separation emerges around L50–L56. Neutral develops a much larger late R1 advantage than Game. Game is therefore best described here as weaker late amplification of the recalled first-pass winner—not as undirected noise added uniformly to all four answers.

Conditioning on the eventual outcome gives the expected but useful decomposition. On no-switch trials, R1 becomes dominant late. On switch trials, R2 overtakes R1 late in both Game and Neutral and in both datasets. Thus the R2 takeover is not a Game-only computation. These selected panels cannot establish why a question switched, because they are defined using the final choice itself.

**Paper-figure interpretation.** The displayed-letter-controlled companions are the preferred paper figures for this result. In their switch panels, once a meaningful semantic-rank separation becomes visible at the final decision position, R2 is already above R1; there is no phase readable through the Standard logit lens in which R1 first becomes the leading candidate and is only later overtaken. This weighs against a serial final-position story in which the model first reconstructs W1 as its prospective answer and then suppresses it. It does not show that remembered information first arrives only near L50: separate causal experiments place matching 1P-to-2P semantic-history transmission across ordinary-attention layers 4–48 and policy-dependent use of that route before the late readout. Nor does this activation plot alone prove a direct suppressive operation. Combined with the matching-history lesions, the supported account is that semantic recollection identifies the old winner, Game uses that rank information to reduce W1 relative to alternatives, and the resulting answer ordering becomes output-readable at the final decision position around L50–L56. Late sublayer decomposition further shows that much of the final Game-versus-Neutral difference is weaker Game-side amplification of W1 rather than one large negative W1 write.

The standard-lens centered plots contain a stable early R1 advantage, but the displayed-letter-controlled companions remove it almost completely. That early pattern is therefore attributable to stable A/B/C/D output-row geometry rather than mapping-independent first-presentation rank. The controlled companion is the appropriate matched comparison with Seed-OSS.

The background informativeness scale answers a different question from the rank-line means: whether an individual question's complete four-candidate geometry at a layer resembles the exact final geometry. It remains approximately unaligned through the early layers and rises sharply around L50–L56, with real non-monotonic variation through L63. L64 equals the exact final readout and therefore has similarity 1 by construction.

## Measurement scope

Readouts L1–L63 are post-block residuals transported to final-output space by Qwen3.6-27B's final RMS norm and its own A--D output-embedding rows. L64 is replaced with the exact live aggregated A-D logits from the natural forward. The score for a letter is log-sum-exp over its bare and leading-space token variants. This is activation/decoding evidence at the final decision token; it is not a layerwise causal intervention.

The non-centered companion retains each layer's common A-D offset. Across L1–L63 that offset mixes generic answer-token readiness with layer-dependent Standard logit lens scale, so it should not be read as a calibrated layerwise confidence trajectory. The displayed-letter-controlled companion subtracts the task/layer-specific across-question mean for each displayed letter before aligning candidates by first-presentation rank. It is a sensitivity analysis that removes the stable A/B/C/D geometry; the centered behaviorally complete plot remains canonical.
