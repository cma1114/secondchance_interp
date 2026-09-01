# Second Chance interpretability research

This file is the entry point for the repository. **Do not browse `outputs/`
manually to find conclusions.** That tree contains run artifacts and activation
shards; canonical findings and presentation figures are indexed here.

Scientific corrections are recorded in the
[machine-readable corrections ledger](outputs/operations/scientific_corrections.json)
rather than being left only in conversation history.

Large raw arrays and interactive generated views are intentionally kept out of
the GitHub payload. References to them are marked **local-only** below and
recorded with sizes and hashes in the
[excluded-artifact inventory](version_control/excluded_artifacts.json).

## Integrated cognitive and mechanistic account

The current evidence supports **policy-conditioned semantic reinstatement**.
Neutral uses cross-presentation semantic matches to reinstate every old
candidate in rank order, most strongly W1. Game uses those matches to suppress
W1 and R2, has little net effect on R3, and supports R4; it also selectively
attenuates W1 matching attention from L48 onward. The final repeated-W1
representation remains net pro-W1 in both conditions, so the mechanism is not
a simple final inhibitory edge. Matching 1P history is now causally traced into
the 2P semantic wordpieces and through redundant downstream relays. Reciprocal
Game/Neutral relay crossover now shows that policy is already bound
candidate-by-candidate at those history-bearing semantic positions, then
accumulates further across the remaining pre-prefix tail. The remaining gap is
an independent clean manipulation of old-evidence magnitude for one fixed
semantic candidate; the frozen one-mapping-per-question design does not supply
that donor axis.

The synthesis now starts with a position-by-position information-flow map for
the minimal `incorrect`/`lost` contrast. The candidate-history stream runs from
original option lines to repeated option lines through ordinary attention over
layers 4--48. The policy stream is not confined to evaluation-period GLA
memory. A token-resolved exact-write map shows that the literal feedback word
writes most strongly to every 2P option letter and newline at layer 32; the
evaluation-closing period peaks at option letters at layer 16 and newlines at
layer 44; contextualized `Choose` peaks at layer 60. Direct writes into the
semantic answer wordpieces are real but much smaller, and all four first-pass
ranks receive nearly the same generic policy writes. Separately, old and fresh
candidate scores are held-out decodable in 2P semantic residuals, and the
bivalent Game rank adjustment appears there around layer 34 and strengthens
sharply at layers 45--50. Exact source attribution now shows that matching 1P
lines supply graded old rank through ordinary attention at layers 32--48 in
nearly identical fashion in Game and Neutral. A new causal factorial now shows
that the evaluation-period GLA update changes how this retrieved rank is used:
reciprocal policy transplantation reverses the matching-route rank shift and
moves conflict-trial W1 choice by +15.4 points in Game and -19.1 in Neutral on
confirmation. This historical factorial used the non-output-preserved
transplant: it tests the complete evaluation-period GLA update, including the
donor-conditioned source-token output, rather than isolating persistent memory
alone. MLP 49 still gives the strongest additive readout, but restoring
its natural output at all four final 2P semantic tokens leaves 94--97% of the
final policy effect. MLP 49 is therefore not the necessary local mediator.

The policy source and downstream route are now causally localized. Crossing
only the complete feedback suffix's ordinary-attention and GLA-memory writes
transfers 92--94% of the paired Game/Neutral final-logit difference in either
direction, including the opposite rank-shaped behavior. The literal
`incorrect/lost` token is only one contributor: its first period and the
contextualized `Choose` token are strong in both recipient directions, whereas
the literal keyword is strong into Neutral (0.39 transfer) but weak into Game
(0.08--0.09). A prespecified grouped crossover now resolves the larger source
units. On confirmation, the two-token feedback sentence (`incorrect/lost .`)
transfers 64.3% of the donor task vector into Game and 66.5% into Neutral,
equal to 69.5% and 70.6% of the complete-suffix effect. The five following
tokens (`Choose the answer again .`) separately transfer 45.0% and 57.5%, or
48.7% and 61.0% of complete. Their isolated effects overlap: summing them
overshoots the joint suffix by 16.8 points in Game and 29.8 in Neutral, with
nonzero paired confidence intervals and discovery replication. Thus neither
the feedback sentence nor the identical following instruction is the sole
carrier; the latter has become policy-conditioned by its preceding context,
and the two spans carry partially redundant downstream state. Policy is then
transmitted by both architectures across the stack rather than passing through
a single layer. Exhaustive individual-layer sufficiency and necessity tests
show the practically important writes concentrated at L28--50, with the same
two maxima in both tasks and both frozen splits: ordinary-attention L36 first,
then GLA L45. Restoring all later instruction, question, option-line,
choice-cue, and assistant-prefix writes gives nominal lower-bound mediation
estimates of 51.8% in Game and 41.4% in Neutral on confirmation. A real
restoration-only identity scenario is exact, and every causal intervention
array bit-reproduces the historical run. However, the downstream-only restorer
leaves each restored token's local output on the source-crossed trajectory,
and it does not intercept the short causal GLA q/k/v convolution. Because the
final assistant prefix is adjacent to the readout, source-crossed prefix output
can leak through that convolution. The surviving fraction is not currently a
valid estimate of a physiological bypass. A convolution-safe control on the
candidate-history path now directly confirms this failure mode: freeing only
the final four prefix tokens restores 97.7% of the Game vector and 96.5% of the
Neutral vector on confirmation. The older policy-relay joint values therefore
remain lower bounds; that exact policy-transfer experiment has not itself been
rerun with the convolution-safe boundary.

- **[Read the integrated Game/Neutral mechanistic synthesis](QWEN36_GAME_NEUTRAL_MECHANISTIC_SYNTHESIS.md).**
- **[Policy transport through 2P residual streams](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/REPORT.md).**
- [Canonical policy-transport figure](figures/qwen36_second_presentation_policy_transport.png)
- [Token-resolved feedback-source figure](figures/qwen36_second_presentation_policy_token_cross.png)
- [Raw 2P-to-feedback attention heatmap, Game and Neutral separately](figures/qwen36_second_presentation_policy_attention_heatmap.png)
- [Raw feedback-to-2P residual-write heatmap, Game and Neutral separately](figures/qwen36_second_presentation_policy_write_heatmap.png) — all six panels now share one scale. Game has a strong contextualized `the` route across L32--44 that Neutral largely omits; semantic-wordpiece writes are genuinely much smaller than letter/newline writes.
- [Feedback-token writes as a fraction of the complete attention update](figures/qwen36_second_presentation_policy_write_fraction.png) — the largest single-token contribution is 17.1%; literal Game `incorrect` at L32 supplies 9.2% of the option-letter update and 8.5% of the newline update, but only 1.8% of the semantic-wordpiece update.
- [Exact semantic content of each evaluation-token write into each 2P position](figures/qwen36_second_presentation_policy_write_semantics.png) — this lenses the source-specific attention write rather than the complete 2P residual. Game broadcasts error/correction content from `incorrect`, its period, and contextualized `the` across L28--52; Neutral broadcasts `lost`/`loss` late at L56--60 and a smaller period-sourced recovery signal around L28--36. Letters and newlines receive the strongest writes; semantic wordpieces receive smaller writes; colons receive almost none.
- **Exact-position complete-residual lens audit** — verifies that the task labels are correct and disaggregates every 2P option into letter, semantic wordpieces, and newline across L1--64. `Incorrect`-family tokens occur at structural letters/newlines in both tasks, but never enter the top ten at semantic wordpieces; this is shared correctness-domain geometry in the complete residual, not evidence that Neutral received Game's feedback state. Local-only data: `outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/full_state_top_tokens/full_state_top_tokens.json`.
- [Requested-word trajectories at every 2P destination](figures/qwen36_second_presentation_policy_family_trajectories.png) — raw Game and Neutral J-lens curves across all 64 layers use only capitalization and morphological variants of `incorrect/failed/mistake/wrong` and `lost/again/resend/repeat`. Every 2P option newline develops the first trajectory around L20--48 and peaks at L34, but the tasks remain nearly identical there at L20. The preselected post-list answer-cue space already separates Game from Neutral about fourfold at L20 (0.85 versus 0.22) and later peaks at 5.66 in Game; the final decision has a related, smaller trajectory and no comparably clear L20 separation. This does not establish that the cue space is the unique carrier because intervening scaffold tokens were not included in this plot. Exact arrays and token inventories are [here](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/policy_family_trajectories/policy_family_trajectories.json).
- [Answer-cue to final A–D ranking correspondence](figures/qwen36_cue_final_letter_correspondence.png) — after combining bare and space-prefixed A--D tokens, the final-layer residual at the trailing space after `Your choice (A, B, C, or D):` already predicts much of the final answer ordering. On the 249-question confirmation split, cue/reconstructed-final top-letter agreement is 60.6% in Game and 64.3% in Neutral, pairwise-order agreement is 76.7% and 79.1%, and centered-score correlation is 0.782 and 0.856. The final ranking here is reconstructed from cached bf16 residuals through the final norm/head; its bare-token logits differ from trusted emitted logits by as much as 0.18, changing about 3--6% of argmaxes. More importantly, the separate provisional-switching comparison uses trusted emitted answers: 67.9% in Game versus 49.4% in Neutral at the cue, an 18.5-point gap, compared with a 17.7-point emitted-answer gap. Conflict-trial switching barely changes after the cue; later processing primarily reduces no-conflict switching by 14.2 points in Game and 10.6 in Neutral. Thus the scaffold modifies individual decisions but does not create the distinctive Game policy. See the [canonical report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/REPORT.md#does-the-answer-cue-space-already-contain-the-final-ad-ranking), [ranking summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/cue_final_letter_correspondence/summary.json), and [provisional-switching statistics](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/cue_final_letter_correspondence/provisional_switching.json).
- [Old evidence, fresh evidence, and rank policy at the answer cue](figures/qwen36_cue_score_integration.png) — frozen four-output decoders show fresh 2P evidence clearly at layer 35 (`r=.239` for the task-mean cue state on confirmation) and weaker unique old 1P evidence at layer 50 (`r=.121`). At layer 50, Neutral retains a top-heavy old-rank profile while Game is nearly flat; the Game-minus-Neutral `R4 - mean(R1,R2)` redistribution is `+.265` `[+.070,+.470]` and remains directionally clear through layers 50--64. The corresponding frozen-layer fresh-evidence redistribution is uncertain. See the [canonical report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/REPORT.md#does-the-answer-cue-separately-contain-old-evidence-fresh-evidence-and-the-rank-policy) and [compact statistics](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/cue_score_integration/summary.json).
- [Exhaustive all-layer source map for the answer cue](figures/qwen36_cue_attention_distribution.png) — at L36 the cue reads the 2P stem (21.5%), all four 2P lines (15.7%), its own prefix (15.7%), and feedback (18.5%) in Game; direct raw-1P-line attention is only 2.7%. Neutral instead gives 20.2% to the first-answer boundary and only 3.4% to feedback. At L48/L52, Neutral's cue read of the 2P R1 line exceeds Game by 3.16/4.52 points, while pooled R2--R4 differences are uncertain. This maps all 16 ordinary-attention layers and every causal-prefix token; see the [full report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/cue_attention_distribution/REPORT.md).
- [Causal role of the answer cue](figures/qwen36_cue_memory_causality.png) — the corrected all-64-layer intervention now removes both downstream ordinary-attention K/V access and GLA memory writes while preserving the cue token's own residual. Complete cue-route ablation changes individual final rankings beyond the neighboring-colon control, but it does not materially change the main Game-minus-Neutral switching difference: +1.6 points `[-3.6,+6.8]` on discovery and -1.2 `[-5.2,+3.2]` on confirmation. Reciprocal cue-memory swaps do reduce preferential Game switching by 6.0 points on discovery and 7.2 on confirmation. The cue is therefore a causal, policy-bearing summary channel, but it is not a necessary bottleneck for the main behavioral effect. See the [full causal report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/cue_memory_causality/analysis/REPORT.md).
- [Final-decision evidence and causal source program](figures/qwen36_final_query_attention.png) — the exact final position contains separately decodable old 1P evidence (held-out peak at L56, `r=.323`) and fresh 2P evidence (L60, `r=.295`). Ordinary-attention writes at L52/L56 create the late task-specific old-rank geometry. Reciprocal Game/Neutral swaps localize its causal input to contextualized 2P question/cue states: the old-rank logit transfer replicates on both splits, and Neutral scaffold patched into Game lowers held-out conflict switching from 82.5% to 78.1%, although that winner-change effect is absent on discovery. L40 reads from the four 2P option lines are largely task-shared. Blocking direct final-query reads from all four raw 1P option lines across the complete ordinary-attention range L4--64 is not a replicated explanation of preferential switching. See the [full report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/final_position_program/REPORT.md).
- [Final-position state crossover](figures/qwen36_final_position_state_crossover.png) — reciprocal Game/Neutral residual swaps at the exact final answer position cover every layer 1--64. Task-state transfer is small through L44, becomes practical at L48, reaches roughly one third at L52--60, and jumps to 82--85% at L63; L64 is the exact-donor control. Replacing all final-position sequence-mixer writes transfers essentially 100% of the paired donor task vector and donor switch behavior on both frozen splits. Replacing all MLP writes transfers only 10--22% of the continuous vector and does not transfer behavior. This establishes a late mixer-written final receiver state, while leaving the evaluation-to-relay-to-final source chain open. See the [full report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/final_position_state_crossover/REPORT.md).
- [Evaluation-feedback source localization](figures/qwen36_feedback_source_localization.png) — crossing only downstream writes from the complete `incorrect/lost . Choose the answer again .` suffix transfers 92--94% of the paired donor task state in both directions and both frozen splits. Individual-token effects show that the policy source is contextualized and distributed across the keyword, first period, and following instruction rather than confined to one literal word.
- [Grouped feedback-source crossover](figures/qwen36_feedback_source_groups.png) — the feedback sentence alone carries 64--70% donor-vector transfer (69--75% of the complete suffix), while the following identical instruction separately carries 45--58% (49--61% of complete). The separate effects sum to more than the joint suffix: the replicated complete-minus-sum interaction is -0.168 in Game and -0.298 in Neutral on confirmation, showing substantial overlap/redundancy rather than additive independent channels. See the [full report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/evaluation_relay_final_mediation/source_group_crossover/analysis/REPORT.md) and [machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/evaluation_relay_final_mediation/source_group_crossover/analysis/summary.json).
- [Complete individual-layer policy transmission](figures/qwen36_feedback_policy_individual_layers.png) — exhaustive L1--64 sufficiency and all-except-layer necessity tests, shown separately for Game and Neutral and for discovery and confirmation, identify L36 as the dominant individual policy-transmission layer and L45 as the second in both tasks. At both peaks, donor writes move the rank profile in the expected opposite directions: Neutral writes make Game more top-heavy, while Game writes make Neutral more bottom-supporting; both sufficiency and necessity directions replicate on the frozen discovery split. Large individual effects are concentrated at L28--50 and alternate between ordinary attention and GLA; no single layer explains the 92--94% complete transfer. See the [full report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/policy_source_layer_decomposition/individual_all_layers_analysis/REPORT.md) and [exact estimates](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/policy_source_layer_decomposition/individual_all_layers_analysis/individual_layer_estimates.csv).
- [Corrected evaluation-to-final relay mediation](figures/qwen36_feedback_relay_mediation.png) — every exhaustive post-feedback region relays a replicated portion of causal policy state. The nominal joint restoration mediates 51.8% `[49.4%,54.0%]` of Game-recipient transfer and 41.4% `[38.6%,44.4%]` of Neutral-recipient transfer on confirmation; discovery gives 53.8% and 43.6%. These are lower bounds, not bypass estimates: the restored prefix retains its source-crossed local output, and the adjacent readout can receive it through the unintercepted short GLA convolution. The real no-source-swap cache-restoration scenario is exactly 0.0-error, but that identity control cannot expose lesion-dependent convolution leakage. See the [corrected report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/evaluation_relay_final_mediation/relay_corrected_analysis/REPORT.md), [source report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/evaluation_relay_final_mediation/source_analysis/REPORT.md), and [machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/evaluation_relay_final_mediation/relay_corrected_analysis/summary.json).
- [Old-score/fresh-score residual figure](figures/qwen36_second_presentation_score_integration.png)
- [Exact old/fresh score source-attribution report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/score_source_attribution/REPORT.md)
- [Exact score source-attribution figure](figures/qwen36_second_presentation_score_source_attribution.png)
- [Causal policy × retrieved-rank report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/policy_rank_factorial/analysis/REPORT.md)
- [Causal policy × retrieved-rank figure](figures/qwen36_policy_rank_factorial.png)

## Latest major result: active semantic-winner targeting

- [TriviaMC difficulty-filtered strategic-switching replication — Step 1](outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/step1/analysis/REPORT.md) — on the untouched 250-question confirmation half, remapped Game switches away from the semantic first-presentation winner on 32.0% of questions versus 26.0% in action-matched Neutral, a paired +6.0-point difference `[+2.0,+10.4]`; discovery gives +8.8 `[+4.8,+12.8]`. Old-letter avoidance instead moves -4.0 points `[-7.2,-0.8]`, confirming that the difference follows the winner's semantic content rather than its former A-D character. Relative to Neutral, Game's final centered W1/W2/W3/W4 evidence is -0.708/+0.212/+0.231/+0.265 logits on confirmation, while A-D entropy is +0.131 bits. Thus the qualitative SimpleMC behavioral target reproduces on the frozen 500-question difficulty-filtered TriviaMC set, at a smaller behavioral magnitude. [Plan](outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/PLAN.md) · [machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/step1/analysis/summary.json) · [figure](figures/qwen36_triviamc_strategic_replication_step1.png).
- [TriviaMC difficulty-filtered strategic-switching replication — Step 2 causal matching history](outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/step2/analysis/REPORT.md) — across every ordinary-attention layer, blocking all four true 1P→2P semantic option-line reads rather than four cyclic wrong-line reads raises held-out Game W1 evidence by +0.698 logits and lowers W3/W4 by -0.164/-0.442. Natural aggregated-A-D old-W1 choice is 68.4% in Game versus 73.2% in Neutral; after the matching blockade it is 74.4% versus 74.0%, eliminating preferential Game W1 avoidance. The matching-minus-cyclic change in that task gap is +9.6 points `[+4.4,+14.8]`, with discovery +8.0 `[+2.4,+13.6]`. The Game policy-dependent recollection mechanism therefore reproduces causally. Neutral's rankwise matching-specific profile does not replicate consistently across halves, so the stronger task-shared Neutral support claim does not independently reproduce on TriviaMC. [Machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/step2/analysis/summary.json) · [figure](figures/qwen36_triviamc_matching_history_step2.png).
- [TriviaMC difficulty-filtered strategic-switching replication — Step 3 complete feedback-suffix policy crossover](outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/step3/analysis/REPORT.md) — crossing only the downstream ordinary-attention K/V and recurrent GLA writes emitted by the seven tokens `incorrect/lost . Choose the answer again .` transfers the opposite task's natural A-D scoring pattern on both frozen halves. Confirmation donor-policy transfer is 90.5% `[89.2%,91.8%]` into Game and 92.0% `[90.5%,93.4%]` into Neutral; discovery is 92.0% and 93.3%. The held-out rank reversal is explicit: Neutral suffix state installed into Game changes W1/W2/W3/W4 by +0.621/-0.199/-0.187/-0.235 centered logits, while Game suffix state installed into Neutral changes them by -0.656/+0.210/+0.207/+0.239. The real duplicated-row transplant identity and trusted-natural controls are exactly 0.0-error. Thus the contextualized policy source strongly cross-dataset replicates even though Step 2's stronger Neutral route-use profile did not. [Machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/step3/analysis/summary.json) · [figure](figures/qwen36_triviamc_feedback_suffix_step3.png) · [plan](outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/PLAN.md).
- [TriviaMC difficulty-filtered strategic-switching replication — Step 4 policy × retrieved-rank factorial](outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/step4/analysis/REPORT.md) — the reciprocal evaluation-period GLA transplant causally changes how the matching 1P→2P candidate-history route uses old rank. On confirmation, natural Game's matching-minus-cyclic lesion effect is +0.698/-0.090/-0.166/-0.442 logits across W1--W4; installing Neutral's period update reduces the W1 effect by 0.607 logits to +0.091. Natural Neutral again has no stable route profile, but installing Game's period update creates a Game-like +0.822/-0.097/-0.206/-0.519 profile. On W1≠fresh-W2 conflict trials, Neutral policy installed in Game raises old-W1 choice by +13.5 points `[+5.4,+21.6]`, while Game policy installed in Neutral lowers it by -12.2 `[-21.6,-2.7]`; discovery independently gives +16.4 and -19.7. Thus the Game-conditioned policy×recollection mechanism cross-dataset replicates causally, while stable natural-Neutral support through this exact route still does not. [Machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/step4/analysis/summary.json) · [figure](figures/qwen36_triviamc_policy_rank_step4.png) · [plan](outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/PLAN.md).
- [Evidence that Game switching is structured rather than pure noise](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/strategic_switching_evidence/analysis/REPORT.md) — a direct within-Game comparison aligns the same semantic candidates between the first and final decisions. Their mean A-D probability profile changes from 58.6/24.0/12.0/5.3% by old rank to 31.5/25.5/22.9/20.1%; in raw logits, R1/R2/R3/R4 change by -2.114/-1.181/-0.410/+0.632. On the 135 questions where the fresh 2P winner differs from both the old winner and old runner-up, natural Game switches select the fresh winner 65.8% of the time versus the old runner-up 22.5%; the preference replicates on both frozen splits. Under the joint matching-history blockade that difference shrinks to +6.6 points with a confidence interval spanning zero. Independently, the fresh-state crossover produces modest but replicating discrete donor-choice redirection beyond both equal-alternative and recipient-conditioned answer-frequency drift nulls. Together with the held-out matching-edge lesion, these results separate a history-dependent choice of whom to leave from a causally movable current-presentation influence on where to go. The following removal test now addresses necessity for the specific frozen decoded fresh subspace. No correctness label is used. [Machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/strategic_switching_evidence/analysis/summary.json).
- [Fresh-2P × recollected-history causal removal](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/fresh_history_double_dissociation/analysis/REPORT.md) — after every L1--64 block, a 500-question intervention removes more than 98% of the frozen decoded unique-fresh coordinate from every 2P semantic wordpiece and option newline while immediately restoring the old-score coordinate. Against an exact same-dose random edit, it does not reliably reduce the held-out Game-minus-Neutral old-W1-avoidance choice gap (-5.1 points `[-13.2,+2.9]`), whereas the matching-history blockade removes 24.3 points `[-34.6,-14.7]`. At score resolution, however, fresh removal reduces the differential old-W1 suppression by +0.055 logits `[+0.025,+0.084]` on the complete confirmation split and +0.041 `[+0.002,+0.080]` on conflicts, compared with natural gaps of -0.521/-0.522. The decoded fresh coordinate therefore makes a small but reliable policy-conditioned logit contribution (about 10%/8% of those natural gaps); it is not merely task-shared. Matching recollection remains much larger and necessary for the choice-level effect, while the fresh coordinate is contributory but not necessary for that effect's existence. This is bounded to the decoded linear coordinate and does not remove every distributed recomputation channel. [Figure](figures/qwen36_fresh_history_double_dissociation.png) · [machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/fresh_history_double_dissociation/analysis/summary.json).
- [Original/repeated question-stem access factorial](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/question_stem_access_factorial/analysis/REPORT.md) — a 500-question 2x2 intervention blocks direct ordinary-attention rereading of the original question wording after the first answer, the repeated question wording during 2P processing, or both, across all 16 ordinary-attention layers. Original-stem blockade leaves held-out Game old-W1 avoidance unchanged (-0.7 points `[-6.6,+5.2]`) but raises Neutral avoidance by 14.0 points `[+6.6,+22.1]`, replicating on discovery. Repeated-stem blockade modestly reduces the held-out fresh-evidence alignment at 2P semantic tokens around L40--48, yet increases fresh-W2 choice by 11.0 points in Game and 8.8 in Neutral. Joint blockade shrinks the held-out task gap from 22.1 to 5.9 points because Neutral becomes 21.3 points more switch-prone, not because Game loses old-winner suppression. Direct question-stem rereading is therefore a causal stabilizing/reconsideration input, especially in Neutral, but is not necessary for preferential Game switching or fresh-W2 selection. Question information already embedded in option states and GLA memory remains outside the intervention. [Figure](figures/qwen36_question_stem_access_factorial.png) · [machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/question_stem_access_factorial/analysis/summary.json).
- [Candidate-history entry, relay, and policy-binding program](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/candidate_history_pathway/PLAN.md) — Stage A's exhaustive 32-mask factorial shows that matching 1P history enters the 2P option lines chiefly through semantic wordpieces; newlines are secondary and structural tokens are individually small. Stage B then traces that history through redundant downstream relays, with convolution-safe complete-tail recovery of 97.7% `[96.3%,99.1%]` in Game and 96.5% `[95.7%,97.4%]` in Neutral on held-out conflicts. Stage C reciprocally crosses same-question Game/Neutral outgoing relay state while leaving the final prefix free. The complete feedback source transfers 93.0% of the donor task vector in both directions; all four semantic relays transfer 19.6% `[16.6%,22.6%]` into Game and 24.3% `[19.6%,29.2%]` into Neutral; the complete pre-prefix tail transfers 52.5% `[46.9%,58.0%]` and 60.5% `[55.3%,65.3%]`. Swapping one semantic candidate at a time produces 7.5--12.2 percentage points more donor-policy transfer on the swapped rank than on untouched ranks in Game and 10.3--14.4 points more in Neutral, with every paired interval above zero. Policy is therefore already candidate-bound at 2P semantic wordpieces, then accumulates across later pre-prefix relays; it is not applied only at the final decision. All 500-question Stage-C natural and restoration-only controls are exactly 0.0-error. [Stage A report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/candidate_history_pathway/stage_a/analysis/REPORT.md) · [Stage B report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/candidate_history_pathway/stage_b/analysis/REPORT.md) · [Convolution-safe report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/candidate_history_pathway/convolution_control/analysis/REPORT.md) · [Stage C report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/candidate_history_pathway/stage_c/analysis/REPORT.md) · [Stage C figure](figures/qwen36_candidate_history_policy_binding.png).

The all-candidate remapping factorial now gives a direct causal account of
preferential Game switching. First-presentation option lines feed their
semantically matching repeated lines through ordinary attention. Blocking W1's
matching edge raises held-out W1 evidence in Game by 0.362 logits but lowers it
in Neutral by 1.178 logits: the same semantic match actively disfavors W1 under
`incorrect` and supports it under `lost`.

The effect is graded and nonlinear in first-pass candidate evidence. An earlier
linear regression appeared to find an additional W1-specific increment, but a
prespecified audit with flexible controls for both candidate score and the gap
to the best competitor supersedes that interpretation: the extra R1 term
includes zero on both splits and slightly worsens held-out prediction. Jointly
blocking all four matching relays reduces the held-out conflict-trial
W1-choice gap to a zero point estimate (Game 17.6%→27.2%; Neutral
39.7%→27.2%); the held-out interval permits partial through over-elimination.
It nearly eliminates the point-estimate gap in discovery. Continuous W1--W2 margin
mediation is partial rather than complete.

An existing-data two-score analysis now sharpens the interpretation. After
controlling fresh remapped-presentation evidence and both displayed positions,
first-pass candidate evidence improves held-out prediction of final logits in
both tasks (+0.020 R-squared in Game; +0.041 in Neutral). On the causal
matching-edge endpoint, stronger old evidence produces a reliably more
opposing effect in Game relative to Neutral on both frozen splits; current
second-pass evidence contributes independently as well. A simple old-by-current
interaction is not robust under flexible score control. This makes combined
old/current evidence the leading economical account. A new causal crossover
now locates a portable part of that old evidence after the complete 1P option
list: the final option line and its closing newline. A smaller W1-specific
remainder remains unresolved.

The new crossover holds a semantic candidate at literal D and keeps its full
line text and token positions identical, while reordering A--C so that the
candidate had high versus low first-pass evidence. Replacing the complete D
line's ordinary-attention history across every ordinary-attention layer
(4, 8, ..., 64) moves its later centered logit on held-out questions by
+0.387/+0.359 in Game and +0.440/+0.446 in Neutral under low/high fresh 2P
evidence. Replaying only the D-closing newline's state through all 64 layers
produces smaller but reliable effects: +0.168/+0.190 and +0.186/+0.224.

The closing newline also carries information about the rest of the completed
comparison. After excluding D itself, its four-candidate transfer vector
aligns with the exact complete-history vector on confirmation: cosine +0.510
[+0.354, +0.649] in Game and +0.516 [+0.341, +0.679] in Neutral. This is the
first causal localization of a portable, distributed first-pass comparison
summary to a single post-list token. It is shared by Game and Neutral; it does
not itself explain their policy difference. Local transfer did not reliably
interact with fresh evidence, whereas complete-history transfer did, so the
later old/current integration still depends on additional history or
downstream computation.

The corrected natural-attention trajectory now covers every ordinary-attention
layer through L64. Matching attention peaks around L48--52 and then falls by
about five percentage points at L56. R1--R4 are approximately similar through
L48. The important policy contrast begins at L48: Game selectively attenuates
the matched-line read for R1, the first-pass winner, relative to R2--R4 at
every ordinary-attention layer through L64, with held-out paired intervals excluding zero. In
absolute terms both policies still show a small preference for reading R1, but
that winner preference is consistently weaker under Game.

The missing causal extension is now complete. Blocking only the matching
edges at ordinary-attention layers 52--64 changes candidate evidence by only
about 0.01 logits and has no reliable W1-choice effect. Extending the full
all-candidate blockade from layers 4--48 through layer 64 changes every
held-out R1--R4 effect by approximately 0.01 logits (maximum 0.0101 before
rounding). Thus the answer-specific
semantic relay is causally carried by layers 4--48; the conspicuous late
attention allocation at 52--64 is descriptive rather than an additional
necessary route.

The exhaustive source map now shows what the repeated option lines read in
addition to their semantic matches. Across layers 12--52, the four original
option lines jointly receive about **33--41%** of each repeated line's
attention. The matching original line receives about **13--16%**, while the
other three original lines jointly receive **20--25%**. The first-answer cue
and boundary receive another **6--9%** through most of the model. Game also
reads the policy-bearing feedback sentence much more strongly at layers
28--48 (for example, **4.15% versus 1.07%** at layer 36). Thus a repeated line
has simultaneous access to its semantic match, the whole first-pass candidate
set, the first-answer boundary, and the feedback policy. This is an activation
map. The causal follow-up now shows what the nonmatching reads do.

Blocking all three nonmatching 1P lines from every 2P option line across
ordinary-attention layers 4--64, while preserving each semantic match, shifts
evidence away from R1 and toward R3/R4 almost identically in both tasks. The
held-out within-task rank-slope changes are +0.093 in Game and +0.089 in
Neutral. Crucially, the Game-minus-Neutral policy slope is unchanged: +0.243
naturally versus +0.247 after the lesion (change +0.004
[-0.021, +0.029]). The other three lines therefore provide shared rank
evidence; they do not create the distinctive Game policy, which survives in
the preserved matching-line computation or another signal already bound to
it.

- **Start with the [human-readable all-candidate report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/all_candidate_matched_relay/analysis/REPORT.md).**
- [Canonical four-panel figure](figures/qwen36_all_candidate_matched_relay.png)
- [Absolute Game/Neutral attention trajectories by first-pass rank](figures/qwen36_all_candidate_matched_relay_absolute_attention.png)
- [Direct late winner-attention attenuation](figures/qwen36_late_winner_attention_attenuation.png)
- [Exhaustive Game/Neutral source distribution for every repeated option line](figures/qwen36_second_presentation_attention_distribution.png)
- [Exhaustive source-distribution report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_attention_distribution/analysis/REPORT.md)
- [Nonmatching-history causal report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/nonmatching_history_factorial/analysis/REPORT.md)
- [Nonmatching-history causal figure](figures/qwen36_nonmatching_history_factorial.png)
- [Old-score/current-score integration report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/old_current_score_integration/analysis/REPORT.md)
- [Old/current score machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/old_current_score_integration/analysis/summary.json)
- [D-line old-score and comparison-state causal report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/d_line_score_transfer/analysis/REPORT.md)
- [D-line causal figure](figures/qwen36_d_line_score_transfer.png)
- [D-line four-candidate vector summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/d_line_score_transfer/analysis/global_vector_summary.json)
- [Categorical-winner nonlinear audit](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/categorical_winner_audit/REPORT.md)
- [Categorical-winner audit figure](figures/qwen36_categorical_winner_nonlinearity_audit.png)
- [Machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/all_candidate_matched_relay/analysis/summary.json)
- [Complete late-layer attention contrasts](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/all_candidate_matched_relay/analysis/full_attention_trajectory_summary.json)
- [Complete-range causal report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/all_candidate_matched_relay_full_range/analysis/REPORT.md)
- [Complete-range causal figure](figures/qwen36_all_candidate_matched_relay_full_range.png)
- [Canonical remapping synthesis](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/analysis/REPORT.md#all-candidate-matched-relay-nonlinear-rank-dependence-at-semantic-matches)

The fixed-A result has now been recalibrated with exactly the same complete
24-ordering cohort and causal pipeline used for B--D. Replacing the complete
first selected-option line's ordinary-attention history with a paired history
carrying different answer content robustly moves **Neutral** toward that donor
answer at every literal position: held-out A/B/C/D effects are
+2.44/+1.93/+1.48/+1.06 logits. **Game does not reliably follow the donor** at
any position: held-out A is +0.064 [-0.265, +0.380], and B--D likewise include
zero. The old two-mapping fixed-A Game estimate (+0.548 logits) therefore does
not replicate under the matched pipeline; it was cohort/design-contingent,
not a robust A-specific mechanism.

The repeated-line mediation is also consistent across A--D. Blocking the
matching repeated option removes only a modest portion of Neutral donor
transfer, while the token-count-matched nonmatching blockade removes more
(held-out fixed-A matching-specific estimate -0.398 logits).
This does not negate the natural-history matching-edge lesion that explains
preferential Game switching; it shows that counterfactual donor substitution
and natural winner retrieval are not interchangeable interventions. Neutral
accepts substituted semantic history, whereas `incorrect` largely prevents it
from propagating to the final decision.

- [Same-pipeline fixed-A calibration report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/fixed_a_full24_calibration/analysis/REPORT.md)
- [Canonical A--D calibration figure](figures/qwen36_fixed_a_full24_calibration.png)
- [Machine-readable fixed-A calibration summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/fixed_a_full24_calibration/analysis/summary.json)
- [Fixed-B/C/D position-generalization report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/fixed_bcd_line_generalization/mediation_analysis/REPORT.md)
- [Canonical fixed-B/C/D figure](figures/qwen36_fixed_bcd_selected_line_semantic_transfer.png)
- [Machine-readable fixed-B/C/D summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/fixed_bcd_line_generalization/mediation_analysis/summary.json)
- [Frozen fixed-B/C/D design](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/fixed_bcd_line_generalization/PLAN.md)

## Current canonical experiment

- **Model:** Qwen3.6-27B
- **Datasets:** remapped SimpleMC and difficulty-filtered remapped TriviaMC,
  500 questions each
- **Prompt version:** cleaned action-matched explicit ChatML; the task-defining
  feedback differs only by `incorrect` versus `lost`, and both say `Choose the
  answer again.`
- **Run identifier:** `qwen36_27b_clean_final_residual_variance`
- **Exact prompts:** [PROMPT_SPEC.md](PROMPT_SPEC.md)

Earlier Qwen reports using `None`, `[redacted]`, mismatched chat templates, or
different placement of the answer-only instruction are historical diagnostics,
not the canonical evidence for this prompt version.

## Current headline finding

The cleaned one-token paradigm replicates selective final answer-choice
contraction. On SimpleMC, Game's layer-64 A–D contrast variance is **0.387
[0.348, 0.430]** times prompt-matched Baseline while variance in the other
5,117 dimensions is **1.368 [1.283, 1.462]**. On TriviaMC, the corresponding
ratios are **0.369 [0.354, 0.385]** and **0.771 [0.748, 0.798]**: there is broad
contraction there, but the answer subspace contracts about twice as strongly.
The direct Game/Neutral comparison isolates the distinctive task effect even
more cleanly. Game has 0.507 times Neutral's A–D variance on SimpleMC and 0.652
times on TriviaMC, while the non-answer complement ratios are 1.056 and 0.985.
Thus higher Game answer entropy reflects selective flattening of final answer
evidence rather than indiscriminate disorder throughout the residual stream.
This is activation/geometry evidence; the separate remapping and causal route
experiments establish the semantic rank structure.

- **[Clean final-residual variance report](outputs/prompt_variant_tests/qwen36_27b_clean_final_residual_variance/analysis/REPORT.md)**
- [SimpleMC machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_clean_final_residual_variance/analysis/simplemc/summary.json)
- [TriviaMC machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_clean_final_residual_variance/analysis/triviamc/summary.json)
- [Canonical figure](figures/qwen36_clean_final_residual_variance.png)

The corresponding **non-remapped, full-layer final-decision trajectories** are
now available for all 500 SimpleMC and all 500 difficulty-filtered TriviaMC
questions. Candidates are aligned by their same-format first-presentation rank
R1--R4, and Game/Neutral differ only at `incorrect`/`lost`. Rank separation in
the **fixed output readout** is predominantly late (visibly growing around
L48--52 and steepening around L54--56). At L64, R1's centered score is 0.953 in Game versus 1.524 in Neutral
on SimpleMC, and 2.432 versus 4.065 on TriviaMC. In both datasets and both
tasks, eventual switch trials show a late R2 takeover, while no-switch trials
show late R1 dominance; the outcome-conditioned panels are descriptive
postselection rather than causal tests. The background tint now shows each
layer's mean similarity to the exact final four-candidate score pattern: the
decision becomes strongly output-readable around L50--56. The apparent inverse
rank ordering at L8--15 vanishes after controlling displayed A/B/C/D means and
is therefore displayed-letter geometry rather than early semantic-rank signal.
The displayed-letter-controlled companions are **paper-figure candidates**:
on switch trials, R2 already leads R1 when semantic-rank separation first
becomes appreciable, with no visible final-position phase in which R1 first
emerges as the prospective answer and is then overtaken. This argues against
that specific serial recomputation-then-suppression account. It does not mean
history first arrives at L50: the causal matching-history work places semantic
1P-to-2P transmission across ordinary-attention layers 4--48. Read together,
the evidence says that recollection identifies the old winner, Game uses that
rank information to reduce W1 relative to alternatives, and the resulting
answer ordering becomes output-readable late. The trajectory itself is
activation evidence, and late sublayer analyses show that much of the final
Game--Neutral contrast is weaker Game amplification of W1 rather than a single
large negative W1 write.

A held-out **prospective-answer decoder** now refines that timing claim. Ridge
decoders trained on frozen discovery questions recover the exact eventual
four-answer score pattern from final-decision residuals well before JLens can
read it: on TriviaMC, shared-decoder cosine is 0.369 at L32 and 0.676 at L40,
while fixed-JLens cosine is -0.007 and 0.016. Game-trained decoders transfer to
Neutral and Neutral-trained decoders transfer to Game with only a modest
penalty (matched minus cross-condition cosine at L48: 0.070 on SimpleMC and
0.040 on TriviaMC; at L56: 0.021 and 0.011). Thus the prospective answer code is
predominantly condition-general, although not perfectly identical between
conditions. The specific R2-over-R1 ordering on held-out eventual-switch trials
becomes reliable only around L44--48. On paired questions where Game switches
and Neutral stays, the Game-minus-Neutral R2−R1 difference is already linearly
decodable at L34--35 even though the across-condition mean remains R1-favoring.
The task-dependent precursor therefore exists earlier in a non-output-aligned
linear basis and is rotated/amplified into answer-token space late. This is
activation/decoding evidence and the outcome slices are descriptive
postselection, not causal localization.

A completed **matched standard-logit-lens check** rules out the main readout
confound in the Seed comparison. The same cached Qwen residuals were passed
through Qwen's pinned final RMS norm and its own A--D output rows. After the
same displayed-letter control used for Seed, switch-trial R2-minus-R1 becomes
sustained positive at L50/L51 on SimpleMC Game/Neutral and L50/L52 on
TriviaMC Game/Neutral. R2 is already ahead when a reliable output-readable
ordering appears; the standard lens does not reveal a preceding Qwen R1-leading
phase. Seed's R1-first-then-R2 pattern is therefore a genuine descriptive
cross-model timing difference, not a Jacobian-lens-versus-logit-lens artifact.
No new transformer forwards were required for the direct comparison.

The decoder figures include a stringent chance control. A fully shuffled
question-to-target null is approximately zero. A stronger null shuffles final
targets only among questions with the same displayed 1P winner letter, thereby
preserving the easiest W1 structure while destroying question-specific final
geometry. The learned decoder exceeds that W1-matched null: at L40 the
actual/null cosines are 0.403/0.170 on SimpleMC and 0.676/0.462 on TriviaMC.

The corresponding **all-question policy-difference analysis** removes the
outcome-selection caveat. On every held-out paired question, it asks whether
the final-decision residual predicts that question's exact final
Game-minus-Neutral A--D score change. The question-specific policy pattern is
stably decodable from L33 on SimpleMC and L32 on TriviaMC; at L40 its cosine is
0.311 and 0.520 while fixed JLens is 0.006 and -0.035. The held-out final rank
profile is strategic and replicates: SimpleMC Game-minus-Neutral is R1 -0.532,
R2 +0.001, R3 +0.181, R4 +0.350; TriviaMC is R1 -1.600, R2 +0.392, R3 +0.604,
R4 +0.604. Thus policy-adjusted information is present at the final decision
position in a non-output-aligned basis by the low-to-mid 30s: Game lowers the
old winner and redistributes toward lower old ranks before the difference
becomes readable as answer-token logits. This is held-out activation decoding,
not a causal localization.

- **[Canonical non-remapped trajectory report](outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/analysis/REPORT.md)**
- SimpleMC: [all](figures/qwen36_simplemc_nonremapped_rank_trajectories_all.png) · [switch](figures/qwen36_simplemc_nonremapped_rank_trajectories_switch.png) · [no switch](figures/qwen36_simplemc_nonremapped_rank_trajectories_stay.png)
- TriviaMC: [all](figures/qwen36_triviamc_nonremapped_rank_trajectories_all.png) · [switch](figures/qwen36_triviamc_nonremapped_rank_trajectories_switch.png) · [no switch](figures/qwen36_triviamc_nonremapped_rank_trajectories_stay.png)
- Non-centered companions: [SimpleMC](figures/qwen36_simplemc_nonremapped_rank_trajectories_raw.png) · [TriviaMC](figures/qwen36_triviamc_nonremapped_rank_trajectories_raw.png)
- **Paper-figure candidates — displayed-letter-controlled trajectories:** [SimpleMC](figures/qwen36_simplemc_nonremapped_rank_trajectories_letter_controlled.png) · [TriviaMC](figures/qwen36_triviamc_nonremapped_rank_trajectories_letter_controlled.png)
- [Machine-readable trajectory summary](outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/analysis/summary.json) · [frozen design](outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/PLAN.md)
- **[Prospective-answer decoder report](outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/prospective_decoding/analysis/REPORT.md)** · [machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/prospective_decoding/analysis/summary.json)
- Prospective-decoder figures: [SimpleMC](figures/prospective_decoding/qwen36_simplemc_prospective_answer_decoding.png) · [TriviaMC](figures/prospective_decoding/qwen36_triviamc_prospective_answer_decoding.png)
- **[All-question policy-adjusted decoder report](outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/prospective_decoding/policy_analysis/REPORT.md)** · [machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/prospective_decoding/policy_analysis/summary.json)
- Policy-adjusted figures: [SimpleMC](figures/prospective_decoding/qwen36_simplemc_policy_adjusted_prospective_decoding.png) · [TriviaMC](figures/prospective_decoding/qwen36_triviamc_policy_adjusted_prospective_decoding.png)
- **[Matched Qwen Jacobian-lens versus standard-logit-lens report](outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/standard_logit_lens/comparison/REPORT.md)** · [canonical figure](figures/model_replications/qwen36_jlens_vs_standard_logit_lens_switch_r2_r1.png) · [standard-lens trajectory report](outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/standard_logit_lens/analysis/REPORT.md)

The earlier full layerwise trajectory used a superseded prompt with
`Choose a different answer` in Game and longer, unmatched Neutral wording. It
remains a historical localization result: its layer-64 Game full/A–D/complement
ratios were 1.300/0.336/1.314, closely matching the cleaned SimpleMC endpoint.
[Historical scalar report](outputs/mechanistic/qwen36_27b_jlens_corrected_empty_history_full/analysis/contraction/REPORT.md) ·
[historical layerwise report](outputs/mechanistic/qwen36_27b_jlens_corrected_empty_history_full/analysis/compression/JLENS_LAYERWISE_COMPRESSION_REPORT.md) ·
[historical compact estimates](outputs/mechanistic/qwen36_27b_jlens_corrected_empty_history_full/analysis/full_residual_variance/full_residual_variance_summary.json).

### Condition-dependent function of the strongest late mixers

A held-out causal experiment tested Mixers 56 and 63 under the same canonical
prompt. Mixer 56 normally sharpens Baseline discrimination by supporting the
top two candidates and suppressing rank 4. Mixer 63 normally performs a late
rank-opposed flattening operation, suppressing the top two candidates and
supporting ranks 3–4. Inserting both paired Baseline outputs into Game does not
reliably reduce switching. The earlier eight-output mediation result therefore
reflects a context-dependent coordinated late transformation, not the simple
removal of two ordinary sharpening components.

A direct held-out test of Mixer 56 in all three conditions corrects an earlier
informal interpretation: **Mixer 56 does not reverse sign and flatten in Game.**
Its question-specific output sharpens the A–D distribution in Baseline, Game,
and Neutral, but the sharpening is much weaker in Game. Removing Mixer 56
reduces winner advantage by 0.134 logits in Baseline, 0.043 in Game, and 0.132
in Neutral. The paired Neutral-minus-Game difference in natural sharpening is
0.089 logits (95% CI 0.074–0.105); Baseline and Neutral do not reliably differ
on winner-advantage sharpening. Thus Game-specific compression partly reflects
**attenuated late sharpening**, not an opposite flattening write from Mixer 56.

- [Baseline mixer-function report](outputs/causal/qwen36_27b_baseline_mixer_function/analysis/REPORT.md)
- [Compact numerical summary](outputs/causal/qwen36_27b_baseline_mixer_function/analysis/baseline_mixer_function_summary.json)
- [Mixer 56 across-condition causal report](outputs/causal/qwen36_27b_mixer56_across_conditions/analysis/REPORT.md)
- [Mixer 56 across-condition numerical summary](outputs/causal/qwen36_27b_mixer56_across_conditions/analysis/mixer56_across_conditions_summary.json)

### Feedback-end causal carrier

A complete-residual replacement at the period ending the feedback sentence was
repeated under the canonical prompt on all 249 held-out questions. Replacing
Game's feedback-end residuals with same-question Neutral residuals at all 64
readouts restores 13.6% of the natural original-winner-advantage gap, 23.4% of
the A–D spread gap, and 26.0% of the entropy gap. The strongest isolated window
is layers 33–40. It does **not** reliably reduce switching: the net change is
-1.2 percentage points (95% CI -5.2 to +2.8), despite changing 30 answers.

The reverse Game-into-Neutral replacement has much smaller continuous effects
and also does not increase switching. The feedback-end state is therefore a
real, asymmetric carrier of part of the continuous compression, but not a
portable or sufficient switching instruction. Its asymmetry is more consistent
with Neutral supplying sharpening that Game lacks.

- [Canonical feedback-end residual-replacement report](outputs/causal/qwen36_27b_feedback_end_residual_replacement_canonical/analysis/REPORT.md)
- [Compact numerical summary](outputs/causal/qwen36_27b_feedback_end_residual_replacement_canonical/analysis/feedback_end_residual_replacement_summary.json)

### Where the prior semantic answer is carried

A remapping experiment first established that Game preferentially suppresses
the *semantic answer reached on the first presentation*, rather than merely the
letter emitted there. A frozen causal follow-up then removed selected token
writes from all 48 Gated DeltaNet (GLA) layers while preserving every target's
exact historical four-question batch cohort.

The single-word evaluation effect also reproduces in the simpler **non-remapped**
500-question SimpleMC game. With identical option mappings and the shared
instruction `Choose the answer again`, changing only `lost` to `incorrect`
raises switching from **26.4% to 35.2%**, a paired difference of **+8.8 points**
(95% CI +4.6 to +12.8). Of the 176 Evaluation switches, **71.0%** go to the
Baseline runner-up; among 109 changed Baseline-wrong trials, **49.5%** move to
the correct answer. The effect is not confined to original answer A: the
Evaluation-minus-Neutral switch differences are +5.8, +11.0, +14.2, and +8.6
points for Baseline A, B, C, and D respectively. It is also not entropy-free:
Evaluation raises A-D entropy by 0.232 bits relative to Neutral. Thus the
one-word paradigm passes lift, runner-up, and changed-wrong accuracy checks but
fails entropy preservation. It reproduces 62% of the raw switching lift in the
existing `incorrect + different` condition under the same current Baseline
comparator.

- [Non-remapped action-matched behavioral report](outputs/prompt_variant_tests/qwen36_27b_simplemc_action_matched_nonremapped/analysis/REPORT.md)
- [Machine-readable non-remapped summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_action_matched_nonremapped/analysis/summary.json)
- [Frozen non-remapped plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_action_matched_nonremapped/PLAN.md)

The matched 500-question attention comparison shows that remapping does not
create the model's broad return to first-pass candidate history. Across all 16
ordinary-attention layers on held-out questions, complete 2P option lines give
all four 1P option lines 31.0% of their attention in non-remapped Game versus
31.4% when remapped; Neutral gives 31.5% versus 31.9%. What changes is the
allocation within that nearly fixed history budget. The semantically matching
1P line receives 15.6% without remapping versus 12.3% with remapping in Game,
and 15.7% versus 12.4% in Neutral. In the remapped prompt, where semantic
identity is separated from displayed letter/position, the semantic match still
beats the same-letter 1P line by 3.8 points in both tasks. Thus both paradigms
read the old option set equally strongly; identity order concentrates more of
that read onto the one line where semantic, letter, and positional cues agree.
Because the complete question and all four answers are already visible again
in 2P, this backward read is not needed merely to recover missing problem text.
Attention is observational, but combined with the separate balanced
matching-line lesions—which causally change final candidate scores by old rank
and remove the discrete Game-minus-Neutral switching difference—the result
supports a common mechanism in both paradigms: 2P reuses prior candidate
evaluation while also constructing fresh 2P evidence. Non-remapped identity
order makes the retrieval sharper; remapping separates letter and position
from semantic identity without eliminating the semantic match.

- [Remapped versus non-remapped 2P→1P attention report](outputs/prompt_variant_tests/qwen36_27b_simplemc_action_matched_nonremapped/attention_comparison/analysis/REPORT.md)
- [Attention comparison figure](figures/qwen36_remapped_nonremapped_2p_1p_attention.png)
- [Machine-readable attention comparison](outputs/prompt_variant_tests/qwen36_27b_simplemc_action_matched_nonremapped/attention_comparison/analysis/summary.json)

For future navigation, the related experiments are deliberately separate:

| Experiment | Option mappings | Feedback contrast | Baseline comparator | Location |
|---|---|---|---|---|
| Current non-remapped action-matched test | Unchanged | `incorrect + again` vs `lost + again` | Current same-format `answer` | `qwen36_27b_simplemc_action_matched_nonremapped/` |
| Semantic-remapping factorial | Changed between presentations | 2x2 evaluation and action clauses | Semantic W1/W2 endpoints | `qwen36_27b_simplemc_option_remapping/feedback_factorial/` |
| Historical token-matched action test | Unchanged | `incorrect + different` vs `lost + again` | Legacy report used imported `baseline_answer`; explicitly marked historical | `qwen36_27b_simplemc_token_matched_feedback/analysis/` |

The raw Baseline artifact contains both `answer` (the current matched-format
run) and `baseline_answer` (imported from an earlier compiled run); they differ
on 59/500 questions. Current analyses use `answer`. The historical report is
preserved only so its provenance remains auditable.

A full 500-question 2x2 behavioral factorial now separates the evaluation and
action clauses. With the action held fixed at `Choose the answer again`, changing
`Your answer was lost` to `Your answer was incorrect` increases avoidance of the
first-pass semantic answer W1 by **15.2 percentage points** (95% CI 11.0–19.4)
and lowers its centered logit by **0.516** (0.455–0.578). The complete standard
Game-versus-Neutral W1-avoidance gap is 18.8 points, so the evaluation-only
condition reproduces about 81% of it. This is direct behavioral evidence that
the incorrectness evaluation is sufficient for most of the semantic revision
effect; it does not require the explicit instruction to choose a different
answer.

The revision is not entropy-free. Evaluation-only raises A–D entropy by 0.136
bits (0.111–0.163). Conversely, changing only `again` to `different` under the
`lost` evaluation raises entropy by a nearly identical 0.135 bits but increases
W1 avoidance by only 3.4 points. The evaluation clause is therefore the main
determinant of *which semantic answer loses*, while both clauses contribute to
overall uncertainty. On the 273 W1 != W2 questions, W1 is selected on 21.6% of
evaluation-only trials and 20.9% of standard-Game trials, versus 39.2% under
Neutral. Thus the explicit different-answer instruction is essentially
unnecessary for the key semantic-avoidance behavior.

- [Feedback-clause factorial report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/analysis/REPORT.md)
- [Machine-readable factorial summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/analysis/summary.json)

The corrected action-matched mechanistic follow-up finds a real but substantially
smaller portable recurrent state. At the period after `Your answer was
incorrect.` or `Your answer was lost.`, the corrected intervention copies each
GLA's recurrent-memory update while restoring that period token's own output to
its target-natural value. Across all 273 W1 != W2 questions, this transfers
0.097 logits (95% CI 0.056–0.139) Evaluation-to-Neutral and 0.091 logits
(0.054–0.127) Neutral-to-Evaluation, versus a natural 0.469-logit task gap.
Thus the persistent GLA route carries about one fifth of the natural margin
difference; the much larger historical estimate also allowed donor information
to escape through the source token's residual and is superseded.

Both frozen splits pass the joint gate, though the effect is heterogeneous:
0.047 logits in discovery and 0.141 in confirmation. Only blocks 25–32 pass the
band screen and replicate, carrying 58.4% (42.5–82.5%) of the corrected all-GLA
effect. Blocks 26 and 27 have small pooled effects alone, but every individual
deletion leaves nearly the full route intact. Policy memory is therefore causal,
distributed, and redundant rather than localized to one indispensable GLA.

- [Corrected consolidated evaluation-update transplant report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/evaluation_update_transplant/output_preserved/analysis/REPORT.md)
- [Corrected machine-readable transplant summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/evaluation_update_transplant/output_preserved/analysis/summary.json)
- [Canonical corrected figure](figures/qwen36_evaluation_update_transplant.png)
- [Historical non-output-preserved report (superseded)](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/evaluation_update_transplant/analysis/REPORT.md)
- [Frozen experimental plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/evaluation_update_transplant/PLAN.md)

A direct characterization of what the 48 GLAs actually add to the residual
stream sharpens that result. At the evaluation-closing period, Evaluation and
Matched-Neutral GLA outputs become strongly different early, but the difference
is largely not aligned with A-D answer evidence. At the final decision,
answer-aligned GLA contrasts remain negligible through block 47 and then form a
late, non-monotonic sequence. On W1 != W2 trials, the cumulative raw GLA write
ends with previous-answer W1 at -0.226 [-0.363, -0.087], W2 at +0.043
[-0.086, +0.173], and the mean of the other options at +0.092 [+0.014,
+0.169] Evaluation-minus-Neutral centered units. The analogous no-conflict W1
effect is null. Thus the GLAs do not gradually repeat one suppression vector:
an early distributed condition state is expressed as a conflict-sensitive,
oscillatory answer computation concentrated in blocks 49-63. JLens decoding of
individual mean writes does not expose a stable interpretable revision feature.

- [Canonical GLA residual-write report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/gla_residual_writes/analysis/REPORT.md)
- [Compact numerical summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/gla_residual_writes/analysis/summary.json)
- [GLA output-geometry figure](figures/qwen36_action_matched_gla_output_geometry.png)
- [GLA answer-write figure](figures/qwen36_action_matched_gla_answer_writes.png)

A question-specific JLens follow-up tested whether each final-position
Evaluation-minus-Neutral GLA output directly encodes the semantic content of
the previous answer W1. It does not provide a robust affirmative result.
Individual block directions alternate, and at block 63 the conflict-trial W1
score is -0.087 [-0.235, +0.067] centered JLens units; the cumulative
transported W1 score is -0.074 [-0.231, +0.086]. A maximum-over-option-tokens
rank statistic makes W1 bottom-ranked 14.9 points [6.5, 23.7] more often on
conflict than no-conflict questions, but the effect falls to 2.6 points [-5.3,
10.6] when option tokens are averaged. Per-question unrestricted tokens are
mostly generic or uninterpretable. Thus the late GLA difference contains at
most a fragile semantic signature, not a clean decodable `not W1` direction.

- [Question-specific GLA JLens report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/gla_question_specific_jlens/analysis/REPORT.md)
- [Question-specific GLA JLens figure](figures/qwen36_action_matched_question_specific_gla_jlens.png)
- [Machine-readable JLens summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/gla_question_specific_jlens/analysis/summary.json)

The corrected exact source trace resolves the route-scope limitation of that
null JLens result. Each GLA is replayed with only the evaluation-closing
period's recurrent write removed while the period token's own output remains
target-natural. On the 273 W1 != W2 questions, complete deletion raises Game
W1 selection by 5.5 points [2.9, 8.4] and changes Matched Neutral by 0.0
[-2.6, 2.6]. The natural 18.3-point Neutral-minus-Game gap therefore falls to
12.8 points: **5.5 points [1.5, 9.5], or 30.0% [9.5, 50.0%], are removed**.
The continuous W1-minus-W2 margin gap shrinks by 0.097 logits [0.059, 0.135],
with positive effects on both frozen splits. Direct traces show strongest
final-decision Evaluation-minus-Neutral retrieval-norm differences at GLA
blocks 49, 33, and 47, and a small cumulative direct contribution favoring W2
over W1. The corrected route has no reliable entropy effect. Persistent GLA
memory is thus a causal carrier of roughly one third—not most—of the semantic
revision effect.

- [Corrected evaluation-period source-trace report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/evaluation_period_source_trace/output_preserved_analysis/REPORT.md)
- [Canonical corrected source-trace figure](figures/qwen36_evaluation_period_source_trace.png)
- [Corrected machine-readable source-trace summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/evaluation_period_source_trace/output_preserved_analysis/summary.json)
- [Historical non-output-preserved source trace (superseded)](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/evaluation_period_source_trace/analysis/REPORT.md)
- [Frozen source-trace plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/evaluation_period_source_trace/PLAN.md)

The action-period mediation experiment identifies how that upstream signal
survives the shared instruction `Choose the answer again.` At the period ending
that instruction, swapping all 48 accumulated GLA recurrent states transfers
51.2% [42.5, 60.8] of the conflict-trial W1-minus-W2 margin gap from Neutral
into Evaluation and 69.2% [59.2, 81.2] in the reverse direction. Jointly
swapping the GLA state and the period residual transfers 55.0% [45.6, 64.3]
and 83.7% [75.4, 93.9], respectively. Swapping the complete 64-block residual
trajectory alone transfers only 3.0% [-3.1, 8.6] and 2.4% [-2.9, 7.0]. The
effect independently replicates on the frozen confirmation split.

This is the missing causal bridge: the revision signal is carried through the
action clause primarily in hidden recurrent GLA memory, not as a stable
direction in that token's ordinary residual stream. The state carries broad
revision/compression as well as conflict-specific W1 reduction; replacing
Evaluation's action-period state with Neutral's removes roughly half of the
W1-targeting coefficient while also affecting no-conflict trials. The result
therefore supports a distributed revision state rather than a single `not W1`
direction.

- [Action-period mediation report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/action_period_mediation/analysis/REPORT.md)
- [Canonical action-period mediation figure](figures/qwen36_action_period_mediation.png)
- [Machine-readable mediation summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/action_period_mediation/analysis/summary.json)
- [Frozen mediation plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/action_period_mediation/PLAN.md)

A corrected source-specific follow-up now isolates the action-ending period's
own GLA-memory write and later ordinary-attention reads while restoring that
period's local residual output. The former claim of exactly zero attention
impact was an instrument artifact: the Boolean SDPA mask had made that arm a
silent no-op. In the corrected 500-question run, blocking later reads changes
Neutral conflict-trial W1-minus-W2 margin by -0.033 logits [-0.051, -0.014],
with nearly identical discovery and confirmation estimates (-0.033 on each).
In Game no-conflict trials it instead raises W1 centered advantage by +0.048
[+0.031, +0.066], again replicating by split (+0.043 and +0.054). The
action-ending period is therefore genuinely read through ordinary attention,
with task- and conflict-dependent effects.

Those effects are modest: no pooled W1-selection change excludes zero, and the
joint lesion does not reproduce the main Game-versus-Neutral behavioral gap.
The period's output-preserved own GLA write is also small. The defensible
conclusion is that the **upstream revision state survives through** the shared
action clause in recurrent memory, while its closing period adds a secondary
causal source rather than creating the main policy or serving as its dominant
bottleneck.

- [Action-period source-lesion report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/action_period_source_lesion/analysis/REPORT.md)
- [Canonical action-period source-lesion figure](figures/qwen36_action_period_source_lesion.png)
- [Machine-readable source-lesion summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/action_period_source_lesion/analysis/summary.json)
- [Frozen source-lesion plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/action_period_source_lesion/PLAN.md)

A matched workspace-lens follow-up tested whether RelP's R-lens could decode
the otherwise opaque early GLA signal more clearly than J-lens. It could not.
At the final decision, both lenses remain incoherent at block 33. Both become
clearly task-related around blocks 42--47, recovering a progression from
`incorrect`/`wrong`, through `replace`/`instead`/`exclude`, to
`again`/`override` opposed to `previous`/`last`. R-lens provides no material
advantage over J-lens here. This corroborates a later vocabulary-aligned
evaluation-to-replacement computation, but it does not reveal the content of
the causal answer-specific state at its block-33 onset.

- [Matched J-lens versus R-lens report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/gla_workspace_lenses/analysis/REPORT.md)
- **Interactive workspace-lens explorer** — local-only: `outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/gla_workspace_lenses/analysis/workspace_lens_explorer.html`.
- **Complete displayed token readouts** — local-only: `outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/gla_workspace_lenses/analysis/readouts.json`.

Applying the same workspace lenses to their intended object—the **complete
final-position residual before and after each GLA write**—is substantially more
informative. R-lens begins surfacing `different` around blocks 18--19 and
incorrectness/failure semantics around blocks 21--23; by blocks 25--29 both
lenses strongly decode the Evaluation-versus-Neutral state as
`incorrect`/`wrong`/`failure`. The accumulated state then evolves through
`previous`/`another`/`again` and `try`/`retry` semantics. Individual contextual
GLA changes become jointly interpretable around blocks 42--47 as a progression
from incorrectness to replacement and then to another/second versus previous.

However, those readable writes do not immediately demote W1. At block 33 the
GLA-specific contextual change remains opaque and has essentially zero
behavior-aligned J-lens margin effect. The first replicating negative
W1-minus-W2 margin movements in both lenses occur at GLAs 49 and 53. The
supported staged account is therefore early distributed instruction decoding,
middle/late retry-replacement semantics, and later non-monotonic candidate
redistribution—not one standalone suppression direction.

- [Complete-residual boundary-lens report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/gla_boundary_workspace_lenses/analysis/REPORT.md)
- **Interactive before/after GLA explorer** — local-only: `outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/gla_boundary_workspace_lenses/analysis/workspace_lens_boundary_explorer.html`.
- **Question-level boundary-lens results** — local-only: `outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/gla_boundary_workspace_lenses/run/results.npz`.
- [Boundary-lens numerical summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/gla_boundary_workspace_lenses/analysis/summary.json)

The complementary complete-residual JLens comparison at the **two punctuation
tokens in the feedback turn** now clarifies the readable division of labor.
At the period after `Your answer was incorrect.` versus `lost.`, the
Evaluation-minus-Neutral state becomes explicitly about error/incorrectness at
readout 43 and later shifts toward feedback, admission, and reflection. At the
shared final period after `Choose the answer again.`, correction/revision is
readable around 39--43, then `exclude`/`reject` emerges around readout 49 and
exclusion/elimination dominates through readout 64. Thus the earlier period is
the causal source and readable evaluation state, whereas the later period more
clearly expresses the downstream revision action. This does not identify which
semantic answer is excluded; that answer-specific claim still rests on the
remapping behavior and causal source trace.

- [Two-period complete-residual JLens report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/period_jlens/analysis/REPORT.md)
- **Interactive two-period JLens explorer** — local-only: `outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/period_jlens/analysis/action_matched_period_jlens_explorer.html`.
- [Compact two-period token readouts](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/period_jlens/run/top_tokens_with_baseline_ranks.json)

A consolidated reanalysis of the saved period-JLens, source-trace, global
deletion, and historical final-period replacement tensors requires no new model
run. The intact Evaluation-period write flattens the A-D distribution: on
non-conflict trials its centered final-logit effects are -0.324 for W1 and
+0.092, +0.124, and +0.108 for Baseline ranks 2--4. The associated entropy
increase is therefore structured answer-space redistribution rather than
isotropic noise. Rank ordering alone, however, cannot distinguish generic
compression from explicit W1 targeting.

A direct existing-data test now makes that distinction. On 268 W1 != W2
conflict trials, a per-question scalar-compression model explains 45.7% of the
Game causal-update squared magnitude. This R² is descriptive rather than a
generative variance decomposition because the response is natural minus
ablated evidence and the predictor includes that same ablated evidence; the
fitted mean contraction (about 0.20, with 32% clamped at zero) nevertheless
shows the fit is substantive. Adding prior-winner identity raises the
descriptive fit to 53.1%
and gives an extra W1 update of -0.257 logits [-0.308, -0.207], compared with
+0.069 [+0.028, +0.110] in Matched Neutral. On the 156 questions where W1 is
already below W2 without the write, generic compression predicts that W1 should
rise relative to W2; Game instead lowers it by -0.142 logits [-0.229, -0.053].

Crucially, this pooled targeting effect is **initial-letter heterogeneous**.
For conflict trials where the first answer was A, Game's extra W1 coefficient
is -0.375 [-0.433, -0.317]; for first answers B-D pooled it is
-0.012 [-0.105, +0.079]. The initial-A content is displayed as B, C, or D after
remapping, so this is not a simple final output-letter-A bias: the extra
suppression follows content that was originally selected as A. The current
evidence therefore supports broad compression plus an initial-A-contingent
old-answer-targeting component—not a letter-general winner-targeting mechanism.
The same heterogeneity appears on no-conflict trials: pooled Game residual
suppression beyond compression is -0.094 [-0.166, -0.025], but it is
-0.404 [-0.538, -0.270] for W1=A and +0.003 [-0.080, +0.086] for W1=B-D.
Because W1 equals the fresh winner there, no-conflict trials cannot by
themselves identify remembered-answer targeting; they corroborate the
initial-A dependence.
Game nevertheless remains active when W1 is B-D: on conflict trials its
evaluation-period contribution to W1 is -0.054 centered logits versus +0.055
in Matched Neutral (paired difference -0.110 [-0.201, -0.016]); on non-conflict
trials it is -0.271 versus -0.005 (difference -0.266 [-0.354, -0.176]). Thus
non-A Game behavior is genuine, stronger generic compression—not an additional
W1-specific term.
The widely cited 86% behavioral and 91% margin transfer estimates are from the
conflict-only transplant. On no-conflict trials, global evaluation-write
deletion nevertheless eliminates and slightly reverses the 11.6-point natural
W1-selection gap (to -2.2 points) and removes 54.7% of the centered-W1 logit
contrast. The evaluation period is therefore causally important in both
subsets, although transplant sufficiency has only been measured on conflict.

At the shared action-closing period, question-specific JLens A--D evidence is
already behaviorally aligned. At L56, Game-minus-Neutral W1 advantage is -0.271
on conflict trials but +0.099 on non-conflict trials, and -0.208 on eventual
Game switches versus +0.074 on repeats. The earlier Evaluation-period source
write remains present there in high-dimensional norm, but its directly
W1-aligned component is weak and does not predict the final causal effect
(conflict-trial Spearman r=-0.011). Thus the action period is a readable,
behaviorally aligned downstream state, but existing data do not establish it
as a causal bottleneck.

- [Existing-data period-transmission report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/period_transmission/analysis/REPORT.md)
- [Canonical period-transmission figure](figures/qwen36_action_matched_period_transmission.png)
- [Machine-readable period-transmission summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/period_transmission/analysis/summary.json)
- [Targeting-versus-compression results](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/period_transmission/analysis/targeting_vs_compression.json)
- [Targeting-versus-compression figure](figures/qwen36_evaluation_targeting_vs_compression.png)

A systematic semantic clustering of the saved JLens tails identifies several
stable but isolated readouts: answer language at block 39; incorrectness at
42; replacement at 43; `again` opposed to `previous`/`last` at 47; failure/new
at 49; and failure/retry opposed to `another` at 50. These individual entries
are stable using top/bottom 6, 12, or 24 tokens and under cosine clustering
thresholds from 0.20 to 0.30. The heatmap does **not** show a continuous or
clearly coherent semantic trajectory across blocks. It therefore supports the
reality of those punctate vocabulary-lens readouts, not a demonstrated
multi-step mechanism or circuit.

- [Semantic-cluster heatmap](figures/qwen36_jlens_gla_semantic_clusters.png)
- [Auditable cluster and block-score tables](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/gla_boundary_workspace_lenses/semantic_clustering/)

Question-level semantic-family scores show one modest behavioral correlate.
The Evaluation-minus-Neutral GLA-47 change toward `again`/`retry`/`another`/
`second` is larger on trials that avoid W1, in both historical splits and both
workspace lenses. On the frozen 249-question confirmation split, the switched
minus repeated difference is +0.048 [+0.023, +0.075] under J-lens and +0.055
[+0.021, +0.091] under R-lens (AUC 0.625 and 0.597). However, confidence
intervals include no association after controlling the fresh remapped-Baseline
W1 margin. Neither block-42 incorrectness nor block-43 replacement strength
reliably predicts switching. Thus retry semantics track behavior weakly, but
most of the readable revision sequence is shared by switched and repeated
trials.

- [Revision-semantics / switching report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/gla_semantic_switch_association/analysis/REPORT.md)
- [Revision-semantics / switching figure](figures/qwen36_revision_semantics_switch_association.png)
- **Question-level semantic scores** — local-only: `outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/gla_semantic_switch_association/run/results.npz`.

A new exact-run layerwise analysis localizes where that semantic targeting
becomes explicit at the final decision position. On the 273 W1 != W2 questions,
both JLens and ordinary logit lens show little Game--Neutral separation through
the early and middle model. The W1 contrast begins to fall around readouts
52--54, makes its largest step at readout 56, and remains negative through the
output. At readout 64, Game minus Neutral is -0.476 logits for W1 (95% CI
-0.553 to -0.401), -0.065 for W2 (-0.143 to +0.009), and +0.270 for the mean
of the other two contents (+0.229 to +0.313). This is a late, semantically
ranked redistribution away from the first-pass answer, not merely uniform
A--D compression. It localizes the expression of the effect but not yet its
upstream cause.

- [Layerwise W1-suppression report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/layerwise_decision_trajectories/analysis/REPORT.md)
- [Layerwise W1-suppression figure](figures/qwen36_simplemc_remapped_w1_layerwise.png)
- [Original-versus-fresh-remapped winner report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/first_vs_regenerated_analysis/REPORT.md)

A new semantic-binding factorial removes the remaining literal-letter
confound. For each question, two first presentations had matched Baseline
decisions of literal `A`, but `A` named different semantic contents X and Y;
the second presentation was then identical across all four Evaluation/Neutral
cells. As in the canonical empty-history paradigm, no answer token was inserted
in the historical assistant turn. A local post-run audit found 138 eligible
same-letter/different-content pairs for A in the two available Baseline
mappings, versus only 6 for B, 8 for C, and 9 for D; the current result is
therefore explicitly A-conditional rather than letter-general. On
the frozen confirmation split, `incorrect` selectively penalized whichever
semantic content had been selected first by 1.018 centered logits
[0.851, 1.200] and 40.4 selection points [29.5, 51.4]. This is strong direct
evidence that the revision computation depends on the semantic identity of the
first answer, not merely its emitted letter.

Whole-module causal removal did not reveal a dominant ordinary-attention or
MLP binding mechanism at the evaluation period, repeated W1 option end, or
final decision. The one clean replicated effect occurred while processing the
repeated W1 option: removing both module classes' evaluation-by-semantic-history
interaction restored 0.082 logits [0.058, 0.107], only about 4% of the directly
comparable 2.036-logit natural W1-versus-alternative targeting effect, and
changed W1 selection by an unreliable 2.1 points. At the
final decision, MLP-interaction removal moved the model away from W1, opposite
to a simple suppressive MLP-conjunction account. The experiment therefore
strengthens the behavioral semantic-binding claim while ruling out the simplest
version of the proposed local MLP mechanism; most of the causal binding remains
distributed or located elsewhere in the computation.

- [Semantic-binding factorial report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_module_factorial/analysis/REPORT.md)
- [Semantic-binding factorial figure](figures/qwen36_semantic_binding_module_factorial.png)
- [Machine-readable factorial summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_module_factorial/analysis/summary.json)
- [Frozen factorial plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_module_factorial/PLAN.md)

A direct fixed-A follow-up tested whether the complete residual at the internal
first-decision position is itself a portable semantic-answer state. Within the
same 64/73 discovery/confirmation cohort, it exchanged the X-history and
Y-history residuals at eleven frozen post-block readouts while preserving
literal first decision `A`, feedback, and the entire second presentation.
Natural semantic targeting reproduced (held-out 1.016 logits [0.853, 1.205]),
but the residual swap did not transfer the later suppression target. Discovery
selected readout 52; held-out Game transfer was -0.010 logits [-0.035, 0.017],
Neutral transfer was -0.013 [-0.038, 0.013], and Game minus Neutral was +0.002
[-0.017, 0.021]. Identity-patch residuals and A--D logits reproduced exactly.
Therefore the behavioral semantic history is not carried as a portable
complete residual vector at any tested single first-decision readout. This
does not exclude semantic information already written into same-layer recurrent
state or distributed across earlier option positions.

- [Fixed-A first-decision transfer report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_first_decision_transfer/analysis/REPORT.md)
- [Fixed-A first-decision transfer figure](figures/qwen36_fixed_a_first_decision_transfer.png)
- [Fixed-A first-decision transfer summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_first_decision_transfer/analysis/summary.json)
- [Frozen fixed-A transfer plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_first_decision_transfer/PLAN.md)

An internal-memory follow-up exchanged the **delta-rule recurrent matrix in
all 48 GLA layers immediately after the first-answer boundary** within the same
fixed-A X/Y cohort. A post-run architecture audit corrected the initial claim
that this was the "complete accumulated GLA state": the hook was downstream of
Qwen's causal Q/K/V convolution, and it did not exchange convolutional history,
conventional-attention prefix memory, or the recipient residual trajectory.
The natural semantic-history interaction again reproduced
(held-out 1.018 logits [0.856, 1.201]), but the state transplant did not
reliably redirect the suppression target. Held-out Game transfer was +0.030
logits [-0.038, 0.095], Neutral was +0.023 [-0.072, 0.111], and Game minus
Neutral was +0.008 [-0.043, 0.059]. The discovery result was similarly small,
and host/shard estimates were heterogeneous. Untouched rows were bit-exact,
and targeted rows changed answers, so the patch did execute; its effects were
not reliably aligned with donor semantics. The valid conclusion is only that
the delta-rule matrices are not independently sufficient as a clean portable
semantic-answer memory. The experiment does not exclude a representation that
depends on the omitted state pathways or their consistency with one another.
Replacement-host natural logits were not bit-identical to the trusted run
(19/548 cell-level answers differed), although the causal comparisons are
within-host identity-reinsertion contrasts and the key natural semantic effect
reproduced closely.

- [Fixed-A accumulated GLA-state transplant report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_gla_state_transplant/analysis/REPORT.md)
- [Fixed-A accumulated GLA-state transplant summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_gla_state_transplant/analysis/summary.json)
- [Frozen fixed-A accumulated-state plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_gla_state_transplant/PLAN.md)

A corrected **complete causal-cache factorial** now closes the main memory-
carrier gap left by those nulls. At the internal first-decision boundary it
separately transplanted (1) all 16 conventional-attention K/V caches, (2) all
48 GLA causal-convolution states, and (3) all 48 GLA delta-rule recurrent
matrices, including every combination. The full-cache X/Y transplant exactly
reproduced the donor continuation on every analyzed question (maximum A--D
logit error 0.0). Under the experiment's exact four-cell batch regime, 56
discovery and 63 held-out confirmation questions retained the required fixed-A
first decisions; 18 historically selected items were screened out before any
intervention because of Qwen's batch-sensitive numerics.

The result is clear and replicates as a **persistent-state-family localization**:
the causal influence of which semantic answer was selected first is transmitted
overwhelmingly through conventional-attention K/V history. On held-out
confirmation questions, K/V-only transplantation moved the final semantic
margin toward the donor history by 0.663 logits in Game and 2.797 logits in
Neutral; the complete-cache effects were 0.614 and 2.762 logits. Without K/V,
the corresponding continuous semantic transfer was near zero. GLA convolution
and recurrent states can still perturb individual outputs, but they do not
reliably transfer which semantic answer was selected first. The cached and
unsplit computations reproduce the crucial Neutral-minus-Game prior-answer
margin gap to within 0.005 logits on confirmation. This localizes the relevant
persistent-state family, not a single token position: the K/V transplant
includes every prefix position through the first-decision boundary, so it may
carry either an explicit previous-answer record or distributed
first-presentation information. The symmetric X↔Y transplant also cannot
change aggregate switching under the complete-cache cell—it exactly permutes
the two histories' outputs. Game subsequently discounts the prior semantic
history much more than Neutral, but the relevant prefix position(s) and
downstream comparison/revision computation remain to be localized.

- [Complete causal-cache factorial report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_full_cache_factorial/analysis/REPORT.md)
- [Complete causal-cache factorial figure](figures/qwen36_fixed_a_full_cache_factorial.png)
- [Machine-readable factorial summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_full_cache_factorial/analysis/summary.json)
- [Frozen complete-cache plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_full_cache_factorial/PLAN.md)

The source-localization follow-up now identifies where that conventional-
attention memory resides. In the same fixed-A crossover, it transplanted K/V
from disjoint first-presentation token regions while holding the visible
recipient prompt, GLA state, feedback, and second presentation fixed. The
dominant carrier is the **first option line labeled `A`**, whose semantic
content differs between X and Y histories even though the selected letter is
always A. On held-out confirmation questions, transplanting only that line
moved the final semantic margin toward the donor answer by 0.558 logits in
Game and 3.161 logits in Matched Neutral; the resulting +2.603-logit
Game-minus-Neutral contrast replicated discovery (+2.187) and exceeded the
+2.135-logit contrast from transplanting all conventional-attention K/V. The
same patch increased donor-answer selection by 15.1 points in Game versus 43.7
points in Neutral.

The entire first question reproduced essentially all of the all-K/V contrast
(+2.124 versus +2.135 logits), while the first-decision boundary and the later
choice-cue/assistant-header span were near zero. The rest of the question
weakly opposed the selected-option effect. Selected-line entropy reductions
were similar in Game and Neutral (-0.110 versus -0.129 bits), ruling out a
generic uncertainty explanation for the large condition difference. Thus the
first answer's semantic content is causally available in ordinary-attention
K/V at the selected option line, and `incorrect` makes subsequent computation
use that memory far less strongly than `lost`. The direction remains weakly
reinstating in Game, so the best-supported story is attenuated semantic
reinstatement rather than a sign-reversed inhibitory read. This remains
explicitly fixed-A rather than letter-general; layer-band localization is the
next justified step.

- [K/V source-localization report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_kv_source_transplant/analysis/REPORT.md)
- [K/V source-localization figure](figures/qwen36_fixed_a_kv_source_localization.png)
- [Machine-readable K/V source summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_kv_source_transplant/analysis/summary.json)
- [Frozen K/V source-localization plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_kv_source_transplant/PLAN.md)

The selected-option K/V follow-up localizes a strong causal semantic-history
contribution to **ordinary-attention block 44**, while showing that the complete
condition difference is cooperative across blocks 4--48. Block 44 alone replicated a
large Game-minus-Neutral semantic-transfer effect (+0.324 logits discovery;
+0.418 confirmation), driven by substantially stronger donor-answer
reinstatement in Neutral. The 36--48 band was jointly sufficient (+0.418 and
+0.548), but leave-one-band-out tests showed nonlinear dependence on 4--16 and
especially 20--32. The 52--64 band was neither sufficient nor necessary.
The all-layer positive control closely reproduced the prior source-localization
effect on both splits. This is therefore not a diffuse all-depth result: block
44 is a clean individual causal contributor, early/middle layers enable its
condition-dependent use, and the last quarter is dispensable. Because these
K/V interventions alter what *all later queries* can read, they did not by
themselves establish that the final answer query directly reads that option
line. The result remains fixed-literal-`A` rather than letter-general.

- [K/V layer-localization report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_kv_layer_localization/analysis/REPORT.md)
- [K/V layer-localization figure](figures/qwen36_fixed_a_kv_layer_localization.png)
- [Machine-readable layer summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_kv_layer_localization/analysis/summary.json)
- [Frozen layer-localization plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_kv_layer_localization/PLAN.md)

A targeted follow-up tested that missing direct-read claim. It removed only
the **final decision query's** ordinary-attention edges to the first selected
option line, leaving every earlier query and the rest of the computation
untouched. The prespecified result did not occur. On held-out confirmation,
blocking the selected line at block 44 changed Game W1 choice by -2.34 points,
the 36--48 band by -0.78 points, and all ordinary-attention blocks 4--48 by
0.00 points; none showed the predicted increase in Game choosing its old
semantic answer. The selected-line effects were also not reliably different
from token-count-matched unselected-option controls. The confirmation
selected-minus-control changes in preferential W1 avoidance were +1.56 points
for block 44, +0.78 for blocks 36--48, and -0.78 for blocks 4--48, with all
95% intervals crossing zero.

Thus the selected-option K/V history is causally important, but not through a
clean edge from that line directly into the final decision query. Its effect
must be mediated through earlier downstream token positions or a distributed
trajectory that subsequently reaches the answer decision. This correction is
why the earlier block-44 result should be described as a causal K/V
contribution rather than a demonstrated final-query reader.

- [Final-query edge-ablation report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_final_query_edge_ablation/analysis/REPORT.md)
- [Final-query edge-ablation figure](figures/qwen36_fixed_a_final_query_edge_ablation.png)
- [Machine-readable final-query summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_final_query_edge_ablation/analysis/summary.json)
- [Frozen final-query edge-ablation plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_final_query_edge_ablation/PLAN.md)

The same clean final-query test has now been run on the full canonical remapped
paradigm, without the fixed-A cohort restriction, and analyzed separately on
273 conflict questions (W1 differs from the fresh-remapped Baseline answer W2)
and 227 no-conflict questions. The natural conflict effect is large: Game
chooses W1 on 20.5% of questions versus 38.8% in Neutral. Nevertheless,
blocking the final decision query's ordinary-attention edges to W1's complete
original option line does not undo that 18.3-point avoidance difference.
Blocking block 44, blocks 36/40/44/48, or all ordinary-attention blocks 4--48
changes Game W1 choice by -0.73, 0.00, and 0.00 percentage points; effects
relative to token-count-matched unselected option lines are likewise null.
Discovery and held-out confirmation agree, and no-conflict trials show no
stable selected-line-specific effect.

This letter-general result confirms the fixed-A correction: selected-option
K/V history is causally important, but the final answer query does not retrieve
it through one clean direct edge. The semantic history must be read by earlier
downstream queries and relayed through intermediate states before the final
decision. A predecessor pass was invalidated as a Boolean-mask no-op; the
corrected run passes Boolean/additive/implicit-causal mask tests, changes
intervention logits, and exactly reproduces all trusted natural logits.

- [Canonical remapped final-query report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/final_query_edge_ablation/analysis/REPORT.md)
- [Canonical remapped final-query figure](figures/qwen36_remapped_final_query_edge_ablation.png)
- [Machine-readable canonical remapped summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/final_query_edge_ablation/analysis/summary.json)
- [Canonical remapped final-query plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/final_query_edge_ablation/PLAN.md)

The receiver-path follow-up finds the missing relay. It screened every earlier
query position, froze candidates on the 251-question discovery split, and then
causally tested them on all 500 questions. The validated receiver is the
**second-presentation option line containing W1**: jointly blocking its reads
from the matching first-presentation option line across ordinary-attention
blocks 4--48 has a large causal effect. This experiment did not test blocks
4--48 one at a time. An automatic discovery screen based on source-specific
projected-write magnitude/onset—not the raw attention trajectory—selected only
late repeated-W1 edges at 52, 60, and 64 for individual tests, and those were
null. The causal layer or smaller causal band inside 4--48 therefore remains
unlocalized.

The complete all-candidate causal extension now rules out the omitted late
quarter as a necessary route. Layers 52--64 alone produce only approximately
0.01-logit matching-specific candidate effects and no reliable held-out
W1-choice change. Extending the intervention from 4--48 through 64 changes
every held-out R1--R4 effect by approximately 0.01 logits (maximum 0.0101).
The joint causal range is
therefore 4--48; what remains unlocalized is the responsible layer or smaller
cooperative band inside that range.

- [Complete-range causal report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/all_candidate_matched_relay_full_range/analysis/REPORT.md)
- [Complete-range causal figure](figures/qwen36_all_candidate_matched_relay_full_range.png)

On 136 held-out conflict trials, blocking this W1-line-to-repeated-W1-line read
increased Game's W1 choice by 10.3 points [2.9, 17.6], but decreased Neutral's
W1 choice by 29.4 points [21.3, 37.5]. Relative to a token-count-matched
unselected source line, the intervention reduced—and reversed—the natural
Game--Neutral W1-avoidance gap by 38.2 points [27.9, 48.5]. Discovery showed the
same direction (+32.1 points). The result also replicated on held-out
no-conflict trials: the selected-minus-control W1-choice effects were +10.6
points in Game and -24.8 points in Neutral. Thus the model performs a semantic
match while rereading the answer options; the same retrieved match supports
retaining W1 after `lost` and avoiding W1 after `incorrect`. What remains
unlocalized is the downstream operation that applies those opposite policies.

- [Receiver-path causal report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/receiver_path_search/validation/analysis/REPORT.md)
- [Receiver-path causal figure](figures/qwen36_remapped_receiver_edge_validation.png)
- [Machine-readable receiver-path summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/receiver_path_search/validation/analysis/summary.json)
- [Frozen receiver-path plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/receiver_path_search/PLAN.md)

The direct final-hop follow-up clarifies what happens after that receiver. At
only the final pre-answer query, ordinary attention to the complete repeated
W1 option line was blocked and compared with separately blocking all three
other repeated option lines. The final query **does** use the W1 line, but its
net effect is pro-W1 rather than suppressive. On held-out conflict trials, the
W1-line-minus-other-lines **contrast** changed W1--W2 margin by -0.135
[-0.159, -0.111] logits in Game and -0.320 [-0.364, -0.279] in Neutral
(the raw W1-line lesions were -0.098 and -0.259). The condition difference replicated
in discovery (+0.169 logits) and confirmation (+0.186 logits).

This means the final direct edge is principally a W1 reinstatement route, and
Neutral relies on it much more strongly. Blocking it shrinks the behavioral
Game--Neutral W1-avoidance gap by 8.3 [3.2, 13.7] points on held-out conflict
trials, but by harming Neutral retention rather than undoing Game suppression.
The earlier original-W1-line → repeated-W1-line causal effect must therefore
influence Game through an intermediate state before the final answer query;
the simple last-hop suppression story is rejected.

- [Final-query repeated-option report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/final_query_repeated_option_ablation/analysis/REPORT.md)
- [Canonical final-query repeated-option figure](figures/qwen36_final_query_repeated_option_ablation.png)
- [Machine-readable final-query repeated-option summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/final_query_repeated_option_ablation/analysis/summary.json)
- [Final-query repeated-option plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/final_query_repeated_option_ablation/PLAN.md)

The comprehensive pre-final relay test reaches the same conclusion and shows
that the direct final edge is not the whole reinstatement route. Blocking the
repeated W1 option line from **every later pre-final query** reduced held-out
Game W1 choice by 12.5 points [7.4, 18.4] and W1-minus-W2 margin by 0.62 logits
[0.49, 0.75]. Neutral fell much more: 30.9 points [23.5, 39.0] and 1.01 logits
[0.83, 1.21]. Discovery replicated both directions. Almost the entire effect
came from queries after all four repeated options; reads by later option lines
were null.

Thus earlier post-option states and the final answer query both read the
repeated W1 line to **reinstate** W1, with substantially greater gain in
Neutral. The repeated-line pathway does not carry an active anti-W1 signal in
Game. Combined with the original-line to repeated-line lesion, the supported
account is differential reinstatement: Game weakens or negatively
contextualizes a shared pro-W1 route rather than sending a separate suppressive
message downstream. The prespecified suppression prerequisite failed, so the
gated depth-band run was not performed.

- [Repeated-W1 relay report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/repeated_w1_relay/analysis/REPORT.md)
- [Repeated-W1 relay figure](figures/qwen36_repeated_w1_relay.png)
- [Machine-readable repeated-W1 relay summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/repeated_w1_relay/analysis/summary.json)
- [Repeated-W1 relay plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/repeated_w1_relay/PLAN.md)

A new W1-fixed six-permutation screen creates the clean cohort needed to test
the still-missing first-choice binding signal. Keeping W1's semantic content,
letter, and option position fixed while permuting only the other three options
produced 222 questions where W1 wins in the identity ordering but loses in at
least one alternative ordering (115 discovery, 107 held-out confirmation).
The key W1=A subset contains 77 questions (41/36): every token through the A
option line is identical across the chosen and unchosen presentations, so the
selectedness difference must arise after later competitors are processed.

The resulting causal test is now complete. In the exact cached execution regime,
38/41 discovery and 32/36 confirmation pairs retained the required A-chosen
donor / A-unchosen recipient contrast. Importing conventional-attention K/V
from the donor at only the empty first-answer decision boundary did not produce
the predicted Game/Neutral-opposed effect. The centered semantic-A interaction
was −0.010 [−0.039, +0.017] logits in discovery and +0.005 [−0.029, +0.038]
in confirmation; semantic-A choice was unchanged in confirmation. Both the
complete-cache positive control and untouched donor rows reproduced with zero
A-D logit error. Thus the null is interpretable: the selectedness-binding signal
is not stored in ordinary-attention K/V at that single boundary token, and the
prespecified localization gate failed.

- [W1-fixed permutation screen report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/w1_fixed_permutation_screen/analysis/REPORT.md)
- [Machine-readable screen summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/w1_fixed_permutation_screen/analysis/summary.json)
- [Frozen eligible-pair artifact](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/w1_fixed_permutation_screen/analysis/eligible_pairs.json)
- [Frozen six-permutation plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/w1_fixed_permutation_screen/plan.json)
- [First-decision boundary-K/V causal report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/w1_selectedness_boundary_kv/analysis/REPORT.md)
- [Boundary-K/V canonical figure](figures/qwen36_w1_selectedness_boundary_kv.png)
- [Boundary-K/V machine-readable summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/w1_selectedness_boundary_kv/analysis/summary.json)

The tighter identical-source attention-edge test is also complete. It held
every token through the original semantic-A line identical and verified that
the line's conventional-attention K/V was bit-exact across histories where A
later won or lost; only the ordering of B--D differed. Blocking the
original-A→repeated-A edge had a large pro-A relay effect in Neutral but
approximately none in Game. In held-out confirmation, the centered-A lesion
effect was -1.281 logits when A had won and -1.121 when it had lost in Neutral,
versus +0.161 and +0.128 in Game.

The winner-specific Game-minus-Neutral interaction pointed the same way in
both splits (+0.300 logits in discovery; +0.193 in confirmation), but the
held-out interval narrowly crossed zero [-0.020, +0.414]. Therefore this is
strong evidence for a policy-dependent difference in how the original option
line supports its repeated counterpart, but not a completed explanation of
how the model identifies the first-pass winner. In particular, the large
effect is a general Neutral reinstatement route; the extra dependence on
whether A actually won is modest and not conclusively replicated.

- [Identical-source selectedness-edge report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/w1_selectedness_edge/analysis/REPORT.md)
- [Identical-source selectedness-edge figure](figures/qwen36_w1_identical_source_selectedness_edge.png)
- [Machine-readable selectedness-edge summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/w1_selectedness_edge/analysis/summary.json)
- [Identical-source selectedness-edge plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/w1_selectedness_edge/PLAN.md)

The canonical all-candidate follow-up individually blocked each semantic
matching relay (original W1→repeated W1 through W4→W4) across ordinary-attention
blocks 4--48, using a cyclic nonmatching source lesion of the same structural
form as the control (not exactly token-count matched).
The Game--Neutral causal interaction is graded and nonlinear in first-pass
candidate evidence. A linear model gave an apparent W1 excess, but a later
audit with flexible candidate-score and competitor-gap controls leaves the R1
term uncertain on both splits and slightly worsens held-out prediction. The
data therefore do not establish a separate categorical winner state.

The direct W1 lesion also has opposite causal signs. Relative to natural,
blocking the matching W1 edge raises held-out W1 evidence by 0.362
[0.204, 0.521] logits in Game but lowers it by 1.178 [1.013, 1.350] logits in
Neutral. Game is therefore not merely omitting Neutral's reinstatement: its
semantic W1 match actively contributes against W1.

The joint intervention connects that mechanism directly to preferential
switching. On held-out conflict trials, Game W1 choice moved from 17.6% to
27.2% while Neutral moved from 39.7% to 27.2%; the in-sample point estimates
coincided, while the held-out gap-reduction interval [+11.8, +32.4] points
allows partial through over-elimination. Discovery independently reduced its
-15.3-point gap to
-1.5 points. The held-out W1--W2 margin gap shrank by 0.314 logits (49%), while
discovery's margin mediation was weaker. Thus policy-dependent semantic
matching relays explain essentially all of the replicated discrete choice gap
and part, but not all, of the continuous-logit gap. The remaining open question
is how the model computes the nonlinear rank transformation upstream.

The complete all-candidate natural-attention companion corrects an earlier
design error that truncated descriptive plots at block 48. Across all 16
ordinary-attention blocks, matching attention peaks around blocks 48--52 and
drops sharply at block 56 for every rank and condition. The last quarter also
reveals a rank effect that the truncated plot missed: from block 52 through 64,
both Game and Neutral allocate more matching attention to R1 than to R2--R4.
Neutral's R1 advantage is larger than Game's at blocks 52, 56, and 60. This is
direct observational evidence that late matching reads know which semantic
candidate won; it complements, but does not replace, the causal edge lesions.

- [All-candidate matched-relay report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/all_candidate_matched_relay/analysis/REPORT.md)
- [All-candidate canonical figure](figures/qwen36_all_candidate_matched_relay.png)
- [Machine-readable all-candidate summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/all_candidate_matched_relay/analysis/summary.json)
- [Complete late-layer attention summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/all_candidate_matched_relay/analysis/full_attention_trajectory_summary.json)
- [All-candidate frozen plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/all_candidate_matched_relay/PLAN.md)

The original restricted fixed-A serial-mediation run appeared to support the
inferred first-option→matching-repeated-option relay. That interpretation is
superseded by the exact full24 recalibration and the B/C/D generalization:
under the matched pipeline, the token-count-matched nonmatching receiver
removes at least as much counterfactual donor transfer. Natural-history lesions
still establish that repeated options read their semantic matches, but
matching-specific mediation of selected-line transplantation is not
established.

- [Historical fixed-A serial-mediation report, superseded by full24](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/fixed_a_donor_receiver_mediation/analysis/REPORT.md)

The direct option-state probe now finds the representation that the boundary
test missed. Using only the exact option-closing newline residuals, a shared
linear ranker trained on 251 questions predicts the model's choice on 249
held-out questions across all six mappings with 64.9% [60.5, 69.1] accuracy at
readout 53 and 62.0% [57.8, 66.3] at readout 64. The majority-letter A baseline
is 51.9% [46.7, 56.9]; the paired readout-53 gain is 13.0 points [5.3, 20.8].

More decisively, among 107 held-out same-content/same-letter pairs, the fitted
score for W1 at readout 53 is 4.36 units higher [3.06, 5.79] when distractor
ordering makes W1 win than when it loses. The difference is exactly zero for A,
whose prefix is identical before later options are seen, but positive for B,
C, and D. Thus the option-closing newline carries a context-dependent
candidate-value/selectedness signal. The newline was also used to construct
earlier content-aligned directions, but corrected cross-mapping tests found
only modest mapping-invariant content geometry around layers 32--43 and strong
late displayed-letter/position structure. This is robust held-out value
decoding, not proof of a clean semantic code at that exact token.

- [Option-newline selected-answer probe report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/option_newline_choice_probe/analysis/REPORT.md)
- [Canonical option-newline probe figure](figures/qwen36_option_newline_choice_probe.png)
- [Machine-readable option-newline probe summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/option_newline_choice_probe/analysis/summary.json)
- [Frozen option-newline probe plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/option_newline_choice_probe/PLAN.md)

The prespecified causal follow-up is now complete. It clamped only that fitted
coordinate at the first-presentation W1 newline over readouts 33--56, moving it
from the chosen-presentation score to the matched unchosen-presentation score.
On 38 held-out conflict questions, the Game-minus-Neutral W1-minus-W2
interaction was +0.040 [-0.029, +0.102] logits; the W1-choice interaction was
+5.3 [-5.3, +18.4] points. Discovery's continuous interaction was +0.057
[+0.006, +0.115], but its discrete effect differed, and the equal/opposite sign
control did not replicate. The exact zero-dose sham had zero effect.

The conclusion is narrower than the decoder finding: the one-dimensional
candidate-value direction is readable, but moving it does not reproducibly
control preferential Game revision. The relevant binding may be nonlinear,
multidimensional, distributed across option states, or encoded in a different
basis.

- [Option-newline candidate-value causal report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/option_newline_value_causal/analysis/REPORT.md)
- [Option-newline candidate-value causal plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/option_newline_value_causal/PLAN.md)
- [Option-newline candidate-value causal summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/option_newline_value_causal/analysis/summary.json)
- [Option-newline candidate-value causal figure](figures/qwen36_option_newline_value_causal.png)

**Correction and resolved rerun:** the first all-four projection run is invalid
as a causal test. It omitted the fitted displayed-letter centering and used
absolute cached-K/V replacement, thereby removing static A/B/C/D structure and
injecting cache reconstruction error. Those outputs remain audit-only.

The corrected run removed only the affine centered candidate-value coordinate
from all four first-presentation option-newline carriers at every ordinary
attention block 4--64. It added projected-minus-unprojected K/V deltas to the
live model state and included an exact zero-delta identity path. Across all
1,000 identity comparisons, the maximum A--D logit difference was 0.0 and no
answer changed.

The corrected effect is much smaller. On 136 held-out conflict questions, W1
choice changed by +1.5 [-2.2, +5.1] points in Game and -0.7 [-3.0, +1.5] in
Neutral; interaction +2.2 [-2.2, +6.6]. W1-minus-W2 margin changed by +0.020
[+0.003, +0.038] logits in Game and -0.002 [-0.022, +0.017] in Neutral;
interaction +0.023 [-0.003, +0.049]. Discovery's corresponding interaction was
only +0.002 [-0.025, +0.028]. This is weakly suggestive of W1 reinstatement in
held-out Game logits, but it is not a robust condition-specific mechanism. The
decoded one-dimensional score affects some choices; it does not explain the
preferential Game revision effect.

- [All-four candidate-value projection report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/option_newline_all_four_projection/analysis/REPORT.md)
- [All-four projection summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/option_newline_all_four_projection/analysis/summary.json)
- [All-four projection plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/option_newline_all_four_projection/PLAN.md)
- [Canonical all-four projection figure](figures/qwen36_option_newline_all_four_projection.png)
- [Corrected centered all-four report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/option_newline_all_four_centered_projection/analysis/REPORT.md)
- [Corrected centered all-four summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/option_newline_all_four_centered_projection/analysis/summary.json)
- [Corrected centered all-four plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/option_newline_all_four_centered_projection/PLAN.md)
- [Corrected centered all-four figure](figures/qwen36_option_newline_all_four_centered_projection.png)

The direct redundancy test is also complete. In one matched five-mode runner,
we removed (i) the centered candidate-value coordinate at all four original
option newlines, (ii) centered A--D identity at the first-decision position, or
(iii) both. The joint lesion did **not** recover W1 more strongly than either
single lesion. On 136 held-out conflict questions, letter-only removal produced
an attractive Game-minus-Neutral W1-choice effect of +3.7 [+0.7, +7.4] points,
but discovery's interaction was exactly 0.0 [-4.4, +4.4], and the held-out
continuous-margin interaction was only +0.006 [-0.022, +0.035] logits. Joint
removal changed Game W1 choice by 0.0 [-2.2, +2.2] points and had a negative
factorial interaction. Thus the readable option score and first-decision letter
identity are not established as redundant causal routes for selectedness
binding; that mechanism remains unresolved.

- [Joint score/letter causal report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/joint_option_score_decision_letter/analysis/REPORT.md)
- [Joint score/letter summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/joint_option_score_decision_letter/analysis/summary.json)
- [Joint score/letter plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/joint_option_score_decision_letter/PLAN.md)
- [Joint score/letter canonical figure](figures/qwen36_joint_option_score_decision_letter.png)

A complete first-decision boundary crossover now resolves the strongest
remaining version of the “stored winner” hypothesis. For 105 discovery and 93
held-out exact questions, it crossed the entire 64-block update at the empty
first-answer boundary between histories that naturally selected different
semantic answers. The manipulation changed the immediate decision to the donor
on 97.1%/98.9% of questions, but condition-specific donor-semantic transfer was
essentially zero: Neutral-minus-Game -0.019 [-0.045, +0.005] logits in
discovery and -0.009 [-0.035, +0.016] in confirmation. Complete donor history
gave +0.428 [+0.252, +0.603] and +0.466 [+0.268, +0.672] logits.

The crossed boundary carried a small trace of the donor's literal A--D letter,
not the semantic answer currently attached to that letter. The first-decision
position therefore contains a portable output-letter state but not a portable
semantic-winner binding: on confirmation its final literal-letter echo was only
+0.087/+0.077 centered logits in Game/Neutral, and final donor-semantic choice
changed only +0.5/-1.1 points. Which answer won is reconstructed from the distributed
first-presentation option history rather than read from the empty answer slot.
This fits the earlier
ordinary-attention K/V localization to the original selected option line and
explains why letter and one-dimensional score lesions repeatedly failed to
eliminate preferential semantic revision.

- [First-decision boundary-crossover report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/decision_boundary_crossover/analysis/REPORT.md)
- [Boundary-crossover summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/decision_boundary_crossover/analysis/summary.json)
- [Boundary-crossover plan](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/decision_boundary_crossover/PLAN.md)
- [Boundary-crossover canonical figure](figures/qwen36_first_decision_boundary_crossover.png)

A subsequent mechanism audit reconciles these nulls with the strongest
positive results and records three genuinely untested hypotheses. The most
direct missing test is a **cross-semantic transplant of the evaluation-period
GLA update**: existing transplants establish that this update carries the
`incorrect` policy, but always held W1 fixed and therefore never tested whether
the update also carries W1's semantic identity. If it does not, the leading
alternative is receiver-side reconstruction: the evaluation state gates the
original-option to repeated-option semantic match. This can be tested cleanly
with W1=A chosen/unchosen permutation pairs whose original A-line K/V is
byte-identical. The full evidence audit, falsifying outcomes, and a third
sequential-comparator hypothesis are recorded in the [canonical remapping
report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/analysis/REPORT.md#mechanism-audit-after-the-winner-letter-and-candidate-score-nulls).

A corrected follow-up resolved what the late W1 "suppression" actually consists
of. (The predecessor token-swap runner omitted per-row left-padding offsets, so
its causal token-swap result is superseded.) Swapping the complete `incorrect`
and `lost` token states at each residual readout 48--56 still has essentially no
effect: the largest W1-evidence change is 0.0072 centered units and the largest
W1-choice change is 1.10 percentage points. Swapping `different` with its
position-aligned Neutral counterpart `answer`, or swapping the feedback-end
periods, is similarly small; across all tested tokens and directions, the
largest W1-choice change is 1.83 percentage points. The literal late feedback
token states are therefore not the controller of the final semantic
transformation.

The natural within-block trajectory is much more informative. Mixers 52 and 53,
MLP 54, and especially Mixer 56 create the explicit Game--Neutral W1 divergence
(incremental contrasts -0.286, -0.182, -0.128, and -0.966). Mixer 56 does not
write W1 downward in Game: it increases centered W1 evidence from +0.219 to
+0.448 in Game but from +0.600 to +1.795 in Neutral. The largest apparent
"suppression" is therefore attenuated Game sharpening versus very strong
Neutral reinstatement. MLP 56 subsequently reverses about 0.304 logits of that
difference. Natural logits reproduce the trusted run bit-for-bit.

This localization generalizes across the semantic split. The same Mixer
52/Mixer 53/MLP 54/Mixer 56 sequence appears on the 227 W1 = W2 trials, with
Mixer 56 again dominant (-1.096 versus -0.966 on the 273 W1 != W2 trials).
However, the groups enter block 52 differently: Game-minus-Neutral W1 evidence
is already -0.160 on W1 = W2 trials but is +0.163 on conflict trials. Thus the
late transformation is general; the conflict design is what makes its target
interpretable as the previous semantic answer.

On conflict trials the pre-52 sign reflects opposing candidate trajectories,
not Game preparing to repeat W1: Game initially foregrounds W1 over W2
(+0.156 versus -0.122), while Neutral foregrounds W2 over W1 (-0.007 versus
+0.487). By MLP 56, Game favors W2 and Neutral favors W1. This is consistent
with Game first representing the historical answer as a rejection target and
then moving away from it, although that two-stage reading is not yet causally
established.

- [Late feedback-state swap and sublayer report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_state_swap/analysis/REPORT.md)
- [Feedback-token state-swap figure](figures/qwen36_simplemc_remapped_feedback_state_swap.png)
- [Late-sublayer trajectory figure](figures/qwen36_simplemc_remapped_late_sublayers.png)

An exact token-by-head source decomposition now shows where ordinary-attention
Mixers 52 and 56 obtain their W1-directed writes. On the 273 conflict trials,
Mixer 56 writes +0.168 centered W1 units in Game and +1.000 in Neutral. Its
-0.832 Game-minus-Neutral difference is dominated by contextual boundary states
inside the repeated presentation (-0.407), the final
`Your choice (A, B, C, or D):` cue (-0.314), and the repeated W1 option
(-0.226), not by direct reads from `incorrect`, `different`, or the feedback
period. A finer token audit shows that the first aggregate is almost entirely
the newline states after options B and C, while the choice-cue aggregate is
mostly its final space and literal `A` token. These are contextual summary
positions, not evidence that whitespace itself is semantically meaningful. H6
supplies about half of the total Mixer-56 contrast, with H15, H0, and H2
supplying most of the remainder. Mixer 52 is smaller (-0.175 total contrast)
and is led by H23 (-0.090).

The best current interpretation is therefore feedback-conditioned reprocessing
of the second presentation: Neutral strongly reconstructs or reinstates W1
from the repeated question, while Game does much less. The decomposition is an
exact additive account of the natural mixer write, not proof that each source
region is independently causal.

- [Exact Mixer 52/56 source and corrected token-swap report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/source_contributions_and_swaps/analysis/REPORT.md)
- [Mixer source-contribution figure](figures/qwen36_simplemc_remapped_mixer_source_contributions.png)
- [Corrected feedback-token swap figure](figures/qwen36_simplemc_remapped_feedback_token_state_swaps.png)

A within-Game semantic-target transfer test now distinguishes what Mixer 56 is
doing. Holding feedback and the complete repeated presentation fixed, we
changed the semantic answer implicit in the first presentation and transplanted
that donor's Mixer-56 query/gate and repeated-option K/V state. On the 104
held-out changed-winner questions, the joint patch moved Mixer 56 **toward** the
donor answer by 0.437 logit units (95% CI 0.194--0.693), the opposite of
transferring a suppression target. The reverse component-level effect also
appeared in discovery, but downstream layers nearly cancelled it: the held-out
final effect was -0.036 in the preregistered target-transfer sign convention
(95% CI -0.094 to +0.019), with only small choice changes. Mixer 56 therefore
carries content-specific reinstatement/reconstruction of the first-pass answer;
it is not itself the causal suppression mechanism. Natural logits were
bit-exact to the frozen donor-defining run across all 500 questions.

- [Mixer 56 semantic-target transfer report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/mixer56_target_transfer/analysis/REPORT.md)
- [Mixer 56 semantic-target transfer figure](figures/qwen36_simplemc_mixer56_semantic_target_transfer.png)

The clean causal carrier is the empty first-assistant boundary immediately
after the first question. Removing its GLA writes increased the Game margin of
the original semantic winner (W1) over the fresh-remapped winner (W2) by 0.243
logits (95% CI 0.108–0.381) on the 249-question confirmation split and
increased W1 selection by 9.6 percentage points. The direct Game margin effect
also appeared in discovery: +0.146 logits (95% CI 0.011–0.278). In confirmation,
the same lesion changed Neutral's W1–W2 margin by only +0.009 logits (95% CI
-0.113–0.128).

Option-token GLA writes have a distinct role: ablating them reduces W1 support,
especially in Neutral. They therefore help retain or reconstruct the earlier
answer but are not themselves the Game-specific suppression operation. The
joint content-plus-boundary lesion has a large difference-in-differences effect,
but it mixes these two functions and should not be described as one unitary
suppression mechanism.

A complementary final-decision experiment tested the semantic representation
directly. At every one of the 64 post-block readouts, it measured and removed
the projection of the second-answer decision state onto the question-specific
W1 semantic vector derived from four option remappings. The unmodified companion
passes exactly reproduced all 500 historical Game and Neutral logit rows, and
the intervention removed more than 99.7% of the measured projection after each
layer.

The result is real but partial. On the 273 questions where W1 differs from the
answer selected by a fresh solution of the remapped question (W2), continuous
ablation reduced the pooled Game-specific W1-versus-W2 targeting contrast by
0.130 logits (95% CI 0.071–0.189), or 31.6%. The held-out confirmation estimate
was 0.206 logits (95% CI 0.140–0.275). However, it reduced the pooled behavioral
W1-avoidance gap by only 2.2 percentage points (95% CI -2.9 to +7.3), or 12.5%,
and the held-out reduction was only 1.5 points.

A post-hoc split reanalysis materially qualifies the original interpretation.
On held-out W1 != W2 questions, ablation increased W1 selection in both Game
(+5.9 points) and Neutral (+4.4 points), chiefly while reducing W2 selection.
On held-out W1 = W2 questions, it instead reduced W1 selection in both Game
(-6.2 points) and Neutral (-12.4 points). The constructed direction is therefore
causally involved in W1-related decision computation, but cannot currently be
described simply as positive W1 evidence or as a Game-specific suppression
signal. The reason for the counterintuitive Neutral effect on W1 != W2 trials is
now largely resolved: signed projection-zeroing was not a pure ablation. Whenever
the W1 projection was negative, it added the W1-defined direction in order to
move the projection to zero. The result still leaves open how much of the natural
condition gap is Game suppression versus Neutral reinstatement. Full numbers and
interpretation limits are recorded in the linked final-decision report.

A frozen positive-only follow-up removed only positive W1 projections and left
negative projections unchanged. Its natural companion reproduced the prior run
bit-for-bit across all 500 questions. This modification did not rescue a simple
"positive W1 evidence" interpretation. On held-out W1 != W2 questions, W1
selection increased by 5.9 points in Game but also by 3.7 points in Neutral,
reducing the natural Neutral-minus-Game W1-selection gap by only 2.2 points
(95% CI -3.7 to +8.1). The corresponding discovery estimate was 1.5 points
(95% CI -5.1 to +8.0). A confirmation logit contrast was larger (+0.11 centered
W1 logits), but its discovery estimate was only +0.04 and included zero. Thus
the direction is causally involved in answer computation, but positive-only
removal still does not locate a robust Game-specific semantic-suppression
mechanism.

The original **bidirectional** W1 projection-zeroing intervention shows a
replicated agreement-dependent reversal. On W1 != W2 questions it increased W1
choice in both Game and Neutral (pooled +5.1 and +3.3 points, respectively). On
W1 = W2 questions it decreased W1 choice in both conditions (pooled -4.4 and
-8.8 points). Every direction replicated in the frozen discovery and
confirmation halves. Thus the constructed direction is causally relevant, but
it is not context-independent positive evidence for W1: its effect depends on
whether W1 agrees with the answer supported by a fresh solution of the remapped
second presentation. The canonical signed-ablation report now contains the
complete split-wise table.

Crucially, the positive-only Neutral effect does not replicate across splits.
On W1 != W2 questions it was -2.2 points in discovery (5 questions entered W1,
8 left) and +3.7 points in confirmation (8 entered, 3 left). Pooled over all 273
questions, the change was only +0.7 points (95% CI -2.6 to +4.0): 13 entered W1
and 11 left. Nor did positive-only removal generally raise W1's score. In pooled
Neutral trials, the centered W1 logit fell by 0.037; W2 fell by 0.055; and the
other two options rose by 0.046 on average. Thus the remaining confirmation
increase consists of a few heterogeneous boundary crossings, not a general
boost to W1. The focused diagnostic and figure linked below make this comparison
explicit.

An audit of every causal analysis using remapped option letters found and
corrected one analysis-only tie-breaking error: a few scripts reordered A-D
logits into semantic-content order before taking `argmax`. Exact ties must be
resolved in displayed A-D order first. Across the affected arrays, this changed
0.20%–1.59% of discrete answer cells and did not change any qualitative
conclusion. It cannot affect raw logits, margins, entropy, projections,
activation norms, or other continuous effects. The canonical reports linked
below have been regenerated with the correct rule.

A paired follow-up directly measured whether the validated first-answer-
boundary GLA lesion changes the old-answer (W1) semantic activation at the
final decision position. It does not do so in a replicated Game-specific way.
On held-out W1≠W2 questions, the lesion reproduced the Game-specific output
effect (Game W1−W2 margin +0.222 logits, 95% CI +0.087 to +0.352; Neutral
-0.016, 95% CI -0.138 to +0.099), but its effect on the final W1 semantic
projection was similar in Game and Neutral (Game -1.017 residual units;
Neutral -1.173; interaction +0.156, 95% CI -0.864 to +1.175). Discovery also
showed no reliable projection interaction and did not reproduce the held-out
projection direction. Thus the boundary route changes how W1 is ranked without
simply adding or removing the measured one-dimensional W1 activation.

A subsequent content-rewriting test directly challenged the semantic-memory
interpretation of the first-answer boundary. It transplanted all four tensors
that define the boundary GLA memory updates (`key`, `value`, decay gate `g`, and
`beta`) from a same-question alternative presentation that produced a different
semantic winner. The intended transfer did not occur: pooled Game change in the
W1-minus-donor-answer margin was -0.001 logits (95% CI -0.030 to +0.028), and
the Game-minus-Neutral divergence was -0.020 logits (95% CI -0.048 to +0.008).

Instead, a small replicated effect followed the donor's old literal response
letter in both Game and Neutral (about -0.043 logits). The boundary GLA route is
therefore causally relevant, but its portable content looks mapping- or
letter-specific rather than like a semantic record of W1. The earlier boundary
lesion should not by itself be described as locating the remembered semantic
answer.

- [First-presentation GLA-memory causal report](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/first_span_gla_ablation/analysis/REPORT.md)
- [Compact numerical summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/first_span_gla_ablation/analysis/summary.json)
- [Boundary-lesion / final-W1-activation report](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/first_boundary_semantic_projection/analysis/REPORT.md)
- [Boundary-lesion / final-W1-activation numerical summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/first_boundary_semantic_projection/analysis/summary.json)
- [Continuous final-decision W1-semantic ablation report](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/final_decision_semantic_ablation/analysis/REPORT.md)
- [Continuous final-decision numerical summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/final_decision_semantic_ablation/analysis/summary.json)
- [Positive-only final-decision W1 ablation report](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/final_decision_positive_only_exact/analysis/REPORT.md)
- [Positive-only numerical summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/final_decision_positive_only_exact/analysis/summary.json)
- [Why Neutral sometimes chose W1 more](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/final_decision_positive_only_exact/analysis/NEUTRAL_W1_DIAGNOSTIC.md)
- [Neutral W1 diagnostic numbers](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/final_decision_positive_only_exact/analysis/neutral_w1_diagnostic.json)
- [Positive-only per-question data](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/final_decision_positive_only_exact/data/per_question_condition.csv)
- **Positive-only per-question layerwise data** — local-only: `outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/final_decision_positive_only_exact/data/per_question_condition_layer.csv`.
- [Negative-only final-decision W1 ablation report](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/final_decision_negative_only/analysis/REPORT.md) — On pooled W1≠W2 trials, removing negative W1 projection raised W1 choices in both Game (+5.49 pp) and Neutral (+4.40 pp), but the +1.10 pp condition difference was unreliable. Negative projection therefore carries causal `not-W1` content but is not the Game-specific suppression mechanism.
- [Negative-only numerical summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/final_decision_negative_only/analysis/summary.json)
- [Negative-only pooled summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/final_decision_negative_only/analysis/pooled_summary.json)
- [Positive-only W2 semantic ablation report](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/final_decision_w2_positive_only/analysis/REPORT.md) — On W1≠W2 trials, removing the fresh remapped-Baseline winner W2 reduced W2 choices in both frozen splits and both conditions (Game: -7.3/-8.8 pp; Neutral: -11.7/-9.6 pp). It reduced switching similarly in Game and Neutral and showed no reliable condition-specific effect. W2 is causally used as a candidate, but this direction does not explain Game-specific revision.
- [Positive-only W2 numerical summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/final_decision_w2_positive_only/analysis/summary.json)
- [Remapped-answer tie audit](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/REMAPPED_ARGMAX_TIE_AUDIT.md)
- [Machine-readable tie audit](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/REMAPPED_ARGMAX_TIE_AUDIT.json)
- [First-boundary GLA semantic-memory rewrite report](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/first_boundary_gla_memory_rewrite/analysis/REPORT.md)
- [First-boundary GLA rewrite numerical summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/first_boundary_gla_memory_rewrite/analysis/summary.json)
- [First-boundary accumulated GLA-state transplant report](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/first_boundary_gla_state_transplant/analysis/REPORT.md)
- [First-boundary accumulated-state numerical summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/first_boundary_gla_state_transplant/analysis/summary.json)

The accumulated-state transplant held the recipient prompt fixed and replaced
all 48 GLA recurrent matrix states immediately after the first answer. It did
not selectively transfer suppression of a donor semantic winner: on frozen
confirmation, Game minus Neutral was +0.020 logits (95% CI -0.068 to +0.109).
A same-winner/different-mapping state was at least as behaviorally disruptive.
Thus neither the local boundary write tensors nor the complete GLA matrix state
at that boundary is a portable semantic-answer memory. The earlier boundary-
write lesion still establishes causal involvement, but the remembered answer
is likely distributed across state types or relationally encoded.

## Curated figures

These are the current presentation figures. Final figures belong here as PNGs;
plots buried under `outputs/` are run artifacts.

- [Answer-contraction summary](figures/qwen36_27b_simplemc_corrected/answer_contraction_summary.png)
- [Rank-contraction decomposition](figures/qwen36_27b_simplemc_corrected/game_rank_contraction_decomposition.png)
- [Full residual-stream variance control](figures/qwen36_27b_simplemc_corrected/full_residual_variance.png)
- [Baseline function of Mixers 56 and 63](figures/qwen36_27b_simplemc_corrected/baseline_mixer_function.png)
- [Mixer 56 across Baseline, Game, and Neutral](figures/qwen36_27b_simplemc_corrected/mixer56_across_conditions.png)
- [Feedback-end complete-residual replacement](figures/qwen36_27b_simplemc_corrected/feedback_end_residual_replacement.png)
- [First-presentation GLA-memory lesions](figures/qwen36_27b_simplemc_corrected/first_span_gla_ablation.png)
- [Boundary GLA lesion versus final W1 semantic activation](figures/qwen36_first_boundary_semantic_projection.png)
- [Final-decision W1-semantic activation and ablation](figures/qwen36_27b_simplemc_corrected/final_decision_semantic_ablation.png)
- [Final-decision positive/negative W1 activation by condition and W1–W2 conflict](figures/qwen36_27b_simplemc_corrected/final_decision_semantic_sign_strength.png) — Held-out natural executions show that Game and Neutral have nearly identical gross positive and negative projection-strength trajectories in both conflict and non-conflict trials; the causal interaction is not explained by a large difference in activation amount along this one-dimensional axis.
- [Why Neutral sometimes chose W1 more after ablation](figures/qwen36_27b_simplemc_corrected/neutral_w1_ablation_diagnostic.png)
- [First-boundary GLA semantic-memory rewrite](figures/qwen36_27b_simplemc_corrected/first_boundary_gla_memory_rewrite.png)
- [Action-ending-period source lesion](figures/qwen36_action_period_source_lesion.png)
- [Semantic binding: attention versus MLP factorial](figures/qwen36_semantic_binding_module_factorial.png)
- [Option-newline candidate-value causal intervention](figures/qwen36_option_newline_value_causal.png)
- [Continuous first-decision A–D letter scrub](figures/qwen36_first_decision_letter_scrub.png) — Removing the complete centered A–D JLens subspace from the live first-decision residual after every readout 48–63 explained 0% of the held-out 22.8-point preferential-W1-avoidance gap (Game and Neutral each changed +0.7 points; interaction 0.0, 95% CI -2.2 to +2.2). The late explicit answer-letter state is not the semantic binding route.
- [Continuous first-decision letter-scrub report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/first_decision_letter_scrub/analysis/REPORT.md)
- [Continuous first-decision letter-scrub summary](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/first_decision_letter_scrub/analysis/summary.json)
- [OLMo 2 32B clean SimpleMC behavioral gate](figures/model_replications/olmo2_32b_simplemc_behavioral_gate.png) — cross-model behavioral non-replication: `incorrect` does not increase switching relative to `lost` and instead gives the remapped semantic old winner more relative support.
- [Gemma 4 31B two-dataset negative-model comparison](outputs/model_replications/gemma4_31b_negative_model_comparison/REPORT.md) — Gemma fails the clean remapped choice-rate gate (SimpleMC +0.6 points `[-2.2,+3.4]`; TriviaMC +0.8 `[-0.6,+2.2]`) but retains the strategic computation in continuous scores. Game versus Neutral selectively lowers semantic old-R1 by 0.850/0.650 logits and raises alternatives; the seven-token feedback suffix transfers 99.95%/95.54% of the held-out policy vector; and matching-history blockade plus the direct policy × recollection factorial causally show that the installed suffix policy changes how recalled semantic rank is used. On confirmation, the installed-Game versus installed-Neutral matching-route interaction is `R1 +1.398, R2 -0.195, R3 -0.499, R4 -0.705` on SimpleMC and `+0.683/-0.348/-0.212/-0.124` on TriviaMC, identically in native Game and Neutral recipients. Displayed-choice interactions remain small. The negative-model difference is therefore expression across the argmax boundary, not absence of policy, recollection, or their interaction. Policy-factorial figures: [SimpleMC](figures/model_replications/gemma4_31b_simplemc_policy_recollection_factorial.png) / [TriviaMC](figures/model_replications/gemma4_31b_triviamc_policy_recollection_factorial.png); displayed-letter-controlled trajectories: [SimpleMC](figures/model_replications/gemma4_31b_simplemc_nonremapped_rank_trajectories_letter_controlled.png) / [TriviaMC](figures/model_replications/gemma4_31b_triviamc_nonremapped_rank_trajectories_letter_controlled.png); [plan](outputs/model_replications/gemma4_31b_negative_model_comparison/PLAN.md).
- [Seed-OSS 36B clean SimpleMC behavioral replication](figures/model_replications/seed_oss_36b_simplemc_clean_behavioral_gate.png) — positive cross-model replication: +14.2-point same-order and +8.6-point remapped semantic switching gaps, with selective suppression of old-winner content rather than its former letter.
- [Seed-OSS 36B clean TriviaMC behavioral replication](figures/model_replications/seed_oss_36b_triviamc_clean_behavioral_gate.png) — prespecified dataset generalization: +7.6-point same-order and +5.4-point remapped semantic switching gaps, again with selective semantic W1 suppression.
- [Seed-OSS 36B SimpleMC causal matching-history replication](figures/model_replications/seed_oss_36b_simplemc_matching_history.png) — across all 64 grouped-query attention layers, blocking the four true matching 1P→2P option-line reads rather than four cyclic wrong-line reads raises held-out Game W1 evidence by +1.197 logits and lowers W3/W4 by -0.527/-0.625. Natural held-out old-W1 choice is 36.5% Game versus 45.8% Neutral; matching blockade changes this to 49.0% versus 47.8%, eliminating the preferential-avoidance point estimate. The primary matching-minus-cyclic change is +7.2 points `[+1.6,+12.9]`, with independent discovery +8.8 `[+2.0,+15.5]`. This is a causal cross-architecture replication of Game's recollection mechanism. Natural Neutral's matching-specific profile is not stable. [Report](outputs/model_replications/seed_oss_36b_mechanistic_replication/simplemc/matching_history/analysis/REPORT.md) · [summary](outputs/model_replications/seed_oss_36b_mechanistic_replication/simplemc/matching_history/analysis/summary.json) · [plan](outputs/model_replications/seed_oss_36b_mechanistic_replication/PLAN.md).
- [Seed-OSS 36B TriviaMC causal matching-history replication](figures/model_replications/seed_oss_36b_triviamc_matching_history.png) — the complete all-64-layer test independently passes both frozen confirmation endpoints: the Game-minus-Neutral W1 matching-versus-cyclic interaction is +1.277 `[+0.942,+1.618]` centered logits, and the old-W1 choice-gap change is +10.0 `[+5.2,+14.8]` points. Natural Game/Neutral old-W1 choice is 68.4%/75.6%; matching blockade makes both 76.8%, while cyclic wrong-line blockade retains 68.0%/78.0%. [Report](outputs/model_replications/seed_oss_36b_mechanistic_replication/triviamc/matching_history/analysis/REPORT.md) · [summary](outputs/model_replications/seed_oss_36b_mechanistic_replication/triviamc/matching_history/analysis/summary.json).
- [Seed-OSS 36B complete feedback-suffix policy crossover](outputs/model_replications/seed_oss_36b_mechanistic_replication/REPORT.md) — crossing only the seven tokens from `incorrect/lost` through `Choose the answer again.` at every Seed layer transfers essentially the entire paired natural Game/Neutral A--D score difference through ordinary-attention K/V. Confirmation transfer is 1.0002 in both directions on SimpleMC and 0.9999 on TriviaMC, with both frozen discovery halves agreeing. Old-W1 rank evidence and displayed choice reverse with the donor suffix. Natural and real distinct-row identity controls are exactly 0.0-error. [SimpleMC figure](figures/model_replications/seed_oss_36b_simplemc_feedback_suffix.png) · [TriviaMC figure](figures/model_replications/seed_oss_36b_triviamc_feedback_suffix.png) · [plan](outputs/model_replications/seed_oss_36b_mechanistic_replication/PLAN.md).
- [Seed-OSS 36B complete final-decision trajectories](outputs/model_replications/seed_oss_36b_final_position_trajectories/PLAN.md) — every final-position post-block residual L1--L64 was collected for all 500 SimpleMC and 500 difficulty-filtered TriviaMC questions. The standard Seed logit lens makes the answer-rank pattern directly readable around L40--L42. Held-out all-question decoders recover the paired final Game-minus-Neutral pattern earlier, persistently from L36 on SimpleMC and L39 on TriviaMC; at L40 learned/readout cosine is 0.575/0.223 and 0.446/0.154. Exact final Game-minus-Neutral R1--R4 effects are `-1.530/+0.121/+0.450/+0.958` and `-2.142/+0.766/+0.720/+0.656`, replicating selective old-winner suppression plus lower-rank support. The narrow Qwen switch-panel timing does not exactly replicate: Seed often shows R1 already readable before R2 later overtakes it. A matched conventional-logit-lens reread of Qwen preserves Qwen's R2-already-leading pattern, so this difference is not caused by the former readout mismatch. [Trajectory report](outputs/model_replications/seed_oss_36b_final_position_trajectories/analysis/REPORT.md) · [decoder report](outputs/model_replications/seed_oss_36b_final_position_trajectories/prospective_decoding/analysis/REPORT.md) · [matched Qwen readout](outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/standard_logit_lens/comparison/REPORT.md) · displayed-letter-controlled [SimpleMC](figures/model_replications/seed_oss_36b_simplemc_nonremapped_rank_trajectories_letter_controlled.png) / [TriviaMC](figures/model_replications/seed_oss_36b_triviamc_nonremapped_rank_trajectories_letter_controlled.png) · policy-decoder [SimpleMC](figures/prospective_decoding/seed_oss_36b_simplemc_policy_adjusted_prospective_decoding.png) / [TriviaMC](figures/prospective_decoding/seed_oss_36b_triviamc_policy_adjusted_prospective_decoding.png).
- [Seed-OSS 36B direct policy × recollection factorial](outputs/model_replications/seed_oss_36b_mechanistic_replication/REPORT.md) — crossing the installed Game/Neutral seven-token feedback-suffix state with intact, matching-blocked, and cyclic-wrong 1P→2P history access shows that policy causally changes how matching candidate history is used. On frozen confirmation questions, the installed Game-versus-Neutral suffix changes the matching-history old-W1 route by +1.078 centered logits on SimpleMC and +1.279 on TriviaMC, identically in native Game and Neutral recipient prompts. Every prior-result and identity cell reproduced exactly. [SimpleMC report](outputs/model_replications/seed_oss_36b_mechanistic_replication/simplemc/policy_recollection_factorial/analysis/REPORT.md) · [TriviaMC report](outputs/model_replications/seed_oss_36b_mechanistic_replication/triviamc/policy_recollection_factorial/analysis/REPORT.md).
- [Seed-OSS 36B fresh-2P × recollection decomposition](outputs/model_replications/seed_oss_36b_mechanistic_replication/REPORT.md) — corrected analysis separates complete, conflict, and non-conflict confirmation questions. The original conflict-choice endpoint was weak before intervention (+4.8 points SimpleMC and +5.5 TriviaMC, both intervals including zero), so its null scrub interaction did not establish a Seed analogue of Qwen's dissociation. At score resolution, removing 97.26%/98.31% of the discovery-fitted fresh coordinate reduces the old-W1 task gap by +0.215 `[+0.143,+0.292]` logits on full SimpleMC and +0.249 `[+0.010,+0.491]` on TriviaMC conflicts. Matching-history blockade removes the aggregate choice gap (-10.4/-7.2 points) but not the conflict gap (-0.8/0.0), and the joint lesion leaves significant conflict score gaps (-0.259/-0.362). Thus aggregate matching-recollection dependence replicates, but the Qwen recollection-versus-fresh-evidence dissociation does not: Seed uses a material fresh-score contribution plus an unlocalized conflict-route component. [SimpleMC report](outputs/model_replications/seed_oss_36b_mechanistic_replication/simplemc/fresh_removal/analysis/REPORT.md) · [TriviaMC report](outputs/model_replications/seed_oss_36b_mechanistic_replication/triviamc/fresh_removal/analysis/REPORT.md) · figures [SimpleMC](figures/model_replications/seed_oss_36b_simplemc_fresh_history_double_dissociation.png) / [TriviaMC](figures/model_replications/seed_oss_36b_triviamc_fresh_history_double_dissociation.png).

- [Canonical non-remapped option-line versus first-answer-boundary test](outputs/model_replications/canonical_history_decision_factorial/analysis/REPORT.md) — the reduced five-cell causal factorial completed on Qwen3.6-27B, Seed-OSS 36B, and Gemma 4 31B across both 500-question datasets. On frozen confirmation, blocking each complete 2P option line from reading its identical complete 1P option line weakened the Game-minus-Neutral old-W1 logit effect in all six cells by 0.55/1.27 (Qwen), 1.95/1.92 (Seed), and 0.86/0.91 logits (Gemma SimpleMC/TriviaMC); cyclic wrong-line controls were approximately null. Cutting every outgoing route from the exact first answer-generation boundary was small (−0.04 to +0.12 logits) and never reproducibly reduced the choice gap. Joint cells add no reliable choice effect beyond matching alone; Qwen shows a continuous-score-only interaction after the matching route is cut. Thus the candidate-matched first-option-line route is used in the ordinary prompt, while the missing first-answer boundary is not the primary recollection source. The semantic interpretation comes from combining this canonical result with the earlier remapped mapping-invariance evidence; non-remapped lines alone confound meaning with displayed letter and line identity. [Figure](figures/model_replications/canonical_history_decision_factorial.png) · [summary](outputs/model_replications/canonical_history_decision_factorial/analysis/summary.json) · [plan](outputs/model_replications/canonical_history_decision_factorial/PLAN.md).

- [Confidence-slope supplement to the canonical factorial](outputs/model_replications/canonical_history_decision_factorial/confidence_slope_supplement/REPORT.md) — where the confidence dose-response travels. Inside the factorial's causal cells, the slope of the Game-minus-Neutral old-W1 push on first-pass confidence is untouched by severing every outgoing signal from the would-be first-answer position (e.g., Qwen +0.73→+0.79, Seed +0.95→+0.93), but collapses by 77–97% under the matching-line blockade in four of the five cells that show scaling (Qwen +0.73→+0.17 and +1.26→+0.24, Seed SimpleMC +0.95→+0.11, Gemma SimpleMC +0.56→+0.02; Seed TriviaMC partial). On Qwen TriviaMC the remnant after the matching cut (+0.24) is removed by additionally cutting the first-decision position (+0.02), so Qwen's backup route there carries graded confidence only once the line route is gone. The graded confidence the policy consumes travels with the retrieved option-line scores, not as a summary stored at the answer position. Natural slopes exactly reproduce the audited dose-response values and baseline rankings exactly reproduce the factorial's stored rank order (both asserted in the script).

- [Cross-model R2-redistribution contrast](figures/model_replications/r2_redistribution_contrast.png) — the direct test of a second-choice-targeting reading of the natural rank profiles. On paired natural runs, the old runner-up's share of the score redistributed away from the old winner, relative to the mean R3/R4 gain, is `-0.253 [-0.312,-0.193]` on Qwen SimpleMC, `-0.173 [-0.274,-0.074]` on Qwen TriviaMC, and `-0.504 [-0.616,-0.389]` on Seed SimpleMC; Seed TriviaMC is `+0.134 [-0.042,+0.311]`. The runner-up therefore gains **less** than the lower-ranked candidates in three of four model-by-dataset cells and never reliably more, with each negative contrast replicating independently on both frozen splits. R2 wins most switches despite being the policy's least-favored alternative, because it starts closest to the top. Observational natural-run evidence; the causal case remains the Qwen categorical-winner audit, matching-route lesions, and destination analysis. [Report](outputs/model_replications/r2_redistribution_contrast/analysis/REPORT.md) · [summary](outputs/model_replications/r2_redistribution_contrast/analysis/summary.json).

- [Relative margin expression](figures/model_replications/relative_margin_expression.png) — descriptive cross-model comparison, on the canonical non-remapped trajectory endpoint, of the natural Game-minus-Neutral R1 score adjustment against each model's own final-position top-2 decision margin. Absolute adjustments overlap across models (0.57--2.26 logits), but median Neutral margins span an order of magnitude (Qwen 0.73/3.25, Seed 2.25/12.0, Gemma 5.5/18.4), so the adjustment-to-margin ratio is within-dataset monotone in exactly the behavioral order: SimpleMC 0.78/0.67/0.16 and TriviaMC 0.50/0.19/0.05 for Qwen/Seed/Gemma, with the fraction of questions whose margin is smaller than the adjustment falling from 43% to 2%. This quantifies why similar-sized internal policies express as a large choice-rate effect in Qwen, a moderate one in Seed, and none in Gemma; no margin is intervened on, so it is observational. [Report](outputs/model_replications/relative_margin_expression/analysis/REPORT.md) · [summary](outputs/model_replications/relative_margin_expression/analysis/summary.json).
- [Confidence dose-response of the Game revision policy](outputs/model_replications/confidence_dose_response/analysis/REPORT.md) — descriptive natural-run analysis of each model's own first-presentation top-1/top-2 logit margin against its final Game-minus-Neutral suppression of that old winner. The W1-suppression slope is positive on both frozen splits in Qwen SimpleMC/TriviaMC (+0.727/+1.260 logits per one-SD confidence), Seed SimpleMC/TriviaMC (+0.947/+0.318), and Gemma SimpleMC (+0.557); Gemma TriviaMC is nonlinear and split-unstable (+0.099 `[-0.015,+0.214]`). A magnitude-versus-direction control rules out a merely generic gain account: scale-free alignment with selective W1 suppression rises reliably with confidence in all six cells, including Gemma TriviaMC. Choice-level dose response does not rise correspondingly after the highly collinear signed Neutral margin, so graded internal adjustment and argmax expression dissociate. This is observational: downstream saturation or correlated question features can still produce the association, and no causal confidence readout is established. [Figure](figures/model_replications/confidence_dose_response.png) · [machine-readable summary](outputs/model_replications/confidence_dose_response/analysis/summary.json).

- [Per-condition attribution of the confidence dose-response](outputs/model_replications/confidence_dose_response/per_condition_attribution/REPORT.md) — the primary dose-response is a Game-minus-Neutral difference, which either condition can move. Measuring each condition's old-W1 change against the first-pass centered score: only Qwen shows Game-side confidence scaling on both datasets (SimpleMC Game slope -0.80 per SD of confidence with Neutral flat at -0.07; TriviaMC -2.31 versus -1.05). Seed shows the pattern on SimpleMC only (-1.11 versus -0.16); Gemma's Game slopes are flat and its SimpleMC Neutral slope is positive (+0.60), so its differential may be Neutral-side reinstatement — one cell, hypothesis only. Per-condition slopes conflate policy with shared second-reading re-scoring (the difference cancels it), and the Game-minus-Neutral slope differences reproduce the primary differential exactly (asserted in the script). Descriptive only.

- [General MCQ uncertainty direction at the second decision](outputs/model_replications/mcq_uncertainty_policy/analysis/REPORT.md) — a discovery-fitted high-minus-low 1P entropy direction transfers across SimpleMC and TriviaMC at L64 (`r=.855/.773` in Qwen and `.585/.486` in Seed), remains predictive after controlling the displayed old-winner letter, and has at most 0.19–5.41% direct overlap with the centered A–D output subspace. Its natural projection is strongly structured at the 2P decision position. All-layer ±3 steering with an equal-dose orthogonal control causally changes the final rank vector in both models and datasets: middle-to-late positive uncertainty steering lowers old R1 and raises lower ranks. This is not a Game-specific readout—Seed's Game/Neutral effects are nearly equal, and Qwen's effect is substantially stronger in Neutral. Choice effects at this dose are small. The measured axis is a general uncertainty-like candidate-ranking control available at 2P, not the condition-specific trigger for preferential Game switching. [Figure](figures/model_replications/mcq_uncertainty_policy.png) · summary (local-only): `outputs/model_replications/mcq_uncertainty_policy/analysis/summary.json` · [plan](outputs/model_replications/mcq_uncertainty_policy/PLAN.md).

Directory: [`figures/qwen36_27b_simplemc_corrected/`](figures/qwen36_27b_simplemc_corrected/)

## Other indexed research threads

These remain useful for research history and cross-model comparison, but their
prompt versions and status must be checked before treating them as directly
comparable with the current canonical Qwen run.

### Behavioral screening and replication

- [Seed-OSS 36B clean SimpleMC + TriviaMC replication](outputs/model_replications/seed_oss_36b_clean_behavioral_replication/REPORT.md) — the same pinned Seed revision completed both 500-question clean-paradigm datasets. SimpleMC switching was 31.0% Game versus 16.8% Neutral in the same order and 63.6% versus 55.0% after complete remapping; TriviaMC was 16.0% versus 8.4% and 30.6% versus 25.2%. Both frozen confirmation splits pass the prespecified behavioral gate. In both datasets Game selectively lowers the semantic old winner after remapping, while the old literal letter control is much smaller. Format-complete unrestricted-output sensitivity analyses agree with the all-question conditional A-D results. [SimpleMC report](outputs/model_replications/seed_oss_36b_clean_behavioral_replication/simplemc/analysis/REPORT.md) · [TriviaMC report](outputs/model_replications/seed_oss_36b_clean_behavioral_replication/triviamc/analysis/REPORT.md) · [plan](outputs/model_replications/seed_oss_36b_clean_behavioral_replication/PLAN.md).
- [OLMo 2 32B clean SimpleMC behavioral gate](outputs/model_replications/olmo2_32b_simplemc_behavioral_gate/analysis/REPORT.md) — all 500 clean-paradigm questions completed with 100% answer-only compliance. Same-order switching was 18.6% in Game versus 18.4% in Neutral (+0.2 points [−3.4,+4.0]). After every option moved letters, Game switched semantic answers less than Neutral, 65.6% versus 72.0% (−6.4 points [−9.6,−3.2]); Game-minus-Neutral W1 evidence was +0.206 logits and Game entropy was 0.143 bits lower. The semantic-versus-literal control confirms an inverse semantic policy effect, so OLMo fails the mechanistic-replication gate. [Summary](outputs/model_replications/olmo2_32b_simplemc_behavioral_gate/analysis/summary.json) · [figure](figures/model_replications/olmo2_32b_simplemc_behavioral_gate.png) · [plan](outputs/model_replications/olmo2_32b_simplemc_behavioral_gate/PLAN.md).
- [SimpleMC candidate comparison](outputs/reproduction/SIMPLEMC_CANDIDATE_COMPARISON.md)
- [Large-model SimpleMC screen](outputs/reproduction/LARGE_MODEL_SIMPLEMC_SCREEN.md)
- [Cross-model selective-reranking analysis](outputs/reproduction/selective_reranking_cross_model/REPORT.md)
- [Qwen cross-dataset behavioral summary](outputs/reproduction/QWEN_CROSS_DATASET_SUMMARY.md)
- [Llama-405B cross-dataset summary](outputs/reproduction/llama31_405b_cross_dataset_summary.md)

### Mechanistic and causal work

- [Original mechanistic plan](MECHINTERP_PLAN.md)
- [Implementation details](MECHINTERP_IMPLEMENTATION.md)
- [Qwen3.6-27B component sweep](CAUSAL_COMPONENT_SWEEP_RESULTS.md)
- [TriviaMC component transfer](TRIVIAMC_COMPONENT_TRANSFER_RESULTS.md)
- [Source-route interventions](SOURCE_ROUTE_CAUSAL_RESULTS.md)
- [Position-component interventions](POSITION_COMPONENT_CAUSAL_RESULTS.md)
- **Exact positive/negative W1 semantic-direction JLens explorer** — local-only: `outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/contextual_option_representations/analysis/semantic_direction_jlens_explorer.html`.
- [Qwen3.5-397B SimpleMC report](outputs/mechanistic/qwen35_397b_simplemc/MECHINTERP_REPORT.md)
- [Llama-405B mechanistic report](outputs/mechanistic/LLAMA31_405B_MECHINTERP_REPORT.md)

The earlier Qwen3.6-27B mechanistic and causal documents are indexed for
traceability, not endorsed as prompt-matched replications of the current run.

## Repository map

| Location | Purpose | Browse manually? |
|---|---|---|
| `README.md` | Canonical index and current conclusions | **Start here** |
| `PROMPT_SPEC.md` | Exact current model-visible prompts | Yes |
| `figures/` | Small set of curated, final PNG figures | Yes |
| `mechanistic/` | Analysis and intervention code | When reproducing an analysis |
| `outputs/reproduction/` | Behavioral replication reports and derived results | Through indexed reports |
| `outputs/mechanistic/` | Residual/JLens arrays, activation shards, and run-local analyses | No; use links from this index |
| `outputs/causal/` | Intervention outputs and run-local reports | No; use links from this index |
| `compiled_results_*` | Compiled trial-level behavioral/API results | Only for reanalysis |
| root `*_PLAN.md` and `*_RESULTS.md` files | Historical plans and results from earlier phases | Only when tracing research history |

The repository currently contains about 203,000 files. Most are computational
artifacts under `outputs/causal/` and `outputs/mechanistic/`; their presence does
not imply that each is a document a person needs to inspect.

## Documentation rule

Every future completed analysis must:

1. update this file with its status and canonical report link;
2. place any final presentation PNG in the appropriate `figures/` directory;
3. keep large arrays and shards under `outputs/`; and
4. clearly mark superseded prompt versions or analyses as historical.

This index, rather than the directory tree itself, is the research map.
