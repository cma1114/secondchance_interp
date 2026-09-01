# Second Chance mechanistic-interpretability plan

## Objective

Identify how Qwen changes its answer after being told that its first answer was incorrect. The main working hypothesis is **regenerate, then suppress**:

1. In the redacted Second Chance prompt, Qwen reconstructs approximately the same answer preference it formed during ordinary capabilities testing.
2. Negative feedback recruits a later control process that suppresses the reconstructed first choice.
3. The baseline runner-up either wins automatically after first-choice suppression or receives a separate, selective boost.

The experiment must distinguish targeted first-choice suppression, independent runner-up boosting, generic flattening or reconsideration, and full recomputation under the incorrect prompt.

## Preserve the original paradigm

The primary experiment will reproduce the original redacted design rather than introduce a new multi-turn condition.

- The capabilities test is a separate ordinary-answer run.
- In the Second Chance transcript, the previous assistant response remains `[redacted]`.
- The incorrect condition says that the previous answer was incorrect and asks for a different answer.
- The neutral condition says that the response was lost and asks the model to answer again.
- Qwen is therefore not shown its previous answer letter. The baseline answer and runner-up are labels used externally in the analysis.

The exact checkpoint, chat template, system/setup text, question formatting, and decoding configuration should match the original Qwen runs as closely as possible.

## Preliminary output checks

Before analyzing activations:

1. Render and tokenize the complete capabilities, neutral, and incorrect prompts.
2. Identify the final existing token in the assistant-generation prefix. Its final-layer output, `logits[:, -1, :]`, chooses the first response token. This is the primary answer-prediction position.
3. Verify whether A, B, C, and D are unambiguous single-token continuations at that position.
4. If multiple token variants receive meaningful mass, specify an option-scoring rule before analysis. Do not silently select a convenient tokenization or change the substantive prompt.
5. Score all four options directly so that every trial has complete A-D information; do not rely on top-k storage.
6. Confirm that the generated answer agrees with the highest-scoring canonical A-D option, or document and resolve any discrepancies.

## Phase 1: ordinary baseline answer formation

The first analysis will characterize how answer preferences emerge during the capabilities test, before studying negative feedback.

### Baseline ranks

For each question, use the final capabilities-test A-D logits to define:

- Rank 1: Qwen's ordinary first choice.
- Rank 2: the baseline runner-up.
- Ranks 3 and 4: the remaining options in final-logit order.

These are retrospective, trial-specific labels. Rank 1 is guaranteed to be highest at the final layer, so the baseline layerwise analysis is descriptive rather than causal.

### Native logit lens

At the answer-prediction position, cache the post-layer residual vector `h_l` after every transformer block. For each layer:

1. Apply Qwen's actual final RMSNorm to `h_l`.
2. Apply Qwen's actual language-model output matrix.
3. Read off the A-D scores.

At the final layer these are the actual next-token logits. At intermediate layers they are logit-lens pseudo-logits: a post hoc reading of the current residual through the model's native final output head.

Raw logits will not be the primary quantities because common shifts are behaviorally irrelevant. Within each question and layer, calculate:

\[
\text{first-choice strength}_l
=
z_l(a_1)-\operatorname{mean}[z_l(a_2),z_l(a_3),z_l(a_4)],
\]

\[
\text{runner-up strength}_l
=
z_l(a_2)-\operatorname{mean}[z_l(a_3),z_l(a_4)],
\]

and

\[
\text{top-two gap}_l=z_l(a_1)-z_l(a_2).
\]

Also retain the four A-D logits centered within each trial and layer. Centered rank-2 movement alone will not be interpreted as runner-up boosting, because suppressing rank 1 mechanically raises the centered values of all other options.

### Baseline visualizations

For GPQA and SimpleMC separately, plot:

- Centered pseudo-logit trajectories for final ranks 1-4.
- First-choice strength by layer.
- Runner-up strength by layer.
- The rank-1 versus rank-2 gap by layer.

Show trial-bootstrap uncertainty. Because Qwen has produced heavy-tailed logit changes, report mean trajectories together with median or trimmed-mean sensitivity analyses. Do not pool datasets initially. Also report secondary splits by whether the ordinary baseline answer was actually correct.

The purpose is to determine when the eventual winner and runner-up become natively output-readable and which layer ranges construct their final separation.

## Phase 1b: simple linear probes

The logit lens can miss answer information that is linearly present but not yet aligned with Qwen's final unembedding basis. Use simple probes to test this possibility without initially asking a single probe to recover the complete final logit distribution.

### First-choice probe

At each layer, train a separate four-class linear classifier:

\[
h_l \longrightarrow \{A,B,C,D\},
\]

where the target is Qwen's final capabilities-test choice.

Evaluate on held-out questions:

- Overall and per-letter accuracy.
- Confusion matrices.
- Accuracy by layer relative to 25% chance.
- Generalization from SimpleMC to GPQA and from GPQA to SimpleMC.
- Correct versus incorrect baseline trials.

A transparent low-flexibility version can be built from the training-set mean residual for each eventual answer letter, followed by held-out classification using those class directions. A regularized multinomial classifier can be used as the main version, but it must not be evaluated in sample.

### Runner-up probe

Train a separate four-class classifier at each layer whose target is the final baseline runner-up letter. This asks when runner-up identity becomes linearly decodable without also requiring the probe to recover exact margins, confidence, or the complete A-D distribution.

Compare first-choice decoding, runner-up decoding, and native logit-lens rank recovery by layer.

### Probe safeguards

- The residual dimension will likely exceed the number of questions, so use strong regularization and, if needed, train-only dimensionality reduction.
- Split by question. All variants of a question must remain in the same fold.
- Balance or reweight answer letters and report per-letter performance.
- Do not average raw residuals across answer letters. Decode individual trials first; rank-align and aggregate decoded scores only afterward.
- Use label-permuted versions of questions as a later robustness test to separate semantic-option information from fixed A-D token effects.
- Treat probe success as evidence of linear decodability, not causal use.

### Tuned-lens decision

Do not begin by training a full tuned lens for the 235B checkpoint. A full vocabulary-wide affine translator per layer is substantially more expensive and introduces another learned decoder. Consider a tuned lens or low-rank calibrated lens only if simple probes reliably decode the answer much earlier than the native logit lens, especially when that gap generalizes across datasets and label permutations.

## Phase 2: test regenerate-then-suppress

After establishing ordinary answer formation, apply the identical layerwise readouts to the incorrect and neutral redacted runs. Use the baseline-defined ranks from the separate capabilities run.

Plot the three trajectories directly rather than beginning with a difference curve:

- Ordinary capabilities baseline.
- Incorrect-feedback condition.
- Neutral re-answering control.

The hypothesized trajectory is:

1. The baseline first choice emerges in early or middle layers of the incorrect run similarly to ordinary answering.
2. In later layers, its strength declines specifically under incorrect feedback.
3. The runner-up either remains unchanged relative to ranks 3-4 and wins by default, or receives a separate increase over ranks 3-4.

### Comparison hierarchy

The primary descriptive comparison is **incorrect versus ordinary baseline**, because it measures the total change from ordinary first-choice generation to answering under negative feedback.

The neutral condition is a strategic-specificity control:

- Incorrect minus baseline: total Second Chance effect.
- Neutral minus baseline: generic effect of answering again in the unusual second-turn context.
- Incorrect minus neutral: feedback-specific increment.

Thus neutral remains essential, but incorrect-minus-neutral is not the conceptual definition of suppression.

The primary analysis will include all trials and will not condition on whether the incorrect run changed its final answer. Changed-trial analyses may be shown descriptively but cannot define the mechanism because they select for large logit movements.

## Phase 3: component attribution and causal validation

Layerwise unembedding and probes are localization tools. Mechanistic claims require identifying components and intervening on them.

### Component attribution

At the answer-prediction position, decompose the outputs added by:

- Attention blocks and, where feasible, individual heads.
- MoE blocks, router changes, and weighted expert outputs.

For every trial, project each component's contribution onto that trial's first-choice, runner-up, and top-two contrast directions before averaging. Never average activation vectors across trials with different answer identities and then project.

Look for late components that:

- Contribute negatively to whichever option was baseline rank 1.
- Are present or stronger in the incorrect run than in ordinary baseline and neutral.
- Contribute selectively to rank 2 over ranks 3-4 if an independent runner-up boost exists.

### Activation patching

Use paired same-question runs to test necessity and sufficiency:

1. Patch the neutral residual or component output into the incorrect run and measure recovery of the ordinary first-choice margin.
2. Patch the incorrect activation into neutral and test whether it induces suppression.
3. Narrow successful patches from residual-stream blocks to attention, MoE, head, router, or expert contributions.

Neutral is likely the best patch source because it is structurally closer to the incorrect prompt, even though ordinary baseline remains the primary conceptual and descriptive reference.

Measure interventions using actual final A-D logit contrasts and final choices, not intermediate pseudo-logits alone.

## Interpretation criteria

### Targeted first-choice suppression

Evidence would include:

- The ordinary first choice becomes readable or decodable in the incorrect run before later declining.
- A component contributes negatively to the trial-specific first-choice direction.
- Removing or neutral-patching that component restores the first-choice margin and reduces behavioral lift.
- The targeted letter follows the model's reconstructed answer across questions and label permutations.

### Independent runner-up boosting

Evidence would require more than centered rank-2 movement:

- Rank 2 gains relative to ranks 3-4.
- A distinct component contributes selectively to that contrast.
- Its intervention changes rank 2 without merely undoing rank-1 suppression.

### Generic flattening or reconsideration

Evidence would include broadly reduced logit gaps, increased lower-rank competitiveness, or similar changes under neutral, without a component that conditionally targets the reconstructed winner.

### Full recomputation

If the incorrect trajectory differs from ordinary baseline from the earliest answer-relevant layers and the baseline first choice is never reconstructed, the regenerate-then-suppress hypothesis is disfavored. The prompt may instead cause a different answer computation from the outset.

## Recommended implementation order

1. Reproduce complete A-D logits for capabilities, neutral, and incorrect prompts.
2. Verify the exact answer-prediction position and A-D tokenization.
3. Produce baseline native-logit-lens plots for GPQA and SimpleMC.
4. Train and validate separate first-choice and runner-up linear probes.
5. Compare probe emergence with native output-readability.
6. Overlay baseline, incorrect, and neutral trajectories.
7. Attribute late changes to layers, heads, MoE blocks, routers, and experts.
8. Causally patch candidate components.
9. Add option-label permutations and other counterfactual tests only after the faithful redacted mechanism is localized.
