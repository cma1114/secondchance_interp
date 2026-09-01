# Semantic-W1 binding: whole-attention versus whole-MLP factorial

## Question

Where does the model combine the incorrectness evaluation with the semantic
identity of its first-pass answer?  The experiment distinguishes two broad
implementations without screening individual attention heads:

1. complete outputs of the model's ordinary-attention modules;
2. complete outputs of its MLPs;
3. both classes together.

## Semantic manipulation

For every included question, two first-presentation option mappings are used.
The model's standalone answer is literal `A` under both mappings, but `A`
refers to different option content.  Call those contents X and Y.  A third
mapping is fixed for the repeated question, so the entire second presentation
is identical across the X and Y histories.

The four natural cells are:

1. first answer X, `Your answer was incorrect.`
2. first answer X, `Your answer was lost.`
3. first answer Y, `Your answer was incorrect.`
4. first answer Y, `Your answer was lost.`

Every cell continues with the identical sentence `Choose the answer again.`
and the identical repeated question.  Thus the factorial interaction asks
whether incorrectness has a different effect depending on the semantic answer
the model reached in the first presentation.  The literal first answer is held
at `A`.

Eligibility is determined only from existing standalone Baseline runs.  The
frozen established split yields 64 discovery and 73 confirmation questions.

## Component-level intervention

At each component and token position, collect its complete output vector in all
four natural cells.  For the Evaluation/X cell, its no-interaction output is

`Neutral/X + Evaluation/Y - Neutral/Y`.

The corresponding expression is used for Evaluation/Y.  Replacing the natural
component output with this value removes only the measured
evaluation-by-semantic-history interaction while retaining both main effects.

Run this intervention jointly over:

- all 16 ordinary-attention modules;
- all 64 MLPs;
- both sets together.

No attention-head selection is performed.

Test three positions separately:

1. the period closing `Your answer was incorrect.` / `lost.`;
2. the closing newline of the repeated option containing the relevant prior
   answer content (X or Y);
3. the final answer-decision position.

All nine position-by-module-class interventions are frozen in advance and are
run on both splits.  A same-batch untouched row controls Qwen's batch-dependent
numerical drift.

## Outcomes

Primary natural endpoint: the symmetric semantic-targeting contrast, averaging
(a) extra suppression of X when X rather than Y was the first answer and (b)
extra suppression of Y when Y rather than X was the first answer.

Primary causal endpoints:

- recovery of the relevant prior-answer centered logit;
- recovery of the relevant-prior-answer versus alternative-prior-answer margin;
- change in selection of the relevant prior answer.

Secondary endpoints are the full centered A-D redistribution and A-D entropy.
Report discovery and confirmation separately, plus a pooled descriptive
estimate.  Confidence intervals use paired question bootstrap resampling.

## Interpretation

- Attention-only mediation supports semantic matching in ordinary attention.
- MLP-only mediation supports a nonlinear conjunction computed from information
  already present in the residual stream.
- Mediation only when both are replaced supports attention supplying a weak
  match that MLPs amplify or gate.
- A null for all three means the binding is distributed through another route
  or is not represented by this factorial component-output interaction.

