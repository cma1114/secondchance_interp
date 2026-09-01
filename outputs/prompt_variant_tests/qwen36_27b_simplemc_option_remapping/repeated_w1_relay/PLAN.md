# Repeated-W1 downstream relay localization

## Question

The repeated option line containing W1 causally reads the matching original
option line, but the final answer query's direct read from the repeated line is
pro-W1 rather than suppressive. Does Game use the repeated W1 line through an
earlier downstream ordinary-attention relay, and where is that relay?

## Frozen design

- Use the canonical action-matched remapped SimpleMC prompts and exact
  historical batch-of-four SDPA cohorts.
- W1 is the semantic original-Baseline answer; conflict and no-conflict trials
  are reported separately.
- Source is the complete second-presentation option line containing W1.
- The already-completed direct final-query edge test is not rerun.
- Block only ordinary-attention edges from the W1 source line to causally later
  queries before the final query. Preserve every other attention edge, GLA
  state, residual, and token.

## Prespecified hierarchy

1. **Prerequisite:** block W1 from every later pre-final query in all 16
   ordinary-attention blocks. The predicted signature is recovery of W1 choice
   and W1-minus-W2 margin in Game, especially on conflict trials.
2. **Query-region decomposition:** separately block reads made by later option
   lines and by tokens after all four options. Together these exactly partition
   the prerequisite query set.
3. **Depth localization:** for the complete all-later query set, separately
   block four disjoint block bands: 4--16, 20--32, 36--48, and 52--64. This
   localizes depth without treating the non-additive region-by-depth
   interaction as identified; a focused interaction test is warranted only if
   both prerequisite and marginal localizations validate.
4. **Source control:** for the post-options region, where all repeated options
   are causally available, compare the W1-line lesion with a token-count-matched
   unselected repeated option line.
5. Report the frozen 251/249 discovery/confirmation halves independently and
   stratify the later-option effect by W1's displayed second-presentation
   letter. W1 displayed as D is an exact zero-query control for that region.

The relay is supported only if the all-later prerequisite has the predicted
Game effect and a region/depth localization points in the same direction on
held-out confirmation. A Neutral-only effect is not evidence for the Game
suppression route.

## Sequential execution rule

Run the prerequisite, two query-region cells, and post-options source control
first. Run the four depth bands only if the all-later Game effect points toward
W1 recovery on both frozen splits (positive W1 choice and/or W1-minus-W2
margin). This prevents spending another full pass localizing a route already
falsified by its joint lesion.

## Completed result

The prerequisite failed in the opposite direction on both frozen splits.
Blocking all later pre-final reads reduced Game W1 choice by 23.4 points in
discovery and 12.5 points in confirmation, and reduced its W1-minus-W2 margin
by 0.57 and 0.62 logits. Neutral depended even more strongly on the route.
Therefore the repeated-W1 line supplies pro-W1 reinstatement evidence; it does
not carry the hypothesized active Game suppression signal. Under the frozen
sequential rule, the depth-band stage was not run.

Canonical result: [analysis/REPORT.md](analysis/REPORT.md).
