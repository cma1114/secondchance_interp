# Is there a separate categorical first-pass-winner effect?

## Question

The all-candidate matching-route lesion initially appeared to affect the first-pass winner (R1) more than could be explained by a linear first-pass score. This audit asks whether that excess is evidence for a separate categorical winner state, or whether it is the nonlinear consequence of how far each candidate lies above or below the other candidates.

## Method

The outcome is the candidate-level **Game minus Neutral** matching-route lesion effect from the completed all-candidate experiment. The audit fits three question-centered models on the frozen 251-question discovery split and applies their feature definitions to the frozen 249-question confirmation split:

1. linear first-pass score plus display positions;
2. a cubic first-pass-score curve plus display positions;
3. flexible spline curves for both the candidate's first-pass score and its gap from the best competing candidate, plus display positions.

Each model then asks whether adding an R1 indicator explains residual variation. A separate near-tie analysis compares R1 with R2 when their first-pass scores are close.

## Results

| Control model | Discovery R1 term | Confirmation R1 term | Confirmation prediction gain from R1 |
|---|---:|---:|---:|
| Linear score | +0.238 [0.081, 0.393] | +0.473 [0.304, 0.645] | +4.1% |
| Cubic score | +0.056 [-0.116, 0.222] | +0.328 [0.157, 0.493] | +0.7% |
| Flexible score + competitor gap | -0.161 [-0.434, 0.098] | +0.183 [-0.115, 0.458] | -0.7% |

The apparent winner increment is strong only when the graded score relation is forced to be linear. Once the model can represent a nonlinear dependence on both absolute first-pass evidence and the gap to the best competitor, the R1 coefficient includes zero on both splits and worsens held-out prediction slightly.

Near-tie R1-minus-R2 contrasts are likewise uncertain: +0.197 [-0.046, 0.452] logits for gaps up to 0.25; +0.133 [-0.069, 0.335] up to 0.5; and +0.160 [-0.008, 0.332] up to 1.0.

## Conclusion

The completed data establish a **graded and nonlinear first-pass-rank effect**, but they do not establish an additional categorical representation meaning “this candidate won.” The earlier linear-regression claim of a distinct winner increment is superseded by this audit. Consequently, a GPU search for a categorical winner bit was not justified.

