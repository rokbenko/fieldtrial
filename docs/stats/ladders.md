# Checkpoint ladders

A ladder evaluates several checkpoints of one training run, in order of training step.
Two questions are fixed before the data are seen.

```python
from fieldtrial.stats import plateau, trend_test

counts = [(30, 60), (44, 60), (51, 60), (53, 60)]  # (successes, trials) per step
trend_test(counts, scores=[10_000, 20_000, 40_000, 80_000])
plateau(counts, margin=0.10)  # where it levels off
```

## Trend

**Independent arms:** the Cochran–Armitage test. With step scores x_i, p̄ = Σk_i/N and
x̄ = Σn_i x_i/N,

Z = Σ x_i (k_i − n_i p̄) / √( p̄(1 − p̄) Σ n_i (x_i − x̄)² ) ~ N(0, 1).

**Blocks** (every checkpoint runs once per condition): Mantel's stratified extension. Each
block is a stratum; with N_s trials and K_s successes in stratum s,

T_s = Σ x_i (k_is − n_is K_s/N_s),
V_s = K_s(N_s − K_s) / (N_s(N_s − 1)) · (Σ n_is x_i² − (Σ n_is x_i)²/N_s),
Z = ΣT_s / √(ΣV_s).

V_s is the exact permutation variance, so blocks where every checkpoint agrees carry no
weight. The test asks about a *linear* trend in the scores. Use the training step (or its
logarithm) as the score, and decide which before the data come in. A rejected trend test
says that success changes with the step. It does not say that every later checkpoint is
higher.

## Plateau

From which checkpoint on is every checkpoint within a margin δ of the final one? For each
earlier checkpoint j, fieldtrial tests

H₀ⱼ: p_final − p_j ≥ δ against H₁ⱼ: p_final − p_j < δ,

rejecting when the upper bound of the two-sided 1 − 2α interval for p_final − p_j is below
δ: Newcombe's interval for independent arms, and Tango's for paired blocks.
The hypotheses are tested in a fixed order, from the checkpoint just before the final one
back to the first, each at level α, stopping at the first one that is not rejected. A
pre-specified testing order controls the family-wise error at α with no further
adjustment. The plateau is the earliest checkpoint in the run of rejections.

The margin δ must be fixed before the data are seen, in `study.yaml`. Reports say
"from step N on, every checkpoint was within δ of the final checkpoint". They never name a
"best" checkpoint.

## References

- Armitage, P. (1955). Tests for linear trends in proportions and frequencies.
  *Biometrics* 11, 375–386.
- Mantel, N. (1963). Chi-square tests with one degree of freedom; extensions of the
  Mantel–Haenszel procedure. *JASA* 58, 690–700.
- Agresti, A. (2013). *Categorical Data Analysis*, 3rd ed., section 3.4.6 (the worked
  example used as a golden test: M² = 6.57).
- Maurer, W., Hothorn, L. A. and Lehmacher, W. (1995). Multiple comparisons in drug
  clinical trials and preclinical assays: a-priori ordered hypotheses.
- Tango, T. (1998). Equivalence test and confidence interval for the difference in
  proportions for the paired-sample design. *Statistics in Medicine* 17, 891–908.
- Newcombe, R. G. (1998). *Statistics in Medicine* 17, 873–890.
