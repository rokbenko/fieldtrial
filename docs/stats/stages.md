# Progress stages

A rubric orders a task into stages, for example *lift → handover → inserted → clean*. Each
trial records the furthest stage it reached (−1 if none), so two arms with the same success
rate can still differ in *where* they fail.

```python
from fieldtrial.stats import stage_distribution, stage_funnel, compare_stages

stage_funnel([-1, 0, 0, 1, 1, 1, 2, 3, 3, 3], n_stages=4)
compare_stages(stages_new, stages_baseline)  # Brunner–Munzel, secondary
```

## Funnel

For stage s, Rₛ = #{trials with stage ≥ s}. The conversion into stage s is Rₛ / Rₛ₋₁, with
R₋₁ = n, and it gets a Wilson interval. The last step's cumulative share is the success
rate.

## Comparing arms: Brunner–Munzel

Stage indices are ordinal, with many ties, and arms can differ in spread. The
Brunner–Munzel test handles both. It tests

```
H₀: P(X_A < X_B) + ½ P(X_A = X_B) = ½
```

with a t approximation (scipy's `brunnermunzel`). `alternative="greater"` means arm A tends to
get further. This is a secondary analysis; the primary test is still on success.

## References

- Brunner, E. and Munzel, U. (2000). The nonparametric Behrens–Fisher problem. *Biometrical
  Journal* 42, 17–25.
