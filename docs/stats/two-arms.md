# Two independent arms

Use this when the two arms were evaluated on separate trials (for example 80 rollouts of a
new checkpoint and 120 of the baseline).

```python
from fieldtrial.stats import compare_independent

res = compare_independent(74, 80, 91, 120)  # arm 1 = new, arm 2 = baseline
res.difference  # +0.1667
res.interval  # Newcombe 95%: [+0.0625, +0.2596]
res.primary.pvalue  # Boschloo: 0.0020
res.secondary[0].pvalue  # Fisher: 0.0022
```

## Difference and its interval: Newcombe hybrid score

d = p̂₁ − p̂₂. With Wilson intervals [l₁, u₁] and [l₂, u₂] for each arm:

```
lower = d − √( (p̂₁ − l₁)² + (u₂ − p̂₂)² )
upper = d + √( (u₁ − p̂₁)² + (p̂₂ − l₂)² )
```

This is method 10 of Newcombe (1998). It behaves well at small samples and near 0 or 1,
where the simple Wald interval fails.

## Primary test: Boschloo's exact test

Arm sizes are fixed by the design, and nothing else is. Boschloo's unconditional test
uses Fisher's one-sided p-value as its statistic and takes the largest tail probability
over the unknown common success rate π under H₀. It is uniformly more powerful than
Fisher's exact test.

The two-sided p-value is twice the smaller one-sided p-value, capped at 1, as in
`scipy.stats.boschloo_exact`.

!!! note "Table orientation"
    `scipy.stats.boschloo_exact` treats each **column** of the 2×2 table as one arm, so
    fieldtrial passes `[[k₁, k₂], [n₁ − k₁, n₂ − k₂]]`. Transposing the table changes
    Boschloo's p-value (for 50/80 vs 13/40 it gives 0.00201 instead of the correct
    0.00228). It does not change Fisher's.

## Secondary: Fisher's exact test and the odds ratio

Fisher's test conditions on both margins. It is reported for comparability with papers.
The conditional maximum-likelihood odds ratio is reported with its exact confidence
interval.

## Choosing the test in advance

Choose the primary test before seeing the data (`test="boschloo"` by default). Switching
to whichever test gives the smaller p-value inflates the false-positive rate.

## References

- Newcombe, R. G. (1998). Interval estimation for the difference between independent
  proportions. *Statistics in Medicine* 17, 873–890.
- Boschloo, R. D. (1970). *Statistica Neerlandica* 24, 1–9.
- Fisher, R. A. (1935). The logic of inductive inference. *JRSS* 98, 39–82.
