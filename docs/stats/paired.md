# Paired designs

Use these when every arm ran on the same conditions (complete blocks), for example both
checkpoints tried from each of 40 start positions. Pairing removes the condition-to-condition
variation, so it usually needs fewer trials than an independent comparison.

## Two arms: exact McNemar and Tango's interval

Only the discordant blocks carry information: **b** blocks where only arm 1 succeeded,
and **c** blocks where only arm 2 succeeded.

```python
from fieldtrial.stats import compare_paired

res = compare_paired(b=10, c=2, n=40)
res.test.pvalue  # exact McNemar: 0.0386
res.difference  # (b − c)/n = +0.20
res.interval  # Tango 95%: [+0.035, +0.366]
```

**Exact McNemar test.** Under H₀ each discordant block favors either arm with probability
½, so b ~ Binomial(b + c, ½). The two-sided p-value is `min(1, 2·min(P(B ≤ b), P(B ≥ b)))`.

**Tango's score interval** for Δ = p₁ − p₂. The score statistic for H₀: Δ = δ is

```
T(δ) = (b − c − nδ) / √( n (2 q̃ + δ(1 − δ)) )
```

where q̃ is the constrained maximum-likelihood estimate of the probability that only
arm 2 succeeds. It is the positive root of `2n q² + (−b − c + (2n − b + c)δ) q − cδ(1 − δ) = 0`.
The interval is the set of δ with |T(δ)| ≤ z. With no discordant pairs it is
±z²/(n + z²).

fieldtrial checks its implementation against an independent version that finds q̃
numerically.

## More than two arms

`cochran_q(outcomes)` takes a 0/1 matrix with one row per block and one column per arm.

```
Q = (k − 1)(k Σⱼ Cⱼ² − N²) / (k N − Σᵢ Rᵢ²)   ~  χ²(k − 1)
```

Here Cⱼ are the column totals, Rᵢ the row totals and N the grand total. Pairwise exact
McNemar tests follow, either all pairs or each arm against the first column, adjusted with
Holm by default.

## Several replicates per condition

With several replicates per condition, the study's primary analysis is the
[Cochran–Mantel–Haenszel test](stratified.md), stratified by condition.

## References

- McNemar, Q. (1947). *Psychometrika* 12, 153–157.
- Tango, T. (1998). Equivalence test and confidence interval for the difference in
  proportions for the paired-sample design. *Statistics in Medicine* 17, 891–908.
- Fagerland, M. W., Lydersen, S. and Laake, P. (2014). Recommended tests and confidence
  intervals for paired binomial proportions. *Statistics in Medicine* 33, 2850–2875.
- Cochran, W. G. (1950). *Biometrika* 37, 256–266.
