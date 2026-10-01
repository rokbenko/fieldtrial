# One arm

```python
from fieldtrial.stats import proportion_ci, test_vs_threshold

proportion_ci(36, 40)  # Wilson 95%: 0.900 [0.7695, 0.9604]
proportion_ci(36, 40, method="jeffreys", level=0.9)
test_vs_threshold(39, 40, 0.90, alternative="greater")
```

## Confidence intervals

Notation: k successes in n trials, p̂ = k/n, α = 1 − level, z = Φ⁻¹(1 − α/2).

**Wilson (score), the default.**

```
center      = (p̂ + z²/(2n)) / (1 + z²/n)
half-width  = z / (1 + z²/n) · √( p̂(1 − p̂)/n + z²/(4n²) )
```

Its coverage stays close to the nominal level at the sample sizes typical of robot
evaluations (20–200 trials) and it never leaves [0, 1]. At n = 40 and p̂ = 0.5 the
half-width is 14.8 percentage points: the "±15 pp at 40 rollouts" rule of thumb.

**Clopper–Pearson ("exact").** `[B(α/2; k, n − k + 1), B(1 − α/2; k + 1, n − k)]` with
B the Beta quantile; 0 when k = 0, 1 when k = n. Coverage is always at least nominal, so
it is wider than it needs to be.

**Jeffreys.** The equal-tailed interval of the Beta(k + ½, n − k + ½) posterior. The lower
bound is 0 when k = 0 and the upper bound 1 when k = n (Brown, Cai and DasGupta).

**Agresti–Coull.** With ñ = n + z² and p̃ = (k + z²/2)/ñ: `p̃ ± z √(p̃(1 − p̃)/ñ)`,
clipped to [0, 1].

**When to use which.** Use Wilson unless you have a reason not to. Use Clopper–Pearson
when you must guarantee coverage (for example a safety threshold).

## Threshold test

`test_vs_threshold(k, n, p0)` is the exact binomial test of H₀: p = p₀ (scipy's
`binomtest`). Use `alternative="greater"` for questions like "is the success rate above
0.90?".

## References

- Wilson, E. B. (1927). *JASA* 22, 209–212.
- Clopper, C. J. and Pearson, E. S. (1934). *Biometrika* 26, 404–413.
- Agresti, A. and Coull, B. A. (1998). *The American Statistician* 52, 119–126.
- Brown, L. D., Cai, T. T. and DasGupta, A. (2001). Interval estimation for a binomial
  proportion. *Statistical Science* 16, 101–133.
