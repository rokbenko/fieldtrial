# Planning: sample size, power and MDE

```python
from fieldtrial.stats import sample_size, power, mde, boschloo_power, mcnemar_power

sample_size(0.76, 0.90).n1  # 112 per arm
mde(0.76, 40).effect  # +0.211: 40 per arm detects 76% → 97%
mde(0.76, 40, 120)  # unequal arms: 40 new vs 120 baseline
boschloo_power(0.76, 0.90, 112)  # exact power of the test you will run
```

## Sample size

Notation: Δ = p₁ − p₂, r = n₂/n₁ (the allocation ratio), p̄ = (p₁ + r p₂)/(1 + r), q = 1 − p,
z_α = Φ⁻¹(1 − α/2) for two-sided tests, and z_β = Φ⁻¹(power).

**pooled-z (default).**

```
n₁ = ( z_α √(p̄ q̄ (1 + 1/r)) + z_β √(p₁q₁ + p₂q₂/r) )² / Δ²
```

**fleiss-cc** adds Fleiss' continuity correction:

```
n₁' = n₁/4 · (1 + √(1 + 2(r + 1)/(n₁ r |Δ|)))²
```

**arcsine** uses Cohen's effect size h = 2 asin √p₂ − 2 asin √p₁:

```
n₁ = (z_α + z_β)² (1 + 1/r) / h²
```

n₂ = r·n₁, and both are rounded up. As usual, the far tail of a two-sided test is ignored.

| p₁ → p₂ | pooled-z | Fleiss | arcsine |
|---|---|---|---|
| 0.76 → 0.90 | 112 | 126 | 109 |
| 0.76 → 0.93 | 70 | 82 | 66 |
| 0.30 → 0.60 | 42 | 49 | 42 |

## Minimum detectable effect

`mde(p_baseline, n1, n2)` solves the sample-size formula for the second rate. It gives
the smallest true change the study detects with the requested power. Report it with every
non-significant result: it says what the study could and could not resolve.

| Baseline, n per arm | Detectable increase | Detectable decrease |
|---|---|---|
| 0.76, 40 | +21.1 pp | −30.1 pp |
| 0.76, 120 | +13.6 pp | −16.8 pp |
| 0.50, 40 | +29.5 pp | −29.5 pp |

## Exact power

The closed forms are normal approximations. fieldtrial also computes the exact power of
the tests you will actually run, by enumerating every possible outcome, so there is no
simulation error:

- **`boschloo_power`.** Boschloo's test rejects a set of 2×2 outcomes. That set is
  computed for all (n₁ + 1)(n₂ + 1) outcomes at once, with the supremum over the
  nuisance rate taken on a fixed grid of 4001 points (within about 1e-7 of scipy).
  Power is the binomial probability of the set.
- **`mcnemar_power(p10, p01, n)`.** The number of discordant pairs is
  D ~ Binomial(n, p10 + p01), and given D, b ~ Binomial(D, p10/(p10 + p01)). Power sums
  exactly over both.
- **`simulate_power`** is a seeded Monte Carlo cross-check that calls scipy's tests
  directly.

## References

- Fleiss, J. L., Levin, B. and Paik, M. C. (2003). *Statistical Methods for Rates and
  Proportions*, 3rd ed., chapter 4.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, chapter 6.
- Chow, S.-C., Shao, J. and Wang, H. (2008). *Sample Size Calculations in Clinical
  Research*, 2nd ed., chapter 4.
