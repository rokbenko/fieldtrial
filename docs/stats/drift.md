# Drift and validity checks

Real-world evaluations drift: grippers wear, lighting changes, operators change. Dream
Machines had to rerun every evaluation after a gripper-plasticity problem. fieldtrial checks
whether results changed across sessions and operators, and whether one arm had more invalid
trials than another.

```python
from fieldtrial.stats import homogeneity_test

homogeneity_test([(18, 20), (12, 20), (17, 20)])  # (successes, trials) per session
```

- **Two groups:** Fisher's exact test.
- **More groups:** Pearson's chi-square test of homogeneity on the g × 2 table,
  Σ (O − E)² / E ~ χ²(g − 1). `small_expected` warns when an expected count is below 5,
  where the p-value is only approximate.

A flag is a reason to look, not proof of a problem. Check the per-session numbers in the
report.

## References

- Agresti, A. (2013). *Categorical Data Analysis*, 3rd ed., sections 3.2 and 3.5.
