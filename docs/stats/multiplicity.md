# Multiplicity

Comparing many arms against a baseline raises the chance that at least one comparison looks
significant by luck. With 8 comparisons at α = 0.05, the chance is up to 34%.

```python
from fieldtrial.stats import adjust_pvalues

adjust_pvalues([0.00893, 0.00302, 0.01624], method="holm")
```

With m p-values sorted p₍₁₎ ≤ … ≤ p₍ₘ₎:

| Method | Adjusted p-value | Controls |
|---|---|---|
| Holm (default) | p̃₍ᵢ₎ = max over j ≤ i of min(1, (m − j + 1) p₍ⱼ₎) | family-wise error rate |
| Bonferroni | p̃ᵢ = min(1, m pᵢ) | family-wise error rate (more conservative) |
| Benjamini–Hochberg | p̃₍ᵢ₎ = min over j ≥ i of min(1, m p₍ⱼ₎ / j) | false discovery rate |

Reject when the adjusted p-value is at most α. Holm is never less powerful than Bonferroni,
so it is the default.

**Example.** In the Dream Machines serving sweep, eight arms were each compared against the
91/120 baseline. Two of them stay significant after Holm: R25-B50 (adjusted p = 0.021) and
R50-B20 (0.018). The `sync` arm (raw p = 0.009) does not (0.054).

## References

- Holm, S. (1979). *Scandinavian Journal of Statistics* 6, 65–70.
- Benjamini, Y. and Hochberg, Y. (1995). *JRSS B* 57, 289–300.
