# Stratified comparison (CMH)

When each arm runs several replicates per condition, the trials are no longer simple pairs.
The Cochran–Mantel–Haenszel test compares arms within each condition (stratum) and combines
the evidence.

```python
from fieldtrial.stats import cmh_test

cmh_test([(k1, n1, k2, n2) for each condition])
```

Write aⱼ for arm 1's successes in stratum j, and Eⱼ, Vⱼ for its hypergeometric mean and
variance given the margins. Then

```
χ²_MH = ( |Σ (aⱼ − Eⱼ)| − c )² / Σ Vⱼ   ~  χ²(1),   c = ½ with continuity correction, else 0
```

The pooled odds ratio is the Mantel–Haenszel estimator Σ aⱼdⱼ/Nⱼ ÷ Σ bⱼcⱼ/Nⱼ, with the
Robins–Breslow–Greenland interval. fieldtrial matches statsmodels' `StratifiedTable`.

## References

- Mantel, N. and Haenszel, W. (1959). *JNCI* 22, 719–748.
- Robins, J., Breslow, N. and Greenland, S. (1986). *Biometrics* 42, 311–323.
