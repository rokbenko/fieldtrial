# Agreement with a reward model

Before a reward model's suggested labels are trusted for anything, measure how often it
agrees with people. Raw agreement overstates it: if 80% of trials succeed, a model that
always says "success" agrees 80% of the time. Cohen's κ corrects for the agreement expected
by chance.

```python
from fieldtrial.stats import cohens_kappa, kappa_table

table = kappa_table(human_labels, model_labels)  # rows: human, columns: model
res = cohens_kappa(table)
res.kappa  # 0.72 for [[30, 5], [8, 57]]
res.interval  # 95% CI 0.58–0.86
```

## Formula

With cell proportions p_ij, row margins p_i· (rater A) and column margins p_·j (rater B):

- observed agreement p_o = Σ p_ii
- chance agreement p_e = Σ p_i· p_·i
- κ = (p_o − p_e) / (1 − p_e)

κ = 1 is perfect agreement, κ = 0 is chance level. The interval uses the large-sample
standard error of Fleiss, Cohen and Everitt (1969),

Var(κ̂) = (A + B − C) / (n (1 − p_e)²), where

- A = Σ p_ii [1 − (p_i· + p_·i)(1 − κ)]²
- B = (1 − κ)² Σ_{i≠j} p_ij (p_·i + p_j·)²
- C = [κ − p_e (1 − κ)]²

and the test of κ = 0 uses the variance under that hypothesis,
[p_e + p_e² − Σ p_i· p_·i (p_i· + p_·i)] / (n (1 − p_e)²). The results match statsmodels'
`cohens_kappa` to machine precision.

## Honest agreement needs blind labels

If the reviewer sees the model's suggestion before labeling, they anchor on it and κ goes
up for the wrong reason. fieldtrial's review page asks for the human label first and only
then shows the suggestion; κ in the report is computed from those first, blind labels.
Changes made after seeing the suggestion are kept and logged, but do not count towards κ.

As a rough guide (Landis and Koch, 1977), κ above 0.8 is often called almost perfect and
0.6–0.8 substantial. For evaluation, the question is narrower: is the model good enough to
help, which is what [proxy-assisted estimates](ppi.md) quantify.

## References

- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and
  Psychological Measurement* 20, 37–46.
- Fleiss, J. L., Cohen, J. and Everitt, B. S. (1969). Large sample standard errors of kappa
  and weighted kappa. *Psychological Bulletin* 72(5), 323–327.
- Landis, J. R. and Koch, G. G. (1977). The measurement of observer agreement for
  categorical data. *Biometrics* 33(1), 159–174.
