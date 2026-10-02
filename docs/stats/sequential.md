# Group-sequential stopping

A fixed design runs every planned trial and tests once. A group-sequential design looks at
the data at a few planned points and stops early when the difference is already clear,
which can save many rollouts. Looking repeatedly with an ordinary test inflates the false
positive rate; with 5 unadjusted looks at α = 0.05 it is about 14%. Group-sequential
boundaries raise the bar at each look so the overall error stays at α.

```python
from fieldtrial.stats import sequential_test, spending_boundaries

design = spending_boundaries([0.25, 0.5, 0.75, 1.0], alpha=0.05)   # O'Brien–Fleming type
design.boundaries          # (4.33, 2.96, 2.36, 2.01)
sequential_test([1.2, 3.1], design)   # stops at look 2
```

## Error spending

Lan and DeMets' spending function α(t) says how much of the error may be used up by
information fraction t (for fieldtrial, the share of planned trials or blocks completed).
Look k spends α(t_k) − α(t_{k−1}), and its boundary b_k solves

P₀(no crossing before look k, crossing at look k) = α(t_k) − α(t_{k−1}).

| `spending` | α(t), per tail | Behaviour |
|---|---|---|
| `obrien_fleming` (default) | 2 − 2Φ(Φ⁻¹(1 − α/2)/√t) | Very strict early, close to the fixed design at the end |
| `pocock` | α ln(1 + (e − 1)t) | Similar boundaries at every look; easier to stop early, costs more at the end |

A two-sided design spends α/2 in each tail. Because only the information fractions enter,
the looks need not be equally spaced, and a look may be moved without breaking the error
guarantee as long as the decision to look does not depend on the data. If a study ends
with fewer trials than planned, `spend_all=True` spends the remaining error at the last
look.

The classical constant boundaries of Pocock (1977) and O'Brien and Fleming (1979) are
available as `constant_boundaries(K, shape=...)`.

## Computing the crossing probabilities

The z statistics at the looks behave like a Brownian motion observed at times t_k: the score
S_k = Z_k √t_k has independent normal increments with variance t_k − t_{k−1}. The density of
Z_k on the continuation region is propagated look by look by numerical integration
(Armitage, McPherson and Rowe, 1969) with Simpson's rule on a fine grid. The boundaries
reproduce the published tables to 4 decimals; tests also check the crossing probabilities
against scipy's multivariate normal CDF.

## P-values and intervals after stopping

An ordinary p-value is not valid after a sequential test. fieldtrial reports the stage-wise
ordering p-value: stopping earlier counts as more extreme, and within a look a larger |Z|
counts as more extreme. Stopping at look k* with statistic z* gives

p = Σ_{j<k*} P₀(cross at j) + P₀(no crossing before k*, |Z_{k*}| ≥ |z*|).

It is at most α exactly when the design rejects. Repeated confidence intervals
θ̂_k ± b_k·se_k (`repeated_interval`) cover the true difference at every look
simultaneously, so they stay valid whenever the study stops.

## In a study

```yaml
analysis:
  stopping: {rule: group_sequential, looks: 4, spending: obrien_fleming}
```

`fieldtrial interim` runs a planned look while the study stays blinded: it reports only
"continue" or "stop", never per-arm results. Each look is logged, and an unplanned look is
listed as a deviation in the report.

## References

- Armitage, P., McPherson, C. K. and Rowe, B. C. (1969). Repeated significance tests on
  accumulating data. *JRSS A* 132, 235–244.
- Pocock, S. J. (1977). Group sequential methods in the design and analysis of clinical
  trials. *Biometrika* 64, 191–199.
- O'Brien, P. C. and Fleming, T. R. (1979). A multiple testing procedure for clinical
  trials. *Biometrics* 35, 549–556.
- Lan, K. K. G. and DeMets, D. L. (1983). Discrete sequential boundaries for clinical
  trials. *Biometrika* 70, 659–663.
- Fairbanks, K. and Madsen, R. (1982). P values for tests using a repeated significance
  test design. *Biometrika* 69, 69–74.
- Jennison, C. and Turnbull, B. W. (1989). Interim analyses: the repeated confidence
  interval approach. *JRSS B* 51, 305–361.
- Jennison, C. and Turnbull, B. W. (2000). *Group Sequential Methods with Applications to
  Clinical Trials*. Chapman & Hall/CRC. Tables 2.1 and 2.3.
- Reboussin, D. M., DeMets, D. L., Kim, K. and Lan, K. K. G. (2000). Computations for group
  sequential boundaries using the Lan–DeMets spending function method. *Controlled Clinical
  Trials* 21, 190–207.
