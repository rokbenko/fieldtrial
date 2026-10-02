# Proxy-assisted estimates (PPI++)

Labeling every episode by hand is slow. A reward model can score every episode in seconds,
but its scores are not labels. Prediction-powered inference uses both: many proxy scores
and a smaller random sample of human labels. It gives a valid confidence interval for the
true success rate, narrower than the human labels alone when the proxy is good, and no
wider (asymptotically) when it is not.

```python
from fieldtrial.stats import ppi_mean

res = ppi_mean(y, f, f_unlabeled)  # labels and scores of 60 reviewed episodes; 300 more scores
res.interval  # 95% CI 0.41–0.61
res.classical  # 0.41–0.66 from the 60 labels alone
res.lam  # 0.72: how much the proxy is trusted
```

## The estimator

Humans label n episodes at random (labels Y_i, proxy scores f(X_i)); N further episodes have
proxy scores f(X̃_j) only. For a weight λ in [0, 1],

θ̂_λ = λ · mean(f(X̃)) + mean(Y − λ f(X)).

The first term is the proxy's estimate over many episodes; the second corrects its bias on
the episodes humans labeled. The standard error is

se = √( Var(λ f(X̃)) / N + Var(Y − λ f(X)) / n ),

and the interval is θ̂_λ ± z · se. λ = 0 is the classical estimate from the labels alone;
λ = 1 is the original PPI of Angelopoulos et al. (2023). PPI++ chooses the λ that minimizes
the variance,

λ̂ = Cov(Y, f(X)) / ((1 + n/N) · Var(f)),

clipped to [0, 1]: a proxy that tracks the labels gets a weight near 1; a useless one gets
a weight near 0, which falls back to the labels alone. The implementation follows
`ppi_py.ppi_mean_ci` (ppi-python 0.2.3) and matches it to machine precision.

For two arms with separate episodes, `ppi_difference` combines the two estimates:
θ̂_A − θ̂_B with standard error √(se_A² + se_B²).

## When it is valid

- **The human-labeled episodes must be a random sample of the scored ones.** fieldtrial
  draws the review sample with the study seed and records it before anyone reviews, so it
  cannot be chosen after seeing scores.
- The proxy may be biased or badly calibrated; the correction term takes care of that. Its
  quality only affects the width.
- The interval is a large-sample (normal) interval. With fewer than about 30 human labels
  per arm, treat it as approximate.

In fieldtrial, proxy-assisted estimates are a pre-registered **secondary** analysis. The
primary analysis always uses the human labels of the scheduled trials.

## Related: SureSim

SureSim (Badithela et al., 2025, arXiv:2510.04354) applies the same idea to simulation:
paired real and simulated rollouts are the labeled set, extra simulated rollouts the
unlabeled set. `fieldtrial import-proxy` accepts such scores, so the same estimate can be
computed for simulation results.

## References

- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I. and Zrnic, T. (2023).
  Prediction-powered inference. *Science* 382(6671), 669–674.
- Angelopoulos, A. N., Duchi, J. C. and Zrnic, T. (2023). PPI++: Efficient
  prediction-powered inference. arXiv:2311.01453.
