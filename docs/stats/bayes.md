# Bayesian summaries

!!! warning "Descriptive only"
    These numbers help you communicate uncertainty. They are never the primary test of a
    study, and a high probability is not evidence that a pre-registered comparison
    succeeded.

```python
from fieldtrial.stats import prob_superiority, credible_interval

prob_superiority(91, 120, 74, 80)  # P(p_B > p_A) = 0.99902
credible_interval(36, 40)  # Beta(37, 5) equal-tailed 95%
```

With independent Beta(a₀, b₀) priors (uniform by default), each arm's posterior is
Beta(a₀ + k, b₀ + n − k), and

```
P(p_B > p_A) = ∫₀¹ f_B(x) F_A(x) dx
```

This is computed by adaptive quadrature and checked against Evan Miller's exact closed form.

## References

- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed., chapter 2. CRC Press.
