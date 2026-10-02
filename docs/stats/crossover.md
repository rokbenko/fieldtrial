# Crossover rounds

Some tasks change the scene as they go: filling a tray, clearing a table, stacking parts.
Then trials within a pass are not independent, and alternating arms trial by trial would
leave each arm with the other's mess. In a `crossover_rounds` design, an arm runs a whole
round (one pass over the conditions), and the **round** is the unit of analysis.

Rounds come in cycles of two periods. In each cycle one arm runs first and the other
second. The order (AB or BA) is randomized so that each order occurs equally often. This
balances a period effect, such as the scene wearing down or the operator tiring, between
the arms.

```python
from fieldtrial.stats import crossover_test

periods = [(0.8, 0.6), (0.7, 0.65), (0.55, 0.85), (0.6, 0.8)]  # success per round
orders = ["AB", "AB", "BA", "BA"]
crossover_test(periods, orders)
```

## Estimate

With period differences u_c = y_c1 − y_c2 per cycle (Hills and Armitage, 1979):

- arm difference τ̂ = ½(ū_AB − ū_BA), adjusted for the period effect
- period effect π̂ = −½(ū_AB + ū_BA)
- standard error se(τ̂) = ½ s_u √(1/m_AB + 1/m_BA), where s_u is the pooled within-order
  standard deviation on m − 2 degrees of freedom

## Test

Under no arm difference, u_c does not depend on which order cycle c was given. So every
assignment of the m_AB "AB" labels to the m cycles was equally likely. fieldtrial
enumerates all of them and compares τ̂ with its exact randomization distribution. A
one-sided p-value is the share at least as extreme; the two-sided p-value doubles the
smaller one-sided p-value. Above 200,000 assignments it uses the Hills–Armitage t-test
instead.

The confidence interval inverts the same test, so it excludes 0 exactly when the test
rejects. If the true difference is δ, the shifted period differences u_c − δ s_c (s_c = +1
for AB, −1 for BA) no longer depend on the order. So δ belongs to the interval when the
randomization test applied to the shifted differences does not reject. The ends are found
by bisection. With very few cycles the interval can span every possible difference: with 2
cycles, no result is ever significant.

Few cycles give little power: with 4 cycles the smallest possible two-sided p-value is
1/3. Plan at least 6–8 cycles, or more conditions per round so each round is measured
more precisely. In v0.2, crossover designs compare exactly 2 arms.

## References

- Hills, M. and Armitage, P. (1979). The two-period cross-over clinical trial. *British
  Journal of Clinical Pharmacology* 8, 7–20.
- Senn, S. (2002). *Cross-over Trials in Clinical Research*, 2nd ed., chapter 3. Wiley.
- Edgington, E. S. and Onghena, P. (2007). *Randomization Tests*, 4th ed. Chapman &
  Hall/CRC.
