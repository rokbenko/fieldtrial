# Anytime-valid comparisons

[Group-sequential stopping](sequential.md) looks at the data at a few planned points. An
anytime-valid test lets you look after every block instead, and stop at whichever block
the difference becomes clear, with the false positive rate still at most α. The price is
a wider interval than a fixed design would give at the same sample size; the gain is that
a large difference is found after a handful of blocks, and nothing has to be planned
except α.

```python
from fieldtrial.stats import betting_cs, paired_anytime_test

betting_cs([1, 1, 0, 1, 1, 1, 0, 1, 1, 1] * 3, alpha=0.05).interval()
# Interval(low=0.558, high=0.959, level=0.95, method='betting')

res = paired_anytime_test([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0] * 3, alpha=0.05)
res.rejected_at  # 15: the difference was clear after 15 blocks
```

## Confidence sequences by betting

A confidence sequence (L_t, U_t) covers the mean μ of observations in [0, 1] at every
sample size at once:

P(L_t ≤ μ ≤ U_t for all t) ≥ 1 − α.

For each candidate mean m, imagine betting against "the mean is m". After each
observation the capital is multiplied by 1 + λ_t (x_t − m) (betting that the mean is
higher) or 1 − λ_t (x_t − m) (lower):

K⁺_t(m) = ∏ (1 + λ⁺_i (x_i − m)),  K⁻_t(m) = ∏ (1 − λ⁻_i (x_i − m)).

The bet λ_t may depend only on the observations before t. If the mean really is m, both
capitals are fair games (nonnegative martingales), and Ville's inequality says the hedged
capital K_t = max(θ K⁺_t, (1 − θ) K⁻_t) ever reaches 1/α with probability at most α.
The confidence sequence at time t is every m whose capital is still at most 1/α.

fieldtrial uses the hedged capital process of Waudby-Smith and Ramdas (2024) with their
predictable-mixture empirical-Bernstein bets

λ_t = √( 2 log(1/α′) / (σ̂²_{t−1} · t · log(1 + t)) ),

where σ̂²_{t−1} is the running variance (regularized towards 1/4 with one pseudo
observation), α′ = θα for K⁺ and (1 − θ)α for K⁻, and the bets are capped at 1/(2m) and
1/(2(1 − m)). Two-sided sequences use θ = 1/2; `alternative="greater"` (θ = 1) gives a
lower bound only. The set is found on a grid of 1001 candidate means, padded outwards by
one grid step and intersected over time, exactly as in `confseq.betting.hedged_cs`. The
golden tests reproduce confseq's output to machine precision.

## The paired anytime test

In a randomized block each block runs both arms once, so the block gives a paired
difference d = y_treatment − y_control in [−1, 1]. `paired_anytime_test` maps it to
(d + 1)/2 in [0, 1] and tests the mean difference Δ:

| `alternative` | Null hypothesis | Rejects when |
|---|---|---|
| `two-sided` | Δ = null | the capital against `null` exceeds 1/α |
| `greater` | Δ ≤ null | the upward capital exceeds 1/α |
| `less` | Δ ≥ null | the downward capital exceeds 1/α |

A negative `null` with `greater` is a non-inferiority test (for example `null=-0.1`: the
treatment is at most 10 points worse). The anytime-valid p-value is
min(1, 1 / max_{s ≤ t} K_s), valid however the stopping time was chosen; the test rejects
exactly when it falls below α. The confidence sequence for Δ comes with the result.

Pairing removes the variation between blocks (an easy position helps both arms), which is
why fieldtrial tests block differences rather than the two arms separately.

## Compared with group-sequential designs

| | Group-sequential | Anytime-valid |
|---|---|---|
| Looks | A few, planned (2 to 10) | After every block |
| Statistic | z test with error-spending boundaries | Betting capital, no normal approximation |
| Planned sample size | Needed | Optional (a cap) |
| Final-look power | Close to a fixed design | Lower; needs more blocks when the effect is small |
| Best when | The effect size is roughly known | Effects may be large, and stopping early matters most |

See [the STEP evaluation](step-evaluation.md) for how these compare with STEP, a
near-optimal sequential test for two policies.

## References

- Waudby-Smith, I. and Ramdas, A. (2024). Estimating means of bounded random variables by
  betting. *Journal of the Royal Statistical Society B* 86(1), 1–27.
- Howard, S. R., Ramdas, A., McAuliffe, J. and Sekhon, J. (2021). Time-uniform,
  nonparametric, nonasymptotic confidence sequences. *Annals of Statistics* 49(2),
  1055–1080.
- Ville, J. (1939). *Étude critique de la notion de collectif*. Gauthier-Villars.
