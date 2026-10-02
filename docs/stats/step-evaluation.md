# STEP: an evaluation

Snyder et al. (RSS 2025) propose STEP, a sequential test for comparing two robot
policies with binary success that stops as early as possible while controlling the false
positive rate. The fieldtrial plan asked for an evaluation before porting anything. This
page records it.

## What STEP does

- **Test.** A one-sided sequential test on the success counts of two policies, run as a
  pair of mirrored tests so either direction can be concluded. The type I error is
  controlled at every point of the null p₀ = p₁.
- **Rejection region.** Built offline per (maximum trials n_max, α, shape of the risk
  budget) by an optimization that approximates the optimal stopping rule, and stored as a
  policy file. Inside a band near the boundary the decision is randomized.
- **Data.** Each step takes one outcome from each policy, but the state is the two running
  success counts, so the two policies are treated as independent streams.
- **Reported result.** Fewer trials than earlier sequential tests (the Lai and SAVI
  baselines); the paper and its versions report savings of roughly a third.

## Why fieldtrial does not port it

1. **License.** The reference implementation
   ([TRI-ML/sequentialized_barnard_tests](https://github.com/TRI-ML/sequentialized_barnard_tests))
   is released under CC BY-NC 4.0, a non-commercial license, and its README notes that
   the code may be covered by patents. fieldtrial is Apache-2.0, so it can neither ship
   that code nor depend on it.
2. **Pairing.** fieldtrial runs arms in randomized blocks, where both arms meet the same
   condition. Testing the paired differences removes the variation between conditions,
   which STEP's independent-streams model does not use.
3. **Pre-computation.** STEP needs a policy file for each n_max and α (shipped for
   n_max = 100, 200 and 500 at α = 0.05; others are synthesized, which takes a while),
   and decisions near the boundary are randomized, which is hard to explain in a report.

## What fieldtrial offers instead

[Anytime-valid comparisons](anytime.md) test the paired block differences with a betting
confidence sequence (Waudby-Smith and Ramdas, 2024): no pre-computation, any α, no cap
on the number of blocks, a deterministic decision, and an anytime-valid p-value and
confidence sequence in the report. [Group-sequential stopping](sequential.md) remains the
choice when the effect size is roughly known.

## Simulation

Paired blocks, two-sided α = 0.05, at most 100 blocks, 400 simulated studies per row
(seed 20261003). "Blocks" is the mean number of blocks run before stopping (100 when the
study ran to the end).

| Control | Treatment | Anytime: power | Anytime: blocks | Group-sequential (5 looks, O'Brien–Fleming): power | blocks | Fixed (McNemar, 100 blocks): power |
|---|---|---|---|---|---|---|
| 0.60 | 0.60 | 0.04 | 98 | 0.06 | 99 | 0.04 |
| 0.60 | 0.75 | 0.30 | 86 | 0.55 | 90 | 0.52 |
| 0.60 | 0.90 | 0.99 | 40 | 1.00 | 58 | 1.00 |
| 0.40 | 0.90 | 1.00 | 20 | 1.00 | 42 | 1.00 |

The first row is the null: all three keep the false positive rate near 5% (the Monte
Carlo standard error is about 1 point). For large differences the anytime test stops
after about half as many blocks as the group-sequential design. For a small difference it
has much less power within 100 blocks, so if the expected effect is around 15 points,
plan a group-sequential or fixed design instead.

We did not run STEP in this comparison, to keep fieldtrial clear of non-commercially
licensed code. To compare it on your own setting, install `sequentialized_barnard_tests`
in a separate environment, respecting its license, and run its mirrored test on the
per-arm outcome sequences that `fieldtrial export` writes.

## References

- Snyder, D. et al. (2025). Is your imitation learning policy better than mine? Policy
  comparison with near-optimal stopping. *Robotics: Science and Systems*.
  arXiv:2503.10966.
- Waudby-Smith, I. and Ramdas, A. (2024). Estimating means of bounded random variables by
  betting. *Journal of the Royal Statistical Society B* 86(1), 1–27.
