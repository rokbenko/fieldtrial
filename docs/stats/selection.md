# Best-arm selection

A serving-config sweep compares several configurations of one checkpoint (for example
RTC queue thresholds) and asks which is best. Testing every pair at the end wastes trials
on configurations that are clearly worse early on. Successive elimination drops them as
soon as the data allow, and keeps running only the ones still in contention.

```python
from fieldtrial.stats import eliminate

res = eliminate(blocks, ["default", "q50", "q70", "sync"], delta=0.05)
res.survivors  # ('q70', 'sync'): could not be told apart yet
res.eliminations  # default dropped after block 28, q50 after block 39 (both beaten by sync)
res.best  # None until exactly one arm survives
```

`blocks[b][i]` is arm i's outcome in block b (1 for success, 0 for failure), or `None`
when the arm did not run in that block because it had been dropped.

## The procedure

Each block runs every surviving arm once, so two arms in the same block give a paired
difference. For every pair of arms (i, j), a two-sided
[betting confidence sequence](anytime.md) bounds Δ_ij = p_i − p_j from the blocks that
ran both. With K arms there are P = K(K − 1)/2 pairs, and each sequence uses level δ/P,
so with probability at least 1 − δ all of them cover their difference at every block
(a union bound).

After each block, from `min_blocks` on, arm i is dropped when some arm j beats it with
confidence: the lower bound of Δ_ji is above 0. On the event that all sequences cover, a
best arm is never dropped, whenever the study stops. Therefore:

- a single survivor is the best arm with probability at least 1 − δ;
- several survivors are the arms that could not be told apart from the best, and the best
  is among them with probability at least 1 − δ.

The union bound covers all pairs, not just comparisons with the current leader, because
the leader is chosen from the data. This is the successive-elimination scheme of Even-Dar,
Mannor and Mansour (2006), with time-uniform confidence sequences in place of fixed-time
Hoeffding bounds, so that looking after every block is allowed.

## Choosing δ and the number of blocks

δ is the probability that the procedure drops a best arm. With 4 arms and δ = 0.05 each
pairwise sequence runs at 0.05/6 ≈ 0.008, so elimination needs clear evidence. A
simulation with 4 arms, δ = 0.05, at most 100 blocks and 100 studies per row (seed 5):

| Success rates | Studies ending with a single survivor | Median blocks run |
|---|---|---|
| 0.50, 0.50, 0.50, 0.80 | 60% | 87 |
| 0.55, 0.60, 0.65, 0.90 | 72% | 84 |
| 0.60, 0.65, 0.70, 0.70 | 0% | 100 |

Clearly worse arms are dropped along the way, which saves their trials, but singling out
the best arm takes most of a 100-block budget even for a 25 to 30 point lead. Arms within
5 to 10 points of each other all survive, which is the honest answer: they cannot be told
apart at that budget. Use the surviving set as the result, and run a confirmatory
comparison of the survivors if one configuration must be chosen.

`min_blocks` (default 1) delays the first elimination. It does not affect the error
guarantee; it can help when early blocks are unrepresentative (for example a warm-up
effect).

## References

- Even-Dar, E., Mannor, S. and Mansour, Y. (2006). Action elimination and stopping
  conditions for the multi-armed bandit and reinforcement learning problems. *Journal of
  Machine Learning Research* 7, 1079–1105.
- Waudby-Smith, I. and Ramdas, A. (2024). Estimating means of bounded random variables by
  betting. *Journal of the Royal Statistical Society B* 86(1), 1–27.
- Gupta, S. S. (1965). On some multiple decision (selection and ranking) rules.
  *Technometrics* 7(2), 225–245 (subset selection, the fixed-sample analogue of the
  surviving set).
