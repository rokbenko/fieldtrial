# Statistics reference

`fieldtrial.stats` is the statistics core. It depends only on numpy and scipy, and every
function is tested against an independent reference implementation (statsmodels, scipy,
closed forms or brute-force enumeration).

| Question | Method | Page |
|---|---|---|
| What is the success rate of one arm? | Wilson interval (default), Clopper–Pearson, Jeffreys, Agresti–Coull | [One arm](intervals.md) |
| Is an arm above a fixed threshold? | Exact binomial test | [One arm](intervals.md#threshold-test) |
| Did arm 1 beat arm 2 (separate trials)? | Newcombe CI, Boschloo exact test, Fisher | [Two independent arms](two-arms.md) |
| Did arm 1 beat arm 2 (same conditions)? | Exact McNemar, Tango CI | [Paired designs](paired.md) |
| More than two arms on the same conditions? | Cochran's Q + pairwise McNemar | [Paired designs](paired.md#more-than-two-arms) |
| Several comparisons at once? | Holm, Bonferroni, Benjamini–Hochberg | [Multiplicity](multiplicity.md) |
| Where do arms fail along the task? | Stage funnel, Brunner–Munzel | [Progress stages](stages.md) |
| How fast do arms succeed? | Cumulative success curve, median time with bootstrap CI | [Time to success](timing.md) |
| Several replicates per condition? | Cochran–Mantel–Haenszel | [Stratified (CMH)](stratified.md) |
| Did results change across sessions? | Fisher / chi-square homogeneity | [Drift checks](drift.md) |
| Does success change with training step? Where does it level off? | Cochran–Armitage / Mantel trend, fixed-sequence non-inferiority | [Checkpoint ladders](ladders.md) |
| The scene carries over between trials? | Two-period crossover of whole rounds, exact randomization test | [Crossover rounds](crossover.md) |
| Can I stop early? | Lan–DeMets error spending, stage-wise p-values | [Group-sequential stopping](sequential.md) |
| Can I look after every block? | Betting confidence sequences, anytime-valid p-values | [Anytime-valid comparisons](anytime.md) |
| Does a reward model agree with people? | Cohen's κ on blind labels | [Agreement](agreement.md) |
| Can reward-model scores tighten my interval? | Prediction-powered inference (PPI++) | [Proxy-assisted estimates](ppi.md) |
| Which of several configurations is best? | Successive elimination with confidence sequences | [Best-arm selection](selection.md) |
| How many rollouts do I need? What can I detect? | Sample size, power, MDE, exact power | [Planning](planning.md) |
| How sure am I that B beats A? | Posterior probability (descriptive only) | [Bayesian summaries](bayes.md) |

## Conventions

- Defaults: 95% intervals, two-sided tests, α = 0.05.
- `alternative="greater"` always means *arm 1 is higher*.
- Results are frozen dataclasses (`Interval`, `ProportionEstimate`, `TestResult`, …).
- Invalid input (no trials, more successes than trials, a level outside (0, 1)) raises
  `ValueError` with a message that names the argument.

## Further reading

Kress-Gazit, H. et al. (2024). *Robot Learning as an Empirical Science: Best Practices for
Policy Evaluation.* arXiv:2409.09491.
