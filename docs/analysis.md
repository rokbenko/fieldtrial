# Analysis and reports

`fieldtrial analyze` turns a study's trials into a versioned `Results` model. Reports
(`report.html`, `report.md`, `results.json`) are rendered from it. The analysis plan comes from the locked
design, and nothing is chosen after seeing the data.

## Primary analysis

| Locked design | Primary analysis |
|---|---|
| `randomized_block`, 2 arms, 1 replicate | [Exact McNemar test](stats/paired.md) with a Tango score interval, on blocks where both arms have a completed trial |
| `randomized_block`, 2 arms, several replicates | [Cochran–Mantel–Haenszel test](stats/stratified.md) stratified by condition; reports the Mantel–Haenszel odds ratio |
| `randomized_block`, more than 2 arms | [Cochran's Q](stats/paired.md) on complete blocks, then each arm vs the control with exact McNemar tests and the pre-registered `multiplicity` adjustment (Holm by default) |
| `randomized_block`, 2 arms, `stopping: group_sequential` | [Group-sequential](stats/sequential.md) McNemar score statistic at the planned looks; stage-wise p-value and a repeated confidence interval |
| `randomized_block` with `analysis.ladder` and no `comparison` | [Mantel's test](stats/ladders.md) of a linear association between training step and success, stratified by condition |
| `crossover_rounds` | [Period-adjusted difference](stats/crossover.md) over cycles of whole rounds, exact randomization test and interval |
| `single_arm` with `threshold` | [Exact binomial test](stats/intervals.md) against the threshold |
| `single_arm` without `threshold` | Success rate and interval only |

Blocks without a completed trial for every arm are left out of paired analyses. The report
says how many, and lists them as a deviation.

With more than two arms, the claim about the pre-registered treatment uses its adjusted
McNemar p-value. Cochran's Q is reported as the omnibus test.

With a one-sided `alternative`, the CMH p-value is halved in the hypothesized direction,
and set to 1 − p/2 in the other.

Flagged [rig checks](guides/capture.md) are listed as deviations, and a table shows every
check. Trials [linked to LeRobot episodes](guides/lerobot.md#linking-trials-to-dataset-episodes)
get a descriptive **dataset episodes** table: linked trials, frames and interventions per
arm.

With a switching runner, a **runner** table lists per arm what the runner measured:
requests, errors and latency for the openpi router, and abnormal exits for the command
runner. It is descriptive, a check that the arms were served alike, and no test is run
on it.

A pre-registered `analysis.ladder` is always analyzed: the association with step and,
with a `margin`, the plateau. It is the primary analysis without a `comparison`, and a
pre-registered secondary analysis alongside one.

In a group-sequential study, the report lists every look with its information fraction,
statistic and boundary. Looks that were planned but not run, and interim decisions that
no longer reproduce because labels were edited later, are listed as deviations.

## Sensitivity analysis

Each arm is also compared with the control as if all trials were independent:
[Boschloo's exact test and a Newcombe interval](stats/two-arms.md), with Fisher's exact test
alongside. If the paired and independent analyses disagree, look at the excluded blocks and
the drift checks.

## Secondary analyses

These are descriptive. They are marked "pre-registered" when they are listed under
`analysis.secondary`.

- [Progress stages](stats/stages.md): the furthest stage reached per arm, a funnel with
  Wilson intervals, and a Brunner–Munzel test on the stage index.
- [Time to success](stats/timing.md): the median time among successful trials with a seeded
  bootstrap interval, and the cumulative success curve.
- Outcomes per condition, and per session.

## Validity checks

These are screens for a human to look at, flagged at a fixed 0.05 level. They are not tests
of the study hypothesis.

- **Drift**: each arm's success rate across sessions and across operators
  ([homogeneity test](stats/drift.md)). "Results changed across sessions" usually means
  something on the rig changed.
- **Invalid-trial imbalance**: whether invalid trials concentrate in some arms. Voiding a
  trial can hide a failure.

## Deviations

The report lists everything that departs from the locked plan:

- amendments, with reason and design hashes
- unblinding while trials were still pending
- an incomplete study (pending trials)
- trials run out of the scheduled order
- blocks left out of the paired analysis

## Wording

Every sentence about results comes from `fieldtrial.analysis.wording`, one tested template
per situation. A difference is only described when the pre-registered primary test
rejected:

> q50 succeeded in 92.5% of trials (74/80; 95% CI 84.6–96.5%) vs 75.8% (91/120) for
> baseline: +16.7 pp (95% CI +6.2 to +26.0; Boschloo p = 0.0020).

Otherwise the report says so, and states what the study could have detected:

> No significant difference detected: +14.2 pp (95% CI −0.5 to +24.5; p = 0.056). With 40
> and 120 trials, this study had 80% power only for differences of at least 18.7 pp.

The detectable difference uses the [independent-samples formula](stats/planning.md) with
the observed control rate, so it is conservative for paired designs. Report text never says
"better", "worse", "trend toward significance" or "almost significant"; the wording module
rejects these.

## Charts

`report.html` is one self-contained file: inline CSS and SVG charts, no scripts, nothing
loaded from other hosts, so it opens offline and can be attached anywhere. Colors follow
the Okabe–Ito palette, which stays distinguishable with the common forms of color
blindness. The charts:

- success rate per arm with its confidence interval
- a forest plot of the differences against the control: the paired primary analysis and
  the independent-samples sensitivity analysis
- the furthest stage reached, as stacked bars per arm
- the stage funnel: the share of trials reaching each stage
- cumulative success over time
- success per condition and arm, as a heatmap
- success per session, in time order (drift)

`report.md` has the same tables without charts, for GitHub and pull requests.

## Results schema

`results.json` follows the `Results` model. Its JSON schema is published at
[`reference/results.schema.json`](reference/results.schema.json) and versioned by
`schema_version`. The version changes when a field is renamed or removed; adding optional
fields does not change it.

Provenance records the fieldtrial version used to lock and to analyze, the Python, numpy and
scipy versions, the design hash and seed, and a fingerprint of each arm's policy, serving
settings and runner.
