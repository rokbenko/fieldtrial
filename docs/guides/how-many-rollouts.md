# How many rollouts do I need?

Fewer than you hope. Real-world rollouts are expensive, so it pays to know before you start
what an evaluation can and cannot show.

## One arm: how precise is a success rate?

With 40 rollouts and 32 successes (80%), the 95% Wilson interval runs from 65% to 90%:

| Rollouts | 95% interval at 80% success | Width |
|---|---|---|
| 20 | 58–92% | 34 pp |
| 40 | 65–90% | 24 pp |
| 80 | 70–87% | 17 pp |
| 160 | 73–86% | 12 pp |

```console
$ fieldtrial ci 32/40
```

Halving the width takes four times the rollouts.

## Two arms: what difference can I detect?

To detect a change from 76% to 90% with 80% power (two-sided α = 0.05), you need about
**112 rollouts per arm**:

```console
$ fieldtrial power --p1 0.76 --p2 0.90
112 per arm (pooled-z)
```

| Baseline → target | Rollouts per arm |
|---|---|
| 50% → 70% | 93 |
| 60% → 80% | 82 |
| 76% → 90% | 112 |
| 80% → 90% | 199 |
| 90% → 95% | 435 |

Turned around: with a 75% baseline, what is the smallest change you would detect with 80%
power?

| Rollouts per arm | Detectable increase | Detectable decrease |
|---|---|---|
| 40 | +21.6 pp | −30.3 pp |
| 80 | +16.4 pp | −21.0 pp |
| 160 | +12.2 pp | −14.6 pp |
| 300 | +9.2 pp | −10.5 pp |

```console
$ fieldtrial mde --p1 0.75 --n 80
+16.4 pp (to 0.9143) / −21.0 pp (to 0.5396)
```

So 40 rollouts per arm can only detect very large improvements. With 40 per arm, a true
76% → 90% improvement is detected only about a third of the time.

## Unequal arms

Comparing a new checkpoint against a baseline you have run many times? `--ratio` gives
the split: with three baseline rollouts per new one, detecting 76% → 90% needs 69 new
and 207 baseline rollouts.

```console
$ fieldtrial power --p1 0.76 --p2 0.90 --ratio 3
```

Reusing old baseline runs is cheap but risky. If the rig drifted since then, the
comparison picks up the drift. Interleaving the arms in one study avoids that.

## Paired designs need fewer rollouts

When both arms run on the same starting conditions (a `randomized_block` study), the
analysis compares them within each condition, so easy and hard conditions cancel out. What
counts is how often exactly one arm succeeds. If arm B succeeds where A fails in 20% of
conditions and the reverse happens in 6%, the exact McNemar test has about 30% power with
40 conditions, 49% with 60 and 64% with 80. The gain over independent samples depends on
how strongly the outcomes are correlated across conditions, and `fieldtrial plan` reports
the conservative independent-samples figure.

## Before you lock

`fieldtrial plan` prints the schedule and the minimum detectable effect for your design:

```console
$ fieldtrial plan my-study --baseline 0.75
```

If the detectable difference is larger than anything you expect, either run more
conditions or replicates, or decide in advance that this study is a sanity check rather
than a comparison. A non-significant result from an underpowered study says "we could not
tell", not "no difference", and fieldtrial's reports say exactly that.

See [Planning](../stats/planning.md) for the formulas.
