# Time to success

Every trial is observed until it ends, by success, failure or timeout, so there is no
censoring and no survival model is needed.

```python
from fieldtrial.stats import success_curve, median_time_to_success

curve = success_curve(durations, successes)
curve.at(20.0)  # fraction of all trials that had succeeded within 20 s
median_time_to_success(success_durations, seed=20261001)
```

**Cumulative success curve.** F(t) = #{successful trials with duration ≤ t} / n. The
denominator is all trials, so the curve ends at the success rate. It shows whether an arm
succeeds more often, faster, or both.

**Median time to success** among successful trials, with a percentile-bootstrap interval. It
uses 2000 resamples by default, drawn from the study seed through the stable PCG64 stream, so
the interval is the same on every machine.

## References

- Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*, chapter 13.
