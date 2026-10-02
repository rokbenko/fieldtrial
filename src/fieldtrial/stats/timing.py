"""Time to success.

Every trial is observed until it ends, by success, failure or timeout, so there is no
censoring: the fraction of trials that had succeeded by time ``t`` is a plain proportion.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fieldtrial.stats._rng import STREAM_BOOTSTRAP, StableRng
from fieldtrial.stats._types import Interval
from fieldtrial.stats._validation import check_open_unit


@dataclass(frozen=True, slots=True)
class SuccessCurve:
    """Step function: ``fraction[i]`` of all ``n`` trials had succeeded by ``times[i]``."""

    times: tuple[float, ...]
    fraction: tuple[float, ...]
    n: int

    def at(self, t: float) -> float:
        """Fraction of trials that had succeeded by time ``t``."""
        idx = int(np.searchsorted(self.times, t, side="right"))
        return 0.0 if idx == 0 else self.fraction[idx - 1]


@dataclass(frozen=True, slots=True)
class MedianTime:
    """Median time to success among successful trials, with a bootstrap interval."""

    median: float
    interval: Interval
    n_successes: int


def _check_durations(durations: Sequence[float]) -> np.ndarray:
    d = np.asarray(durations, dtype=float)
    if d.ndim != 1:
        raise ValueError("durations must be a flat sequence")
    if d.size and (not np.all(np.isfinite(d)) or d.min() < 0):
        raise ValueError("durations must be finite and non-negative")
    return d


def success_curve(durations: Sequence[float], successes: Sequence[bool]) -> SuccessCurve:
    r"""Cumulative success curve: :math:`F(t) = \#\{i : \text{success}_i,\ d_i \le t\} / n`.

    The denominator is *all* trials, so the curve ends at the success rate. Comparing curves
    shows whether an arm succeeds more often, faster, or both.
    """
    d = _check_durations(durations)
    s = np.asarray(successes, dtype=bool)
    if s.shape != d.shape:
        raise ValueError("durations and successes must have the same length")
    n = int(d.size)
    times = np.sort(d[s])
    unique_times, counts = np.unique(times, return_counts=True)
    fraction = np.cumsum(counts) / n if n else np.zeros(0)
    return SuccessCurve(
        times=tuple(float(t) for t in unique_times),
        fraction=tuple(float(f) for f in fraction),
        n=n,
    )


def median_time_to_success(
    success_durations: Sequence[float],
    *,
    level: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> MedianTime:
    r"""Median time to success with a seeded percentile-bootstrap confidence interval.

    The interval takes the :math:`\alpha/2` and :math:`1-\alpha/2` quantiles of the medians of
    ``n_boot`` resamples drawn with replacement. Randomness comes from the study seed through
    the stable PCG64 stream, so the interval is identical on every platform. With no
    successes the median is NaN; with one, the interval is that single value.

    References
    ----------
    Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*, chapter 13.
    Chapman & Hall.
    """
    d = _check_durations(success_durations)
    level = check_open_unit("level", level)
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    n = int(d.size)
    if n == 0:
        return MedianTime(math.nan, Interval(math.nan, math.nan, level, "bootstrap"), 0)
    median = float(np.median(d))
    rng = StableRng(seed, STREAM_BOOTSTRAP)
    idx = rng.integers(n, n_boot * n).reshape(n_boot, n)
    medians = np.median(d[idx], axis=1)
    alpha = 1.0 - level
    low, high = np.quantile(medians, [alpha / 2.0, 1.0 - alpha / 2.0])
    return MedianTime(median, Interval(float(low), float(high), level, "bootstrap"), n)
