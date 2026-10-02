"""Small conversions shared by the analysis modules."""

import math

from fieldtrial.analysis.results import CI
from fieldtrial.stats import Interval

LEVEL = 0.95


def ci_model(interval: Interval | None) -> CI | None:
    """A stats ``Interval`` as a results ``CI`` (None when undefined)."""
    if interval is None or math.isnan(interval.low) or math.isnan(interval.high):
        return None
    return CI(low=interval.low, high=interval.high, level=interval.level, method=interval.method)


def num(x: float) -> float | None:
    """A float, or None for NaN."""
    return None if math.isnan(x) else float(x)
