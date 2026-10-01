"""Result types shared by the statistics modules.

Every public function in :mod:`fieldtrial.stats` returns one of these frozen dataclasses.
"""

import math
from dataclasses import dataclass
from typing import Literal

Alternative = Literal["two-sided", "greater", "less"]
"""Direction of the alternative hypothesis. ``"greater"`` means the first arm is higher."""

ALTERNATIVES: tuple[Alternative, ...] = ("two-sided", "greater", "less")


@dataclass(frozen=True, slots=True)
class Interval:
    """A confidence or credible interval.

    Attributes
    ----------
    low, high
        Interval bounds.
    level
        Nominal coverage, for example 0.95.
    method
        Name of the method that produced the interval.
    """

    low: float
    high: float
    level: float
    method: str

    def contains(self, value: float) -> bool:
        """Return whether ``value`` lies inside the closed interval."""
        return self.low <= value <= self.high

    @property
    def width(self) -> float:
        """Width of the interval, ``high - low``."""
        return self.high - self.low


@dataclass(frozen=True, slots=True)
class ProportionEstimate:
    """A success proportion ``successes / trials`` with its interval."""

    successes: int
    trials: int
    estimate: float
    interval: Interval


@dataclass(frozen=True, slots=True)
class TestResult:
    """Result of a hypothesis test.

    Attributes
    ----------
    test
        Name of the test, for example ``"boschloo"``.
    statistic
        The test statistic (its meaning depends on the test). NaN when undefined.
    pvalue
        The p-value. NaN when the test is undefined for the data.
    alternative
        The alternative hypothesis that was tested.
    """

    __test__ = False  # not a pytest test class

    test: str
    statistic: float
    pvalue: float
    alternative: Alternative

    def rejects(self, alpha: float) -> bool:
        """Return whether the test rejects at level ``alpha`` (False when the p-value is NaN)."""
        return not math.isnan(self.pvalue) and self.pvalue <= alpha


@dataclass(frozen=True, slots=True)
class OddsRatio:
    """Conditional maximum-likelihood odds ratio with an exact confidence interval."""

    estimate: float
    interval: Interval


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Comparison of two independent arms.

    ``difference`` is ``arm1.estimate - arm2.estimate`` and ``interval`` is its confidence
    interval. ``primary`` is the test chosen in advance; ``secondary`` holds the other tests,
    reported for comparability.
    """

    arm1: ProportionEstimate
    arm2: ProportionEstimate
    difference: float
    interval: Interval
    primary: TestResult
    secondary: tuple[TestResult, ...]
    odds_ratio: OddsRatio


@dataclass(frozen=True, slots=True)
class PairedComparisonResult:
    """Comparison of two arms evaluated on the same conditions (paired binary outcomes).

    ``b`` counts pairs where only arm 1 succeeded, ``c`` pairs where only arm 2 succeeded,
    and ``n`` is the total number of pairs (``None`` when it was not given, in which case the
    difference and its interval are not available).
    """

    b: int
    c: int
    n: int | None
    difference: float | None
    interval: Interval | None
    test: TestResult


@dataclass(frozen=True, slots=True)
class AdjustedPValues:
    """P-values adjusted for multiple comparisons, in the order they were given."""

    raw: tuple[float, ...]
    adjusted: tuple[float, ...]
    reject: tuple[bool, ...]
    method: str
    alpha: float


@dataclass(frozen=True, slots=True)
class SampleSize:
    """Trials needed per arm to reach the requested power.

    ``n1_exact`` and ``n2_exact`` are the unrounded solutions of the sample-size formula;
    ``n1`` and ``n2`` are rounded up. ``ratio`` is ``n2 / n1``.
    """

    p1: float
    p2: float
    n1_exact: float
    n2_exact: float
    n1: int
    n2: int
    alpha: float
    power: float
    ratio: float
    method: str
    alternative: Alternative


@dataclass(frozen=True, slots=True)
class PowerResult:
    """Power of a planned comparison."""

    p1: float
    p2: float
    n1: int
    n2: int
    alpha: float
    power: float
    method: str
    alternative: Alternative


@dataclass(frozen=True, slots=True)
class MinimumDetectableEffect:
    """The smallest change from a baseline that a study detects with the requested power.

    ``target`` is the success rate at which power reaches the requested level, and ``effect``
    is ``target - baseline``. Both are ``None`` when no such rate exists in (0, 1).
    """

    baseline: float
    n1: int
    n2: int
    direction: Literal["increase", "decrease"]
    target: float | None
    effect: float | None
    alpha: float
    power: float
    method: str


@dataclass(frozen=True, slots=True)
class PairwiseMcNemar:
    """One pairwise exact McNemar comparison inside a multi-arm paired design.

    ``arm_a`` and ``arm_b`` are column indices of the outcome matrix; ``b`` counts blocks
    where only ``arm_a`` succeeded and ``c`` blocks where only ``arm_b`` succeeded.
    """

    arm_a: int
    arm_b: int
    b: int
    c: int
    test: TestResult
    adjusted_pvalue: float
    reject: bool


@dataclass(frozen=True, slots=True)
class CochranQResult:
    """Cochran's Q omnibus test plus pairwise McNemar comparisons with a multiplicity adjustment."""

    statistic: float
    df: int
    pvalue: float
    comparisons: tuple[PairwiseMcNemar, ...]
    adjustment: str
    alpha: float
