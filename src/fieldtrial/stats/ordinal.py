"""Progress stages: how far trials got, not just whether they succeeded.

Each trial's outcome is the index of the furthest stage it reached (``-1`` for none). Reaching
stage ``s`` implies reaching every earlier stage.
"""

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from fieldtrial.stats._types import Alternative, Interval, TestResult
from fieldtrial.stats._validation import check_alternative, check_open_unit
from fieldtrial.stats.proportions import wilson_interval


@dataclass(frozen=True, slots=True)
class StageCount:
    """How many trials stopped exactly at ``stage`` (``-1`` = no stage) and their share."""

    stage: int
    count: int
    share: float


@dataclass(frozen=True, slots=True)
class FunnelStep:
    """Reaching ``stage``, among trials that reached the previous stage.

    ``reached`` counts trials that got at least this far and ``entered`` the trials that
    reached the previous stage (all trials for stage 0). ``conversion`` is
    ``reached / entered`` with its Wilson ``interval``; ``cumulative`` is ``reached / n``.
    """

    stage: int
    entered: int
    reached: int
    conversion: float
    interval: Interval
    cumulative: float


def _check_stages(stage_indices: Sequence[int], n_stages: int) -> np.ndarray:
    if n_stages < 1:
        raise ValueError("n_stages must be at least 1")
    x = np.asarray(stage_indices, dtype=np.int64)
    if x.ndim != 1:
        raise ValueError("stage_indices must be a flat sequence")
    if x.size and (x.min() < -1 or x.max() >= n_stages):
        raise ValueError(f"stage indices must be between -1 and {n_stages - 1}")
    return x


def stage_distribution(stage_indices: Sequence[int], n_stages: int) -> tuple[StageCount, ...]:
    """Distribution of the furthest stage reached, from ``-1`` (none) to ``n_stages - 1``."""
    x = _check_stages(stage_indices, n_stages)
    n = x.size
    return tuple(
        StageCount(
            stage=s, count=int(np.sum(x == s)), share=float(np.sum(x == s)) / n if n else 0.0
        )
        for s in range(-1, n_stages)
    )


def stage_funnel(
    stage_indices: Sequence[int], n_stages: int, *, level: float = 0.95
) -> tuple[FunnelStep, ...]:
    r"""Stage funnel with conditional conversion rates and Wilson intervals.

    For stage :math:`s`, :math:`R_s = \#\{i: x_i \ge s\}` trials reached it and
    :math:`R_{s-1}` (with :math:`R_{-1} = n`) entered it; the conversion is
    :math:`R_s / R_{s-1}`. A step with nobody entering has conversion NaN and the interval
    [0, 1].

    The funnel shows *where* a policy fails: two arms with the same success rate can fail
    at different stages.
    """
    x = _check_stages(stage_indices, n_stages)
    level = check_open_unit("level", level)
    n = int(x.size)
    steps = []
    entered = n
    for s in range(n_stages):
        reached = int(np.sum(x >= s))
        if entered > 0:
            low, high = wilson_interval(reached, entered, level)
            conversion = reached / entered
        else:
            low, high, conversion = 0.0, 1.0, math.nan
        steps.append(
            FunnelStep(
                stage=s,
                entered=entered,
                reached=reached,
                conversion=conversion,
                interval=Interval(low, high, level, "wilson"),
                cumulative=reached / n if n else math.nan,
            )
        )
        entered = reached
    return tuple(steps)


def compare_stages(
    stages_a: Sequence[int],
    stages_b: Sequence[int],
    *,
    alternative: Alternative = "two-sided",
) -> TestResult:
    r"""Brunner–Munzel test on the furthest stage reached (secondary comparison).

    Tests :math:`H_0: P(X_A < X_B) + \tfrac12 P(X_A = X_B) = \tfrac12`, a rank test that
    handles the many ties of ordinal stage data and does not assume equal variances
    (:func:`scipy.stats.brunnermunzel`, t distribution). ``"greater"`` means arm A tends to
    get further. When all observations are identical the test is undefined: the statistic
    and p-value are NaN.

    References
    ----------
    Brunner, E. and Munzel, U. (2000). The nonparametric Behrens–Fisher problem: asymptotic
    theory and a small-sample approximation. *Biometrical Journal* 42, 17–25.
    """
    alternative = check_alternative(alternative)
    a = np.asarray(stages_a, dtype=float)
    b = np.asarray(stages_b, dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError("each arm needs at least 2 trials")
    if np.all(np.concatenate([a, b]) == a[0]):
        return TestResult("brunner-munzel", math.nan, math.nan, alternative)
    # scipy's "greater" means the first sample tends to be larger, which matches fieldtrial's
    # convention that "greater" favors the first argument.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = stats.brunnermunzel(a, b, alternative=alternative)
    return TestResult("brunner-munzel", float(res.statistic), float(res.pvalue), alternative)
