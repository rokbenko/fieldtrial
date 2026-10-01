"""Two independent arms: difference in success rates, its interval, and exact tests."""

import math
import warnings
from typing import Literal

import numpy as np
from scipy import stats
from scipy.stats.contingency import odds_ratio as _odds_ratio

from fieldtrial.stats._types import (
    Alternative,
    ComparisonResult,
    Interval,
    OddsRatio,
    TestResult,
)
from fieldtrial.stats._validation import (
    check_alternative,
    check_choice,
    check_counts,
    check_open_unit,
)
from fieldtrial.stats.proportions import proportion_ci, wilson_interval

IndependentTest = Literal["boschloo", "fisher"]
INDEPENDENT_TESTS: tuple[IndependentTest, ...] = ("boschloo", "fisher")


def arms_table(k1: int, n1: int, k2: int, n2: int) -> np.ndarray:
    """2x2 table with one column per arm: ``[[k1, k2], [n1 - k1, n2 - k2]]``.

    :func:`scipy.stats.boschloo_exact` treats each column as an independent binomial sample,
    so the arms must be the columns. Fisher's test does not depend on the orientation.
    """
    return np.array([[k1, k2], [n1 - k1, n2 - k2]], dtype=np.int64)


def newcombe_interval(
    k1: int, n1: int, k2: int, n2: int, *, level: float = 0.95
) -> tuple[float, float]:
    r"""Newcombe hybrid-score interval for :math:`p_1 - p_2` (method 10).

    With Wilson intervals :math:`[l_i, u_i]` for each arm and :math:`d = \hat p_1 - \hat p_2`:

    .. math::

        \left[\, d - \sqrt{(\hat p_1 - l_1)^2 + (u_2 - \hat p_2)^2},\;
                d + \sqrt{(u_1 - \hat p_1)^2 + (\hat p_2 - l_2)^2} \,\right]

    Reference: Newcombe, R. G. (1998). Interval estimation for the difference between
    independent proportions: comparison of eleven methods. *Statistics in Medicine* 17,
    873–890.
    """
    k1, n1 = check_counts(k1, n1, k_name="k1", n_name="n1")
    k2, n2 = check_counts(k2, n2, k_name="k2", n_name="n2")
    level = check_open_unit("level", level)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1, level)
    l2, u2 = wilson_interval(k2, n2, level)
    d = p1 - p2
    low = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return max(-1.0, low), min(1.0, high)


def boschloo_test(
    k1: int, n1: int, k2: int, n2: int, *, alternative: Alternative = "two-sided"
) -> TestResult:
    r"""Boschloo's exact unconditional test of :math:`p_1 = p_2`.

    The test statistic is Fisher's one-sided p-value; the p-value is the largest
    probability, over the common success rate :math:`\pi` under :math:`H_0`, of a table
    at least as extreme. Arm sizes are fixed by design and nothing else is conditioned on,
    which makes the test uniformly more powerful than Fisher's exact test. Computed by
    :func:`scipy.stats.boschloo_exact`; the two-sided p-value is twice the smaller one-sided
    p-value, capped at 1. ``"greater"`` means :math:`p_1 > p_2`.

    Reference: Boschloo, R. D. (1970). Raised conditional level of significance for the
    2x2-table when testing the equality of two probabilities. *Statistica Neerlandica* 24,
    1–9.
    """
    k1, n1 = check_counts(k1, n1, k_name="k1", n_name="n1")
    k2, n2 = check_counts(k2, n2, k_name="k2", n_name="n2")
    alternative = check_alternative(alternative)
    res = stats.boschloo_exact(arms_table(k1, n1, k2, n2), alternative=alternative)
    return TestResult(
        test="boschloo",
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        alternative=alternative,
    )


def fisher_test(
    k1: int, n1: int, k2: int, n2: int, *, alternative: Alternative = "two-sided"
) -> TestResult:
    r"""Fisher's exact test of :math:`p_1 = p_2`, conditional on both margins.

    The two-sided p-value sums the hypergeometric probabilities of all tables no more
    likely than the observed one (:func:`scipy.stats.fisher_exact`). The statistic is the
    sample odds ratio. ``"greater"`` means :math:`p_1 > p_2`.

    Reference: Fisher, R. A. (1935). The logic of inductive inference. *JRSS* 98, 39–82.
    """
    k1, n1 = check_counts(k1, n1, k_name="k1", n_name="n1")
    k2, n2 = check_counts(k2, n2, k_name="k2", n_name="n2")
    alternative = check_alternative(alternative)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 0/0 sample odds ratio
        res = stats.fisher_exact(arms_table(k1, n1, k2, n2), alternative=alternative)
    return TestResult(
        test="fisher",
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        alternative=alternative,
    )


def odds_ratio(k1: int, n1: int, k2: int, n2: int, *, level: float = 0.95) -> OddsRatio:
    r"""Conditional maximum-likelihood odds ratio of arm 1 versus arm 2, with an exact CI.

    The estimate maximizes the noncentral hypergeometric likelihood given the margins; the
    interval inverts the one-sided conditional tests (:func:`scipy.stats.contingency.odds_ratio`
    with ``kind="conditional"``). It is 0 or infinite when a cell is empty.

    Reference: Cornfield, J. (1956). A statistical problem arising from retrospective
    studies. *Proc. Third Berkeley Symposium* 4, 135–148.
    """
    k1, n1 = check_counts(k1, n1, k_name="k1", n_name="n1")
    k2, n2 = check_counts(k2, n2, k_name="k2", n_name="n2")
    level = check_open_unit("level", level)
    table = arms_table(k1, n1, k2, n2)
    if np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
        # A zero margin leaves the odds ratio unidentified.
        return OddsRatio(
            estimate=math.nan,
            interval=Interval(0.0, math.inf, level, "conditional-exact"),
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = _odds_ratio(table, kind="conditional")
        ci = res.confidence_interval(confidence_level=level)
    return OddsRatio(
        estimate=float(res.statistic),
        interval=Interval(float(ci.low), float(ci.high), level, "conditional-exact"),
    )


def compare_independent(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    *,
    test: IndependentTest | str = "boschloo",
    alternative: Alternative = "two-sided",
    level: float = 0.95,
) -> ComparisonResult:
    r"""Compare the success rates of two independent arms.

    Parameters
    ----------
    k1, n1
        Successes and trials of arm 1 (for example the new checkpoint).
    k2, n2
        Successes and trials of arm 2 (for example the baseline).
    test
        Primary test, chosen before seeing the data: ``"boschloo"`` (default) or
        ``"fisher"``. The other test is reported in ``secondary``.
    alternative
        ``"two-sided"`` (default), ``"greater"`` (:math:`p_1 > p_2`) or ``"less"``.
    level
        Confidence level of the intervals.

    Returns
    -------
    ComparisonResult
        Each arm's Wilson estimate, the difference :math:`\hat p_1 - \hat p_2` with its
        Newcombe hybrid-score interval (:func:`newcombe_interval`), the primary and secondary
        tests (:func:`boschloo_test`, :func:`fisher_test`), and the conditional odds ratio
        (:func:`odds_ratio`).

    Notes
    -----
    The interval is always two-sided at ``level``, whatever the alternative of the test.
    """
    k1, n1 = check_counts(k1, n1, k_name="k1", n_name="n1")
    k2, n2 = check_counts(k2, n2, k_name="k2", n_name="n2")
    check_choice("test", test, INDEPENDENT_TESTS)
    alternative = check_alternative(alternative)
    level = check_open_unit("level", level)

    low, high = newcombe_interval(k1, n1, k2, n2, level=level)
    boschloo = boschloo_test(k1, n1, k2, n2, alternative=alternative)
    fisher = fisher_test(k1, n1, k2, n2, alternative=alternative)
    primary, secondary = (boschloo, fisher) if test == "boschloo" else (fisher, boschloo)
    return ComparisonResult(
        arm1=proportion_ci(k1, n1, level=level),
        arm2=proportion_ci(k2, n2, level=level),
        difference=k1 / n1 - k2 / n2,
        interval=Interval(low, high, level, "newcombe"),
        primary=primary,
        secondary=(secondary,),
        odds_ratio=odds_ratio(k1, n1, k2, n2, level=level),
    )
