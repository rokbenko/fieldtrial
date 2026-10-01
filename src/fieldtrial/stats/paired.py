"""Paired designs: arms evaluated on the same conditions (complete blocks).

For two arms, every block contributes one pair of binary outcomes. Only the discordant pairs
carry information about the difference: ``b`` blocks where only arm 1 succeeded and ``c``
blocks where only arm 2 succeeded.
"""

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy import optimize, stats

from fieldtrial.stats._types import (
    Alternative,
    CochranQResult,
    Interval,
    PairedComparisonResult,
    PairwiseMcNemar,
    TestResult,
)
from fieldtrial.stats._validation import (
    check_alternative,
    check_choice,
    check_count,
    check_open_unit,
)
from fieldtrial.stats.multiplicity import ADJUST_METHODS, AdjustMethod, adjust_pvalues


def mcnemar_exact(b: int, c: int, *, alternative: Alternative = "two-sided") -> TestResult:
    r"""Exact McNemar test on the discordant pairs.

    Under :math:`H_0` (both arms have the same success rate) each discordant pair is equally
    likely to favor either arm, so :math:`b \sim \text{Binomial}(b + c, 1/2)`. The two-sided
    p-value is :math:`\min\{1, 2\min[P(B \le b), P(B \ge b)]\}`; ``"greater"`` means arm 1
    succeeds more often (large :math:`b`). With no discordant pairs the p-value is 1.

    The returned ``statistic`` is :math:`b`.

    References
    ----------
    McNemar, Q. (1947). Note on the sampling error of the difference between correlated
    proportions or percentages. *Psychometrika* 12, 153–157.

    Fagerland, M. W., Lydersen, S. and Laake, P. (2013). The McNemar test for binary
    matched-pairs data: mid-p and asymptotic are better than exact conditional.
    *BMC Medical Research Methodology* 13, 91.
    """
    b = check_count("b", b)
    c = check_count("c", c)
    alternative = check_alternative(alternative)
    if b + c == 0:
        return TestResult("mcnemar-exact", 0.0, 1.0, alternative)
    p = stats.binomtest(b, b + c, 0.5, alternative=alternative).pvalue
    return TestResult("mcnemar-exact", float(b), float(p), alternative)


def _tango_score(delta: float, b: int, c: int, n: int) -> float:
    """Tango's score statistic for H0: p1 - p2 = delta (0 where it is 0/0)."""
    numerator = b - c - n * delta
    a_coef = 2.0 * n
    b_coef = -b - c + (2.0 * n - b + c) * delta
    c_coef = -c * delta * (1.0 - delta)
    disc = max(b_coef * b_coef - 4.0 * a_coef * c_coef, 0.0)
    q = max((-b_coef + math.sqrt(disc)) / (2.0 * a_coef), 0.0)  # constrained MLE of p21
    variance = n * (2.0 * q + delta * (1.0 - delta))
    if variance <= 0.0:
        return 0.0 if numerator == 0 else math.copysign(math.inf, numerator)
    return numerator / math.sqrt(variance)


def tango_interval(b: int, c: int, n: int, *, level: float = 0.95) -> tuple[float, float]:
    r"""Tango's score confidence interval for a paired difference :math:`p_1 - p_2`.

    With :math:`\Delta = p_1 - p_2 = p_{12} - p_{21}` (the discordant cell probabilities),
    the score statistic for :math:`H_0: \Delta = \delta` is

    .. math::

        T(\delta) = \frac{b - c - n\delta}{\sqrt{n\,(2\tilde p_{21} + \delta(1 - \delta))}},

    where :math:`\tilde p_{21}` is the maximum-likelihood estimate of :math:`p_{21}` under the
    constraint, the positive root of :math:`A x^2 + B x + C = 0` with :math:`A = 2n`,
    :math:`B = -b - c + (2n - b + c)\delta` and :math:`C = -c\,\delta(1 - \delta)`. The
    interval is :math:`\{\delta : |T(\delta)| \le z_{1-\alpha/2}\}`, found by root-finding on
    either side of :math:`\hat\Delta = (b - c)/n`. When :math:`b = c = 0` it reduces to
    :math:`\pm z^2/(n + z^2)`.

    Parameters
    ----------
    b, c
        Discordant pairs: blocks where only arm 1 succeeded, and where only arm 2 succeeded.
    n
        Total number of pairs (blocks), at least ``b + c`` and at least 1.
    level
        Confidence level.

    References
    ----------
    Tango, T. (1998). Equivalence test and confidence interval for the difference in
    proportions for the paired-sample design. *Statistics in Medicine* 17, 891–908.

    Fagerland, M. W., Lydersen, S. and Laake, P. (2014). Recommended tests and confidence
    intervals for paired binomial proportions. *Statistics in Medicine* 33, 2850–2875.
    """
    b = check_count("b", b)
    c = check_count("c", c)
    n = check_count("n", n)
    if n == 0:
        raise ValueError("n must be at least 1, got 0")
    if b + c > n:
        raise ValueError(f"b + c ({b + c}) cannot exceed n ({n})")
    level = check_open_unit("level", level)
    z = float(stats.norm.isf((1.0 - level) / 2.0))
    estimate = (b - c) / n
    eps = 1e-12

    if estimate >= 1.0:
        upper = 1.0
    else:
        upper = optimize.brentq(
            lambda d: _tango_score(d, b, c, n) + z, estimate, 1.0 - eps, xtol=1e-14
        )
    if estimate <= -1.0:
        lower = -1.0
    else:
        lower = optimize.brentq(
            lambda d: _tango_score(d, b, c, n) - z, -1.0 + eps, estimate, xtol=1e-14
        )
    return max(-1.0, float(lower)), min(1.0, float(upper))


def compare_paired(
    b: int,
    c: int,
    n: int | None = None,
    *,
    alternative: Alternative = "two-sided",
    level: float = 0.95,
) -> PairedComparisonResult:
    r"""Compare two arms evaluated on the same blocks.

    The test is the exact McNemar test (:func:`mcnemar_exact`). When the total number of
    pairs ``n`` is given, the result also contains the difference :math:`(b - c)/n` and its
    Tango score interval (:func:`tango_interval`).
    """
    test = mcnemar_exact(b, c, alternative=alternative)
    if n is None:
        return PairedComparisonResult(b=b, c=c, n=None, difference=None, interval=None, test=test)
    low, high = tango_interval(b, c, n, level=level)
    return PairedComparisonResult(
        b=b,
        c=c,
        n=n,
        difference=(b - c) / n,
        interval=Interval(low, high, level, "tango"),
        test=test,
    )


def cochran_q(
    outcomes: Sequence[Sequence[int]] | np.ndarray,
    *,
    pairs: Literal["all", "vs-first"] = "all",
    adjustment: AdjustMethod | str = "holm",
    alpha: float = 0.05,
) -> CochranQResult:
    r"""Cochran's Q test for more than two arms on the same blocks, plus pairwise McNemar tests.

    Parameters
    ----------
    outcomes
        Binary matrix with one row per block and one column per arm (1 = success).
    pairs
        ``"all"`` compares every pair of arms; ``"vs-first"`` compares each arm with the
        first column (the control).
    adjustment
        Multiplicity adjustment for the pairwise tests: ``"holm"`` (default),
        ``"bonferroni"`` or ``"bh"``.
    alpha
        Level for the pairwise rejection decisions.

    Notes
    -----
    With :math:`k` arms, column totals :math:`C_j`, row totals :math:`R_i` and grand total
    :math:`N`:

    .. math::

        Q = \frac{(k - 1)\left(k \sum_j C_j^2 - N^2\right)}{k N - \sum_i R_i^2},

    referred to a :math:`\chi^2_{k-1}` distribution. Blocks where all arms agree do not
    affect :math:`Q`. When every block agrees, :math:`Q = 0` and the p-value is 1.

    References
    ----------
    Cochran, W. G. (1950). The comparison of percentages in matched samples. *Biometrika*
    37, 256–266.
    """
    x = np.asarray(outcomes)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2:
        raise ValueError("outcomes must be a matrix with at least one block and two arms")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("outcomes must contain only 0 and 1")
    check_choice("pairs", pairs, ("all", "vs-first"))
    check_choice("adjustment", adjustment, ADJUST_METHODS)
    x = x.astype(np.int64)
    k = x.shape[1]
    col = x.sum(axis=0)
    row = x.sum(axis=1)
    total = int(row.sum())
    denominator = k * total - int((row**2).sum())
    if denominator == 0:
        q, p = 0.0, 1.0
    else:
        q = (k - 1) * (k * int((col**2).sum()) - total**2) / denominator
        p = float(stats.chi2.sf(q, k - 1))

    if pairs == "all":
        index_pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    else:
        index_pairs = [(j, 0) for j in range(1, k)]
    tests = []
    for i, j in index_pairs:
        b = int(np.sum((x[:, i] == 1) & (x[:, j] == 0)))
        c = int(np.sum((x[:, i] == 0) & (x[:, j] == 1)))
        tests.append((i, j, b, c, mcnemar_exact(b, c)))
    adjusted = adjust_pvalues([t[4].pvalue for t in tests], method=adjustment, alpha=alpha)
    comparisons = tuple(
        PairwiseMcNemar(arm_a=i, arm_b=j, b=b, c=c, test=test, adjusted_pvalue=adj, reject=rej)
        for (i, j, b, c, test), adj, rej in zip(
            tests, adjusted.adjusted, adjusted.reject, strict=True
        )
    )
    return CochranQResult(
        statistic=float(q),
        df=k - 1,
        pvalue=p,
        comparisons=comparisons,
        adjustment=adjustment,
        alpha=adjusted.alpha,
    )
