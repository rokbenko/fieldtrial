"""One arm: confidence intervals for a success proportion, and a test against a threshold."""

import math
from typing import Literal

from scipy import stats

from fieldtrial.stats._types import Alternative, Interval, ProportionEstimate, TestResult
from fieldtrial.stats._validation import (
    check_alternative,
    check_choice,
    check_counts,
    check_open_unit,
)

IntervalMethod = Literal["wilson", "clopper-pearson", "jeffreys", "agresti-coull"]
INTERVAL_METHODS: tuple[IntervalMethod, ...] = (
    "wilson",
    "clopper-pearson",
    "jeffreys",
    "agresti-coull",
)


def _clip(x: float) -> float:
    return min(1.0, max(0.0, x))


def wilson_interval(k: int, n: int, level: float) -> tuple[float, float]:
    r"""Wilson score interval bounds (no argument checks; see :func:`proportion_ci`).

    With :math:`z = \Phi^{-1}(1 - \alpha/2)` and :math:`\hat p = k/n`:

    .. math::

        \frac{\hat p + z^2/(2n)}{1 + z^2/n}
        \pm \frac{z}{1 + z^2/n} \sqrt{\frac{\hat p (1-\hat p)}{n} + \frac{z^2}{4n^2}}
    """
    z = float(stats.norm.isf((1.0 - level) / 2.0))
    z2 = z * z
    denom = n + z2
    center = (k + z2 / 2.0) / denom
    half = z / denom * math.sqrt(k * (n - k) / n + z2 / 4.0)
    low = 0.0 if k == 0 else _clip(center - half)
    high = 1.0 if k == n else _clip(center + half)
    return low, high


def proportion_ci(
    k: int,
    n: int,
    *,
    level: float = 0.95,
    method: IntervalMethod | str = "wilson",
) -> ProportionEstimate:
    r"""Estimate a success proportion with a confidence interval.

    Parameters
    ----------
    k
        Number of successes.
    n
        Number of trials, at least 1.
    level
        Confidence level, strictly between 0 and 1.
    method
        ``"wilson"`` (default), ``"clopper-pearson"``, ``"jeffreys"`` or ``"agresti-coull"``.

    Returns
    -------
    ProportionEstimate
        The estimate :math:`\hat p = k/n` and its interval, clipped to [0, 1].

    Notes
    -----
    With :math:`\alpha = 1 - \text{level}` and :math:`z = \Phi^{-1}(1-\alpha/2)`:

    - **Wilson** (score) interval: see :func:`wilson_interval`. It has close to nominal
      coverage at the sample sizes typical of robot evaluations and is the default.
    - **Clopper–Pearson** ("exact"): :math:`[B(\alpha/2; k, n-k+1),\ B(1-\alpha/2; k+1, n-k)]`,
      where :math:`B(q; a, b)` is the Beta quantile; the bound is 0 when :math:`k=0` and 1
      when :math:`k=n`. Conservative.
    - **Jeffreys**: the equal-tailed interval of the
      :math:`\text{Beta}(k + 1/2, n - k + 1/2)` posterior, with the lower bound set to 0 when
      :math:`k = 0` and the upper bound set to 1 when :math:`k = n` (Brown, Cai and DasGupta,
      2001).
    - **Agresti–Coull**: with :math:`\tilde n = n + z^2` and
      :math:`\tilde p = (k + z^2/2)/\tilde n`, the interval is
      :math:`\tilde p \pm z\sqrt{\tilde p(1-\tilde p)/\tilde n}`.

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession, and statistical
    inference. *JASA* 22, 209–212.

    Clopper, C. J. and Pearson, E. S. (1934). The use of confidence or fiducial limits
    illustrated in the case of the binomial. *Biometrika* 26, 404–413.

    Agresti, A. and Coull, B. A. (1998). Approximate is better than "exact" for interval
    estimation of binomial proportions. *The American Statistician* 52, 119–126.

    Brown, L. D., Cai, T. T. and DasGupta, A. (2001). Interval estimation for a binomial
    proportion. *Statistical Science* 16, 101–133.
    """
    k, n = check_counts(k, n)
    level = check_open_unit("level", level)
    check_choice("method", method, INTERVAL_METHODS)
    alpha = 1.0 - level

    if method == "wilson":
        low, high = wilson_interval(k, n, level)
    elif method == "clopper-pearson":
        low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
        high = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    elif method == "jeffreys":
        posterior = stats.beta(k + 0.5, n - k + 0.5)
        low = 0.0 if k == 0 else float(posterior.ppf(alpha / 2.0))
        high = 1.0 if k == n else float(posterior.ppf(1.0 - alpha / 2.0))
    else:  # agresti-coull
        z = float(stats.norm.isf(alpha / 2.0))
        n_tilde = n + z * z
        p_tilde = (k + z * z / 2.0) / n_tilde
        half = z * math.sqrt(p_tilde * (1.0 - p_tilde) / n_tilde)
        low, high = p_tilde - half, p_tilde + half

    interval = Interval(low=_clip(low), high=_clip(high), level=level, method=method)
    return ProportionEstimate(successes=k, trials=n, estimate=k / n, interval=interval)


def test_vs_threshold(
    k: int,
    n: int,
    p0: float,
    *,
    alternative: Alternative = "two-sided",
) -> TestResult:
    r"""Exact binomial test of a success proportion against a fixed threshold ``p0``.

    Tests :math:`H_0: p = p_0` against the chosen alternative (``"greater"`` means
    :math:`p > p_0`, for example "is success above 0.90?"). The p-value is computed from the
    :math:`\text{Binomial}(n, p_0)` distribution by :func:`scipy.stats.binomtest`; the
    two-sided p-value sums the probabilities of all outcomes no more likely than the observed
    one.

    The returned ``statistic`` is the observed proportion :math:`k/n`.

    References
    ----------
    Clopper, C. J. and Pearson, E. S. (1934). *Biometrika* 26, 404–413.
    """
    k, n = check_counts(k, n)
    p0 = check_open_unit("p0", p0)
    alternative = check_alternative(alternative)
    result = stats.binomtest(k, n, p0, alternative=alternative)
    return TestResult(
        test="binomial", statistic=k / n, pvalue=float(result.pvalue), alternative=alternative
    )


test_vs_threshold.__test__ = False  # type: ignore[attr-defined]  # not a pytest test
