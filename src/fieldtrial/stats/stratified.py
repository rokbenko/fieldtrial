"""Comparisons stratified by condition: the Cochran–Mantel–Haenszel test.

Used when each arm runs several replicates per condition, so a simple pairing of one trial
per arm is not available. Each condition is a stratum with its own 2x2 table.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from scipy import stats

from fieldtrial.stats._types import Interval, TestResult
from fieldtrial.stats._validation import check_count, check_open_unit


@dataclass(frozen=True, slots=True)
class StratifiedResult:
    """CMH test plus the Mantel–Haenszel pooled odds ratio of arm 1 versus arm 2."""

    test: TestResult
    odds_ratio: float
    interval: Interval
    strata_used: int


def cmh_test(
    strata: Sequence[tuple[int, int, int, int]],
    *,
    correction: bool = False,
    level: float = 0.95,
) -> StratifiedResult:
    r"""Cochran–Mantel–Haenszel test of no association between arm and success.

    Parameters
    ----------
    strata
        One ``(k1, n1, k2, n2)`` tuple per stratum: successes and trials of arm 1 and arm 2.
        Strata with fewer than 2 trials in total carry no information and are skipped.
    correction
        Apply the continuity correction (off by default, as in statsmodels).
    level
        Confidence level of the pooled odds ratio.

    Notes
    -----
    With :math:`a_j = k_{1j}`, stratum size :math:`N_j`, and the hypergeometric mean
    :math:`E_j` and variance :math:`V_j` of :math:`a_j` given the margins,

    .. math::

        \chi^2_{MH} = \frac{\left(\left|\sum_j (a_j - E_j)\right| - c\right)^2}{\sum_j V_j}
        \sim \chi^2_1,

    with :math:`c = 1/2` under the continuity correction. The pooled odds ratio is the
    Mantel–Haenszel estimator :math:`\sum_j a_j d_j / N_j \big/ \sum_j b_j c_j / N_j` with
    the Robins–Breslow–Greenland variance for its log.

    References
    ----------
    Mantel, N. and Haenszel, W. (1959). *JNCI* 22, 719–748.

    Robins, J., Breslow, N. and Greenland, S. (1986). *Biometrics* 42, 311–323.
    """
    level = check_open_unit("level", level)
    used = []
    for k1, n1, k2, n2 in strata:
        for k, n, label in ((k1, n1, "1"), (k2, n2, "2")):
            check_count(f"k{label}", k)
            check_count(f"n{label}", n)
            if k > n:
                raise ValueError(f"k{label} ({k}) cannot exceed n{label} ({n})")
        if n1 + n2 >= 2:
            used.append((k1, n1, k2, n2))
    if not used:
        raise ValueError("no stratum has at least 2 trials")

    sum_dev = sum_var = 0.0
    r_sum = s_sum = 0.0
    p_r = p_s_q_r = q_s = 0.0
    for k1, n1, k2, n2 in used:
        a, b, c, d = k1, k2, n1 - k1, n2 - k2  # rows: success/failure, columns: arm 1/arm 2
        total = n1 + n2
        successes = a + b
        failures = c + d
        expected = n1 * successes / total
        variance = n1 * n2 * successes * failures / (total**2 * (total - 1))
        sum_dev += a - expected
        sum_var += variance
        r = a * d / total
        s = b * c / total
        r_sum += r
        s_sum += s
        p = (a + d) / total
        q = (b + c) / total
        p_r += p * r
        p_s_q_r += p * s + q * r
        q_s += q * s

    if sum_var == 0:
        statistic, pvalue = math.nan, math.nan
    else:
        deviation = max(abs(sum_dev) - (0.5 if correction else 0.0), 0.0)
        statistic = deviation**2 / sum_var
        pvalue = float(stats.chi2.sf(statistic, 1))

    if r_sum > 0 and s_sum > 0:
        odds_ratio = r_sum / s_sum
        var_log = p_r / (2 * r_sum**2) + p_s_q_r / (2 * r_sum * s_sum) + q_s / (2 * s_sum**2)
        z = float(stats.norm.isf((1 - level) / 2))
        half = z * math.sqrt(var_log)
        interval = Interval(
            math.exp(math.log(odds_ratio) - half),
            math.exp(math.log(odds_ratio) + half),
            level,
            "robins-breslow-greenland",
        )
    else:
        odds_ratio = math.nan if r_sum == s_sum == 0 else (math.inf if s_sum == 0 else 0.0)
        interval = Interval(0.0, math.inf, level, "robins-breslow-greenland")

    return StratifiedResult(
        test=TestResult("cmh", float(statistic), float(pvalue), "two-sided"),
        odds_ratio=odds_ratio,
        interval=interval,
        strata_used=len(used),
    )
