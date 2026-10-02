r"""Crossover rounds: arms alternate whole rounds, for tasks whose scene carries over.

When the scene evolves across trials (filling a tray, clearing a table), the trials in a
round are not independent, so the round is the unit of analysis. Rounds come in cycles of
two periods. In each cycle one arm runs first and the other second, with the order (``AB``
or ``BA``) randomized so that each order occurs equally often. Each round gives one
success proportion.

The analysis is the classical two-period crossover analysis of Hills and Armitage (1979),
with an exact randomization test that matches how the orders were assigned.
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from fieldtrial.stats._types import Alternative, Interval, TestResult
from fieldtrial.stats._validation import check_alternative, check_open_unit

Order = Literal["AB", "BA"]
EXACT_LIMIT = 200_000
"""Largest number of relabellings enumerated exactly; above it the t-test p-value is used."""


@dataclass(frozen=True, slots=True)
class CrossoverResult:
    """Period-adjusted comparison of arm A with arm B over crossover cycles.

    Attributes
    ----------
    cycles, ab, ba
        Number of cycles in total and in each order.
    difference
        Estimated success-rate difference A − B, adjusted for the period effect.
    interval
        Hills–Armitage t interval for the difference, or None with fewer than 3 cycles.
    period_effect
        Estimated change from the first to the second period of a cycle (scene drift,
        fatigue), the same for both arms.
    test
        Exact randomization test of no difference between the arms (or the t-test when the
        number of relabellings exceeds :data:`EXACT_LIMIT`).
    """

    cycles: int
    ab: int
    ba: int
    difference: float
    interval: Interval | None
    period_effect: float
    test: TestResult


def crossover_test(
    periods: Sequence[tuple[float, float]],
    orders: Sequence[Order],
    *,
    alternative: Alternative = "two-sided",
    level: float = 0.95,
) -> CrossoverResult:
    r"""Compare two arms run in crossover cycles of whole rounds.

    Parameters
    ----------
    periods
        Per cycle, the success proportion of the first and the second round.
    orders
        Per cycle, ``"AB"`` when arm A ran first, ``"BA"`` when arm B ran first.
    alternative
        ``"greater"`` means arm A has the higher success rate.
    level
        Confidence level of the interval.

    Notes
    -----
    Let :math:`u_c = y_{c1} - y_{c2}` be the period difference of cycle :math:`c`. Under
    the model :math:`y = \mu + \text{arm} + \text{period} + \varepsilon`,
    :math:`E[u_c] = (\tau_A - \tau_B) + (\pi_1 - \pi_2)` in ``AB`` cycles and
    :math:`-(\tau_A - \tau_B) + (\pi_1 - \pi_2)` in ``BA`` cycles. Hence

    .. math::

        \hat\tau = \tfrac12(\bar u_{AB} - \bar u_{BA}), \qquad
        \hat\pi = -\tfrac12(\bar u_{AB} + \bar u_{BA}), \qquad
        \mathrm{se}(\hat\tau) = \tfrac12 s_u \sqrt{1/m_{AB} + 1/m_{BA}},

    with :math:`s_u` the pooled within-order standard deviation of :math:`u` on
    :math:`m - 2` degrees of freedom. The test statistic :math:`\hat\tau` is referred to its
    exact randomization distribution: under no arm difference, :math:`u_c` does not depend
    on the order, so every assignment of :math:`m_{AB}` ``AB`` labels to the :math:`m`
    cycles is equally likely. A one-sided p-value is the share of assignments at least as
    extreme; the two-sided p-value doubles the smaller one-sided p-value (capped at 1).

    References
    ----------
    Hills, M. and Armitage, P. (1979). The two-period cross-over clinical trial. *British
    Journal of Clinical Pharmacology* 8, 7–20.

    Senn, S. (2002). *Cross-over Trials in Clinical Research*, 2nd ed., chapter 3. Wiley.
    """
    alternative = check_alternative(alternative)
    level = check_open_unit("level", level)
    if len(periods) != len(orders):
        raise ValueError("periods and orders must have the same length")
    if any(o not in ("AB", "BA") for o in orders):
        raise ValueError(f"orders must be 'AB' or 'BA', got {sorted(set(orders))!r}")
    y = np.array(periods, dtype=float).reshape(-1, 2) if periods else np.zeros((0, 2))
    if not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
        raise ValueError("period success proportions must lie in [0, 1]")
    is_ab = np.array([o == "AB" for o in orders], dtype=bool)
    m_ab, m_ba = int(is_ab.sum()), int((~is_ab).sum())
    if m_ab == 0 or m_ba == 0:
        raise ValueError("both orders (AB and BA) need at least one cycle")
    m = m_ab + m_ba
    u = y[:, 0] - y[:, 1]
    tau = 0.5 * (u[is_ab].mean() - u[~is_ab].mean())
    period = -0.5 * (u[is_ab].mean() + u[~is_ab].mean())

    interval: Interval | None = None
    se = math.nan
    if m >= 3:
        ss = ((u[is_ab] - u[is_ab].mean()) ** 2).sum() + ((u[~is_ab] - u[~is_ab].mean()) ** 2).sum()
        s2 = float(ss) / (m - 2)
        se = 0.5 * math.sqrt(s2 * (1.0 / m_ab + 1.0 / m_ba))
        q = float(stats.t.isf((1.0 - level) / 2.0, m - 2))
        interval = Interval(
            low=float(tau - q * se),
            high=float(tau + q * se),
            level=level,
            method="hills_armitage_t",
        )

    combos = math.comb(m, m_ab)
    if combos <= EXACT_LIMIT:
        pvalue = _exact_pvalue(u, m_ab, float(tau), alternative)
        name = "crossover_randomization"
    else:
        pvalue = _t_pvalue(float(tau), se, m - 2, alternative)
        name = "hills_armitage_t"
    return CrossoverResult(
        cycles=m,
        ab=m_ab,
        ba=m_ba,
        difference=float(tau),
        interval=interval,
        period_effect=float(period),
        test=TestResult(test=name, statistic=float(tau), pvalue=pvalue, alternative=alternative),
    )


def _exact_pvalue(u: np.ndarray, m_ab: int, tau: float, alternative: Alternative) -> float:
    m = len(u)
    total = float(u.sum())
    sums = np.array([u[list(c)].sum() for c in itertools.combinations(range(m), m_ab)], dtype=float)
    taus = 0.5 * (sums / m_ab - (total - sums) / (m - m_ab))
    eps = 1e-12 * max(1.0, abs(tau))
    greater = float((taus >= tau - eps).mean())
    less = float((taus <= tau + eps).mean())
    if alternative == "greater":
        return greater
    if alternative == "less":
        return less
    return min(1.0, 2.0 * min(greater, less))


def _t_pvalue(tau: float, se: float, df: int, alternative: Alternative) -> float:
    if not (se > 0):
        return 1.0 if tau == 0 else 0.0
    t = tau / se
    if alternative == "greater":
        return float(stats.t.sf(t, df))
    if alternative == "less":
        return float(stats.t.cdf(t, df))
    return float(min(1.0, 2.0 * stats.t.sf(abs(t), df)))
