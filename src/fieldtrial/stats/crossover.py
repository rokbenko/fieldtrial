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
        Interval for the difference that inverts the randomization test (or the
        Hills–Armitage t interval above :data:`EXACT_LIMIT` relabellings).
    standard_error
        Hills–Armitage standard error of the difference (None with fewer than 3 cycles).
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
    standard_error: float | None = None


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

    The confidence interval inverts the same randomization test, so it excludes 0 exactly
    when the two-sided test rejects at :math:`1 - \text{level}`. Under
    :math:`H_\delta: \tau_A - \tau_B = \delta`, the shifted differences
    :math:`u_c - \delta s_c` (with :math:`s_c = +1` for ``AB`` and :math:`-1` for ``BA``)
    do not depend on the order, so the test applies to them; the interval collects the
    :math:`\delta` it does not reject, with its ends found by bisection. Above
    :data:`EXACT_LIMIT` relabellings, the test and interval use the t distribution with
    the standard error above on :math:`m - 2` degrees of freedom.

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

    combos = math.comb(m, m_ab)
    if combos <= EXACT_LIMIT:
        rand = _Randomization(u, is_ab)
        pvalue = rand.pvalue(0.0, alternative)
        name = "crossover_randomization"
        bounds = rand.interval(float(tau), 1.0 - level)
        if bounds is not None:
            interval = Interval(
                low=bounds[0], high=bounds[1], level=level, method="crossover_randomization"
            )
    else:
        pvalue = _t_pvalue(float(tau), se, m - 2, alternative)
        name = "hills_armitage_t"
        if m >= 3:
            q = float(stats.t.isf((1.0 - level) / 2.0, m - 2))
            interval = Interval(
                low=float(tau - q * se),
                high=float(tau + q * se),
                level=level,
                method="hills_armitage_t",
            )
    return CrossoverResult(
        cycles=m,
        ab=m_ab,
        ba=m_ba,
        difference=float(tau),
        interval=interval,
        period_effect=float(period),
        test=TestResult(test=name, statistic=float(tau), pvalue=pvalue, alternative=alternative),
        standard_error=None if math.isnan(se) else se,
    )


class _Randomization:
    r"""Every relabelling of the cycles with the observed number of AB orders.

    For assignment g, ``a[g]`` is the estimate computed from the period differences ``u``
    and ``b[g]`` the estimate computed from the order signs ``s`` (+1 AB, -1 BA). Under
    :math:`H_\delta`, the statistic of assignment g is ``a[g] - delta * b[g]``, and the
    observed one is ``tau - delta`` (``b`` is 1 for the observed labels).
    """

    def __init__(self, u: np.ndarray, is_ab: np.ndarray) -> None:
        m = len(u)
        m_ab = int(is_ab.sum())
        rows = np.zeros((math.comb(m, m_ab), m), dtype=bool)
        for g, chosen in enumerate(itertools.combinations(range(m), m_ab)):
            rows[g, list(chosen)] = True
        signs = np.where(is_ab, 1.0, -1.0)
        self.a = self._estimates(rows, u, m_ab)
        self.b = self._estimates(rows, signs, m_ab)
        self.tau = float(self._estimates(is_ab[None, :], u, m_ab)[0])

    @staticmethod
    def _estimates(rows: np.ndarray, x: np.ndarray, m_ab: int) -> np.ndarray:
        m_ba = rows.shape[1] - m_ab
        sums = np.asarray(rows.astype(float) @ x, dtype=float)
        out: np.ndarray = 0.5 * (sums / m_ab - (float(x.sum()) - sums) / m_ba)
        return out

    def pvalue(self, delta: float, alternative: Alternative) -> float:
        null = self.a - delta * self.b
        observed = self.tau - delta
        eps = 1e-12 * max(1.0, abs(observed), float(np.abs(null).max(initial=0.0)))
        greater = float((null >= observed - eps).mean())
        less = float((null <= observed + eps).mean())
        if alternative == "greater":
            return greater
        if alternative == "less":
            return less
        return min(1.0, 2.0 * min(greater, less))

    def interval(self, tau: float, alpha: float) -> tuple[float, float] | None:
        """Shifts not rejected by the two-sided test at ``alpha``.

        None when even the estimate is rejected, which happens only with ties.
        """
        if self.pvalue(tau, "two-sided") <= alpha:
            return None

        def edge(outer: float) -> float:
            if self.pvalue(outer, "two-sided") > alpha:
                return outer
            inside, out = tau, outer
            for _ in range(60):
                mid = 0.5 * (inside + out)
                if self.pvalue(mid, "two-sided") > alpha:
                    inside = mid
                else:
                    out = mid
            return inside

        return edge(-1.0), edge(1.0)


def _t_pvalue(tau: float, se: float, df: int, alternative: Alternative) -> float:
    if not (se > 0):
        return 1.0 if tau == 0 else 0.0
    t = tau / se
    if alternative == "greater":
        return float(stats.t.sf(t, df))
    if alternative == "less":
        return float(stats.t.cdf(t, df))
    return float(min(1.0, 2.0 * stats.t.sf(abs(t), df)))
