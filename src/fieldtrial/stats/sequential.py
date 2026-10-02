r"""Group-sequential designs: planned interim looks with error-spending boundaries.

A group-sequential study looks at the data K times. At each look it compares the
standardized statistic :math:`Z_k` with a boundary :math:`b_k` and stops if the boundary is
crossed. The boundaries are chosen so that the probability of crossing any of them when the
arms do not differ is the planned :math:`\alpha`.

Crossing probabilities come from the joint distribution of :math:`(Z_1, \dots, Z_K)`. With
information fractions :math:`t_1 < \dots < t_K`, the score process
:math:`S_k = Z_k \sqrt{t_k}` has independent normal increments with variance
:math:`t_k - t_{k-1}` (Jennison and Turnbull, 2000, section 3.1). The density of
:math:`Z_k` on the continuation region is computed by the recursive numerical integration
of Armitage, McPherson and Rowe (1969), using Simpson's rule on a fine grid.

References
----------
Armitage, P., McPherson, C. K. and Rowe, B. C. (1969). Repeated significance tests on
accumulating data. *JRSS A* 132, 235–244.

Lan, K. K. G. and DeMets, D. L. (1983). Discrete sequential boundaries for clinical trials.
*Biometrika* 70, 659–663.

Jennison, C. and Turnbull, B. W. (2000). *Group Sequential Methods with Applications to
Clinical Trials*. Chapman & Hall/CRC.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, special, stats

from fieldtrial.stats._types import Alternative, Interval, TestResult
from fieldtrial.stats._validation import check_alternative, check_choice, check_open_unit

Spending = Literal["obrien_fleming", "pocock"]
SPENDING_FUNCTIONS: tuple[Spending, ...] = ("obrien_fleming", "pocock")

_GRID = 241  # Simpson points per look, odd: boundaries are stable to 1e-6 from 161 points up
_TAIL = 9.0  # one-sided designs: the lower end of the continuation region, in sd units
_MAX_Z = 12.0


@dataclass(frozen=True, slots=True)
class SequentialDesign:
    r"""Boundaries of a group-sequential design.

    Attributes
    ----------
    information
        Information fraction of each look, increasing, the last at most 1.
    boundaries
        Critical value of each look on the z scale. A two-sided design stops when
        :math:`|Z_k| \ge b_k`; a one-sided design when :math:`Z_k \ge b_k` (``"greater"``)
        or :math:`Z_k \le -b_k` (``"less"``).
    alpha_spent
        Cumulative type I error spent up to each look.
    alpha
        Overall type I error.
    method
        ``"obrien_fleming"`` or ``"pocock"`` spending, or ``"pocock_constant"`` /
        ``"obrien_fleming_constant"`` for the classical boundaries.
    alternative
        Direction of the test.
    """

    information: tuple[float, ...]
    boundaries: tuple[float, ...]
    alpha_spent: tuple[float, ...]
    alpha: float
    method: str
    alternative: Alternative

    @property
    def looks(self) -> int:
        """Number of planned looks."""
        return len(self.boundaries)


@dataclass(frozen=True, slots=True)
class SequentialResult:
    """Outcome of a group-sequential test on the looks seen so far.

    ``stopped_at`` is the 1-based look at which a boundary was crossed, or None.
    ``test`` holds the statistic of the deciding look (the crossing look, or the latest
    one) and the stage-wise ordering p-value, which is valid only once the study has
    stopped or reached its final look (``final`` is then True); before that it is NaN.
    """

    stopped_at: int | None
    final: bool
    z: tuple[float, ...]
    boundaries: tuple[float, ...]
    test: TestResult


def alpha_spending(t: float, alpha: float, spending: Spending) -> float:
    r"""Cumulative error spent at information fraction ``t``.

    Notes
    -----
    Lan–DeMets O'Brien–Fleming type:
    :math:`\alpha(t) = 2 - 2\Phi\!\left(\Phi^{-1}(1 - \alpha/2)/\sqrt{t}\right)`.

    Lan–DeMets Pocock type: :math:`\alpha(t) = \alpha \ln(1 + (e - 1)t)`.

    Both spend :math:`\alpha` at :math:`t = 1`. For a two-sided symmetric design each tail
    spends :math:`\alpha/2` with these functions (Jennison and Turnbull, 2000, section 7.2).
    """
    check_choice("spending", spending, SPENDING_FUNCTIONS)
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"information fraction must be between 0 and 1, got {t!r}")
    if t == 0.0:
        return 0.0
    if spending == "obrien_fleming":
        return float(2.0 * stats.norm.sf(stats.norm.isf(alpha / 2.0) / math.sqrt(t)))
    return float(alpha * math.log1p((math.e - 1.0) * t))


# --- numerical integration --------------------------------------------------------------


_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.exp(-0.5 * x * x) * _INV_SQRT_2PI, dtype=np.float64)


def _sf(x: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return np.asarray(special.ndtr(-np.asarray(x)), dtype=np.float64)


def _simpson(lo: float, hi: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Grid points and Simpson weights on [lo, hi]."""
    m = _GRID
    z = np.linspace(lo, hi, m)
    w = np.ones(m)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return z, w * (hi - lo) / (3.0 * (m - 1))


class _Recursion:
    """Sub-density of Z_k on the continuation region, look by look.

    ``drift`` is the mean of Z at full information (t = 1); the mean of Z_k is
    ``drift * sqrt(t_k)``.
    """

    def __init__(self, information: Sequence[float], drift: float) -> None:
        self.t = [float(x) for x in information]
        self.drift = drift
        self.k = 0
        self.z: NDArray[np.float64] = np.zeros(0)
        self.w: NDArray[np.float64] = np.zeros(0)
        self.h: NDArray[np.float64] = np.zeros(0)

    def _conditional(self, k: int) -> tuple[NDArray[np.float64], float]:
        """Mean (per grid point of look k-1) and sd of Z_k given Z_{k-1}."""
        t0, t1 = self.t[k - 1], self.t[k]
        dt = t1 - t0
        mean = (self.z * math.sqrt(t0) + self.drift * dt) / math.sqrt(t1)
        return mean, math.sqrt(dt / t1)

    def exit_probability(self, upper: float, lower: float) -> tuple[float, float]:
        """P(no crossing so far, then Z_k >= upper) and (..., Z_k <= lower) at the next look."""
        k = self.k
        if k == 0:
            mu = self.drift * math.sqrt(self.t[0])
            return float(_sf(upper - mu)), float(_sf(mu - lower))
        mean, sd = self._conditional(k)
        wh = self.w * self.h
        up = float(wh @ _sf((upper - mean) / sd))
        low = float(wh @ _sf((mean - lower) / sd))
        return up, low

    def advance(self, upper: float, lower: float) -> None:
        """Move to the next look with continuation region (lower, upper)."""
        k = self.k
        z, w = _simpson(lower, upper)
        if k == 0:
            h = _pdf(z - self.drift * math.sqrt(self.t[0]))
        else:
            mean, sd = self._conditional(k)
            dens = _pdf((z[:, None] - mean[None, :]) / sd) / sd
            h = dens @ (self.w * self.h)
        self.z, self.w, self.h = z, w, np.asarray(h, dtype=np.float64)
        self.k = k + 1


def _lower_end(drift: float, t: float) -> float:
    return drift * math.sqrt(t) - _TAIL


def _region(b: float, alternative: Alternative, drift: float, t: float) -> tuple[float, float]:
    """Continuation region (lower, upper) on the z scale for a boundary b."""
    if alternative == "two-sided":
        return -b, b
    # One-sided designs are computed for "greater"; "less" is the mirror image.
    return _lower_end(drift, t), b


def crossing_probabilities(
    boundaries: Sequence[float],
    information: Sequence[float],
    *,
    alternative: Alternative = "two-sided",
    drift: float = 0.0,
) -> tuple[float, ...]:
    """Probability of first crossing the boundary at each look.

    ``drift`` is the mean of the z statistic at full information (0 under the null
    hypothesis). For ``"less"`` designs pass the drift of ``-Z``. The sum of the returned
    probabilities is the type I error (drift 0) or the power.
    """
    alternative = check_alternative(alternative)
    info = _check_information(information)
    if len(boundaries) != len(info):
        raise ValueError("boundaries and information must have the same length")
    rec = _Recursion(info, drift)
    probs = []
    for k, b in enumerate(boundaries):
        lo, hi = _region(float(b), alternative, drift, info[k])
        up, low = rec.exit_probability(hi, lo)
        probs.append(up + low if alternative == "two-sided" else up)
        if k < len(boundaries) - 1:
            rec.advance(hi, lo)
    return tuple(probs)


def _check_information(information: Sequence[float]) -> list[float]:
    info = [float(x) for x in information]
    if not info:
        raise ValueError("at least one look is needed")
    prev = 0.0
    for t in info:
        if not (math.isfinite(t) and prev < t <= 1.0):
            raise ValueError(
                "information fractions must be strictly increasing and in (0, 1], "
                f"got {list(information)!r}"
            )
        prev = t
    return info


def _solve(target: float, exit_prob: "_ExitFn") -> float:
    """Boundary b with exit_prob(b) == target; exit_prob decreases in b."""
    if target <= exit_prob(_MAX_Z):
        return _MAX_Z
    lo = 0.0
    if exit_prob(lo) <= target:
        return lo
    return float(optimize.brentq(lambda b: exit_prob(b) - target, lo, _MAX_Z, xtol=1e-10))


class _ExitFn:
    def __init__(self, rec: _Recursion, alternative: Alternative, t: float) -> None:
        self.rec, self.alternative, self.t = rec, alternative, t

    def __call__(self, b: float) -> float:
        lo, hi = _region(b, self.alternative, self.rec.drift, self.t)
        up, low = self.rec.exit_probability(hi, lo)
        return up + low if self.alternative == "two-sided" else up


def spending_boundaries(
    information: Sequence[float],
    *,
    alpha: float = 0.05,
    spending: Spending = "obrien_fleming",
    alternative: Alternative = "two-sided",
    spend_all: bool = False,
) -> SequentialDesign:
    r"""Lan–DeMets error-spending boundaries for looks at the given information fractions.

    Parameters
    ----------
    information
        Information fraction of each look, strictly increasing in (0, 1]. For trial counts,
        ``n_k / n_max``.
    alpha
        Overall type I error (two-sided when ``alternative`` is ``"two-sided"``).
    spending
        ``"obrien_fleming"`` (conservative early, close to a fixed design at the end) or
        ``"pocock"`` (roughly equal boundaries).
    alternative
        ``"two-sided"`` for symmetric boundaries, or a one-sided direction.
    spend_all
        Spend all remaining error at the last look even if its fraction is below 1, as
        when a study ends with fewer trials than planned.

    Notes
    -----
    Look k spends :math:`\alpha(t_k) - \alpha(t_{k-1})`. Its boundary :math:`b_k` solves

    .. math::

        P_0(\text{no crossing before } k,\ \text{crossing at } k) =
        \alpha(t_k) - \alpha(t_{k-1}),

    with :math:`\alpha(\cdot)` from :func:`alpha_spending`. A two-sided design splits the
    error equally between the tails.

    References
    ----------
    Lan, K. K. G. and DeMets, D. L. (1983). *Biometrika* 70, 659–663.

    Jennison, C. and Turnbull, B. W. (2000), chapter 7.
    """
    alpha = check_open_unit("alpha", alpha)
    alternative = check_alternative(alternative)
    check_choice("spending", spending, SPENDING_FUNCTIONS)
    info = _check_information(information)

    tail_alpha = alpha / 2.0 if alternative == "two-sided" else alpha
    factor = 2.0 if alternative == "two-sided" else 1.0
    rec = _Recursion(info, 0.0)
    bounds: list[float] = []
    spent: list[float] = []
    prev = 0.0
    for k, t in enumerate(info):
        last = k == len(info) - 1
        cum = alpha if (last and spend_all) else factor * alpha_spending(t, tail_alpha, spending)
        b = _solve(max(cum - prev, 0.0), _ExitFn(rec, alternative, t))
        bounds.append(b)
        spent.append(cum)
        prev = cum
        if not last:
            lo, hi = _region(b, alternative, 0.0, t)
            rec.advance(hi, lo)
    return SequentialDesign(
        information=tuple(info),
        boundaries=tuple(bounds),
        alpha_spent=tuple(spent),
        alpha=alpha,
        method=spending,
        alternative=alternative,
    )


def constant_boundaries(
    looks: int,
    *,
    alpha: float = 0.05,
    shape: Literal["pocock", "obrien_fleming"] = "pocock",
    alternative: Alternative = "two-sided",
) -> SequentialDesign:
    r"""Classical Pocock or O'Brien–Fleming boundaries for equally spaced looks.

    Pocock uses the same critical value :math:`c` at every look; O'Brien–Fleming uses
    :math:`b_k = c\sqrt{K/k}`. :math:`c` is chosen so that the overall type I error is
    ``alpha``. These are the tabulated designs in Jennison and Turnbull (2000), table 2.1
    (Pocock) and table 2.3 (O'Brien–Fleming). Error-spending boundaries
    (:func:`spending_boundaries`) are preferred in practice because they allow unequal
    and unplanned look times.

    References
    ----------
    Pocock, S. J. (1977). *Biometrika* 64, 191–199.

    O'Brien, P. C. and Fleming, T. R. (1979). *Biometrics* 35, 549–556.
    """
    alpha = check_open_unit("alpha", alpha)
    alternative = check_alternative(alternative)
    check_choice("shape", shape, ("pocock", "obrien_fleming"))
    if looks < 1:
        raise ValueError(f"looks must be at least 1, got {looks}")
    info = [k / looks for k in range(1, looks + 1)]

    def shape_of(c: float) -> list[float]:
        if shape == "pocock":
            return [c] * looks
        return [c * math.sqrt(looks / k) for k in range(1, looks + 1)]

    def total(c: float) -> float:
        return sum(crossing_probabilities(shape_of(c), info, alternative=alternative)) - alpha

    c = float(optimize.brentq(total, 0.5, 6.0, xtol=1e-10))
    probs = crossing_probabilities(shape_of(c), info, alternative=alternative)
    return SequentialDesign(
        information=tuple(info),
        boundaries=tuple(shape_of(c)),
        alpha_spent=tuple(float(x) for x in np.cumsum(probs)),
        alpha=alpha,
        method=f"{shape}_constant",
        alternative=alternative,
    )


def sequential_test(z: Sequence[float], design: SequentialDesign) -> SequentialResult:
    r"""Apply a group-sequential design to the z statistics observed at each look so far.

    The p-value uses the stage-wise ordering (Fairbanks and Madsen, 1982; Jennison and
    Turnbull, 2000, section 8.4): an outcome stopping at an earlier look is more extreme
    than any outcome stopping later, and within a look a larger :math:`|Z|` is more
    extreme. Stopping at look :math:`k^*` with statistic :math:`z^*` gives

    .. math::

        p = \sum_{j < k^*} P_0(\text{cross at } j) +
        P_0(\text{no crossing before } k^*,\ |Z_{k^*}| \ge |z^*|).

    The p-value is at most :math:`\alpha` exactly when the design rejects. It is defined
    only when the study has stopped or reached its last look; otherwise it is NaN.

    References
    ----------
    Fairbanks, K. and Madsen, R. (1982). P values for tests using a repeated significance
    test design. *Biometrika* 69, 69–74.
    """
    if len(z) == 0:
        raise ValueError("at least one look must have been observed")
    if len(z) > design.looks:
        raise ValueError(f"{len(z)} looks observed but the design plans {design.looks}")
    zs = tuple(float(x) for x in z)
    if not all(math.isfinite(x) for x in zs):
        raise ValueError("z statistics must be finite")
    alt = design.alternative
    oriented = [-x if alt == "less" else x for x in zs]
    stat_alt: Alternative = "two-sided" if alt == "two-sided" else "greater"

    def crossed(k: int) -> bool:
        x = oriented[k]
        return abs(x) >= design.boundaries[k] if alt == "two-sided" else x >= design.boundaries[k]

    stop = next((k for k in range(len(zs)) if crossed(k)), None)
    final = stop is not None or len(zs) == design.looks
    k_star = stop if stop is not None else len(zs) - 1
    pvalue = math.nan
    if final:
        x = oriented[k_star]
        b_star = abs(x) if alt == "two-sided" else x
        bounds = [*design.boundaries[:k_star], b_star]
        info = design.information[: k_star + 1]
        if stat_alt == "greater" and b_star < _lower_end(0.0, info[-1]):
            pvalue = 1.0
        else:
            pvalue = min(1.0, sum(crossing_probabilities(bounds, info, alternative=stat_alt)))
    return SequentialResult(
        stopped_at=None if stop is None else stop + 1,
        final=final,
        z=zs,
        boundaries=design.boundaries[: len(zs)],
        test=TestResult(
            test=f"group_sequential_{design.method}",
            statistic=zs[k_star],
            pvalue=pvalue,
            alternative=alt,
        ),
    )


def repeated_interval(
    estimate: float, standard_error: float, boundary: float, *, level: float
) -> Interval:
    r"""Repeated confidence interval at a look: :math:`\hat\theta_k \pm b_k\,\mathrm{se}_k`.

    Using the design's two-sided boundaries as multipliers, the intervals at all looks
    cover the true difference simultaneously with probability ``level`` (Jennison and
    Turnbull, 1989), so they stay valid whenever the study stops.

    References
    ----------
    Jennison, C. and Turnbull, B. W. (1989). Interim analyses: the repeated confidence
    interval approach. *JRSS B* 51, 305–361.
    """
    level = check_open_unit("level", level)
    if not (math.isfinite(standard_error) and standard_error >= 0.0):
        raise ValueError(f"standard_error must be finite and non-negative, got {standard_error!r}")
    half = boundary * standard_error
    return Interval(low=estimate - half, high=estimate + half, level=level, method="repeated_ci")
