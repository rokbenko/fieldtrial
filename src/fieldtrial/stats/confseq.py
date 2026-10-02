r"""Anytime-valid confidence sequences for bounded means, by betting.

A confidence sequence :math:`(L_t, U_t)` covers the mean :math:`\mu` of observations in
:math:`[0, 1]` at every sample size at once: :math:`P(\forall t: L_t \le \mu \le U_t) \ge
1 - \alpha`. So the data can be looked at after every observation, and the study can stop
whenever the sequence excludes a value, without inflating the error rate.

For each candidate mean :math:`m` a gambler bets against :math:`m` and grows the capital

.. math::

    K_t^{+}(m) = \prod_{i \le t} \bigl(1 + \lambda_i^{+} (x_i - m)\bigr), \qquad
    K_t^{-}(m) = \prod_{i \le t} \bigl(1 - \lambda_i^{-} (x_i - m)\bigr),

with bets :math:`\lambda_i` that depend only on :math:`x_1, \dots, x_{i-1}`. Under
:math:`\mu = m` both are nonnegative martingales, so by Ville's inequality the hedged
capital :math:`K_t = \max(\theta K_t^{+}, (1 - \theta) K_t^{-})` exceeds :math:`1/\alpha`
with probability at most :math:`\alpha`. The confidence sequence is the set of :math:`m`
whose capital stays at or below :math:`1/\alpha`, found on a grid.

The bets are the predictable-mixture empirical-Bernstein bets

.. math::

    \lambda_t = \sqrt{\frac{2 \log(1/\alpha')}{\hat\sigma_{t-1}^2\, t \log(1 + t)}},

with regularized running mean and variance (prior mean 1/2, prior variance 1/4, one
pseudo-observation), truncated at :math:`1/(2m)` for :math:`K^{+}` and :math:`1/(2(1-m))`
for :math:`K^{-}`. :math:`\alpha' = \theta\alpha` for :math:`K^{+}` and
:math:`(1-\theta)\alpha` for :math:`K^{-}`. This is the hedged capital process of
Waudby-Smith and Ramdas (2024), as implemented in ``confseq.betting.hedged_cs``.

References
----------
Waudby-Smith, I. and Ramdas, A. (2024). Estimating means of bounded random variables by
betting. *JRSS B* 86(1), 1–27.

Howard, S. R., Ramdas, A., McAuliffe, J. and Sekhon, J. (2021). Time-uniform,
nonparametric, nonasymptotic confidence sequences. *Annals of Statistics* 49(2),
1055–1080.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fieldtrial.stats._types import Alternative, Interval, TestResult
from fieldtrial.stats._validation import check_alternative, check_open_unit

BREAKS = 1000
"""Grid resolution for inverting the capital process: candidate means 0, 1/1000, ..., 1."""

_PRIOR_MEAN = 0.5
_PRIOR_VARIANCE = 0.25
_TRUNCATION = 0.5

Values = Sequence[float] | NDArray[np.float64]
"""Observations: a sequence of floats or a float array."""


@dataclass(frozen=True, slots=True)
class ConfidenceSequence:
    """A confidence sequence: one interval per sample size, all valid at once.

    ``lower[t - 1]`` and ``upper[t - 1]`` bound the mean after ``t`` observations.
    """

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    level: float
    method: str

    def __len__(self) -> int:
        return len(self.lower)

    def interval(self, t: int | None = None) -> Interval:
        """The interval after ``t`` observations (1-based; the latest when omitted)."""
        i = len(self.lower) if t is None else t
        if not 1 <= i <= len(self.lower):
            raise ValueError(f"t must be between 1 and {len(self.lower)}, got {t}")
        return Interval(self.lower[i - 1], self.upper[i - 1], self.level, self.method)


@dataclass(frozen=True, slots=True)
class AnytimeTestResult:
    """Anytime-valid test of a mean difference from paired observations.

    Attributes
    ----------
    n
        Number of paired observations.
    estimate
        Mean paired difference.
    sequence
        Confidence sequence for the mean difference (on the difference scale).
    rejected_at
        First observation (1-based) after which the null was rejected, or None.
    test
        ``statistic`` is the largest capital reached against the null and ``pvalue`` the
        anytime-valid p-value ``min(1, 1 / max capital)``.
    """

    n: int
    estimate: float
    sequence: ConfidenceSequence
    rejected_at: int | None
    test: TestResult


def _theta(alternative: Alternative) -> float:
    return {"two-sided": 0.5, "greater": 1.0, "less": 0.0}[alternative]


def _as_unit(x: Values, name: str = "x") -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0) or np.any(arr > 1):
        raise ValueError(f"{name} must contain values between 0 and 1")
    return arr


def predmix_bets(x: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    r"""Predictable-mixture empirical-Bernstein bets :math:`\lambda_1, \dots, \lambda_n`.

    Bet :math:`t` uses only :math:`x_1, \dots, x_{t-1}`.
    """
    n = len(x)
    t = np.arange(1, n + 1, dtype=np.float64)
    mu = np.minimum((_PRIOR_MEAN + np.cumsum(x)) / (t + 1), 1.0)
    sigma2 = (_PRIOR_VARIANCE + np.cumsum((x - mu) ** 2)) / (t + 1)
    sigma2_prev = np.append(_PRIOR_VARIANCE, sigma2[: n - 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        bets: NDArray[np.float64] = np.sqrt(
            2.0 * math.log(1.0 / alpha) / (t * np.log1p(t) * sigma2_prev)
        )
    bets[np.isnan(bets)] = 0.0
    return bets


def _side(
    x: NDArray[np.float64], m: NDArray[np.float64], bets: NDArray[np.float64], sign: float
) -> NDArray[np.float64]:
    """Capital of one side for each candidate mean (rows) after each observation (columns)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cap = _TRUNCATION / m if sign > 0 else _TRUNCATION / (1.0 - m)
        lam = np.minimum(bets[None, :], cap[:, None])
        step = x[None, :] - m[:, None]
        factor = 1.0 + sign * lam * step
        factor[np.logical_and(np.isinf(lam), step == 0)] = 1.0
        capital = np.cumprod(factor, axis=1)
    capital[np.isnan(capital)] = 0.0
    return capital


def capital_process(
    x: Values,
    null: float | Values,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> NDArray[np.float64]:
    """Hedged capital against each candidate mean after each observation.

    Returns an array of shape ``(len(null), len(x))`` (a 1-d array for a scalar ``null``).
    Values above ``1 / alpha`` reject that mean.
    """
    data = _as_unit(x)
    a = check_open_unit("alpha", alpha)
    theta = _theta(check_alternative(alternative))
    scalar = np.ndim(null) == 0
    m = np.atleast_1d(np.asarray(null, dtype=np.float64))
    if np.any(m < 0) or np.any(m > 1):
        raise ValueError("null must lie between 0 and 1")
    zeros = np.zeros((len(m), len(data)))
    plus = _side(data, m, predmix_bets(data, a * theta), 1.0) if theta > 0 else zeros
    minus = _side(data, m, predmix_bets(data, a * (1 - theta)), -1.0) if theta < 1 else zeros
    if theta == 1.0:
        capital = plus
    elif theta == 0.0:
        capital = minus
    else:
        capital = np.maximum(theta * plus, (1 - theta) * minus)
    return capital[0] if scalar else capital


def betting_cs(
    x: Values,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    *,
    breaks: int = BREAKS,
) -> ConfidenceSequence:
    """Confidence sequence for the mean of observations in [0, 1] (hedged capital, betting).

    ``alternative="greater"`` gives a lower confidence sequence (the upper bound is 1) at
    level ``1 - alpha``, ``"less"`` an upper one; ``"two-sided"`` splits ``alpha`` evenly.
    Intervals are padded outwards by one grid step and intersected over time.
    """
    data = _as_unit(x)
    if not data.size:
        raise ValueError("x must contain at least one value")
    if breaks < 10:
        raise ValueError("breaks must be at least 10")
    grid = np.arange(0, 1 + 1 / breaks, step=1 / breaks)
    inside = capital_process(data, grid, alpha, alternative) <= 1.0 / alpha
    # The smallest and largest candidate means still in the sequence; (0, 1) if none is.
    found = inside.any(axis=0)
    first = inside.argmax(axis=0)
    last = len(grid) - 1 - inside[::-1].argmax(axis=0)
    lower = np.where(found, grid[first], 0.0)
    upper = np.where(found, grid[last], 1.0)
    lower = np.maximum.accumulate(np.maximum(0.0, lower - 1 / breaks))
    upper = np.minimum.accumulate(np.minimum(1.0, upper + 1 / breaks))
    return ConfidenceSequence(
        lower=tuple(float(v) for v in lower),
        upper=tuple(float(v) for v in upper),
        level=1.0 - alpha,
        method="betting",
    )


def paired_anytime_test(
    differences: Values,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    null: float = 0.0,
    *,
    breaks: int = BREAKS,
) -> AnytimeTestResult:
    r"""Anytime-valid test of the mean paired difference, for differences in [-1, 1].

    Each difference (for example treatment minus control success in one block) is mapped to
    :math:`(d + 1)/2 \in [0, 1]` and tested with :func:`capital_process`. The null is
    :math:`\Delta = \text{null}` (two-sided), :math:`\Delta \le \text{null}` (``"greater"``)
    or :math:`\Delta \ge \text{null}` (``"less"``); a nonzero ``null`` gives a
    non-inferiority or superiority-by-margin test. The study may stop at the first
    rejection; the error rate stays at most ``alpha`` however often it looks.
    """
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 1 or not d.size:
        raise ValueError("differences must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(d)) or np.any(np.abs(d) > 1):
        raise ValueError("differences must lie between -1 and 1")
    if not -1.0 < null < 1.0:
        raise ValueError(f"null must lie strictly between -1 and 1, got {null}")
    alt = check_alternative(alternative)
    z = (d + 1.0) / 2.0
    capital = capital_process(z, (null + 1.0) / 2.0, alpha, alt)
    running = np.maximum.accumulate(capital)
    crossed = np.flatnonzero(running > 1.0 / alpha)
    cs = betting_cs(z, alpha, alt, breaks=breaks)
    sequence = ConfidenceSequence(
        lower=tuple(2 * v - 1 for v in cs.lower),
        upper=tuple(2 * v - 1 for v in cs.upper),
        level=cs.level,
        method="betting (paired)",
    )
    peak = float(running[-1])
    return AnytimeTestResult(
        n=len(d),
        estimate=float(d.mean()),
        sequence=sequence,
        rejected_at=int(crossed[0]) + 1 if crossed.size else None,
        test=TestResult(
            test="betting_paired",
            statistic=peak,
            pvalue=min(1.0, 1.0 / peak) if peak > 0 else 1.0,
            alternative=alt,
        ),
    )
