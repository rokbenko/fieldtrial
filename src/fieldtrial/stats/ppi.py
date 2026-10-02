r"""Prediction-powered inference (PPI++) for a success rate.

A proxy, for example a reward model, scores every episode: :math:`f(X)`. Humans label a
random subset of :math:`n` episodes (:math:`Y`, with their proxy scores :math:`f(X_i)`); the
other :math:`N` episodes have proxy scores :math:`f(\tilde X_j)` only. The power-tuned
estimate of the mean of :math:`Y` is

.. math::

    \hat\theta_\lambda = \lambda \,\overline{f(\tilde X)} + \overline{Y - \lambda f(X)},

with standard error

.. math::

    \widehat{\mathrm{se}} = \sqrt{\frac{\widehat{\operatorname{Var}}(\lambda f(\tilde X))}{N}
    + \frac{\widehat{\operatorname{Var}}(Y - \lambda f(X))}{n}}

(population variances) and the Wald interval :math:`\hat\theta_\lambda \pm z\,
\widehat{\mathrm{se}}`. :math:`\lambda = 0` is the classical estimate from the labeled
episodes alone; :math:`\lambda = 1` is the original PPI. The tuned value

.. math::

    \hat\lambda = \frac{\widehat{\operatorname{Cov}}(Y, f(X))}
    {(1 + n/N)\,\widehat{\operatorname{Var}}(f)}

(covariance over the labeled episodes, variance of the scores of all episodes), clipped to
[0, 1], minimizes the variance, so a poor proxy costs nothing beyond the classical interval
asymptotically. The interval is valid only when the labeled episodes are a random subset
of the scored ones. This follows ``ppi_py.ppi_mean_ci`` (ppi-python 0.2.3).

References
----------
Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I. and Zrnic, T. (2023).
Prediction-powered inference. *Science* 382(6671), 669–674.

Angelopoulos, A. N., Duchi, J. C. and Zrnic, T. (2023). PPI++: Efficient prediction-powered
inference. arXiv:2311.01453.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import special

from fieldtrial.stats._types import Alternative, Interval
from fieldtrial.stats._validation import check_alternative, check_open_unit


@dataclass(frozen=True, slots=True)
class PPIResult:
    r"""A prediction-powered estimate of a mean.

    Attributes
    ----------
    labeled, unlabeled
        Number of labeled and of proxy-only episodes.
    lam
        The power-tuning parameter used (0: classical, 1: plain PPI).
    estimate
        :math:`\hat\theta_\lambda`.
    standard_error
        Its standard error.
    interval
        Wald interval at the requested level and alternative.
    classical
        The classical interval from the labeled episodes alone (:math:`\lambda = 0`), for
        comparison.
    """

    labeled: int
    unlabeled: int
    lam: float
    estimate: float
    standard_error: float
    interval: Interval
    classical: Interval


@dataclass(frozen=True, slots=True)
class PPIDifference:
    """Difference of two independent prediction-powered estimates, ``first - second``."""

    estimate: float
    standard_error: float
    interval: Interval
    first: PPIResult
    second: PPIResult


Values = Sequence[float] | NDArray[np.float64]
"""Labels or scores: a sequence of floats or a float array."""


def _array(values: Values, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _wald(
    estimate: float, se: float, level: float, alternative: Alternative, method: str
) -> Interval:
    if alternative == "two-sided":
        z = float(special.ndtri(1 - (1 - level) / 2))
        return Interval(estimate - z * se, estimate + z * se, level, method)
    z = float(special.ndtri(level))
    if alternative == "greater":  # a lower bound
        return Interval(estimate - z * se, math.inf, level, method)
    return Interval(-math.inf, estimate + z * se, level, method)


def tuned_lambda(y: Values, f: Values, f_unlabeled: Values) -> float:
    r"""The variance-minimizing :math:`\lambda`, clipped to [0, 1].

    Returns 0 when the scores are constant.
    """
    yy, ff, fu = _array(y, "y"), _array(f, "f"), _array(f_unlabeled, "f_unlabeled")
    n, big_n = len(yy), len(fu)
    cov = float(np.mean((yy - yy.mean()) * (ff - ff.mean())))
    pooled = np.concatenate([ff, fu])
    var = float(np.var(pooled, ddof=1)) if len(pooled) > 1 else 0.0
    if var <= 0:
        return 0.0
    return min(max(cov / ((1 + n / big_n) * var), 0.0), 1.0)


def ppi_mean(
    y: Values,
    f: Values,
    f_unlabeled: Values,
    level: float = 0.95,
    alternative: Alternative = "two-sided",
    lam: float | None = None,
) -> PPIResult:
    """PPI++ estimate and interval for the mean of ``y``.

    ``y`` are the human labels (1 success, 0 failure) and ``f`` the proxy scores of the same
    episodes; ``f_unlabeled`` are the proxy scores of the episodes nobody labeled. With
    ``lam=None`` the power-tuning parameter is estimated (:func:`tuned_lambda`).
    """
    yy, ff, fu = _array(y, "y"), _array(f, "f"), _array(f_unlabeled, "f_unlabeled")
    if len(yy) != len(ff):
        raise ValueError("y and f must have the same length")
    if len(yy) < 2:
        raise ValueError("at least 2 labeled episodes are needed")
    if len(fu) < 1:
        raise ValueError("at least 1 unlabeled episode is needed")
    lvl = check_open_unit("level", level)
    alt = check_alternative(alternative)
    lam_used = tuned_lambda(yy, ff, fu) if lam is None else float(lam)
    if not 0.0 <= lam_used <= 1.0:
        raise ValueError(f"lam must lie between 0 and 1, got {lam}")
    n, big_n = len(yy), len(fu)
    estimate = float(lam_used * fu.mean() + (yy - lam_used * ff).mean())
    se = math.sqrt(float(np.var(lam_used * fu)) / big_n + float(np.var(yy - lam_used * ff)) / n)
    classical_se = float(np.std(yy)) / math.sqrt(n)
    return PPIResult(
        labeled=n,
        unlabeled=big_n,
        lam=lam_used,
        estimate=estimate,
        standard_error=se,
        interval=_wald(estimate, se, lvl, alt, "ppi++"),
        classical=_wald(float(yy.mean()), classical_se, lvl, alt, "wald"),
    )


def ppi_difference(
    first: PPIResult,
    second: PPIResult,
    level: float = 0.95,
    alternative: Alternative = "two-sided",
) -> PPIDifference:
    """``first - second`` for estimates from independent episodes (for example two arms)."""
    lvl = check_open_unit("level", level)
    alt = check_alternative(alternative)
    estimate = first.estimate - second.estimate
    se = math.hypot(first.standard_error, second.standard_error)
    return PPIDifference(
        estimate=estimate,
        standard_error=se,
        interval=_wald(estimate, se, lvl, alt, "ppi++ difference"),
        first=first,
        second=second,
    )
