"""Bayesian summaries. Descriptive only: never used as the primary test of a study."""

import math

from scipy import integrate, stats

from fieldtrial.stats._types import Interval
from fieldtrial.stats._validation import check_counts, check_open_unit


def _check_prior(prior: tuple[float, float]) -> tuple[float, float]:
    a, b = (float(x) for x in prior)
    if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
        raise ValueError(f"prior parameters must be positive numbers, got {prior!r}")
    return a, b


def prob_superiority(
    ka: int,
    na: int,
    kb: int,
    nb: int,
    *,
    prior: tuple[float, float] = (1.0, 1.0),
) -> float:
    r"""Posterior probability that arm B's success rate exceeds arm A's, :math:`P(p_B > p_A)`.

    With independent :math:`\text{Beta}(a_0, b_0)` priors the posteriors are
    :math:`p_A \sim \text{Beta}(a_0 + k_A,\ b_0 + n_A - k_A)` and likewise for B, and

    .. math::

        P(p_B > p_A) = \int_0^1 f_B(x)\, F_A(x)\, dx,

    computed by adaptive quadrature. The default ``prior=(1, 1)`` is uniform.

    This is a descriptive summary, not a test: a value near 1 is not evidence that a
    pre-registered comparison succeeded.

    References
    ----------
    Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed., section 2.4. CRC Press.
    """
    ka, na = check_counts(ka, na, k_name="ka", n_name="na")
    kb, nb = check_counts(kb, nb, k_name="kb", n_name="nb")
    a0, b0 = _check_prior(prior)
    post_a = stats.beta(a0 + ka, b0 + na - ka)
    post_b = stats.beta(a0 + kb, b0 + nb - kb)
    value, _ = integrate.quad(
        lambda x: post_b.pdf(x) * post_a.cdf(x), 0.0, 1.0, epsabs=1e-13, epsrel=1e-11, limit=200
    )
    return min(1.0, max(0.0, float(value)))


def credible_interval(
    k: int,
    n: int,
    *,
    level: float = 0.95,
    prior: tuple[float, float] = (1.0, 1.0),
) -> Interval:
    r"""Equal-tailed credible interval of the :math:`\text{Beta}(a_0 + k,\ b_0 + n - k)` posterior.

    The bounds are the :math:`\alpha/2` and :math:`1 - \alpha/2` posterior quantiles with
    :math:`\alpha = 1 - \text{level}`. With ``prior=(0.5, 0.5)`` this is the Jeffreys
    interval without the boundary adjustment.

    References
    ----------
    Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed., section 2.3. CRC Press.
    """
    k, n = check_counts(k, n)
    level = check_open_unit("level", level)
    a0, b0 = _check_prior(prior)
    posterior = stats.beta(a0 + k, b0 + n - k)
    alpha = 1.0 - level
    return Interval(
        low=float(posterior.ppf(alpha / 2.0)),
        high=float(posterior.ppf(1.0 - alpha / 2.0)),
        level=level,
        method=f"beta-posterior({a0:g},{b0:g})",
    )
