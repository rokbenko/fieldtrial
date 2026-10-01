"""Planning: sample size, power and the minimum detectable effect.

Closed-form methods follow Chow, Shao and Wang (2008), *Sample Size Calculations in Clinical
Research*, chapter 4, and Fleiss, Levin and Paik (2003), *Statistical Methods for Rates and
Proportions*, chapter 4. Exact power for Boschloo's and McNemar's tests is computed by
enumerating every possible outcome, so it carries no Monte Carlo error.
"""

import math
from functools import cache
from typing import Literal

import numpy as np
from scipy import optimize, stats

from fieldtrial.stats._types import (
    Alternative,
    MinimumDetectableEffect,
    PairedPowerResult,
    PowerResult,
    SampleSize,
)
from fieldtrial.stats._validation import (
    check_alternative,
    check_choice,
    check_count,
    check_open_unit,
)

PowerMethod = Literal["pooled-z", "fleiss-cc", "arcsine"]
POWER_METHODS: tuple[PowerMethod, ...] = ("pooled-z", "fleiss-cc", "arcsine")


def _z_alpha(alpha: float, alternative: Alternative) -> float:
    return float(stats.norm.isf(alpha / 2.0 if alternative == "two-sided" else alpha))


def _check_ratio(ratio: float) -> float:
    r = float(ratio)
    if not (math.isfinite(r) and r > 0):
        raise ValueError(f"ratio must be a positive number, got {ratio!r}")
    return r


def _check_rates(p1: float, p2: float) -> tuple[float, float]:
    p1 = check_open_unit("p1", p1)
    p2 = check_open_unit("p2", p2)
    if p1 == p2:
        raise ValueError("p1 and p2 must differ")
    return p1, p2


def _n1_pooled(p1: float, p2: float, z_a: float, z_b: float, r: float) -> float:
    pbar = (p1 + r * p2) / (1.0 + r)
    term_a = z_a * math.sqrt(pbar * (1.0 - pbar) * (1.0 + 1.0 / r))
    term_b = z_b * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2) / r)
    return (term_a + term_b) ** 2 / (p1 - p2) ** 2


def _n1_exact(p1: float, p2: float, z_a: float, z_b: float, r: float, method: str) -> float:
    if method == "arcsine":
        h = 2.0 * math.asin(math.sqrt(p2)) - 2.0 * math.asin(math.sqrt(p1))
        return (z_a + z_b) ** 2 * (1.0 + 1.0 / r) / h**2
    n = _n1_pooled(p1, p2, z_a, z_b, r)
    if method == "fleiss-cc":
        a = 2.0 * (r + 1.0) / (r * abs(p1 - p2))
        return n / 4.0 * (1.0 + math.sqrt(1.0 + a / n)) ** 2
    return n


def sample_size(
    p1: float,
    p2: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    method: PowerMethod | str = "pooled-z",
    ratio: float = 1.0,
    alternative: Alternative = "two-sided",
) -> SampleSize:
    r"""Trials per arm needed to detect a difference between two success rates.

    Parameters
    ----------
    p1, p2
        Success rates of arm 1 and arm 2 (for example baseline and hoped-for new rate).
    alpha
        Significance level of the planned test.
    power
        Desired power, :math:`1 - \beta`.
    method
        ``"pooled-z"`` (default), ``"fleiss-cc"`` or ``"arcsine"``.
    ratio
        Allocation ratio :math:`r = n_2 / n_1`, for example 3 for 40 new trials against 120
        baseline trials.
    alternative
        ``"two-sided"`` (default) uses :math:`z_{1-\alpha/2}`; one-sided alternatives use
        :math:`z_{1-\alpha}`.

    Notes
    -----
    With :math:`\Delta = p_1 - p_2`, :math:`\bar p = (p_1 + r p_2)/(1 + r)` and
    :math:`q = 1 - p`:

    - **pooled-z** (the normal approximation to the pooled two-proportion z-test):

      .. math::

          n_1 = \frac{\left(z_\alpha \sqrt{\bar p \bar q (1 + 1/r)}
                + z_\beta \sqrt{p_1 q_1 + p_2 q_2 / r}\right)^2}{\Delta^2}

    - **fleiss-cc** adds Fleiss' continuity correction to the pooled-z :math:`n_1`:
      :math:`n_1' = \tfrac{n_1}{4}\left(1 + \sqrt{1 + 2(r + 1)/(n_1 r |\Delta|)}\right)^2`.
    - **arcsine** uses Cohen's effect size
      :math:`h = 2\arcsin\sqrt{p_2} - 2\arcsin\sqrt{p_1}`:
      :math:`n_1 = (z_\alpha + z_\beta)^2 (1 + 1/r) / h^2`.

    As usual for sample-size formulas, the far tail of a two-sided test is ignored.
    :math:`n_2 = r n_1`; both are rounded up in ``n1`` and ``n2``.

    References
    ----------
    Fleiss, J. L., Levin, B. and Paik, M. C. (2003). *Statistical Methods for Rates and
    Proportions*, 3rd ed., section 4.2. Wiley.

    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed.,
    chapter 6. Erlbaum.
    """
    p1, p2 = _check_rates(p1, p2)
    alpha = check_open_unit("alpha", alpha)
    power = check_open_unit("power", power)
    check_choice("method", method, POWER_METHODS)
    r = _check_ratio(ratio)
    alternative = check_alternative(alternative)
    z_a = _z_alpha(alpha, alternative)
    z_b = float(stats.norm.isf(1.0 - power))
    n1 = _n1_exact(p1, p2, z_a, z_b, r, method)
    n2 = n1 * r
    return SampleSize(
        p1=p1,
        p2=p2,
        n1_exact=n1,
        n2_exact=n2,
        n1=math.ceil(n1 - 1e-9),
        n2=math.ceil(n2 - 1e-9),
        alpha=alpha,
        power=power,
        ratio=r,
        method=method,
        alternative=alternative,
    )


def _power_closed_form(
    p1: float, p2: float, n1: float, r: float, alpha: float, method: str, alternative: Alternative
) -> float:
    # Under the alternative the standardized test statistic is approximately normal with
    # mean `shift` and standard deviation 1; it rejects beyond `crit`.
    z_a = _z_alpha(alpha, alternative)
    delta = p2 - p1
    if method == "arcsine":
        h = 2.0 * math.asin(math.sqrt(p2)) - 2.0 * math.asin(math.sqrt(p1))
        shift = h * math.sqrt(n1 / (1.0 + 1.0 / r))
        crit = z_a
    else:
        if method == "fleiss-cc":
            # Undo the continuity correction: n1 is the corrected size n1'.
            a = 2.0 * (r + 1.0) / (r * abs(delta)) if delta != 0 else math.inf
            s = 2.0 * math.sqrt(n1)
            n1 = 0.0 if s * s <= a else ((s * s - a) / (2.0 * s)) ** 2
        pbar = (p1 + r * p2) / (1.0 + r)
        null_sd = math.sqrt(pbar * (1.0 - pbar) * (1.0 + 1.0 / r))
        alt_sd = math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2) / r)
        shift = delta * math.sqrt(n1) / alt_sd
        crit = z_a * null_sd / alt_sd
    # "greater" means p1 > p2, i.e. a negative delta = p2 - p1.
    if alternative == "greater":
        return float(stats.norm.cdf(-shift - crit))
    if alternative == "less":
        return float(stats.norm.cdf(shift - crit))
    return float(stats.norm.cdf(-shift - crit) + stats.norm.cdf(shift - crit))


def power(
    p1: float,
    p2: float,
    n1: int,
    n2: int | None = None,
    *,
    alpha: float = 0.05,
    method: PowerMethod | str = "pooled-z",
    alternative: Alternative = "two-sided",
) -> PowerResult:
    r"""Approximate power of a comparison of two independent arms (closed form).

    ``n2`` defaults to ``n1``. The formulas are the ones in :func:`sample_size`, solved for
    power; for two-sided tests both tails are included, so the power at
    :math:`p_1 = p_2` equals :math:`\alpha`. ``"greater"`` means :math:`p_1 > p_2`.

    For the exact power of Boschloo's test, see :func:`boschloo_power`.
    """
    p1 = check_open_unit("p1", p1)
    p2 = check_open_unit("p2", p2)
    n1 = check_count("n1", n1)
    n2 = n1 if n2 is None else check_count("n2", n2)
    if n1 == 0 or n2 == 0:
        raise ValueError("n1 and n2 must be at least 1")
    alpha = check_open_unit("alpha", alpha)
    check_choice("method", method, POWER_METHODS)
    alternative = check_alternative(alternative)
    value = _power_closed_form(p1, p2, float(n1), n2 / n1, alpha, method, alternative)
    return PowerResult(
        p1=p1,
        p2=p2,
        n1=n1,
        n2=n2,
        alpha=alpha,
        power=min(1.0, max(0.0, value)),
        method=method,
        alternative=alternative,
    )


def mde(
    p_baseline: float,
    n1: int,
    n2: int | None = None,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    direction: Literal["increase", "decrease"] = "increase",
    method: PowerMethod | str = "pooled-z",
) -> MinimumDetectableEffect:
    r"""Minimum detectable effect: the smallest change from a baseline a study can detect.

    Solves :func:`sample_size` for the second arm's rate: the returned ``target`` is the rate
    :math:`p_2` for which ``sample_size(p_baseline, p2, ratio=n2/n1)`` equals ``n1``
    (two-sided test at ``alpha`` with the requested ``power``). Arm 1 is the baseline with
    ``n1`` trials and arm 2 the comparison arm with ``n2`` trials (default ``n1``).

    ``target`` and ``effect`` are ``None`` when even a rate of 0 or 1 cannot be detected
    with this many trials.

    Example: with a baseline of 0.76 and 40 trials per arm, the study detects increases to
    about 0.971 (+21.1 percentage points) and decreases to about 0.459 (−30.1 points).
    """
    p_baseline = check_open_unit("p_baseline", p_baseline)
    n1 = check_count("n1", n1)
    n2 = n1 if n2 is None else check_count("n2", n2)
    if n1 == 0 or n2 == 0:
        raise ValueError("n1 and n2 must be at least 1")
    alpha = check_open_unit("alpha", alpha)
    power = check_open_unit("power", power)
    check_choice("direction", direction, ("increase", "decrease"))
    check_choice("method", method, POWER_METHODS)
    r = n2 / n1
    z_a = _z_alpha(alpha, "two-sided")
    z_b = float(stats.norm.isf(1.0 - power))

    def excess(p2: float) -> float:
        return _n1_exact(p_baseline, p2, z_a, z_b, r, method) - n1

    eps = 1e-12
    if direction == "increase":
        bracket = (p_baseline + 1e-9, 1.0 - eps)
    else:
        bracket = (eps, p_baseline - 1e-9)
    far_end = bracket[1] if direction == "increase" else bracket[0]
    target: float | None = None
    if excess(far_end) <= 0:
        target = float(optimize.brentq(excess, *bracket, xtol=1e-12))
    return MinimumDetectableEffect(
        baseline=p_baseline,
        n1=n1,
        n2=n2,
        direction=direction,
        target=target,
        effect=None if target is None else target - p_baseline,
        alpha=alpha,
        power=power,
        method=method,
    )


# --- exact power ----------------------------------------------------------------------------

_GRID_SIZE = 4001  # nuisance-parameter grid: sup accurate to about 1e-7 relative
_GRID_CHUNK = 64


def boschloo_pvalues(n1: int, n2: int, *, alternative: Literal["greater", "less"]) -> np.ndarray:
    r"""One-sided Boschloo p-values for every possible outcome ``(x1, x2)`` at once.

    Returns an array of shape ``(n1 + 1, n2 + 1)`` indexed by successes in arm 1 and arm 2.
    The p-value of a table is
    :math:`\sup_\pi \sum_{t:\,F(t) \le F(\text{obs})} P_\pi(t)`, where :math:`F` is Fisher's
    one-sided p-value (as in :func:`scipy.stats.boschloo_exact`, including its 1e-13
    relative tie tolerance) and :math:`P_\pi` the product of two binomials with common rate
    :math:`\pi`. The supremum is taken over a fixed grid of 4001 values of :math:`\pi`, so
    the p-values are exactly monotone in :math:`F`.
    """
    return _boschloo_pvalues_cached(n1, n2, alternative).copy()


@cache
def _boschloo_pvalues_cached(n1: int, n2: int, alternative: str) -> np.ndarray:
    x1 = np.arange(n1 + 1)[:, None]
    x2 = np.arange(n2 + 1)[None, :]
    total = n1 + n2
    if alternative == "less":
        fisher = np.asarray(stats.hypergeom.cdf(x1, total, x1 + x2, n1), dtype=float)
    else:
        fisher = np.asarray(stats.hypergeom.cdf(x2, total, x1 + x2, n2), dtype=float)
    flat = fisher.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_stat = flat[order]
    # Last sorted position whose statistic is within scipy's tie tolerance of each table's.
    ends = np.searchsorted(sorted_stat, sorted_stat * (1.0 + 1e-13), side="right") - 1

    grid = np.linspace(0.0, 1.0, _GRID_SIZE)
    k1 = np.arange(n1 + 1)
    k2 = np.arange(n2 + 1)
    best = np.zeros(flat.size)
    for start in range(0, grid.size, _GRID_CHUNK):
        pis = grid[start : start + _GRID_CHUNK][:, None]
        f1 = np.asarray(stats.binom.pmf(k1[None, :], n1, pis), dtype=float)  # (chunk, n1 + 1)
        f2 = np.asarray(stats.binom.pmf(k2[None, :], n2, pis), dtype=float)  # (chunk, n2 + 1)
        probs = (f1[:, :, None] * f2[:, None, :]).reshape(pis.shape[0], -1)[:, order]
        cumulative = np.cumsum(probs, axis=1)[:, ends]
        np.maximum(best, cumulative.max(axis=0), out=best)
    flat_pvalues = np.empty_like(best)
    flat_pvalues[order] = np.minimum(best, 1.0)
    pvalues: np.ndarray = flat_pvalues.reshape(n1 + 1, n2 + 1)
    pvalues.setflags(write=False)
    return pvalues


def boschloo_rejection_region(
    n1: int, n2: int, *, alpha: float = 0.05, alternative: Alternative = "two-sided"
) -> np.ndarray:
    """Boolean ``(n1 + 1, n2 + 1)`` array: which outcomes Boschloo's test rejects at ``alpha``.

    The two-sided test rejects when twice the smaller one-sided p-value is at most ``alpha``,
    as in :func:`scipy.stats.boschloo_exact`.
    """
    n1 = check_count("n1", n1)
    n2 = check_count("n2", n2)
    if n1 == 0 or n2 == 0:
        raise ValueError("n1 and n2 must be at least 1")
    alpha = check_open_unit("alpha", alpha)
    alternative = check_alternative(alternative)
    tol = 1e-12
    if alternative == "two-sided":
        less = _boschloo_pvalues_cached(n1, n2, "less")
        greater = _boschloo_pvalues_cached(n1, n2, "greater")
        result: np.ndarray = 2.0 * np.minimum(less, greater) <= alpha + tol
        return result
    one_sided: np.ndarray = _boschloo_pvalues_cached(n1, n2, alternative) <= alpha + tol
    return one_sided


def boschloo_power(
    p1: float,
    p2: float,
    n1: int,
    n2: int | None = None,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> PowerResult:
    r"""Exact power of Boschloo's test.

    Sums :math:`P(x_1, x_2) = \text{Bin}(x_1; n_1, p_1)\,\text{Bin}(x_2; n_2, p_2)` over the
    rejection region (:func:`boschloo_rejection_region`). No simulation is involved, so the
    result is deterministic. Practical up to a few hundred trials per arm.
    """
    p1 = check_open_unit("p1", p1)
    p2 = check_open_unit("p2", p2)
    n1 = check_count("n1", n1)
    n2 = n1 if n2 is None else check_count("n2", n2)
    alternative = check_alternative(alternative)
    region = boschloo_rejection_region(n1, n2, alpha=alpha, alternative=alternative)
    f1 = stats.binom.pmf(np.arange(n1 + 1), n1, p1)
    f2 = stats.binom.pmf(np.arange(n2 + 1), n2, p2)
    value = float(np.sum(np.outer(f1, f2)[region]))
    return PowerResult(
        p1=p1,
        p2=p2,
        n1=n1,
        n2=n2,
        alpha=alpha,
        power=min(1.0, value),
        method="boschloo-exact",
        alternative=alternative,
    )


def mcnemar_power(
    p10: float,
    p01: float,
    n: int,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> PairedPowerResult:
    r"""Exact power of the exact McNemar test with ``n`` pairs.

    The number of discordant pairs is :math:`D \sim \text{Bin}(n, p_{10} + p_{01})`, and
    given :math:`D = d`, :math:`b \sim \text{Bin}(d, p_{10}/(p_{10} + p_{01}))`. Power is
    the probability of the rejection region of :func:`fieldtrial.stats.mcnemar_exact`,
    summed exactly over :math:`d` and :math:`b`. ``"greater"`` means arm 1 succeeds more
    often (:math:`p_{10} > p_{01}`).
    """
    p10 = check_open_unit("p10", p10)
    p01 = check_open_unit("p01", p01)
    if p10 + p01 > 1.0:
        raise ValueError("p10 + p01 cannot exceed 1")
    n = check_count("n", n)
    if n == 0:
        raise ValueError("n must be at least 1")
    alpha = check_open_unit("alpha", alpha)
    alternative = check_alternative(alternative)
    share = p10 / (p10 + p01)
    discordant = stats.binom.pmf(np.arange(n + 1), n, p10 + p01)
    total = 0.0
    for d in range(1, n + 1):
        b = np.arange(d + 1)
        lower = stats.binom.cdf(b, d, 0.5)  # P(B <= b)
        upper = stats.binom.sf(b - 1, d, 0.5)  # P(B >= b)
        if alternative == "greater":
            pvals = upper
        elif alternative == "less":
            pvals = lower
        else:
            pvals = np.minimum(1.0, 2.0 * np.minimum(lower, upper))
        reject = pvals <= alpha + 1e-12
        total += discordant[d] * float(np.sum(stats.binom.pmf(b[reject], d, share)))
    return PairedPowerResult(
        p10=p10, p01=p01, n=n, alpha=alpha, power=min(1.0, total), alternative=alternative
    )


def simulate_power(
    p1: float,
    p2: float,
    n1: int,
    n2: int | None = None,
    *,
    test: Literal["boschloo", "fisher"] = "boschloo",
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    n_sims: int = 2000,
    seed: int = 0,
) -> PowerResult:
    r"""Power by seeded Monte Carlo simulation, calling the scipy test on each simulated study.

    Draws ``n_sims`` pairs :math:`x_1 \sim \text{Bin}(n_1, p_1)`,
    :math:`x_2 \sim \text{Bin}(n_2, p_2)` from ``numpy.random.Generator(PCG64(seed))`` and
    counts how often the test rejects at ``alpha``. ``standard_error`` is
    :math:`\sqrt{\hat\beta(1-\hat\beta)/n_\text{sims}}`. Useful as an independent cross-check
    of :func:`boschloo_power`.
    """
    p1 = check_open_unit("p1", p1)
    p2 = check_open_unit("p2", p2)
    n1 = check_count("n1", n1)
    n2 = n1 if n2 is None else check_count("n2", n2)
    if n1 == 0 or n2 == 0:
        raise ValueError("n1 and n2 must be at least 1")
    check_choice("test", test, ("boschloo", "fisher"))
    alpha = check_open_unit("alpha", alpha)
    alternative = check_alternative(alternative)
    n_sims = check_count("n_sims", n_sims)
    if n_sims == 0:
        raise ValueError("n_sims must be at least 1")
    rng = np.random.Generator(np.random.PCG64(seed))
    x1 = rng.binomial(n1, p1, size=n_sims)
    x2 = rng.binomial(n2, p2, size=n_sims)
    pvalue_of: dict[tuple[int, int], float] = {}
    rejections = 0
    for a, b in zip(x1.tolist(), x2.tolist(), strict=True):
        key = (a, b)
        if key not in pvalue_of:
            table = np.array([[a, b], [n1 - a, n2 - b]], dtype=np.int64)
            if test == "boschloo":
                p = stats.boschloo_exact(table, alternative=alternative).pvalue
            else:
                p = stats.fisher_exact(table, alternative=alternative).pvalue
            pvalue_of[key] = float(p)
        rejections += pvalue_of[key] <= alpha
    estimate = rejections / n_sims
    return PowerResult(
        p1=p1,
        p2=p2,
        n1=n1,
        n2=n2,
        alpha=alpha,
        power=estimate,
        method=f"{test}-simulated",
        alternative=alternative,
        standard_error=math.sqrt(estimate * (1.0 - estimate) / n_sims),
    )
