r"""Checkpoint ladders: trend across training steps, and where success levels off.

A ladder evaluates checkpoints of one training run in order of training step. Two questions
are asked, both fixed before the data are seen:

- **Trend:** does the success rate change monotonically with the step? The
  Cochran–Armitage test (independent arms) or its stratified Mantel extension (arms run in
  blocks) tests for a linear trend in the step scores.
- **Plateau:** from which checkpoint on is every checkpoint within a margin :math:`\delta`
  of the final one? Each earlier checkpoint is tested for non-inferiority against the final
  checkpoint, in a fixed sequence from the last to the first, so the family-wise error stays
  at :math:`\alpha` without adjustment.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from fieldtrial.stats._types import Alternative, Interval, TestResult
from fieldtrial.stats._validation import check_alternative, check_count, check_open_unit
from fieldtrial.stats.compare import newcombe_interval
from fieldtrial.stats.paired import tango_interval


@dataclass(frozen=True, slots=True)
class TrendResult:
    """Linear trend test across ordered arms.

    ``test.statistic`` is the standardized statistic :math:`Z` (positive when success rises
    with the score). ``strata_used`` counts the strata that carried information.
    """

    test: TestResult
    scores: tuple[float, ...]
    strata_used: int


@dataclass(frozen=True, slots=True)
class PlateauStep:
    r"""Non-inferiority of one checkpoint against the reference (final) checkpoint.

    ``difference`` is reference minus this checkpoint, with a two-sided
    :math:`1 - 2\alpha` interval; the checkpoint is non-inferior when the upper bound is
    below the margin. ``tested`` is False for checkpoints after the fixed sequence stopped.
    """

    index: int
    difference: float
    interval: Interval
    tested: bool
    noninferior: bool


@dataclass(frozen=True, slots=True)
class PlateauResult:
    """Fixed-sequence non-inferiority of each checkpoint against the final one.

    ``plateau_index`` is the earliest checkpoint from which every checkpoint was shown to be
    within ``margin`` of the reference, or None when the checkpoint right before the
    reference was not.
    """

    reference: int
    margin: float
    alpha: float
    steps: tuple[PlateauStep, ...]
    plateau_index: int | None
    method: str


def _scores(scores: Sequence[float] | None, arms: int) -> tuple[float, ...]:
    out = tuple(float(s) for s in scores) if scores is not None else tuple(map(float, range(arms)))
    if len(out) != arms:
        raise ValueError(f"expected {arms} scores, got {len(out)}")
    if not all(math.isfinite(s) for s in out):
        raise ValueError("scores must be finite")
    if len(set(out)) < 2:
        raise ValueError("scores must not all be equal")
    return out


def _pvalue(z: float, alternative: Alternative) -> float:
    if math.isnan(z):
        return 1.0
    if alternative == "greater":
        return float(stats.norm.sf(z))
    if alternative == "less":
        return float(stats.norm.cdf(z))
    return float(min(1.0, 2.0 * stats.norm.sf(abs(z))))


def _check_arm_counts(counts: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for i, (k, n) in enumerate(counts):
        k = check_count(f"successes[{i}]", k)
        n = check_count(f"trials[{i}]", n)
        if k > n:
            raise ValueError(f"successes ({k}) cannot exceed trials ({n}) for arm {i}")
        out.append((k, n))
    return out


def trend_test(
    counts: Sequence[tuple[int, int]],
    *,
    scores: Sequence[float] | None = None,
    alternative: Alternative = "two-sided",
) -> TrendResult:
    r"""Cochran–Armitage test for a linear trend in success across ordered, independent arms.

    Parameters
    ----------
    counts
        ``(successes, trials)`` per arm, in ladder order.
    scores
        Score of each arm, for example the training step. Defaults to 0, 1, 2, …
    alternative
        ``"greater"`` tests for success rising with the score.

    Notes
    -----
    With :math:`\bar p = \sum k_i / N` and :math:`\bar x = \sum n_i x_i / N`,

    .. math::

        Z = \frac{\sum_i x_i (k_i - n_i \bar p)}
                 {\sqrt{\bar p (1 - \bar p) \sum_i n_i (x_i - \bar x)^2}}
        \sim N(0, 1).

    :math:`Z^2 = N r^2`, where :math:`r` is the Pearson correlation between score and
    outcome; Mantel's statistic :math:`M^2 = (N - 1) r^2` differs only by the factor
    :math:`(N - 1)/N`. The test is undefined (p = 1) when every trial succeeded or every
    trial failed.

    References
    ----------
    Armitage, P. (1955). Tests for linear trends in proportions and frequencies.
    *Biometrics* 11, 375–386.

    Agresti, A. (2013). *Categorical Data Analysis*, 3rd ed., section 3.4.6.
    """
    alternative = check_alternative(alternative)
    arms = _check_arm_counts(counts)
    x = np.array(_scores(scores, len(arms)))
    k = np.array([a for a, _ in arms], dtype=float)
    n = np.array([b for _, b in arms], dtype=float)
    total = n.sum()
    if total < 2:
        raise ValueError("at least 2 trials are needed")
    p = k.sum() / total
    xbar = float((n * x).sum() / total)
    var = p * (1.0 - p) * float((n * (x - xbar) ** 2).sum())
    t = float((x * (k - n * p)).sum())
    z = t / math.sqrt(var) if var > 0 else math.nan
    return TrendResult(
        test=TestResult(
            test="cochran_armitage",
            statistic=z,
            pvalue=_pvalue(z, alternative),
            alternative=alternative,
        ),
        scores=tuple(float(s) for s in x),
        strata_used=1 if var > 0 else 0,
    )


def stratified_trend_test(
    strata: Sequence[Sequence[tuple[int, int]]],
    *,
    scores: Sequence[float] | None = None,
    alternative: Alternative = "two-sided",
) -> TrendResult:
    r"""Mantel's stratified test for a linear trend, for arms run in blocks.

    Each stratum is one block (or condition) and lists ``(successes, trials)`` per arm in
    ladder order. A randomized complete block design with one trial per arm gives strata
    of ``(0, 1)`` or ``(1, 1)`` entries.

    Notes
    -----
    Conditional on each stratum's margins, with :math:`N_s` trials, :math:`K_s` successes,
    :math:`\bar p_s = K_s / N_s` and :math:`T_s = \sum_i x_i (k_{is} - n_{is} \bar p_s)`,

    .. math::

        V_s = \frac{K_s (N_s - K_s)}{N_s (N_s - 1)}
              \left(\sum_i n_{is} x_i^2 - \frac{(\sum_i n_{is} x_i)^2}{N_s}\right), \qquad
        Z = \frac{\sum_s T_s}{\sqrt{\sum_s V_s}}.

    :math:`V_s` is the exact permutation variance of :math:`T_s`, so strata where every
    trial agrees contribute nothing. With one stratum, :math:`Z^2` is Mantel's
    :math:`M^2 = (N - 1) r^2`.

    References
    ----------
    Mantel, N. (1963). Chi-square tests with one degree of freedom; extensions of the
    Mantel–Haenszel procedure. *JASA* 58, 690–700.
    """
    alternative = check_alternative(alternative)
    if not strata:
        raise ValueError("at least one stratum is needed")
    width = len(strata[0])
    x = np.array(_scores(scores, width))
    t_sum = v_sum = 0.0
    used = 0
    for s in strata:
        if len(s) != width:
            raise ValueError("every stratum must list the same number of arms")
        arms = _check_arm_counts(s)
        k = np.array([a for a, _ in arms], dtype=float)
        n = np.array([b for _, b in arms], dtype=float)
        big_n, big_k = n.sum(), k.sum()
        if big_n < 2 or big_k in (0.0, big_n):
            continue
        sxx = float((n * x * x).sum() - (n * x).sum() ** 2 / big_n)
        if sxx <= 0:
            continue
        t_sum += float((x * (k - n * big_k / big_n)).sum())
        v_sum += big_k * (big_n - big_k) / (big_n * (big_n - 1.0)) * sxx
        used += 1
    z = t_sum / math.sqrt(v_sum) if v_sum > 0 else math.nan
    return TrendResult(
        test=TestResult(
            test="mantel_trend",
            statistic=z,
            pvalue=_pvalue(z, alternative),
            alternative=alternative,
        ),
        scores=tuple(float(v) for v in x),
        strata_used=used,
    )


def _fixed_sequence(
    diffs: list[tuple[float, tuple[float, float]]],
    *,
    margin: float,
    alpha: float,
    method: str,
) -> PlateauResult:
    reference = len(diffs)
    steps: list[PlateauStep] = []
    testing = True
    plateau: int | None = None
    for j in range(reference - 1, -1, -1):
        d, (low, high) = diffs[j]
        ni = testing and high < margin
        steps.append(
            PlateauStep(
                index=j,
                difference=d,
                interval=Interval(low=low, high=high, level=1.0 - 2.0 * alpha, method=method),
                tested=testing,
                noninferior=ni,
            )
        )
        if ni:
            plateau = j
        else:
            testing = False
    return PlateauResult(
        reference=reference,
        margin=margin,
        alpha=alpha,
        steps=tuple(sorted(steps, key=lambda s: s.index)),
        plateau_index=plateau,
        method=method,
    )


def _check_plateau_args(margin: float, alpha: float) -> tuple[float, float]:
    margin = float(margin)
    if not (math.isfinite(margin) and 0.0 < margin < 1.0):
        raise ValueError(f"margin must be strictly between 0 and 1, got {margin!r}")
    alpha = check_open_unit("alpha", alpha)
    if alpha >= 0.5:
        raise ValueError(f"alpha must be below 0.5, got {alpha}")
    return margin, alpha


def plateau(
    counts: Sequence[tuple[int, int]], *, margin: float, alpha: float = 0.05
) -> PlateauResult:
    r"""Find where a ladder of independent arms levels off, relative to its final checkpoint.

    Parameters
    ----------
    counts
        ``(successes, trials)`` per checkpoint, in ladder order; the last is the reference.
    margin
        Non-inferiority margin :math:`\delta` on the success-rate scale, fixed in advance.
    alpha
        One-sided level of each non-inferiority test.

    Notes
    -----
    For checkpoint :math:`j`, test :math:`H_{0j}: p_K - p_j \ge \delta` against
    :math:`p_K - p_j < \delta`. :math:`H_{0j}` is rejected when the upper bound of the
    two-sided :math:`1 - 2\alpha` Newcombe interval for :math:`p_K - p_j` is below
    :math:`\delta`. Hypotheses are tested in the fixed order :math:`j = K-1, K-2, \dots, 1`,
    each at level :math:`\alpha`, stopping at the first that is not rejected. A fixed
    testing sequence controls the family-wise error at :math:`\alpha`.

    References
    ----------
    Maurer, W., Hothorn, L. A. and Lehmacher, W. (1995). Multiple comparisons in drug
    clinical trials and preclinical assays: a-priori ordered hypotheses. In *Biometrie in
    der chemisch-pharmazeutischen Industrie* 6, 3–18.

    Newcombe, R. G. (1998). *Statistics in Medicine* 17, 873–890.
    """
    margin, alpha = _check_plateau_args(margin, alpha)
    arms = _check_arm_counts(counts)
    if len(arms) < 2:
        raise ValueError("a ladder needs at least 2 checkpoints")
    kr, nr = arms[-1]
    diffs = []
    for k, n in arms[:-1]:
        low, high = newcombe_interval(kr, nr, k, n, level=1.0 - 2.0 * alpha)
        diffs.append((kr / nr - k / n, (low, high)))
    return _fixed_sequence(diffs, margin=margin, alpha=alpha, method="newcombe")


def plateau_paired(
    outcomes: Sequence[Sequence[int]] | np.ndarray, *, margin: float, alpha: float = 0.05
) -> PlateauResult:
    r"""Find where a ladder levels off when every checkpoint ran once per block.

    ``outcomes`` is a binary matrix with one row per complete block and one column per
    checkpoint, in ladder order; the last column is the reference. The procedure is the
    same as :func:`plateau` with Tango's score interval for the paired difference
    (Tango, 1998, which was designed for exactly this non-inferiority question).

    References
    ----------
    Tango, T. (1998). Equivalence test and confidence interval for the difference in
    proportions for the paired-sample design. *Statistics in Medicine* 17, 891–908.
    """
    margin, alpha = _check_plateau_args(margin, alpha)
    m = np.asarray(outcomes)
    if m.ndim != 2 or m.shape[1] < 2 or m.shape[0] < 1:
        raise ValueError("outcomes must be a blocks × checkpoints matrix with at least 2 columns")
    if not np.isin(m, (0, 1)).all():
        raise ValueError("outcomes must be 0 or 1")
    m = m.astype(int)
    n = m.shape[0]
    ref = m[:, -1]
    diffs = []
    for j in range(m.shape[1] - 1):
        b = int(((ref == 1) & (m[:, j] == 0)).sum())
        c = int(((ref == 0) & (m[:, j] == 1)).sum())
        low, high = tango_interval(b, c, n, level=1.0 - 2.0 * alpha)
        diffs.append(((b - c) / n, (low, high)))
    return _fixed_sequence(diffs, margin=margin, alpha=alpha, method="tango")
