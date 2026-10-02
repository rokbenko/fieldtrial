r"""Agreement between two raters: Cohen's kappa.

For a :math:`k \times k` table of counts (rows: rater A, columns: rater B) with cell
proportions :math:`p_{ij}` and margins :math:`p_{i\cdot}`, :math:`p_{\cdot j}`, the observed
agreement is :math:`p_o = \sum_i p_{ii}`, the agreement expected by chance is
:math:`p_e = \sum_i p_{i\cdot} p_{\cdot i}`, and

.. math::

    \kappa = \frac{p_o - p_e}{1 - p_e}.

The large-sample variance of Fleiss, Cohen and Everitt (1969) is

.. math::

    \operatorname{Var}(\hat\kappa) = \frac{A + B - C}{n (1 - p_e)^2}, \quad
    A = \sum_i p_{ii}\,[1 - (p_{i\cdot} + p_{\cdot i})(1 - \kappa)]^2, \quad
    B = (1 - \kappa)^2 \sum_{i \ne j} p_{ij} (p_{\cdot i} + p_{j\cdot})^2, \quad
    C = [\kappa - p_e (1 - \kappa)]^2,

which gives the confidence interval :math:`\hat\kappa \pm z_{1-\alpha/2} \sqrt{\cdot}`. Under
:math:`\kappa = 0` the variance is
:math:`[p_e + p_e^2 - \sum_i p_{i\cdot} p_{\cdot i} (p_{i\cdot} + p_{\cdot i})] / (n (1 - p_e)^2)`,
used for the test of no agreement beyond chance.

References
----------
Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and
Psychological Measurement* 20, 37–46.

Fleiss, J. L., Cohen, J. and Everitt, B. S. (1969). Large sample standard errors of kappa
and weighted kappa. *Psychological Bulletin* 72(5), 323–327.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import special

from fieldtrial.stats._types import Interval, TestResult
from fieldtrial.stats._validation import check_open_unit


@dataclass(frozen=True, slots=True)
class KappaResult:
    r"""Cohen's kappa with its interval and the test of no agreement beyond chance.

    Attributes
    ----------
    n
        Number of rated items.
    agreement
        Observed proportion of agreement :math:`p_o`.
    expected
        Agreement expected by chance :math:`p_e`.
    kappa
        Cohen's kappa (NaN when :math:`p_e = 1`, for example when both raters always give
        the same single category).
    interval
        Wald interval with the Fleiss–Cohen–Everitt standard error.
    test
        z test of :math:`\kappa = 0` (two-sided).
    """

    n: int
    agreement: float
    expected: float
    kappa: float
    standard_error: float
    interval: Interval
    test: TestResult


def kappa_table(rater_a: Sequence[object], rater_b: Sequence[object]) -> list[list[int]]:
    """Count table of two raters' labels (categories in sorted order of their union)."""
    if len(rater_a) != len(rater_b):
        raise ValueError("both raters must label the same items")
    categories = sorted({*rater_a, *rater_b}, key=repr)
    index = {c: i for i, c in enumerate(categories)}
    table = [[0] * len(categories) for _ in categories]
    for a, b in zip(rater_a, rater_b, strict=True):
        table[index[a]][index[b]] += 1
    return table


def cohens_kappa(table: Sequence[Sequence[int]], level: float = 0.95) -> KappaResult:
    """Cohen's kappa from a square table of counts (rows: rater A, columns: rater B)."""
    t = np.asarray(table, dtype=np.float64)
    if t.ndim != 2 or t.shape[0] != t.shape[1] or t.shape[0] < 2:
        raise ValueError("table must be square with at least 2 categories")
    if np.any(t < 0) or not np.all(np.isfinite(t)) or np.any(t != np.round(t)):
        raise ValueError("table must hold non-negative counts")
    n = float(t.sum())
    if n == 0:
        raise ValueError("table is empty")
    lvl = check_open_unit("level", level)
    p = t / n
    row = p.sum(axis=1)
    col = p.sum(axis=0)
    po = float(np.trace(p))
    pe = float(np.sum(row * col))
    z = float(special.ndtri(1 - (1 - lvl) / 2))
    if pe >= 1.0:
        nan = math.nan
        return KappaResult(
            n=int(n),
            agreement=po,
            expected=pe,
            kappa=nan,
            standard_error=nan,
            interval=Interval(nan, nan, lvl, "fleiss-cohen-everitt"),
            test=TestResult("kappa_z", nan, nan, "two-sided"),
        )
    kappa = (po - pe) / (1 - pe)
    term_a = float(np.sum(np.diag(p) * (1 - (row + col) * (1 - kappa)) ** 2))
    cross = p * (col[:, None] + row[None, :]) ** 2
    np.fill_diagonal(cross, 0.0)
    term_b = (1 - kappa) ** 2 * float(cross.sum())
    term_c = (kappa - pe * (1 - kappa)) ** 2
    var = max((term_a + term_b - term_c) / (1 - pe) ** 2 / n, 0.0)
    var0 = (pe + pe**2 - float(np.sum(row * col * (row + col)))) / ((1 - pe) ** 2 * n)
    se = math.sqrt(var)
    stat = kappa / math.sqrt(var0) if var0 > 0 else math.nan
    pvalue = float(2 * special.ndtr(-abs(stat))) if math.isfinite(stat) else math.nan
    return KappaResult(
        n=int(n),
        agreement=po,
        expected=pe,
        kappa=kappa,
        standard_error=se,
        interval=Interval(kappa - z * se, kappa + z * se, lvl, "fleiss-cohen-everitt"),
        test=TestResult("kappa_z", stat, pvalue, "two-sided"),
    )
