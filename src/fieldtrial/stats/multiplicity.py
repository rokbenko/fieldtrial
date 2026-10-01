"""Adjusting p-values when a study makes several comparisons."""

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

from fieldtrial.stats._types import AdjustedPValues
from fieldtrial.stats._validation import check_choice

AdjustMethod = Literal["holm", "bonferroni", "bh"]
ADJUST_METHODS: tuple[AdjustMethod, ...] = ("holm", "bonferroni", "bh")


def adjust_pvalues(
    pvalues: Sequence[float],
    *,
    method: AdjustMethod | str = "holm",
    alpha: float = 0.05,
) -> AdjustedPValues:
    r"""Adjust p-values for multiple comparisons.

    Parameters
    ----------
    pvalues
        Raw p-values, each in [0, 1].
    method
        ``"holm"`` (default), ``"bonferroni"`` or ``"bh"`` (Benjamini–Hochberg).
    alpha
        Level at which ``reject`` is decided, in (0, 1).

    Returns
    -------
    AdjustedPValues
        Adjusted p-values and rejection decisions, in the input order.

    Notes
    -----
    With :math:`m` p-values sorted as :math:`p_{(1)} \le \dots \le p_{(m)}`:

    - **Bonferroni**: :math:`\tilde p_i = \min(1, m\,p_i)`. Controls the family-wise error
      rate (FWER).
    - **Holm** (step-down):
      :math:`\tilde p_{(i)} = \max_{j \le i} \min\{1, (m - j + 1)\,p_{(j)}\}`. Controls the
      FWER and is uniformly more powerful than Bonferroni.
    - **Benjamini–Hochberg** (step-up):
      :math:`\tilde p_{(i)} = \min_{j \ge i} \min\{1, m\,p_{(j)} / j\}`. Controls the false
      discovery rate under independence or positive dependence.

    A hypothesis is rejected when :math:`\tilde p_i \le \alpha`.

    References
    ----------
    Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    *Scandinavian Journal of Statistics* 6, 65–70.

    Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate.
    *JRSS B* 57, 289–300.
    """
    check_choice("method", method, ADJUST_METHODS)
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")
    if len(pvalues) == 0:
        raise ValueError("pvalues must not be empty")
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be a flat sequence")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("every p-value must be a number in [0, 1]")

    m = p.size
    if method == "bonferroni":
        adjusted = np.minimum(1.0, m * p)
    else:
        order = np.argsort(p, kind="stable")
        sorted_p = p[order]
        if method == "holm":
            factors = m - np.arange(m)  # m, m-1, ..., 1
            stepped = np.maximum.accumulate(np.minimum(1.0, factors * sorted_p))
        else:  # bh
            ranks = np.arange(1, m + 1)
            stepped = np.minimum.accumulate((m * sorted_p / ranks)[::-1])[::-1]
            stepped = np.minimum(1.0, stepped)
        adjusted = np.empty(m)
        adjusted[order] = stepped

    adjusted_t = tuple(float(x) for x in adjusted)
    return AdjustedPValues(
        raw=tuple(float(x) for x in p),
        adjusted=adjusted_t,
        reject=tuple(x <= alpha and not math.isnan(x) for x in adjusted_t),
        method=method,
        alpha=alpha,
    )
