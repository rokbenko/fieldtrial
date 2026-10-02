"""Validity checks: whether results changed across sessions or trials were voided unevenly."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from fieldtrial.stats._validation import check_counts


@dataclass(frozen=True, slots=True)
class HomogeneityResult:
    """Test that several groups share one success rate.

    ``method`` is ``"fisher"`` for two groups and ``"chi-square"`` otherwise.
    ``small_expected`` is True when an expected cell count is below 5, in which case the
    chi-square p-value is only approximate.
    """

    method: str
    statistic: float
    pvalue: float
    df: int
    groups: int
    small_expected: bool


def homogeneity_test(groups: Sequence[tuple[int, int]]) -> HomogeneityResult:
    r"""Test whether groups (sessions, operators, arms) share one success rate.

    Each group is ``(successes, trials)``; groups with no trials are skipped.

    - **Two groups:** Fisher's exact test (two-sided); the statistic is the sample odds
      ratio.
    - **More groups:** Pearson's chi-square test of homogeneity on the
      :math:`g \times 2` table, :math:`\sum (O - E)^2 / E \sim \chi^2_{g-1}`, without
      continuity correction.

    If every trial in every group has the same outcome, the rates are trivially homogeneous
    and the p-value is 1.

    Use it to flag session drift: for example the gripper wear that forced Dream Machines to
    rerun their evaluations would show up as different success rates across sessions.

    References
    ----------
    Agresti, A. (2013). *Categorical Data Analysis*, 3rd ed., sections 3.2 and 3.5. Wiley.
    """
    rows = []
    for k, n in groups:
        if n == 0:
            continue
        check_counts(k, n)
        rows.append((k, n - k))
    if len(rows) < 2:
        raise ValueError("need at least 2 groups with trials")
    table = np.array(rows, dtype=np.int64)
    if np.any(table.sum(axis=0) == 0):
        return HomogeneityResult(
            "fisher" if len(rows) == 2 else "chi-square", 0.0, 1.0, len(rows) - 1, len(rows), False
        )
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0) / table.sum()
    small = bool(np.any(expected < 5))
    if len(rows) == 2:
        res = stats.fisher_exact(table)
        statistic = float(res.statistic)
        return HomogeneityResult(
            "fisher",
            statistic if math.isfinite(statistic) else math.inf,
            float(res.pvalue),
            1,
            2,
            small,
        )
    chi = stats.chi2_contingency(table, correction=False)
    return HomogeneityResult(
        "chi-square", float(chi.statistic), float(chi.pvalue), int(chi.dof), len(rows), small
    )
