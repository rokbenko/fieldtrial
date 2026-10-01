"""Two independent arms. Golden values: docs/PLAN.md section 12.

The Boschloo values were corrected on 2026-10-01 (decision log): scipy treats each column of
the 2x2 table as one arm.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep

from fieldtrial.stats import compare_independent
from fieldtrial.stats.compare import arms_table, boschloo_test, newcombe_interval, odds_ratio

GOLDEN = [
    # (k1, n1, k2, n2, diff, newcombe_low, newcombe_high, fisher, boschloo)
    (50, 80, 13, 40, 0.3000, 0.1104, 0.4582, 0.00337, 0.00228),
    (36, 40, 91, 120, 0.1417, -0.0054, 0.2450, 0.07060, 0.05574),
    (74, 80, 91, 120, 0.1667, 0.0625, 0.2596, 0.00223, 0.00196),
]


@pytest.mark.parametrize(
    ("k1", "n1", "k2", "n2", "diff", "low", "high", "fisher", "boschloo"), GOLDEN
)
def test_golden_comparisons(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    diff: float,
    low: float,
    high: float,
    fisher: float,
    boschloo: float,
) -> None:
    res = compare_independent(k1, n1, k2, n2)
    assert res.difference == pytest.approx(diff, abs=1e-4)
    assert res.interval.low == pytest.approx(low, abs=1e-4)
    assert res.interval.high == pytest.approx(high, abs=1e-4)
    assert res.interval.method == "newcombe"
    assert res.primary.test == "boschloo"
    assert res.primary.pvalue == pytest.approx(boschloo, abs=1e-5)
    assert res.secondary[0].test == "fisher"
    assert res.secondary[0].pvalue == pytest.approx(fisher, abs=1e-5)


def test_dream_machines_reported_fisher_p() -> None:
    # The Dream Machines post reports Fisher p = 0.0034 for 50/80 vs 13/40.
    res = compare_independent(50, 80, 13, 40, test="fisher")
    assert round(res.primary.pvalue, 4) == 0.0034


@pytest.mark.parametrize(
    ("k1", "n1", "k2", "n2"),
    [(k1, n1, k2, n2) for k1, n1, k2, n2, *_ in GOLDEN] + [(3, 9, 7, 11), (0, 12, 4, 15)],
)
def test_newcombe_matches_statsmodels_live(k1: int, n1: int, k2: int, n2: int) -> None:
    lo, hi = confint_proportions_2indep(k1, n1, k2, n2, method="newcomb", compare="diff")
    ours = newcombe_interval(k1, n1, k2, n2)
    assert ours[0] == pytest.approx(lo, abs=1e-10)
    assert ours[1] == pytest.approx(hi, abs=1e-10)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize(
    ("k1", "n1", "k2", "n2"), [(50, 80, 13, 40), (36, 40, 91, 120), (5, 20, 9, 18)]
)
def test_tests_match_scipy_with_arms_as_columns(
    k1: int, n1: int, k2: int, n2: int, alternative: str
) -> None:
    table = [[k1, k2], [n1 - k1, n2 - k2]]
    res = compare_independent(k1, n1, k2, n2, alternative=alternative)  # type: ignore[arg-type]
    assert res.primary.pvalue == pytest.approx(
        stats.boschloo_exact(table, alternative=alternative).pvalue, rel=1e-12
    )
    assert res.secondary[0].pvalue == pytest.approx(
        stats.fisher_exact(table, alternative=alternative).pvalue, rel=1e-12
    )


def test_boschloo_uses_arms_as_columns_not_rows() -> None:
    # Guard against the transposed table that produced the original (wrong) golden values.
    k1, n1, k2, n2 = 50, 80, 13, 40
    rows_as_arms = stats.boschloo_exact([[k1, n1 - k1], [k2, n2 - k2]]).pvalue
    ours = boschloo_test(k1, n1, k2, n2).pvalue
    assert ours == pytest.approx(0.00228, abs=1e-5)
    assert abs(ours - rows_as_arms) > 1e-4


def test_greater_means_arm1_higher() -> None:
    res = compare_independent(50, 80, 13, 40, alternative="greater")
    assert res.primary.pvalue < 0.01
    assert compare_independent(50, 80, 13, 40, alternative="less").primary.pvalue > 0.99


def test_arms_table_orientation() -> None:
    assert arms_table(3, 10, 4, 12).tolist() == [[3, 4], [7, 8]]


def test_odds_ratio_matches_scipy() -> None:
    oratio = odds_ratio(50, 80, 13, 40)
    ref = stats.contingency.odds_ratio([[50, 13], [30, 27]], kind="conditional")
    assert oratio.estimate == pytest.approx(ref.statistic)
    ci = ref.confidence_interval()
    assert oratio.interval.low == pytest.approx(ci.low)
    assert oratio.interval.high == pytest.approx(ci.high)
    assert oratio.estimate > 1  # arm 1 has higher odds


def test_degenerate_tables_do_not_crash() -> None:
    all_success = compare_independent(40, 40, 30, 30)
    assert all_success.difference == 0.0
    assert math.isnan(all_success.odds_ratio.estimate)
    zero_cell = compare_independent(0, 40, 10, 40)
    assert zero_cell.odds_ratio.estimate == 0.0
    assert zero_cell.primary.pvalue < 0.01


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="k1"):
        compare_independent(5, 4, 1, 4)
    with pytest.raises(ValueError, match="n2"):
        compare_independent(1, 4, 0, 0)
    with pytest.raises(ValueError, match="test"):
        compare_independent(1, 4, 1, 4, test="chi2")
    with pytest.raises(ValueError, match="alternative"):
        compare_independent(1, 4, 1, 4, alternative="bigger")  # type: ignore[arg-type]


# --- properties ------------------------------------------------------------------------

arm = st.integers(min_value=1, max_value=60).flatmap(
    lambda n: st.tuples(st.integers(min_value=0, max_value=n), st.just(n))
)


@given(arm, arm)
def test_newcombe_contains_difference_and_mirrors(a1: tuple[int, int], a2: tuple[int, int]) -> None:
    (k1, n1), (k2, n2) = a1, a2
    low, high = newcombe_interval(k1, n1, k2, n2)
    d = k1 / n1 - k2 / n2
    assert -1.0 <= low <= d <= high <= 1.0
    low_swapped, high_swapped = newcombe_interval(k2, n2, k1, n1)
    assert low == pytest.approx(-high_swapped, abs=1e-12)
    assert high == pytest.approx(-low_swapped, abs=1e-12)


@settings(max_examples=25, deadline=None)
@given(arm, arm)
def test_one_sided_boschloo_never_exceeds_fisher(a1: tuple[int, int], a2: tuple[int, int]) -> None:
    (k1, n1), (k2, n2) = a1, a2
    b = boschloo_test(k1, n1, k2, n2, alternative="greater").pvalue
    f = stats.fisher_exact(arms_table(k1, n1, k2, n2), alternative="greater").pvalue
    assert np.isnan(b) or b <= f + 1e-9
