"""Stages, timing, stratified comparison and drift (docs/PLAN.md section 12)."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable, Table

from fieldtrial.stats import proportion_ci
from fieldtrial.stats._rng import StableRng
from fieldtrial.stats.drift import homogeneity_test
from fieldtrial.stats.ordinal import compare_stages, stage_distribution, stage_funnel
from fieldtrial.stats.stratified import cmh_test
from fieldtrial.stats.timing import median_time_to_success, success_curve

# --- stages ----------------------------------------------------------------------------------

STAGES = [-1, 0, 0, 1, 1, 1, 2, 3, 3, 3]  # 4 stages, success = 3


def test_stage_distribution() -> None:
    dist = stage_distribution(STAGES, 4)
    assert [(d.stage, d.count) for d in dist] == [(-1, 1), (0, 2), (1, 3), (2, 1), (3, 3)]
    assert sum(d.share for d in dist) == pytest.approx(1.0)


def test_stage_funnel_matches_manual_counts_and_wilson() -> None:
    funnel = stage_funnel(STAGES, 4)
    assert [(f.entered, f.reached) for f in funnel] == [(10, 9), (9, 7), (7, 4), (4, 3)]
    for f in funnel:
        ref = proportion_ci(f.reached, f.entered).interval
        assert (f.interval.low, f.interval.high) == pytest.approx((ref.low, ref.high))
    assert funnel[-1].cumulative == pytest.approx(0.3)  # = success rate


def test_funnel_with_nobody_entering() -> None:
    funnel = stage_funnel([-1, -1], 2)
    assert funnel[1].entered == 0
    assert math.isnan(funnel[1].conversion)


@pytest.mark.parametrize("bad", [[4], [-2], [[1]]])
def test_invalid_stage_indices(bad: list[int]) -> None:
    with pytest.raises(ValueError, match="stage"):
        stage_distribution(bad, 4)


def _brunner_munzel_reference(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Brunner and Munzel (2000), written from the paper: statistic for H0 p = 1/2, df, p."""
    n1, n2 = len(x), len(y)
    allr = stats.rankdata(np.concatenate([x, y]))
    r1, r2 = allr[:n1], allr[n1:]
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    m1, m2 = r1.mean(), r2.mean()
    s1 = np.sum((r1 - rx - m1 + (n1 + 1) / 2) ** 2) / (n1 - 1)
    s2 = np.sum((r2 - ry - m2 + (n2 + 1) / 2) ** 2) / (n2 - 1)
    w = n1 * n2 * (m2 - m1) / ((n1 + n2) * math.sqrt(n1 * s1 + n2 * s2))
    df = (n1 * s1 + n2 * s2) ** 2 / ((n1 * s1) ** 2 / (n1 - 1) + (n2 * s2) ** 2 / (n2 - 1))
    return w, 2 * stats.t.sf(abs(w), df)


def test_brunner_munzel_matches_paper_formula_and_scipy_example() -> None:
    # Example from scipy's documentation of brunnermunzel.
    x1 = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1])
    x2 = np.array([3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4])
    res = compare_stages(x2, x1)  # arm A = x2 tends to be higher
    w, p = _brunner_munzel_reference(x1, x2)
    assert abs(res.statistic) == pytest.approx(abs(w), rel=1e-12)
    assert res.pvalue == pytest.approx(p, rel=1e-9)
    assert res.pvalue == pytest.approx(0.0057862086661515377, rel=1e-9)


def test_brunner_munzel_direction() -> None:
    high, low = [3, 3, 2, 3, 3, 2, 3], [0, 1, 0, 1, 2, 0, 1]
    assert compare_stages(high, low, alternative="greater").pvalue < 0.01
    assert compare_stages(high, low, alternative="less").pvalue > 0.99


def test_brunner_munzel_degenerate_and_invalid() -> None:
    assert math.isnan(compare_stages([2, 2, 2], [2, 2]).pvalue)
    with pytest.raises(ValueError, match="at least 2"):
        compare_stages([1], [1, 2])


# --- timing ----------------------------------------------------------------------------------


def test_success_curve() -> None:
    curve = success_curve([5.0, 3.0, 8.0, 3.0, 45.0], [True, True, False, True, False])
    assert curve.times == (3.0, 5.0)
    assert curve.fraction == pytest.approx((0.4, 0.6))
    assert curve.at(2.9) == 0.0
    assert curve.at(4.0) == pytest.approx(0.4)
    assert curve.at(100.0) == pytest.approx(0.6)  # ends at the success rate


def test_median_time_and_bootstrap() -> None:
    d = [12.0, 15.5, 9.0, 20.0, 11.0, 14.0, 13.5, 30.0]
    res = median_time_to_success(d, seed=3)
    assert res.median == pytest.approx(float(np.median(d)))
    assert res.interval.low <= res.median <= res.interval.high
    assert median_time_to_success(d, seed=3) == res  # seeded: identical
    assert median_time_to_success(d, seed=4).interval != res.interval
    assert math.isnan(median_time_to_success([]).median)
    single = median_time_to_success([7.0])
    assert (single.interval.low, single.interval.high) == (7.0, 7.0)


def test_timing_invalid() -> None:
    with pytest.raises(ValueError, match="same length"):
        success_curve([1.0, 2.0], [True])
    with pytest.raises(ValueError, match="non-negative"):
        median_time_to_success([-1.0])
    with pytest.raises(ValueError, match="n_boot"):
        median_time_to_success([1.0], n_boot=0)


def test_vectorized_integers_are_uniform_and_seeded() -> None:
    a = StableRng(9, 4).integers(7, 7000)
    assert set(np.unique(a)) == set(range(7))
    assert np.array_equal(a, StableRng(9, 4).integers(7, 7000))
    assert np.all(StableRng(1, 1).integers(1, 5) == 0)


# --- CMH -------------------------------------------------------------------------------------

STRATA = [(8, 10, 5, 10), (6, 9, 2, 8), (9, 12, 7, 11), (4, 6, 3, 7)]


def _sm_table(strata: list[tuple[int, int, int, int]]) -> StratifiedTable:
    tables = np.array([[[k1, k2], [n1 - k1, n2 - k2]] for k1, n1, k2, n2 in strata], dtype=float)
    return StratifiedTable(np.moveaxis(tables, 0, -1))


@pytest.mark.parametrize("correction", [False, True])
def test_cmh_matches_statsmodels(correction: bool) -> None:
    ours = cmh_test(STRATA, correction=correction)
    ref = _sm_table(STRATA)
    null = ref.test_null_odds(correction=correction)
    assert ours.test.statistic == pytest.approx(null.statistic, rel=1e-12)
    assert ours.test.pvalue == pytest.approx(null.pvalue, rel=1e-12)
    assert ours.odds_ratio == pytest.approx(ref.oddsratio_pooled, rel=1e-12)
    low, high = ref.oddsratio_pooled_confint()
    assert ours.interval.low == pytest.approx(low, rel=1e-9)
    assert ours.interval.high == pytest.approx(high, rel=1e-9)


@given(
    st.lists(
        st.tuples(st.integers(2, 15), st.integers(2, 15)).flatmap(
            lambda nn: st.tuples(
                st.integers(0, nn[0]), st.just(nn[0]), st.integers(0, nn[1]), st.just(nn[1])
            )
        ),
        min_size=2,
        max_size=6,
    )
)
def test_cmh_matches_statsmodels_on_random_strata(strata: list[tuple[int, int, int, int]]) -> None:
    ours = cmh_test(strata)
    if math.isnan(ours.test.statistic):
        return  # no variation in any stratum: the test is undefined
    ref = _sm_table(strata).test_null_odds()
    assert ours.test.pvalue == pytest.approx(ref.pvalue, rel=1e-9, abs=1e-12)


def test_cmh_skips_empty_strata_and_validates() -> None:
    assert cmh_test([*STRATA, (0, 1, 0, 0)]).strata_used == 4
    with pytest.raises(ValueError, match="no stratum"):
        cmh_test([(0, 1, 0, 0)])
    with pytest.raises(ValueError, match="cannot exceed"):
        cmh_test([(3, 2, 0, 2)])
    assert math.isnan(cmh_test([(0, 2, 0, 2)]).test.pvalue)
    one_sided = cmh_test([(5, 5, 0, 5), (4, 4, 0, 4)])
    assert one_sided.odds_ratio == math.inf


# --- drift -----------------------------------------------------------------------------------


def test_two_groups_use_fisher_golden() -> None:
    res = homogeneity_test([(50, 80), (13, 40)])
    assert res.method == "fisher"
    assert res.pvalue == pytest.approx(0.00337, abs=1e-5)  # docs/PLAN.md section 12


def test_several_groups_match_statsmodels_chi_square() -> None:
    groups = [(30, 40), (25, 40), (36, 40), (20, 40)]
    res = homogeneity_test(groups)
    table = Table(np.array([[k, n - k] for k, n in groups], dtype=float))
    ref = table.test_nominal_association()
    assert res.method == "chi-square"
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert res.pvalue == pytest.approx(ref.pvalue, rel=1e-12)
    assert res.df == 3


def test_homogeneity_edge_cases() -> None:
    assert homogeneity_test([(10, 10), (5, 5), (0, 0)]).pvalue == 1.0
    assert homogeneity_test([(1, 3), (2, 3), (0, 3)]).small_expected
    with pytest.raises(ValueError, match="at least 2 groups"):
        homogeneity_test([(3, 4), (0, 0)])


def test_inputs_must_be_flat() -> None:
    with pytest.raises(ValueError, match="flat"):
        stage_funnel([[0, 1]], 2)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="flat"):
        success_curve([[1.0]], [[True]])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="n_stages"):
        stage_funnel([0], 0)
