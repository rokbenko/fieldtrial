"""One-arm intervals and the threshold test.

Golden values: docs/PLAN.md section 12 (scipy 1.17.1, statsmodels 0.15.0). Where statsmodels
implements the same method, the value is also re-derived live.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from fieldtrial.stats import proportion_ci

# Aliased so pytest does not collect the library function as a test.
from fieldtrial.stats import test_vs_threshold as threshold_test

GOLDEN = [
    # (k, n, method, low, high)
    (91, 120, "wilson", 0.6745, 0.8261),
    (36, 40, "wilson", 0.7695, 0.9604),
    (13, 40, "wilson", 0.2008, 0.4798),
    (39, 40, "wilson", 0.8712, 0.9956),
    (0, 40, "wilson", 0.0000, 0.0876),
    (40, 40, "wilson", 0.9124, 1.0000),
    (91, 120, "clopper-pearson", 0.6717, 0.8318),
    (36, 40, "clopper-pearson", 0.7634, 0.9721),
    (13, 40, "clopper-pearson", 0.1857, 0.4913),
    (39, 40, "clopper-pearson", 0.8684, 0.9994),
    (0, 40, "clopper-pearson", 0.0000, 0.0881),
    (40, 40, "clopper-pearson", 0.9119, 1.0000),
    (91, 120, "jeffreys", 0.6762, 0.8282),
    (36, 40, "jeffreys", 0.7796, 0.9653),
    (13, 40, "jeffreys", 0.1960, 0.4784),
    (39, 40, "jeffreys", 0.8891, 0.9973),
    (36, 40, "agresti-coull", 0.7638, 0.9661),
]

STATSMODELS_NAME = {
    "wilson": "wilson",
    "clopper-pearson": "beta",
    "jeffreys": "jeffreys",
    "agresti-coull": "agresti_coull",
}
METHODS = tuple(STATSMODELS_NAME)


@pytest.mark.parametrize(("k", "n", "method", "low", "high"), GOLDEN)
def test_golden_intervals(k: int, n: int, method: str, low: float, high: float) -> None:
    est = proportion_ci(k, n, method=method)
    assert est.estimate == pytest.approx(k / n)
    assert est.interval.low == pytest.approx(low, abs=1e-4)
    assert est.interval.high == pytest.approx(high, abs=1e-4)
    assert est.interval.level == 0.95
    assert est.interval.method == method


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(("k", "n"), [(91, 120), (36, 40), (13, 40), (39, 40), (1, 7), (5, 9)])
@pytest.mark.parametrize("level", [0.8, 0.95, 0.99])
def test_matches_statsmodels_live(k: int, n: int, method: str, level: float) -> None:
    lo, hi = proportion_confint(k, n, alpha=1 - level, method=STATSMODELS_NAME[method])
    est = proportion_ci(k, n, level=level, method=method)
    assert est.interval.low == pytest.approx(max(lo, 0.0), abs=1e-10)
    assert est.interval.high == pytest.approx(min(hi, 1.0), abs=1e-10)


def test_wilson_half_width_at_20_of_40_is_about_15_points() -> None:
    est = proportion_ci(20, 40)
    assert est.interval.width / 2 == pytest.approx(0.148, abs=5e-4)


def test_jeffreys_boundary_convention() -> None:
    # Brown, Cai and DasGupta (2001): the bound at the observed edge is 0 or 1.
    assert proportion_ci(0, 40, method="jeffreys").interval.low == 0.0
    assert proportion_ci(40, 40, method="jeffreys").interval.high == 1.0


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((1, 0), "at least 1"),
        ((5, 4), "cannot exceed"),
        ((-1, 4), "non-negative"),
        ((1.5, 4), "integer"),
        ((True, 4), "integer"),
    ],
)
def test_invalid_counts(args: tuple[object, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        proportion_ci(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5, float("nan")])
def test_invalid_level(level: float) -> None:
    with pytest.raises(ValueError, match="level"):
        proportion_ci(3, 10, level=level)


def test_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        proportion_ci(3, 10, method="wald")


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize(
    ("k", "n", "p0"), [(36, 40, 0.9), (39, 40, 0.9), (30, 40, 0.9), (0, 5, 0.2)]
)
def test_threshold_test_matches_scipy(k: int, n: int, p0: float, alternative: str) -> None:
    res = threshold_test(k, n, p0, alternative=alternative)  # type: ignore[arg-type]
    ref = stats.binomtest(k, n, p0, alternative=alternative)
    assert res.pvalue == pytest.approx(ref.pvalue, rel=1e-12)
    assert res.statistic == pytest.approx(k / n)
    assert res.test == "binomial"


def test_threshold_must_be_inside_unit_interval() -> None:
    with pytest.raises(ValueError, match="p0"):
        threshold_test(3, 10, 1.0)


# --- properties ------------------------------------------------------------------------

counts = st.integers(min_value=1, max_value=400).flatmap(
    lambda n: st.tuples(st.integers(min_value=0, max_value=n), st.just(n))
)
levels = st.floats(min_value=0.5, max_value=0.999)


@given(counts, levels, st.sampled_from(METHODS))
def test_interval_is_ordered_inside_unit_and_contains_estimate(
    kn: tuple[int, int], level: float, method: str
) -> None:
    k, n = kn
    est = proportion_ci(k, n, level=level, method=method)
    assert 0.0 <= est.interval.low <= est.estimate <= est.interval.high <= 1.0


@given(counts, levels, st.sampled_from(METHODS))
def test_interval_is_symmetric_under_relabeling(
    kn: tuple[int, int], level: float, method: str
) -> None:
    k, n = kn
    a = proportion_ci(k, n, level=level, method=method).interval
    b = proportion_ci(n - k, n, level=level, method=method).interval
    assert a.low == pytest.approx(1 - b.high, abs=1e-9)
    assert a.high == pytest.approx(1 - b.low, abs=1e-9)


@given(counts, st.sampled_from(METHODS), levels, levels)
def test_interval_widens_with_level(kn: tuple[int, int], method: str, l1: float, l2: float) -> None:
    k, n = kn
    lo_level, hi_level = sorted((l1, l2))
    narrow = proportion_ci(k, n, level=lo_level, method=method).interval
    wide = proportion_ci(k, n, level=hi_level, method=method).interval
    assert wide.low <= narrow.low + 1e-12
    assert wide.high >= narrow.high - 1e-12


def test_result_helpers() -> None:
    interval = proportion_ci(36, 40).interval
    assert interval.contains(0.9)
    assert not interval.contains(0.5)
    result = threshold_test(39, 40, 0.8, alternative="greater")
    assert result.rejects(0.05)
    assert not result.rejects(1e-9)
