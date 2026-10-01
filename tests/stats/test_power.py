"""Planning functions. Golden values: docs/PLAN.md section 12 (sample size and MDE)."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    proportion_effectsize,
    samplesize_proportions_2indep_onetail,
)

from fieldtrial.stats import (
    boschloo_power,
    mcnemar_power,
    mde,
    power,
    sample_size,
    simulate_power,
)
from fieldtrial.stats.power import boschloo_pvalues, boschloo_rejection_region

GOLDEN_SAMPLE_SIZE = [
    # (p1, p2, pooled-z, fleiss-cc, arcsine)
    (0.76, 0.90, 111.8216, 125.7015, 108.46),
    (0.76, 0.93, 69.9517, 81.2908, 65.80),
    (0.30, 0.60, 41.9703, 48.4074, 41.79),
]


@pytest.mark.parametrize(("p1", "p2", "pooled", "fleiss", "arcsine"), GOLDEN_SAMPLE_SIZE)
def test_golden_sample_size(
    p1: float, p2: float, pooled: float, fleiss: float, arcsine: float
) -> None:
    for method, golden, digits in (
        ("pooled-z", pooled, 4),
        ("fleiss-cc", fleiss, 4),
        ("arcsine", arcsine, 2),
    ):
        res = sample_size(p1, p2, method=method)
        assert res.n1_exact == pytest.approx(golden, abs=0.5 * 10**-digits + 1e-12)
        assert res.n1 == math.ceil(res.n1_exact)
        assert res.n2 == res.n1


def test_sample_size_ceilings_match_the_spec() -> None:
    assert sample_size(0.76, 0.90).n1 == 112
    assert sample_size(0.76, 0.90, method="fleiss-cc").n1 == 126
    assert sample_size(0.76, 0.90, method="arcsine").n1 == 109
    assert sample_size(0.30, 0.60, method="fleiss-cc").n1 == 49


@pytest.mark.parametrize("ratio", [1.0, 0.5, 3.0])
@pytest.mark.parametrize(("p1", "p2"), [(0.76, 0.90), (0.30, 0.60), (0.9, 0.7)])
def test_pooled_matches_statsmodels_live(p1: float, p2: float, ratio: float) -> None:
    # statsmodels: diff = prop1 - prop2, prop2 = rate of sample 2, ratio = nobs2 / nobs1.
    ref = samplesize_proportions_2indep_onetail(
        p1 - p2, p2, 0.8, ratio=ratio, alpha=0.05, alternative="two-sided"
    )
    assert sample_size(p1, p2, ratio=ratio).n1_exact == pytest.approx(ref, rel=1e-9)


@pytest.mark.parametrize(("p1", "p2"), [(0.76, 0.90), (0.30, 0.60)])
def test_arcsine_matches_statsmodels_live(p1: float, p2: float) -> None:
    # statsmodels also counts the far tail, which shifts n by about 1e-5 relative.
    ref = NormalIndPower().solve_power(
        effect_size=proportion_effectsize(p2, p1), alpha=0.05, power=0.8, ratio=1
    )
    assert sample_size(p1, p2, method="arcsine").n1_exact == pytest.approx(ref, rel=1e-4)
    for n in (20, 60, 150):
        expected = NormalIndPower().power(
            proportion_effectsize(p2, p1), nobs1=n, alpha=0.05, ratio=1
        )
        assert power(p1, p2, n, method="arcsine").power == pytest.approx(expected, rel=1e-9)


def test_one_sided_and_unequal_allocation() -> None:
    two = sample_size(0.76, 0.90)
    one = sample_size(0.76, 0.90, alternative="less")
    assert one.n1_exact < two.n1_exact
    unequal = sample_size(0.76, 0.90, ratio=3.0)
    assert unequal.n2_exact == pytest.approx(3 * unequal.n1_exact)
    assert unequal.n1_exact < two.n1_exact < unequal.n2_exact


GOLDEN_MDE = [
    # (baseline, n, up_target, up_pp, down_target, down_pp)
    (0.76, 40, 0.9707, 21.1, 0.4588, -30.1),
    (0.76, 120, 0.8958, 13.6, 0.5915, -16.8),
    (0.50, 40, 0.7949, 29.5, 0.2051, -29.5),
]


@pytest.mark.parametrize(("p", "n", "up", "up_pp", "down", "down_pp"), GOLDEN_MDE)
def test_golden_mde(p: float, n: int, up: float, up_pp: float, down: float, down_pp: float) -> None:
    inc = mde(p, n)
    dec = mde(p, n, direction="decrease")
    assert inc.target == pytest.approx(up, abs=5e-5)
    assert dec.target == pytest.approx(down, abs=5e-5)
    assert inc.effect is not None
    assert dec.effect is not None
    assert inc.effect * 100 == pytest.approx(up_pp, abs=0.05)
    assert dec.effect * 100 == pytest.approx(down_pp, abs=0.05)


def test_mde_round_trips_through_sample_size() -> None:
    res = mde(0.76, 40, 120)
    assert res.target is not None
    assert sample_size(0.76, res.target, ratio=3.0).n1_exact == pytest.approx(40, rel=1e-8)


def test_mde_unreachable_returns_none() -> None:
    res = mde(0.5, 3)
    assert res.target is None
    assert res.effect is None


@pytest.mark.parametrize(("p1", "p2"), [(0.76, 0.90), (0.3, 0.6), (0.9, 0.75)])
@pytest.mark.parametrize("method", ["pooled-z", "fleiss-cc", "arcsine"])
def test_power_at_the_planned_size_reaches_the_target(p1: float, p2: float, method: str) -> None:
    n = sample_size(p1, p2, method=method).n1
    assert power(p1, p2, n, method=method).power >= 0.80
    assert power(p1, p2, n - 2, method=method).power < 0.80


def test_power_equals_alpha_without_a_difference() -> None:
    assert power(0.7, 0.7, 50).power == pytest.approx(0.05)


def test_power_alternatives_point_the_right_way() -> None:
    # "greater" means p1 > p2.
    assert power(0.9, 0.76, 80, alternative="greater").power > 0.5
    assert power(0.9, 0.76, 80, alternative="less").power < 0.05


def test_invalid_planning_inputs() -> None:
    with pytest.raises(ValueError, match="differ"):
        sample_size(0.5, 0.5)
    with pytest.raises(ValueError, match="ratio"):
        sample_size(0.5, 0.6, ratio=0)
    with pytest.raises(ValueError, match="method"):
        sample_size(0.5, 0.6, method="exact")
    with pytest.raises(ValueError, match="p1"):
        power(1.0, 0.5, 10)
    with pytest.raises(ValueError, match="at least 1"):
        mde(0.5, 0)


@given(
    st.floats(min_value=0.05, max_value=0.6),
    st.floats(min_value=0.02, max_value=0.15),
    st.floats(min_value=0.01, max_value=0.15),
    st.sampled_from(["pooled-z", "fleiss-cc", "arcsine"]),
)
def test_sample_size_shrinks_as_the_difference_grows(
    p1: float, d: float, extra: float, method: str
) -> None:
    small = sample_size(p1, p1 + d, method=method).n1_exact
    large = sample_size(p1, p1 + d + extra, method=method).n1_exact
    assert large < small


# --- exact power ----------------------------------------------------------------------------


@pytest.mark.parametrize("alternative", ["less", "greater"])
def test_grid_boschloo_pvalues_match_scipy(alternative: str) -> None:
    n1, n2 = 15, 12
    ours = boschloo_pvalues(n1, n2, alternative=alternative)  # type: ignore[arg-type]
    rng = np.random.default_rng(1)
    for x1, x2 in zip(rng.integers(0, n1 + 1, 25), rng.integers(0, n2 + 1, 25), strict=True):
        ref = stats.boschloo_exact([[x1, x2], [n1 - x1, n2 - x2]], alternative=alternative).pvalue
        assert ours[x1, x2] == pytest.approx(ref, rel=1e-6, abs=1e-12)


def test_boschloo_rejection_region_matches_scipy_decisions() -> None:
    n1, n2 = 10, 14
    region = boschloo_rejection_region(n1, n2, alpha=0.05)
    for x1 in range(n1 + 1):
        for x2 in range(0, n2 + 1, 3):
            p = stats.boschloo_exact([[x1, x2], [n1 - x1, n2 - x2]]).pvalue
            if abs(p - 0.05) > 1e-6:
                assert region[x1, x2] == (p <= 0.05)


def test_boschloo_power_is_close_to_closed_form_and_beats_fisher() -> None:
    exact = boschloo_power(0.76, 0.90, 112).power
    assert exact == pytest.approx(0.80, abs=0.03)
    fisher = simulate_power(0.5, 0.85, 20, test="fisher", n_sims=400, seed=3).power
    boschloo = simulate_power(0.5, 0.85, 20, test="boschloo", n_sims=400, seed=3).power
    assert boschloo >= fisher


def _mcnemar_power_by_trinomial(p10: float, p01: float, n: int, alpha: float) -> float:
    total = 0.0
    for b in range(n + 1):
        for c in range(n + 1 - b):
            prob = stats.multinomial.pmf([b, c, n - b - c], n, [p10, p01, 1 - p10 - p01])
            p = 1.0 if b + c == 0 else stats.binomtest(b, b + c, 0.5).pvalue
            total += prob * (p <= alpha)
    return total


@pytest.mark.parametrize(("p10", "p01", "n"), [(0.25, 0.05, 30), (0.2, 0.1, 25), (0.1, 0.1, 20)])
def test_mcnemar_power_matches_trinomial_enumeration(p10: float, p01: float, n: int) -> None:
    ours = mcnemar_power(p10, p01, n).power
    assert ours == pytest.approx(
        _mcnemar_power_by_trinomial(p10, p01, n, 0.05), rel=1e-9, abs=1e-12
    )


def test_mcnemar_power_one_sided_and_invalid() -> None:
    assert (
        mcnemar_power(0.25, 0.05, 30, alternative="greater").power
        > mcnemar_power(0.25, 0.05, 30).power
    )
    with pytest.raises(ValueError, match="exceed"):
        mcnemar_power(0.6, 0.5, 10)


def test_simulation_is_seeded() -> None:
    a = simulate_power(0.6, 0.8, 15, n_sims=200, seed=7)
    b = simulate_power(0.6, 0.8, 15, n_sims=200, seed=7)
    assert a.power == b.power
    assert a.standard_error is not None


@settings(max_examples=20, deadline=None)
@given(st.floats(min_value=0.05, max_value=0.95), st.floats(min_value=0.05, max_value=0.95))
def test_boschloo_power_is_a_probability(p1: float, p2: float) -> None:
    value = boschloo_power(p1, p2, 12, 9).power
    assert 0.0 <= value <= 1.0


def test_one_sided_exact_power_and_validation() -> None:
    region = boschloo_rejection_region(12, 12, alternative="less")
    assert region[0, 12]  # arm 1 far below arm 2
    assert not region[12, 0]
    assert mcnemar_power(0.05, 0.25, 30, alternative="less").power > 0.5
    with pytest.raises(ValueError, match="at least 1"):
        power(0.5, 0.6, 0)
    with pytest.raises(ValueError, match="at least 1"):
        boschloo_rejection_region(0, 5)
    with pytest.raises(ValueError, match="at least 1"):
        mcnemar_power(0.2, 0.1, 0)
    with pytest.raises(ValueError, match="at least 1"):
        simulate_power(0.5, 0.6, 0)
    with pytest.raises(ValueError, match="n_sims"):
        simulate_power(0.5, 0.6, 10, n_sims=0)
