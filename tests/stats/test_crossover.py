"""Crossover rounds (two-period, two-order crossover with rounds as units).

Independent references:
- scipy.stats.ttest_ind on the period differences grouped by order: the Hills–Armitage
  t-test (Hills and Armitage, 1979; Senn, 2002, section 3.6).
- scipy.stats.permutation_test with permutation_type="independent": the exact
  randomization distribution over order labels, enumerated by a separate implementation.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from fieldtrial.stats import crossover
from fieldtrial.stats.crossover import crossover_test

PERIODS = [
    (0.80, 0.60),
    (0.70, 0.65),
    (0.90, 0.70),
    (0.75, 0.70),
    (0.55, 0.85),
    (0.60, 0.80),
    (0.50, 0.70),
    (0.65, 0.75),
]
ORDERS = ["AB", "AB", "AB", "AB", "BA", "BA", "BA", "BA"]


def _split(periods: list[tuple[float, float]], orders: list[str]) -> tuple[np.ndarray, np.ndarray]:
    u = np.array([a - b for a, b in periods])
    ab = np.array([o == "AB" for o in orders])
    return u[ab], u[~ab]


def test_matches_hills_armitage_t_test() -> None:
    res = crossover_test(PERIODS, ORDERS)  # type: ignore[arg-type]
    u_ab, u_ba = _split(PERIODS, ORDERS)
    assert res.difference == pytest.approx((u_ab.mean() - u_ba.mean()) / 2)
    assert res.period_effect == pytest.approx(-(u_ab.mean() + u_ba.mean()) / 2)
    t = stats.ttest_ind(u_ab, u_ba, equal_var=True)
    ci = t.confidence_interval(0.95)
    assert res.interval is not None
    assert (res.interval.low, res.interval.high) == pytest.approx((ci.low / 2, ci.high / 2))
    assert (res.cycles, res.ab, res.ba) == (8, 4, 4)


def _scipy_exact(u_ab: np.ndarray, u_ba: np.ndarray, alternative: str) -> float:
    res = stats.permutation_test(
        (u_ab, u_ba),
        lambda x, y: (np.mean(x) - np.mean(y)) / 2,
        permutation_type="independent",
        alternative=alternative,
        n_resamples=math.inf,
    )
    return float(res.pvalue)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_exact_pvalue_matches_scipy_permutation_test(alternative: str) -> None:
    res = crossover_test(PERIODS, ORDERS, alternative=alternative)  # type: ignore[arg-type]
    u_ab, u_ba = _split(PERIODS, ORDERS)
    assert res.test.test == "crossover_randomization"
    assert res.test.pvalue == pytest.approx(_scipy_exact(u_ab, u_ba, alternative), abs=1e-12)


@settings(max_examples=30, deadline=None)
@given(
    data=st.lists(
        st.tuples(st.integers(0, 10), st.integers(0, 10), st.sampled_from(["AB", "BA"])),
        min_size=3,
        max_size=9,
    )
)
def test_property_randomization_and_symmetry(data: list[tuple[int, int, str]]) -> None:
    periods = [(a / 10, b / 10) for a, b, _ in data]
    orders = [o for _, _, o in data]
    if len(set(orders)) < 2:
        return
    res = crossover_test(periods, orders)  # type: ignore[arg-type]
    u_ab, u_ba = _split(periods, orders)
    if min(len(u_ab), len(u_ba)) >= 2:
        # Integer data (tenths) so that scipy's tie handling is exact as well.
        ref = _scipy_exact(np.rint(u_ab * 10), np.rint(u_ba * 10), "two-sided")
        assert res.test.pvalue == pytest.approx(ref, abs=1e-9)
    # Swapping the arm labels flips the sign and keeps the p-value.
    flipped = crossover_test(periods, ["BA" if o == "AB" else "AB" for o in orders])  # type: ignore[misc]
    assert flipped.difference == pytest.approx(-res.difference, abs=1e-12)
    assert flipped.test.pvalue == pytest.approx(res.test.pvalue, abs=1e-12)
    # Halving the outcomes and lifting every first round by 0.5 halves the arm difference
    # and moves only the period effect.
    lowered = crossover_test([(0.5 + a / 2, b / 2) for a, b in periods], orders)  # type: ignore[arg-type]
    assert lowered.difference == pytest.approx(res.difference / 2, abs=1e-12)
    assert lowered.period_effect == pytest.approx(res.period_effect / 2 - 0.5, abs=1e-12)
    assert 0.0 < res.test.pvalue <= 1.0


def test_period_effect_is_separated_from_the_arm_effect() -> None:
    # Arm A is 0.1 higher; the second period of every cycle loses 0.2 (scene wear).
    periods, orders = [], []
    for i in range(6):
        order = "AB" if i % 2 == 0 else "BA"
        a, b = 0.8, 0.7
        first, second = (a, b) if order == "AB" else (b, a)
        periods.append((first, second - 0.2))
        orders.append(order)
    res = crossover_test(periods, orders)  # type: ignore[arg-type]
    assert res.difference == pytest.approx(0.1)
    assert res.period_effect == pytest.approx(-0.2)


def test_large_studies_fall_back_to_the_t_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crossover, "EXACT_LIMIT", 10)
    res = crossover_test(PERIODS, ORDERS)  # type: ignore[arg-type]
    u_ab, u_ba = _split(PERIODS, ORDERS)
    assert res.test.test == "hills_armitage_t"
    assert res.test.pvalue == pytest.approx(stats.ttest_ind(u_ab, u_ba).pvalue)


def test_invalid_input() -> None:
    with pytest.raises(ValueError, match="both orders"):
        crossover_test([(0.5, 0.5), (0.6, 0.4)], ["AB", "AB"])
    with pytest.raises(ValueError, match="same length"):
        crossover_test([(0.5, 0.5)], ["AB", "BA"])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        crossover_test([(1.5, 0.5), (0.5, 0.5)], ["AB", "BA"])
    with pytest.raises(ValueError, match="'AB' or 'BA'"):
        crossover_test([(0.5, 0.5), (0.5, 0.5)], ["AB", "BB"])  # type: ignore[list-item]
    two = crossover_test([(0.5, 0.4), (0.4, 0.6)], ["AB", "BA"])
    assert two.interval is None
