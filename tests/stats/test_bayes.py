"""Bayesian summaries. Golden values: docs/PLAN.md section 12, Beta(1, 1) priors.

The independent reference is Evan Miller's closed form for P(p_B > p_A), exact when the
posterior's first parameter for B is an integer.
"""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats
from scipy.special import betaln

from fieldtrial.stats import credible_interval, prob_superiority

GOLDEN = [
    ((91, 120), (74, 80), 0.99902),
    ((91, 120), (36, 40), 0.97068),
    ((13, 40), (50, 80), 0.99900),
    ((91, 120), (57, 80), 0.23101),
]


def _miller(ka: int, na: int, kb: int, nb: int) -> float:
    a_a, b_a, a_b, b_b = ka + 1, na - ka + 1, kb + 1, nb - kb + 1
    return sum(
        math.exp(
            betaln(a_a + i, b_a + b_b) - math.log(b_b + i) - betaln(1 + i, b_b) - betaln(a_a, b_a)
        )
        for i in range(a_b)
    )


@pytest.mark.parametrize(("a", "b", "golden"), GOLDEN)
def test_golden_prob_superiority(a: tuple[int, int], b: tuple[int, int], golden: float) -> None:
    value = prob_superiority(*a, *b)
    assert value == pytest.approx(golden, abs=1e-5)
    assert value == pytest.approx(_miller(*a, *b), abs=1e-9)


@given(
    st.integers(1, 60).flatmap(lambda n: st.tuples(st.integers(0, n), st.just(n))),
    st.integers(1, 60).flatmap(lambda n: st.tuples(st.integers(0, n), st.just(n))),
)
def test_prob_superiority_matches_closed_form_and_complements(
    a: tuple[int, int], b: tuple[int, int]
) -> None:
    forward = prob_superiority(*a, *b)
    assert forward == pytest.approx(_miller(*a, *b), abs=1e-8)
    # Continuous posteriors: P(B > A) + P(A > B) = 1.
    assert forward + prob_superiority(*b, *a) == pytest.approx(1.0, abs=1e-8)


def test_identical_data_gives_one_half() -> None:
    assert prob_superiority(30, 40, 30, 40) == pytest.approx(0.5, abs=1e-9)


def test_credible_interval() -> None:
    ci = credible_interval(36, 40, prior=(0.5, 0.5))
    assert ci.low == pytest.approx(0.7796, abs=1e-4)  # Jeffreys, docs/PLAN.md section 12
    assert ci.high == pytest.approx(0.9653, abs=1e-4)
    uniform = credible_interval(36, 40)
    assert uniform.low == pytest.approx(stats.beta(37, 5).ppf(0.025))
    assert uniform.method == "beta-posterior(1,1)"


@pytest.mark.parametrize("prior", [(0, 1), (1, -1), (float("inf"), 1)])
def test_invalid_prior(prior: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="prior"):
        prob_superiority(1, 2, 1, 2, prior=prior)
