"""Paired designs. Golden values: docs/PLAN.md section 12 (exact McNemar).

The Tango interval is checked against an independent implementation that finds the
constrained maximum-likelihood estimate numerically instead of with Tango's closed form.
Reference values from R (PropCIs::scoreci.mp) will be added once available.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import optimize, stats
from statsmodels.stats.contingency_tables import cochrans_q as sm_cochrans_q
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

from fieldtrial.stats import (
    adjust_pvalues,
    cochran_q,
    compare_paired,
    mcnemar_exact,
    tango_interval,
)

GOLDEN_MCNEMAR = [((10, 2), 0.03857), ((7, 1), 0.07031), ((5, 5), 1.0)]


@pytest.mark.parametrize(("bc", "pvalue"), GOLDEN_MCNEMAR)
def test_golden_mcnemar(bc: tuple[int, int], pvalue: float) -> None:
    b, c = bc
    res = mcnemar_exact(b, c)
    assert res.pvalue == pytest.approx(pvalue, abs=1e-5)
    # statsmodels reads b from cell [0, 1] and c from cell [1, 0].
    assert res.pvalue == pytest.approx(sm_mcnemar([[20, b], [c, 9]], exact=True).pvalue, rel=1e-12)


def test_mcnemar_alternatives_and_no_discordant_pairs() -> None:
    assert mcnemar_exact(10, 2, alternative="greater").pvalue == pytest.approx(
        stats.binomtest(10, 12, 0.5, alternative="greater").pvalue
    )
    assert mcnemar_exact(10, 2, alternative="less").pvalue > 0.99
    assert mcnemar_exact(0, 0).pvalue == 1.0


# --- Tango -------------------------------------------------------------------------------


def _independent_tango(b: int, c: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Tango's interval with the constrained MLE found by numerical optimization."""
    z = stats.norm.isf((1 - level) / 2)
    m = n - b - c

    def score(delta: float) -> float:
        lo, hi = max(0.0, -delta), (1.0 - delta) / 2.0
        if hi - lo <= 1e-15:
            q = lo
        else:

            def negloglik(q: float) -> float:
                total = 0.0
                for count, prob in ((b, q + delta), (c, q), (m, 1 - 2 * q - delta)):
                    if count:
                        total -= count * math.log(max(prob, 1e-300))
                return total

            q = optimize.minimize_scalar(
                negloglik, bounds=(lo, hi), method="bounded", options={"xatol": 1e-13}
            ).x
        var = n * (2 * q + delta - delta * delta)
        num = b - c - n * delta
        if var <= 0:
            return 0.0 if num == 0 else math.copysign(math.inf, num)
        return num / math.sqrt(var)

    est = (b - c) / n
    upper = 1.0 if est >= 1 else optimize.brentq(lambda d: score(d) + z, est, 1 - 1e-12)
    lower = -1.0 if est <= -1 else optimize.brentq(lambda d: score(d) - z, -1 + 1e-12, est)
    return lower, upper


TANGO_CASES = [
    (10, 2, 40),
    (2, 10, 40),
    (7, 1, 40),
    (5, 5, 40),
    (0, 6, 30),
    (12, 3, 50),
    (1, 7, 21),
    (20, 10, 200),
]


@pytest.mark.parametrize(("b", "c", "n"), TANGO_CASES)
def test_tango_matches_independent_implementation(b: int, c: int, n: int) -> None:
    ours = tango_interval(b, c, n)
    ref = _independent_tango(b, c, n)
    assert ours[0] == pytest.approx(ref[0], abs=1e-6)
    assert ours[1] == pytest.approx(ref[1], abs=1e-6)


def test_tango_known_value() -> None:
    low, high = tango_interval(10, 2, 40)
    assert low == pytest.approx(0.034743, abs=1e-6)
    assert high == pytest.approx(0.366316, abs=1e-6)


def test_tango_with_no_discordant_pairs() -> None:
    z2 = stats.norm.isf(0.025) ** 2
    low, high = tango_interval(0, 0, 20)
    assert high == pytest.approx(z2 / (20 + z2), abs=1e-9)
    assert low == pytest.approx(-z2 / (20 + z2), abs=1e-9)


def test_tango_at_the_boundary() -> None:
    low, high = tango_interval(30, 0, 30)
    assert high == 1.0
    assert low == pytest.approx(_independent_tango(30, 0, 30)[0], abs=1e-6)
    low, high = tango_interval(0, 30, 30)
    assert low == -1.0
    assert high == pytest.approx(_independent_tango(0, 30, 30)[1], abs=1e-6)


def test_tango_invalid() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        tango_interval(5, 6, 10)
    with pytest.raises(ValueError, match="at least 1"):
        tango_interval(0, 0, 0)


@given(
    st.integers(min_value=1, max_value=150)
    .flatmap(lambda n: st.tuples(st.integers(0, n), st.integers(0, n), st.just(n)))
    .filter(lambda t: t[0] + t[1] <= t[2])
)
def test_tango_contains_estimate_and_mirrors(bcn: tuple[int, int, int]) -> None:
    b, c, n = bcn
    low, high = tango_interval(b, c, n)
    assert -1.0 <= low <= (b - c) / n <= high <= 1.0
    low_swapped, high_swapped = tango_interval(c, b, n)
    assert low == pytest.approx(-high_swapped, abs=1e-9)
    assert high == pytest.approx(-low_swapped, abs=1e-9)


def test_compare_paired() -> None:
    res = compare_paired(10, 2, 40)
    assert res.difference == pytest.approx(0.2)
    assert res.interval is not None
    assert res.interval.method == "tango"
    assert res.test.pvalue == pytest.approx(0.03857, abs=1e-5)
    without_n = compare_paired(10, 2)
    assert without_n.difference is None
    assert without_n.interval is None


# --- Cochran's Q ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_cochran_q_matches_statsmodels(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = (rng.random((30, 4)) < [0.5, 0.6, 0.7, 0.8]).astype(int)
    ours = cochran_q(x)
    ref = sm_cochrans_q(x)
    assert ours.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert ours.pvalue == pytest.approx(ref.pvalue, rel=1e-12)
    assert ours.df == 3
    assert len(ours.comparisons) == 6


def test_cochran_q_pairwise_holm() -> None:
    x = np.array([[1, 0, 1]] * 9 + [[1, 1, 1]] * 5 + [[0, 0, 1]] * 2)
    res = cochran_q(x, pairs="vs-first")
    assert [(c.arm_a, c.arm_b) for c in res.comparisons] == [(1, 0), (2, 0)]
    first = res.comparisons[0]
    assert (first.b, first.c) == (0, 9)
    expected = adjust_pvalues([c.test.pvalue for c in res.comparisons], method="holm")
    assert [c.adjusted_pvalue for c in res.comparisons] == list(expected.adjusted)
    assert [c.reject for c in res.comparisons] == list(expected.reject)


def test_cochran_q_all_blocks_agree() -> None:
    res = cochran_q([[1, 1, 1], [0, 0, 0]])
    assert res.statistic == 0.0
    assert res.pvalue == 1.0


@pytest.mark.parametrize("bad", [[[1]], [[0, 2]], [1, 0], []])
def test_cochran_q_invalid(bad: list[object]) -> None:
    with pytest.raises(ValueError, match="outcomes"):
        cochran_q(bad)  # type: ignore[arg-type]
