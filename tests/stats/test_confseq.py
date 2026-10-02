"""Betting confidence sequences and the paired anytime test (v0.3).

Golden values (``data/confseq_golden.json``) come from ``confseq`` 0.0.11 (MIT,
github.com/gostevehoward/confseq), computed in a separate environment:
``confseq.betting.hedged_cs(x, alpha=0.05, theta=...)`` for theta 1/2, 1 and 0, and the
hedged capital ``max(K+/2, K-/2)`` from ``confseq.betting.betting_mart`` at m = 1/2 with
``lambda_predmix_eb(x, alpha=0.025)`` bets. confseq's C++ extension was not needed; the
pure-Python ``betting`` module was imported on its own.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fieldtrial.stats import betting_cs, capital_process, paired_anytime_test

GOLDEN = json.loads((Path(__file__).parent / "data" / "confseq_golden.json").read_text())
CASES = GOLDEN["hedged_cs"]
SIDES = {"two": "two-sided", "greater": "greater", "less": "less"}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("side", sorted(SIDES))
def test_golden_hedged_cs(case: str, side: str) -> None:
    row = CASES[case]
    cs = betting_cs(row["x"], 0.05, SIDES[side])  # type: ignore[arg-type]
    np.testing.assert_allclose(cs.lower, row[side][0], atol=1e-12)
    np.testing.assert_allclose(cs.upper, row[side][1], atol=1e-12)
    assert cs.level == 0.95


@pytest.mark.parametrize("case", sorted(CASES))
def test_golden_capital_process(case: str) -> None:
    row = CASES[case]
    np.testing.assert_allclose(capital_process(row["x"], 0.5), row["capital_half"], rtol=1e-12)


def test_interval_accessor_and_errors() -> None:
    cs = betting_cs([1, 1, 0, 1], 0.1)
    assert len(cs) == 4
    last = cs.interval()
    assert (last.low, last.high) == (cs.lower[-1], cs.upper[-1])
    assert cs.interval(1).low == cs.lower[0]
    with pytest.raises(ValueError, match="between 1 and 4"):
        cs.interval(5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        betting_cs([0.5, 1.2])
    with pytest.raises(ValueError, match="at least one"):
        betting_cs([])
    with pytest.raises(ValueError, match="alpha"):
        betting_cs([1, 0], alpha=1.5)
    with pytest.raises(ValueError, match="null"):
        capital_process([1, 0], 1.5)
    with pytest.raises(ValueError, match="one-dimensional"):
        betting_cs([[0.5, 1.0]])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="breaks"):
        betting_cs([1, 0], breaks=5)


def test_capital_shape() -> None:
    x = [1, 0, 1]
    assert capital_process(x, 0.5).shape == (3,)
    assert capital_process(x, [0.2, 0.5, 0.8]).shape == (3, 3)


def test_paired_test_detects_a_clear_difference() -> None:
    d = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1] * 4  # treatment wins most blocks
    res = paired_anytime_test(d, 0.05, "two-sided")
    assert res.n == 40
    assert res.estimate == pytest.approx(0.8)
    assert res.rejected_at is not None
    assert res.test.pvalue <= 0.05
    assert res.test.statistic >= 1 / 0.05
    # Once the null is rejected, the sequence (running intersection) excludes it too,
    # up to the grid padding.
    assert res.sequence.lower[-1] > 0
    assert res.sequence.method == "betting (paired)"


def test_paired_test_does_not_reject_balanced_data() -> None:
    d = [1, -1, 0, 0] * 10
    res = paired_anytime_test(d, 0.05)
    assert res.rejected_at is None
    assert res.test.pvalue > 0.05
    assert res.sequence.lower[-1] < 0 < res.sequence.upper[-1]


def test_paired_test_margin_and_validation() -> None:
    d = [0, 0, 1, 0, 0, -1, 0, 0] * 6  # equal arms
    # Non-inferiority with margin 0.3: H0 delta <= -0.3 is rejected for equal arms.
    ni = paired_anytime_test(d, 0.05, "greater", null=-0.3)
    assert ni.rejected_at is not None
    sup = paired_anytime_test(d, 0.05, "greater", null=0.0)
    assert sup.rejected_at is None
    with pytest.raises(ValueError, match="between -1 and 1"):
        paired_anytime_test([2, 0])
    with pytest.raises(ValueError, match="non-empty"):
        paired_anytime_test([])
    with pytest.raises(ValueError, match="strictly between"):
        paired_anytime_test([1, 0], null=1.0)


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=1, max_size=40),
    st.sampled_from(["two-sided", "greater", "less"]),
)
def test_sequence_is_nested_and_within_unit(x: list[float], alternative: str) -> None:
    cs = betting_cs(x, 0.1, alternative, breaks=200)  # type: ignore[arg-type]
    lo, hi = np.array(cs.lower), np.array(cs.upper)
    assert np.all((lo >= 0) & (hi <= 1))
    assert np.all(np.diff(lo) >= 0)
    assert np.all(np.diff(hi) <= 0)
    if alternative == "greater":
        assert np.all(hi == 1)
    if alternative == "less":
        assert np.all(lo == 0)


@settings(max_examples=40, deadline=None)
@given(st.lists(st.sampled_from([0.0, 0.25, 0.5, 0.75, 1.0]), min_size=1, max_size=30))
def test_sequence_is_symmetric_under_reflection(x: list[float]) -> None:
    breaks = 200
    a = betting_cs(x, 0.1, breaks=breaks)
    b = betting_cs([1 - v for v in x], 0.1, breaks=breaks)
    tol = 1 / breaks + 1e-9
    np.testing.assert_allclose(a.lower, 1 - np.array(b.upper), atol=tol)
    np.testing.assert_allclose(a.upper, 1 - np.array(b.lower), atol=tol)


@settings(max_examples=30, deadline=None)
@given(st.lists(st.sampled_from([-1.0, 0.0, 1.0]), min_size=1, max_size=40))
def test_paired_pvalue_matches_decision(d: list[float]) -> None:
    res = paired_anytime_test(d, 0.05, breaks=200)
    assert 0 < res.test.pvalue <= 1
    assert (res.rejected_at is not None) == (res.test.pvalue < 0.05)
    assert math.isclose(res.estimate, float(np.mean(d)))


@pytest.mark.slow
@pytest.mark.parametrize("p", [0.2, 0.5, 0.85])
def test_time_uniform_type_one_error(p: float) -> None:
    """Under the null, the capital ever crosses 1/alpha with probability at most alpha."""
    rng = np.random.default_rng(20261003)
    alpha, n, reps = 0.1, 300, 2000
    crossed = 0
    for _ in range(reps):
        x = (rng.random(n) < p).astype(float)
        crossed += bool(np.any(capital_process(x, p, alpha) > 1 / alpha))
    se = math.sqrt(alpha * (1 - alpha) / reps)
    assert crossed / reps <= alpha + 3 * se


@pytest.mark.slow
def test_time_uniform_coverage_of_the_sequence() -> None:
    rng = np.random.default_rng(7)
    alpha, n, reps, p = 0.1, 150, 400, 0.3
    missed = 0
    for _ in range(reps):
        x = (rng.random(n) < p).astype(float)
        cs = betting_cs(x, alpha, breaks=200)
        missed += bool(np.any((np.array(cs.lower) > p) | (np.array(cs.upper) < p)))
    assert missed / reps <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / reps)
