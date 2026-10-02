"""Checkpoint ladders: trend tests and plateau detection.

Golden values:
- Agresti (2013), Categorical Data Analysis, 3rd ed., section 3.4.6 (infant malformation by
  maternal alcohol consumption, scores 0, 0.5, 1.5, 4, 7): M^2 = 6.57, P = 0.01.
- statsmodels' Table.test_ordinal_association (linear-by-linear association) as an
  independent implementation of Mantel's statistic.
- The stratified variance is checked against the exact permutation variance, computed by
  enumerating every within-stratum arrangement of the outcomes.
"""

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.contingency_tables import Table

from fieldtrial.stats.compare import newcombe_interval
from fieldtrial.stats.ladder import plateau, plateau_paired, stratified_trend_test, trend_test
from fieldtrial.stats.paired import tango_interval

# (successes = malformation present, trials) per alcohol category.
AGRESTI = [(48, 17114), (38, 14502), (5, 793), (1, 127), (1, 38)]
AGRESTI_SCORES = [0.0, 0.5, 1.5, 4.0, 7.0]


def test_golden_agresti_malformation() -> None:
    res = trend_test(AGRESTI, scores=AGRESTI_SCORES)
    n = sum(t for _, t in AGRESTI)
    m2 = res.test.statistic**2 * (n - 1) / n
    assert m2 == pytest.approx(6.57, abs=0.005)
    assert res.test.pvalue == pytest.approx(0.0104, abs=5e-4)
    assert res.test.statistic > 0  # malformation rises with consumption


def _statsmodels_z(counts: list[tuple[int, int]], scores: list[float]) -> float:
    table = np.array([[n - k, k] for k, n in counts])
    r = Table(table, shift_zeros=False).test_ordinal_association(
        row_scores=np.array(scores), col_scores=np.array([0.0, 1.0])
    )
    return float(r.zscore)


def test_single_stratum_mantel_matches_statsmodels() -> None:
    res = stratified_trend_test([AGRESTI], scores=AGRESTI_SCORES)
    assert res.test.statistic == pytest.approx(_statsmodels_z(AGRESTI, AGRESTI_SCORES), rel=1e-9)


@settings(max_examples=40, deadline=None)
@given(
    data=st.lists(st.tuples(st.integers(1, 30), st.floats(0.0, 1.0)), min_size=2, max_size=6),
    seed=st.integers(0, 2**32 - 1),
)
def test_property_cochran_armitage_relation_to_mantel(
    data: list[tuple[int, float]], seed: int
) -> None:
    rng = np.random.default_rng(seed)
    counts = [(int(rng.binomial(n, p)), n) for n, p in data]
    total = sum(n for _, n in counts)
    ks = sum(k for k, _ in counts)
    ca = trend_test(counts).test.statistic
    mh = stratified_trend_test([counts]).test.statistic
    if ks in (0, total):
        assert math.isnan(ca)
        assert math.isnan(mh)
        return
    assert mh == pytest.approx(_statsmodels_z(counts, list(range(len(counts)))), rel=1e-9)
    assert ca**2 * (total - 1) / total == pytest.approx(mh**2, rel=1e-9)
    # Reversing the order flips the sign; an affine rescaling of the scores changes nothing.
    assert trend_test(counts[::-1]).test.statistic == pytest.approx(-ca, rel=1e-9)
    shifted = trend_test(counts, scores=[3.0 + 2.5 * i for i in range(len(counts))])
    assert shifted.test.statistic == pytest.approx(ca, rel=1e-9)


def _exact_moments(stratum: list[tuple[int, int]], scores: list[float]) -> tuple[float, float]:
    """Permutation mean and variance of sum_i x_i k_i given the stratum margins."""
    units = [x for (_, n), x in zip(stratum, scores, strict=True) for _ in range(n)]
    total_k = sum(k for k, _ in stratum)
    values = [
        sum(units[i] for i in chosen)
        for chosen in itertools.combinations(range(len(units)), total_k)
    ]
    return float(np.mean(values)), float(np.var(values))


def test_stratified_variance_is_the_exact_permutation_variance() -> None:
    strata = [
        [(1, 1), (0, 1), (1, 1), (1, 1)],
        [(0, 1), (0, 1), (1, 1), (1, 1)],
        [(2, 3), (1, 2), (0, 2), (2, 2)],
        [(1, 1), (1, 1), (1, 1), (1, 1)],  # no information
    ]
    scores = [0.0, 1.0, 2.0, 4.0]
    t_obs = mean_sum = var_sum = 0.0
    for s in strata:
        mean, var = _exact_moments(s, scores)
        t_obs += sum(x * k for (k, _), x in zip(s, scores, strict=True))
        mean_sum += mean
        var_sum += var
    res = stratified_trend_test(strata, scores=scores)
    assert res.test.statistic == pytest.approx((t_obs - mean_sum) / math.sqrt(var_sum), rel=1e-12)
    assert res.strata_used == 3


def test_alternatives_and_degenerate_data() -> None:
    rising = [(10, 40), (20, 40), (30, 40)]
    two = trend_test(rising)
    assert trend_test(rising, alternative="greater").test.pvalue == pytest.approx(
        two.test.pvalue / 2
    )
    assert trend_test(rising, alternative="less").test.pvalue > 0.99
    flat = trend_test([(40, 40), (40, 40)])
    assert math.isnan(flat.test.statistic)
    assert flat.test.pvalue == 1.0
    assert flat.strata_used == 0
    with pytest.raises(ValueError, match="scores"):
        trend_test(rising, scores=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="expected 3 scores"):
        trend_test(rising, scores=[1.0, 2.0])
    with pytest.raises(ValueError, match="cannot exceed"):
        trend_test([(5, 4), (1, 4)])
    with pytest.raises(ValueError, match="same number of arms"):
        stratified_trend_test([[(1, 1), (0, 1)], [(1, 1)]])


# --- plateau --------------------------------------------------------------------------


def test_plateau_independent_uses_newcombe_and_fixed_sequence() -> None:
    counts = [(20, 60), (45, 60), (52, 60), (54, 60), (55, 60)]
    res = plateau(counts, margin=0.15, alpha=0.05)
    assert res.reference == 4
    assert res.method == "newcombe"
    for step in res.steps:
        low, high = newcombe_interval(55, 60, *counts[step.index], level=0.90)
        assert (step.interval.low, step.interval.high) == pytest.approx((low, high))
    flags = {s.index: s.noninferior for s in res.steps}
    # Checkpoint 0 is far below the final one; the sequence stops there.
    assert flags[0] is False
    assert res.plateau_index is not None
    assert res.plateau_index >= 1
    assert all(flags[j] for j in range(res.plateau_index, 4))


def test_plateau_stops_at_first_failure() -> None:
    # Checkpoint 3 fails, so 0..2 are never tested even though 1 would pass on its own.
    counts = [(10, 40), (39, 40), (30, 40), (20, 40), (39, 40)]
    res = plateau(counts, margin=0.1)
    assert res.plateau_index is None
    assert [s.tested for s in res.steps] == [False, False, False, True]
    assert not any(s.noninferior for s in res.steps)


def test_plateau_paired_uses_tango() -> None:
    rng = np.random.default_rng(3)
    rates = [0.3, 0.8, 0.9, 0.9]
    m = (rng.random((80, 4)) < rates).astype(int)
    res = plateau_paired(m, margin=0.15)
    assert res.method == "tango"
    for step in res.steps:
        b = int(((m[:, -1] == 1) & (m[:, step.index] == 0)).sum())
        c = int(((m[:, -1] == 0) & (m[:, step.index] == 1)).sum())
        assert (step.interval.low, step.interval.high) == pytest.approx(
            tango_interval(b, c, 80, level=0.9)
        )
        assert step.difference == pytest.approx((b - c) / 80)
    assert res.plateau_index in (1, 2)
    with pytest.raises(ValueError, match="0 or 1"):
        plateau_paired([[0, 2]], margin=0.1)
    with pytest.raises(ValueError, match="margin"):
        plateau_paired(m, margin=0.0)


@pytest.mark.slow
def test_plateau_familywise_error_at_the_margin() -> None:
    """With every earlier checkpoint exactly delta below the final one, the chance of
    declaring any of them non-inferior is at most alpha (Monte Carlo, 2000 studies)."""
    rng = np.random.default_rng(20261002)
    rates = [0.75, 0.75, 0.75, 0.85]
    false = 0
    reps = 2000
    for _ in range(reps):
        counts = [(int(rng.binomial(80, p)), 80) for p in rates]
        if plateau(counts, margin=0.10).plateau_index is not None:
            false += 1
    rate = false / reps
    assert rate <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / reps)


@pytest.mark.slow
def test_trend_type_one_error_paired_blocks() -> None:
    rng = np.random.default_rng(7)
    reps, rejects = 3000, 0
    for _ in range(reps):
        m = (rng.random((40, 4)) < 0.7).astype(int)
        strata = [[(int(v), 1) for v in row] for row in m]
        if stratified_trend_test(strata).test.pvalue <= 0.05:
            rejects += 1
    rate = rejects / reps
    assert abs(rate - 0.05) < 3 * math.sqrt(0.05 * 0.95 / reps) + 0.01
    assert stats.norm.isf(0.025) == pytest.approx(1.96, abs=1e-3)
