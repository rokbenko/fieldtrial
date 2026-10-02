"""Cohen's kappa and prediction-powered inference (v0.3).

Golden values:
- Cohen's kappa against ``statsmodels.stats.inter_rater.cohens_kappa`` (an independent
  implementation, test-only dependency): kappa, its standard error (``std_kappa``), the
  interval and the z statistic under kappa = 0 (``z_value``).
- PPI++ against ``ppi_py.ppi_mean_ci`` and ``ppi_mean_pointestimate`` (ppi-python 0.2.3, MIT),
  computed in a separate environment (``data/ppi_golden.json``) with lambda 0, 1 and the
  tuned value, passed explicitly; the tuned value is ppi_py's ``_calc_lam_glm`` with clipping.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from statsmodels.stats.inter_rater import cohens_kappa as sm_kappa

from fieldtrial.stats import (
    cohens_kappa,
    kappa_table,
    ppi_difference,
    ppi_mean,
    tuned_lambda,
)

TABLES = [
    [[30, 5], [8, 57]],
    [[12, 3], [2, 3]],
    [[20, 4, 1], [3, 15, 2], [0, 5, 10]],
    [[50, 0], [0, 50]],
]


@pytest.mark.parametrize("table", TABLES)
def test_golden_kappa_against_statsmodels(table: list[list[int]]) -> None:
    ours = cohens_kappa(table)
    ref = sm_kappa(np.asarray(table))
    assert ours.kappa == pytest.approx(ref.kappa, abs=1e-12)
    assert ours.standard_error == pytest.approx(ref.std_kappa, abs=1e-12)
    assert ours.interval.low == pytest.approx(ref.kappa_low, abs=1e-12)
    assert ours.interval.high == pytest.approx(ref.kappa_upp, abs=1e-12)
    if math.isfinite(ours.test.statistic):
        assert ours.test.statistic == pytest.approx(ref.z_value, abs=1e-12)


def test_kappa_from_labels_and_edge_cases() -> None:
    table = kappa_table([1, 1, 0, 0, 1], [1, 0, 0, 0, 1])
    assert table == [[2, 0], [1, 2]]
    res = cohens_kappa(table)
    assert res.n == 5
    assert res.agreement == pytest.approx(0.8)
    one = cohens_kappa([[10, 0], [0, 0]])  # a single category used by both raters
    assert math.isnan(one.kappa)
    with pytest.raises(ValueError, match="square"):
        cohens_kappa([[1, 2, 3]])
    with pytest.raises(ValueError, match="empty"):
        cohens_kappa([[0, 0], [0, 0]])
    with pytest.raises(ValueError, match="non-negative counts"):
        cohens_kappa([[1, -1], [0, 2]])
    with pytest.raises(ValueError, match="same items"):
        kappa_table([1], [1, 0])


@settings(max_examples=50, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=40), min_size=4, max_size=4))
def test_kappa_is_at_most_one_and_symmetric(cells: list[int]) -> None:
    if sum(cells) == 0:
        return
    table = [cells[:2], cells[2:]]
    res = cohens_kappa(table)
    if math.isnan(res.kappa):
        return
    assert res.kappa <= 1 + 1e-12
    transposed = cohens_kappa([[cells[0], cells[2]], [cells[1], cells[3]]])
    assert transposed.kappa == pytest.approx(res.kappa, abs=1e-12)


PPI = json.loads((Path(__file__).parent / "data" / "ppi_golden.json").read_text())


@pytest.mark.parametrize("case", sorted(PPI))
def test_golden_ppi_against_ppi_py(case: str) -> None:
    row = PPI[case]
    assert tuned_lambda(row["y"], row["f"], row["fu"]) == pytest.approx(row["lam_tuned"], abs=1e-12)
    for tag, lam in (("lam0", 0.0), ("lam1", 1.0), ("lamtuned", row["lam_tuned"])):
        res = ppi_mean(row["y"], row["f"], row["fu"], lam=lam)
        estimate, low, high = row[tag]
        assert res.estimate == pytest.approx(estimate, abs=1e-12)
        assert res.interval.low == pytest.approx(low, abs=1e-12)
        assert res.interval.high == pytest.approx(high, abs=1e-12)
        greater = ppi_mean(row["y"], row["f"], row["fu"], lam=lam, alternative="greater")
        assert greater.interval.low == pytest.approx(row[tag + "_greater"], abs=1e-12)
        assert greater.interval.high == math.inf
    tuned = ppi_mean(row["y"], row["f"], row["fu"])
    assert tuned.lam == pytest.approx(row["lam_tuned"], abs=1e-12)


def test_ppi_difference_and_validation() -> None:
    row = PPI["weak_proxy"]
    a = ppi_mean(row["y"], row["f"], row["fu"])
    b = ppi_mean(PPI["small"]["y"], PPI["small"]["f"], PPI["small"]["fu"])
    diff = ppi_difference(a, b)
    assert diff.estimate == pytest.approx(a.estimate - b.estimate)
    assert diff.standard_error == pytest.approx(math.hypot(a.standard_error, b.standard_error))
    less = ppi_difference(a, b, alternative="less")
    assert less.interval.low == -math.inf
    with pytest.raises(ValueError, match="same length"):
        ppi_mean([1, 0], [0.5], [0.2])
    with pytest.raises(ValueError, match="at least 2 labeled"):
        ppi_mean([1], [0.5], [0.2])
    with pytest.raises(ValueError, match="at least 1 unlabeled"):
        ppi_mean([1, 0], [0.5, 0.2], [])
    with pytest.raises(ValueError, match="lam must lie"):
        ppi_mean([1, 0], [0.5, 0.2], [0.3], lam=1.5)
    with pytest.raises(ValueError, match="finite"):
        ppi_mean([1, float("nan")], [0.5, 0.2], [0.3])
    with pytest.raises(ValueError, match="one-dimensional"):
        ppi_mean([[1, 0]], [0.5, 0.2], [0.3])  # type: ignore[list-item]
    assert tuned_lambda([1, 0], [0.5, 0.5], [0.5]) == 0.0  # constant scores


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.sampled_from([0.0, 1.0]), min_size=3, max_size=30),
    st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=30),
)
def test_lambda_zero_is_the_classical_interval(y: list[float], fu: list[float]) -> None:
    f = [0.5] * len(y)
    res = ppi_mean(y, f, fu, lam=0.0)
    assert res.estimate == pytest.approx(float(np.mean(y)))
    assert res.interval.low == pytest.approx(res.classical.low)
    assert res.interval.high == pytest.approx(res.classical.high)
    assert 0.0 <= tuned_lambda(y, [v * 0.9 for v in y], fu) <= 1.0


@pytest.mark.slow
def test_ppi_coverage_with_a_good_proxy() -> None:
    rng = np.random.default_rng(20261003)
    p, n, big_n, reps = 0.6, 40, 400, 1000
    covered = narrower = 0
    for _ in range(reps):
        y = (rng.random(n) < p).astype(float)
        f = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.15, n), 0, 1)
        yu = (rng.random(big_n) < p).astype(float)
        fu = np.clip(0.2 + 0.6 * yu + rng.normal(0, 0.15, big_n), 0, 1)
        res = ppi_mean(y, f, fu)
        covered += res.interval.low <= p <= res.interval.high
        narrower += res.interval.width < res.classical.width
    assert covered / reps >= 0.95 - 3 * math.sqrt(0.95 * 0.05 / reps)
    assert narrower / reps > 0.9  # a good proxy makes the interval narrower
