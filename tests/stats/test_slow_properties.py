"""Statistical property tests (docs/PLAN.md section 12). CI runs them nightly.

Where the property can be computed exactly by enumerating outcomes, it is: exact coverage and
exact type I error have no Monte Carlo error, so the bounds are tight.
"""

import numpy as np
import pytest
from scipy import stats

from fieldtrial.stats import (
    boschloo_power,
    mcnemar_power,
    mde,
    power,
    proportion_ci,
    simulate_power,
)
from fieldtrial.stats.power import boschloo_rejection_region

pytestmark = pytest.mark.slow


def exact_coverage(p: float, n: int, method: str = "wilson") -> float:
    covered = 0.0
    for k in range(n + 1):
        if proportion_ci(k, n, method=method).interval.contains(p):
            covered += float(stats.binom.pmf(k, n, p))
    return covered


@pytest.mark.parametrize("p", [0.5, 0.75, 0.9])
@pytest.mark.parametrize("n", [20, 40, 80])
def test_wilson_coverage_is_near_nominal(p: float, n: int) -> None:
    assert 0.92 <= exact_coverage(p, n) <= 0.98


@pytest.mark.parametrize("n", [20, 40, 80])
def test_clopper_pearson_is_conservative(n: int) -> None:
    for p in np.linspace(0.05, 0.95, 19):
        assert exact_coverage(float(p), n, "clopper-pearson") >= 0.95 - 1e-12


@pytest.mark.parametrize(("n1", "n2"), [(20, 20), (40, 40), (40, 120)])
@pytest.mark.parametrize("alternative", ["two-sided", "greater"])
def test_boschloo_type_one_error_is_at_most_alpha(n1: int, n2: int, alternative: str) -> None:
    region = boschloo_rejection_region(n1, n2, alpha=0.05, alternative=alternative)  # type: ignore[arg-type]
    x1, x2 = np.arange(n1 + 1), np.arange(n2 + 1)
    for pi in np.linspace(0.01, 0.99, 99):
        size = np.outer(stats.binom.pmf(x1, n1, pi), stats.binom.pmf(x2, n2, pi))[region].sum()
        assert size <= 0.05 * (1 + 1e-5), f"size {size:.6f} at pi = {pi:.2f}"


@pytest.mark.parametrize("n", [20, 40, 80])
def test_mcnemar_type_one_error_is_at_most_alpha(n: int) -> None:
    for q in (0.05, 0.1, 0.2, 0.3, 0.45):
        assert mcnemar_power(q, q, n).power <= 0.05 + 1e-12


@pytest.mark.parametrize(("baseline", "n"), [(0.76, 40), (0.76, 120), (0.5, 40)])
@pytest.mark.parametrize("direction", ["increase", "decrease"])
def test_power_at_the_mde_is_the_requested_power(baseline: float, n: int, direction: str) -> None:
    res = mde(baseline, n, direction=direction)  # type: ignore[arg-type]
    assert res.target is not None
    assert power(baseline, res.target, n).power == pytest.approx(0.80, abs=2e-3)


@pytest.mark.parametrize(("p1", "p2", "n1", "n2"), [(0.6, 0.85, 30, 30), (0.76, 0.9, 40, 60)])
def test_simulated_boschloo_power_agrees_with_exact(p1: float, p2: float, n1: int, n2: int) -> None:
    exact = boschloo_power(p1, p2, n1, n2).power
    sim = simulate_power(p1, p2, n1, n2, n_sims=2000, seed=20261001)
    assert sim.standard_error is not None
    assert abs(sim.power - exact) <= 4 * sim.standard_error
