"""Group-sequential boundaries (docs/PLAN.md section 5, v0.2).

Golden values:
- Classical constants: Jennison and Turnbull (2000), table 2.1 (Pocock C_P) and table 2.3
  (O'Brien–Fleming C_B), two-sided alpha = 0.05, equally spaced looks.
- Lan–DeMets boundaries: Reboussin, DeMets, Kim and Lan (2000), "Computations for group
  sequential boundaries using the Lan-DeMets spending function method", Controlled
  Clinical Trials 21, 190–207 (the ld98 / R ldbounds example: 5 equally spaced looks,
  one-sided alpha = 0.025).

Crossing probabilities are also checked against scipy's multivariate normal CDF (Genz's
algorithm), which is independent of the recursive integration used here.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from fieldtrial.stats.sequential import (
    alpha_spending,
    constant_boundaries,
    crossing_probabilities,
    repeated_interval,
    sequential_test,
    spending_boundaries,
)

POCOCK = {1: 1.960, 2: 2.178, 3: 2.289, 4: 2.361, 5: 2.413, 10: 2.555}
OBRIEN_FLEMING = {1: 1.960, 2: 1.977, 3: 2.004, 4: 2.024, 5: 2.040, 10: 2.087}
FIVE = [0.2, 0.4, 0.6, 0.8, 1.0]
LD_OBF_5 = [4.8769, 3.3569, 2.6803, 2.2898, 2.0310]
LD_POCOCK_5 = [2.4380, 2.4268, 2.4101, 2.3966, 2.3859]


@pytest.mark.parametrize(("looks", "c"), POCOCK.items())
def test_golden_pocock_constants(looks: int, c: float) -> None:
    design = constant_boundaries(looks, shape="pocock")
    assert design.boundaries[0] == pytest.approx(c, abs=6e-4)
    assert design.alpha_spent[-1] == pytest.approx(0.05, abs=1e-9)


@pytest.mark.parametrize(("looks", "c"), OBRIEN_FLEMING.items())
def test_golden_obrien_fleming_constants(looks: int, c: float) -> None:
    design = constant_boundaries(looks, shape="obrien_fleming")
    assert design.boundaries[-1] == pytest.approx(c, abs=6e-4)
    assert design.boundaries[0] == pytest.approx(c * math.sqrt(looks), abs=2e-3)


def test_golden_lan_demets() -> None:
    obf = spending_boundaries(FIVE, alpha=0.025, alternative="greater")
    assert obf.boundaries == pytest.approx(LD_OBF_5, abs=2e-4)
    poc = spending_boundaries(FIVE, alpha=0.025, spending="pocock", alternative="greater")
    assert poc.boundaries == pytest.approx(LD_POCOCK_5, abs=2e-4)
    # A symmetric two-sided design at 0.05 is essentially the one-sided design at 0.025.
    two = spending_boundaries(FIVE, alpha=0.05)
    assert two.boundaries == pytest.approx(LD_OBF_5, abs=2e-4)


def test_spending_functions() -> None:
    assert alpha_spending(1.0, 0.05, "obrien_fleming") == pytest.approx(0.05)
    assert alpha_spending(1.0, 0.05, "pocock") == pytest.approx(0.05)
    assert alpha_spending(0.0, 0.05, "pocock") == 0.0
    assert alpha_spending(0.5, 0.05, "obrien_fleming") == pytest.approx(
        2 * stats.norm.sf(stats.norm.isf(0.025) / math.sqrt(0.5))
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        alpha_spending(1.5, 0.05, "pocock")


def _mvn_continue(bounds: list[float], info: list[float]) -> float:
    """P(|Z_k| < b_k for all k) under H0 by inclusion–exclusion over the box corners."""
    t = np.array(info)
    cov = np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))
    b = np.array(bounds)
    return float(
        stats.multivariate_normal.cdf(
            b, mean=np.zeros(len(t)), cov=cov, lower_limit=-b, abseps=1e-9, releps=1e-9
        )
    )


@pytest.mark.parametrize(
    ("bounds", "info"),
    [
        ([2.5, 2.0], [0.5, 1.0]),
        ([3.0, 2.4, 2.1], [0.3, 0.55, 1.0]),
        ([2.2, 2.2, 2.2], [1 / 3, 2 / 3, 1]),
    ],
)
def test_crossing_probabilities_match_multivariate_normal(
    bounds: list[float], info: list[float]
) -> None:
    probs = crossing_probabilities(bounds, info)
    assert 1.0 - sum(probs) == pytest.approx(_mvn_continue(bounds, info), abs=2e-6)
    assert probs[0] == pytest.approx(2 * stats.norm.sf(bounds[0]), abs=1e-12)


def test_power_under_drift_matches_single_look() -> None:
    # With one look, power is the usual normal power.
    drift = 2.8
    p = crossing_probabilities([1.96], [1.0], drift=drift, alternative="greater")[0]
    assert p == pytest.approx(stats.norm.sf(1.96 - drift), abs=1e-12)
    # More looks with OBF spending cost little power.
    design = spending_boundaries(FIVE)
    pw = sum(crossing_probabilities(design.boundaries, FIVE, drift=drift))
    assert 0.75 < pw < stats.norm.sf(1.96 - drift) + stats.norm.cdf(-1.96 - drift)


@settings(max_examples=25, deadline=None)
@given(
    cuts=st.lists(st.floats(0.05, 0.95), min_size=1, max_size=4, unique=True),
    alpha=st.sampled_from([0.01, 0.05, 0.1]),
    spending=st.sampled_from(["obrien_fleming", "pocock"]),
    alternative=st.sampled_from(["two-sided", "greater"]),
)
def test_property_spending_design_spends_alpha(
    cuts: list[float], alpha: float, spending: str, alternative: str
) -> None:
    info = [*sorted(cuts), 1.0]
    if min(np.diff([0.0, *info])) < 0.02:
        return
    design = spending_boundaries(
        info,
        alpha=alpha,
        spending=spending,
        alternative=alternative,  # type: ignore[arg-type]
    )
    probs = crossing_probabilities(design.boundaries, info, alternative=alternative)  # type: ignore[arg-type]
    assert sum(probs) == pytest.approx(alpha, abs=1e-7)
    assert np.cumsum(probs) == pytest.approx(design.alpha_spent, abs=1e-7)
    # Every boundary is at least the fixed-design critical value.
    fixed = stats.norm.isf(alpha / 2 if alternative == "two-sided" else alpha)
    assert min(design.boundaries) >= fixed - 1e-9


def test_spend_all_at_an_early_final_look() -> None:
    design = spending_boundaries([0.3, 0.6, 0.8], spend_all=True)
    assert design.alpha_spent[-1] == pytest.approx(0.05)
    assert sum(crossing_probabilities(design.boundaries, [0.3, 0.6, 0.8])) == pytest.approx(0.05)


def test_sequential_test_stops_and_pvalue_agrees_with_decision() -> None:
    design = spending_boundaries(FIVE)
    early = sequential_test([1.0, 3.5], design)
    assert early.stopped_at == 2
    assert early.final
    assert early.test.rejects(0.05)
    running = sequential_test([1.0, 1.2], design)
    assert running.stopped_at is None
    assert not running.final
    assert math.isnan(running.test.pvalue)
    # At the last look, the p-value equals alpha exactly at the boundary.
    b = design.boundaries[-1]
    at = sequential_test([0.0, 0.0, 0.0, 0.0, b], design)
    assert at.test.pvalue == pytest.approx(0.05, abs=1e-7)
    near = sequential_test([0.0, 0.0, 0.0, 0.0, b - 0.05], design)
    assert near.test.pvalue > 0.05
    assert near.stopped_at is None
    # Stopping earlier is more extreme than any outcome at a later look.
    assert early.test.pvalue < design.alpha_spent[1]
    assert early.test.pvalue > design.alpha_spent[0]


def test_one_sided_less_mirrors_greater() -> None:
    greater = spending_boundaries(FIVE, alpha=0.025, alternative="greater")
    less = spending_boundaries(FIVE, alpha=0.025, alternative="less")
    assert greater.boundaries == pytest.approx(less.boundaries)
    g = sequential_test([0.5, 2.0, 3.0], greater)
    lo = sequential_test([-0.5, -2.0, -3.0], less)
    assert (g.stopped_at, g.test.pvalue) == (lo.stopped_at, pytest.approx(lo.test.pvalue))
    assert sequential_test([-12.0, -12.0, -12.0, -12.0, -12.0], greater).test.pvalue == 1.0


def test_repeated_interval() -> None:
    ci = repeated_interval(0.1, 0.05, 2.5, level=0.95)
    assert (ci.low, ci.high) == pytest.approx((-0.025, 0.225))
    assert ci.method == "repeated_ci"


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        spending_boundaries([0.5, 0.4, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        spending_boundaries([0.5, 1.2])
    with pytest.raises(ValueError, match="same length"):
        crossing_probabilities([2.0], [0.5, 1.0])
    with pytest.raises(ValueError, match="design plans"):
        sequential_test([0.0] * 6, spending_boundaries(FIVE))
    with pytest.raises(ValueError, match="at least 1"):
        constant_boundaries(0)
    with pytest.raises(ValueError, match="spending"):
        spending_boundaries(FIVE, spending="linear")  # type: ignore[arg-type]


@pytest.mark.slow
def test_type_one_error_with_paired_binary_data() -> None:
    """McNemar score statistics at 4 looks of 20 blocks: the OBF design keeps its level."""
    rng = np.random.default_rng(20261002)
    looks = [20, 40, 60, 80]
    design = spending_boundaries([n / 80 for n in looks])
    reps, rejects = 4000, 0
    for _ in range(reps):
        # Each block: only A (0.15), only B (0.15), or concordant (0.7).
        draws = rng.choice(3, size=80, p=[0.15, 0.15, 0.7])
        z = []
        for n in looks:
            b = int((draws[:n] == 0).sum())
            c = int((draws[:n] == 1).sum())
            z.append((b - c) / math.sqrt(b + c) if b + c else 0.0)
            res = sequential_test(z, design)
            if res.stopped_at is not None:
                break
        rejects += res.test.rejects(0.05)
    assert rejects / reps <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / reps)
