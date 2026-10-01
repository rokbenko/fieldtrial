"""Multiplicity adjustments. Golden values: docs/PLAN.md section 12 (Holm table)."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.multitest import multipletests

from fieldtrial.stats import adjust_pvalues

# Dream Machines serving sweep, each arm against the baseline 91/120 (Fisher two-sided).
SWEEP = [
    # (arm, k, n, raw, holm, reject)
    ("sync", 21, 40, 0.00893, 0.05361, False),
    ("R25-B50", 20, 40, 0.00302, 0.02113, True),
    ("R50-B50", 22, 40, 0.01624, 0.08118, False),
    ("R50-B1", 32, 40, 0.66934, 1.0, False),
    ("R50-B16", 35, 40, 0.17923, 0.60959, False),
    ("R50-B20", 74, 80, 0.00223, 0.01785, True),
    ("R50-B25", 68, 80, 0.15240, 0.60959, False),
    ("R50-B25 compiled", 29, 40, 0.67743, 1.0, False),
]


def test_golden_holm_sweep() -> None:
    raw = [float(stats.fisher_exact([[k, 91], [n - k, 29]]).pvalue) for _, k, n, *_ in SWEEP]
    for (_, _, _, golden_raw, *_), r in zip(SWEEP, raw, strict=True):
        assert r == pytest.approx(golden_raw, abs=1e-5)
    res = adjust_pvalues(raw, method="holm")
    for (_, _, _, _, holm, reject), adj, rej in zip(SWEEP, res.adjusted, res.reject, strict=True):
        assert adj == pytest.approx(holm, abs=1e-5)
        assert rej == reject


@pytest.mark.parametrize(
    ("ours", "theirs"), [("holm", "holm"), ("bonferroni", "bonferroni"), ("bh", "fdr_bh")]
)
@pytest.mark.parametrize("seed", range(5))
def test_matches_statsmodels_live(ours: str, theirs: str, seed: int) -> None:
    p = np.random.default_rng(seed).uniform(0, 0.2, size=9)
    reject, adjusted, _, _ = multipletests(p, alpha=0.05, method=theirs)
    res = adjust_pvalues(p.tolist(), method=ours)
    np.testing.assert_allclose(res.adjusted, adjusted, rtol=1e-12, atol=1e-15)
    assert list(res.reject) == list(reject)


def test_single_pvalue_is_unchanged() -> None:
    res = adjust_pvalues([0.03])
    assert res.adjusted == (0.03,)
    assert res.reject == (True,)


@pytest.mark.parametrize("bad", [[], [0.1, -0.01], [1.2], [float("nan")], [[0.1, 0.2]]])
def test_invalid_input(bad: list[float]) -> None:
    with pytest.raises(ValueError, match=r"empty|flat|number in"):
        adjust_pvalues(bad)


def test_invalid_method_and_alpha() -> None:
    with pytest.raises(ValueError, match="method"):
        adjust_pvalues([0.1], method="sidak")
    with pytest.raises(ValueError, match="alpha"):
        adjust_pvalues([0.1], alpha=1.0)


pvalue_lists = st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=30)


@given(pvalue_lists, st.sampled_from(["holm", "bonferroni", "bh"]))
def test_adjusted_is_at_least_raw_and_at_most_one(p: list[float], method: str) -> None:
    res = adjust_pvalues(p, method=method)
    for raw, adj in zip(res.raw, res.adjusted, strict=True):
        assert raw - 1e-15 <= adj <= 1.0


@given(pvalue_lists, st.sampled_from(["holm", "bonferroni", "bh"]))
def test_adjustment_preserves_order(p: list[float], method: str) -> None:
    res = adjust_pvalues(p, method=method)
    order = np.argsort(res.raw, kind="stable")
    adjusted_sorted = np.asarray(res.adjusted)[order]
    assert np.all(np.diff(adjusted_sorted) >= -1e-15)


@given(pvalue_lists)
def test_holm_is_between_bh_and_bonferroni(p: list[float]) -> None:
    holm = adjust_pvalues(p, method="holm").adjusted
    bonf = adjust_pvalues(p, method="bonferroni").adjusted
    bh = adjust_pvalues(p, method="bh").adjusted
    for h, b, f in zip(holm, bonf, bh, strict=True):
        assert f - 1e-12 <= h <= b + 1e-12
