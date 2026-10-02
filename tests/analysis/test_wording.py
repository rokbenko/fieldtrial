"""Report wording: the section 15 templates, and the words that must never appear."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fieldtrial.analysis import wording
from fieldtrial.stats import compare_independent, mde, proportion_ci


def _sentence(k1: int, n1: int, k2: int, n2: int) -> str:
    res = compare_independent(k1, n1, k2, n2)
    rejected = res.primary.rejects(0.05)
    effect = None if rejected else mde(min(max(k2 / n2, 0.01), 0.99), n2, n1).effect
    ci1 = proportion_ci(k1, n1).interval
    return wording.difference(
        treatment="q50",
        control="baseline",
        k1=k1,
        n1=n1,
        ci1=(ci1.low, ci1.high),
        k2=k2,
        n2=n2,
        diff=res.difference,
        ci=(res.interval.low, res.interval.high),
        test="boschloo",
        pvalue=res.primary.pvalue,
        rejected=rejected,
        mde_pp=effect,
    )


def test_significant_template() -> None:
    assert _sentence(74, 80, 91, 120) == (
        "q50 succeeded in 92.5% of trials (74/80; 95% CI 84.6–96.5%) vs 75.8% (91/120) "
        "for baseline: +16.7 pp (95% CI +6.2 to +26.0; Boschloo p = 0.0020)."
    )


def test_not_significant_template() -> None:
    effect = mde(91 / 120, 120, 40).effect
    assert effect is not None
    assert _sentence(36, 40, 91, 120) == (
        "No significant difference detected: +14.2 pp (95% CI −0.5 to +24.5; p = 0.056). "
        "With 40 and 120 trials, this study had 80% power only for differences of at least "
        f"{effect * 100:.1f} pp."
    )


@pytest.mark.parametrize(
    ("p", "text"),
    [
        (0.0019640, "0.0020"),
        (0.05574, "0.056"),
        (0.2668, "0.27"),
        (0.0995, "0.10"),
        (0.5, "0.50"),
        (0.999, "1.0"),
        (0.00005, "< 0.0001"),
        (float("nan"), "n/a"),
    ],
)
def test_fmt_p(p: float, text: str) -> None:
    assert wording.fmt_p(p) == text


def test_fmt_signed() -> None:
    assert wording.fmt_signed(0.1667) == "+16.7"
    assert wording.fmt_signed(-0.005) == "−0.5"
    assert wording.fmt_signed(0.0004) == "0.0"
    assert wording.fmt_pp(-0.2) == "−20.0 pp"


@pytest.mark.parametrize(
    "text",
    [
        "q50 is better",
        "a trend toward significance",
        "almost significant",
        "Worse than baseline",
        "borderline effect",
    ],
)
def test_banned_wording(text: str) -> None:
    with pytest.raises(ValueError, match="must not say"):
        wording.check(text)


@settings(max_examples=40, deadline=None)
@given(
    n1=st.integers(1, 30),
    n2=st.integers(1, 30),
    k1_frac=st.floats(0, 1),
    k2_frac=st.floats(0, 1),
)
def test_no_difference_claim_without_rejection(
    n1: int, n2: int, k1_frac: float, k2_frac: float
) -> None:
    """Property: a comparison is only described in rates when the test rejected."""
    k1, k2 = round(k1_frac * n1), round(k2_frac * n2)
    text = _sentence(k1, n1, k2, n2)
    rejected = compare_independent(k1, n1, k2, n2).primary.rejects(0.05)
    assert text.startswith("No significant difference") != rejected
    assert "pp" in text


def test_other_templates() -> None:
    assert wording.threshold(
        arm="a",
        k=38,
        n=40,
        ci=(0.83, 0.99),
        p0=0.8,
        alternative="greater",
        pvalue=0.01,
        rejected=True,
    ).endswith("above the 80.0% threshold (exact binomial p = 0.010).")
    assert "do not show a success rate above" in wording.threshold(
        arm="a",
        k=30,
        n=40,
        ci=(0.6, 0.86),
        p0=0.8,
        alternative="greater",
        pvalue=0.8,
        rejected=False,
    )
    assert "different from" in wording.threshold(
        arm="a",
        k=30,
        n=40,
        ci=(0.6, 0.86),
        p0=0.8,
        alternative="two-sided",
        pvalue=0.5,
        rejected=False,
    )
    assert wording.descriptive("a", 3, 4, (0.3, 0.95)).endswith("No test was pre-registered.")
    assert "did not have 80% power" in wording.no_difference_power(5, 5, None)
    assert wording.excluded_blocks(1, 9).startswith("1 block without")
    assert wording.omnibus(3, 10, 4.2, 0.12) == (
        "Cochran's Q across 3 arms on 10 complete blocks: Q = 4.20, p = 0.12."
    )
    assert "odds ratio 2.50" in wording.odds_ratio(
        treatment="t",
        control="c",
        k1=30,
        n1=40,
        ci1=(0.6, 0.86),
        k2=20,
        n2=40,
        estimate=2.5,
        ci=(1.1, 5.0),
        pvalue=0.02,
        rejected=True,
        mde_pp=None,
    )
    assert wording.odds_ratio(
        treatment="t",
        control="c",
        k1=30,
        n1=40,
        ci1=(0.6, 0.86),
        k2=28,
        n2=40,
        estimate=1.2,
        ci=(0.5, 3.0),
        pvalue=0.6,
        rejected=False,
        mde_pp=0.25,
    ).startswith("No significant difference detected: Mantel–Haenszel odds ratio 1.20")


def test_v02_sentences() -> None:
    assert "rises with training step" in wording.step_association(
        test="mantel", statistic=3.1, pvalue=0.002, rejected=True, checkpoints=4
    )
    flat = wording.step_association(
        test="mantel", statistic=0.4, pvalue=0.6, rejected=False, checkpoints=4
    )
    assert flat.startswith("No significant association")
    assert "Not enough data" in wording.step_association(
        test="mantel", statistic=None, pvalue=None, rejected=False, checkpoints=4
    )
    assert "From s2 on" in wording.plateau(plateau_arm="s2", final_arm="s4", margin=0.1, alpha=0.05)
    assert "No earlier checkpoint" in wording.plateau(
        plateau_arm=None, final_arm="s4", margin=0.1, alpha=0.05
    )
    common = {
        "treatment": "b",
        "control": "a",
        "cycles": 8,
        "rate_t": 0.9,
        "rate_c": 0.6,
        "diff": 0.3,
        "ci": (0.1, 0.5),
        "test": "crossover_randomization",
        "level": 0.95,
    }
    yes = wording.crossover(pvalue=0.01, rejected=True, mde_pp=None, **common)  # type: ignore[arg-type]
    assert "period-adjusted difference +30.0 pp" in yes
    assert "randomization test p = 0.010" in yes
    no = wording.crossover(pvalue=0.2, rejected=False, mde_pp=0.25, **common)  # type: ignore[arg-type]
    assert no.startswith("No significant difference detected")
    assert "at least 25.0 pp" in no
    assert "O'Brien–Fleming" in wording.sequential_stop(
        look=2, looks=4, blocks=20, planned=40, spending="obrien_fleming"
    )
    assert "continue" in wording.interim("continue", 1, 4)
    assert "Stop the study" in wording.interim("stop", 2, 4)
