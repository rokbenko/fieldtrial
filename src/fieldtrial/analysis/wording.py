"""Every sentence a report says about results comes from here (docs/PLAN.md section 15).

One tested template per situation. The rules:

- A difference is only described as a difference when the pre-registered primary test
  rejected. Otherwise the text says "No significant difference detected" and states the
  smallest difference the study could have detected with 80% power.
- Never "better", "trend toward significance", "almost significant" or similar. ``check``
  enforces this on every sentence produced here.
"""

import math
import re

BANNED = (
    "better",
    "worse",
    "trend",
    "almost significant",
    "nearly significant",
    "marginally significant",
    "approaching significance",
    "borderline",
)
_BANNED_RE = re.compile("|".join(re.escape(b) for b in BANNED), re.IGNORECASE)
MINUS = "−"  # U+2212, as in the report templates

TEST_LABELS = {
    "boschloo": "Boschloo",
    "fisher": "Fisher",
    "mcnemar": "McNemar",
    "cmh": "CMH",
    "binomial": "exact binomial",
}


def check(text: str) -> str:
    """Return ``text`` unchanged, or raise ``ValueError`` if it uses banned wording."""
    match = _BANNED_RE.search(text)
    if match:
        raise ValueError(f"report wording must not say {match.group(0)!r}: {text}")
    return text


def fmt_p(p: float) -> str:
    """A p-value with two significant digits: ``0.0020``, ``0.056``, ``0.45``, ``< 0.0001``."""
    if math.isnan(p):
        return "n/a"
    if p < 0.0001:
        return "< 0.0001"
    if p >= 0.995:
        return "1.0"
    decimals = max(2, 1 - math.floor(math.log10(p)))
    text = f"{p:.{decimals}f}"
    # Rounding can add a digit (0.0995 -> 0.100); keep two significant digits.
    if float(text) >= 10 ** (math.floor(math.log10(p)) + 1):
        text = f"{p:.{decimals - 1}f}"
    return text


def fmt_p_eq(p: float) -> str:
    """``p = 0.0020`` or ``p < 0.0001``."""
    text = fmt_p(p)
    return f"p {text}" if text.startswith("<") else f"p = {text}"


def fmt_rate(x: float) -> str:
    """A proportion as a percentage with one decimal: ``92.5%``."""
    return f"{x * 100:.1f}%"


def fmt_range(low: float, high: float) -> str:
    """A rate interval: ``84.6–96.5%``."""
    return f"{low * 100:.1f}–{high * 100:.1f}%"


def fmt_signed(x: float) -> str:
    """A signed number of percentage points without unit: ``+16.7``, ``−0.5``, ``0.0``."""
    value = round(x * 100, 1)
    if value == 0:
        return "0.0"
    return f"{'+' if value > 0 else MINUS}{abs(value):.1f}"


def fmt_pp(x: float) -> str:
    """A signed difference in percentage points: ``+16.7 pp``."""
    return f"{fmt_signed(x)} pp"


def fmt_level(level: float) -> str:
    """``95%`` for 0.95."""
    return f"{level * 100:g}%"


def _test(test: str) -> str:
    return TEST_LABELS.get(test, test)


def arm_rate(arm: str, k: int, n: int, ci: tuple[float, float], level: float = 0.95) -> str:
    """``q50 succeeded in 92.5% of trials (74/80; 95% CI 84.6–96.5%)``."""
    return (
        f"{arm} succeeded in {fmt_rate(k / n)} of trials "
        f"({k}/{n}; {fmt_level(level)} CI {fmt_range(*ci)})"
    )


def no_difference_power(n_treatment: int, n_control: int, mde_pp: float | None) -> str:
    """The power statement that follows every non-significant result."""
    if mde_pp is None:
        return (
            f"With {n_treatment} and {n_control} trials, this study did not have 80% power "
            "for any difference."
        )
    return (
        f"With {n_treatment} and {n_control} trials, this study had 80% power only for "
        f"differences of at least {abs(mde_pp) * 100:.1f} pp."
    )


def difference(
    *,
    treatment: str,
    control: str,
    k1: int,
    n1: int,
    ci1: tuple[float, float],
    k2: int,
    n2: int,
    diff: float,
    ci: tuple[float, float],
    test: str,
    pvalue: float,
    rejected: bool,
    mde_pp: float | None,
    level: float = 0.95,
    note: str = "",
) -> str:
    """A two-arm comparison of success rates.

    Significant: ``q50 succeeded in 92.5% of trials (74/80; 95% CI 84.6–96.5%) vs 75.8%
    (91/120) for baseline: +16.7 pp (95% CI +6.2 to +26.0; Boschloo p = 0.0020).``

    Not significant: ``No significant difference detected: +14.2 pp (95% CI −0.5 to +24.5;
    p = 0.056). With 40 and 120 trials, this study had 80% power only for differences of at
    least X pp.``

    ``note`` is appended after the p-value, for example ``", Holm-adjusted"``.
    """
    interval = f"{fmt_level(level)} CI {fmt_signed(ci[0])} to {fmt_signed(ci[1])}"
    if rejected:
        text = (
            f"{arm_rate(treatment, k1, n1, ci1, level)} vs {fmt_rate(k2 / n2)} ({k2}/{n2}) "
            f"for {control}: {fmt_pp(diff)} ({interval}; {_test(test)} {fmt_p_eq(pvalue)}"
            f"{note})."
        )
    else:
        text = (
            f"No significant difference detected: {fmt_pp(diff)} ({interval}; "
            f"{fmt_p_eq(pvalue)}{note}). {no_difference_power(n1, n2, mde_pp)}"
        )
    return check(text)


def odds_ratio(
    *,
    treatment: str,
    control: str,
    k1: int,
    n1: int,
    ci1: tuple[float, float],
    k2: int,
    n2: int,
    estimate: float,
    ci: tuple[float, float],
    pvalue: float,
    rejected: bool,
    mde_pp: float | None,
    level: float = 0.95,
) -> str:
    """A stratified (CMH) comparison, reported as a Mantel–Haenszel odds ratio."""
    interval = f"{fmt_level(level)} CI {ci[0]:.2f} to {ci[1]:.2f}"
    if rejected:
        text = (
            f"{arm_rate(treatment, k1, n1, ci1, level)} vs {fmt_rate(k2 / n2)} ({k2}/{n2}) "
            f"for {control}: Mantel–Haenszel odds ratio {estimate:.2f} ({interval}; "
            f"CMH {fmt_p_eq(pvalue)})."
        )
    else:
        text = (
            f"No significant difference detected: Mantel–Haenszel odds ratio {estimate:.2f} "
            f"({interval}; {fmt_p_eq(pvalue)}). {no_difference_power(n1, n2, mde_pp)}"
        )
    return check(text)


def threshold(
    *,
    arm: str,
    k: int,
    n: int,
    ci: tuple[float, float],
    p0: float,
    alternative: str,
    pvalue: float,
    rejected: bool,
    level: float = 0.95,
) -> str:
    """A single arm tested against a fixed success-rate threshold."""
    rate = arm_rate(arm, k, n, ci, level)
    p_text = f"exact binomial {fmt_p_eq(pvalue)}"
    if rejected:
        side = "above" if k / n > p0 else "below"
        text = f"{rate}, {side} the {fmt_rate(p0)} threshold ({p_text})."
    else:
        wanted = {"greater": "above", "less": "below"}.get(alternative, "different from")
        text = (
            f"{rate}. The data do not show a success rate {wanted} the {fmt_rate(p0)} "
            f"threshold ({p_text})."
        )
    return check(text)


def descriptive(arm: str, k: int, n: int, ci: tuple[float, float], level: float = 0.95) -> str:
    """A single arm without a pre-registered test."""
    return check(f"{arm_rate(arm, k, n, ci, level)}. No test was pre-registered.")


def omnibus(n_arms: int, n_blocks: int, statistic: float, pvalue: float) -> str:
    """Cochran's Q across all arms."""
    return check(
        f"Cochran's Q across {n_arms} arms on {n_blocks} complete blocks: "
        f"Q = {statistic:.2f}, {fmt_p_eq(pvalue)}."
    )


def no_data(what: str) -> str:
    """When an analysis cannot run yet."""
    return check(f"Not enough data for {what} yet.")


def sensitivity(
    *, treatment: str, control: str, diff: float, ci: tuple[float, float], pvalue: float
) -> str:
    """The independent-samples sensitivity analysis, stated neutrally."""
    return check(
        f"Sensitivity analysis treating trials as independent, {treatment} vs {control}: "
        f"{fmt_pp(diff)} (95% CI {fmt_signed(ci[0])} to {fmt_signed(ci[1])}; Boschloo "
        f"{fmt_p_eq(pvalue)})."
    )


def excluded_blocks(n: int, used: int) -> str:
    """Blocks left out of a paired analysis."""
    return check(
        f"{n} block{'s' if n != 1 else ''} without a completed trial for every arm "
        f"{'were' if n != 1 else 'was'} left out of the paired analysis ({used} used)."
    )


def drift_flag(arm: str, grouping: str, groups: int, pvalue: float) -> str:
    """Success rate changed across sessions or operators."""
    return check(
        f"Results changed across {grouping}s for {arm}: the success rate differs between "
        f"its {groups} {grouping}s ({fmt_p_eq(pvalue)}). Check the rig and setup log."
    )


def invalid_flag(per_arm: dict[str, int], attempts: dict[str, int], pvalue: float) -> str:
    """Invalid trials concentrated in some arms."""
    parts = ", ".join(f"{arm} {per_arm[arm]}/{attempts[arm]}" for arm in per_arm)
    return check(
        f"Invalid trials are unevenly spread across arms ({parts}; {fmt_p_eq(pvalue)}). "
        "Invalid trials can hide failures; check why they were voided."
    )
