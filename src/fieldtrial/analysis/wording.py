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
    "mantel": "Mantel",
    "cochran_armitage": "Cochran–Armitage",
    "crossover_randomization": "randomization test",
    "hills_armitage_t": "Hills–Armitage t-test",
    "group_sequential": "group-sequential McNemar",
    "betting_paired": "anytime-valid betting test",
}

SPENDING_LABELS = {"obrien_fleming": "O'Brien–Fleming", "pocock": "Pocock"}


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
    power_note: str | None = None,
) -> str:
    """A two-arm comparison of success rates.

    Significant: ``q50 succeeded in 92.5% of trials (74/80; 95% CI 84.6–96.5%) vs 75.8%
    (91/120) for baseline: +16.7 pp (95% CI +6.2 to +26.0; Boschloo p = 0.0020).``

    Not significant: ``No significant difference detected: +14.2 pp (95% CI −0.5 to +24.5;
    p = 0.056). With 40 and 120 trials, this study had 80% power only for differences of at
    least X pp.``

    ``note`` is appended after the p-value, for example ``", Holm-adjusted"``.
    ``power_note`` replaces the fixed-design power statement after a non-significant
    result (anytime-valid designs, whose power differs).
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
            f"{fmt_p_eq(pvalue)}{note}). "
            f"{power_note if power_note is not None else no_difference_power(n1, n2, mde_pp)}"
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


def step_association(
    *,
    test: str,
    statistic: float | None,
    pvalue: float | None,
    rejected: bool,
    checkpoints: int,
) -> str:
    """Association between training step and success across a checkpoint ladder."""
    if pvalue is None or statistic is None:
        return no_data("the association with training step")
    label = _test(test)
    if rejected:
        direction = "rises" if statistic > 0 else "falls"
        text = (
            f"Success {direction} with training step across the {checkpoints} checkpoints "
            f"({label} test of a linear association with step, {fmt_p_eq(pvalue)})."
        )
    else:
        text = (
            f"No significant association between training step and success detected across "
            f"the {checkpoints} checkpoints ({label} test, {fmt_p_eq(pvalue)})."
        )
    return check(text)


def plateau(*, plateau_arm: str | None, final_arm: str, margin: float, alpha: float) -> str:
    """Where a ladder levels off: non-inferiority against the final checkpoint."""
    tail = (
        f"non-inferiority margin {margin * 100:.1f} pp, one-sided α = {alpha:g}, tested "
        "from the latest checkpoint backwards"
    )
    if plateau_arm is None:
        text = f"No earlier checkpoint was shown to be within the margin of {final_arm} ({tail})."
    else:
        text = (
            f"From {plateau_arm} on, every checkpoint was within {margin * 100:.1f} pp of "
            f"{final_arm} ({tail})."
        )
    return check(text)


def crossover(
    *,
    treatment: str,
    control: str,
    cycles: int,
    rate_t: float,
    rate_c: float,
    diff: float,
    ci: tuple[float, float] | None,
    test: str,
    pvalue: float,
    rejected: bool,
    mde_pp: float | None,
    level: float = 0.95,
) -> str:
    """A crossover-rounds comparison: period-adjusted difference over cycles."""
    interval = f"{fmt_level(level)} CI {fmt_signed(ci[0])} to {fmt_signed(ci[1])}; " if ci else ""
    if rejected:
        text = (
            f"Over {cycles} crossover cycles, {treatment} succeeded in {fmt_rate(rate_t)} of "
            f"trials vs {fmt_rate(rate_c)} for {control}: period-adjusted difference "
            f"{fmt_pp(diff)} ({interval}{_test(test)} {fmt_p_eq(pvalue)})."
        )
    else:
        power = (
            f" With {cycles} cycles and the observed round-to-round variation, this study "
            f"had 80% power only for differences of at least {abs(mde_pp) * 100:.1f} pp."
            if mde_pp is not None
            else ""
        )
        text = (
            f"No significant difference detected over {cycles} crossover cycles: "
            f"period-adjusted difference {fmt_pp(diff)} ({interval}{_test(test)} "
            f"{fmt_p_eq(pvalue)}).{power}"
        )
    return check(text)


def sequential_stop(*, look: int, looks: int, blocks: int, planned: int, spending: str) -> str:
    """A study that stopped at an interim look."""
    label = SPENDING_LABELS.get(spending, spending)
    return check(
        f"The study stopped at interim look {look} of {looks} ({blocks} of {planned} planned "
        f"blocks) because the pre-registered {label} boundary was crossed."
    )


def sequential_note(*, looks_run: int, looks: int, spending: str) -> str:
    """How the p-value of a group-sequential study was computed."""
    label = SPENDING_LABELS.get(spending, spending)
    return check(
        f"Group-sequential design with {label} error spending ({looks_run} of {looks} planned "
        "looks): the p-value uses stage-wise ordering and the interval is a repeated "
        "confidence interval."
    )


def interim(decision: str, look: int, looks: int) -> str:
    """What an interim look reports while the study is blinded."""
    if decision == "stop":
        return check(
            f"Interim look {look} of {looks}: the pre-registered boundary was crossed. Stop "
            "the study and unblind to see the results."
        )
    return check(
        f"Interim look {look} of {looks}: continue. The boundary was not crossed; no "
        "results are shown while the study is blinded."
    )


def rig_drift(checked_at: object, reasons: list[str]) -> str:
    """A flagged rig check, for the deviations list."""
    when = checked_at.strftime("%Y-%m-%d %H:%M") if hasattr(checked_at, "strftime") else checked_at
    detail = "; ".join(reasons) or "the rig differs from its reference photo"
    return check(f"Rig check on {when} UTC differs from the reference photo: {detail}.")


def anytime_stop(*, blocks: int, planned: int) -> str:
    """A study stopped by its anytime-valid test."""
    return check(
        f"The study stopped after {blocks} of {planned} planned blocks because the "
        "pre-registered anytime-valid test rejected the null hypothesis."
    )


def anytime_note(*, blocks: int) -> str:
    """How the p-value and interval of an anytime-valid study were computed."""
    return check(
        f"Anytime-valid design checked after every block ({blocks} complete): the p-value is "
        "anytime-valid and the interval is a confidence sequence, both valid whenever the "
        "study stopped."
    )


ANYTIME_POWER = (
    "An anytime-valid test needs more blocks than a fixed design to detect the same "
    "difference; see the confidence sequence for the differences still compatible with "
    "the data."
)


def anytime_look(blocks: int) -> str:
    """What the console says when an anytime study stops (while blinded)."""
    return check(
        f"After {blocks} complete blocks the pre-registered anytime-valid test rejected the "
        "null hypothesis. The remaining trials are cancelled; unblind to see the results."
    )


def _elimination_list(eliminations: list[tuple[str, int]]) -> str:
    return ", ".join(f"{arm} after block {block}" for arm, block in eliminations)


def selection(
    *,
    survivors: list[str],
    eliminations: list[tuple[str, int]],
    blocks: int,
    planned: int,
    delta: float,
) -> str:
    """The result of best-arm selection by successive elimination.

    An arm is called the highest only when it is the single survivor.
    """
    method = f"successive elimination, δ = {delta:g}"
    dropped = _elimination_list(eliminations)
    if len(survivors) == 1:
        return check(
            f"Selected arm: {survivors[0]}. Every other arm was dropped because another arm "
            f"had a higher success rate with confidence ({method}): {dropped}. With "
            f"probability at least {fmt_rate(1 - delta)}, {survivors[0]} has the highest "
            f"success rate."
        )
    names = ", ".join(survivors)
    if not eliminations:
        return check(
            f"No arm could be singled out after {blocks} of {planned} planned blocks: "
            f"{names} could not be told apart ({method})."
        )
    return check(
        f"No single arm could be selected after {blocks} of {planned} planned blocks: "
        f"{names} could not be told apart ({method}). Dropped: {dropped}."
    )


def selection_look(dropped: list[str], remaining: int) -> str:
    """What the console says when arms are dropped (by blind code, while blinded)."""
    names = ", ".join(dropped)
    plural = "s" if len(dropped) != 1 else ""
    if remaining <= 1:
        return check(
            f"Arm{plural} {names} dropped; one arm remains, so the study is complete. "
            "Unblind to see the results."
        )
    return check(
        f"Arm{plural} {names} dropped: another arm had a higher success rate with "
        f"confidence. {'Their' if plural else 'Its'} remaining trials are cancelled; "
        f"{remaining} arms continue."
    )


def agreement(
    *, model: str, n: int, agreement: float, kappa: float | None, ci: tuple[float, float] | None
) -> str:
    """How often a reward model's suggestions matched blind human labels (descriptive)."""
    if kappa is None or ci is None:
        return check(
            f"The reward model {model} agreed with the human label on {fmt_rate(agreement)} "
            f"of {n} episodes; Cohen's κ is undefined because one label was never used."
        )
    return check(
        f"The reward model {model} agreed with the blind human label on {fmt_rate(agreement)} "
        f"of {n} episodes (Cohen's κ = {kappa:.2f}, 95% CI {ci[0]:.2f} to {ci[1]:.2f})."
    )


def proxy_estimate(
    *,
    arm: str,
    source: str,
    estimate: float,
    ci: tuple[float, float],
    classical: tuple[float, float],
    labeled: int,
    unlabeled: int,
) -> str:
    """A proxy-assisted (PPI++) estimate: a pre-registered secondary analysis."""
    return check(
        f"Secondary, proxy-assisted: {arm} succeeded in an estimated {fmt_rate(estimate)} of "
        f"episodes from {source} (95% CI {fmt_range(*ci)}; {labeled} human-labeled and "
        f"{unlabeled} model-scored episodes; {fmt_range(*classical)} from the human labels "
        "alone)."
    )
