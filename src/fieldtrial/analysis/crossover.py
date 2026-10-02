"""Crossover rounds in a study: per-round results and the period-adjusted comparison.

Rounds ``2c - 1`` and ``2c`` form cycle ``c`` (docs/stats/crossover.md). The arm of each
round comes from the locked schedule, so rounds without completed trials are still listed.
A cycle is analyzed when both of its rounds have at least one completed trial.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from fieldtrial.analysis.records import TrialRecord
from fieldtrial.analysis.results import CrossoverSummary, RoundRow
from fieldtrial.design import StudySpec, build_schedule
from fieldtrial.stats import CrossoverResult, crossover_test


@dataclass(frozen=True, slots=True)
class CrossoverOutcome:
    """The crossover summary plus the test, for the engine."""

    summary: CrossoverSummary
    result: CrossoverResult | None
    successes: dict[str, int]  # per arm, over the cycles used
    completed: dict[str, int]


def crossover_analysis(spec: StudySpec, records: Sequence[TrialRecord]) -> CrossoverOutcome:
    """Summarize the rounds and run the crossover test on the usable cycles."""
    comparison = spec.analysis.primary.comparison
    assert comparison is not None
    treatment, control = comparison.treatment, comparison.control
    round_arm = {s.block: s.arm for s in build_schedule(spec)}
    counts: dict[int, list[int]] = {r: [0, 0] for r in round_arm}
    for r in records:
        if r.status == "completed" and r.block in counts:
            counts[r.block][0] += int(bool(r.success))
            counts[r.block][1] += 1

    rows = [
        RoundRow(
            round=rnd,
            cycle=(rnd + 1) // 2,
            period=2 - rnd % 2,
            arm=round_arm[rnd],
            successes=counts[rnd][0],
            completed=counts[rnd][1],
            rate=counts[rnd][0] / counts[rnd][1] if counts[rnd][1] else None,
        )
        for rnd in sorted(round_arm)
    ]
    periods: list[tuple[float, float]] = []
    orders: list[str] = []
    successes = {treatment: 0, control: 0}
    completed = {treatment: 0, control: 0}
    for first, second in zip(rows[0::2], rows[1::2], strict=True):
        if first.rate is None or second.rate is None:
            continue
        periods.append((first.rate, second.rate))
        orders.append("AB" if first.arm == treatment else "BA")
        for row in (first, second):
            successes[row.arm] += row.successes
            completed[row.arm] += row.completed
    result = None
    if "AB" in orders and "BA" in orders:
        result = crossover_test(
            periods,
            orders,  # type: ignore[arg-type]
            alternative=spec.analysis.primary.alternative,
        )
    first_rows = rows[0::2]
    summary = CrossoverSummary(
        treatment=treatment,
        control=control,
        cycles=len(first_rows),
        cycles_used=len(periods),
        ab=sum(r.arm == treatment for r in first_rows),
        ba=sum(r.arm == control for r in first_rows),
        rounds=rows,
        period_effect=result.period_effect if result else None,
    )
    return CrossoverOutcome(
        summary=summary, result=result, successes=successes, completed=completed
    )
