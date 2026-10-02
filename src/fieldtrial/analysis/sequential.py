"""Group-sequential looks for 2-arm randomized block designs (docs/stats/sequential.md).

The statistic at a look is the McNemar score statistic on the complete blocks so far,
``Z = (b - c) / sqrt(b + c)``, where ``b`` counts blocks where only the treatment succeeded
and ``c`` blocks where only the control did. Information is measured in complete blocks.

An interim look records only which blocks it used, the information fraction, the boundary
and the decision, never ``Z``: the event log stays free of per-arm results while the study
is blinded. The final analysis recomputes ``Z`` at every look from the recorded blocks.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from fieldtrial.analysis.records import TrialRecord
from fieldtrial.analysis.results import LookRow, SequentialSummary
from fieldtrial.design import StudySpec
from fieldtrial.stats import Interval, repeated_interval, sequential_test, spending_boundaries

Decision = Literal["continue", "stop"]


class InterimError(ValueError):
    """An interim look cannot be run now (not due, already stopped, or the study is done)."""


@dataclass(frozen=True, slots=True)
class InterimDecision:
    """The outcome of an interim look. Holds nothing that reveals per-arm results."""

    look: int
    planned_looks: int
    fraction: float
    blocks: tuple[int, ...]
    boundary: float
    decision: Decision

    def payload(self) -> dict[str, Any]:
        """The ``interim_look`` event payload."""
        return {
            "look": self.look,
            "planned_looks": self.planned_looks,
            "fraction": self.fraction,
            "blocks": list(self.blocks),
            "boundary": self.boundary,
            "decision": self.decision,
        }


@dataclass(frozen=True, slots=True)
class SequentialOutcome:
    """The final group-sequential analysis, for the engine."""

    summary: SequentialSummary
    pvalue: float | None
    rejected: bool
    interval: Interval | None
    pairs: int
    b: int
    c: int
    notes: tuple[str, ...]  # deviations: skipped looks, decisions that do not reproduce


def planned_blocks(spec: StudySpec) -> int:
    """Number of complete blocks at full information."""
    return spec.conditions.count() * spec.conditions.replicates


def complete_pairs(
    records: Sequence[TrialRecord], treatment: str, control: str
) -> dict[int, tuple[bool, bool]]:
    """Block -> (treatment succeeded, control succeeded), for blocks with both arms done."""
    seen: dict[int, dict[str, bool]] = {}
    for r in records:
        if r.status == "completed" and r.arm in (treatment, control):
            seen.setdefault(r.block, {})[r.arm] = bool(r.success)
    return {
        block: (arms[treatment], arms[control])
        for block, arms in sorted(seen.items())
        if treatment in arms and control in arms
    }


def score_z(pairs: Sequence[tuple[bool, bool]]) -> tuple[float, int, int]:
    """McNemar score statistic ``(b - c) / sqrt(b + c)`` (0 without discordant pairs)."""
    b = sum(t and not c for t, c in pairs)
    c = sum(c and not t for t, c in pairs)
    z = (b - c) / math.sqrt(b + c) if b + c else 0.0
    return z, b, c


def _settings(spec: StudySpec) -> tuple[str, str, list[float]]:
    comparison = spec.analysis.primary.comparison
    stopping = spec.analysis.stopping
    if stopping.rule != "group_sequential" or comparison is None:
        raise InterimError("this study has no group-sequential stopping rule")
    return comparison.treatment, comparison.control, stopping.fractions()


def _crossed(z: float, boundary: float, alternative: str) -> bool:
    if alternative == "greater":
        return z >= boundary
    if alternative == "less":
        return -z >= boundary
    return abs(z) >= boundary


def next_look_due(
    spec: StudySpec, looks_done: Sequence[Mapping[str, Any]], complete_blocks: int
) -> tuple[int, int] | None:
    """``(look, blocks needed)`` of the next interim look, or None when none remains.

    The final look is the final analysis, not an interim look, so at most ``looks - 1``
    interim looks are run. Returns None once a look has decided to stop.
    """
    _, _, fractions = _settings(spec)
    if any(look.get("decision") == "stop" for look in looks_done):
        return None
    k = len(looks_done)
    if k >= len(fractions) - 1:
        return None
    return k + 1, math.ceil(fractions[k] * planned_blocks(spec) - 1e-9)


def interim_decision(
    spec: StudySpec,
    records: Sequence[TrialRecord],
    looks_done: Sequence[Mapping[str, Any]],
) -> InterimDecision:
    """Run the next planned interim look on the complete blocks so far.

    Raises :class:`InterimError` when no look is due. Error spending makes the actual
    timing of a look valid as long as it is planned, so a look taken later than its
    planned point uses the information actually available.
    """
    treatment, control, fractions = _settings(spec)
    pairs = complete_pairs(records, treatment, control)
    due = next_look_due(spec, looks_done, len(pairs))
    if due is None:
        if any(look.get("decision") == "stop" for look in looks_done):
            raise InterimError("the study already stopped at an interim look")
        raise InterimError(
            "all interim looks are done; the next analysis is the final one "
            "(after the remaining trials)"
        )
    look, needed = due
    total = planned_blocks(spec)
    if len(pairs) < needed:
        raise InterimError(
            f"interim look {look} of {len(fractions)} is due after {needed} complete blocks; "
            f"{len(pairs)} so far"
        )
    fraction = len(pairs) / total
    if fraction >= 1.0:
        raise InterimError("every block is complete; run the final analysis instead")
    previous = [float(x["fraction"]) for x in looks_done]
    if previous and fraction <= previous[-1]:
        raise InterimError("no new complete blocks since the last look")
    stopping = spec.analysis.stopping
    primary = spec.analysis.primary
    design = spending_boundaries(
        [*previous, fraction],
        alpha=primary.alpha,
        spending=stopping.spending,
        alternative=primary.alternative,
    )
    z, _, _ = score_z(list(pairs.values()))
    boundary = design.boundaries[-1]
    return InterimDecision(
        look=look,
        planned_looks=len(fractions),
        fraction=fraction,
        blocks=tuple(pairs),
        boundary=boundary,
        decision="stop" if _crossed(z, boundary, primary.alternative) else "continue",
    )


def final_analysis(
    spec: StudySpec,
    records: Sequence[TrialRecord],
    looks_done: Sequence[Mapping[str, Any]],
) -> SequentialOutcome | None:
    """Recompute every look and apply the design. None when no block is complete.

    The repeated confidence interval for the paired difference uses the boundary of the
    deciding look as its multiplier, with the Wald standard error
    ``sqrt(b + c - (b - c)^2 / n) / n``.
    """
    treatment, control, fractions = _settings(spec)
    primary = spec.analysis.primary
    stopping = spec.analysis.stopping
    total = planned_blocks(spec)
    pairs = complete_pairs(records, treatment, control)
    looks = sorted(looks_done, key=lambda x: int(x["look"]))
    notes: list[str] = []

    stop_index = next((i for i, x in enumerate(looks) if x.get("decision") == "stop"), None)
    used = looks if stop_index is None else looks[: stop_index + 1]
    look_pairs = [[pairs[b] for b in x.get("blocks", []) if b in pairs] for x in used]
    looks_fr = [float(x["fraction"]) for x in used]
    kinds: list[Literal["interim", "final"]] = ["interim"] * len(used)
    recorded: list[Decision | None] = [x.get("decision") for x in used]
    if stop_index is None:
        final_fraction = min(len(pairs) / total, 1.0)
        if not looks_fr or final_fraction > looks_fr[-1]:
            looks_fr.append(final_fraction)
            look_pairs.append(list(pairs.values()))
            kinds.append("final")
            recorded.append(None)
    if not look_pairs or not look_pairs[-1]:
        return None

    design = spending_boundaries(
        looks_fr,
        alpha=primary.alpha,
        spending=stopping.spending,
        alternative=primary.alternative,
        spend_all=stop_index is None,
    )
    zs = [score_z(p)[0] for p in look_pairs]
    result = sequential_test(zs, design)
    for i, (z, boundary) in enumerate(zip(zs, design.boundaries, strict=True)):
        decision = recorded[i]
        crossed = _crossed(z, boundary, primary.alternative)
        if decision is not None and (decision == "stop") != crossed:
            notes.append(
                f"Interim look {i + 1} recorded '{decision}', but recomputing it from the "
                f"current labels gives '{'stop' if crossed else 'continue'}' (labels were "
                "edited after the look)."
            )
    skipped = len(fractions) - 1 - len([k for k in kinds if k == "interim"])
    if stop_index is None and skipped > 0:
        notes.append(
            f"{skipped} planned interim look{'s were' if skipped != 1 else ' was'} not run; "
            "error spending keeps the overall error rate at the planned level."
        )

    last = look_pairs[result.stopped_at - 1 if result.stopped_at else len(look_pairs) - 1]
    _, b, c = score_z(last)
    n = len(last)
    estimate = (b - c) / n
    se = math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
    k_star = (result.stopped_at or len(zs)) - 1
    rci_level = 1 - primary.alpha if primary.alternative == "two-sided" else 1 - 2 * primary.alpha
    interval = repeated_interval(estimate, se, design.boundaries[k_star], level=rci_level)

    rows = [
        LookRow(
            look=i + 1,
            kind=kinds[i],
            fraction=looks_fr[i],
            blocks=len(look_pairs[i]),
            z=zs[i],
            boundary=design.boundaries[i],
            crossed=_crossed(zs[i], design.boundaries[i], primary.alternative),
            recorded_decision=recorded[i],
        )
        for i in range(len(zs))
    ]
    pvalue = result.test.pvalue
    summary = SequentialSummary(
        spending=stopping.spending,
        planned_looks=len(fractions),
        planned_fractions=fractions,
        planned_blocks=total,
        looks=rows,
        stopped_at=stop_index + 1 if stop_index is not None else None,
    )
    return SequentialOutcome(
        summary=summary,
        pvalue=None if math.isnan(pvalue) else pvalue,
        rejected=result.test.rejects(primary.alpha),
        interval=interval,
        pairs=n,
        b=b,
        c=c,
        notes=tuple(notes),
    )
