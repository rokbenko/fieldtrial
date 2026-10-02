"""Looks that run by themselves after each completed trial (docs/stats/anytime.md, selection.md).

For ``analysis.stopping.rule: anytime`` a look that stops the study cancels every pending
trial; for ``analysis.selection`` a look that drops arms cancels their pending trials, and
all of them once one arm remains. Each look is one transaction plus one event
(``interim_look`` or ``selection_look``). A trial that is running is never cancelled; it
finishes normally.
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from fieldtrial.analysis.adaptive import (
    anytime_blocks,
    anytime_check,
    recorded_eliminations,
    selection_check,
    selection_rows,
)
from fieldtrial.services._context import ServiceError, StudyContext
from fieldtrial.services.events import append_event, list_events
from fieldtrial.services.study import SYSTEM_ACTOR
from fieldtrial.services.trial import collect_records
from fieldtrial.store import models as m


@dataclass(frozen=True, slots=True)
class LookOutcome:
    """What an automatic look did. Arms appear by blind code only."""

    rule: str
    decision: str
    dropped: tuple[str, ...]
    voided_slots: int


@dataclass(frozen=True, slots=True)
class AdaptiveStatus:
    """Where an anytime or selection study stands. Blind codes only, no rates."""

    rule: str  # "anytime" or "elimination"
    complete_blocks: int
    min_blocks: int
    stopped: bool
    dropped: tuple[str, ...]  # blind codes of dropped arms, in the order they were dropped
    remaining: int  # arms still running


def _kind(ctx: StudyContext) -> str | None:
    analysis = ctx.spec.analysis
    if analysis.selection is not None:
        return "selection_look"
    if analysis.stopping.rule == "anytime":
        return "interim_look"
    return None


def _payloads(ctx: StudyContext, kind: str) -> list[dict[str, Any]]:
    with ctx.db() as db:
        return [dict(e.payload) for e in list_events(db, ctx.study_id, kind)]


def _blind_codes(ctx: StudyContext) -> dict[str, str]:
    with ctx.db() as db:
        arms = db.scalars(select(m.Arm).where(m.Arm.study_id == ctx.study_id))
        return {a.key: a.blind_code for a in arms}


def adaptive_status(ctx: StudyContext) -> AdaptiveStatus | None:
    """The study's adaptive-look status, or None when it checks nothing after each block."""
    kind = _kind(ctx)
    if kind is None:
        return None
    looks = _payloads(ctx, kind)
    records, _ = collect_records(ctx)
    spec = ctx.spec
    stopped = any(x.get("decision") == "stop" for x in looks)
    if kind == "interim_look":
        comparison = spec.analysis.primary.comparison
        assert comparison is not None
        blocks = len(anytime_blocks(records, comparison.treatment, comparison.control))
        return AdaptiveStatus(
            rule="anytime",
            complete_blocks=blocks,
            min_blocks=spec.analysis.stopping.min_blocks or 1,
            stopped=stopped,
            dropped=(),
            remaining=0 if stopped else len(spec.arms),
        )
    selection = spec.analysis.selection
    assert selection is not None
    dropped = recorded_eliminations(looks)
    codes = _blind_codes(ctx)
    arms = [a.id for a in spec.arms]
    order = sorted(dropped, key=lambda a: (dropped[a], arms.index(a)))
    return AdaptiveStatus(
        rule="elimination",
        complete_blocks=len(selection_rows(records, arms, dropped)),
        min_blocks=selection.min_blocks,
        stopped=stopped,
        dropped=tuple(codes[a] for a in order),
        remaining=0 if stopped else len(arms) - len(dropped),
    )


def _void(db: Any, ctx: StudyContext, arm_keys: set[str] | None) -> int:
    """Cancel pending slots (of ``arm_keys``, or all), except those with a running trial."""
    arm_ids = {a.id: a.key for a in db.scalars(select(m.Arm).where(m.Arm.study_id == ctx.study_id))}
    running = set(
        db.scalars(
            select(m.Trial.slot_id).where(
                m.Trial.study_id == ctx.study_id, m.Trial.status == "running"
            )
        )
    )
    voided = 0
    for slot in db.scalars(
        select(m.ScheduleSlot).where(
            m.ScheduleSlot.study_id == ctx.study_id, m.ScheduleSlot.status == "pending"
        )
    ):
        if slot.id in running:
            continue
        if arm_keys is None or arm_ids[slot.arm_id] in arm_keys:
            slot.status = "void"
            voided += 1
    return voided


def check_after_block(ctx: StudyContext, *, actor: str = SYSTEM_ACTOR) -> LookOutcome | None:
    """Run the automatic look, if the study has one and something changed.

    Returns what the look did, or None when the study goes on unchanged.
    """
    kind = _kind(ctx)
    if kind is None:
        return None
    with ctx.db() as db:
        status = db.get_one(m.Study, ctx.study_id).status
    if status not in ("locked", "running"):
        return None
    looks = _payloads(ctx, kind)
    records, _ = collect_records(ctx)
    spec = ctx.spec
    if kind == "interim_look":
        stop = anytime_check(spec, records, looks)
        if stop is None:
            return None
        payload: dict[str, Any] = stop.payload()
        dropped: set[str] | None = None
        decision = stop.decision
        rule = "anytime"
    else:
        look = selection_check(spec, records, looks)
        if look is None:
            return None
        payload = look.payload()
        dropped = {a for a, _, _ in look.eliminated}
        decision = look.decision
        rule = "elimination"

    with ctx.db() as db, db.begin():
        if len(list_events(db, ctx.study_id, kind)) != len(looks):
            raise ServiceError("another look was recorded meanwhile; try again")
        voided = _void(db, ctx, None if decision == "stop" else dropped)
        append_event(db, ctx.study_id, kind, actor, {**payload, "voided_slots": voided})
    codes = _blind_codes(ctx)
    return LookOutcome(
        rule=rule,
        decision=decision,
        dropped=tuple(codes[a] for a in sorted(dropped or ())),
        voided_slots=voided,
    )
