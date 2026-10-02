"""Interim looks of a group-sequential study (docs/stats/sequential.md).

An interim look reports only "continue" or "stop", so it can run while the study is
blinded. A look that stops the study voids the remaining pending trials in the same
transaction, and both are recorded in one ``interim_look`` event.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from fieldtrial.analysis.sequential import (
    InterimError,
    complete_pairs,
    interim_decision,
    next_look_due,
)
from fieldtrial.services._context import ServiceError, StudyContext, open_study
from fieldtrial.services.events import append_event, list_events
from fieldtrial.services.study import SYSTEM_ACTOR
from fieldtrial.services.trial import collect_records
from fieldtrial.store import models as m


@dataclass(frozen=True, slots=True)
class InterimStatus:
    """Where a group-sequential study stands. Counts only, nothing per arm."""

    planned_looks: int
    looks_done: int
    complete_blocks: int
    next_look: int | None  # None when no interim look remains or the study stopped
    blocks_needed: int | None
    stopped_at: int | None

    @property
    def due(self) -> bool:
        """Whether the next interim look can be run now."""
        return self.blocks_needed is not None and self.complete_blocks >= self.blocks_needed


@dataclass(frozen=True, slots=True)
class InterimResult:
    """The outcome of an interim look."""

    look: int
    planned_looks: int
    decision: str
    complete_blocks: int
    voided_slots: int


def _looks(ctx: StudyContext) -> list[dict[str, Any]]:
    with ctx.db() as db:
        return [dict(e.payload) for e in list_events(db, ctx.study_id, "interim_look")]


def interim_status(ctx: StudyContext) -> InterimStatus | None:
    """The study's interim-look status, or None when it has no group-sequential rule."""
    spec = ctx.spec
    comparison = spec.analysis.primary.comparison
    stopping = spec.analysis.stopping
    if stopping.rule != "group_sequential" or comparison is None:
        return None
    looks = _looks(ctx)
    records, _ = collect_records(ctx)
    complete = len(complete_pairs(records, comparison.treatment, comparison.control))
    due = next_look_due(spec, looks, complete)
    stopped = next((int(x["look"]) for x in looks if x.get("decision") == "stop"), None)
    return InterimStatus(
        planned_looks=len(stopping.fractions()),
        looks_done=len(looks),
        complete_blocks=complete,
        next_look=due[0] if due else None,
        blocks_needed=due[1] if due else None,
        stopped_at=stopped,
    )


def run_interim_in(ctx: StudyContext, *, actor: str = SYSTEM_ACTOR) -> InterimResult:
    """Run the next planned interim look on an open study."""
    with ctx.db() as db:
        study = db.get_one(m.Study, ctx.study_id)
        if study.status == "draft":
            raise ServiceError("lock the study before running an interim look")
        running = db.scalar(
            select(func.count())
            .select_from(m.Trial)
            .where(m.Trial.study_id == ctx.study_id, m.Trial.status == "running")
        )
        if running:
            raise ServiceError("a trial is running; finish it before the interim look")
    records, _ = collect_records(ctx)
    looks = _looks(ctx)
    try:
        decision = interim_decision(ctx.spec, records, looks)
    except InterimError as exc:
        raise ServiceError(str(exc)) from exc

    with ctx.db() as db, db.begin():
        if len(list_events(db, ctx.study_id, "interim_look")) != len(looks):
            raise ServiceError("another interim look was recorded meanwhile; try again")
        voided = 0
        if decision.decision == "stop":
            for slot in db.scalars(
                select(m.ScheduleSlot).where(
                    m.ScheduleSlot.study_id == ctx.study_id, m.ScheduleSlot.status == "pending"
                )
            ):
                slot.status = "void"
                voided += 1
        append_event(
            db,
            ctx.study_id,
            "interim_look",
            actor,
            {**decision.payload(), "voided_slots": voided},
        )
    return InterimResult(
        look=decision.look,
        planned_looks=decision.planned_looks,
        decision=decision.decision,
        complete_blocks=len(decision.blocks),
        voided_slots=voided,
    )


def run_interim(folder: str | Path, *, actor: str = SYSTEM_ACTOR) -> InterimResult:
    """Run the next planned interim look (see :func:`run_interim_in`)."""
    with open_study(folder) as ctx:
        return run_interim_in(ctx, actor=actor)
