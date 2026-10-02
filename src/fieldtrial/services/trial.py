"""Sessions and trials: running the schedule, recording outcomes, voiding and rescheduling."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from fieldtrial.analysis.records import StudyContextInfo, TrialRecord
from fieldtrial.services._context import ConcurrencyError, ServiceError, StudyContext
from fieldtrial.services.events import append_event, list_events
from fieldtrial.store import models as m
from fieldtrial.store.models import utcnow

TERMINATIONS = (
    "success",
    "timeout",
    "stuck",
    "operator_stop",
    "robot_fault",
    "safety_stop",
    "other",
)
Reschedule = Literal["block", "end"]


@dataclass(frozen=True, slots=True)
class SlotView:
    """A schedule slot as an operator or runtime sees it."""

    slot_id: str
    seq: int
    block: int
    replicate: int
    position: int
    condition: str
    factors: dict[str, Any]
    arm: str
    blind_code: str
    total: int


@dataclass(frozen=True, slots=True)
class TrialView:
    """A trial and its concurrency version."""

    trial_id: str
    slot_id: str
    status: str
    attempt: int
    version: int


def _study(db: Session, ctx: StudyContext) -> m.Study:
    return db.get_one(m.Study, ctx.study_id)


def _require_open(study: m.Study) -> None:
    if study.status == "closed":
        raise ServiceError("the study is closed")


# --- sessions --------------------------------------------------------------------------------


def start_session(
    ctx: StudyContext,
    *,
    operator: str,
    rig: str,
    rig_check: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    software: dict[str, Any] | None = None,
    notes: str | None = None,
    started_at: datetime | None = None,
) -> str:
    """Start a working session (one operator on one rig) and return its id."""
    if not operator.strip() or not rig.strip():
        raise ServiceError("a session needs an operator and a rig")
    with ctx.db() as db, db.begin():
        _require_open(_study(db, ctx))
        session = m.Session(
            study_id=ctx.study_id,
            operator=operator,
            rig=rig,
            rig_check=rig_check or {},
            environment=environment or {},
            software=software or {},
            notes=notes,
            started_at=started_at or utcnow(),
        )
        db.add(session)
        db.flush()
        append_event(
            db, ctx.study_id, "session_started", operator, {"session_id": session.id, "rig": rig}
        )
        return session.id


def end_session(ctx: StudyContext, session_id: str, *, ended_at: datetime | None = None) -> None:
    """End a session."""
    with ctx.db() as db, db.begin():
        session = db.get(m.Session, session_id)
        if session is None or session.study_id != ctx.study_id:
            raise ServiceError(f"no session {session_id}")
        if session.ended_at is not None:
            raise ServiceError("the session has already ended")
        session.ended_at = ended_at or utcnow()
        append_event(
            db, ctx.study_id, "session_ended", session.operator, {"session_id": session_id}
        )


# --- schedule --------------------------------------------------------------------------------


def _slot_view(db: Session, slot: m.ScheduleSlot, total: int) -> SlotView:
    cond = db.get_one(m.Condition, slot.condition_id)
    arm = db.get_one(m.Arm, slot.arm_id)
    return SlotView(
        slot_id=slot.id,
        seq=slot.seq,
        block=slot.block,
        replicate=slot.replicate,
        position=slot.position,
        condition=cond.key,
        factors=dict(cond.factors),
        arm=arm.key,
        blind_code=arm.blind_code,
        total=total,
    )


def next_slot(ctx: StudyContext) -> SlotView | None:
    """The first pending slot in run order that has no trial running, or None when done."""
    with ctx.db() as db:
        running = select(m.Trial.slot_id).where(
            m.Trial.study_id == ctx.study_id, m.Trial.status == "running"
        )
        slot = db.scalars(
            select(m.ScheduleSlot)
            .where(
                m.ScheduleSlot.study_id == ctx.study_id,
                m.ScheduleSlot.status == "pending",
                m.ScheduleSlot.id.not_in(running),
            )
            .order_by(m.ScheduleSlot.seq)
            .limit(1)
        ).first()
        if slot is None:
            return None
        total = db.scalar(
            select(func.count())
            .select_from(m.ScheduleSlot)
            .where(m.ScheduleSlot.study_id == ctx.study_id, m.ScheduleSlot.status != "void")
        )
        return _slot_view(db, slot, int(total or 0))


def pending_slots(ctx: StudyContext) -> list[SlotView]:
    """All pending slots in run order."""
    with ctx.db() as db:
        slots = list(
            db.scalars(
                select(m.ScheduleSlot)
                .where(m.ScheduleSlot.study_id == ctx.study_id, m.ScheduleSlot.status == "pending")
                .order_by(m.ScheduleSlot.seq)
            )
        )
        return [_slot_view(db, s, len(slots)) for s in slots]


# --- trials ----------------------------------------------------------------------------------


def _check_outcome(
    ctx: StudyContext, stage_index: int, termination: str, failure_tags: Sequence[str]
) -> bool:
    n_stages = len(ctx.spec.rubric.stages)
    if not -1 <= stage_index < n_stages:
        raise ServiceError(f"stage_index must be between -1 and {n_stages - 1}")
    if termination not in TERMINATIONS:
        raise ServiceError(f"termination must be one of {TERMINATIONS}")
    unknown = set(failure_tags) - set(ctx.spec.rubric.failure_tags)
    if unknown:
        raise ServiceError(f"unknown failure tags: {', '.join(sorted(unknown))}")
    return stage_index >= ctx.spec.rubric.success_index


def _pending_slot(db: Session, ctx: StudyContext, slot_id: str) -> m.ScheduleSlot:
    slot = db.get(m.ScheduleSlot, slot_id)
    if slot is None or slot.study_id != ctx.study_id:
        raise ServiceError(f"no slot {slot_id}")
    if slot.status != "pending":
        raise ServiceError(f"slot {slot.seq} is {slot.status}, not pending")
    return slot


def _new_trial(
    db: Session, ctx: StudyContext, slot: m.ScheduleSlot, session_id: str, started_at: datetime
) -> m.Trial:
    session = db.get(m.Session, session_id)
    if session is None or session.study_id != ctx.study_id:
        raise ServiceError(f"no session {session_id}")
    if session.ended_at is not None:
        raise ServiceError("the session has ended; start a new one")
    if db.scalar(
        select(func.count())
        .select_from(m.Trial)
        .where(m.Trial.slot_id == slot.id, m.Trial.status == "running")
    ):
        raise ServiceError(f"slot {slot.seq} already has a running trial")
    attempt = 1 + int(
        db.scalar(select(func.count()).select_from(m.Trial).where(m.Trial.slot_id == slot.id)) or 0
    )
    study = _study(db, ctx)
    _require_open(study)
    if study.status == "locked":
        study.status = "running"
    trial = m.Trial(
        study_id=ctx.study_id,
        slot_id=slot.id,
        session_id=session_id,
        attempt=attempt,
        status="running",
        started_at=started_at,
    )
    db.add(trial)
    db.flush()
    return trial


def start_trial(
    ctx: StudyContext, slot_id: str, session_id: str, *, started_at: datetime | None = None
) -> TrialView:
    """Start a trial for a pending slot."""
    with ctx.db() as db, db.begin():
        slot = _pending_slot(db, ctx, slot_id)
        trial = _new_trial(db, ctx, slot, session_id, started_at or utcnow())
        actor = db.get_one(m.Session, session_id).operator
        append_event(
            db, ctx.study_id, "trial_started", actor, {"trial_id": trial.id, "seq": slot.seq}
        )
        return TrialView(trial.id, slot.id, trial.status, trial.attempt, trial.version)


def _locked_trial(
    db: Session, ctx: StudyContext, trial_id: str, expected_version: int | None
) -> m.Trial:
    trial = db.get(m.Trial, trial_id)
    if trial is None or trial.study_id != ctx.study_id:
        raise ServiceError(f"no trial {trial_id}")
    if expected_version is not None and trial.version != expected_version:
        raise ConcurrencyError(
            f"trial {trial_id} changed (version {trial.version}, expected {expected_version})"
        )
    return trial


def _bump_version(db: Session, trial: m.Trial) -> None:
    # Compare-and-set on the version column: a concurrent writer makes this update miss.
    result = db.execute(
        update(m.Trial)
        .where(m.Trial.id == trial.id, m.Trial.version == trial.version)
        .values(version=trial.version + 1)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:  # type: ignore[attr-defined]
        raise ConcurrencyError(f"trial {trial.id} was changed by someone else")
    trial.version += 1


def complete_trial(
    ctx: StudyContext,
    trial_id: str,
    *,
    stage_index: int,
    termination: str,
    failure_tags: Sequence[str] = (),
    notes: str | None = None,
    duration_s: float | None = None,
    ended_at: datetime | None = None,
    expected_version: int | None = None,
) -> TrialView:
    """Record the outcome of a running trial. Success means reaching the success stage."""
    success = _check_outcome(ctx, stage_index, termination, failure_tags)
    with ctx.db() as db, db.begin():
        trial = _locked_trial(db, ctx, trial_id, expected_version)
        if trial.status != "running":
            raise ServiceError(f"trial {trial_id} is {trial.status}, not running")
        end = ended_at or utcnow()
        trial.ended_at = end
        trial.duration_s = (
            duration_s if duration_s is not None else (end - trial.started_at).total_seconds()
        )
        trial.stage_index, trial.success = stage_index, success
        trial.termination, trial.failure_tags, trial.notes = termination, list(failure_tags), notes
        trial.status = "completed"
        _bump_version(db, trial)
        slot = db.get_one(m.ScheduleSlot, trial.slot_id)
        slot.status = "done"
        actor = db.get_one(m.Session, trial.session_id).operator
        append_event(
            db,
            ctx.study_id,
            "trial_completed",
            actor,
            {"trial_id": trial.id, "seq": slot.seq, "stage_index": stage_index, "success": success},
        )
        return TrialView(trial.id, slot.id, trial.status, trial.attempt, trial.version)


def _reschedule(
    db: Session, ctx: StudyContext, slot: m.ScheduleSlot, where: Reschedule
) -> m.ScheduleSlot:
    """Insert a replacement slot at the end of the slot's block (default) or of the study."""
    slots = select(m.ScheduleSlot).where(m.ScheduleSlot.study_id == ctx.study_id)
    if where == "block":
        new_seq = 1 + int(
            db.scalar(
                select(func.max(m.ScheduleSlot.seq)).where(
                    m.ScheduleSlot.study_id == ctx.study_id, m.ScheduleSlot.block == slot.block
                )
            )
            or 0
        )
        # Make room by shifting everything after the block back by one.
        for later in db.scalars(
            slots.where(m.ScheduleSlot.seq >= new_seq).order_by(m.ScheduleSlot.seq.desc())
        ):
            later.seq += 1
    else:
        new_seq = 1 + int(
            db.scalar(
                select(func.max(m.ScheduleSlot.seq)).where(m.ScheduleSlot.study_id == ctx.study_id)
            )
            or 0
        )
    replacement = m.ScheduleSlot(
        study_id=ctx.study_id,
        seq=new_seq,
        block=slot.block,
        replicate=slot.replicate,
        position=slot.position,
        condition_id=slot.condition_id,
        arm_id=slot.arm_id,
        origin="rescheduled",
        status="pending",
    )
    db.add(replacement)
    db.flush()
    return replacement


def invalidate_trial(
    ctx: StudyContext,
    trial_id: str,
    reason: str,
    *,
    reschedule: Reschedule = "block",
    expected_version: int | None = None,
    actor: str | None = None,
) -> TrialView:
    """Mark a trial invalid (robot fault, setup error, ...) and reschedule its slot.

    Invalid trials are never dropped silently: they stay in the database and are counted per
    arm in every report. The slot is voided and a replacement is inserted at the end of the
    current block (or the end of the study), keeping the design balanced.
    """
    if not reason.strip():
        raise ServiceError("an invalid trial needs a reason")
    with ctx.db() as db, db.begin():
        trial = _locked_trial(db, ctx, trial_id, expected_version)
        if trial.status == "invalid":
            raise ServiceError(f"trial {trial_id} is already invalid")
        trial.status = "invalid"
        trial.invalid_reason = reason
        trial.ended_at = trial.ended_at or utcnow()
        _bump_version(db, trial)
        slot = db.get_one(m.ScheduleSlot, trial.slot_id)
        slot.status = "void"
        replacement = _reschedule(db, ctx, slot, reschedule)
        who = actor or db.get_one(m.Session, trial.session_id).operator
        append_event(
            db,
            ctx.study_id,
            "trial_invalidated",
            who,
            {"trial_id": trial.id, "reason": reason, "replacement_seq": replacement.seq},
        )
        return TrialView(trial.id, slot.id, trial.status, trial.attempt, trial.version)


def record_trial(
    ctx: StudyContext,
    slot_id: str,
    session_id: str,
    *,
    stage_index: int,
    termination: str,
    duration_s: float,
    failure_tags: Sequence[str] = (),
    notes: str | None = None,
    started_at: datetime | None = None,
    invalid_reason: str | None = None,
    source: str = "manual",
) -> TrialView:
    """Record a whole trial at once (for imports and simulations), as one write and one event.

    With ``invalid_reason`` the trial is stored as invalid and the slot is rescheduled at the
    end of its block.
    """
    success = _check_outcome(ctx, stage_index, termination, failure_tags)
    with ctx.db() as db, db.begin():
        slot = _pending_slot(db, ctx, slot_id)
        start = started_at or utcnow()
        trial = _new_trial(db, ctx, slot, session_id, start)
        trial.ended_at = start + timedelta(seconds=duration_s)
        trial.duration_s = duration_s
        trial.stage_index, trial.success = stage_index, success
        trial.termination, trial.failure_tags, trial.notes = termination, list(failure_tags), notes
        payload: dict[str, Any] = {"trial_id": trial.id, "seq": slot.seq, "source": source}
        if invalid_reason:
            trial.status, trial.invalid_reason = "invalid", invalid_reason
            slot.status = "void"
            payload["replacement_seq"] = _reschedule(db, ctx, slot, "block").seq
            payload["reason"] = invalid_reason
        else:
            trial.status = "completed"
            slot.status = "done"
            payload["success"] = success
        actor = db.get_one(m.Session, session_id).operator
        append_event(db, ctx.study_id, "trial_recorded", actor, payload)
        return TrialView(trial.id, slot.id, trial.status, trial.attempt, trial.version)


# --- reading -----------------------------------------------------------------------------------


def collect_records(ctx: StudyContext) -> tuple[list[TrialRecord], StudyContextInfo]:
    """All finished trials (completed and invalid) in run order, plus study facts."""
    with ctx.db() as db:
        study = _study(db, ctx)
        rows = db.execute(
            select(m.Trial, m.ScheduleSlot, m.Arm, m.Condition, m.Session)
            .join(m.ScheduleSlot, m.Trial.slot_id == m.ScheduleSlot.id)
            .join(m.Arm, m.ScheduleSlot.arm_id == m.Arm.id)
            .join(m.Condition, m.ScheduleSlot.condition_id == m.Condition.id)
            .join(m.Session, m.Trial.session_id == m.Session.id)
            .where(m.Trial.study_id == ctx.study_id, m.Trial.status != "running")
            .order_by(m.Trial.started_at, m.Trial.id)
        ).all()
        records = [
            TrialRecord(
                trial_id=t.id,
                seq=s.seq,
                block=s.block,
                replicate=s.replicate,
                condition=c.key,
                arm=a.key,
                blind_code=a.blind_code,
                session_id=sess.id,
                operator=sess.operator,
                rig=sess.rig,
                status=t.status,
                attempt=t.attempt,
                origin=s.origin,
                started_at=t.started_at,
                stage_index=t.stage_index,
                success=t.success,
                termination=t.termination,
                duration_s=t.duration_s,
                failure_tags=tuple(t.failure_tags or ()),
                notes=t.notes,
                invalid_reason=t.invalid_reason,
            )
            for t, s, a, c, sess in rows
        ]
        statuses = list(
            db.scalars(select(m.ScheduleSlot.status).where(m.ScheduleSlot.study_id == ctx.study_id))
        )
        amendments = tuple(
            {
                "ts": e.ts.isoformat(),
                "actor": e.actor,
                **{k: v for k, v in e.payload.items() if k != "diff"},
            }
            for e in list_events(db, ctx.study_id, "amendment")
        )
        info = StudyContextInfo(
            design_hash=study.design_hash,
            status=study.status,
            locked_at=study.locked_at,
            unblinded_at=study.unblinded_at,
            fieldtrial_version=study.fieldtrial_version,
            amendments=amendments,
            planned_slots=sum(1 for st in statuses if st != "void"),
            pending_slots=sum(1 for st in statuses if st == "pending"),
        )
        return records, info
