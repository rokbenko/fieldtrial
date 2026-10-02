"""Sessions: one operator on one rig for one continuous stretch of trials."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from fieldtrial.services._context import ServiceError, StudyContext
from fieldtrial.services.events import append_event, idempotent
from fieldtrial.services.trial import (
    UNDO_WINDOW_S,
    SlotView,
    TrialDetail,
    _require_open,
    _study,
    labelled_at,
    list_trials,
    next_slot,
)
from fieldtrial.store import models as m
from fieldtrial.store.models import utcnow


@dataclass(frozen=True, slots=True)
class SessionView:
    """A session as the console shows it."""

    session_id: str
    operator: str
    rig: str
    started_at: datetime
    ended_at: datetime | None
    rig_check: dict[str, Any]
    notes: str | None


def _session_view(row: m.Session) -> SessionView:
    return SessionView(
        session_id=row.id,
        operator=row.operator,
        rig=row.rig,
        started_at=row.started_at,
        ended_at=row.ended_at,
        rig_check=dict(row.rig_check or {}),
        notes=row.notes,
    )


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
    idempotency_key: str | None = None,
) -> str:
    """Start a working session (one operator on one rig) and return its id."""
    if not operator.strip() or not rig.strip():
        raise ServiceError("a session needs an operator and a rig")

    def write() -> str:
        with ctx.db() as db, db.begin():
            _require_open(_study(db, ctx))
            session = m.Session(
                study_id=ctx.study_id,
                operator=operator.strip(),
                rig=rig.strip(),
                rig_check=rig_check or {},
                environment=environment or {},
                software=software or {},
                notes=notes,
                started_at=started_at or utcnow(),
            )
            db.add(session)
            db.flush()
            append_event(
                db,
                ctx.study_id,
                "session_started",
                session.operator,
                {
                    "session_id": session.id,
                    "rig": session.rig,
                    "result": {"session_id": session.id},
                },
                idempotency_key=idempotency_key,
            )
            return session.id

    return idempotent(
        ctx, idempotency_key, "session_started", lambda d: str(d["session_id"]), write
    )


def end_session(
    ctx: StudyContext,
    session_id: str,
    *,
    ended_at: datetime | None = None,
    idempotency_key: str | None = None,
) -> None:
    """End a session. A trial still running in it must be finished first."""

    def write() -> None:
        with ctx.db() as db, db.begin():
            session = db.get(m.Session, session_id)
            if session is None or session.study_id != ctx.study_id:
                raise ServiceError(f"no session {session_id}")
            if session.ended_at is not None:
                raise ServiceError("the session has already ended")
            running = db.scalar(
                select(func.count())
                .select_from(m.Trial)
                .where(m.Trial.session_id == session_id, m.Trial.status == "running")
            )
            if running:
                raise ServiceError("finish or invalidate the running trial first")
            session.ended_at = ended_at or utcnow()
            append_event(
                db,
                ctx.study_id,
                "session_ended",
                session.operator,
                {"session_id": session_id, "result": {}},
                idempotency_key=idempotency_key,
            )

    idempotent(ctx, idempotency_key, "session_ended", lambda _d: None, write)


def get_session(ctx: StudyContext, session_id: str) -> SessionView:
    """One session."""
    with ctx.db() as db:
        row = db.get(m.Session, session_id)
        if row is None or row.study_id != ctx.study_id:
            raise ServiceError(f"no session {session_id}")
        return _session_view(row)


def list_sessions(ctx: StudyContext, *, open_only: bool = False) -> list[SessionView]:
    """Sessions, newest first."""
    with ctx.db() as db:
        query = select(m.Session).where(m.Session.study_id == ctx.study_id)
        if open_only:
            query = query.where(m.Session.ended_at.is_(None))
        rows = db.scalars(query.order_by(m.Session.started_at.desc(), m.Session.id.desc()))
        return [_session_view(r) for r in rows]


@dataclass(frozen=True, slots=True)
class ConsoleState:
    """Everything the trial console shows for one session.

    ``running`` is the trial in progress (it may be waiting for its label after Stop);
    ``up_next`` is the slot to run next. ``undoable`` is the session's latest finished trial
    while it can still be undone.
    """

    session: SessionView
    running: TrialDetail | None
    up_next: SlotView | None
    undoable: TrialDetail | None
    done: int
    total: int
    undo_until: datetime | None = None


def console_state(
    ctx: StudyContext, session_id: str, *, now: datetime | None = None
) -> ConsoleState:
    """The console's view of a session."""
    session = get_session(ctx, session_id)
    recent = list_trials(ctx, session_id=session_id, limit=1)
    latest = recent[0] if recent else None
    running = latest if latest is not None and latest.status == "running" else None
    undoable = None
    undo_until = None
    if latest is not None and latest.status in ("completed", "invalid"):
        finished = labelled_at(ctx, latest.trial_id) or latest.ended_at
        if finished is not None:
            age = ((now or utcnow()) - finished).total_seconds()
            if age <= UNDO_WINDOW_S:
                undoable = latest
                undo_until = finished + timedelta(seconds=UNDO_WINDOW_S)
    with ctx.db() as db:
        counts = dict(
            db.execute(
                select(m.ScheduleSlot.status, func.count())
                .where(m.ScheduleSlot.study_id == ctx.study_id)
                .group_by(m.ScheduleSlot.status)
            ).all()
        )
    return ConsoleState(
        session=session,
        running=running,
        up_next=None if running else next_slot(ctx),
        undoable=undoable,
        undo_until=undo_until,
        done=int(counts.get("done", 0)),
        total=int(counts.get("done", 0) + counts.get("pending", 0)),
    )
