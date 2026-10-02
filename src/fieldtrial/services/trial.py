"""Trials: running the schedule, recording outcomes, voiding, rescheduling, undo and edits.

Every write takes an optional ``idempotency_key`` (a retried request is applied once) and,
where a trial already exists, an optional ``expected_version`` (optimistic concurrency).
"""

import dataclasses
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from fieldtrial.analysis.records import StudyContextInfo, TrialRecord
from fieldtrial.services._context import ConcurrencyError, ServiceError, StudyContext
from fieldtrial.services.events import append_event, idempotent, list_events
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
UNDO_WINDOW_S = 10.0
MAX_MEDIA_BYTES = 200 * 1024 * 1024
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


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


def _view(trial: m.Trial) -> TrialView:
    return TrialView(trial.id, trial.slot_id, trial.status, trial.attempt, trial.version)


def _decode_view(data: dict[str, Any]) -> TrialView:
    return TrialView(**data)


def _result(view: TrialView) -> dict[str, Any]:
    return dataclasses.asdict(view)


def _study(db: Session, ctx: StudyContext) -> m.Study:
    return db.get_one(m.Study, ctx.study_id)


def _require_open(study: m.Study) -> None:
    if study.status == "closed":
        raise ServiceError("the study is closed")


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


def _active_total(db: Session, ctx: StudyContext) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(m.ScheduleSlot)
            .where(m.ScheduleSlot.study_id == ctx.study_id, m.ScheduleSlot.status != "void")
        )
        or 0
    )


def slot_view(ctx: StudyContext, slot_id: str) -> SlotView:
    """One slot, as an operator sees it."""
    with ctx.db() as db:
        slot = db.get(m.ScheduleSlot, slot_id)
        if slot is None or slot.study_id != ctx.study_id:
            raise ServiceError(f"no slot {slot_id}")
        return _slot_view(db, slot, _active_total(db, ctx))


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
        return _slot_view(db, slot, _active_total(db, ctx))


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
    ctx: StudyContext,
    slot_id: str,
    session_id: str,
    *,
    started_at: datetime | None = None,
    idempotency_key: str | None = None,
) -> TrialView:
    """Start a trial for a pending slot."""

    def write() -> TrialView:
        with ctx.db() as db, db.begin():
            slot = _pending_slot(db, ctx, slot_id)
            trial = _new_trial(db, ctx, slot, session_id, started_at or utcnow())
            actor = db.get_one(m.Session, session_id).operator
            view = _view(trial)
            append_event(
                db,
                ctx.study_id,
                "trial_started",
                actor,
                {"trial_id": trial.id, "seq": slot.seq, "result": _result(view)},
                idempotency_key=idempotency_key,
            )
            return view

    return idempotent(ctx, idempotency_key, "trial_started", _decode_view, write)


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
    idempotency_key: str | None = None,
) -> TrialView:
    """Record the outcome of a running trial. Success means reaching the success stage."""
    success = _check_outcome(ctx, stage_index, termination, failure_tags)

    def write() -> TrialView:
        with ctx.db() as db, db.begin():
            trial = _locked_trial(db, ctx, trial_id, expected_version)
            if trial.status != "running":
                raise ServiceError(f"trial {trial_id} is {trial.status}, not running")
            # A stopped trial keeps the time it was stopped, not the time it was labelled.
            end = ended_at or trial.ended_at or utcnow()
            trial.ended_at = end
            if duration_s is not None:
                trial.duration_s = duration_s
            elif trial.duration_s is None or ended_at is not None:
                trial.duration_s = (end - trial.started_at).total_seconds()
            trial.stage_index, trial.success = stage_index, success
            trial.termination, trial.failure_tags = termination, list(failure_tags)
            trial.notes = notes
            trial.status = "completed"
            _bump_version(db, trial)
            slot = db.get_one(m.ScheduleSlot, trial.slot_id)
            slot.status = "done"
            actor = db.get_one(m.Session, trial.session_id).operator
            view = _view(trial)
            append_event(
                db,
                ctx.study_id,
                "trial_completed",
                actor,
                {
                    "trial_id": trial.id,
                    "seq": slot.seq,
                    "stage_index": stage_index,
                    "success": success,
                    "result": _result(view),
                },
                idempotency_key=idempotency_key,
            )
            return view

    return idempotent(ctx, idempotency_key, "trial_completed", _decode_view, write)


def stop_trial(
    ctx: StudyContext,
    trial_id: str,
    *,
    ended_at: datetime | None = None,
    expected_version: int | None = None,
    idempotency_key: str | None = None,
) -> TrialView:
    """Stop the clock on a running trial; it stays running until it is labelled.

    The duration counts from start to stop, so time spent labelling is not included.
    """

    def write() -> TrialView:
        with ctx.db() as db, db.begin():
            trial = _locked_trial(db, ctx, trial_id, expected_version)
            if trial.status != "running":
                raise ServiceError(f"trial {trial_id} is {trial.status}, not running")
            if trial.ended_at is not None:
                raise ServiceError("the trial is already stopped; label it")
            end = ended_at or utcnow()
            trial.ended_at = end
            trial.duration_s = max((end - trial.started_at).total_seconds(), 0.0)
            _bump_version(db, trial)
            actor = db.get_one(m.Session, trial.session_id).operator
            view = _view(trial)
            append_event(
                db,
                ctx.study_id,
                "trial_stopped",
                actor,
                {"trial_id": trial.id, "duration_s": trial.duration_s, "result": _result(view)},
                idempotency_key=idempotency_key,
            )
            return view

    return idempotent(ctx, idempotency_key, "trial_stopped", _decode_view, write)


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
    idempotency_key: str | None = None,
) -> TrialView:
    """Mark a trial invalid (robot fault, setup error, ...) and reschedule its slot.

    Invalid trials are never dropped silently: they stay in the database and are counted per
    arm in every report. The slot is voided and a replacement is inserted at the end of the
    current block (or the end of the study), keeping the design balanced.
    """
    if not reason.strip():
        raise ServiceError("an invalid trial needs a reason")

    def write() -> TrialView:
        with ctx.db() as db, db.begin():
            trial = _locked_trial(db, ctx, trial_id, expected_version)
            if trial.status == "invalid":
                raise ServiceError(f"trial {trial_id} is already invalid")
            trial.status = "invalid"
            trial.invalid_reason = reason
            trial.ended_at = trial.ended_at or utcnow()
            if trial.duration_s is None:
                trial.duration_s = (trial.ended_at - trial.started_at).total_seconds()
            _bump_version(db, trial)
            slot = db.get_one(m.ScheduleSlot, trial.slot_id)
            slot.status = "void"
            replacement = _reschedule(db, ctx, slot, reschedule)
            who = actor or db.get_one(m.Session, trial.session_id).operator
            view = _view(trial)
            append_event(
                db,
                ctx.study_id,
                "trial_invalidated",
                who,
                {
                    "trial_id": trial.id,
                    "reason": reason,
                    "replacement_seq": replacement.seq,
                    "replacement_slot_id": replacement.id,
                    "result": _result(view),
                },
                idempotency_key=idempotency_key,
            )
            return view

    return idempotent(ctx, idempotency_key, "trial_invalidated", _decode_view, write)


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
    idempotency_key: str | None = None,
) -> TrialView:
    """Record a whole trial at once (for imports and simulations), as one write and one event.

    With ``invalid_reason`` the trial is stored as invalid and the slot is rescheduled at the
    end of its block.
    """
    success = _check_outcome(ctx, stage_index, termination, failure_tags)

    def write() -> TrialView:
        with ctx.db() as db, db.begin():
            slot = _pending_slot(db, ctx, slot_id)
            start = started_at or utcnow()
            trial = _new_trial(db, ctx, slot, session_id, start)
            trial.ended_at = start + timedelta(seconds=duration_s)
            trial.duration_s = duration_s
            trial.stage_index, trial.success = stage_index, success
            trial.termination, trial.failure_tags = termination, list(failure_tags)
            trial.notes = notes
            payload: dict[str, Any] = {"trial_id": trial.id, "seq": slot.seq, "source": source}
            if invalid_reason:
                trial.status, trial.invalid_reason = "invalid", invalid_reason
                slot.status = "void"
                replacement = _reschedule(db, ctx, slot, "block")
                payload["replacement_seq"] = replacement.seq
                payload["replacement_slot_id"] = replacement.id
                payload["reason"] = invalid_reason
            else:
                trial.status = "completed"
                slot.status = "done"
                payload["success"] = success
            actor = db.get_one(m.Session, session_id).operator
            view = _view(trial)
            payload["result"] = _result(view)
            append_event(
                db, ctx.study_id, "trial_recorded", actor, payload, idempotency_key=idempotency_key
            )
            return view

    return idempotent(ctx, idempotency_key, "trial_recorded", _decode_view, write)


_FINISH_KINDS = ("trial_completed", "trial_invalidated", "trial_recorded")


def _labelled_at(db: Session, ctx: StudyContext, trial_id: str) -> datetime | None:
    """When a trial was last labelled (completed, invalidated or recorded)."""
    events = db.scalars(
        select(m.Event)
        .where(m.Event.study_id == ctx.study_id, m.Event.kind.in_(_FINISH_KINDS))
        .order_by(m.Event.id.desc())
    )
    for event in events:
        if event.payload.get("trial_id") == trial_id:
            return event.ts
    return None


def labelled_at(ctx: StudyContext, trial_id: str) -> datetime | None:
    """When a trial was last labelled; the undo window counts from here."""
    with ctx.db() as db:
        return _labelled_at(db, ctx, trial_id)


def _replacement_slot(db: Session, ctx: StudyContext, trial_id: str) -> str | None:
    for event in reversed(list_events(db, ctx.study_id)):
        if event.kind in ("trial_invalidated", "trial_recorded") and (
            event.payload.get("trial_id") == trial_id
        ):
            slot_id = event.payload.get("replacement_slot_id")
            return str(slot_id) if slot_id else None
    return None


def reopen_trial(
    ctx: StudyContext,
    trial_id: str,
    *,
    session_id: str | None = None,
    window_s: float | None = UNDO_WINDOW_S,
    now: datetime | None = None,
    expected_version: int | None = None,
    idempotency_key: str | None = None,
) -> TrialView:
    """Undo a finished trial's outcome: the trial is running again, waiting for a new label.

    Allowed within ``window_s`` seconds of the trial ending (``None``: no limit), and only
    for the latest trial of its session (``session_id``, when given, must match). Undoing an
    invalid trial also voids its replacement slot, which must not have been run yet.
    """

    def write() -> TrialView:
        with ctx.db() as db, db.begin():
            trial = _locked_trial(db, ctx, trial_id, expected_version)
            if trial.status not in ("completed", "invalid"):
                raise ServiceError(f"trial {trial_id} is {trial.status}; nothing to undo")
            if session_id is not None and trial.session_id != session_id:
                raise ServiceError("only the session that ran a trial can undo it")
            current = now or utcnow()
            finished = _labelled_at(db, ctx, trial.id) or trial.ended_at or trial.started_at
            if window_s is not None and (current - finished).total_seconds() > window_s:
                raise ServiceError(f"undo is only possible for {window_s:g} s after a trial")
            later = db.scalar(
                select(func.count())
                .select_from(m.Trial)
                .where(
                    m.Trial.session_id == trial.session_id,
                    m.Trial.started_at > trial.started_at,
                )
            )
            if later:
                raise ServiceError("only the latest trial of a session can be undone")
            previous = {
                "status": trial.status,
                "stage_index": trial.stage_index,
                "termination": trial.termination,
                "failure_tags": list(trial.failure_tags or []),
                "notes": trial.notes,
                "invalid_reason": trial.invalid_reason,
            }
            if trial.status == "invalid":
                replacement_id = _replacement_slot(db, ctx, trial.id)
                if replacement_id is not None:
                    replacement = db.get_one(m.ScheduleSlot, replacement_id)
                    has_trials = db.scalar(
                        select(func.count())
                        .select_from(m.Trial)
                        .where(m.Trial.slot_id == replacement_id)
                    )
                    if replacement.status != "pending" or has_trials:
                        raise ServiceError("the replacement trial has already been run")
                    replacement.status = "void"
            trial.status = "running"
            trial.stage_index = trial.success = trial.termination = None
            trial.invalid_reason = None  # ended_at and duration_s keep the stop time
            trial.failure_tags = []
            _bump_version(db, trial)
            slot = db.get_one(m.ScheduleSlot, trial.slot_id)
            slot.status = "pending"
            actor = db.get_one(m.Session, trial.session_id).operator
            view = _view(trial)
            append_event(
                db,
                ctx.study_id,
                "trial_reopened",
                actor,
                {"trial_id": trial.id, "previous": previous, "result": _result(view)},
                idempotency_key=idempotency_key,
            )
            return view

    return idempotent(ctx, idempotency_key, "trial_reopened", _decode_view, write)


def edit_trial(
    ctx: StudyContext,
    trial_id: str,
    *,
    actor: str,
    reason: str,
    stage_index: int | None = None,
    termination: str | None = None,
    failure_tags: Sequence[str] | None = None,
    notes: str | None = None,
    expected_version: int | None = None,
    idempotency_key: str | None = None,
) -> TrialView:
    """Correct a completed trial's label. The old and new values are logged with a reason.

    Edits after unblinding are listed as deviations in every report.
    """
    if not reason.strip():
        raise ServiceError("an edit needs a reason")

    def write() -> TrialView:
        with ctx.db() as db, db.begin():
            trial = _locked_trial(db, ctx, trial_id, expected_version)
            if trial.status != "completed":
                raise ServiceError(
                    f"only completed trials can be edited (this one is {trial.status})"
                )
            old_stage, old_term = trial.stage_index, trial.termination
            old_tags, old_notes = list(trial.failure_tags or []), trial.notes
            new_stage = old_stage if stage_index is None else stage_index
            new_term = old_term if termination is None else termination
            new_tags = old_tags if failure_tags is None else list(failure_tags)
            new_notes = old_notes if notes is None else (notes or None)
            old = {
                "stage_index": old_stage,
                "termination": old_term,
                "failure_tags": old_tags,
                "notes": old_notes,
            }
            new = {
                "stage_index": new_stage,
                "termination": new_term,
                "failure_tags": new_tags,
                "notes": new_notes,
            }
            if new == old:
                raise ServiceError("nothing to change")
            if new_stage is None or new_term is None:
                raise ServiceError("a completed trial needs a stage and a termination")
            trial.success = _check_outcome(ctx, new_stage, new_term, new_tags)
            trial.stage_index, trial.termination = new_stage, new_term
            trial.failure_tags, trial.notes = new_tags, new_notes
            _bump_version(db, trial)
            view = _view(trial)
            append_event(
                db,
                ctx.study_id,
                "trial_edited",
                actor,
                {
                    "trial_id": trial.id,
                    "reason": reason,
                    "old": old,
                    "new": new,
                    "result": _result(view),
                },
                idempotency_key=idempotency_key,
            )
            return view

    return idempotent(ctx, idempotency_key, "trial_edited", _decode_view, write)


def attach_media(
    ctx: StudyContext,
    trial_id: str,
    filename: str,
    data: bytes,
    *,
    actor: str,
    idempotency_key: str | None = None,
) -> str:
    """Save a clip or photo for a trial under ``media/`` and return its path in the folder."""
    if len(data) > MAX_MEDIA_BYTES:
        raise ServiceError(f"media files are limited to {MAX_MEDIA_BYTES // 2**20} MB")
    name = _SAFE_NAME.sub("_", filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]).strip("._")
    if not name:
        raise ServiceError("the media file needs a name")

    def write() -> str:
        with ctx.db() as db, db.begin():
            trial = db.get(m.Trial, trial_id)
            if trial is None or trial.study_id != ctx.study_id:
                raise ServiceError(f"no trial {trial_id}")
            folder = ctx.folder / "media" / trial.id
            folder.mkdir(parents=True, exist_ok=True)
            target, n = folder / name, 1
            while target.exists():
                stem, dot, ext = name.partition(".")
                target = folder / f"{stem}-{n}{dot}{ext}"
                n += 1
            target.write_bytes(data)
            relative = target.relative_to(ctx.folder).as_posix()
            trial.media = [*(trial.media or []), relative]
            append_event(
                db,
                ctx.study_id,
                "media_attached",
                actor,
                {
                    "trial_id": trial.id,
                    "path": relative,
                    "bytes": len(data),
                    "result": {"path": relative},
                },
                idempotency_key=idempotency_key,
            )
            return relative

    return idempotent(ctx, idempotency_key, "media_attached", lambda d: str(d["path"]), write)


# --- reading -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrialDetail:
    """A trial with its slot, for the console and the API."""

    trial_id: str
    slot: SlotView
    session_id: str
    status: str
    attempt: int
    version: int
    started_at: datetime
    ended_at: datetime | None
    duration_s: float | None
    stage_index: int | None
    success: bool | None
    termination: str | None
    failure_tags: tuple[str, ...]
    notes: str | None
    invalid_reason: str | None
    media: tuple[str, ...]


def _detail(db: Session, ctx: StudyContext, trial: m.Trial, total: int) -> TrialDetail:
    return TrialDetail(
        trial_id=trial.id,
        slot=_slot_view(db, db.get_one(m.ScheduleSlot, trial.slot_id), total),
        session_id=trial.session_id,
        status=trial.status,
        attempt=trial.attempt,
        version=trial.version,
        started_at=trial.started_at,
        ended_at=trial.ended_at,
        duration_s=trial.duration_s,
        stage_index=trial.stage_index,
        success=trial.success,
        termination=trial.termination,
        failure_tags=tuple(trial.failure_tags or ()),
        notes=trial.notes,
        invalid_reason=trial.invalid_reason,
        media=tuple(trial.media or ()),
    )


def record_runner_output(
    ctx: StudyContext,
    trial_id: str,
    *,
    metrics: dict[str, float],
    log: str | None = None,
    actor: str = "runner",
) -> None:
    """Store what a runner measured during a trial (exit code, request latency, log path).

    Kept as a ``runner_output`` event, so it never changes the trial's label. While the
    study is blinded the event names only the trial; the arm follows from its slot.
    """
    with ctx.db() as db, db.begin():
        trial = db.get(m.Trial, trial_id)
        if trial is None or trial.study_id != ctx.study_id:
            raise ServiceError(f"no trial {trial_id}")
        append_event(
            db,
            ctx.study_id,
            "runner_output",
            actor,
            {"trial_id": trial_id, "metrics": dict(metrics), "log": log},
        )


def get_trial(ctx: StudyContext, trial_id: str) -> TrialDetail:
    """One trial with its slot."""
    with ctx.db() as db:
        trial = db.get(m.Trial, trial_id)
        if trial is None or trial.study_id != ctx.study_id:
            raise ServiceError(f"no trial {trial_id}")
        return _detail(db, ctx, trial, _active_total(db, ctx))


def list_trials(
    ctx: StudyContext, *, session_id: str | None = None, limit: int | None = None
) -> list[TrialDetail]:
    """Trials (all statuses), newest first."""
    with ctx.db() as db:
        query = select(m.Trial).where(m.Trial.study_id == ctx.study_id)
        if session_id is not None:
            query = query.where(m.Trial.session_id == session_id)
        query = query.order_by(m.Trial.started_at.desc(), m.Trial.id.desc())
        if limit is not None:
            query = query.limit(limit)
        total = _active_total(db, ctx)
        return [_detail(db, ctx, t, total) for t in db.scalars(query)]


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
        outputs = {
            str(e.payload.get("trial_id")): dict(e.payload.get("metrics") or {})
            for e in list_events(db, ctx.study_id, "runner_output")
        }
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
                runner_metrics=outputs.get(t.id, {}),
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
        unblinded = list_events(db, ctx.study_id, "unblinded")
        late_edits = 0
        if study.unblinded_at is not None:
            late_edits = sum(
                1
                for e in list_events(db, ctx.study_id, "trial_edited")
                if e.ts > study.unblinded_at
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
            pending_at_unblinding=(
                int(unblinded[0].payload.get("pending_slots", 0)) if unblinded else None
            ),
            edits_after_unblinding=late_edits,
            interim_looks=tuple(e.payload for e in list_events(db, ctx.study_id, "interim_look")),
            rig_checks=tuple(
                {"ts": e.ts, **dict(e.payload)} for e in list_events(db, ctx.study_id, "rig_check")
            ),
        )
        return records, info
