"""REST API v1: lets any runtime (a ROS 2 node, a script, C++ over HTTP) run a study.

The flow for one trial: ``GET .../sessions/{id}`` gives ``up_next``; ``POST .../trials``
starts it; ``POST .../trials/{id}/complete`` (or ``/invalidate``) records the outcome.
Send an ``Idempotency-Key`` header on writes you may retry. Errors are JSON
``{"detail": ...}``: 400 for a request the study cannot accept, 404 for an unknown study,
409 when a trial changed since you read it (``expected_version``).
"""

import functools
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, Depends, File, Header, Query, Request, UploadFile
from sse_starlette import EventSourceResponse, ServerSentEvent

from fieldtrial.analysis import wording
from fieldtrial.api.schemas import (
    CompleteIn,
    ConsoleOut,
    EditIn,
    ErrorOut,
    InterimOut,
    InterimStatusOut,
    InvalidateIn,
    MediaOut,
    SessionIn,
    SessionOut,
    SlotOut,
    StartIn,
    StatusOut,
    StopIn,
    StudyOut,
    TrialOut,
    UndoIn,
)
from fieldtrial.services import ServiceError, StudyContext
from fieldtrial.services.events import events_after, latest_event_id
from fieldtrial.services.interim import interim_status, run_interim_in
from fieldtrial.services.registry import StudyRegistry
from fieldtrial.services.session import (
    ConsoleState,
    SessionView,
    console_state,
    end_session,
    get_session,
    start_session,
)
from fieldtrial.services.study import is_blinded, status_of
from fieldtrial.services.trial import (
    SlotView,
    TrialDetail,
    attach_media,
    complete_trial,
    edit_trial,
    get_trial,
    invalidate_trial,
    list_trials,
    next_slot,
    reopen_trial,
    start_trial,
    stop_trial,
)
from fieldtrial.web.runners import Runners

ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorOut},
    404: {"model": ErrorOut},
    409: {"model": ErrorOut},
}
router = APIRouter(prefix="/api/v1", responses=ERRORS)
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key", max_length=64)]
SSE_POLL_S = 0.5


def registry(request: Request) -> StudyRegistry:
    """The studies this server serves."""
    reg: StudyRegistry = request.app.state.registry
    return reg


Registry = Annotated[StudyRegistry, Depends(registry)]


# --- conversions -------------------------------------------------------------------------------


def stage_id(ctx: StudyContext, index: int | None) -> str | None:
    """The stage id for an index (``None`` for -1 or unlabelled)."""
    if index is None or index < 0:
        return None
    return ctx.spec.rubric.stages[index].id


def stage_index(ctx: StudyContext, stage: str | None) -> int:
    """The index of a stage id (``None`` means no stage reached: -1)."""
    if stage is None or stage == "":
        return -1
    ids = [s.id for s in ctx.spec.rubric.stages]
    if stage not in ids:
        raise ServiceError(f"unknown stage {stage!r}; stages are {ids}")
    return ids.index(stage)


def slot_out(ctx: StudyContext, slot: SlotView, blinded: bool) -> SlotOut:
    """A slot, with arm details only when the study is not blinded."""
    arm = None if blinded else ctx.spec.arm(slot.arm)
    return SlotOut(
        slot_id=slot.slot_id,
        seq=slot.seq,
        total=slot.total,
        block=slot.block,
        replicate=slot.replicate,
        condition=slot.condition,
        factors=slot.factors,
        blind_code=slot.blind_code,
        arm=None if arm is None else arm.id,
        policy=None if arm is None else dict(arm.policy),
        serving=None if arm is None else dict(arm.serving),
    )


def trial_out(ctx: StudyContext, trial: TrialDetail, blinded: bool) -> TrialOut:
    """A trial, blinded as needed."""
    return TrialOut(
        trial_id=trial.trial_id,
        slot=slot_out(ctx, trial.slot, blinded),
        session_id=trial.session_id,
        status=trial.status,  # type: ignore[arg-type]
        attempt=trial.attempt,
        version=trial.version,
        started_at=trial.started_at,
        ended_at=trial.ended_at,
        duration_s=trial.duration_s,
        stage=stage_id(ctx, trial.stage_index),
        stage_index=trial.stage_index,
        success=trial.success,
        termination=trial.termination,
        failure_tags=list(trial.failure_tags),
        notes=trial.notes,
        invalid_reason=trial.invalid_reason,
        media=list(trial.media),
    )


def session_out(session: SessionView) -> SessionOut:
    """A session."""
    return SessionOut(
        session_id=session.session_id,
        operator=session.operator,
        rig=session.rig,
        started_at=session.started_at,
        ended_at=session.ended_at,
    )


def console_out(ctx: StudyContext, state: ConsoleState) -> ConsoleOut:
    """A session's console state."""
    blinded = is_blinded(ctx)
    return ConsoleOut(
        session=session_out(state.session),
        running=None if state.running is None else trial_out(ctx, state.running, blinded),
        up_next=None if state.up_next is None else slot_out(ctx, state.up_next, blinded),
        undoable=None if state.undoable is None else trial_out(ctx, state.undoable, blinded),
        done=state.done,
        total=state.total,
    )


# --- studies -----------------------------------------------------------------------------------


@router.get("/studies", response_model=list[StudyOut])
def studies(reg: Registry) -> list[StudyOut]:
    """Studies in the served folder."""
    return [
        StudyOut(slug=e.slug, name=e.name, title=e.title, status=e.status, locked=e.locked)
        for e in reg.entries()
    ]


@router.get("/studies/{slug}", response_model=StatusOut)
def status(slug: str, reg: Registry) -> StatusOut:
    """Progress of a study. No per-arm results while blinded."""
    ctx = reg.get(slug)
    report = status_of(ctx)
    rubric = ctx.spec.rubric
    return StatusOut(
        slug=slug,
        name=report.name,
        status=report.status,
        design_hash=report.design_hash,
        blinded=report.blinded,
        planned=report.planned - report.void,
        done=report.done,
        pending=report.pending,
        invalid_trials=report.invalid_trials,
        sessions=report.sessions,
        amendments=report.amendments,
        stages=[s.id for s in rubric.stages],
        success_stage=rubric.success,
        failure_tags=list(rubric.failure_tags),
        timeout_s=ctx.spec.limits.timeout_s,
        per_arm=report.per_arm,
    )


@router.get("/studies/{slug}/next", response_model=SlotOut | None)
def next_trial(slug: str, reg: Registry) -> SlotOut | None:
    """The next slot to run, or null when every slot is done or running."""
    ctx = reg.get(slug)
    slot = next_slot(ctx)
    return None if slot is None else slot_out(ctx, slot, is_blinded(ctx))


@router.get("/studies/{slug}/interim", response_model=InterimStatusOut | None)
def interim(slug: str, reg: Registry) -> InterimStatusOut | None:
    """Interim-look status, or null when the study has no group-sequential rule."""
    found = interim_status(reg.get(slug))
    if found is None:
        return None
    return InterimStatusOut(
        planned_looks=found.planned_looks,
        looks_done=found.looks_done,
        complete_blocks=found.complete_blocks,
        next_look=found.next_look,
        blocks_needed=found.blocks_needed,
        due=found.due,
        stopped_at=found.stopped_at,
    )


@router.post("/studies/{slug}/interim", response_model=InterimOut)
def run_interim(slug: str, reg: Registry) -> InterimOut:
    """Run the interim look that is due. A stop cancels the remaining trials."""
    result = run_interim_in(reg.get(slug), actor="api")
    return InterimOut(
        look=result.look,
        planned_looks=result.planned_looks,
        decision=result.decision,
        complete_blocks=result.complete_blocks,
        voided_slots=result.voided_slots,
        message=wording.interim(result.decision, result.look, result.planned_looks),
    )


# --- sessions ----------------------------------------------------------------------------------


@router.post("/studies/{slug}/sessions", response_model=ConsoleOut, status_code=201)
def create_session(
    slug: str, body: SessionIn, reg: Registry, key: IdempotencyKey = None
) -> ConsoleOut:
    """Start a session and return its console state."""
    ctx = reg.get(slug)
    session_id = start_session(
        ctx,
        operator=body.operator,
        rig=body.rig,
        rig_check=dict(body.rig_check),
        notes=body.notes,
        idempotency_key=key,
    )
    return console_out(ctx, console_state(ctx, session_id))


@router.get("/studies/{slug}/sessions/{session_id}", response_model=ConsoleOut)
def session(slug: str, session_id: str, reg: Registry) -> ConsoleOut:
    """A session's running trial, next slot and progress."""
    ctx = reg.get(slug)
    return console_out(ctx, console_state(ctx, session_id))


@router.post("/studies/{slug}/sessions/{session_id}/end", response_model=SessionOut)
def finish_session(
    slug: str, session_id: str, reg: Registry, key: IdempotencyKey = None
) -> SessionOut:
    """End a session (a running trial must be finished first)."""
    ctx = reg.get(slug)
    end_session(ctx, session_id, idempotency_key=key)
    return session_out(get_session(ctx, session_id))


# --- trials ------------------------------------------------------------------------------------


@router.get("/studies/{slug}/trials", response_model=list[TrialOut])
def trials(
    slug: str,
    reg: Registry,
    session_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=10_000)] = 200,
) -> list[TrialOut]:
    """Trials, newest first."""
    ctx = reg.get(slug)
    blinded = is_blinded(ctx)
    return [
        trial_out(ctx, t, blinded) for t in list_trials(ctx, session_id=session_id, limit=limit)
    ]


@router.get("/studies/{slug}/trials/{trial_id}", response_model=TrialOut)
def trial(slug: str, trial_id: str, reg: Registry) -> TrialOut:
    """One trial."""
    ctx = reg.get(slug)
    return trial_out(ctx, get_trial(ctx, trial_id), is_blinded(ctx))


def _runners(request: Request) -> Runners:
    found: Runners = request.app.state.runners
    return found


def _trial(ctx: StudyContext, trial_id: str) -> TrialOut:
    return trial_out(ctx, get_trial(ctx, trial_id), is_blinded(ctx))


@router.post("/studies/{slug}/trials", response_model=TrialOut, status_code=201)
def start(
    request: Request, slug: str, body: StartIn, reg: Registry, key: IdempotencyKey = None
) -> TrialOut:
    """Start a trial for a pending slot (and its arm, when a runner runs the policy)."""
    ctx = reg.get(slug)
    view = start_trial(ctx, body.slot_id, body.session_id, idempotency_key=key)
    _runners(request).started(slug, ctx, get_trial(ctx, view.trial_id))
    return _trial(ctx, view.trial_id)


@router.post("/studies/{slug}/trials/{trial_id}/stop", response_model=TrialOut)
def stop(
    request: Request,
    slug: str,
    trial_id: str,
    body: StopIn,
    reg: Registry,
    key: IdempotencyKey = None,
) -> TrialOut:
    """Stop the clock; the duration ends here even if labelling takes longer."""
    ctx = reg.get(slug)
    stop_trial(ctx, trial_id, expected_version=body.expected_version, idempotency_key=key)
    _runners(request).stopped(slug, ctx, trial_id)
    return _trial(ctx, trial_id)


@router.post("/studies/{slug}/trials/{trial_id}/complete", response_model=TrialOut)
def complete(
    request: Request,
    slug: str,
    trial_id: str,
    body: CompleteIn,
    reg: Registry,
    key: IdempotencyKey = None,
) -> TrialOut:
    """Record the outcome of a running trial."""
    ctx = reg.get(slug)
    _runners(request).stopped(slug, ctx, trial_id)
    complete_trial(
        ctx,
        trial_id,
        stage_index=stage_index(ctx, body.stage),
        termination=body.termination,
        failure_tags=body.failure_tags,
        notes=body.notes,
        duration_s=body.duration_s,
        expected_version=body.expected_version,
        idempotency_key=key,
    )
    return _trial(ctx, trial_id)


@router.post("/studies/{slug}/trials/{trial_id}/invalidate", response_model=TrialOut)
def invalidate(
    request: Request,
    slug: str,
    trial_id: str,
    body: InvalidateIn,
    reg: Registry,
    key: IdempotencyKey = None,
) -> TrialOut:
    """Void a trial and reschedule its slot."""
    ctx = reg.get(slug)
    invalidate_trial(
        ctx,
        trial_id,
        body.reason,
        reschedule=body.reschedule,
        expected_version=body.expected_version,
        idempotency_key=key,
    )
    _runners(request).cancelled(slug, ctx, trial_id)
    return _trial(ctx, trial_id)


@router.post("/studies/{slug}/trials/{trial_id}/undo", response_model=TrialOut)
def undo(
    slug: str, trial_id: str, body: UndoIn, reg: Registry, key: IdempotencyKey = None
) -> TrialOut:
    """Undo the outcome of a session's latest trial, within 10 s; it is running again."""
    ctx = reg.get(slug)
    reopen_trial(
        ctx,
        trial_id,
        session_id=body.session_id,
        expected_version=body.expected_version,
        idempotency_key=key,
    )
    return _trial(ctx, trial_id)


@router.patch("/studies/{slug}/trials/{trial_id}", response_model=TrialOut)
def edit(
    slug: str, trial_id: str, body: EditIn, reg: Registry, key: IdempotencyKey = None
) -> TrialOut:
    """Correct a completed trial's label (logged with old and new values)."""
    ctx = reg.get(slug)
    if body.clear_stage and body.stage is not None:
        raise ServiceError("give either stage or clear_stage, not both")
    new_stage = -1 if body.clear_stage else None
    if body.stage is not None:
        new_stage = stage_index(ctx, body.stage)
    edit_trial(
        ctx,
        trial_id,
        actor=body.actor,
        reason=body.reason,
        stage_index=new_stage,
        termination=body.termination,
        failure_tags=body.failure_tags,
        notes=body.notes,
        expected_version=body.expected_version,
        idempotency_key=key,
    )
    return _trial(ctx, trial_id)


@router.post("/studies/{slug}/trials/{trial_id}/media", response_model=MediaOut, status_code=201)
def media(
    slug: str,
    trial_id: str,
    file: Annotated[UploadFile, File()],
    reg: Registry,
    actor: str = "api",
    key: IdempotencyKey = None,
) -> MediaOut:
    """Attach a clip or photo to a trial; it is stored under the study's media/ folder."""
    ctx = reg.get(slug)
    path = attach_media(
        ctx, trial_id, file.filename or "media", file.file.read(), actor=actor, idempotency_key=key
    )
    return MediaOut(path=path)


# --- live updates ------------------------------------------------------------------------------


@router.get(
    "/studies/{slug}/events",
    response_class=EventSourceResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def events(
    slug: str,
    request: Request,
    reg: Registry,
    after: str | None = None,
    once: bool = False,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> EventSourceResponse:
    """Server-Sent Events: one ``changed`` event per new event-log row.

    Starts after ``Last-Event-ID`` (or ``after``), or at the newest event. With
    ``once=true`` the stream sends what is new and closes, which suits simple polling.
    """
    ctx = await anyio.to_thread.run_sync(reg.get, slug)
    start_id = last_event_id or after or await anyio.to_thread.run_sync(latest_event_id, ctx)

    async def stream() -> AsyncIterator[ServerSentEvent]:
        last = start_id
        while True:
            new = await anyio.to_thread.run_sync(functools.partial(events_after, ctx, last))
            for event in new:
                last = event.id
                yield ServerSentEvent(
                    data=json.dumps({"kind": event.kind, "ts": event.ts.isoformat()}),
                    event="changed",
                    id=event.id,
                )
            if once or await request.is_disconnected():
                return
            await anyio.sleep(SSE_POLL_S)

    return EventSourceResponse(stream(), ping=15)
