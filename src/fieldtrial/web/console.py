"""The operator console: server-rendered pages and HTMX partials (docs/PLAN.md section 14).

Pages never show arm identities or per-arm results while a study is blinded. Every write
goes through a service with an idempotency key from the form, so a double tap or a
retried request is applied once.
"""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.datastructures import UploadFile as StarletteUpload

from fieldtrial.analysis import wording
from fieldtrial.analysis.wording import fmt_rate
from fieldtrial.report import render_html, render_markdown
from fieldtrial.services import ServiceError, StudyContext
from fieldtrial.services.analysis import analyze_study
from fieldtrial.services.events import latest_event_id
from fieldtrial.services.interim import interim_status, run_interim_in
from fieldtrial.services.registry import StudyRegistry
from fieldtrial.services.rig import check_rig, has_reference, latest_check
from fieldtrial.services.session import (
    console_state,
    end_session,
    list_sessions,
    start_session,
)
from fieldtrial.services.study import is_blinded, list_studies, status_of, unblind_study
from fieldtrial.services.trial import (
    TERMINATIONS,
    complete_trial,
    edit_trial,
    get_trial,
    invalidate_trial,
    list_trials,
    reopen_trial,
    start_trial,
    stop_trial,
)
from fieldtrial.web.runners import Runners
from fieldtrial.web.security import csrf_token

TEMPLATES = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=TEMPLATES)
router = APIRouter(include_in_schema=False)

Text = Annotated[str, Form()]
OptionalText = Annotated[str | None, Form()]


def _registry(request: Request) -> StudyRegistry:
    reg: StudyRegistry = request.app.state.registry
    return reg


def _key() -> str:
    return uuid.uuid4().hex


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}:{rest:02d}"


templates.env.filters["duration"] = _duration
templates.env.filters["rate"] = fmt_rate
templates.env.filters["time"] = lambda dt: "" if dt is None else dt.strftime("%H:%M:%S")
templates.env.filters["date"] = lambda dt: "" if dt is None else dt.strftime("%Y-%m-%d %H:%M")


def render(request: Request, name: str, status_code: int = 200, **context: Any) -> HTMLResponse:
    """Render a template with the shared context (CSRF token, LAN flag)."""
    cookies = dict(request.cookies)
    context.setdefault("title", "fieldtrial")
    context["csrf"] = csrf_token(request.scope, cookies)
    context["lan"] = request.app.state.lan
    context["htmx_headers"] = json.dumps({"X-CSRF-Token": context["csrf"]})
    context["terminations"] = TERMINATIONS
    return templates.TemplateResponse(request, name, context, status_code=status_code)


# --- runners -----------------------------------------------------------------------------------


def runners(request: Request) -> Runners:
    """The app's runners."""
    found: Runners = request.app.state.runners
    return found


# --- pages -------------------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> Response:
    """The list of studies (or the study itself when serving one folder)."""
    reg = _registry(request)
    entries = reg.entries()
    if reg.single and entries:
        return RedirectResponse(f"/studies/{entries[0].slug}", 303)
    return render(request, "studies.html", entries=entries, title="Studies · fieldtrial")


@router.get("/studies/{slug}", response_class=HTMLResponse)
def study_page(request: Request, slug: str) -> Response:
    """Design summary, progress, sessions and the form to start a session."""
    reg = _registry(request)
    folder = reg.folder(slug)
    entry = list_studies(folder)[0]
    if not entry.locked:
        return render(
            request, "draft.html", entry=entry, folder=folder, title=f"{slug} · fieldtrial"
        )
    ctx = reg.get(slug)
    report = status_of(ctx)
    return render(
        request,
        "study.html",
        rig_reference=has_reference(ctx),
        slug=slug,
        spec=ctx.spec,
        report=report,
        sessions=list_sessions(ctx),
        title=f"{ctx.spec.title or ctx.spec.name} · fieldtrial",
        idempotency_key=_key(),
    )


@router.post("/studies/{slug}/sessions")
async def create_session(request: Request, slug: str) -> Response:
    """Start a session from the study page; the browser goes to its console.

    An optional rig photo is compared with the study's reference photo; a flagged result
    shows as a warning in the console and as a deviation in the report.
    """
    ctx = _registry(request).get(slug)
    async with request.form() as form:
        checklist = ctx.spec.rubric.rig_checklist
        checks = {item: form.get(f"check_{i}") == "on" for i, item in enumerate(checklist)}
        if not all(checks.values()):
            raise ServiceError("tick every rig check before starting")
        keys = ("operator", "rig", "notes", "idempotency_key")
        fields = {k: str(form.get(k) or "") for k in keys}
        photo = form.get("rig_photo")
        photo_bytes = b""
        if isinstance(photo, StarletteUpload) and photo.filename:
            photo_bytes = await photo.read(MAX_RIG_PHOTO + 1)
            if len(photo_bytes) > MAX_RIG_PHOTO:
                raise ServiceError("the rig photo is larger than 25 MB")
    session_id = start_session(
        ctx,
        operator=fields["operator"],
        rig=fields["rig"],
        rig_check=checks,
        notes=fields["notes"] or None,
        idempotency_key=fields["idempotency_key"] or None,
    )
    if photo_bytes and has_reference(ctx):
        check_rig(ctx, photo_bytes, session_id=session_id, source="upload", actor="console")
    elif not photo_bytes:
        runners(request).camera_rig_check(slug, ctx, session_id=session_id)
    return _redirect(request, f"/studies/{slug}/sessions/{session_id}")


def _redirect(request: Request, url: str) -> Response:
    # An XHR follows a 303 by itself and HTMX never sees it, so HTMX gets a 200 with
    # HX-Redirect; a plain form post gets the 303.
    if request.headers.get("hx-request") == "true":
        return Response(status_code=200, headers={"HX-Redirect": url})
    return RedirectResponse(url, status_code=303)


MAX_RIG_PHOTO = 25 * 1024 * 1024


def _rig_alert(ctx: StudyContext, session_id: str) -> str | None:
    latest = latest_check(ctx, session_id)
    if latest is None or not latest.get("flagged"):
        return None
    reasons = "; ".join(latest.get("reasons", [])) or "the rig differs from its reference photo"
    return f"Rig check: {reasons}. Check the camera, the light and the scene before running trials."


def _panel_context(
    request: Request, slug: str, session_id: str, notice: str | None = None
) -> dict[str, Any]:
    ctx = _registry(request).get(slug)
    state = console_state(ctx, session_id)
    suggestion = None
    if state.running is not None:
        suggestion = runners(request).suggestion(state.running.trial_id)
    return {
        "slug": slug,
        "spec": ctx.spec,
        "state": state,
        "suggestion": suggestion,
        "blinded": is_blinded(ctx),
        "server_now": datetime.now(UTC).isoformat(),
        "last_event": latest_event_id(ctx) or "",
        "idempotency_key": _key(),
        "interim": interim_status(ctx),
        "notice": notice,
        "runner": runners(request).status(slug),
        "rig_alert": _rig_alert(ctx, session_id),
    }


@router.get("/studies/{slug}/sessions/{session_id}", response_class=HTMLResponse)
def console_page(request: Request, slug: str, session_id: str) -> HTMLResponse:
    """The trial console for one session."""
    context = _panel_context(request, slug, session_id)
    spec = context["spec"]
    return render(
        request,
        "console.html",
        mirror=False,
        title=f"Console · {spec.title or spec.name}",
        **context,
    )


@router.get("/studies/{slug}/sessions/{session_id}/mirror", response_class=HTMLResponse)
def mirror_page(request: Request, slug: str, session_id: str) -> HTMLResponse:
    """A read-only second screen that follows the console live."""
    context = _panel_context(request, slug, session_id)
    return render(request, "console.html", mirror=True, title="Mirror · fieldtrial", **context)


@router.get("/studies/{slug}/sessions/{session_id}/panel", response_class=HTMLResponse)
def panel(request: Request, slug: str, session_id: str, mirror: bool = False) -> HTMLResponse:
    """The console's live part (re-fetched after every change)."""
    return render(
        request, "_panel.html", mirror=mirror, **_panel_context(request, slug, session_id)
    )


@router.post("/studies/{slug}/sessions/{session_id}/start", response_class=HTMLResponse)
def start(
    request: Request,
    slug: str,
    session_id: str,
    slot_id: Text,
    idempotency_key: OptionalText = None,
) -> HTMLResponse:
    """Start the next trial."""
    ctx = _registry(request).get(slug)
    view = start_trial(ctx, slot_id, session_id, idempotency_key=idempotency_key)
    runners(request).started(slug, ctx, get_trial(ctx, view.trial_id))
    return panel(request, slug, session_id)


@router.post(
    "/studies/{slug}/sessions/{session_id}/trials/{trial_id}/stop", response_class=HTMLResponse
)
def stop(
    request: Request,
    slug: str,
    session_id: str,
    trial_id: str,
    version: Annotated[int, Form()],
    idempotency_key: OptionalText = None,
) -> HTMLResponse:
    """Stop the clock; the label form appears."""
    ctx = _registry(request).get(slug)
    stop_trial(ctx, trial_id, expected_version=version, idempotency_key=idempotency_key)
    runners(request).stopped(slug, ctx, trial_id)
    return panel(request, slug, session_id)


@router.post(
    "/studies/{slug}/sessions/{session_id}/trials/{trial_id}/complete",
    response_class=HTMLResponse,
)
def complete(
    request: Request,
    slug: str,
    session_id: str,
    trial_id: str,
    stage: Text,
    termination: Text,
    version: Annotated[int, Form()],
    failure_tags: Annotated[list[str] | None, Form()] = None,
    notes: OptionalText = None,
    idempotency_key: OptionalText = None,
) -> HTMLResponse:
    """Record the label the operator chose (the duration was fixed when it was stopped)."""
    ctx = _registry(request).get(slug)
    complete_trial(
        ctx,
        trial_id,
        stage_index=int(stage),
        termination=termination,
        failure_tags=failure_tags or [],
        notes=notes or None,
        expected_version=version,
        idempotency_key=idempotency_key,
    )
    runners(request).forget(trial_id)
    return panel(request, slug, session_id)


@router.post(
    "/studies/{slug}/sessions/{session_id}/trials/{trial_id}/invalidate",
    response_class=HTMLResponse,
)
def invalidate(
    request: Request,
    slug: str,
    session_id: str,
    trial_id: str,
    reason: Text,
    version: Annotated[int, Form()],
    idempotency_key: OptionalText = None,
) -> HTMLResponse:
    """Void the running trial; its slot is rescheduled at the end of the block."""
    ctx = _registry(request).get(slug)
    invalidate_trial(
        ctx, trial_id, reason, expected_version=version, idempotency_key=idempotency_key
    )
    runners(request).cancelled(slug, ctx, trial_id)
    return panel(request, slug, session_id)


@router.post(
    "/studies/{slug}/sessions/{session_id}/trials/{trial_id}/undo", response_class=HTMLResponse
)
def undo(
    request: Request,
    slug: str,
    session_id: str,
    trial_id: str,
    idempotency_key: OptionalText = None,
) -> HTMLResponse:
    """Undo the last trial's label (within 10 s)."""
    ctx = _registry(request).get(slug)
    reopen_trial(ctx, trial_id, session_id=session_id, idempotency_key=idempotency_key)
    return panel(request, slug, session_id)


@router.post("/studies/{slug}/sessions/{session_id}/interim", response_class=HTMLResponse)
def interim(request: Request, slug: str, session_id: str) -> HTMLResponse:
    """Run the interim look that is due. Shows only "continue" or "stop"."""
    ctx = _registry(request).get(slug)
    result = run_interim_in(ctx, actor="console")
    notice = wording.interim(result.decision, result.look, result.planned_looks)
    return render(
        request,
        "_panel.html",
        mirror=False,
        **_panel_context(request, slug, session_id, notice=notice),
    )


@router.post("/studies/{slug}/sessions/{session_id}/end")
def finish(
    request: Request, slug: str, session_id: str, idempotency_key: OptionalText = None
) -> Response:
    """End the session and go back to the study page."""
    ctx = _registry(request).get(slug)
    end_session(ctx, session_id, idempotency_key=idempotency_key)
    return _redirect(request, f"/studies/{slug}")


# --- history -----------------------------------------------------------------------------------


@router.get("/studies/{slug}/history", response_class=HTMLResponse)
def history(request: Request, slug: str) -> HTMLResponse:
    """Every trial, newest first, with corrections."""
    ctx = _registry(request).get(slug)
    return render(
        request,
        "history.html",
        slug=slug,
        spec=ctx.spec,
        trials=list_trials(ctx),
        blinded=is_blinded(ctx),
        title=f"History · {ctx.spec.title or ctx.spec.name}",
    )


@router.get("/studies/{slug}/trials/{trial_id}/edit", response_class=HTMLResponse)
def edit_form(request: Request, slug: str, trial_id: str) -> HTMLResponse:
    """The correction form for one trial (a table row)."""
    ctx = _registry(request).get(slug)
    return render(
        request,
        "_edit_row.html",
        slug=slug,
        spec=ctx.spec,
        trial=get_trial(ctx, trial_id),
        blinded=is_blinded(ctx),
        idempotency_key=_key(),
    )


@router.get("/studies/{slug}/trials/{trial_id}/row", response_class=HTMLResponse)
def trial_row(request: Request, slug: str, trial_id: str) -> HTMLResponse:
    """One history row (to cancel an edit)."""
    ctx = _registry(request).get(slug)
    return render(
        request,
        "_trial_row.html",
        slug=slug,
        spec=ctx.spec,
        trial=get_trial(ctx, trial_id),
        blinded=is_blinded(ctx),
    )


@router.post("/studies/{slug}/trials/{trial_id}/edit", response_class=HTMLResponse)
def save_edit(
    request: Request,
    slug: str,
    trial_id: str,
    actor: Text,
    reason: Text,
    stage: Text,
    termination: Text,
    version: Annotated[int, Form()],
    failure_tags: Annotated[list[str] | None, Form()] = None,
    notes: OptionalText = None,
    idempotency_key: OptionalText = None,
) -> HTMLResponse:
    """Save a correction; the old and new labels go to the event log."""
    ctx = _registry(request).get(slug)
    edit_trial(
        ctx,
        trial_id,
        actor=actor,
        reason=reason,
        stage_index=int(stage),
        termination=termination,
        failure_tags=failure_tags or [],
        notes=notes or "",
        expected_version=version,
        idempotency_key=idempotency_key,
    )
    return trial_row(request, slug, trial_id)


# --- report ------------------------------------------------------------------------------------


@router.get("/studies/{slug}/report", response_class=HTMLResponse)
def report(request: Request, slug: str) -> HTMLResponse:
    """The analysis, or the unblinding step while blinded."""
    ctx = _registry(request).get(slug)
    title = f"Report · {ctx.spec.title or ctx.spec.name}"
    if is_blinded(ctx):
        return render(
            request,
            "report.html",
            slug=slug,
            spec=ctx.spec,
            report=status_of(ctx),
            results=None,
            title=title,
            idempotency_key=_key(),
        )
    return render(
        request,
        "report.html",
        slug=slug,
        spec=ctx.spec,
        report=status_of(ctx),
        results=analyze_study(ctx.folder),
        title=title,
    )


@router.post("/studies/{slug}/unblind")
def unblind(request: Request, slug: str, confirm: OptionalText = None) -> Response:
    """Reveal the arms (logged). Needs the confirmation box ticked."""
    if confirm != "yes":
        raise ServiceError("tick the box to confirm that you want to unblind")
    ctx = _registry(request).get(slug)
    unblind_study(ctx.folder, actor="console")
    return _redirect(request, f"/studies/{slug}/report")


@router.get("/studies/{slug}/report.md", response_class=PlainTextResponse)
def report_markdown(request: Request, slug: str) -> Response:
    """The Markdown report as a download."""
    ctx = _registry(request).get(slug)
    text = render_markdown(analyze_study(ctx.folder))
    return PlainTextResponse(
        text,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{slug}-report.md"'},
    )


REPORT_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


@router.get("/studies/{slug}/report.html", response_class=HTMLResponse)
def report_html(request: Request, slug: str, download: bool = False) -> Response:
    """The self-contained HTML report (no scripts; inline styles and charts only)."""
    ctx = _registry(request).get(slug)
    headers = {"Content-Security-Policy": REPORT_CSP}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{slug}-report.html"'
    return HTMLResponse(render_html(analyze_study(ctx.folder)), headers=headers)


@router.get("/studies/{slug}/results.json")
def results_json(request: Request, slug: str) -> Response:
    """The Results model as a download."""
    ctx = _registry(request).get(slug)
    return Response(
        analyze_study(ctx.folder).model_dump_json(indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{slug}-results.json"'},
    )


def error_page(request: Request, message: str, status_code: int) -> HTMLResponse:
    """An error for the console: a flash message for HTMX requests, else a page."""
    if request.headers.get("hx-request") == "true":
        response = render(request, "_flash.html", message=message, status_code=status_code)
        response.headers["HX-Retarget"] = "#flash"
        response.headers["HX-Reswap"] = "innerHTML"
        return response
    return render(request, "error.html", message=message, status_code=status_code, title="Error")
