"""Blind review of reward-model-scored episodes in the console (docs/guides/reward-models.md).

The reviewer sees the episode's video and labels it first; only then does the page show the
model's suggestion, and the reviewer may change the label with a reason. Nothing on the page
says which arm ran the episode.
"""

from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from fieldtrial.services import ServiceError
from fieldtrial.services.rewards import (
    next_review,
    record_review,
    review_progress,
    review_state,
    revise_review,
)
from fieldtrial.web.console import _redirect, _registry, render

router = APIRouter(include_in_schema=False)

Text = Annotated[str, Form()]


def _state_url(slug: str, root: str, episode: int) -> str:
    return f"/studies/{slug}/review?root={quote(root, safe='')}&episode={episode}"


@router.get("/studies/{slug}/review", response_class=HTMLResponse)
def review_page(
    request: Request, slug: str, root: str | None = None, episode: int | None = None
) -> HTMLResponse:
    """The next episode to label, or a labelled episode with the model's suggestion."""
    ctx = _registry(request).get(slug)
    done, total = review_progress(ctx)
    state = None
    if root is not None and episode is not None:
        state = review_state(ctx, root, episode)
    item = None if state is not None else next_review(ctx)
    return render(
        request,
        "review.html",
        slug=slug,
        spec=ctx.spec,
        done=done,
        total=total,
        item=item,
        state=state,
        reviewer=request.cookies.get("fieldtrial_reviewer", ""),
        title=f"Review · {ctx.spec.title or ctx.spec.name}",
    )


@router.post("/studies/{slug}/review/label")
def label(
    request: Request,
    slug: str,
    root: Text,
    episode_index: Annotated[int, Form()],
    success: Text,
    reviewer: Text,
) -> Response:
    """Record the blind first label, then show the suggestion."""
    if not reviewer.strip():
        raise ServiceError("enter your name before labelling")
    ctx = _registry(request).get(slug)
    record_review(ctx, root, episode_index, success == "1", reviewer=reviewer.strip())
    response = _redirect(request, _state_url(slug, root, episode_index))
    response.set_cookie("fieldtrial_reviewer", reviewer.strip(), samesite="strict")
    return response


@router.post("/studies/{slug}/review/revise")
def revise(
    request: Request,
    slug: str,
    root: Text,
    episode_index: Annotated[int, Form()],
    success: Text,
    reason: Text,
    reviewer: Text,
) -> Response:
    """Change a label after seeing the suggestion (logged with the reason)."""
    ctx = _registry(request).get(slug)
    revise_review(
        ctx,
        root,
        episode_index,
        success == "1",
        reason=reason,
        reviewer=reviewer.strip() or "console",
    )
    return _redirect(request, _state_url(slug, root, episode_index))


@router.get("/studies/{slug}/review/video")
def video(request: Request, slug: str, root: str, episode: int) -> Response:
    """The video file of the next episode to review (the page seeks to the episode's span)."""
    ctx = _registry(request).get(slug)
    item = next_review(ctx)
    if item is None or item.video is None or (item.root, item.episode_index) != (root, episode):
        raise ServiceError("only the episode under review can be played")
    return FileResponse(item.video.path, media_type="video/mp4")
