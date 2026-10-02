"""Rig drift checks: compare photos of the rig with a reference photo.

The reference lives at ``rig/reference.png`` in the study folder; each check keeps its
photo under ``rig/checks/`` and is recorded as a ``rig_check`` event. Flagged checks are
listed as deviations in every report.
"""

import hashlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fieldtrial.capture.drift import DriftResult, DriftThresholds, compare
from fieldtrial.capture.images import load_image, save_png
from fieldtrial.services._context import ServiceError, StudyContext
from fieldtrial.services.events import append_event, list_events

REFERENCE = Path("rig") / "reference.png"
CHECKS = Path("rig") / "checks"
Source = Literal["upload", "camera", "cli"]


@dataclass(frozen=True, slots=True)
class RigCheck:
    """One check of the rig against the reference photo."""

    session_id: str | None
    source: str
    image: str
    drift: DriftResult

    @property
    def flagged(self) -> bool:
        """Whether any threshold was crossed."""
        return self.drift.flagged


def reference_path(ctx: StudyContext) -> Path:
    """Where the study's reference photo is (it may not exist yet)."""
    return Path(ctx.folder) / REFERENCE


def has_reference(ctx: StudyContext) -> bool:
    """Whether a reference photo has been set."""
    return reference_path(ctx).exists()


def _thresholds(ctx: StudyContext) -> DriftThresholds:
    capture = ctx.spec.capture
    if capture is None:
        return DriftThresholds()
    d = capture.drift
    return DriftThresholds(
        max_shift_px=d.max_shift_px,
        max_brightness_change=d.max_brightness_change,
        min_similarity=d.min_similarity,
    )


def _decode(image: bytes | str | Path) -> Any:
    try:
        return load_image(image)
    except ValueError as exc:
        raise ServiceError(str(exc)) from exc


def set_reference(
    ctx: StudyContext, image: bytes | str | Path, *, actor: str = "fieldtrial"
) -> Path:
    """Store the reference photo (the previous one is kept with a timestamp)."""
    pixels = _decode(image)
    target = reference_path(ctx)
    if target.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        target.rename(target.with_name(f"reference-{stamp}.png"))
    save_png(pixels, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    with ctx.db() as db, db.begin():
        append_event(
            db,
            ctx.study_id,
            "rig_reference",
            actor,
            {
                "path": REFERENCE.as_posix(),
                "sha256": digest,
                "width": int(pixels.shape[1]),
                "height": int(pixels.shape[0]),
            },
        )
    return target


def check_rig(
    ctx: StudyContext,
    image: Any,
    *,
    session_id: str | None = None,
    source: Source = "upload",
    actor: str = "fieldtrial",
) -> RigCheck:
    """Compare a photo (file, bytes or RGB array) with the reference and record the result."""
    if not has_reference(ctx):
        raise ServiceError(
            "this study has no rig reference photo yet; set one with "
            f"`fieldtrial rig-check {ctx.folder} PHOTO --set-reference`"
        )
    pixels = image if hasattr(image, "shape") else _decode(image)
    reference = load_image(reference_path(ctx))
    try:
        result = compare(reference, pixels, _thresholds(ctx))
    except ValueError as exc:
        raise ServiceError(f"cannot compare the photos: {exc}") from exc
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    relative = CHECKS / f"{stamp}.png"
    save_png(pixels, Path(ctx.folder) / relative)
    check = RigCheck(session_id=session_id, source=source, image=relative.as_posix(), drift=result)
    with ctx.db() as db, db.begin():
        append_event(
            db,
            ctx.study_id,
            "rig_check",
            actor,
            {
                "session_id": session_id,
                "source": source,
                "image": check.image,
                **{k: v for k, v in asdict(result).items() if k != "reasons"},
                "reasons": list(result.reasons),
            },
        )
    return check


def rig_checks(ctx: StudyContext) -> list[dict[str, Any]]:
    """Every recorded rig check, oldest first, with its time."""
    with ctx.db() as db:
        return [{"ts": e.ts, **dict(e.payload)} for e in list_events(db, ctx.study_id, "rig_check")]


def latest_check(ctx: StudyContext, session_id: str) -> dict[str, Any] | None:
    """The latest rig check of a session, if any."""
    found = [c for c in rig_checks(ctx) if c.get("session_id") == session_id]
    return found[-1] if found else None
