"""The server's runners: one per study, driven by trial starts and stops.

The console and the REST API both call :class:`Runners` when a trial starts, stops or is
voided, so a study whose arms use the ``command`` or ``openpi_router`` runner runs the
policy itself, whichever client the operator uses. Simulated arms produce a suggested
outcome at once; switching runners report one when they can judge the outcome (a
command's exit code).

A runner that fails to start a trial voids it with the reason, so the slot is
rescheduled and the operator sees what went wrong.
"""

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from fieldtrial.capture.recorder import CaptureError, FrameSource, TrialRecorder
from fieldtrial.design.capture_config import CaptureConfig
from fieldtrial.runners.base import (
    ArmSpec,
    RunArtifacts,
    Runner,
    RunnerError,
    RunnerStatus,
    TrialContext,
)
from fieldtrial.runners.manual import ManualRunner
from fieldtrial.runners.sim import SimArm, SimRunner
from fieldtrial.services import ServiceError, StudyContext
from fieldtrial.services.registry import StudyRegistry
from fieldtrial.services.rig import RigCheck, check_rig, has_reference
from fieldtrial.services.trial import (
    TrialDetail,
    attach_media,
    get_trial,
    invalidate_trial,
    record_runner_output,
)

log = logging.getLogger("fieldtrial.runners")
SWITCHING = ("command", "openpi_router")


def runner_kind(ctx: StudyContext) -> str:
    """``command``, ``openpi_router``, ``sim`` or ``manual``."""
    kinds = {a.runner for a in ctx.spec.arms}
    for kind in (*SWITCHING, "sim"):
        if kind in kinds:
            return kind
    return "manual"


def build_runner(ctx: StudyContext) -> Runner:
    """The runner a study's arms ask for (an openpi router starts listening at once)."""
    spec = ctx.spec
    kind = runner_kind(ctx)
    if kind == "command":
        from fieldtrial.runners.command import CommandRunner

        assert spec.runners is not None
        assert spec.runners.command is not None
        return CommandRunner(
            spec.runners.command,
            study=spec.name,
            folder=Path(ctx.folder),
            success_index=spec.rubric.success_index,
        )
    if kind == "openpi_router":
        from fieldtrial.runners.openpi_router import OpenpiRouter

        config = spec.runners.openpi_router if spec.runners else None
        router = OpenpiRouter(
            {a.id: str(a.policy["url"]) for a in spec.arms},
            listen=config.listen if config else "127.0.0.1:8000",
            metadata=config.metadata if config else "identical",
        )
        router.serve()
        return router
    if kind == "sim":
        return SimRunner(
            {
                a.id: SimArm(float(a.policy.get("sim_success_rate", 0.5)))
                for a in spec.arms
                if a.runner == "sim"
            },
            n_stages=len(spec.rubric.stages),
            success_index=spec.rubric.success_index,
            failure_tags=tuple(spec.rubric.failure_tags),
            seed=spec.design.seed,
        )
    return ManualRunner()


def _open_camera(config: CaptureConfig) -> FrameSource:
    from fieldtrial.capture.recorder import CameraSource

    assert config.camera is not None
    return CameraSource(config.camera, width=config.width, height=config.height, fps=config.fps)


class Runners:
    """One runner per study, and the outcome a runner suggested for each running trial.

    With ``capture.camera`` in ``study.yaml``, the evaluation camera is recorded from Start
    to Stop and the clip attached to the trial; ``capture.drift.every_trials`` also checks
    the rig from the camera. ``open_camera`` replaces the OpenCV camera (for tests).
    """

    def __init__(self, open_camera: Callable[[CaptureConfig], FrameSource] | None = None) -> None:
        self._runners: dict[str, Runner] = {}
        self._running: dict[str, str] = {}  # slug -> trial id the runner is running
        self._suggested: dict[str, RunArtifacts] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.RLock()
        self._recorders: dict[str, TrialRecorder] = {}
        self._recording: dict[str, str] = {}  # slug -> trial id being recorded
        self._camera_errors: dict[str, str] = {}
        self._open_camera = open_camera or _open_camera

    def _runner(self, slug: str, ctx: StudyContext) -> Runner:
        if slug not in self._runners:
            self._runners[slug] = build_runner(ctx)
        return self._runners[slug]

    def warm(self, registry: StudyRegistry) -> None:
        """Start the openpi routers of the served studies, so the robot can connect early."""
        for entry in registry.entries():
            if not entry.locked:
                continue
            try:
                ctx = registry.get(entry.slug)
                if runner_kind(ctx) == "openpi_router":
                    with self._lock:
                        self._runner(entry.slug, ctx)
            except (RunnerError, ServiceError) as exc:
                self._errors[entry.slug] = str(exc)
                log.warning("runner for %s: %s", entry.slug, exc)

    @staticmethod
    def _context(ctx: StudyContext, trial: TrialDetail) -> TrialContext:
        return TrialContext(
            seq=trial.slot.seq,
            condition=trial.slot.condition,
            factors=trial.slot.factors,
            instruction=ctx.spec.task.instruction,
            timeout_s=ctx.spec.limits.timeout_s,
            trial_id=trial.trial_id,
        )

    def started(self, slug: str, ctx: StudyContext, trial: TrialDetail) -> None:
        """Run the trial's arm and start the camera.

        A runner that cannot start voids the trial and raises; a camera that fails is
        reported but never blocks the trial. A repeated call for the same trial (a retried
        request) does nothing.
        """
        if trial.status != "running" or self._recording.get(slug) == trial.trial_id:
            return
        self._start_runner(slug, ctx, trial)
        self._start_camera(slug, ctx, trial)

    def _start_runner(self, slug: str, ctx: StudyContext, trial: TrialDetail) -> None:
        if self._running.get(slug) == trial.trial_id:
            return
        arm = ctx.spec.arm(trial.slot.arm)
        spec_arm = ArmSpec(arm.id, trial.slot.blind_code, dict(arm.policy), dict(arm.serving))
        kind = runner_kind(ctx)
        with self._lock:
            if kind == "sim":
                if arm.runner != "sim" or trial.trial_id in self._suggested:
                    return
                runner = self._runner(slug, ctx)
                runner.prepare(spec_arm)
                runner.start(self._context(ctx, trial))
                self._suggested[trial.trial_id] = runner.stop("other")
                return
            if kind not in SWITCHING:
                return
            try:
                runner = self._runner(slug, ctx)
                runner.prepare(spec_arm)
                runner.start(self._context(ctx, trial))
            except RunnerError as exc:
                self._errors[slug] = str(exc)
                current = get_trial(ctx, trial.trial_id)
                invalidate_trial(
                    ctx,
                    trial.trial_id,
                    f"runner could not start: {exc}",
                    expected_version=current.version,
                    actor="runner",
                )
                raise ServiceError(
                    f"the runner could not start trial {trial.slot.seq}: {exc}. The trial "
                    "was marked invalid and rescheduled."
                ) from exc
            self._errors.pop(slug, None)
            self._running[slug] = trial.trial_id

    def stopped(self, slug: str, ctx: StudyContext, trial_id: str) -> None:
        """Stop the runner and the camera; keep the suggestion, record output and clip."""
        self._stop_camera(slug, ctx, trial_id)
        self._stop_runner(slug, ctx, trial_id)

    def _stop_runner(self, slug: str, ctx: StudyContext, trial_id: str) -> None:
        with self._lock:
            if self._running.get(slug) != trial_id:
                return
            runner = self._runners[slug]
            try:
                artifacts = runner.stop("operator_stop")
            except RunnerError as exc:
                self._errors[slug] = str(exc)
                return
            finally:
                self._running.pop(slug, None)
            if artifacts.stage_index is not None:
                self._suggested[trial_id] = artifacts
        if artifacts.metrics or artifacts.log:
            record_runner_output(ctx, trial_id, metrics=artifacts.metrics, log=artifacts.log)

    # --- camera ---------------------------------------------------------------------------

    def _camera(self, slug: str, ctx: StudyContext) -> TrialRecorder | None:
        config = ctx.spec.capture
        if config is None or config.camera is None:
            return None
        with self._lock:
            if slug not in self._recorders:
                self._recorders[slug] = TrialRecorder(
                    lambda: self._open_camera(config), fps=config.fps
                )
            return self._recorders[slug]

    def _camera_failed(self, slug: str, what: str, exc: Exception) -> None:
        self._camera_errors[slug] = f"camera: {what}: {exc}"
        log.warning("camera for %s: %s: %s", slug, what, exc)

    def _start_camera(self, slug: str, ctx: StudyContext, trial: TrialDetail) -> None:
        recorder = self._camera(slug, ctx)
        if recorder is None:
            return
        config = ctx.spec.capture
        assert config is not None
        every = config.drift.every_trials
        if every and trial.slot.seq % every == 0 and has_reference(ctx):
            self.camera_rig_check(slug, ctx, session_id=trial.session_id)
        if not config.record:
            return
        try:
            recorder.start(Path(ctx.folder) / "media" / ".recording" / f"{trial.trial_id}.mp4")
        except (CaptureError, OSError) as exc:
            self._camera_failed(slug, "recording did not start", exc)
            return
        self._camera_errors.pop(slug, None)
        self._recording[slug] = trial.trial_id

    def _stop_camera(self, slug: str, ctx: StudyContext, trial_id: str) -> None:
        if self._recording.get(slug) != trial_id:
            return
        self._recording.pop(slug, None)
        recorder = self._recorders[slug]
        try:
            clip = recorder.stop()
        except CaptureError as exc:
            self._camera_failed(slug, "recording failed", exc)
            return
        if clip is None:
            self._camera_failed(slug, "no frames", CaptureError("the camera gave no frame"))
            return
        try:
            attach_media(ctx, trial_id, "camera.mp4", clip.read_bytes(), actor="camera")
        except ServiceError as exc:
            self._camera_failed(slug, f"clip kept at {clip}", exc)
            return
        clip.unlink(missing_ok=True)

    def camera_rig_check(
        self, slug: str, ctx: StudyContext, *, session_id: str | None = None
    ) -> RigCheck | None:
        """Check the rig from the camera, if the study has a camera and a reference photo."""
        recorder = self._camera(slug, ctx)
        if recorder is None or not has_reference(ctx):
            return None
        try:
            frame = recorder.grab()
            return check_rig(ctx, frame, session_id=session_id, source="camera", actor="camera")
        except (CaptureError, ServiceError) as exc:
            self._camera_failed(slug, "rig check failed", exc)
            return None

    def cancelled(self, slug: str, ctx: StudyContext, trial_id: str) -> None:
        """A running trial was voided: stop the runner too."""
        self.stopped(slug, ctx, trial_id)
        self.forget(trial_id)

    def status(self, slug: str) -> RunnerStatus | None:
        """The runner's state for the console, or None for manual and simulated studies."""
        with self._lock:
            runner = self._runners.get(slug)
            error = "; ".join(
                e for e in (self._errors.get(slug), self._camera_errors.get(slug)) if e
            )
            camera = slug in self._recording
        note = "camera recording" if camera else ""
        if runner is None or isinstance(runner, ManualRunner | SimRunner):
            message = "; ".join(x for x in (note, error) if x)
            return RunnerStatus("running" if camera else "idle", message) if message else None
        found = runner.status()
        extra = "; ".join(x for x in (note, error) if x and not found.message.endswith(x))
        if extra:
            return RunnerStatus(found.state, f"{found.message}; {extra}".strip("; "))
        return found

    def suggestion(self, trial_id: str) -> RunArtifacts | None:
        """The runner's suggested outcome for a trial."""
        return self._suggested.get(trial_id)

    def forget(self, trial_id: str) -> None:
        """Drop a trial's suggestion once it is labelled."""
        self._suggested.pop(trial_id, None)

    def close(self) -> None:
        """Stop every runner (processes, routers)."""
        with self._lock:
            for runner in self._runners.values():
                try:
                    runner.close()
                except Exception as exc:
                    log.warning("closing a runner failed: %s", exc)
            self._runners.clear()
            self._running.clear()
            for recorder in self._recorders.values():
                try:
                    recorder.close()
                except Exception as exc:
                    log.warning("closing a camera failed: %s", exc)
            self._recorders.clear()
            self._recording.clear()
