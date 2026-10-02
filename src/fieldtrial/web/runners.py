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
from pathlib import Path

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
from fieldtrial.services.trial import (
    TrialDetail,
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


class Runners:
    """One runner per study, and the outcome a runner suggested for each running trial."""

    def __init__(self) -> None:
        self._runners: dict[str, Runner] = {}
        self._running: dict[str, str] = {}  # slug -> trial id the runner is running
        self._suggested: dict[str, RunArtifacts] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.RLock()

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
        """Run the trial's arm. A runner that cannot start voids the trial and raises.

        A repeated call for the same trial (a retried request) does nothing.
        """
        if trial.status != "running" or self._running.get(slug) == trial.trial_id:
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
        """Stop the runner's trial; keep its suggested outcome and record its output."""
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

    def cancelled(self, slug: str, ctx: StudyContext, trial_id: str) -> None:
        """A running trial was voided: stop the runner too."""
        self.stopped(slug, ctx, trial_id)
        self.forget(trial_id)

    def status(self, slug: str) -> RunnerStatus | None:
        """The runner's state for the console, or None for manual and simulated studies."""
        with self._lock:
            runner = self._runners.get(slug)
            error = self._errors.get(slug)
        if runner is None or isinstance(runner, ManualRunner | SimRunner):
            return RunnerStatus("idle", error) if error else None
        found = runner.status()
        if error and not found.message.endswith(error):
            return RunnerStatus(found.state, f"{found.message}; {error}".strip("; "))
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
