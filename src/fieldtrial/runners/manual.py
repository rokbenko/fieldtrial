"""Manual runner: no robot control. The operator runs the policy and labels the outcome."""

from datetime import UTC, datetime

from fieldtrial.runners.base import (
    ArmSpec,
    RunArtifacts,
    RunnerCapabilities,
    RunnerStatus,
    Termination,
    TrialContext,
)


class ManualRunner:
    """The console shows which arm to load (by blind code); the operator does the rest.

    It cannot switch arms or judge outcomes, so it reports only the elapsed time.
    """

    capabilities = RunnerCapabilities()

    def __init__(self) -> None:
        self._arm: ArmSpec | None = None
        self._started: datetime | None = None
        self._state = RunnerStatus("idle")

    def prepare(self, arm: ArmSpec) -> None:
        """Ask the operator to load ``arm`` (shown by its blind code)."""
        self._arm = arm
        self._state = RunnerStatus("ready", f"load arm {arm.blind_code}")

    def start(self, trial: TrialContext) -> None:
        """Note the start time."""
        if self._arm is None:
            raise RuntimeError("call prepare() first")
        self._started = datetime.now(UTC)
        self._state = RunnerStatus("running", f"trial {trial.seq}")

    def stop(self, reason: Termination = "other") -> RunArtifacts:
        """Return the elapsed time; the operator labels the outcome."""
        if self._started is None:
            raise RuntimeError("no trial is running")
        elapsed = (datetime.now(UTC) - self._started).total_seconds()
        self._started = None
        self._state = RunnerStatus("ready")
        return RunArtifacts(duration_s=elapsed, termination=reason)

    def status(self) -> RunnerStatus:
        """Current state."""
        return self._state

    def close(self) -> None:
        """Nothing to release."""
        self._state = RunnerStatus("closed")
