"""The runner protocol (docs/PLAN.md section 13).

A runner executes one trial for an arm. fieldtrial schedules and records; runners only run.
Optional dependencies (lerobot, openpi, ...) are imported lazily inside adapters.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Termination = Literal[
    "success", "timeout", "stuck", "operator_stop", "robot_fault", "safety_stop", "other"
]


@dataclass(frozen=True, slots=True)
class RunnerCapabilities:
    """What a runner can do on its own."""

    can_switch_arms: bool = False
    can_stop: bool = False
    records_media: bool = False
    reports_outcome: bool = False


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """The arm a runner should load. ``blind_code`` is what an operator may see."""

    arm_id: str
    blind_code: str
    policy: dict[str, Any] = field(default_factory=dict)
    serving: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrialContext:
    """The trial about to run."""

    seq: int
    condition: str
    factors: dict[str, Any]
    instruction: str | None = None
    timeout_s: float | None = None


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """What a runner reports when a trial stops.

    Runners that cannot judge the outcome leave ``stage_index`` as None; the operator then
    labels it. ``invalid_reason`` marks a trial that must be voided and rescheduled.
    """

    duration_s: float
    termination: Termination
    stage_index: int | None = None
    failure_tags: tuple[str, ...] = ()
    media: tuple[str, ...] = ()
    invalid_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RunnerStatus:
    """Current state, shown in the console (for example "loading" while a policy loads)."""

    state: Literal["idle", "loading", "ready", "running", "closed"]
    message: str = ""


@runtime_checkable
class Runner(Protocol):
    """Runs trials for one arm at a time."""

    capabilities: RunnerCapabilities

    def prepare(self, arm: ArmSpec) -> None:
        """Load or switch to the arm's policy. May block."""

    def start(self, trial: TrialContext) -> None:
        """Start a trial."""

    def stop(self, reason: Termination) -> RunArtifacts:
        """Stop the running trial and report what happened."""

    def status(self) -> RunnerStatus:
        """Current state."""

    def close(self) -> None:
        """Release resources."""
