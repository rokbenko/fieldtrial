"""Simulated runner: synthetic, seeded outcomes for tests, docs and ``fieldtrial demo``."""

import math
from dataclasses import dataclass

from fieldtrial.runners.base import (
    ArmSpec,
    RunArtifacts,
    RunnerCapabilities,
    RunnerStatus,
    Termination,
    TrialContext,
)
from fieldtrial.stats._rng import STREAM_SIMULATION, StableRng


@dataclass(frozen=True, slots=True)
class SimArm:
    """True behavior of a simulated arm.

    ``success_rate`` is the probability of reaching the success stage. Failed trials stop at a
    stage drawn uniformly below the success stage. Successful trials take a log-normal time
    with median ``median_success_s``; failures end at the timeout (or after
    ``median_failure_s`` when there is none). ``invalid_rate`` is the probability that a trial
    is voided (simulated robot fault).
    """

    success_rate: float
    median_success_s: float = 12.0
    median_failure_s: float = 20.0
    invalid_rate: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success_rate must be between 0 and 1")
        if not 0.0 <= self.invalid_rate < 0.5:
            raise ValueError("invalid_rate must be in [0, 0.5)")
        if self.median_success_s <= 0 or self.median_failure_s <= 0:
            raise ValueError("median durations must be positive")


class SimRunner:
    """A runner that draws outcomes instead of moving a robot. Deterministic for a given seed."""

    capabilities = RunnerCapabilities(
        can_switch_arms=True, can_stop=True, records_media=False, reports_outcome=True
    )

    def __init__(
        self,
        arms: dict[str, SimArm],
        *,
        n_stages: int,
        success_index: int,
        failure_tags: tuple[str, ...] = (),
        seed: int = 0,
    ) -> None:
        if not 0 <= success_index < n_stages:
            raise ValueError("success_index must be a stage index")
        self._arms = arms
        self._n_stages = n_stages
        self._success_index = success_index
        self._failure_tags = failure_tags
        self._rng = StableRng(seed, STREAM_SIMULATION)
        self._arm: ArmSpec | None = None
        self._trial: TrialContext | None = None
        self._state: RunnerStatus = RunnerStatus("idle")

    def prepare(self, arm: ArmSpec) -> None:
        """Switch to an arm (instant)."""
        if arm.arm_id not in self._arms:
            raise ValueError(f"no simulated behavior for arm {arm.arm_id!r}")
        self._arm = arm
        self._state = RunnerStatus("ready", f"arm {arm.blind_code}")

    def start(self, trial: TrialContext) -> None:
        """Begin a trial."""
        if self._arm is None:
            raise RuntimeError("call prepare() first")
        self._trial = trial
        self._state = RunnerStatus("running")

    def _lognormal(self, median: float, sigma: float = 0.35) -> float:
        u1 = max(self._rng.uniform(), 1e-300)
        u2 = self._rng.uniform()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return median * math.exp(sigma * z)

    def stop(self, reason: Termination = "other") -> RunArtifacts:
        """End the trial and report its simulated outcome."""
        if self._arm is None or self._trial is None:
            raise RuntimeError("no trial is running")
        behavior = self._arms[self._arm.arm_id]
        timeout = self._trial.timeout_s
        self._trial = None
        self._state = RunnerStatus("ready")
        if self._rng.uniform() < behavior.invalid_rate:
            return RunArtifacts(
                duration_s=round(self._lognormal(5.0), 2),
                termination="robot_fault",
                invalid_reason="simulated robot fault",
            )
        if self._rng.uniform() < behavior.success_rate:
            top = self._n_stages - 1
            stage = self._success_index + self._rng.below(top - self._success_index + 1)
            duration = self._lognormal(behavior.median_success_s)
            if timeout is not None:
                duration = min(duration, timeout * 0.95)
            return RunArtifacts(
                duration_s=round(duration, 2), termination="success", stage_index=stage
            )
        stage = self._rng.below(self._success_index + 1) - 1  # -1 .. success_index - 1
        tags: tuple[str, ...] = ()
        if self._failure_tags:
            tags = (self._failure_tags[self._rng.below(len(self._failure_tags))],)
        if timeout is not None:
            return RunArtifacts(
                duration_s=timeout, termination="timeout", stage_index=stage, failure_tags=tags
            )
        return RunArtifacts(
            duration_s=round(self._lognormal(behavior.median_failure_s), 2),
            termination="stuck",
            stage_index=stage,
            failure_tags=tags,
        )

    def status(self) -> RunnerStatus:
        """Current state."""
        return self._state

    def close(self) -> None:
        """Nothing to release."""
        self._state = RunnerStatus("closed")
