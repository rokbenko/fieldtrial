"""The auto-operator: fill a locked study with simulated trials."""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from fieldtrial.runners.base import ArmSpec, TrialContext
from fieldtrial.runners.sim import SimArm, SimRunner
from fieldtrial.services._context import ServiceError, open_study
from fieldtrial.services.session import end_session, start_session
from fieldtrial.services.trial import next_slot, record_trial
from fieldtrial.store.models import utcnow


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """How many trials the auto-operator ran."""

    completed: int
    invalid: int
    sessions: int


def simulate_study(
    folder: str | Path,
    rates: dict[str, float],
    *,
    seed: int = 1,
    invalid_rate: float = 0.0,
    trials_per_session: int = 40,
    max_trials: int | None = None,
) -> SimulationResult:
    """Run every pending slot with the sim runner and record the outcomes.

    ``rates`` gives the true success rate of every arm. Trials are spread over sessions of
    ``trials_per_session`` trials with a simulated clock, so the report's drift checks have
    something to look at. ``max_trials`` stops early (for partially filled studies).
    """
    with open_study(folder) as ctx:
        arm_ids = [a.id for a in ctx.spec.arms]
        missing = sorted(set(arm_ids) - set(rates))
        unknown = sorted(set(rates) - set(arm_ids))
        if missing or unknown:
            raise ServiceError(
                f"--rates must give every arm exactly once; missing {missing}, unknown {unknown}"
            )
        rubric = ctx.spec.rubric
        runner = SimRunner(
            {arm: SimArm(rates[arm], invalid_rate=invalid_rate) for arm in arm_ids},
            n_stages=len(rubric.stages),
            success_index=rubric.success_index,
            failure_tags=tuple(rubric.failure_tags),
            seed=seed,
        )
        clock = utcnow()
        session_id: str | None = None
        in_session = completed = invalid = sessions = 0
        while max_trials is None or completed + invalid < max_trials:
            slot = next_slot(ctx)
            if slot is None:
                break
            if session_id is None or in_session >= trials_per_session:
                if session_id is not None:
                    end_session(ctx, session_id, ended_at=clock)
                    clock += timedelta(minutes=30)
                sessions += 1
                session_id = start_session(
                    ctx,
                    operator="sim-operator",
                    rig="sim-rig",
                    software={"runner": "sim", "seed": str(seed)},
                    started_at=clock,
                )
                in_session = 0
            arm = ctx.spec.arm(slot.arm)
            runner.prepare(ArmSpec(arm.id, slot.blind_code, dict(arm.policy), dict(arm.serving)))
            runner.start(
                TrialContext(
                    seq=slot.seq,
                    condition=slot.condition,
                    factors=slot.factors,
                    instruction=ctx.spec.task.instruction,
                    timeout_s=ctx.spec.limits.timeout_s,
                )
            )
            artifacts = runner.stop("other")
            record_trial(
                ctx,
                slot.slot_id,
                session_id,
                stage_index=-1 if artifacts.stage_index is None else artifacts.stage_index,
                termination=artifacts.termination,
                duration_s=artifacts.duration_s,
                failure_tags=artifacts.failure_tags,
                started_at=clock,
                invalid_reason=artifacts.invalid_reason,
                source="sim",
            )
            clock += timedelta(seconds=artifacts.duration_s + 20.0)  # reset time between trials
            in_session += 1
            if artifacts.invalid_reason:
                invalid += 1
            else:
                completed += 1
        if session_id is not None:
            end_session(ctx, session_id, ended_at=clock)
        runner.close()
        return SimulationResult(completed=completed, invalid=invalid, sessions=sessions)
