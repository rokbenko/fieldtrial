"""The auto-operator: fill a locked study with simulated trials."""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from fieldtrial.runners.base import ArmSpec, TrialContext
from fieldtrial.runners.sim import SimArm, SimRunner
from fieldtrial.services._context import ServiceError, StudyContext, open_study
from fieldtrial.services.adaptive import check_after_block
from fieldtrial.services.interim import interim_status, run_interim_in
from fieldtrial.services.session import end_session, start_session
from fieldtrial.services.study import init_study, lock_study
from fieldtrial.services.trial import collect_records, next_slot, pending_slots, record_trial
from fieldtrial.store.models import utcnow


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """How many trials the auto-operator ran."""

    completed: int
    invalid: int
    sessions: int
    interim_looks: int = 0
    stopped_at: int | None = None  # interim look that stopped the study
    dropped: tuple[str, ...] = ()  # blind codes of arms dropped by best-arm selection


RESET_S = 20.0  # simulated time between trials
BREAK_S = 1800.0  # simulated time between sessions
MAX_TRIAL_S = 600.0  # bound on a simulated trial without a timeout


def _span_bound(
    pending: int,
    *,
    max_trials: int | None,
    invalid_rate: float,
    timeout_s: float | None,
    trials_per_session: int,
) -> timedelta:
    """An upper bound on the simulated run's length, so it can end before the current time.

    Simulated trials then never carry future timestamps, and trials run later in the console
    follow them in time.
    """
    trials = max_trials if max_trials is not None else math.ceil(pending / (1 - invalid_rate)) + 10
    sessions = trials // max(trials_per_session, 1) + 1
    per_trial = (timeout_s or MAX_TRIAL_S) + RESET_S
    return timedelta(seconds=trials * per_trial + sessions * BREAK_S)


def _latest_end(ctx: StudyContext) -> datetime | None:
    records, _info = collect_records(ctx)
    ends = [r.started_at + timedelta(seconds=r.duration_s or 0.0) for r in records]
    return max(ends, default=None)


def simulate_study(
    folder: str | Path,
    rates: dict[str, float],
    *,
    seed: int = 1,
    invalid_rate: float = 0.0,
    trials_per_session: int = 40,
    max_trials: int | None = None,
    interim: bool = True,
) -> SimulationResult:
    """Run every pending slot with the sim runner and record the outcomes.

    ``rates`` gives the true success rate of every arm. Trials are spread over sessions of
    ``trials_per_session`` trials with a simulated clock, so the report's drift checks have
    something to look at. ``max_trials`` stops early (for partially filled studies). With
    ``interim``, planned interim looks of a group-sequential study run as they come due,
    as the protocol asks of an operator, and anytime or selection studies check after every
    block as they would live.
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
        clock = utcnow() - _span_bound(
            len(pending_slots(ctx)),
            max_trials=max_trials,
            invalid_rate=invalid_rate,
            timeout_s=ctx.spec.limits.timeout_s,
            trials_per_session=trials_per_session,
        )
        # Continue after trials already recorded, so a second run stays in run order.
        last_end = _latest_end(ctx)
        if last_end is not None and last_end + timedelta(seconds=BREAK_S) > clock:
            clock = last_end + timedelta(seconds=BREAK_S)
        session_id: str | None = None
        in_session = completed = invalid = sessions = looks = 0
        stopped_at: int | None = None
        sequential = interim and ctx.spec.analysis.stopping.rule == "group_sequential"
        adaptive = interim and (
            ctx.spec.analysis.stopping.rule == "anytime" or ctx.spec.analysis.selection is not None
        )
        dropped: list[str] = []
        while max_trials is None or completed + invalid < max_trials:
            slot = next_slot(ctx)
            if slot is None:
                break
            if session_id is None or in_session >= trials_per_session:
                if session_id is not None:
                    end_session(ctx, session_id, ended_at=clock)
                    clock += timedelta(seconds=BREAK_S)
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
            clock += timedelta(seconds=artifacts.duration_s + RESET_S)
            in_session += 1
            if artifacts.invalid_reason:
                invalid += 1
            else:
                completed += 1
            if sequential:
                status = interim_status(ctx)
                if status is not None and status.due:
                    look = run_interim_in(ctx, actor="sim-operator")
                    looks += 1
                    if look.decision == "stop":
                        stopped_at = look.look
            if adaptive and not artifacts.invalid_reason:
                outcome = check_after_block(ctx, actor="sim-operator")
                if outcome is not None:
                    looks += 1
                    dropped.extend(outcome.dropped)
                    if outcome.decision == "stop":
                        stopped_at = looks
        if session_id is not None:
            end_session(ctx, session_id, ended_at=clock)
        runner.close()
        return SimulationResult(
            completed=completed,
            invalid=invalid,
            sessions=sessions,
            interim_looks=looks,
            stopped_at=stopped_at,
            dropped=tuple(dropped),
        )


def sim_rates(folder: str | Path) -> dict[str, float]:
    """The true success rates of a study's simulated arms (``policy.sim_success_rate``)."""
    with open_study(folder) as ctx:
        return {
            a.id: float(a.policy.get("sim_success_rate", 0.5))
            for a in ctx.spec.arms
            if a.runner == "sim"
        }


def prepare_demo(folder: str | Path, *, prefill: int = 40, seed: int = 1) -> Path:
    """Create, lock and half-fill the demo study (a simulated robot) in ``folder``."""
    study_file = init_study(folder, template="demo", name="demo")
    lock_study(study_file.parent)
    if prefill:
        simulate_study(
            study_file.parent,
            sim_rates(study_file.parent),
            seed=seed,
            invalid_rate=0.03,
            trials_per_session=10,
            max_trials=prefill,
        )
    return study_file.parent
