"""The sim runner and the auto-operator."""

from pathlib import Path

import pytest

from fieldtrial.runners.base import ArmSpec, Runner, TrialContext
from fieldtrial.runners.sim import SimArm, SimRunner
from fieldtrial.services import ServiceError, open_study
from fieldtrial.services.simulate import simulate_study
from fieldtrial.services.study import study_status
from fieldtrial.services.trial import collect_records


def _runner(seed: int = 0, **arm: float) -> SimRunner:
    return SimRunner(
        {"a": SimArm(**arm)}, n_stages=4, success_index=3, failure_tags=("x", "y"), seed=seed
    )


def _run(runner: SimRunner, n: int, timeout: float | None = 45.0) -> list[object]:
    runner.prepare(ArmSpec("a", "A3"))
    out = []
    for seq in range(n):
        runner.start(TrialContext(seq=seq, condition="c", factors={}, timeout_s=timeout))
        out.append(runner.stop())
    return out


def test_sim_runner_is_a_runner() -> None:
    runner = _runner(success_rate=0.5)
    assert isinstance(runner, Runner)
    assert runner.status().state == "idle"
    runner.close()
    assert runner.status().state == "closed"


def test_sim_runner_is_deterministic() -> None:
    assert _run(_runner(7, success_rate=0.6), 50) == _run(_runner(7, success_rate=0.6), 50)
    assert _run(_runner(7, success_rate=0.6), 50) != _run(_runner(8, success_rate=0.6), 50)


def test_sim_runner_outcomes() -> None:
    results = _run(_runner(success_rate=0.7, invalid_rate=0.1), 2000)
    invalid = [r for r in results if r.invalid_reason]  # type: ignore[attr-defined]
    valid = [r for r in results if not r.invalid_reason]  # type: ignore[attr-defined]
    rate = sum(r.termination == "success" for r in valid) / len(valid)  # type: ignore[attr-defined]
    assert 0.05 < len(invalid) / len(results) < 0.15
    assert 0.66 < rate < 0.74
    for r in valid:
        if r.termination == "success":  # type: ignore[attr-defined]
            assert r.stage_index == 3  # type: ignore[attr-defined]
            assert r.duration_s < 45  # type: ignore[attr-defined]
        else:
            assert r.termination == "timeout"  # type: ignore[attr-defined]
            assert -1 <= r.stage_index < 3  # type: ignore[attr-defined]
            assert r.failure_tags in (("x",), ("y",))  # type: ignore[attr-defined]


def test_sim_runner_without_timeout() -> None:
    results = _run(_runner(success_rate=0.0), 20, timeout=None)
    assert {r.termination for r in results} == {"stuck"}  # type: ignore[attr-defined]


def test_sim_runner_misuse() -> None:
    runner = _runner(success_rate=0.5)
    with pytest.raises(RuntimeError):
        runner.start(TrialContext(seq=1, condition="c", factors={}))
    with pytest.raises(RuntimeError):
        runner.stop()
    with pytest.raises(ValueError, match="no simulated behavior"):
        runner.prepare(ArmSpec("b", "B4"))
    with pytest.raises(ValueError, match="success_rate"):
        SimArm(success_rate=1.5)
    with pytest.raises(ValueError, match="invalid_rate"):
        SimArm(success_rate=0.5, invalid_rate=0.5)
    with pytest.raises(ValueError, match="positive"):
        SimArm(success_rate=0.5, median_success_s=0)
    with pytest.raises(ValueError, match="stage index"):
        SimRunner({}, n_stages=2, success_index=2)


def test_simulate_fills_study(locked_study: Path) -> None:
    result = simulate_study(
        locked_study, {"baseline": 0.5, "q50": 0.9}, seed=2, invalid_rate=0.2, trials_per_session=3
    )
    assert result.completed == 8
    assert result.sessions >= 3
    status = study_status(locked_study)
    assert status.pending == 0
    assert status.done == 8
    assert status.invalid_trials == result.invalid
    with open_study(locked_study) as ctx:
        records, _ = collect_records(ctx)
    assert sum(r.status == "invalid" for r in records) == result.invalid
    assert len({r.session_id for r in records}) == result.sessions


def test_simulate_max_trials_and_rates(locked_study: Path) -> None:
    with pytest.raises(ServiceError, match="every arm"):
        simulate_study(locked_study, {"baseline": 0.5})
    with pytest.raises(ServiceError, match="every arm"):
        simulate_study(locked_study, {"baseline": 0.5, "q50": 0.5, "zzz": 0.1})
    partial = simulate_study(locked_study, {"baseline": 0.5, "q50": 0.9}, max_trials=3)
    assert partial.completed + partial.invalid == 3
    assert study_status(locked_study).pending == 5


def test_manual_runner() -> None:
    from fieldtrial.runners.manual import ManualRunner

    runner = ManualRunner()
    assert isinstance(runner, Runner)
    assert not runner.capabilities.reports_outcome
    with pytest.raises(RuntimeError):
        runner.start(TrialContext(seq=1, condition="c", factors={}))
    with pytest.raises(RuntimeError):
        runner.stop()
    runner.prepare(ArmSpec("a", "N4"))
    assert runner.status().message == "load arm N4"
    runner.start(TrialContext(seq=1, condition="c", factors={}))
    assert runner.status().state == "running"
    artifacts = runner.stop("operator_stop")
    assert artifacts.termination == "operator_stop"
    assert artifacts.stage_index is None
    assert artifacts.duration_s >= 0
    runner.close()
    assert runner.status().state == "closed"


def test_simulated_trials_end_before_now(locked_study: Path) -> None:
    from fieldtrial.store.models import utcnow

    simulate_study(locked_study, {"baseline": 0.5, "q50": 0.9}, invalid_rate=0.3, seed=5)
    now = utcnow()
    with open_study(locked_study) as ctx:
        records, _ = collect_records(ctx)
    ends = [
        r.started_at + __import__("datetime").timedelta(seconds=r.duration_s or 0) for r in records
    ]
    assert max(ends) <= now
    starts = [r.started_at for r in records]
    assert starts == sorted(starts)


def test_second_simulation_continues_in_order(locked_study: Path) -> None:
    rates = {"baseline": 0.5, "q50": 0.9}
    simulate_study(locked_study, rates, seed=1, max_trials=3)
    simulate_study(locked_study, rates, seed=2)
    with open_study(locked_study) as ctx:
        records, _ = collect_records(ctx)
    seqs = [r.seq for r in sorted(records, key=lambda r: r.started_at)]
    assert seqs == sorted(seqs)
