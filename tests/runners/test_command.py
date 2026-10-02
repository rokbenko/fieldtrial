"""The command runner and its template, with small Python scripts as rollout commands."""

import json
import shlex
import sys
import time
from pathlib import Path

import pytest

from fieldtrial.design import StudyValidationError, parse_study
from fieldtrial.design.runner_config import (
    CommandRunnerConfig,
    TemplateError,
    render_command,
    sample_values,
)
from fieldtrial.runners.base import ArmSpec, RunnerError, TrialContext
from fieldtrial.runners.command import CommandRunner
from fieldtrial.templates import template_text

PY = shlex.quote(sys.executable)

# A fake rollout: records its argv and environment, then waits for a signal.
ROLLOUT = """
import json, os, signal, sys, time
out = os.environ["OUT"]
with open(out, "a") as fh:
    fh.write(json.dumps({"argv": sys.argv[1:], "arm": os.environ["FIELDTRIAL_ARM"],
                         "trial": os.environ["FIELDTRIAL_TRIAL_ID"],
                         "seq": os.environ["FIELDTRIAL_SEQ"]}) + "\\n")
print("rollout running", flush=True)
mode = sys.argv[1] if len(sys.argv) > 1 else "wait"
if mode == "exit0":
    sys.exit(0)
if mode == "ignore":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
def stop(*_):
    print("got SIGINT", flush=True)
    sys.exit(7)
if mode != "ignore":
    signal.signal(signal.SIGINT, stop)
while True:
    time.sleep(0.05)
"""

# A per-arm process: reads start/stop lines and logs them.
SERVER = """
import json, os, sys
out = os.environ["OUT"]
for line in sys.stdin:
    with open(out, "a") as fh:
        fh.write(json.dumps({"arm": os.environ["FIELDTRIAL_ARM"], "line": line.strip()}) + "\\n")
"""


def _arm(code: str = "K7", path: str = "ckpt/a") -> ArmSpec:
    return ArmSpec(
        arm_id="secret-arm",
        blind_code=code,
        policy={"path": path},
        serving={"inference.queue_threshold": 30, "rtc": True},
    )


def _trial(seq: int = 1) -> TrialContext:
    return TrialContext(
        seq=seq, condition="slot=3", factors={"slot": 3}, trial_id=f"t-{seq}", timeout_s=30
    )


@pytest.fixture
def scripts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "rollout.py").write_text(ROLLOUT)
    (tmp_path / "server.py").write_text(SERVER)
    monkeypatch.setenv("OUT", str(tmp_path / "out.jsonl"))
    return tmp_path


def _records(folder: Path) -> list[dict]:  # type: ignore[type-arg]
    path = folder / "out.jsonl"
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- template -----------------------------------------------------------------------------


def test_render_fills_placeholders_per_argument() -> None:
    values = sample_values(
        policy={"path": "out/my ckpt"},
        serving={"inference.queue_threshold": 50, "rtc": True},
        factors={"slot": 4},
    )
    argv = render_command(
        "lerobot-rollout --policy.path={policy.path} "
        "--inference.queue_threshold={serving[inference.queue_threshold]} "
        "--rtc={serving[rtc]} --slot={factors.slot} --code {blind_code} '{{literal}}'",
        values,
    )
    assert argv == [
        "lerobot-rollout",
        "--policy.path=out/my ckpt",  # one argument, despite the space
        "--inference.queue_threshold=50",
        "--rtc=true",
        "--slot=4",
        "--code",
        "XX",
        "{literal}",
    ]


@pytest.mark.parametrize(
    ("template", "fragment"),
    [
        ("run {policy.nope}", "policy has no key 'nope' (keys: path)"),
        ("run {arm}", "unknown placeholder {arm}"),
        ("run {policy}", "unknown placeholder"),
        ("run {seq", "unbalanced brace"),
        ("run 'unterminated", "cannot be split"),
    ],
)
def test_template_errors(template: str, fragment: str) -> None:
    values = sample_values(policy={"path": "p"}, serving={}, factors={})
    with pytest.raises(
        TemplateError, match=fragment.replace("{", r"\{").replace("(", r"\(").replace(")", r"\)")
    ):
        render_command(template, values)


def _study(runner_block: str, runner: str = "command") -> str:
    text = template_text("basic", "s").replace("runner: manual", f"runner: {runner}")
    return text + runner_block


def test_study_validates_the_template_for_every_arm() -> None:
    ok = _study(
        "runners:\n  command:\n    template: rollout --path={policy.path} "
        "--q={serving[inference.queue_threshold]}\n"
    )
    spec = parse_study(ok).spec
    assert spec.runners is not None
    assert spec.runners.command is not None
    assert spec.runners.command.launch == "per_trial"
    with pytest.raises(StudyValidationError, match="has no key 'missing'"):
        parse_study(_study("runners:\n  command:\n    template: rollout {policy.missing}\n"))
    with pytest.raises(StudyValidationError, match=r"add runners\.command\.template"):
        parse_study(_study(""))
    mixed = ok.replace("runner: command", "runner: manual", 1)
    with pytest.raises(StudyValidationError, match="must run every arm"):
        parse_study(mixed)
    router = _study("", runner="openpi_router")
    with pytest.raises(StudyValidationError, match=r"policy\.url"):
        parse_study(router)


# --- runner, per trial ----------------------------------------------------------------------


def _runner(folder: Path, template: str, **kwargs: object) -> CommandRunner:
    config = CommandRunnerConfig(template=template, grace_s=2, **kwargs)  # type: ignore[arg-type]
    return CommandRunner(config, study="s", folder=folder, success_index=3)


def test_per_trial_launch_signal_and_log(scripts: Path) -> None:
    runner = _runner(scripts, f"{PY} rollout.py wait --path={{policy.path}}")
    assert runner.capabilities.can_switch_arms
    assert not runner.capabilities.reports_outcome
    runner.prepare(_arm())
    runner.start(_trial())
    [rec] = _records(scripts)
    assert rec == {"argv": ["wait", "--path=ckpt/a"], "arm": "K7", "trial": "t-1", "seq": "1"}
    time.sleep(0.2)
    artifacts = runner.stop("operator_stop")
    assert artifacts.metrics == {"exit_code": 7.0, "exited_before_stop": 0.0}
    assert artifacts.stage_index is None
    assert artifacts.termination == "operator_stop"
    assert artifacts.log is not None
    log = Path(artifacts.log).read_text()
    assert "rollout running" in log
    assert "got SIGINT" in log
    assert Path(artifacts.log) == scripts / "logs" / "t-1.log"
    with pytest.raises(RunnerError, match="no trial is running"):
        runner.stop()


def test_exit_code_suggests_success(scripts: Path) -> None:
    runner = _runner(scripts, f"{PY} rollout.py exit0", success_exit_code=0)
    assert runner.capabilities.reports_outcome
    runner.prepare(_arm())
    runner.start(_trial())
    deadline = time.monotonic() + 5
    while runner.status().message.find("command ended") < 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert "command ended (exit 0)" in runner.status().message
    artifacts = runner.stop()
    assert artifacts.stage_index == 3
    assert artifacts.termination == "success"
    assert artifacts.metrics["exited_before_stop"] == 1.0


def test_stubborn_process_is_killed(scripts: Path) -> None:
    runner = CommandRunner(
        CommandRunnerConfig(template=f"{PY} rollout.py ignore", grace_s=0.3),
        study="s",
        folder=scripts,
    )
    import fieldtrial.runners.command as command

    command.TERM_GRACE_S = 0.3
    try:
        runner.prepare(_arm())
        runner.start(_trial())
        _records(scripts)
        artifacts = runner.stop()
    finally:
        command.TERM_GRACE_S = 5.0
    assert artifacts.metrics["exit_code"] == -9.0


def test_missing_executable_is_a_clear_error(scripts: Path) -> None:
    runner = _runner(scripts, "no-such-rollout-binary --x")
    runner.prepare(_arm())
    with pytest.raises(RunnerError, match="could not start 'no-such-rollout-binary'"):
        runner.start(_trial())
    # The runner is still usable afterwards.
    with pytest.raises(RunnerError, match="no trial is running"):
        runner.stop()


def test_cannot_switch_arms_mid_trial(scripts: Path) -> None:
    runner = _runner(scripts, f"{PY} rollout.py wait")
    runner.prepare(_arm())
    runner.start(_trial())
    with pytest.raises(RunnerError, match="trial is running"):
        runner.prepare(_arm("Q2"))
    runner.close()
    assert runner.status().state == "closed"


# --- runner, per arm --------------------------------------------------------------------------


def test_per_arm_keeps_one_process_and_restarts_on_switch(scripts: Path) -> None:
    runner = _runner(scripts, f"{PY} server.py", launch="per_arm")
    runner.prepare(_arm("K7"))
    runner.start(_trial(1))
    runner.stop()
    runner.prepare(_arm("K7"))  # same arm: same process
    runner.start(_trial(2))
    runner.stop()
    runner.prepare(_arm("Q2"))  # another arm: restart
    runner.start(_trial(3))
    runner.stop()
    runner.close()
    deadline = time.monotonic() + 5
    while len(_records(scripts)) < 6 and time.monotonic() < deadline:
        time.sleep(0.05)
    lines = [(r["arm"], r["line"].split(" ")[0]) for r in _records(scripts)]
    assert lines == [
        ("K7", "start"),
        ("K7", "stop"),
        ("K7", "start"),
        ("K7", "stop"),
        ("Q2", "start"),
        ("Q2", "stop"),
    ]
    first = json.loads(_records(scripts)[0]["line"][len("start ") :])
    assert first == {"trial_id": "t-1", "seq": 1, "condition": "slot=3", "factors": {"slot": 3}}
    assert len(list((scripts / "logs").glob("arm-*.log"))) == 2
