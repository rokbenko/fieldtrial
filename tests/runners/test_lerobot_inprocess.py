"""The in-process ``lerobot`` runner, against a fake LeRobot 0.6.1 (tests/runners/fake_lerobot)."""

import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from fieldtrial.design import StudyValidationError, parse_study
from fieldtrial.design.hashing import design_hash, normalized_design
from fieldtrial.design.runner_config import LeRobotRunnerConfig, lerobot_arm_error
from fieldtrial.runners import lerobot_inprocess as lr
from fieldtrial.runners.base import ArmSpec, RunnerError, TrialContext
from fieldtrial.templates import template_text

FAKE = Path(__file__).parent / "fake_lerobot"


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """The fake ``lerobot`` package, importable for one test."""
    monkeypatch.syspath_prepend(str(FAKE))
    monkeypatch.setattr(lr, "_STRATEGY", None)
    import lerobot
    import lerobot.robots

    lerobot.CALLS.clear()
    lerobot.robots.ROBOTS.clear()
    yield lerobot
    for name in list(sys.modules):
        if name == "draccus" or name == "lerobot" or name.startswith(("lerobot.", "draccus.")):
            del sys.modules[name]


def checkpoint(root: Path, name: str, offset: float = 0.1, **config: Any) -> str:
    path = root / name
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "offset": offset,
                "input_features": {
                    "observation.images.top": {"type": "VISUAL", "shape": [3, 4, 6]}
                },
                **config,
            }
        )
    )
    return str(path)


def config(**kw: Any) -> LeRobotRunnerConfig:
    return LeRobotRunnerConfig.model_validate(
        {"robot": {"type": "fake_arm", "port": "/dev/ttyACM0"}, "fps": 50, "reset_s": 0.1, **kw}
    )


def arm(code: str, path: str, **serving: Any) -> ArmSpec:
    return ArmSpec(arm_id=code.lower(), blind_code=code, policy={"path": path}, serving=serving)


def trial(seq: int, timeout_s: float | None = None) -> TrialContext:
    return TrialContext(
        seq=seq, condition="c1", factors={}, trial_id=f"trial-{seq}", timeout_s=timeout_s
    )


def run(runner: lr.LeRobotRunner, spec: ArmSpec, seq: int, seconds: float = 0.15) -> Any:
    runner.prepare(spec)
    runner.start(trial(seq))
    time.sleep(seconds)
    return runner.stop()


def calls(fake: ModuleType, kind: str) -> list[tuple[Any, ...]]:
    return [c for c in fake.CALLS if c[0] == kind]


# --- configuration --------------------------------------------------------------------------


def _study(block: str, runner: str = "lerobot") -> str:
    text = template_text("basic", "s").replace("runner: manual", f"runner: {runner}")
    return text + block


ROBOT = "runners:\n  lerobot:\n    robot: {type: so101_follower, port: /dev/ttyACM0}\n"


def test_study_validates_the_lerobot_runner() -> None:
    spec = parse_study(_study(ROBOT)).spec
    assert spec.runners is not None
    assert spec.runners.lerobot is not None
    assert spec.runners.lerobot.fps == 30
    assert spec.runners.lerobot.keep_loaded == 1
    with pytest.raises(StudyValidationError, match=r"add runners\.lerobot\.robot"):
        parse_study(_study(""))
    with pytest.raises(StudyValidationError, match="needs a type"):
        parse_study(_study("runners:\n  lerobot:\n    robot: {port: /dev/ttyACM0}\n"))
    with pytest.raises(StudyValidationError, match="keep_loaded"):
        parse_study(_study(ROBOT + "    keep_loaded: 0\n"))
    no_path = _study(ROBOT).replace(
        "policy: {path: outputs/pi05_21h/checkpoints/050000/pretrained_model}", "policy: {}", 1
    )
    with pytest.raises(StudyValidationError, match=r"needs policy\.path"):
        parse_study(no_path)
    mixed = _study(ROBOT).replace("runner: lerobot", "runner: manual", 1)
    with pytest.raises(StudyValidationError, match="must run every arm"):
        parse_study(mixed)


@pytest.mark.parametrize(
    ("policy", "serving", "fragment"),
    [
        ({"path": "p"}, {}, None),
        ({"path": "p", "revision": "main"}, {"inference": {"type": "rtc"}}, None),
        ({"path": " "}, {}, "policy.path"),
        ({"path": "p", "revision": 3}, {}, "revision"),
        ({"path": "p"}, {"inference": {"type": "async"}}, "sync or rtc"),
        ({"path": "p"}, {"inference": "rtc"}, "sync or rtc"),
        ({"path": "p"}, {"interpolation_multiplier": 0}, "interpolation_multiplier"),
        ({"path": "p"}, {"interpolation_multiplier": True}, "interpolation_multiplier"),
    ],
)
def test_arm_settings(
    policy: dict[str, Any], serving: dict[str, Any], fragment: str | None
) -> None:
    error = lerobot_arm_error(policy, serving)
    if fragment is None:
        assert error is None
    else:
        assert error is not None
        assert fragment in error


def test_setup_settings_stay_out_of_the_design_hash() -> None:
    base = parse_study(_study(ROBOT)).spec
    moved = parse_study(
        _study(
            ROBOT.replace("ttyACM0", "ttyACM1")
            + "    device: cuda\n    keep_loaded: all\n    dataset: {root: elsewhere}\n"
        )
    ).spec
    faster = parse_study(_study(ROBOT + "    fps: 15\n")).spec
    assert design_hash(base) == design_hash(moved)
    assert design_hash(base) != design_hash(faster)
    # A study with another runner keeps the hash it had before the lerobot runner existed.
    command = parse_study(
        _study("runners:\n  command:\n    template: rollout {policy.path}\n", runner="command")
    ).spec
    assert "lerobot" not in normalized_design(command)["runners"]


# --- versions and imports -------------------------------------------------------------------


@pytest.mark.parametrize("version", ["0.6.1", "0.6.9", "0.6.2.dev0"])
def test_supported_versions(version: str) -> None:
    lr.check_version(version)


@pytest.mark.parametrize("version", ["0.6.0", "0.7.0", "1.0.0", "unknown"])
def test_unsupported_versions(version: str) -> None:
    with pytest.raises(RunnerError, match=r">=0\.6\.1,<0\.7"):
        lr.check_version(version)


def test_missing_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "lerobot", None)
    with pytest.raises(RunnerError, match=r"fieldtrial\[lerobot-runner\]"):
        lr.import_lerobot()
    assert lr.probe([("K7", {"path": "p"})]) == [
        ("lerobot", False, lr.probe([])[0][2]),
    ]


def test_wrong_lerobot_version(fake: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fake, "__version__", "0.5.0")
    with pytest.raises(RunnerError, match=r"found 0\.5\.0"):
        lr.import_lerobot()


# --- trials -----------------------------------------------------------------------------------


def test_trials_are_recorded_as_episodes(fake: ModuleType, tmp_path: Path) -> None:
    a = checkpoint(tmp_path / "ckpt", "a", offset=0.5)
    runner = lr.LeRobotRunner(config(), study="s", folder=tmp_path, instruction="stack the cups")
    assert runner.status().state == "idle"
    out = run(runner, arm("K7", a), 1)
    robot = fake.robots.ROBOTS[0]
    assert robot.actions, "the policy drove the robot"
    # The fake policy adds its offset to the joint positions it sees.
    assert robot.actions[0] == {"j0.pos": 0.5, "j1.pos": 0.5}
    assert out.termination == "operator_stop"
    assert out.stage_index is None  # the operator labels the outcome
    assert out.episode is not None
    assert out.episode.episode_index == 0
    assert out.episode.codebase_version == "v3.0"
    assert out.episode.length == out.metrics["frames"] > 0
    root = tmp_path / "lerobot" / "s"
    assert Path(out.episode.root) == root.resolve()
    metrics = out.metrics
    assert metrics["episode_index"] == 0
    assert metrics["arm_load_s"] >= 0
    assert metrics["policy_loaded"] == 1
    assert metrics["policy_mb"] == 4.0
    assert metrics["timed_out"] == 0
    assert "record_hz" in metrics
    assert "save_s" in metrics
    assert lr.read_sidecar(root) == [{"episode_index": 0, "trial_id": "trial-1", "seq": 1}]
    # The task is the study's instruction; the dataset says nothing about the arm.
    assert calls(fake, "engine")[0][2] == "stack the cups"
    episodes = json.loads((root / "meta" / "fake_episodes.json").read_text())
    assert episodes[0]["tasks"] == ["stack the cups"]
    # Processors get the stats lerobot-rollout passes for a new dataset, never recorded ones.
    assert all(c[2] == {} for c in calls(fake, "processors"))
    # The dataset is finalized after each episode and reopened for the next.
    assert calls(fake, "dataset_resume")
    assert calls(fake, "return_to_initial")  # reset between trials
    out2 = run(runner, arm("K7", a), 2)
    assert out2.episode is not None
    assert out2.episode.episode_index == 1
    assert out2.metrics["arm_load_s"] == 0  # same arm: nothing loaded
    runner.close()
    assert runner.status().state == "closed"
    assert ("disconnect",) in fake.CALLS
    assert len(calls(fake, "connect")) == 1  # connected once


def test_switching_arms_loads_and_evicts(fake: ModuleType, tmp_path: Path) -> None:
    a = checkpoint(tmp_path / "ckpt", "a")
    b = checkpoint(tmp_path / "ckpt", "b")
    runner = lr.LeRobotRunner(config(), study="s", folder=tmp_path)
    # Two arms can share one checkpoint (a serving sweep): its weights load once.
    run(runner, arm("K7", a), 1)
    out = run(runner, arm("Q2", a, interpolation_multiplier=2), 2)
    assert len(calls(fake, "load")) == 1
    assert out.metrics["policy_loaded"] == 0
    run(runner, arm("M4", b), 3)
    run(runner, arm("K7", a), 4)  # keep_loaded: 1 evicted a
    assert [c[1] for c in calls(fake, "load")] == [a, b, a]
    runner.close()

    fake.CALLS.clear()
    keep = lr.LeRobotRunner(config(keep_loaded="all"), study="t", folder=tmp_path)
    for seq, spec in enumerate([arm("K7", a), arm("M4", b), arm("K7", a), arm("M4", b)], 1):
        run(keep, spec, seq, 0.05)
    assert [c[1] for c in calls(fake, "load")] == [a, b]
    keep.close()


def test_interpolation_records_at_the_dataset_rate(fake: ModuleType, tmp_path: Path) -> None:
    a = checkpoint(tmp_path / "ckpt", "a")
    runner = lr.LeRobotRunner(config(fps=20), study="s", folder=tmp_path)
    out = run(runner, arm("K7", a, interpolation_multiplier=3), 1, seconds=0.5)
    robot = fake.robots.ROBOTS[0]
    frames = out.metrics["frames"]
    # The robot is commanded about three times per recorded frame.
    assert len(robot.actions) >= 2 * frames
    assert out.metrics["record_hz"] <= 25
    runner.close()


def test_rtc_settings_follow_the_arm(fake: ModuleType, tmp_path: Path) -> None:
    rtc = checkpoint(tmp_path / "ckpt", "rtc", supports_rtc=True)
    runner = lr.LeRobotRunner(config(), study="s", folder=tmp_path)
    policy_rtc = {"type": "rtc", "rtc": {"execution_horizon": 8}, "queue_threshold": 20}
    run(runner, arm("K7", rtc, inference=policy_rtc), 1, 0.05)
    run(runner, arm("Q2", rtc, inference={"type": "sync"}), 2, 0.05)
    assert [c[1] for c in calls(fake, "engine")] == ["rtc", "sync"]
    # The shared policy gets the RTC options of the arm that runs, and none for sync.
    assert [c[1] for c in calls(fake, "init_rtc")] == [{"execution_horizon": 8}, None]
    assert len(calls(fake, "load")) == 1
    runner.close()


def test_time_limit(fake: ModuleType, tmp_path: Path) -> None:
    a = checkpoint(tmp_path / "ckpt", "a")
    runner = lr.LeRobotRunner(config(), study="s", folder=tmp_path)
    runner.prepare(arm("K7", a))
    runner.start(trial(1, timeout_s=0.05))
    deadline = time.monotonic() + 5
    while "time limit" not in runner.status().message and time.monotonic() < deadline:
        time.sleep(0.01)
    assert runner.status().message == "time limit reached; stop the trial"
    out = runner.stop()
    assert out.termination == "timeout"
    assert out.metrics["timed_out"] == 1
    # A trial stopped before any frame leaves no episode.
    runner.prepare(arm("K7", a))
    runner.start(trial(2, timeout_s=0.0))
    time.sleep(0.05)
    empty = runner.stop()
    assert empty.episode is None
    assert empty.metrics["frames"] == 0
    runner.close()


def test_a_robot_fault_stops_the_loop(fake: ModuleType, tmp_path: Path) -> None:
    a = checkpoint(tmp_path / "ckpt", "a")
    cfg = config(robot={"type": "fake_arm", "fail_after": 4})
    runner = lr.LeRobotRunner(cfg, study="s", folder=tmp_path)
    runner.prepare(arm("K7", a))
    runner.start(trial(1))
    deadline = time.monotonic() + 5
    while "void the trial" not in runner.status().message and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "OSError: motor bus timeout" in runner.status().message
    out = runner.stop()
    assert out.termination == "robot_fault"
    assert out.metrics["loop_error"] == 1
    assert "control loop stopped" in runner.status().message
    runner.close()


def test_an_engine_failure_stops_the_loop(fake: ModuleType, tmp_path: Path) -> None:
    a = checkpoint(tmp_path / "ckpt", "a", engine_fails=True)
    runner = lr.LeRobotRunner(config(), study="s", folder=tmp_path)
    out = run(runner, arm("K7", a), 1, 0.1)
    assert out.termination == "robot_fault"
    assert "inference engine failed" in runner.status().message
    runner.close()


def test_a_new_server_continues_the_dataset(fake: ModuleType, tmp_path: Path) -> None:
    a = checkpoint(tmp_path / "ckpt", "a")
    first = lr.LeRobotRunner(config(), study="s", folder=tmp_path)
    run(first, arm("K7", a), 1, 0.05)
    first.close()
    second = lr.LeRobotRunner(config(), study="s", folder=tmp_path)
    out = run(second, arm("K7", a), 2, 0.05)
    second.close()
    assert out.episode is not None
    assert out.episode.episode_index == 1
    assert [e["trial_id"] for e in lr.read_sidecar(tmp_path / "lerobot" / "s")] == [
        "trial-1",
        "trial-2",
    ]
    assert len(calls(fake, "dataset_create")) == 1


def test_dataset_settings(fake: ModuleType, tmp_path: Path) -> None:
    a = checkpoint(tmp_path / "ckpt", "a")
    cfg = config(dataset={"repo_id": "me/evals", "root": "data/evals", "streaming_encoding": True})
    runner = lr.LeRobotRunner(cfg, study="s", folder=tmp_path)
    out = run(runner, arm("K7", a), 1, 0.05)
    runner.close()
    assert out.episode is not None
    assert Path(out.episode.root) == (tmp_path / "data" / "evals").resolve()
    created = calls(fake, "dataset_create")[0]
    assert created[1] == "me/evals"
    assert created[4]["streaming_encoding"] is True
    assert lr.default_root(config(), "x", tmp_path) == (
        "local/x",
        (tmp_path / "lerobot" / "x").resolve(),
    )


# --- errors -----------------------------------------------------------------------------------


def test_protocol_errors(fake: ModuleType, tmp_path: Path) -> None:
    a = checkpoint(tmp_path / "ckpt", "a")
    runner = lr.LeRobotRunner(config(), study="s", folder=tmp_path)
    with pytest.raises(RunnerError, match="prepare"):
        runner.start(trial(1))
    with pytest.raises(RunnerError, match="no trial"):
        runner.stop()
    runner.prepare(arm("K7", a))
    runner.start(trial(1))
    with pytest.raises(RunnerError, match="already running"):
        runner.start(trial(2))
    with pytest.raises(RunnerError, match="stop it before switching"):
        runner.prepare(arm("Q2", a))
    runner.close()  # stops the running trial first
    assert runner.status().state == "closed"


@pytest.mark.parametrize(
    ("robot", "fragment"),
    [
        ({"type": "no_such_robot"}, "unknown type"),
        ({"type": "fake_arm", "fail_connect": True}, "connecting the robot failed"),
    ],
)
def test_robot_errors(
    fake: ModuleType, tmp_path: Path, robot: dict[str, Any], fragment: str
) -> None:
    a = checkpoint(tmp_path / "ckpt", "a")
    runner = lr.LeRobotRunner(config(robot=robot), study="s", folder=tmp_path)
    with pytest.raises(RunnerError, match=fragment):
        runner.prepare(arm("K7", a))
    status = runner.status()
    assert status.state == "idle"
    assert fragment.split()[0] in status.message


@pytest.mark.parametrize(
    ("settings", "serving", "fragment"),
    [
        ({"use_peft": True}, {}, "PEFT"),
        ({"fail_load": True}, {}, r"arm K7: the policy could not be loaded \(RuntimeError"),
        ({}, {"inference": {"type": "rtc"}}, "does not support RTC"),
        ({"relative_actions": True}, {}, "relative actions"),
        ({}, {"inference": {"type": "rtc", "bogus": 1}}, r"serving\.inference"),
        (
            {
                "input_features": {
                    "observation.images.wrist": {"type": "VISUAL", "shape": [3, 4, 6]}
                }
            },
            {},
            "rename_map",
        ),
    ],
)
def test_policy_errors(
    fake: ModuleType,
    tmp_path: Path,
    settings: dict[str, Any],
    serving: dict[str, Any],
    fragment: str,
) -> None:
    path = checkpoint(tmp_path / "ckpt", "a", **settings)
    runner = lr.LeRobotRunner(config(), study="s", folder=tmp_path)
    with pytest.raises(RunnerError, match=fragment) as caught:
        runner.prepare(arm("K7", path, **serving))
    # Errors name the blind code, never the checkpoint.
    assert path not in str(caught.value)
    runner.close()


def test_rename_map_skips_the_camera_check(fake: ModuleType, tmp_path: Path) -> None:
    wrist = {"observation.images.wrist": {"type": "VISUAL", "shape": [3, 4, 6]}}
    path = checkpoint(tmp_path / "ckpt", "a", input_features=wrist)
    cfg = config(rename_map={"observation.images.top": "observation.images.wrist"})
    runner = lr.LeRobotRunner(cfg, study="s", folder=tmp_path)
    runner.prepare(arm("K7", path))
    overrides = calls(fake, "processors")[0][3]
    assert overrides["rename_observations_processor"] == {
        "rename_map": {"observation.images.top": "observation.images.wrist"}
    }
    assert overrides["device_processor"] == {"device": "cpu"}  # auto-selected
    runner.close()


def test_probe(fake: ModuleType, tmp_path: Path) -> None:
    a = checkpoint(tmp_path / "ckpt", "a")
    found = lr.probe([("K7", {"path": a}), ("Q2", {"path": str(tmp_path / "missing")})])
    assert found[0] == ("lerobot", True, "LeRobot is installed")
    assert found[1] == ("K7", True, "policy configuration found")
    assert found[2][:2] == ("Q2", False)
    assert "config.json not found" in found[2][2]


def test_bad_sidecar(tmp_path: Path) -> None:
    (tmp_path / lr.SIDECAR).write_text("{")
    with pytest.raises(RunnerError, match="cannot read"):
        lr.read_sidecar(tmp_path)
    assert lr.read_sidecar(tmp_path / "none") == []


# --- the server -------------------------------------------------------------------------------


def test_the_server_runs_trials_and_links_episodes(fake: ModuleType, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from tests.web.conftest import API
    from typer.testing import CliRunner

    from fieldtrial.cli.main import app as cli
    from fieldtrial.report.html import render_html
    from fieldtrial.report.markdown import render_markdown
    from fieldtrial.services import open_study
    from fieldtrial.services.analysis import analyze_study
    from fieldtrial.services.events import list_events
    from fieldtrial.services.study import init_study, lock_study
    from fieldtrial.web.app import create_app

    paths = [checkpoint(tmp_path / "ckpt", name, offset=o) for name, o in (("a", 0.1), ("b", 0.2))]
    folder = tmp_path / "studies" / "lr"
    init_study(folder, template="basic", name="lr")
    study = folder / "study.yaml"
    text = study.read_text().replace("range: [1, 40]", "range: [1, 2]")
    text = text.replace("runner: manual", "runner: lerobot").replace(
        "blinding: operator", "blinding: none"
    )
    old = "    policy: {path: outputs/pi05_21h/checkpoints/050000/pretrained_model}\n"
    for path in paths:
        text = text.replace(old, f"    policy: {{path: {path}}}\n", 1)
    text += "runners:\n  lerobot:\n    robot: {type: fake_arm}\n    fps: 50\n    reset_s: 0.1\n"
    study.write_text(text)
    lock_study(folder)

    checked = CliRunner().invoke(cli, ["check-runners", str(folder)])
    assert checked.exit_code == 0, checked.output
    assert "policy configuration found" in checked.output

    with TestClient(create_app(folder.parent), base_url="http://127.0.0.1") as client:
        session = client.post(
            "/api/v1/studies/lr/sessions", json={"operator": "a", "rig": "r"}, headers=API
        ).json()["session"]["session_id"]
        for _ in range(4):
            slot = client.get("/api/v1/studies/lr/next").json()
            started = client.post(
                "/api/v1/studies/lr/trials",
                json={"slot_id": slot["slot_id"], "session_id": session},
                headers=API,
            ).json()
            time.sleep(0.1)
            stopped = client.post(
                f"/api/v1/studies/lr/trials/{started['trial_id']}/stop",
                json={"expected_version": started["version"]},
                headers=API,
            ).json()
            done = client.post(
                f"/api/v1/studies/lr/trials/{started['trial_id']}/complete",
                json={
                    "stage": "clean",
                    "termination": "success",
                    "expected_version": stopped["version"],
                },
                headers=API,
            )
            assert done.status_code == 200, done.text
    assert ("disconnect",) in fake.CALLS  # the server closed the runner

    with open_study(folder) as ctx, ctx.db() as db:
        links = list_events(db, ctx.study_id, "dataset_link")
        outputs = list_events(db, ctx.study_id, "runner_output")
    assert len(links) == len(outputs) == 4
    assert [e.payload["links"][0]["episode_index"] for e in links] == [0, 1, 2, 3]
    assert all(e.payload["source"] == "runner" for e in links)
    assert links[0].payload["root"] == str((folder / "lerobot" / "lr").resolve())

    results = analyze_study(folder)
    assert sorted(u.arm for u in results.runner) == ["baseline", "q50"]
    for summary in results.runner:
        assert summary.trials == 2
        assert summary.policy_loads is not None
        assert summary.policy_loads >= 1
        assert summary.loop_errors == 0
        assert summary.record_hz_median is not None
    report = render_markdown(results)
    assert "Median recording rate (Hz)" in report
    assert "Median latency (ms)" not in report
    page = render_html(results, charts=False)
    assert "Median recording rate (Hz)" in page
    assert "Median latency (ms)" not in page
