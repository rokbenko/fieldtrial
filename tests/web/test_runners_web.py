"""Switching runners driven by the console and the REST API, plus ``fieldtrial check-runners``."""

import re
import shlex
import socket
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from tests.runners.test_openpi_router import META, OTHER_META, FakePolicyServer
from tests.web.conftest import API
from tests.web.test_console import Console, field, trial_id
from typer.testing import CliRunner
from websockets.sync.client import connect

from fieldtrial.cli.main import app as cli
from fieldtrial.services import open_study
from fieldtrial.services.analysis import analyze_study
from fieldtrial.services.events import list_events
from fieldtrial.services.study import init_study, lock_study
from fieldtrial.store import models as m
from fieldtrial.web.app import create_app

PY = shlex.quote(sys.executable)
ROLLOUT = """
import sys, time
print("rollout", sys.argv[1:], flush=True)
if sys.argv[1] == "done":
    sys.exit(0)
while True:
    time.sleep(0.05)
"""


def _study(
    folder: Path,
    runner: str,
    extra: str,
    *,
    arms_policy: dict[str, str] | None = None,
    blinding: str = "operator",
) -> Path:
    init_study(folder, template="basic", name=folder.name)
    path = folder / "study.yaml"
    text = path.read_text().replace("range: [1, 40]", "range: [1, 2]")
    text = text.replace("runner: manual", f"runner: {runner}")
    text = text.replace("blinding: operator", f"blinding: {blinding}")
    if arms_policy:
        for arm, url in arms_policy.items():
            old = "    policy: {path: outputs/pi05_21h/checkpoints/050000/pretrained_model}\n"
            marker = f"  - id: {arm}\n"
            head, tail = text.split(marker, 1)
            tail = tail.replace(old, f"    policy: {{url: {url}}}\n", 1)
            text = head + marker + tail
    path.write_text(text + extra)
    return folder


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --- command runner through the console ----------------------------------------------------


@pytest.fixture
def command_root(tmp_path: Path) -> Path:
    root = tmp_path / "studies"
    folder = _study(
        root / "cmd",
        "command",
        "runners:\n  command:\n"
        f"    template: {PY} rollout.py done --path={{policy.path}}\n"
        "    success_exit_code: 0\n    grace_s: 2\n",
    )
    (folder / "rollout.py").write_text(ROLLOUT)
    lock_study(folder)
    broken = _study(
        root / "broken",
        "command",
        "runners:\n  command:\n    template: no-such-rollout-binary {policy.path}\n",
    )
    lock_study(broken)
    return root


def test_console_runs_the_command_and_suggests_the_outcome(command_root: Path) -> None:
    with TestClient(create_app(command_root), base_url="http://127.0.0.1") as client:
        console = Console(client)
        url = console.session("/studies/cmd")
        page = client.get(url).text
        console.post(f"{url}/start", {"slot_id": field(page, "slot_id")})
        running = client.get(url).text
        tid = trial_id(running)
        assert "Runner:" in running
        # Wait for the command to finish by itself, as a real rollout would before Stop.
        deadline = time.monotonic() + 10
        while "command ended (exit 0)" not in running and time.monotonic() < deadline:
            time.sleep(0.05)
            running = client.get(url).text
        assert "command ended (exit 0)" in running
        stopped = console.post(
            f"{url}/trials/{tid}/stop",
            {"version": field(running, "version"), "idempotency_key": "stop-1"},
        )
        # Exit code 0 suggests success: the success stage is preselected.
        assert re.search(r'name="stage" value="3" data-key="4"\s+checked', stopped)
    folder = command_root / "cmd"
    log = folder / "logs" / f"{tid}.log"
    assert (
        "rollout ['done', '--path=outputs/pi05_21h/checkpoints/050000/pretrained_model']"
        in log.read_text()
    )
    with open_study(folder) as ctx, ctx.db() as db:
        [event] = list_events(db, ctx.study_id, "runner_output")
        assert event.payload["trial_id"] == tid
        assert event.payload["metrics"]["exit_code"] == 0.0
        assert event.payload["log"] == str(log)


def test_a_runner_that_cannot_start_voids_the_trial(command_root: Path) -> None:
    with TestClient(create_app(command_root), base_url="http://127.0.0.1") as client:
        console = Console(client)
        url = console.session("/studies/broken")
        page = client.get(url).text
        r = client.post(
            f"{url}/start",
            data={"slot_id": field(page, "slot_id")},
            headers=console.headers,
        )
        assert r.status_code == 400
        assert "could not start" in r.text
        assert "marked invalid and rescheduled" in r.text
    with open_study(command_root / "broken") as ctx, ctx.db() as db:
        [trial] = db.scalars(select(m.Trial).where(m.Trial.study_id == ctx.study_id)).all()
        assert trial.status == "invalid"
        assert trial.invalid_reason is not None
        assert trial.invalid_reason.startswith("runner could not start")


def test_check_runners_for_commands(command_root: Path) -> None:
    runner = CliRunner()
    ok = runner.invoke(cli, ["check-runners", str(command_root / "cmd")])
    assert ok.exit_code == 0, ok.output
    assert "runs" in ok.output
    assert "policy.path" not in ok.output
    bad = runner.invoke(cli, ["check-runners", str(command_root / "broken")])
    assert bad.exit_code == 1
    assert "not found on PATH" in bad.output


# --- openpi router through the REST API --------------------------------------------------------


@pytest.fixture
def servers() -> Iterator[dict[str, FakePolicyServer]]:
    found = {"baseline": FakePolicyServer(b"A"), "q50": FakePolicyServer(b"B")}
    yield found
    for server in found.values():
        server.close()


def test_api_trials_switch_the_router(tmp_path: Path, servers: dict[str, FakePolicyServer]) -> None:
    port = _free_port()
    folder = _study(
        tmp_path / "studies" / "ab",
        "openpi_router",
        f"runners:\n  openpi_router:\n    listen: 127.0.0.1:{port}\n",
        arms_policy={k: s.url for k, s in servers.items()},
        blinding="none",
    )
    lock_study(folder)
    with TestClient(create_app(folder.parent), base_url="http://127.0.0.1") as client:
        session = client.post(
            "/api/v1/studies/ab/sessions", json={"operator": "a", "rig": "r"}, headers=API
        ).json()["session"]["session_id"]
        with connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None) as robot:
            assert robot.recv() == META
            seen = []
            for _ in range(2):
                slot = client.get("/api/v1/studies/ab/next").json()
                trial = client.post(
                    "/api/v1/studies/ab/trials",
                    json={"slot_id": slot["slot_id"], "session_id": session},
                    headers=API,
                ).json()
                robot.send(b"obs")
                reply = robot.recv()
                seen.append((slot["arm"], reply))
                client.post(
                    f"/api/v1/studies/ab/trials/{trial['trial_id']}/complete",
                    json={
                        "stage": "clean",
                        "termination": "success",
                        "expected_version": trial["version"],
                    },
                    headers=API,
                )
        tags = {"baseline": b"A:obs", "q50": b"B:obs"}
        assert all(reply == tags[arm] for arm, reply in seen)
        assert {arm for arm, _ in seen} == {"baseline", "q50"}  # block 1 runs both arms
    res = analyze_study(folder)
    assert {u.arm for u in res.runner} == {"baseline", "q50"}
    assert all(u.requests == 1 for u in res.runner)
    assert all(u.latency_ms_median is not None for u in res.runner)


def test_check_runners_for_the_router(tmp_path: Path, servers: dict[str, FakePolicyServer]) -> None:
    odd = FakePolicyServer(b"C", metadata=OTHER_META)
    try:
        same = _study(
            tmp_path / "same",
            "openpi_router",
            "",
            arms_policy={k: s.url for k, s in servers.items()},
        )
        lock_study(same)
        differ = _study(
            tmp_path / "differ",
            "openpi_router",
            "",
            arms_policy={"baseline": servers["baseline"].url, "q50": odd.url},
        )
        lock_study(differ)
        runner = CliRunner()
        ok = runner.invoke(cli, ["check-runners", str(same)])
        assert ok.exit_code == 0, ok.output
        assert "metadata are identical" in ok.output
        bad = runner.invoke(cli, ["check-runners", str(differ)])
        assert bad.exit_code == 1
        assert "metadata differ" in bad.output
    finally:
        odd.close()
