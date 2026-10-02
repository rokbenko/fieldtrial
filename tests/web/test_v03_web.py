"""Anytime stopping and best-arm selection in the console, the REST API, the client and the CLI."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.web.test_console import Console
from typer.testing import CliRunner

from fieldtrial.cli.main import app as cli
from fieldtrial.services.simulate import simulate_study
from fieldtrial.services.study import init_study, lock_study
from fieldtrial.web.app import create_app

RATES = {"q30": 0.05, "q50": 0.05, "q70": 0.95, "sync": 0.95}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    base = tmp_path / "studies"
    anytime = base / "any"
    init_study(anytime, template="basic", name="any")
    path = anytime / "study.yaml"
    path.write_text(path.read_text().replace("{rule: fixed}", "{rule: anytime}"))
    lock_study(anytime)
    simulate_study(anytime, {"baseline": 0.05, "q50": 0.95}, seed=3)
    sweep = base / "sweep"
    init_study(sweep, template="best-arm", name="sweep")
    lock_study(sweep)
    simulate_study(sweep, RATES, seed=5, max_trials=120)
    init_study(base / "plain", template="basic", name="plain")
    lock_study(base / "plain")
    return base


@pytest.fixture
def client(root: Path) -> Iterator[TestClient]:
    with TestClient(create_app(root), base_url="http://127.0.0.1") as c:
        yield c


def test_api_adaptive(client: TestClient) -> None:
    anytime = client.get("/api/v1/studies/any/adaptive").json()
    assert anytime["rule"] == "anytime"
    assert anytime["stopped"] is True
    assert anytime["dropped"] == []
    sweep = client.get("/api/v1/studies/sweep/adaptive").json()
    assert sweep["rule"] == "elimination"
    assert sweep["stopped"] is False
    assert len(sweep["dropped"]) == 2
    assert sweep["remaining"] == 2
    assert not set(sweep["dropped"]) & set(RATES)  # blind codes, never arm ids
    assert client.get("/api/v1/studies/plain/adaptive").json() is None


def test_console_shows_dropped_arms_and_the_stop(client: TestClient) -> None:
    console = Console(client)
    sweep = client.get("/api/v1/studies/sweep/adaptive").json()
    url = console.session("/studies/sweep")
    panel = client.get(url).text
    assert "Best-arm selection:" in panel
    for code in sweep["dropped"]:
        assert code in panel
    assert "q30" not in panel  # the arm ids stay hidden
    done = client.get(Console(client).session("/studies/any")).text
    assert "The study stopped after" in done
    assert "anytime-valid test decided" in done


def test_client_adaptive_status(root: Path) -> None:
    import threading
    import time

    import uvicorn

    from fieldtrial.client import Client

    config = uvicorn.Config(create_app(root), host="127.0.0.1", port=0, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.01)
    try:
        port = srv.servers[0].sockets[0].getsockname()[1]
        status = Client(f"http://127.0.0.1:{port}", study="sweep").adaptive_status()
        assert status is not None
        assert status.rule == "elimination"
        assert Client(f"http://127.0.0.1:{port}", study="plain").adaptive_status() is None
    finally:
        srv.should_exit = True
        thread.join(timeout=10)


def test_cli_status_and_simulate(root: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    out = runner.invoke(cli, ["status", str(root / "sweep")])
    assert out.exit_code == 0, out.output
    assert "Best-arm selection:" in out.output
    assert "dropped" in out.output
    out = runner.invoke(cli, ["status", str(root / "any"), "--json"])
    assert out.exit_code == 0, out.output
    assert '"rule": "anytime"' in out.output
    fresh = tmp_path / "fresh"
    init_study(fresh, template="best-arm", name="fresh")
    lock_study(fresh)
    out = runner.invoke(
        cli,
        ["simulate", str(fresh), "--rates", "q30=0.05,q50=0.05,q70=0.05,sync=0.97", "--seed", "2"],
    )
    assert out.exit_code == 0, out.output
    assert "Dropped arms (blind codes):" in out.output
