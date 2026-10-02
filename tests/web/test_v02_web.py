"""Interim looks and crossover rounds in the console, the REST API, the client and the CLI."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.web.conftest import API
from tests.web.test_console import Console
from typer.testing import CliRunner

from fieldtrial.cli.main import app as cli
from fieldtrial.services.simulate import simulate_study
from fieldtrial.services.study import init_study, lock_study
from fieldtrial.web.app import create_app


def _sequential(folder: Path) -> Path:
    init_study(folder, template="basic", name=folder.name)
    path = folder / "study.yaml"
    text = path.read_text().replace("range: [1, 40]", "range: [1, 20]")
    text = text.replace(
        "  stopping: {rule: fixed}", "  stopping: {rule: group_sequential, looks: 4}"
    )
    path.write_text(text)
    lock_study(folder)
    return folder


@pytest.fixture
def root(tmp_path: Path) -> Path:
    base = tmp_path / "studies"
    seq = _sequential(base / "seq")
    simulate_study(seq, {"baseline": 0.7, "q50": 0.7}, max_trials=10, interim=False)
    tray = base / "tray"
    init_study(tray, template="crossover-rounds", name="tray")
    path = tray / "study.yaml"
    path.write_text(path.read_text().replace("range: [1, 12]", "range: [1, 3]"))
    lock_study(tray)
    return base


@pytest.fixture
def client(root: Path) -> Iterator[TestClient]:
    with TestClient(create_app(root), base_url="http://127.0.0.1") as c:
        yield c


def test_api_interim(client: TestClient) -> None:
    status = client.get("/api/v1/studies/seq/interim").json()
    assert status == {
        "planned_looks": 4,
        "looks_done": 0,
        "complete_blocks": 5,
        "next_look": 1,
        "blocks_needed": 5,
        "due": True,
        "stopped_at": None,
    }
    assert client.get("/api/v1/studies/tray/interim").json() is None
    assert client.post("/api/v1/studies/seq/interim").status_code == 403  # no client header
    r = client.post("/api/v1/studies/seq/interim", headers=API)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["look"] == 1
    assert body["decision"] in ("continue", "stop")
    assert set(body) == {
        "look",
        "planned_looks",
        "decision",
        "complete_blocks",
        "voided_slots",
        "message",
    }
    again = client.post("/api/v1/studies/seq/interim", headers=API)
    assert again.status_code in (400, 409, 422)
    assert "due after 10 complete blocks" in again.text


def test_console_banner_and_interim_button(client: TestClient) -> None:
    console = Console(client)
    url = console.session("/studies/seq")
    page = client.get(url).text
    assert "Interim look 1 of 4 is due" in page
    panel = console.post(f"{url}/interim")
    assert "Interim look 1 of 4:" in panel
    assert "is due" not in panel


def test_console_crossover_rounds(client: TestClient) -> None:
    console = Console(client)
    url = console.session("/studies/tray")
    page = client.get(url).text
    assert "round 1 of 16" in page
    assert "New round:" in page


def test_cli_interim(root: Path) -> None:
    runner = CliRunner()
    seq = root / "seq"
    result = runner.invoke(cli, ["interim", str(seq)])
    assert result.exit_code == 0, result.output
    assert "Interim look 1 of 4:" in result.output
    result = runner.invoke(cli, ["interim", str(seq), "--json"])
    assert result.exit_code == 1
    assert "due after 10 complete blocks" in result.output
    result = runner.invoke(cli, ["interim", str(root / "tray")])
    assert result.exit_code == 1
    assert "no group-sequential stopping rule" in result.output
