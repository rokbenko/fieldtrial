"""``fieldtrial serve``: URLs, QR code and how uvicorn is started."""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from fieldtrial.cli import serve as serve_module
from fieldtrial.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _fixed_lan_address(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real lookup "connects" a UDP socket, which the test network guard forbids.
    monkeypatch.setattr(serve_module, "lan_address", lambda: "192.168.1.20")


class _FakeSocket:
    def __init__(self, address: str | None) -> None:
        self.address = address
        self.closed = False

    def connect(self, _target: object) -> None:
        if self.address is None:
            raise OSError("no route")

    def getsockname(self) -> tuple[str, int]:
        assert self.address is not None
        return (self.address, 50000)

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(("address", "expected"), [("10.0.0.7", "10.0.0.7"), (None, "127.0.0.1")])
def test_lan_address(monkeypatch: pytest.MonkeyPatch, address: str | None, expected: str) -> None:
    monkeypatch.undo()  # use the real lan_address with a fake socket
    fake = _FakeSocket(address)
    monkeypatch.setattr(serve_module.socket, "socket", lambda *_args: fake)
    assert serve_module.lan_address() == expected
    assert fake.closed


def test_console_url_and_qr(tmp_path: Path) -> None:
    assert serve_module.console_url(tmp_path, lan=False, port=8765, token=None) == (
        "http://127.0.0.1:8765/"
    )
    url = serve_module.console_url(tmp_path, lan=True, port=9000, token="abc")
    assert url == "http://192.168.1.20:9000/?token=abc"
    qr = serve_module.qr_text(url)
    rows = qr.splitlines()
    assert len(rows) >= 10  # two QR rows per text line
    assert len({len(r) for r in rows}) == 1  # a rectangle
    assert set(rows[0]) == {"█"}  # quiet zone (inverted for dark terminals)


def test_serve_starts_uvicorn(studies: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.append(kw))
    result = runner.invoke(app, ["serve", str(studies)])
    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:8765/" in result.output
    assert calls == [{"host": "127.0.0.1", "port": 8765, "log_level": "warning"}]
    result = runner.invoke(app, ["serve", str(studies), "--lan", "--port", "9100"])
    assert result.exit_code == 0
    assert "?token=" in result.output
    assert "Anyone with this URL" in result.output
    assert calls[1]["host"] == "0.0.0.0"


def test_serve_missing_folder(tmp_path: Path) -> None:
    result = runner.invoke(app, ["serve", str(tmp_path / "missing")])
    assert result.exit_code == 1
    assert "is not a folder" in result.output
