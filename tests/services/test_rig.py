"""Rig checks through the service, the CLI, the console and the report."""

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from tests.capture.test_drift import texture
from tests.services.conftest import write_small_study
from tests.web.test_console import Console
from typer.testing import CliRunner

from fieldtrial.cli.main import app as cli
from fieldtrial.report import render_html, render_markdown
from fieldtrial.services import ServiceError, open_study
from fieldtrial.services.analysis import analyze_study
from fieldtrial.services.rig import check_rig, has_reference, rig_checks, set_reference
from fieldtrial.services.simulate import simulate_study
from fieldtrial.services.study import lock_study, unblind_study
from fieldtrial.web.app import create_app


def png(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def study(tmp_path: Path) -> Path:
    folder = write_small_study(tmp_path / "studies" / "alpha")
    lock_study(folder)
    return folder


def test_reference_and_checks_are_recorded(study: Path) -> None:
    with open_study(study) as ctx:
        assert not has_reference(ctx)
        with pytest.raises(ServiceError, match="no rig reference photo"):
            check_rig(ctx, png(texture()))
        set_reference(ctx, png(texture()))
        assert (study / "rig" / "reference.png").exists()
        same = check_rig(ctx, png(texture()), source="cli")
        moved = check_rig(ctx, png(np.roll(texture(), (0, 24), axis=(0, 1))), source="cli")
        set_reference(ctx, png(texture(seed=1)))  # the old reference is kept
        assert len(list((study / "rig").glob("reference-*.png"))) == 1
        with pytest.raises(ServiceError, match="cannot read the image"):
            check_rig(ctx, b"not an image")
        checks = rig_checks(ctx)
    assert not same.flagged
    assert moved.flagged
    assert [c["flagged"] for c in checks] == [False, True]
    assert (study / checks[1]["image"]).exists()

    simulate_study(study, {"baseline": 0.7, "q50": 0.8})
    unblind_study(study)
    res = analyze_study(study)
    assert [c.flagged for c in res.rig_checks] == [False, True]
    [deviation] = [d for d in res.deviations if d.kind == "rig_drift"]
    assert "camera or scene moved by 24 px" in deviation.message
    assert "Rig checks" in render_html(res, charts=False)
    assert "## Rig checks" in render_markdown(res)


def test_cli(study: Path, tmp_path: Path) -> None:
    ref = tmp_path / "ref.png"
    ref.write_bytes(png(texture()))
    cur = tmp_path / "cur.png"
    cur.write_bytes(png(np.clip(texture() * 1.4, 0, 255)))
    runner = CliRunner()
    out = runner.invoke(cli, ["rig-check", str(study), str(ref), "--set-reference"])
    assert out.exit_code == 0, out.output
    assert "Reference photo saved" in out.output
    out = runner.invoke(cli, ["rig-check", str(study), str(cur)])
    assert out.exit_code == 0, out.output
    assert "flagged" in out.output
    assert "brightness changed" in out.output


def test_console_session_with_a_rig_photo(study: Path) -> None:
    with open_study(study) as ctx:
        set_reference(ctx, png(texture()))
    root = study.parent
    with TestClient(create_app(root), base_url="http://127.0.0.1") as client:
        console = Console(client)
        page = client.get("/studies/alpha").text
        assert 'name="rig_photo"' in page
        assert 'hx-encoding="multipart/form-data"' in page
        r = client.post(
            "/studies/alpha/sessions",
            data={
                "operator": "ana",
                "rig": "rig-1",
                "check_0": "on",
                "check_1": "on",
                "check_2": "on",
            },
            files={
                "rig_photo": ("rig.png", png(np.roll(texture(), (30, 0), axis=(0, 1))), "image/png")
            },
            headers=console.headers,
            follow_redirects=False,
        )
        assert r.status_code == 200, r.text
        url = r.headers["hx-redirect"]
        panel = client.get(url).text
        assert "Rig check: camera or scene moved by 30 px" in panel
