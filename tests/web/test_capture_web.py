"""The console records the evaluation camera per trial and checks the rig from it."""

import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from tests.capture.test_drift import texture
from tests.services.conftest import write_small_study
from tests.web.test_console import Console, field, trial_id

from fieldtrial.capture.recorder import CaptureError, Frame
from fieldtrial.design.capture_config import CaptureConfig
from fieldtrial.services import open_study
from fieldtrial.services.events import list_events
from fieldtrial.services.rig import set_reference
from fieldtrial.services.study import lock_study
from fieldtrial.store import models as m
from fieldtrial.web.app import create_app

pytest.importorskip("av")


class SceneCamera:
    """Shows the reference texture, so camera rig checks pass."""

    def __init__(self) -> None:
        self.image = texture().astype(np.uint8)

    def read(self) -> Frame | None:
        return np.repeat(self.image[..., None], 3, axis=2)

    def close(self) -> None:
        pass


def _study(tmp_path: Path) -> Path:
    folder = write_small_study(tmp_path / "studies" / "cam")
    path = folder / "study.yaml"
    path.write_text(
        path.read_text() + "capture:\n  camera: 0\n  fps: 10\n  drift: {every_trials: 2}\n"
    )
    lock_study(folder)
    return folder


def _run_trial(client: TestClient, console: Console, url: str) -> str:
    page = client.get(url).text
    console.post(f"{url}/start", {"slot_id": field(page, "slot_id")})
    running = client.get(url).text
    tid = trial_id(running)
    assert "camera recording" in running
    time.sleep(0.4)
    stopped = console.post(
        f"{url}/trials/{tid}/stop",
        {"version": field(running, "version"), "idempotency_key": f"stop-{tid}"},
    )
    console.post(
        f"{url}/trials/{tid}/complete",
        {
            "stage": "3",
            "termination": "success",
            "version": field(stopped, "version"),
            "idempotency_key": f"done-{tid}",
        },
    )
    return tid


def test_each_trial_is_recorded_and_the_rig_checked(tmp_path: Path) -> None:
    folder = _study(tmp_path)
    with open_study(folder) as ctx:
        set_reference(ctx, _png())
    opened: list[CaptureConfig] = []

    def open_camera(config: CaptureConfig) -> SceneCamera:
        opened.append(config)
        return SceneCamera()

    app = create_app(folder.parent, open_camera=open_camera)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        console = Console(client)
        url = console.session("/studies/cam")
        first = _run_trial(client, console, url)
        second = _run_trial(client, console, url)
    assert len(opened) == 1  # the camera is opened once and kept
    assert opened[0].camera == 0
    with open_study(folder) as ctx, ctx.db() as db:
        trials = {
            t.id: t for t in db.scalars(select(m.Trial).where(m.Trial.study_id == ctx.study_id))
        }
        for tid in (first, second):
            assert trials[tid].media == [f"media/{tid}/camera.mp4"]
            assert (folder / "media" / tid / "camera.mp4").stat().st_size > 0
        checks = [e.payload for e in list_events(db, ctx.study_id, "rig_check")]
    # One check when the session started, one at trial 2 (every_trials: 2).
    assert [c["source"] for c in checks] == ["camera", "camera"]
    assert not any(c["flagged"] for c in checks)
    assert not list((folder / "media" / ".recording").glob("*.mp4"))


def test_a_broken_camera_never_blocks_a_trial(tmp_path: Path) -> None:
    folder = _study(tmp_path)

    def broken(_config: CaptureConfig) -> SceneCamera:
        raise CaptureError("cannot open camera 0")

    with TestClient(
        create_app(folder.parent, open_camera=broken), base_url="http://127.0.0.1"
    ) as client:
        console = Console(client)
        url = console.session("/studies/cam")
        page = client.get(url).text
        console.post(f"{url}/start", {"slot_id": field(page, "slot_id")})
        running = client.get(url).text
        assert "camera: recording did not start: cannot open camera 0" in running
        assert "Stop" in running


def _png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(texture().astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()
