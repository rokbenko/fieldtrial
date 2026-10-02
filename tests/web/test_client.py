"""fieldtrial.client against a real server on 127.0.0.1 (loopback only, as tests allow)."""

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn

from fieldtrial.client import ApiError, Client
from fieldtrial.web.app import create_app


@pytest.fixture
def server(studies: Path) -> Iterator[str]:
    """A live server in a thread; yields its base URL."""
    config = uvicorn.Config(create_app(studies), host="127.0.0.1", port=0, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        if time.monotonic() > deadline:  # pragma: no cover
            raise RuntimeError("server did not start")
        time.sleep(0.01)
    port = srv.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=10)


def test_client_runs_a_study(server: str, tmp_path: Path) -> None:
    api = Client(server, study="alpha")
    assert [s.slug for s in api.studies()] == ["alpha", "draft", "open"]
    assert api.status().blinded
    state = api.start_session(operator="ana", rig="rig-1", rig_check={"Lamp on": True})
    session_id = state.session.session_id
    for _ in range(state.total):
        slot = api.session(session_id).up_next
        assert slot is not None
        assert slot.arm is None  # blinded
        trial = api.start_trial(slot.slot_id, session_id)
        api.complete_trial(
            trial.trial_id, stage="clean", termination="success", expected_version=trial.version
        )
    assert api.next() is None
    assert api.session(session_id).done == 8
    last = api.trials(session_id=session_id, limit=1)[0]
    undone = api.undo(last.trial_id, session_id=session_id)
    assert undone.status == "running"
    assert api.invalidate_trial(last.trial_id, "fault", expected_version=undone.version).status == (
        "invalid"
    )
    clip = tmp_path / 'clip "1".mp4'
    clip.write_bytes(b"\x00\x01\x02")
    path = api.attach_media(last.trial_id, clip, actor="ana")
    assert api.trial(last.trial_id).media == [path]
    replacement = api.session(session_id).up_next
    assert replacement is not None
    retry = api.start_trial(replacement.slot_id, session_id)
    api.complete_trial(retry.trial_id, stage=None, termination="stuck")
    assert api.end_session(session_id).ended_at is not None


def test_client_errors(server: str) -> None:
    with pytest.raises(ApiError) as info:
        Client(server, study="nope").status()
    assert info.value.status == 404
    assert "no study" in info.value.detail
    api = Client(server, study="alpha")
    with pytest.raises(ApiError) as info:
        api.complete_trial("nope", stage="teleported", termination="success")
    assert info.value.status == 400


def test_client_connection_error() -> None:
    from urllib.error import URLError

    with pytest.raises(URLError):
        Client("http://127.0.0.1:9", study="x", timeout=1).studies()
