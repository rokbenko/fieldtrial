"""REST API v1: every route, blinding, errors, idempotency, SSE and the security rules."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.web.conftest import API

from fieldtrial.web.app import create_app


def _session(client: TestClient, study: str = "alpha") -> dict:  # type: ignore[type-arg]
    r = client.post(
        f"/api/v1/studies/{study}/sessions",
        json={"operator": "ana", "rig": "rig-1", "rig_check": {"Lamp on": True}},
        headers=API,
    )
    assert r.status_code == 201, r.text
    return r.json()  # type: ignore[no-any-return]


def test_studies_and_status(client: TestClient) -> None:
    r = client.get("/api/v1/studies")
    assert [(s["slug"], s["status"]) for s in r.json()] == [
        ("alpha", "locked"),
        ("draft", "draft"),
        ("open", "locked"),
    ]
    status = client.get("/api/v1/studies/alpha").json()
    assert status["blinded"] is True
    assert status["per_arm"] is None
    assert status["planned"] == 8
    assert status["stages"] == ["lift", "handover", "inserted", "clean"]
    assert client.get("/api/v1/studies/open").json()["per_arm"] == {
        "baseline": [0, 0],
        "q50": [0, 0],
    }
    assert client.get("/api/v1/studies/nope").status_code == 404
    assert client.get("/api/v1/studies/..").status_code == 404
    draft = client.get("/api/v1/studies/draft")
    assert draft.status_code == 400
    assert "not locked" in draft.json()["detail"]


def test_next_hides_arms_while_blinded(client: TestClient) -> None:
    blind = client.get("/api/v1/studies/alpha/next").json()
    assert blind["seq"] == 1
    assert (blind["arm"], blind["policy"], blind["serving"]) == (None, None, None)
    assert len(blind["blind_code"]) == 2
    visible = client.get("/api/v1/studies/open/next").json()
    assert visible["arm"] in ("baseline", "q50")
    assert "inference.queue_threshold" in visible["serving"]


def test_trial_flow(client: TestClient) -> None:
    state = _session(client)
    session_id = state["session"]["session_id"]
    slot = state["up_next"]
    assert state["running"] is None
    assert (state["done"], state["total"]) == (0, 8)

    r = client.post(
        "/api/v1/studies/alpha/trials",
        json={"slot_id": slot["slot_id"], "session_id": session_id},
        headers={**API, "Idempotency-Key": "start-1"},
    )
    assert r.status_code == 201
    trial = r.json()
    again = client.post(
        "/api/v1/studies/alpha/trials",
        json={"slot_id": slot["slot_id"], "session_id": session_id},
        headers={**API, "Idempotency-Key": "start-1"},
    )
    assert again.json()["trial_id"] == trial["trial_id"]
    assert trial["status"] == "running"
    assert trial["slot"]["arm"] is None

    state = client.get(f"/api/v1/studies/alpha/sessions/{session_id}").json()
    assert state["running"]["trial_id"] == trial["trial_id"]
    assert state["up_next"] is None

    done = client.post(
        f"/api/v1/studies/alpha/trials/{trial['trial_id']}/complete",
        json={"stage": "clean", "termination": "success", "expected_version": 1},
        headers=API,
    ).json()
    assert (done["status"], done["stage"], done["success"], done["version"]) == (
        "completed",
        "clean",
        True,
        2,
    )
    stale = client.post(
        f"/api/v1/studies/alpha/trials/{trial['trial_id']}/invalidate",
        json={"reason": "fault", "expected_version": 1},
        headers=API,
    )
    assert stale.status_code == 409

    undone = client.post(
        f"/api/v1/studies/alpha/trials/{trial['trial_id']}/undo",
        json={"session_id": session_id},
        headers=API,
    ).json()
    assert (undone["status"], undone["stage"]) == ("running", None)
    client.post(
        f"/api/v1/studies/alpha/trials/{trial['trial_id']}/complete",
        json={"stage": None, "termination": "stuck", "failure_tags": ["dropped"]},
        headers=API,
    )
    edited = client.patch(
        f"/api/v1/studies/alpha/trials/{trial['trial_id']}",
        json={"actor": "bo", "reason": "video review", "stage": "lift", "notes": "slipped"},
        headers=API,
    ).json()
    assert (edited["stage"], edited["notes"], edited["success"]) == ("lift", "slipped", False)
    cleared = client.patch(
        f"/api/v1/studies/alpha/trials/{trial['trial_id']}",
        json={"actor": "bo", "reason": "really none", "clear_stage": True},
        headers=API,
    ).json()
    assert cleared["stage"] is None
    both = client.patch(
        f"/api/v1/studies/alpha/trials/{trial['trial_id']}",
        json={"actor": "bo", "reason": "x", "clear_stage": True, "stage": "lift"},
        headers=API,
    )
    assert both.status_code == 400

    media = client.post(
        f"/api/v1/studies/alpha/trials/{trial['trial_id']}/media",
        files={"file": ("clip.mp4", b"\x00\x01", "video/mp4")},
        params={"actor": "ana"},
        headers=API,
    )
    assert media.status_code == 201
    assert media.json()["path"].endswith("/clip.mp4")

    nxt = client.get(f"/api/v1/studies/alpha/sessions/{session_id}").json()["up_next"]
    second = client.post(
        "/api/v1/studies/alpha/trials",
        json={"slot_id": nxt["slot_id"], "session_id": session_id},
        headers=API,
    ).json()
    invalid = client.post(
        f"/api/v1/studies/alpha/trials/{second['trial_id']}/invalidate",
        json={"reason": "camera fell", "reschedule": "end"},
        headers=API,
    ).json()
    assert invalid["status"] == "invalid"

    listed = client.get("/api/v1/studies/alpha/trials", params={"session_id": session_id}).json()
    assert [t["trial_id"] for t in listed] == [second["trial_id"], trial["trial_id"]]
    one = client.get(f"/api/v1/studies/alpha/trials/{trial['trial_id']}").json()
    assert one["media"] == [media.json()["path"]]

    ended = client.post(f"/api/v1/studies/alpha/sessions/{session_id}/end", json={}, headers=API)
    assert ended.status_code == 200
    assert ended.json()["ended_at"] is not None


def test_validation_and_service_errors(client: TestClient) -> None:
    state = _session(client)
    bad_stage = client.post(
        "/api/v1/studies/alpha/trials/nope/complete",
        json={"stage": "teleported", "termination": "success"},
        headers=API,
    )
    assert bad_stage.status_code == 400
    assert "unknown stage" in bad_stage.json()["detail"]
    assert (
        client.post(
            "/api/v1/studies/alpha/trials",
            json={"slot_id": "x", "session_id": state["session"]["session_id"], "extra": 1},
            headers=API,
        ).status_code
        == 422
    )
    assert client.get("/api/v1/studies/alpha/trials/nope").status_code == 400
    long_key = client.post(
        "/api/v1/studies/alpha/sessions",
        json={"operator": "a", "rig": "r"},
        headers={**API, "Idempotency-Key": "k" * 65},
    )
    assert long_key.status_code == 422


def test_events_stream(client: TestClient) -> None:
    first = client.get("/api/v1/studies/alpha/events", params={"once": True})
    assert first.status_code == 200
    assert first.text.strip() == ""
    anchor = client.get("/api/v1/studies/alpha/trials").json()
    assert anchor == []
    # Everything after the design_locked event: one session start.
    with TestClient(client.app, base_url="http://127.0.0.1") as other:
        _session(other)
        from fieldtrial.services.registry import StudyRegistry

        registry: StudyRegistry = other.app.state.registry  # type: ignore[attr-defined]
        from fieldtrial.services.events import events_after

        locked = events_after(registry.get("alpha"), None)[0].id
    r = client.get("/api/v1/studies/alpha/events", params={"once": True, "after": locked})
    lines = [line for line in r.text.splitlines() if line.startswith("data:")]
    assert [json.loads(line[5:])["kind"] for line in lines] == ["session_started"]
    assert "event: changed" in r.text
    ids = [line[3:].strip() for line in r.text.splitlines() if line.startswith("id:")]
    r = client.get(
        "/api/v1/studies/alpha/events", params={"once": True}, headers={"Last-Event-ID": ids[-1]}
    )
    assert "data:" not in r.text


def test_security_headers_and_docs(client: TestClient) -> None:
    r = client.get("/api/v1/studies")
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp
    assert "unsafe" not in csp
    assert r.headers["x-frame-options"] == "DENY"
    assert "fieldtrial_csrf=" in r.headers["set-cookie"]
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/api/v1/openapi.json").json()["info"]["version"] == "1"


def test_api_writes_need_client_header(client: TestClient) -> None:
    r = client.post("/api/v1/studies/alpha/sessions", json={"operator": "a", "rig": "r"})
    assert r.status_code == 403
    assert "x-fieldtrial-client" in r.text


def test_local_mode_rejects_foreign_hosts(studies: Path) -> None:
    with TestClient(create_app(studies), base_url="http://evil.example") as c:
        assert c.get("/api/v1/studies").status_code == 403
    with TestClient(create_app(studies), base_url="http://localhost:8765") as c:
        assert c.get("/api/v1/studies").status_code == 200
    with TestClient(create_app(studies), base_url="http://[::1]:8765") as c:
        assert c.get("/api/v1/studies").status_code == 200


def test_lan_mode_requires_token(studies: Path) -> None:
    app = create_app(studies, lan_token="s3cret")
    with TestClient(app, base_url="http://192.168.1.20:8765") as c:
        assert c.get("/api/v1/studies").status_code == 401
        assert c.get("/api/v1/studies", headers={"Authorization": "Bearer nope"}).status_code == 401
        bearer = c.get("/api/v1/studies", headers={"Authorization": "Bearer s3cret"})
        assert bearer.status_code == 200
        r = c.get("/api/v1/studies?token=s3cret&x=1", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/api/v1/studies?x=1"
        assert "HttpOnly" in r.headers["set-cookie"]
        assert "s3cret" not in r.headers["location"]
        c.cookies.set("fieldtrial_token", "s3cret")
        assert c.get("/api/v1/studies").status_code == 200


def test_telemetry_is_off(studies: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.invalid:4318")
    app = create_app(studies)
    assert app._native_telemetry.enabled() is False  # type: ignore[attr-defined]
    with TestClient(app, base_url="http://127.0.0.1") as c:
        assert c.get("/api/v1/studies").status_code == 200


def test_published_openapi_is_current(studies: Path) -> None:
    published = Path(__file__).parents[2] / "docs" / "reference" / "openapi.json"
    current = create_app(studies).openapi()
    assert json.loads(published.read_text(encoding="utf-8")) == current, (
        "docs/reference/openapi.json is stale; regenerate it with "
        "`uv run python scripts/export_openapi.py`"
    )
