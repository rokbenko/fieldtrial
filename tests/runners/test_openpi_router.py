"""The openpi router, against fake policy servers on loopback.

The fake servers follow the protocol of openpi-client 0.1.2: one binary metadata frame on
connect, then one binary reply per binary request, and a text frame for an error. The
client side mirrors ``WebsocketClientPolicy`` (``websockets.sync.client.connect`` with
``compression=None``, ``max_size=None`` and an optional ``Authorization`` header). Frames
are opaque bytes here; the router never decodes them.
"""

import asyncio
import threading
from collections.abc import Iterator

import pytest
from websockets.asyncio.server import serve
from websockets.sync.client import connect

from fieldtrial.runners.base import ArmSpec, RunnerError, TrialContext
from fieldtrial.runners.openpi_router import OpenpiRouter

# Hand-made msgpack bytes: {"name": "pi05"} and {"name": "pi0"}.
META = b"\x81\xa4name\xa4pi05"
OTHER_META = b"\x81\xa4name\xa3pi0"


class FakePolicyServer:
    """Replies ``<tag>:<request>``; a request of b"boom" gets a text error."""

    def __init__(self, tag: bytes, metadata: bytes = META) -> None:
        self.tag = tag
        self.metadata = metadata
        self.headers: list[str | None] = []
        self.requests: list[bytes] = []
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_until_complete, args=(self._main(),))
        self._thread.daemon = True
        self._thread.start()
        self._ready.wait(5)

    async def _main(self) -> None:
        self._stop = asyncio.Event()
        async with serve(self._handle, "127.0.0.1", 0, compression=None, max_size=None) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    async def _handle(self, ws) -> None:  # type: ignore[no-untyped-def]
        self.headers.append(ws.request.headers.get("Authorization"))
        await ws.send(self.metadata)
        async for message in ws:
            self.requests.append(message)
            if message == b"boom":
                await ws.send("Traceback: something failed")
            else:
                await ws.send(self.tag + b":" + message)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(5)
        self._loop.close()


@pytest.fixture
def servers() -> Iterator[dict[str, FakePolicyServer]]:
    found = {"baseline": FakePolicyServer(b"A"), "candidate": FakePolicyServer(b"B")}
    yield found
    for server in found.values():
        server.close()


@pytest.fixture
def router(servers: dict[str, FakePolicyServer]) -> Iterator[OpenpiRouter]:
    r = OpenpiRouter({k: s.url for k, s in servers.items()}, listen="127.0.0.1:0")
    r.serve()
    yield r
    r.close()


def _arm(arm_id: str, code: str) -> ArmSpec:
    return ArmSpec(arm_id=arm_id, blind_code=code, policy={"url": "ws://unused"})


def _trial(seq: int) -> TrialContext:
    return TrialContext(seq=seq, condition="c", factors={}, trial_id=f"t{seq}")


def _client(router: OpenpiRouter, api_key: str | None = None):  # type: ignore[no-untyped-def]
    headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
    return connect(
        f"ws://127.0.0.1:{router.port}",
        compression=None,
        max_size=None,
        additional_headers=headers,
    )


def test_routes_each_trial_to_its_arm(router: OpenpiRouter) -> None:
    with _client(router) as ws:
        assert ws.recv() == META  # metadata first, as the openpi client expects
        ws.send(b"warmup")
        assert ws.recv().startswith("fieldtrial router: no arm selected")
        router.prepare(_arm("baseline", "K7"))
        router.start(_trial(1))
        for i in range(3):
            ws.send(b"obs%d" % i)
            assert ws.recv() == b"A:obs%d" % i
        first = router.stop()
        router.prepare(_arm("candidate", "Q2"))
        router.start(_trial(2))
        ws.send(b"obs")
        assert ws.recv() == b"B:obs"
        second = router.stop()
    assert first.metrics["requests"] == 3
    assert first.metrics["errors"] == 0
    assert 0 < first.metrics["latency_ms_median"] <= first.metrics["latency_ms_max"]
    assert first.metrics["latency_ms_p95"] <= first.metrics["latency_ms_max"]
    assert second.metrics["requests"] == 1


def test_api_key_passes_through(router: OpenpiRouter, servers: dict[str, FakePolicyServer]) -> None:
    with _client(router, api_key="s3cret") as ws:
        ws.recv()
    for server in servers.values():
        assert server.headers == ["Api-Key s3cret"]


def test_server_errors_reach_the_client_and_are_counted(router: OpenpiRouter) -> None:
    with _client(router) as ws:
        ws.recv()
        router.prepare(_arm("baseline", "K7"))
        router.start(_trial(1))
        ws.send(b"boom")
        reply = ws.recv()
        assert isinstance(reply, str)
        assert "Traceback" in reply
        artifacts = router.stop()
    assert artifacts.metrics["errors"] == 1
    assert "policy server error" in router.status().message


def test_arms_switch_only_between_trials(router: OpenpiRouter) -> None:
    router.prepare(_arm("baseline", "K7"))
    router.start(_trial(1))
    with pytest.raises(RunnerError, match="trial is running"):
        router.prepare(_arm("candidate", "Q2"))
    router.stop()
    with pytest.raises(RunnerError, match="no policy server"):
        router.prepare(_arm("other", "Z9"))
    with pytest.raises(RunnerError, match="no trial is running"):
        router.stop()


def test_different_metadata_is_refused(servers: dict[str, FakePolicyServer]) -> None:
    odd = FakePolicyServer(b"C", metadata=OTHER_META)
    try:
        upstreams = {"baseline": servers["baseline"].url, "odd": odd.url}
        strict = OpenpiRouter(upstreams, listen="127.0.0.1:0")
        strict.serve()
        try:
            with _client(strict) as ws:
                first = ws.recv()
            assert isinstance(first, str)
            assert "different metadata" in first
            assert strict.metadata_mismatch
        finally:
            strict.close()
        lenient = OpenpiRouter(upstreams, listen="127.0.0.1:0", metadata="first")
        lenient.serve()
        try:
            with _client(lenient) as ws:
                assert ws.recv() == META
        finally:
            lenient.close()
    finally:
        odd.close()


def test_unreachable_server_and_status() -> None:
    r = OpenpiRouter({"a": "ws://127.0.0.1:9"}, listen="127.0.0.1:0", open_timeout=2)
    assert r.status().state == "idle"
    r.serve()
    try:
        assert "waiting for the robot" in r.status().message
        with _client(r) as ws:
            message = ws.recv()
        assert isinstance(message, str)
        assert "cannot reach a policy server" in message
    finally:
        r.close()


def test_port_in_use_is_a_clear_error(router: OpenpiRouter) -> None:
    other = OpenpiRouter({"a": "ws://127.0.0.1:9"}, listen=f"127.0.0.1:{router.port}")
    with pytest.raises(RunnerError, match="could not listen"):
        other.serve()
