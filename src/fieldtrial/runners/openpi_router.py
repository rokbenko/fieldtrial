"""The ``openpi_router`` runner: blinded A/B testing for openpi-style policy servers.

The robot's openpi client connects to the router instead of a policy server. For each
client connection the router opens one upstream connection per arm, checks that every
arm's server sent the same metadata, and forwards the robot's frames verbatim to the arm
of the current trial. It never decodes a frame, so it depends only on ``websockets``.

The protocol, as implemented by ``openpi-client`` 0.1.2 (``websocket_client_policy.py``):
the client connects (optionally with ``Authorization: Api-Key ...``), the server sends one
msgpack metadata frame, then each binary request frame gets one binary reply frame. A text
frame is an error message. The router sends its own errors as text frames, so the client
raises them as server errors.

Arms switch only between trials. The router records each request's round-trip time per
trial; :meth:`OpenpiRouter.stop` reports them as runner metrics.

Requires the ``openpi`` extra (``pip install 'fieldtrial[openpi]'``).
"""

import asyncio
import contextlib
import statistics
import threading
import time
from collections.abc import Mapping
from typing import Any, Literal

from fieldtrial.runners.base import (
    ArmSpec,
    RunArtifacts,
    RunnerCapabilities,
    RunnerError,
    RunnerStatus,
    Termination,
    TrialContext,
)

PREFIX = "fieldtrial router: "


def _websockets() -> Any:
    try:
        import websockets.asyncio.client
        import websockets.asyncio.server
        import websockets.exceptions
    except ImportError as exc:  # pragma: no cover - the extra is installed in tests
        raise RunnerError(
            "the openpi router needs the openpi extra: pip install 'fieldtrial[openpi]'"
        ) from exc
    return websockets


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class OpenpiRouter:
    """A websocket proxy that sends each trial's requests to that trial's arm.

    Parameters
    ----------
    upstreams
        Policy-server URL per arm id, in design order (the first is used for
        ``metadata: first``).
    listen
        ``host:port`` for the robot's client. Port 0 picks a free port (see ``port``).
    metadata
        ``identical`` refuses clients when the arms' servers send different metadata;
        ``first`` forwards the first arm's metadata and records the mismatch.
    """

    capabilities = RunnerCapabilities(can_switch_arms=True, can_stop=False)

    def __init__(
        self,
        upstreams: Mapping[str, str],
        *,
        listen: str = "127.0.0.1:8000",
        metadata: Literal["identical", "first"] = "identical",
        open_timeout: float = 10.0,
    ) -> None:
        if not upstreams:
            raise RunnerError("the router needs at least one arm")
        self.upstreams = dict(upstreams)
        host, _, port = listen.rpartition(":")
        self.host, self._port = host, int(port)
        self.metadata = metadata
        self.open_timeout = open_timeout
        self._lock = threading.Lock()
        self._active: str | None = None
        self._trial: TrialContext | None = None
        self._started = 0.0
        self._latencies: list[float] = []
        self._errors = 0
        self._clients = 0
        self._last_error = ""
        self.metadata_mismatch = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._failed: BaseException | None = None

    # --- server ---------------------------------------------------------------------------

    @property
    def port(self) -> int:
        """The port the router listens on (after :meth:`serve`)."""
        return self._port

    def serve(self) -> None:
        """Start listening in a background thread. Returns once the socket is bound."""
        if self._thread is not None:
            return
        _websockets()
        self._thread = threading.Thread(target=self._run, name="openpi-router", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)
        if self._failed is not None:
            raise RunnerError(
                f"the router could not listen on {self.host}:{self._port}: {self._failed}"
            ) from self._failed

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._main())
        except BaseException as exc:
            self._failed = exc
            self._ready.set()
        finally:
            loop.close()

    async def _main(self) -> None:
        ws = _websockets()
        self._stop = asyncio.Event()
        async with ws.asyncio.server.serve(
            self._handle, self.host, self._port, compression=None, max_size=None
        ) as server:
            self._port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    async def _handle(self, client: Any) -> None:
        ws = _websockets()
        headers = {}
        request = getattr(client, "request", None)
        if request is not None and request.headers.get("Authorization"):
            headers["Authorization"] = request.headers["Authorization"]
        upstream: dict[str, Any] = {}
        with self._lock:
            self._clients += 1
        try:
            try:
                for arm, url in self.upstreams.items():
                    upstream[arm] = await ws.asyncio.client.connect(
                        url,
                        additional_headers=headers or None,
                        compression=None,
                        max_size=None,
                        open_timeout=self.open_timeout,
                    )
                metadata = {arm: await conn.recv() for arm, conn in upstream.items()}
            except (OSError, TimeoutError, ws.exceptions.WebSocketException) as exc:
                await self._refuse(client, f"cannot reach a policy server: {exc}")
                return
            frames = list(metadata.values())
            if any(frame != frames[0] for frame in frames[1:]):
                self.metadata_mismatch = True
                if self.metadata == "identical":
                    await self._refuse(
                        client,
                        "the arms' policy servers sent different metadata, so the robot "
                        "could tell the arms apart. Make them identical, or set "
                        "runners.openpi_router.metadata: first.",
                    )
                    return
            await client.send(frames[0])
            async for message in client:
                target = self._active
                if target is None:
                    await client.send(PREFIX + "no arm selected yet; start a trial first")
                    continue
                conn = upstream[target]
                began = time.perf_counter()
                try:
                    await conn.send(message)
                    reply = await conn.recv()
                except ws.exceptions.ConnectionClosed as exc:
                    self._record(None, error=f"policy server closed the connection: {exc}")
                    await client.send(PREFIX + "the policy server closed the connection")
                    return
                self._record(
                    time.perf_counter() - began,
                    error="policy server error" if isinstance(reply, str) else None,
                )
                await client.send(reply)
        except ws.exceptions.ConnectionClosed:
            pass
        finally:
            for conn in upstream.values():
                with contextlib.suppress(Exception):
                    await conn.close()
            with self._lock:
                self._clients -= 1

    async def _refuse(self, client: Any, message: str) -> None:
        self._record(None, error=message)
        with contextlib.suppress(Exception):
            await client.send(PREFIX + message)
            await client.close()

    def _record(self, seconds: float | None, *, error: str | None = None) -> None:
        with self._lock:
            if seconds is not None and self._trial is not None:
                self._latencies.append(seconds)
            if error is not None:
                self._last_error = error
                if self._trial is not None:
                    self._errors += 1

    # --- Runner protocol --------------------------------------------------------------------

    def prepare(self, arm: ArmSpec) -> None:
        """Send the robot's requests to ``arm`` from now on (only between trials)."""
        if arm.arm_id not in self.upstreams:
            raise RunnerError(f"no policy server for arm {arm.blind_code}")
        with self._lock:
            if self._trial is not None:
                raise RunnerError("a trial is running; stop it before switching arms")
            self._active = arm.arm_id

    def start(self, trial: TrialContext) -> None:
        """Begin recording request latency for this trial."""
        with self._lock:
            if self._active is None:
                raise RunnerError("call prepare() first")
            if self._trial is not None:
                raise RunnerError("a trial is already running")
            self._trial = trial
            self._started = time.monotonic()
            self._latencies = []
            self._errors = 0

    def stop(self, reason: Termination = "operator_stop") -> RunArtifacts:
        """End the trial and report its request count and latency (milliseconds)."""
        with self._lock:
            if self._trial is None:
                raise RunnerError("no trial is running")
            elapsed = time.monotonic() - self._started
            ms = [x * 1000 for x in self._latencies]
            metrics: dict[str, float] = {"requests": float(len(ms)), "errors": float(self._errors)}
            if ms:
                metrics.update(
                    {
                        "latency_ms_median": statistics.median(ms),
                        "latency_ms_p95": _percentile(ms, 0.95),
                        "latency_ms_max": max(ms),
                    }
                )
            self._trial = None
        return RunArtifacts(duration_s=elapsed, termination=reason, metrics=metrics)

    def status(self) -> RunnerStatus:
        """Whether a robot client is connected, and the latest error."""
        with self._lock:
            if self._trial is not None:
                state: Literal["idle", "loading", "ready", "running", "closed"] = "running"
            elif self._thread is None:
                state = "idle"
            else:
                state = "ready"
            clients = self._clients
            error = self._last_error
        where = f"router on {self.host}:{self._port}"
        message = f"{where}, robot connected" if clients else f"{where}, waiting for the robot"
        if error:
            message += f"; last error: {error}"
        return RunnerStatus(state, message)

    def close(self) -> None:
        """Stop listening and close every connection."""
        if self._loop is not None and self._stop is not None and not self._loop.is_closed():
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._thread = None


def probe(url: str, *, api_key: str | None = None, timeout: float = 5.0) -> bytes:
    """Connect to a policy server like the openpi client does and return its metadata frame."""
    ws = _websockets()
    import websockets.sync.client

    headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
    try:
        with websockets.sync.client.connect(
            url,
            compression=None,
            max_size=None,
            additional_headers=headers,
            open_timeout=timeout,
        ) as conn:
            frame = conn.recv(timeout=timeout)
    except (OSError, TimeoutError, ws.exceptions.WebSocketException) as exc:
        raise RunnerError(f"cannot reach {url}: {exc}") from exc
    if isinstance(frame, str):
        raise RunnerError(f"{url} sent an error instead of metadata: {frame[:200]}")
    return bytes(frame)
