"""A small REST client for custom robot runtimes (a ROS 2 node, a rollout script).

Standard library only, so it adds no dependencies to your runtime. Every write sends an
idempotency key and is retried once on a connection error, so a flaky network never
records a trial twice.

>>> from fieldtrial.client import Client
>>> api = Client("http://127.0.0.1:8765", study="my-study")      # doctest: +SKIP
>>> state = api.start_session(operator="ana", rig="rig-1")        # doctest: +SKIP
>>> slot = state.up_next                                          # doctest: +SKIP
>>> trial = api.start_trial(slot.slot_id, state.session.session_id)  # doctest: +SKIP
>>> api.complete_trial(trial.trial_id, stage="clean", termination="success")  # doctest: +SKIP
"""

import json
import mimetypes
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, TypeAdapter

from fieldtrial.api.schemas import (
    ConsoleOut,
    InterimOut,
    InterimStatusOut,
    MediaOut,
    SessionOut,
    SlotOut,
    StatusOut,
    StudyOut,
    TrialOut,
)

M = TypeVar("M", bound=BaseModel)
CLIENT_HEADER = "X-Fieldtrial-Client"


class ApiError(RuntimeError):
    """The server refused a request. ``status`` is the HTTP status, ``detail`` its reason."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"{status}: {detail}")


class Client:
    """Talk to ``fieldtrial serve``.

    Parameters
    ----------
    base_url
        For example ``http://127.0.0.1:8765``.
    study
        The study's folder name (its slug), as listed by :meth:`studies`.
    token
        The access token printed by ``fieldtrial serve --lan``; not needed on 127.0.0.1.
    timeout
        Seconds to wait for each response.
    """

    def __init__(
        self, base_url: str, study: str, *, token: str | None = None, timeout: float = 30.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.study = study
        self.token = token
        self.timeout = timeout

    # --- plumbing ------------------------------------------------------------------------

    def _url(self, path: str, query: Mapping[str, Any] | None = None) -> str:
        url = f"{self.base_url}/api/v1{path}"
        params = {k: v for k, v in (query or {}).items() if v is not None}
        return f"{url}?{urlencode(params)}" if params else url

    def _study_path(self, rest: str = "") -> str:
        return f"/studies/{quote(self.study, safe='')}{rest}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        raw: bytes | None = None,
        content_type: str = "application/json",
        query: Mapping[str, Any] | None = None,
        idempotent: bool = False,
    ) -> Any:
        headers = {CLIENT_HEADER: "python", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = raw
        if body is not None:
            data = json.dumps(body).encode()
        if data is not None:
            headers["Content-Type"] = content_type
        if idempotent:
            headers["Idempotency-Key"] = uuid.uuid4().hex
        attempts = 2 if idempotent or method == "GET" else 1
        for attempt in range(attempts):
            request = Request(self._url(path, query), data=data, headers=headers, method=method)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                return json.loads(payload) if payload else None
            except HTTPError as exc:
                text = exc.read().decode("utf-8", "replace")
                try:
                    detail = str(json.loads(text).get("detail", text))
                except (ValueError, AttributeError):
                    detail = text
                raise ApiError(exc.code, detail) from None
            except URLError:
                if attempt + 1 == attempts:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _as(model: type[M], data: Any) -> M:
        return model.model_validate(data)

    # --- reading -------------------------------------------------------------------------

    def studies(self) -> list[StudyOut]:
        """Studies served by this server."""
        data = self._request("GET", "/studies")
        return TypeAdapter(list[StudyOut]).validate_python(data)

    def status(self) -> StatusOut:
        """Progress of the study (no per-arm results while blinded)."""
        return self._as(StatusOut, self._request("GET", self._study_path()))

    def next(self) -> SlotOut | None:
        """The next slot to run, or None when every slot is done or running."""
        data = self._request("GET", self._study_path("/next"))
        return None if data is None else self._as(SlotOut, data)

    def session(self, session_id: str) -> ConsoleOut:
        """A session's running trial, next slot and progress."""
        return self._as(
            ConsoleOut, self._request("GET", self._study_path(f"/sessions/{session_id}"))
        )

    def trial(self, trial_id: str) -> TrialOut:
        """One trial."""
        return self._as(TrialOut, self._request("GET", self._study_path(f"/trials/{trial_id}")))

    def trials(self, *, session_id: str | None = None, limit: int = 200) -> list[TrialOut]:
        """Trials, newest first."""
        data = self._request(
            "GET", self._study_path("/trials"), query={"session_id": session_id, "limit": limit}
        )
        return TypeAdapter(list[TrialOut]).validate_python(data)

    # --- writing -------------------------------------------------------------------------

    def start_session(
        self,
        *,
        operator: str,
        rig: str,
        rig_check: Mapping[str, bool] | None = None,
        notes: str | None = None,
    ) -> ConsoleOut:
        """Start a session; the result's ``up_next`` is the first slot to run."""
        body = {
            "operator": operator,
            "rig": rig,
            "rig_check": dict(rig_check or {}),
            "notes": notes,
        }
        data = self._request("POST", self._study_path("/sessions"), body=body, idempotent=True)
        return self._as(ConsoleOut, data)

    def interim_status(self) -> InterimStatusOut | None:
        """Interim-look status of a group-sequential study (None for other studies)."""
        data = self._request("GET", self._study_path("/interim"))
        return None if data is None else self._as(InterimStatusOut, data)

    def run_interim(self) -> InterimOut:
        """Run the interim look that is due (not retried: a look is not idempotent)."""
        return self._as(InterimOut, self._request("POST", self._study_path("/interim"), body={}))

    def end_session(self, session_id: str) -> SessionOut:
        """End a session."""
        data = self._request(
            "POST", self._study_path(f"/sessions/{session_id}/end"), body={}, idempotent=True
        )
        return self._as(SessionOut, data)

    def start_trial(self, slot_id: str, session_id: str) -> TrialOut:
        """Start the trial for a slot."""
        body = {"slot_id": slot_id, "session_id": session_id}
        data = self._request("POST", self._study_path("/trials"), body=body, idempotent=True)
        return self._as(TrialOut, data)

    def stop_trial(self, trial_id: str, *, expected_version: int | None = None) -> TrialOut:
        """Stop the clock on a running trial; label it afterwards with :meth:`complete_trial`."""
        data = self._request(
            "POST",
            self._study_path(f"/trials/{trial_id}/stop"),
            body={"expected_version": expected_version},
            idempotent=True,
        )
        return self._as(TrialOut, data)

    def complete_trial(
        self,
        trial_id: str,
        *,
        stage: str | None,
        termination: str,
        failure_tags: list[str] | None = None,
        notes: str | None = None,
        duration_s: float | None = None,
        expected_version: int | None = None,
    ) -> TrialOut:
        """Record a running trial's outcome: the furthest ``stage`` reached (None for none)."""
        body = {
            "stage": stage,
            "termination": termination,
            "failure_tags": list(failure_tags or []),
            "notes": notes,
            "duration_s": duration_s,
            "expected_version": expected_version,
        }
        data = self._request(
            "POST", self._study_path(f"/trials/{trial_id}/complete"), body=body, idempotent=True
        )
        return self._as(TrialOut, data)

    def invalidate_trial(
        self,
        trial_id: str,
        reason: str,
        *,
        reschedule: str = "block",
        expected_version: int | None = None,
    ) -> TrialOut:
        """Void a trial (robot fault, setup error); its slot is rescheduled."""
        body = {"reason": reason, "reschedule": reschedule, "expected_version": expected_version}
        data = self._request(
            "POST", self._study_path(f"/trials/{trial_id}/invalidate"), body=body, idempotent=True
        )
        return self._as(TrialOut, data)

    def undo(self, trial_id: str, *, session_id: str | None = None) -> TrialOut:
        """Undo the latest trial's outcome within 10 s; the trial is running again."""
        data = self._request(
            "POST",
            self._study_path(f"/trials/{trial_id}/undo"),
            body={"session_id": session_id},
            idempotent=True,
        )
        return self._as(TrialOut, data)

    def attach_media(self, trial_id: str, path: str | Path, *, actor: str = "client") -> str:
        """Upload a clip or photo for a trial; returns where it is stored in the study folder."""
        file = Path(path)
        boundary = uuid.uuid4().hex
        kind = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        safe_name = file.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{safe_name}"\r\nContent-Type: {kind}\r\n\r\n'
        ).encode()
        raw = head + file.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        data = self._request(
            "POST",
            self._study_path(f"/trials/{trial_id}/media"),
            raw=raw,
            content_type=f"multipart/form-data; boundary={boundary}",
            query={"actor": actor},
            idempotent=True,
        )
        return self._as(MediaOut, data).path
