"""Request and response bodies of the REST API v1 (shared with :mod:`fieldtrial.client`).

While a study is blinded, responses carry blind codes only: ``arm``, ``policy`` and
``serving`` are ``None``.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

API_VERSION = "1"
Termination = Literal[
    "success", "timeout", "stuck", "operator_stop", "robot_fault", "safety_stop", "other"
]


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Out(BaseModel):
    model_config = ConfigDict(extra="ignore")


class StudyOut(_Out):
    """A study under the served folder."""

    slug: str
    name: str | None
    title: str | None
    status: str
    locked: bool


class StatusOut(_Out):
    """Progress of a locked study. ``per_arm`` is ``None`` while blinded."""

    slug: str
    name: str
    status: str
    design_hash: str
    blinded: bool
    planned: int
    done: int
    pending: int
    invalid_trials: int
    sessions: int
    amendments: int
    stages: list[str]
    success_stage: str
    failure_tags: list[str]
    timeout_s: float | None
    per_arm: dict[str, tuple[int, int]] | None


class SlotOut(_Out):
    """A scheduled trial. ``arm``, ``policy`` and ``serving`` are hidden while blinded."""

    slot_id: str
    seq: int
    total: int
    block: int
    replicate: int
    condition: str
    factors: dict[str, Any]
    blind_code: str
    arm: str | None
    policy: dict[str, Any] | None
    serving: dict[str, Any] | None


class TrialOut(_Out):
    """A trial and its concurrency version (send it back as ``expected_version``)."""

    trial_id: str
    slot: SlotOut
    session_id: str
    status: Literal["running", "completed", "invalid"]
    attempt: int
    version: int
    started_at: datetime
    ended_at: datetime | None
    duration_s: float | None
    stage: str | None  # stage id, or None when no stage was reached or not labelled yet
    stage_index: int | None
    success: bool | None
    termination: str | None
    failure_tags: list[str]
    notes: str | None
    invalid_reason: str | None
    media: list[str]


class SessionIn(_In):
    """Start a session: one operator on one rig."""

    operator: str = Field(min_length=1, max_length=128)
    rig: str = Field(min_length=1, max_length=128)
    rig_check: dict[str, bool] = Field(default_factory=dict)
    notes: str | None = None


class SessionOut(_Out):
    """A session."""

    session_id: str
    operator: str
    rig: str
    started_at: datetime
    ended_at: datetime | None


class ConsoleOut(_Out):
    """What a runtime or console needs for one session: the running trial or the next slot."""

    session: SessionOut
    running: TrialOut | None
    up_next: SlotOut | None
    undoable: TrialOut | None
    done: int
    total: int


class StartIn(_In):
    """Start a trial for a slot (normally ``up_next``)."""

    slot_id: str
    session_id: str


class StopIn(_In):
    """Stop the clock on a running trial (it is labelled afterwards)."""

    expected_version: int | None = None


class CompleteIn(_In):
    """The outcome of a running trial.

    ``stage`` is the furthest stage reached: a stage id, or ``None`` for no stage. Success
    follows from reaching the study's success stage.
    """

    stage: str | None
    termination: Termination
    failure_tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    duration_s: float | None = Field(default=None, ge=0)
    expected_version: int | None = None


class InvalidateIn(_In):
    """Void a trial (robot fault, setup error) and reschedule it."""

    reason: str = Field(min_length=1)
    reschedule: Literal["block", "end"] = "block"
    expected_version: int | None = None


class UndoIn(_In):
    """Undo the latest trial of a session (within 10 s)."""

    session_id: str | None = None
    expected_version: int | None = None


class EditIn(_In):
    """Correct a completed trial's label. ``reason`` and ``actor`` are logged."""

    actor: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1)
    stage: str | None = None
    clear_stage: bool = False  # set the stage to "none reached"
    termination: Termination | None = None
    failure_tags: list[str] | None = None
    notes: str | None = None
    expected_version: int | None = None


class MediaOut(_Out):
    """Where an attached file was stored, relative to the study folder."""

    path: str


class ErrorOut(_Out):
    """An error. ``detail`` explains it."""

    detail: str
