"""Plain records that services hand to the analysis engine. No database types leak out."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One trial attempt, as recorded.

    ``status`` is ``"completed"`` or ``"invalid"`` (running trials are never analyzed).
    ``seq`` is the run-order position of the trial's slot (rescheduling shifts later slots).
    """

    trial_id: str
    seq: int
    block: int
    replicate: int
    condition: str
    arm: str
    blind_code: str
    session_id: str
    operator: str
    rig: str
    status: str
    attempt: int
    origin: str
    started_at: datetime
    stage_index: int | None = None
    success: bool | None = None
    termination: str | None = None
    duration_s: float | None = None
    failure_tags: tuple[str, ...] = ()
    notes: str | None = None
    invalid_reason: str | None = None
    runner_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StudyContextInfo:
    """Study facts the analysis needs besides the trials: provenance and deviations."""

    design_hash: str
    status: str
    locked_at: datetime | None
    unblinded_at: datetime | None
    fieldtrial_version: str
    amendments: tuple[dict[str, Any], ...] = ()
    planned_slots: int = 0
    pending_slots: int = 0
    pending_at_unblinding: int | None = None  # from the first ``unblinded`` event
    edits_after_unblinding: int = 0
    interim_looks: tuple[dict[str, Any], ...] = ()  # payloads of ``interim_look`` events
    software: dict[str, str] = field(default_factory=dict)
