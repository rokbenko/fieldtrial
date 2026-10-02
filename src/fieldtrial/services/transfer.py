"""Importing trials from CSV and exporting them to CSV or JSON Lines."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal

from sqlalchemy import select

from fieldtrial.io.csv import ImportRow, read_import_csv, write_trials_csv
from fieldtrial.io.jsonl import write_trials_jsonl
from fieldtrial.services._context import ServiceError, StudyContext, open_study
from fieldtrial.services.trial import (
    collect_records,
    end_session,
    pending_slots,
    record_trial,
    start_session,
)
from fieldtrial.store import models as m
from fieldtrial.store.models import utcnow


@dataclass(frozen=True, slots=True)
class ImportResult:
    """How many trials were imported."""

    completed: int
    invalid: int
    sessions: int


def _stage_of(ctx: StudyContext, row: ImportRow) -> int:
    stages = [s.id for s in ctx.spec.rubric.stages]
    if row.stage is not None and row.stage != "":
        if row.stage in stages:
            return stages.index(row.stage)
        if row.stage.lstrip("-").isdigit():
            index = int(row.stage)
            if -1 <= index < len(stages):
                return index
        raise ServiceError(f"line {row.line}: unknown stage {row.stage!r}; stages are {stages}")
    if row.success is None:
        if row.stage == "" or row.invalid_reason:
            return -1  # an empty stage cell means no stage was reached
        raise ServiceError(f"line {row.line}: need a success value or a stage")
    return len(stages) - 1 if row.success else -1


def import_trials(
    folder: str | Path, csv_path: str | Path, *, mapping: dict[str, str] | None = None
) -> ImportResult:
    """Import trials from a CSV into a locked study.

    Each row fills the next pending slot with the same condition and arm, in run order. All
    rows are checked before anything is written; if any row cannot be placed, nothing is
    imported. Rows are grouped into sessions by their ``session``, ``operator`` and ``rig``.
    """
    rows = read_import_csv(csv_path, mapping)
    with open_study(folder) as ctx:
        arm_ids = {a.id for a in ctx.spec.arms}
        free: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
        for slot in pending_slots(ctx):
            free[(slot.condition, slot.arm)].append((slot.seq, slot.slot_id))
        problems = []
        placements: list[tuple[ImportRow, tuple[int, str], int]] = []
        for row in rows:
            if row.arm not in arm_ids:
                problems.append(f"line {row.line}: unknown arm {row.arm!r}")
                continue
            try:
                stage = _stage_of(ctx, row)
            except ServiceError as exc:
                problems.append(str(exc))
                continue
            queue = free.get((row.condition, row.arm))
            if not queue:
                problems.append(
                    f"line {row.line}: no pending slot for condition {row.condition!r} and arm "
                    f"{row.arm!r} (unknown condition, or more trials than planned)"
                )
                continue
            placements.append((row, queue.pop(0), stage))
        if problems:
            raise ServiceError("cannot import:\n" + "\n".join(problems))
        # Rows carry no timestamps, so record them in schedule order, back to back, ending now.
        placements.sort(key=lambda p: p[1][0])

        sessions: dict[tuple[str, str, str], str] = {}
        clock = utcnow() - timedelta(seconds=sum(max(r.duration_s, 1.0) for r, _, _ in placements))
        completed = invalid = 0
        for row, (_seq, slot_id), stage in placements:
            key = (row.session, row.operator, row.rig)
            if key not in sessions:
                sessions[key] = start_session(
                    ctx,
                    operator=row.operator,
                    rig=row.rig,
                    notes=f"imported session {row.session}",
                    started_at=clock,
                )
            success_stage = ctx.spec.rubric.success_index
            termination = row.termination or ("success" if stage >= success_stage else "other")
            if row.invalid_reason and not row.termination:
                termination = "robot_fault"
            record_trial(
                ctx,
                slot_id,
                sessions[key],
                stage_index=stage,
                termination=termination,
                duration_s=row.duration_s,
                failure_tags=row.failure_tags,
                notes=row.notes,
                started_at=clock,
                invalid_reason=row.invalid_reason,
                source="csv",
            )
            clock += timedelta(seconds=max(row.duration_s, 1.0))
            if row.invalid_reason:
                invalid += 1
            else:
                completed += 1
        for session_id in sessions.values():
            end_session(ctx, session_id, ended_at=clock)
        return ImportResult(completed=completed, invalid=invalid, sessions=len(sessions))


def export_trials(
    folder: str | Path, out: str | Path, *, fmt: Literal["csv", "jsonl"] = "csv"
) -> int:
    """Export all finished trials. While the study is blinded, arms appear as blind codes."""
    with open_study(folder) as ctx:
        records, _info = collect_records(ctx)
        with ctx.db() as db:
            study = db.scalars(select(m.Study)).one()
            blinded = ctx.spec.design.blinding == "operator" and study.unblinded_at is None
        if fmt == "csv":
            return write_trials_csv(records, out, blinded=blinded)
        return write_trials_jsonl(records, out, blinded=blinded)
