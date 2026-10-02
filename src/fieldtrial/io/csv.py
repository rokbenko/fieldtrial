"""CSV import and export of trials."""

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from fieldtrial.analysis.records import TrialRecord

EXPORT_FIELDS = (
    "trial_id",
    "seq",
    "block",
    "replicate",
    "condition",
    "arm",
    "status",
    "attempt",
    "stage_index",
    "success",
    "termination",
    "duration_s",
    "failure_tags",
    "operator",
    "rig",
    "session_id",
    "started_at",
    "origin",
    "invalid_reason",
    "notes",
)

IMPORT_FIELDS = (
    "condition",
    "arm",
    "success",
    "stage",
    "duration_s",
    "termination",
    "failure_tags",
    "operator",
    "rig",
    "session",
    "status",
    "invalid_reason",
    "notes",
)
_TRUE = {"1", "true", "yes", "y", "t", "success", "ok"}
_FALSE = {"0", "false", "no", "n", "f", "fail", "failure"}


class CsvImportError(ValueError):
    """The CSV cannot be imported. ``problems`` lists every bad row."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(problems))


@dataclass(frozen=True, slots=True)
class ImportRow:
    """One trial to import. ``line`` is the CSV line number, for error messages."""

    line: int
    condition: str
    arm: str
    stage: str | None
    success: bool | None
    duration_s: float
    termination: str | None
    failure_tags: tuple[str, ...]
    operator: str
    rig: str
    session: str
    invalid_reason: str | None
    notes: str | None


def parse_mapping(spec: str | None) -> dict[str, str]:
    """Parse ``--map arm=policy,success=ok`` into ``{"arm": "policy", "success": "ok"}``."""
    mapping: dict[str, str] = {}
    if not spec:
        return mapping
    for part in spec.split(","):
        field, sep, column = part.partition("=")
        field, column = field.strip(), column.strip()
        if not sep or not field or not column:
            raise CsvImportError([f"--map: expected field=column, got {part!r}"])
        if field not in IMPORT_FIELDS:
            raise CsvImportError(
                [f"--map: unknown field {field!r}; fields are {', '.join(IMPORT_FIELDS)}"]
            )
        mapping[field] = column
    return mapping


def _parse_bool(text: str) -> bool | None:
    value = text.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return None


def read_import_csv(path: str | Path, mapping: dict[str, str] | None = None) -> list[ImportRow]:
    """Read trials from a CSV file.

    Required columns: ``condition``, ``arm`` and either ``success`` (true/false, yes/no, 1/0)
    or ``stage`` (a stage id, an index, or empty for no stage). Optional: ``duration_s``,
    ``termination``, ``failure_tags`` (separated by ``;``), ``operator``, ``rig``,
    ``session``, ``status`` (``invalid`` to import a voided trial), ``invalid_reason`` and
    ``notes``. ``mapping`` renames fields to the CSV's own column names.
    """
    columns = {field: field for field in IMPORT_FIELDS} | (mapping or {})
    problems: list[str] = []
    rows: list[ImportRow] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        problems.extend(
            f"missing column {columns[required]!r} (for {required})"
            for required in ("condition", "arm")
            if columns[required] not in header
        )
        if columns["success"] not in header and columns["stage"] not in header:
            problems.append(f"need a {columns['success']!r} or {columns['stage']!r} column")
        if problems:
            raise CsvImportError(problems)

        def get(row: dict[str, str], field: str) -> str:
            return (row.get(columns[field]) or "").strip()

        for line, row in enumerate(reader, start=2):
            status = get(row, "status").lower() or "completed"
            if status not in ("completed", "invalid"):
                problems.append(f"line {line}: status must be completed or invalid, got {status!r}")
                continue
            success = None
            if columns["success"] in header:
                raw = get(row, "success")
                success = _parse_bool(raw)
                if success is None and status == "completed" and columns["stage"] not in header:
                    problems.append(f"line {line}: cannot read success value {raw!r}")
                    continue
            duration_text = get(row, "duration_s")
            try:
                duration = float(duration_text) if duration_text else 0.0
                if duration < 0:
                    raise ValueError
            except ValueError:
                problems.append(f"line {line}: duration_s must be a non-negative number")
                continue
            tags = tuple(t.strip() for t in get(row, "failure_tags").split(";") if t.strip())
            invalid_reason = get(row, "invalid_reason") or None
            if status == "invalid" and not invalid_reason:
                invalid_reason = "imported as invalid"
            rows.append(
                ImportRow(
                    line=line,
                    condition=get(row, "condition"),
                    arm=get(row, "arm"),
                    stage=get(row, "stage") if columns["stage"] in header else None,
                    success=success,
                    duration_s=duration,
                    termination=get(row, "termination") or None,
                    failure_tags=tags,
                    operator=get(row, "operator") or "import",
                    rig=get(row, "rig") or "unknown",
                    session=get(row, "session") or "imported",
                    invalid_reason=invalid_reason if status == "invalid" else None,
                    notes=get(row, "notes") or None,
                )
            )
    if problems:
        raise CsvImportError(problems)
    return rows


def write_trials_csv(records: Iterable[TrialRecord], path: str | Path, *, blinded: bool) -> int:
    """Write trials to CSV. While blinded, the ``arm`` column holds blind codes. Returns rows."""
    count = 0
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(_export_row(r, blinded))
            count += 1
    return count


def _export_row(r: TrialRecord, blinded: bool) -> dict[str, object]:
    return {
        "trial_id": r.trial_id,
        "seq": r.seq,
        "block": r.block,
        "replicate": r.replicate,
        "condition": r.condition,
        "arm": r.blind_code if blinded else r.arm,
        "status": r.status,
        "attempt": r.attempt,
        "stage_index": r.stage_index,
        "success": r.success,
        "termination": r.termination,
        "duration_s": r.duration_s,
        "failure_tags": ";".join(r.failure_tags),
        "operator": r.operator,
        "rig": r.rig,
        "session_id": r.session_id,
        "started_at": r.started_at.isoformat(),
        "origin": r.origin,
        "invalid_reason": r.invalid_reason,
        "notes": r.notes,
    }


def export_rows(records: Sequence[TrialRecord], *, blinded: bool) -> list[dict[str, object]]:
    """Export rows as dicts (shared by the CSV and JSONL writers)."""
    return [_export_row(r, blinded) for r in records]
