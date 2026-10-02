"""CSV import and CSV/JSONL export."""

import csv
import json
from pathlib import Path

import pytest

from fieldtrial.io.csv import CsvImportError, parse_mapping, read_import_csv
from fieldtrial.services import ServiceError, open_study
from fieldtrial.services.simulate import simulate_study
from fieldtrial.services.study import study_status, unblind_study
from fieldtrial.services.transfer import export_trials, import_trials
from fieldtrial.services.trial import pending_slots


def _write(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _plan(folder: Path) -> list[tuple[str, str]]:
    with open_study(folder) as ctx:
        return [(s.condition, s.arm) for s in pending_slots(ctx)]


def test_parse_mapping() -> None:
    assert parse_mapping(None) == {}
    assert parse_mapping("arm=policy, success=ok") == {"arm": "policy", "success": "ok"}
    with pytest.raises(CsvImportError, match="expected field=column"):
        parse_mapping("arm")
    with pytest.raises(CsvImportError, match="unknown field"):
        parse_mapping("colour=red")


def test_read_requires_columns(tmp_path: Path) -> None:
    path = _write(tmp_path / "t.csv", ["cond", "policy"], [["slot=1", "baseline"]])
    with pytest.raises(CsvImportError) as info:
        read_import_csv(path)
    assert len(info.value.problems) == 3


def test_read_reports_every_bad_row(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "t.csv",
        ["condition", "arm", "success", "duration_s", "status"],
        [
            ["slot=1", "baseline", "maybe", "1", ""],
            ["slot=1", "baseline", "yes", "-1", ""],
            ["slot=1", "baseline", "yes", "1", "lost"],
            ["slot=1", "baseline", "", "1", "invalid"],
        ],
    )
    with pytest.raises(CsvImportError) as info:
        read_import_csv(path)
    assert [p.split(":")[0] for p in info.value.problems] == ["line 2", "line 3", "line 4"]


def test_import_with_mapping_and_stages(locked_study: Path, tmp_path: Path) -> None:
    plan = _plan(locked_study)
    rows = []
    for i, (cond, arm) in enumerate(plan):
        stage = ["clean", "handover", "3", ""][i % 4]
        status = "invalid" if i == 5 else ""
        rows.append([cond, arm, stage, 10 + i, "dropped;stuck" if i == 1 else "", "op", status])
    path = _write(
        tmp_path / "t.csv",
        ["cond", "policy", "stage", "duration_s", "failure_tags", "operator", "status"],
        rows,
    )
    result = import_trials(locked_study, path, mapping=parse_mapping("condition=cond,arm=policy"))
    assert (result.completed, result.invalid, result.sessions) == (7, 1, 1)
    status = study_status(locked_study)
    assert status.pending == 1  # the invalid row's slot was rescheduled
    assert status.invalid_trials == 1


def test_import_is_all_or_nothing(locked_study: Path, tmp_path: Path) -> None:
    (cond, arm), *_ = _plan(locked_study)
    path = _write(
        tmp_path / "t.csv",
        ["condition", "arm", "success", "stage"],
        [
            [cond, arm, "yes", ""],
            [cond, "nobody", "yes", ""],
            ["slot=99", arm, "yes", ""],
            [cond, arm, "", "teleported"],
            [cond, arm, "no", ""],  # more trials than planned for this cell
        ],
    )
    with pytest.raises(ServiceError) as info:
        import_trials(locked_study, path)
    message = str(info.value)
    assert "line 3: unknown arm" in message
    assert "line 4: no pending slot" in message
    assert "line 5: unknown stage" in message
    assert "line 6: no pending slot" in message
    assert study_status(locked_study).pending == 8


def test_export_round_trip(locked_study: Path, tmp_path: Path) -> None:
    simulate_study(locked_study, {"baseline": 0.5, "q50": 0.9}, seed=4, invalid_rate=0.2)
    blinded = tmp_path / "blind.csv"
    count = export_trials(locked_study, blinded)
    with blinded.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == count
    assert not {r["arm"] for r in rows} & {"baseline", "q50"}

    unblind_study(locked_study)
    out = tmp_path / "open.jsonl"
    assert export_trials(locked_study, out, fmt="jsonl") == count
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["arm"] for r in lines} == {"baseline", "q50"}
    assert sum(r["status"] == "completed" for r in lines) == 8
    assert [r["trial_id"] for r in lines] == [r["trial_id"] for r in rows]
